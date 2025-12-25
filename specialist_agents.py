"""
SPECIALIST AGENTS - ENHANCED
================================

Key Improvements:
1. LDA Topic Features - 30 latent topics from text
2. Section-Based Extraction - Parse discharge note sections  
3. Enhanced Text Statistics - Word count, sentence count, unique ratio
4. Richer Keyword Lists - From SOA pipeline
5. Lab Trajectory Features - Trend detection (worsening/improving)

Specialists:
1. LabSpecialist - Lab values and organ dysfunction with trajectory features
2. NoteSpecialist - Clinical notes with LDA topics (enhanced)
3. PharmacySpecialist - Medication-based risk
4. HistorySpecialist - Diagnosis history patterns
5. PsychosocialSpecialist - Mental, Care, Social sub-specialists
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation

# Import cache manager
try:
    from clinicalbert_cache import ClinicalBERTCache
except ImportError:
    ClinicalBERTCache = None
    print("      ⚠️ ClinicalBERT cache not available (clinicalbert_cache.py not found)")

# GPU Support
try:
    import torch
    from sentence_transformers import SentenceTransformer
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Import centralized configuration
from config import RANDOM_STATE

# Constants - OPTIMIZED REGULARIZATION
DEFAULT_LEARNING_RATE = 0.02      # Slightly slower for better generalization
DEFAULT_MAX_ITER = 500            # More iterations
DEFAULT_MAX_DEPTH = 4             # Shallower to prevent overfitting
DEFAULT_MIN_SAMPLES_LEAF = 100    # Larger leaves for robustness
DEFAULT_L2_REGULARIZATION = 3.0   # Stronger L2
TFIDF_MAX_FEATURES = 600          # Reduced to minimize noise

# LDA configuration
LDA_N_TOPICS = 30                 # Number of latent topics
LDA_MAX_ITER = 20                 # LDA iterations


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def safe_get_column(df: pd.DataFrame, col_name: str, default_value=0):
    """
    Safely get column from DataFrame. If column doesn't exist, return default value.
    
    Args:
        df: DataFrame to get column from
        col_name: Name of column to retrieve
        default_value: Value to return if column doesn't exist (default: 0)
    
    Returns:
        Column values if exists, otherwise array of default_value
    """
    if col_name in df.columns:
        return df[col_name].values
    else:
        return np.full(len(df), default_value)


# =============================================================================
# LAB SPECIALIST (unchanged - working well)
# =============================================================================
class LabSpecialist:
    """
    Lab-based risk assessment.
    Uses critical/panic lab values, organ system dysfunction detection.
    """
    
    def __init__(self) -> None:
        """Initialize Lab Specialist with model and scaler."""
        self.name = "Spec_Labs"
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=0
        )
        self.scaler = StandardScaler()
        
        # Standard Clinical Ranges (synced with extract_data.py)
        self.critical_ranges = {
            'sodium': (136, 145),
            'potassium': (3.5, 5.0),
            'chloride': (96, 106),
            'bicarbonate': (22, 29),
            'bun': (7, 20),
            'creatinine': (0.6, 1.2),
            'glucose': (70, 180),
            'wbc': (4.0, 11.0),
            'hemoglobin': (12.0, 17.5),
            'hematocrit': (36, 50),
            'platelets': (150, 400),
            'albumin': (3.5, 5.5),
            'bilirubin': (0.1, 1.2),
            'alt': (7, 56),
            'ast': (10, 40),
            'alp': (44, 147),
            'inr': (0.8, 1.2),
            'ptt': (25, 35),
            'lactate': (0.5, 2.2),
            'magnesium': (1.7, 2.2),
            'phosphate': (2.5, 4.5),
            'calcium': (8.5, 10.5),
        }
        
        # Life-Threatening Boundaries (Panic Values)
        self.panic_ranges = {
            'sodium': (120, 160),
            'potassium': (2.5, 6.5),
            'glucose': (40, 500),
            'hemoglobin': (5.0, 20.0),
            'lactate': (0.0, 4.0),
            'platelets': (20, 1000),
            'creatinine': (0.0, 4.0),
            'wbc': (0.5, 30.0),
            'bilirubin': (0.0, 10.0),
            'inr': (0.0, 4.0),
        }
        
        # Organ system mapping for multi-organ dysfunction
        self.organ_systems = {
            'renal': ['creatinine', 'bun'],
            'hepatic': ['bilirubin', 'alt', 'ast', 'alp', 'albumin'],
            'hematologic': ['hemoglobin', 'platelets', 'wbc', 'inr'],
            'metabolic': ['sodium', 'potassium', 'glucose', 'lactate', 'bicarbonate'],
            'cardiac': ['troponin', 'bnp'],
        }
        
        # Labs to track trajectories (worsening/improving trends)
        self.trajectory_labs = ['creatinine', 'hemoglobin', 'wbc', 'potassium', 'sodium', 'bun']
    
    def create_lab_risk_features(self, X_labs: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive lab-based risk features.
        
        FIXED: Now includes raw lab values (not just derived flags).
        Model can learn non-linear relationships from actual values.
        """
        risk_features = pd.DataFrame(index=X_labs.index)
        
        # === 0. RAW LAB VALUES (CRITICAL FOR MODEL LEARNING) ===
        # Include actual lab values - model learns non-linear relationships
        key_labs = ['creatinine', 'hemoglobin', 'sodium', 'potassium', 'wbc', 
                    'platelets', 'bun', 'glucose', 'albumin', 'bilirubin',
                    'lactate', 'inr', 'bicarbonate']
        
        for lab in key_labs:
            # Last value (discharge) - most predictive
            last_col = f'{lab}_last'
            if last_col in X_labs.columns:
                risk_features[f'{lab}_value'] = X_labs[last_col].fillna(-1)  # -1 indicates missing
                risk_features[f'{lab}_missing'] = X_labs[last_col].isna().astype(int)
            
            # Mean value during stay
            mean_col = f'{lab}_mean'
            if mean_col in X_labs.columns:
                risk_features[f'{lab}_mean'] = X_labs[mean_col].fillna(-1)
            
            # Variability (std) - high variability = instability
            std_col = f'{lab}_std'
            if std_col in X_labs.columns:
                risk_features[f'{lab}_variability'] = X_labs[std_col].fillna(0)
        
        # Track abnormalities by organ system
        organ_abnormal_counts = {organ: np.zeros(len(X_labs)) for organ in self.organ_systems}
        
        # 1. Critical and Panic Value Detection
        for col in X_labs.columns:
            lab_name = col.rsplit('_', 1)[0] if '_' in col else col
            lab_name_lower = lab_name.lower()
            val = X_labs[col]
            
            # Panic value detection
            if lab_name_lower in self.panic_ranges:
                low, high = self.panic_ranges[lab_name_lower]
                is_panic = (val > 0) & ((val < low) | (val > high))
                risk_features[f'{lab_name}_PANIC'] = is_panic.astype(int)
            
            # Standard range check for discharge values
            if lab_name_lower in self.critical_ranges:
                low, high = self.critical_ranges[lab_name_lower]
                
                if '_last' in col.lower() or '_mean' in col.lower():
                    # Abnormal at discharge = RISK
                    is_abnormal = (val < low) | (val > high)
                    risk_features[f'{lab_name}_abnormal_dc'] = is_abnormal.astype(int)
                    
                    # Normal at discharge = PROTECTIVE
                    is_normal = (val >= low) & (val <= high) & (val > 0)
                    risk_features[f'{lab_name}_normal_dc'] = is_normal.astype(int)
                    
                    # Track organ system abnormalities
                    for organ, labs in self.organ_systems.items():
                        if lab_name_lower in labs:
                            organ_abnormal_counts[organ] += is_abnormal.astype(int).values
        
        # 2. Organ dysfunction scores
        for organ, counts in organ_abnormal_counts.items():
            risk_features[f'{organ}_dysfunction'] = (counts >= 1).astype(int)
            risk_features[f'{organ}_severe_dysfunction'] = (counts >= 2).astype(int)
        
        # 3. Multi-organ dysfunction
        total_organ_dysfunction = sum(
            (counts >= 1).astype(int) for counts in organ_abnormal_counts.values()
        )
        risk_features['multi_organ_dysfunction'] = (total_organ_dysfunction >= 2).astype(int)
        risk_features['severe_multi_organ'] = (total_organ_dysfunction >= 3).astype(int)
        
        # 4. Anemia severity
        if 'hemoglobin_last' in X_labs.columns:
            hgb = X_labs['hemoglobin_last']
            risk_features['severe_anemia'] = (hgb < 7).astype(int)
            risk_features['moderate_anemia'] = ((hgb >= 7) & (hgb < 10)).astype(int)
        
        # 5. Renal function
        if 'creatinine_last' in X_labs.columns:
            cr = X_labs['creatinine_last']
            risk_features['aki_severe'] = (cr > 3.0).astype(int)
            risk_features['aki_moderate'] = ((cr > 1.5) & (cr <= 3.0)).astype(int)
        
        # 6. TRAJECTORY FEATURES (worsening vs improving)
        for lab in self.trajectory_labs:
            first_col = f'{lab}_first'
            last_col = f'{lab}_last'
            
            if first_col in X_labs.columns and last_col in X_labs.columns:
                first_val = X_labs[first_col].fillna(0)
                last_val = X_labs[last_col].fillna(0)
                
                # Calculate trend (last - first)
                risk_features[f'{lab}_trend'] = last_val - first_val
                
                # Worsening trajectory (bad direction)
                if lab in ['creatinine', 'wbc', 'potassium', 'bun']:
                    # These are bad when increasing
                    risk_features[f'{lab}_worsening'] = (
                        (last_val > first_val * 1.2) & (first_val > 0)
                    ).astype(int)
                    risk_features[f'{lab}_improving'] = (
                        (last_val < first_val * 0.9) & (first_val > 0)
                    ).astype(int)
                else:
                    # Hemoglobin/sodium - bad when decreasing significantly
                    risk_features[f'{lab}_worsening'] = (
                        (last_val < first_val * 0.8) & (first_val > 0)
                    ).astype(int)
                    risk_features[f'{lab}_improving'] = (
                        (last_val > first_val * 1.1) & (first_val > 0)
                    ).astype(int)
        
        # 7. Composite scores
        # SOFA-like proxy score
        risk_features['sofa_proxy'] = (
            risk_features.get('renal_dysfunction', pd.Series(0, index=X_labs.index)).astype(int) +
            risk_features.get('hepatic_dysfunction', pd.Series(0, index=X_labs.index)).astype(int) +
            risk_features.get('hematologic_dysfunction', pd.Series(0, index=X_labs.index)).astype(int) +
            risk_features.get('metabolic_dysfunction', pd.Series(0, index=X_labs.index)).astype(int)
        )
        
        # Bleeding risk (anemia + coagulopathy)
        anemia_flag = risk_features.get('moderate_anemia', pd.Series(0, index=X_labs.index)) | \
                     risk_features.get('severe_anemia', pd.Series(0, index=X_labs.index))
        coag_flag = risk_features.get('hematologic_dysfunction', pd.Series(0, index=X_labs.index))
        risk_features['bleeding_risk'] = (anemia_flag & coag_flag).astype(int)
        
        # 8. Discharge stability score (count of normal labs)
        normal_count = sum(
            risk_features[col] for col in risk_features.columns 
            if col.endswith('_normal_dc')
        )
        abnormal_count = sum(
            risk_features[col] for col in risk_features.columns 
            if col.endswith('_abnormal_dc')
        )
        
        total_labs = normal_count + abnormal_count + 1e-8
        risk_features['discharge_stability'] = normal_count / total_labs
        risk_features['discharge_instability'] = abnormal_count / total_labs
        
        # 9. Count of improving vs worsening labs
        improving_cols = [c for c in risk_features.columns if c.endswith('_improving')]
        worsening_cols = [c for c in risk_features.columns if c.endswith('_worsening')]
        
        if improving_cols:
            risk_features['n_improving_labs'] = risk_features[improving_cols].sum(axis=1)
        if worsening_cols:
            risk_features['n_worsening_labs'] = risk_features[worsening_cols].sum(axis=1)
        
        # 10. Lab data completeness (important signal)
        missing_cols = [c for c in risk_features.columns if c.endswith('_missing')]
        if missing_cols:
            risk_features['n_labs_missing'] = risk_features[missing_cols].sum(axis=1)
            risk_features['lab_completeness'] = 1 - (risk_features['n_labs_missing'] / max(len(missing_cols), 1))
        
        return risk_features.fillna(0)
    
    def learn(
        self, 
        X_labs: pd.DataFrame, 
        context: pd.DataFrame, 
        y: np.ndarray
    ) -> None:
        """Learn WITHOUT context fusion - labs only."""
        print(f"   [LAB] [{self.name}] Learning from Labs Only...")
        
        X_enriched = self.create_lab_risk_features(X_labs)
        X_enriched = X_enriched.fillna(0)
        X_enriched = X_enriched.replace([np.inf, -np.inf], 0)
        
        X_scaled = self.scaler.fit_transform(X_enriched)
        
        print(f"      Training on {X_scaled.shape[1]} LAB-ONLY features...")
        self.model.fit(X_scaled, y)
        print(f"      ✅ Lab specialist training complete")

    def give_opinion(
        self, 
        X_labs: pd.DataFrame, 
        context: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate risk opinion based on lab values.
        
        Returns:
            probabilities: Risk probabilities for each patient
            has_data: Binary flags indicating if patient has lab data (1=has data, 0=missing)
        """
        # Detect missing data: patient has no labs if all values are NaN
        has_data = (~X_labs.isna().all(axis=1)).values.astype(int)
        
        X_enriched = self.create_lab_risk_features(X_labs)
        X_enriched = X_enriched.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_enriched)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Keep actual prediction - Doctor learns to weight based on has_data
        return probs, has_data


# =============================================================================
# NOTE SPECIALIST - IMPROVED PROTECTIVE FACTOR HANDLING
# =============================================================================
class NoteSpecialist:
    """
    Clinical notes specialist with balanced risk/protective factor handling.
    
    UPDATED: Aggressive tuning of protective weights to reduce false positives
    in improved/stable patients.
    """
    
    def __init__(self) -> None:
        """Initialize Note Specialist with encoder and model."""
        self.name = "Spec_Notes"
        
        # Load encoder first
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.encoder = SentenceTransformer(
                'pritamdeka/S-PubMedBert-MS-MARCO', 
                device=DEVICE
            )
            if CUDA_AVAILABLE:
                print(f"      ✅ NoteSpecialist: ClinicalBERT on GPU")
            else:
                print(f"      ℹ️  NoteSpecialist: ClinicalBERT on CPU")
            
            # Initialize cache (always available - even if empty, we can store during process)
            if ClinicalBERTCache is not None:
                try:
                    self.cache = ClinicalBERTCache()
                    print(f"      ✅ NoteSpecialist: ClinicalBERT cache enabled")
                except Exception as e:
                    print(f"      ⚠️ NoteSpecialist: Cache initialization failed: {e}")
                    self.cache = None
            else:
                self.cache = None
        else:
            self.encoder = None
            self.cache = None
            print(f"      ⚠️ NoteSpecialist: Using TF-IDF fallback")
            self.tfidf = TfidfVectorizer(
                max_features=500, 
                ngram_range=(1, 2)
            )
        
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        self.scaler = StandardScaler()
        self.manual_feature_names: List[str] = []
        
        # LDA Topic Model
        self.lda = None
        self.count_vectorizer = None
        self.lda_fitted = False
        
        # Pre-compiled patterns for LDA text cleaning
        self._lda_clean_patterns = {
            'deident': re.compile(r'\[\*\*[^\]]*\*\*\]'),
            'punctuation': re.compile(r'[^\w\s]'),
            'whitespace': re.compile(r'\s+'),
        }
        
        # Section patterns for extraction (PRE-COMPILED for 2-3x speedup)
        self.section_patterns = {
            'chief_complaint': re.compile(r'(?:chief complaint|cc|reason for admission)[:\s]*', re.IGNORECASE),
            'history_present_illness': re.compile(r'(?:history of present illness|hpi|present illness)[:\s]*', re.IGNORECASE),
            'past_medical_history': re.compile(r'(?:past medical history|pmh|medical history)[:\s]*', re.IGNORECASE),
            'social_history': re.compile(r'(?:social history|sh|social hx)[:\s]*', re.IGNORECASE),
            'assessment_plan': re.compile(r'(?:assessment and plan|a/p|assessment|impression)[:\s]*', re.IGNORECASE),
            'discharge_diagnosis': re.compile(r'(?:discharge diagnosis|diagnoses|final diagnosis)[:\s]*', re.IGNORECASE),
            'discharge_condition': re.compile(r'(?:discharge condition|condition at discharge)[:\s]*', re.IGNORECASE),
            'discharge_instructions': re.compile(r'(?:discharge instructions|instructions)[:\s]*', re.IGNORECASE),
            'followup': re.compile(r'(?:follow up|followup|follow-up|appointments)[:\s]*', re.IGNORECASE),
        }
        
        # Risk Patterns by Category (PRE-COMPILED for performance)
        self.risk_categories = {
            'social': {
                'homeless': re.compile(r'\b(?:homeless|shelter|undomiciled|unstable.?housing|no\s*(?:fixed\s*)?address)\b', re.IGNORECASE),
                'lives_alone': re.compile(r'\b(?:lives?\s*alone|no\s*family|poor\s*social\s*support|no.?one.?at.?home|isolated)\b', re.IGNORECASE),
                'substance': re.compile(r'\b(?:alcohol(?:ic|ism)?|ETOH|substance\s*(?:use|abuse)|drug\s*(?:use|abuse)|withdrawal|opioid|heroin|cocaine|IVDU|IV\s*drug)\b', re.IGNORECASE),
                'financial': re.compile(r'\b(?:financial|unable\s*to\s*afford|insurance\s*issue|uninsured|no\s*insurance)\b', re.IGNORECASE),
                'transportation': re.compile(r'\b(?:no\s*transportation|transportation\s*issue|cannot\s*get\s*to)\b', re.IGNORECASE),
            },
            'severity': {
                'critical': re.compile(r'\b(?:critical(?:ly)?|unstable|deteriorat|worsen(?:ing)?|decompensated)\b', re.IGNORECASE),
                'respiratory_failure': re.compile(r'\b(?:respiratory\s*failure|intubat|ventilat|BiPAP|CPAP|oxygen\s*dependent)\b', re.IGNORECASE),
                'sepsis': re.compile(r'\b(?:septic|sepsis|bacteremia|severe\s*infection)\b', re.IGNORECASE),
                'shock': re.compile(r'\b(?:shock|hypotensive|pressors|vasopressors|levophed|norepinephrine)\b', re.IGNORECASE),
                'terminal': re.compile(r'\b(?:metasta|stage\s*(?:IV|4)|terminal|end\s*(?:stage|of\s*life)|poor\s*prognosis|limited\s*life)\b', re.IGNORECASE),
            },
            'cognitive': {
                'dementia': re.compile(r'\b(?:dementia|Alzheimer|cognitive\s*impair|memory\s*loss)\b', re.IGNORECASE),
                'delirium': re.compile(r'\b(?:deliri|confusion|disoriented|altered\s*mental|AMS|encephalopathy)\b', re.IGNORECASE),
                'psych': re.compile(r'\b(?:psychiatric|psychosis|schizophren|bipolar|suicidal|SI)\b', re.IGNORECASE),
            },
            'compliance': {
                'ama': re.compile(r'\b(?:AMA|against\s*medical\s*advice|left\s*AMA|eloped)\b', re.IGNORECASE),
                'noncompliant': re.compile(r'\b(?:non.?complian|medication\s*non.?adherence|not\s*taking\s*medications?)\b', re.IGNORECASE),
                'missed_appointments': re.compile(r'\b(?:missed\s*(?:appointments?|follow.?up)|no.?show)\b', re.IGNORECASE),
            },
            'clinical_risk': {
                'readmission_history': re.compile(r'\b(?:readmit|re.?admit|return(?:ed)?\s*to\s*(?:ED|ER|hospital)|frequent\s*(?:flyer|visitor|admission)|multiple\s*admission|bounce.?back)\b', re.IGNORECASE),
                'unable_to_ambulate': re.compile(r'\b(?:unable\s*to\s*(?:ambulate|walk|get\s*out\s*of\s*bed)|bedbound|non.?ambulatory|wheelchair\s*bound)\b', re.IGNORECASE),
                'unable_to_eat': re.compile(r'\b(?:unable\s*to\s*(?:eat|tolerate|take\s*PO)|NPO|nothing\s*by\s*mouth|feeding\s*tube)\b', re.IGNORECASE),
                'family_concern': re.compile(r'\b(?:(?:family|caregiver|daughter|son)\s*(?:concern|worried|fear|anxious)|family\s*expressed\s*concern)\b', re.IGNORECASE),
                'social_isolation': re.compile(r'\b(?:social\s*(?:isolation|issues|problems)|lonely|isolated|no\s*social\s*contact)\b', re.IGNORECASE),
                'left_ama': re.compile(r'\b(?:left\s*(?:against|AMA)|against\s*medical\s*advice|eloped|signed\s*out\s*AMA)\b', re.IGNORECASE),
                'fall_risk': re.compile(r'\b(?:fall(?:s|en)?|fell|unstead|gait\s*disturb|high\s*fall\s*risk)\b', re.IGNORECASE),
                'frailty': re.compile(r'\b(?:frail|debilitat|cachex|malnutrition|failure\s*to\s*thrive|sarcopenia)\b', re.IGNORECASE),
                'polypharmacy': re.compile(r'\b(?:polypharmacy|multiple\s*medications?|medication\s*(?:error|interaction))\b', re.IGNORECASE),
                'heart_failure': re.compile(r'\b(?:CHF|heart\s*failure|HF(?:rEF|pEF)?|cardiomyopathy|EF\s*(?:of\s*)?\d{1,2}%)\b', re.IGNORECASE),
                'copd': re.compile(r'\b(?:COPD|chronic\s*obstructive|emphysema|COPD\s*exacerbation)\b', re.IGNORECASE),
                'ckd': re.compile(r'\b(?:CKD\s*(?:stage\s*)?[345]|ESRD|dialysis|renal\s*failure|kidney\s*disease)\b', re.IGNORECASE),
                'cirrhosis': re.compile(r'\b(?:cirrhosis|hepatic\s*(?:failure|encephalopathy)|MELD|ascites|varice)\b', re.IGNORECASE),
            }
        }
        
        # BOOSTED protective patterns (PRE-COMPILED for performance)
        self.protective_patterns = {
            'dc_snf': (re.compile(r'\b(?:discharged?\s*to\s*(?:SNF|skilled\s*nursing|nursing\s*(?:home|facility)))\b', re.IGNORECASE), -8.0),
            'dc_rehab': (re.compile(r'\b(?:discharged?\s*to\s*(?:rehab|rehabilitation|acute\s*rehab|inpatient\s*rehab))\b', re.IGNORECASE), -7.0),
            'dc_ltac': (re.compile(r'\b(?:discharged?\s*to\s*(?:LTAC|long\s*term\s*acute\s*care))\b', re.IGNORECASE), -6.0),
            'pcp_followup': (re.compile(r'\b(?:(?:PCP|primary\s*care)\s*(?:follow.?up|appointment)|see\s*(?:PCP|doctor)\s*in)\b', re.IGNORECASE), -5.0),
            'specialist_followup': (re.compile(r'\b(?:(?:cardiology|pulmonology|nephrology|GI|neurology)\s*(?:follow.?up|appointment))\b', re.IGNORECASE), -6.0),
            'family_support': (re.compile(r'\b(?:family\s*(?:at\s*bedside|present|supportive|involved)|(?:daughter|son|wife|husband|spouse)\s*(?:present|involved|will\s*help))\b', re.IGNORECASE), -3.0),
            'caregiver': (re.compile(r'\b(?:caregiver|24.?hour\s*care|(?:has|with)\s*help\s*at\s*home)\b', re.IGNORECASE), -3.5),
            'good_support': (re.compile(r'\b(?:good\s*(?:family|social)\s*support|strong\s*support\s*system)\b', re.IGNORECASE), -4.0),
            'stable': (re.compile(r'\b(?:stable\s*(?:condition|for\s*discharge|vitals)|medically\s*stable|clinically\s*stable)\b', re.IGNORECASE), -15.0),
            'improved': (re.compile(r'\b(?:improved|improving|resolv(?:ed|ing)|recovered|better|much\s*better|significantly\s*improved)\b', re.IGNORECASE), -25.0),
            'tolerating': (re.compile(r'\b(?:tolerating\s*(?:diet|PO|oral)|eating\s*well|good\s*(?:appetite|intake))\b', re.IGNORECASE), -5.0),
            'ambulating': (re.compile(r'\b(?:ambulating|walking\s*(?:independently|with\s*assist)|mobile|out\s*of\s*bed)\b', re.IGNORECASE), -6.0),
            'case_management': (re.compile(r'\b(?:case\s*manag|social\s*work(?:er)?|discharge\s*plann(?:er|ing))\s*(?:involved|consulted|arranged)\b', re.IGNORECASE), -3.0),
            'med_reconciliation': (re.compile(r'\b(?:medication\s*reconcil|med\s*rec|medications?\s*reviewed)\b', re.IGNORECASE), -2.0),
            'education': (re.compile(r'\b(?:patient\s*(?:educated|understands|verbalized)|instructions?\s*(?:given|provided|reviewed)|teach.?back)\b', re.IGNORECASE), -2.0),
            'planned_admission': (re.compile(r'\b(?:planned\s*admission|scheduled\s*(?:admission|procedure|surgery)|elective)\b', re.IGNORECASE), -6.0),
            'chemo_protocol': (re.compile(r'\b(?:(?:chemo|chemotherapy)\s*(?:cycle|protocol|course)|scheduled\s*(?:chemo|infusion))\b', re.IGNORECASE), -6.0),
            'hospice': (re.compile(r'\b(?:hospice|comfort\s*(?:care|measures)|palliative|CMO|DNR.?DNI)\b', re.IGNORECASE), -50.0),
        }
        
        # Additional compiled patterns for _extract_clinical_features (PRE-COMPILED)
        self._clinical_patterns = {
            'dc_to_home': re.compile(r'discharged?\s*(?:to\s*)?home(?!\s*health)', re.IGNORECASE),
            'dc_to_snf': re.compile(r'discharged?\s*to\s*(?:SNF|skilled|nursing)', re.IGNORECASE),
            'dc_to_rehab': re.compile(r'discharged?\s*to\s*(?:rehab|rehabilitation)', re.IGNORECASE),
            'dc_to_hospice': re.compile(r'(?:hospice|comfort\s*care)', re.IGNORECASE),
            'has_followup': re.compile(r'follow.?up|return\s*(?:in|to)', re.IGNORECASE),
            'followup_days': re.compile(r'(?:follow.?up|return|see)\s*(?:in\s*)?(\d+)\s*(?:day|week)', re.IGNORECASE),
            'n_diagnoses': re.compile(r'\b(?:diagnos[ei]s|dx)\b', re.IGNORECASE),
            'n_procedures': re.compile(r'\b(?:procedure|surgery|operation)\b', re.IGNORECASE),
            'med_changes': re.compile(r'\b(?:start(?:ed)?|stop(?:ped)?|changed?|new\s+med|discontinue)\b', re.IGNORECASE),
            'social_work': re.compile(r'social\s*work|case\s*manag', re.IGNORECASE),
            'family_meeting': re.compile(r'family\s*meeting|goals\s*of\s*care', re.IGNORECASE),
            'readmission_mentioned': re.compile(r'readmit|re.?admit|return(?:ed)?\s*to\s*(?:ED|ER|hospital)|bounce.?back|frequent', re.IGNORECASE),
            'prior_mention': re.compile(r'(?:prior|previous|past)\s*(?:admission|hospitalization|visit)|admitted\s*(?:before|previously)|history\s*of\s*admission', re.IGNORECASE),
            'first_admission': re.compile(r'first\s*(?:admission|hospitalization|time)|no\s*prior\s*(?:admission|hospitalization)|new\s*patient', re.IGNORECASE),
        }
    
    def _chunk_text(self, text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
        """
        Split long text into overlapping chunks for ClinicalBERT encoding.
        
        ClinicalBERT has 512 token limit (~400-500 words, ~1500-2000 chars).
        We use conservative 1500 chars to stay safely within limits.
        
        Args:
            text: Full clinical note text
            max_chars: Maximum characters per chunk (1500 ≈ 400 tokens)
            overlap: Character overlap between chunks for context continuity
            
        Returns:
            List of text chunks that together cover the full document
        """
        text = str(text).strip()
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end (.!?) within last 200 chars of chunk
                search_start = max(end - 200, start)
                last_period = text.rfind('. ', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                
                # Use the later boundary
                break_point = max(last_period, last_newline)
                if break_point > start + max_chars // 2:  # Don't break too early
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move forward with overlap
            start = end - overlap if end < len(text) else end
        
        return chunks if chunks else [text[:max_chars]]
    
    def _encode_with_chunking(self, text_list: List[str]) -> np.ndarray:
        """
        Encode long texts using chunking + mean pooling with caching.
        
        For each document:
        1. Check cache first (if available) - if found, return immediately
        2. If not cached: Split into overlapping chunks (each fits in ClinicalBERT's 512 token window)
        3. Encode each chunk separately
        4. Mean pool chunk embeddings to get document embedding
        5. Store in cache for future use
        
        This ensures ALL content is seen by ClinicalBERT, not just first 512 tokens.
        """
        all_embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        for text in tqdm(text_list, desc="      Encoding", unit="doc"):
            # Step 1: Check cache first
            cached_embedding = None
            if self.cache is not None:
                cached_embedding = self.cache.get(text)
            
            if cached_embedding is not None:
                # Cache HIT - use cached embedding
                all_embeddings.append(cached_embedding)
                cache_hits += 1
            else:
                # Cache MISS - encode with ClinicalBERT
                cache_misses += 1
                chunks = self._chunk_text(text, max_chars=1500, overlap=200)
                
                if len(chunks) == 1:
                    # Short document - encode directly
                    chunk_embeds = self.encoder.encode(
                        chunks, 
                        batch_size=1,
                        show_progress_bar=False,
                        device=DEVICE,
                        convert_to_numpy=True
                    )
                    doc_embedding = chunk_embeds[0]
                else:
                    # Long document - encode chunks and mean pool
                    chunk_embeds = self.encoder.encode(
                        chunks,
                        batch_size=min(len(chunks), 8),
                        show_progress_bar=False,
                        device=DEVICE,
                        convert_to_numpy=True
                    )
                    # Mean pooling across chunks
                    doc_embedding = np.mean(chunk_embeds, axis=0)
                
                # Store in cache for future use
                if self.cache is not None:
                    try:
                        self.cache.set(text, doc_embedding)
                    except Exception as e:
                        print(f"      ⚠️ Warning: Failed to cache embedding: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Debug: check why cache is None
                    if cache_misses == 1:  # Only print once
                        print(f"      ⚠️ Warning: Cache is None, embeddings not being stored!")
                
                all_embeddings.append(doc_embedding)
        
        # Print cache statistics
        if self.cache is not None and len(text_list) > 0:
            total = len(text_list)
            hit_rate = (cache_hits / total * 100) if total > 0 else 0
            if cache_hits > 0 or cache_misses > 0:
                print(f"      📦 Cache: {cache_hits} HIT ({hit_rate:.1f}%), {cache_misses} MISS")
        
        return np.array(all_embeddings)
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from discharge summary."""
        sections: Dict[str, str] = {}
        
        section_patterns = [
            ('discharge_diagnosis', r'(?:DISCHARGE\s*DIAGNOS[EI]S|FINAL\s*DIAGNOS[EI]S)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('discharge_condition', r'(?:DISCHARGE\s*CONDITION)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('discharge_disposition', r'(?:DISCHARGE\s*(?:DISPOSITION|TO)|DISCHARGED?\s*TO)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('discharge_instructions', r'(?:DISCHARGE\s*INSTRUCTIONS?)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('followup', r'(?:FOLLOW[- ]?UP|APPOINTMENTS?)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('hospital_course', r'(?:(?:BRIEF\s*)?HOSPITAL\s*COURSE)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('social_history', r'(?:SOCIAL\s*(?:HISTORY|HX))[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
            ('assessment', r'(?:ASSESSMENT(?:\s*(?:AND|&)\s*PLAN)?|IMPRESSION)[:\s]*\n?(.*?)(?=\n[A-Z][A-Z\s]+:|$)'),
        ]
        
        for name, pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[name] = match.group(1).strip()[:1500]
        
        return sections
    
    def _compute_risk_score(
        self, 
        text: str
    ) -> Tuple[float, Dict[str, Union[int, float]]]:
        """Extract risk/protective features - let model learn the weights.
        
        OPTIMIZED: Uses pre-compiled regex patterns for 2-3x speedup.
        """
        feature_dict: Dict[str, Union[int, float]] = {}
        
        # Track category presence
        category_hits = {cat: 0 for cat in self.risk_categories}
        
        # Track specific markers for interaction features
        has_severity_marker = False
        has_improvement_marker = False
        
        # 1. Risk factors by category - COUNT only (using compiled patterns)
        total_risk_count = 0
        for category, patterns in self.risk_categories.items():
            category_count = 0
            for name, compiled_pattern in patterns.items():
                # Use compiled pattern's findall method directly
                matches = compiled_pattern.findall(text)
                count = min(len(matches), 3)  # Cap at 3
                feature_dict[f'risk_{category}_{name}'] = count
                category_count += count
                total_risk_count += count
                
                if category == 'severity' and count > 0:
                    has_severity_marker = True
                if count > 0:
                    category_hits[category] += 1
            
            feature_dict[f'risk_{category}_total'] = category_count
        
        # Category presence flags (binary)
        feature_dict['social_vulnerability_flag'] = 1 if category_hits.get('social', 0) > 0 else 0
        feature_dict['compliance_flag'] = 1 if category_hits.get('compliance', 0) > 0 else 0
        feature_dict['severity_flag'] = 1 if category_hits.get('severity', 0) > 0 else 0
        feature_dict['cognitive_flag'] = 1 if category_hits.get('cognitive', 0) > 0 else 0
        feature_dict['clinical_risk_flag'] = 1 if category_hits.get('clinical_risk', 0) > 0 else 0
        
        # 2. Protective factors (using compiled patterns)
        total_protect_count = 0
        for name, (compiled_pattern, _weight) in self.protective_patterns.items():
            matches = compiled_pattern.findall(text)
            count = min(len(matches), 3)
            feature_dict[f'protect_{name}'] = count
            total_protect_count += count
            
            if name in ['improved', 'stable', 'tolerating', 'ambulating'] and count > 0:
                has_improvement_marker = True
        
        # === AGGREGATE FEATURES ===
        feature_dict['total_risk_count'] = total_risk_count
        feature_dict['total_protect_count'] = total_protect_count
        feature_dict['n_risk_categories'] = sum(1 for v in category_hits.values() if v > 0)
        
        feature_dict['risk_minus_protect'] = total_risk_count - total_protect_count
        feature_dict['protect_ratio'] = total_protect_count / (total_risk_count + total_protect_count + 1)
        
        # === INTERACTION FLAGS ===
        feature_dict['has_severity'] = 1 if has_severity_marker else 0
        feature_dict['has_improvement'] = 1 if has_improvement_marker else 0
        feature_dict['improved_despite_severity'] = 1 if (has_severity_marker and has_improvement_marker) else 0
        
        has_facility = (feature_dict.get('protect_dc_snf', 0) > 0 or 
                       feature_dict.get('protect_dc_rehab', 0) > 0 or
                       feature_dict.get('protect_dc_ltac', 0) > 0)
        feature_dict['discharge_to_facility'] = 1 if has_facility else 0
        
        feature_dict['end_of_life_care'] = 1 if feature_dict.get('protect_hospice', 0) > 0 else 0
        
        total_factors = total_risk_count + total_protect_count
        feature_dict['no_factors_extracted'] = 1 if total_factors == 0 else 0
        feature_dict['low_factor_count'] = 1 if total_factors <= 2 else 0
        feature_dict['rich_documentation'] = 1 if total_factors >= 10 else 0
        
        return 0.0, feature_dict

    def _extract_clinical_features(self, text: str) -> Dict[str, Union[int, float]]:
        """Extract structured clinical features (OPTIMIZED with compiled patterns)."""
        features: Dict[str, Union[int, float]] = {}
        text_len = len(str(text).strip())
        
        # Data poverty penalty
        features['missing_note'] = 1 if text_len < 50 else 0
        features['short_note'] = 1 if text_len < 200 else 0
        features['note_length'] = min(text_len / 1000, 20)
        
        # Discharge disposition (using compiled patterns)
        features['dc_to_home'] = 1 if self._clinical_patterns['dc_to_home'].search(text) else 0
        features['dc_to_snf'] = 1 if self._clinical_patterns['dc_to_snf'].search(text) else 0
        features['dc_to_rehab'] = 1 if self._clinical_patterns['dc_to_rehab'].search(text) else 0
        features['dc_to_hospice'] = 1 if self._clinical_patterns['dc_to_hospice'].search(text) else 0
        
        # Follow-up quality (using compiled patterns)
        features['has_followup'] = 1 if self._clinical_patterns['has_followup'].search(text) else 0
        followup_match = self._clinical_patterns['followup_days'].search(text)
        if followup_match:
            days = int(followup_match.group(1))
            if 'week' in followup_match.group(0).lower():
                days *= 7
            features['followup_days'] = min(days, 30)
        else:
            features['followup_days'] = 0
        
        # Complexity indicators (using compiled patterns)
        features['n_diagnoses_text'] = len(self._clinical_patterns['n_diagnoses'].findall(text))
        features['n_procedures_text'] = len(self._clinical_patterns['n_procedures'].findall(text))
        features['med_changes'] = len(self._clinical_patterns['med_changes'].findall(text))
        
        # Care coordination (using compiled patterns)
        features['social_work'] = 1 if self._clinical_patterns['social_work'].search(text) else 0
        features['family_meeting'] = 1 if self._clinical_patterns['family_meeting'].search(text) else 0
        
        # Readmission mention (using compiled patterns)
        features['readmission_mentioned'] = 1 if self._clinical_patterns['readmission_mentioned'].search(text) else 0
        
        # First-timer signals (using compiled patterns)
        features['no_prior_mention'] = 0 if self._clinical_patterns['prior_mention'].search(text) else 1
        features['first_admission_signal'] = 1 if self._clinical_patterns['first_admission'].search(text) else 0
        
        return features
    
    # Text Statistics Method
    def _get_text_statistics(self, text: str) -> Dict[str, float]:
        """Extract text statistics features."""
        if not text or pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'unique_word_ratio': 0,
                'paragraph_count': 0,
            }
        
        text = str(text)
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        unique_words = len(set(w.lower() for w in words))
        
        return {
            'text_length': len(text),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': word_count / max(sentence_count, 1),
            'unique_word_ratio': unique_words / max(word_count, 1),
            'paragraph_count': len(paragraphs),
        }
    
    # Section Features Method
    def _extract_section_features(self, text: str) -> Dict[str, int]:
        """Extract section-based features."""
        sections = self._extract_sections(text)
        features = {}
        
        # Section presence indicators
        for section_name in self.section_patterns.keys():
            features[f'has_section_{section_name}'] = 1 if section_name in sections and len(sections.get(section_name, '')) > 10 else 0
        
        # Section lengths for key sections
        for section_name in ['assessment_plan', 'discharge_diagnosis', 'social_history']:
            section_text = sections.get(section_name, '')
            features[f'section_length_{section_name}'] = min(len(section_text), 5000)
        
        return features
    
    def _clean_text_for_lda(self, text: str) -> str:
        """Clean text for LDA (uses compiled patterns for speed)."""
        text = str(text).lower()
        text = self._lda_clean_patterns['deident'].sub(' ', text)
        text = self._lda_clean_patterns['punctuation'].sub(' ', text)
        text = self._lda_clean_patterns['whitespace'].sub(' ', text).strip()
        return text
    
    # LDA Fit Method
    def _fit_lda(self, text_list: List[str]) -> None:
        """Fit LDA topic model (OPTIMIZED with compiled patterns)."""
        print(f"      Fitting LDA topic model ({LDA_N_TOPICS} topics)...")
        
        # Clean texts for LDA using compiled patterns
        cleaned_texts = [self._clean_text_for_lda(t) for t in text_list]
        
        # Always use sequential processing (no parallelization)
        n_jobs = 1
        
        self.count_vectorizer = CountVectorizer(
            max_features=3000,
            min_df=5,
            max_df=0.95,
            stop_words='english'
        )
        
        count_matrix = self.count_vectorizer.fit_transform(cleaned_texts)
        
        self.lda = LatentDirichletAllocation(
            n_components=LDA_N_TOPICS,
            max_iter=LDA_MAX_ITER,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs  # Conditional: 1 if PyTorch CUDA, -1 otherwise
        )
        self.lda.fit(count_matrix)
        self.lda_fitted = True
        print(f"      ✅ LDA fitted (perplexity: {self.lda.perplexity(count_matrix):.2f})")
    
    # LDA Transform Method
    def _get_lda_features(self, text_list: List[str]) -> np.ndarray:
        """Get LDA topic features (OPTIMIZED with compiled patterns)."""
        if not self.lda_fitted:
            return np.zeros((len(text_list), LDA_N_TOPICS))
        
        # Clean texts using compiled patterns
        cleaned_texts = [self._clean_text_for_lda(t) for t in text_list]
        
        count_matrix = self.count_vectorizer.transform(cleaned_texts)
        topic_features = self.lda.transform(count_matrix)
        
        return topic_features
    
    def _extract_all_features_single(self, text: str) -> Dict[str, Union[int, float]]:
        """Extract all features from a single note (for parallel processing)."""
        _, risk_feats = self._compute_risk_score(text)
        clinical_feats = self._extract_clinical_features(text)
        text_stats = self._get_text_statistics(text)
        section_feats = self._extract_section_features(text)
        return {**risk_feats, **clinical_feats, **text_stats, **section_feats}
    
    def _preprocess(
        self, 
        text_list: List[str]
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Preprocess notes for encoding with enhanced features.
        Uses sequential processing for compatibility.
        
        Returns FULL text for chunked ClinicalBERT encoding (not truncated).
        Manual features include text stats and section features.
        """
        n_notes = len(text_list)
        processed_texts: List[str] = [str(t) for t in text_list]
        
        print(f"      Processing {n_notes} notes (sequential)...")
        
        # Use sequential processing for feature extraction
        all_features: List[Dict[str, Union[int, float]]] = []
        
        # Process each text sequentially
        for text in tqdm(text_list, desc="      Extracting", unit="note"):
            try:
                feats = self._extract_all_features_single(str(text))
                all_features.append(feats)
            except Exception as e:
                print(f"      ⚠️ Feature extraction failed: {e}")
                all_features.append({})

        
        features_df = pd.DataFrame(all_features).fillna(0)
        return processed_texts, features_df
    
    def learn(
        self, 
        text_list: List[str], 
        context: pd.DataFrame, 
        y: np.ndarray
    ) -> Optional[np.ndarray]:
        """Learn with enhanced features including LDA topics."""
        print(f"   [NOTE] [{self.name}] Learning from Text Only (Enhanced)...")
        
        # Fit LDA topic model
        self._fit_lda(text_list)
        
        processed_text, manual_features = self._preprocess(text_list)
        
        # Get LDA topic features
        lda_features = self._get_lda_features(text_list)
        
        if self.encoder is not None:
            print(f"      Encoding {len(processed_text)} notes with ClinicalBERT (chunked)...")
            # Use chunking to handle long notes - ensures ALL content is seen
            embeds = self._encode_with_chunking(processed_text)
            print(f"      Embeddings shape: {embeds.shape}")
        else:
            print(f"      Encoding with TF-IDF...")
            embeds = self.tfidf.fit_transform(processed_text).toarray()
        
        self.manual_feature_names = manual_features.columns.tolist()
        print(f"      Manual features: {len(self.manual_feature_names)}")
        print(f"      LDA topics: {lda_features.shape[1]}")
        
        # Combine all features (includes LDA topics)
        X_final = np.column_stack([embeds, manual_features.values, lda_features])
        
        print(f"      Training on {X_final.shape[1]} TEXT-ONLY features...")
        self.model.fit(X_final, y)
        print(f"      ✅ Note specialist training complete")
        
        return self.model.predict_proba(X_final)[:, 1]
    
    def give_opinion(
        self, 
        text_list: List[str], 
        context: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate risk opinion based on clinical notes (chunked encoding).
        
        Returns:
            probabilities: Risk probabilities for each patient
            has_data: Binary flags indicating if patient has discharge notes (1=has notes, 0=missing)
        """
        # Detect missing notes: empty strings or very short (<10 chars)
        # Missing discharge notes can indicate AMA, rapid discharge, or data quality issues
        has_data = np.array([len(str(text).strip()) > 10 for text in text_list], dtype=int)
        
        processed_text, manual_features = self._preprocess(text_list)
        
        # Get LDA topic features
        lda_features = self._get_lda_features(text_list)
        
        if self.encoder is not None:
            # Use chunking to handle long notes
            embeds = self._encode_with_chunking(processed_text)
        else:
            embeds = self.tfidf.transform(processed_text).toarray()
        
        # Ensure all expected columns exist
        for col in self.manual_feature_names:
            if col not in manual_features.columns:
                manual_features[col] = 0
        manual_features = manual_features[self.manual_feature_names]
        
        # Combine all features (includes LDA topics)
        X_final = np.column_stack([embeds, manual_features.values, lda_features])
        probs = self.model.predict_proba(X_final)[:, 1]
        
        # Keep actual prediction - Doctor learns to discount when has_data=0
        return probs, has_data


# =============================================================================
# PHARMACY SPECIALIST (unchanged - working well)
# =============================================================================
class PharmacySpecialist:
    """
    Medication-based risk assessment.
    """
    
    def __init__(self) -> None:
        """Initialize Pharmacy Specialist with vectorizer and model."""
        self.name = "Spec_Pharm"
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,  # Reduced from 1000
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5  # Increased from 3
        )
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=0
        )
        self.scaler = StandardScaler()
        
        # High-risk medications
        self.high_risk_meds = {
            'anticoagulants': ['warfarin', 'coumadin', 'heparin', 'enoxaparin', 'lovenox', 
                              'rivaroxaban', 'xarelto', 'apixaban', 'eliquis', 'dabigatran'],
            'insulin': ['insulin', 'lantus', 'humalog', 'novolog', 'levemir', 'tresiba'],
            'opioids': ['morphine', 'fentanyl', 'oxycodone', 'hydromorphone', 'dilaudid',
                       'hydrocodone', 'methadone', 'oxycontin'],
            'cardiac': ['digoxin', 'amiodarone', 'sotalol', 'dofetilide'],
            'immunosuppressants': ['tacrolimus', 'cyclosporine', 'sirolimus', 'mycophenolate'],
            'chemotherapy': ['methotrexate', 'cyclophosphamide', 'azathioprine'],
            'psych': ['lithium', 'clozapine'],
            'diuretics': ['furosemide', 'lasix', 'bumetanide', 'torsemide', 'metolazone'],
        }
        
        # Protective medications
        self.protective_meds = {
            'dm_controlled': ['metformin', 'glipizide', 'sitagliptin', 'jardiance', 'ozempic'],
            'htn_controlled': ['lisinopril', 'losartan', 'amlodipine', 'metoprolol', 'carvedilol'],
            'lipid_managed': ['atorvastatin', 'rosuvastatin', 'simvastatin', 'pravastatin'],
            'preventive': ['aspirin 81', 'baby aspirin'],
            'hf_optimized': ['entresto', 'sacubitril', 'spironolactone'],
        }
        
        # Dangerous combinations
        self.dangerous_combos = [
            (['warfarin', 'coumadin'], ['aspirin', 'nsaid', 'ibuprofen', 'naproxen']),
            (['metformin'], ['contrast', 'iodine']),
            (['ace', 'lisinopril', 'enalapril'], ['potassium', 'k-dur', 'klor']),
            (['ssri', 'sertraline', 'fluoxetine'], ['tramadol', 'maoi']),
            (['digoxin'], ['amiodarone']),
        ]
    
    def extract_medication_features(self, text_list: List[str]) -> pd.DataFrame:
        """Extract comprehensive medication features."""
        features = pd.DataFrame()
        
        high_risk_counts = {cat: [] for cat in self.high_risk_meds}
        protective_counts = {cat: [] for cat in self.protective_meds}
        interaction_risks = []
        total_meds = []
        unique_meds = []
        
        for text in text_list:
            text_lower = str(text).lower()
            words = set(text_lower.split())
            
            # High-risk medication counts by category
            for category, meds in self.high_risk_meds.items():
                count = sum(1 for med in meds if med in text_lower)
                high_risk_counts[category].append(count)
            
            # Protective medication counts
            for category, meds in self.protective_meds.items():
                count = sum(1 for med in meds if med in text_lower)
                protective_counts[category].append(count)
            
            # Check dangerous combinations
            combo_count = 0
            for group1, group2 in self.dangerous_combos:
                has_group1 = any(med in text_lower for med in group1)
                has_group2 = any(med in text_lower for med in group2)
                if has_group1 and has_group2:
                    combo_count += 1
            interaction_risks.append(combo_count)
            
            # Total medication burden
            total_meds.append(len(words))
            unique_meds.append(len(set(words)))
        
        # Add high-risk counts
        for category, counts in high_risk_counts.items():
            features[f'high_risk_{category}'] = counts
        
        # Total high-risk
        features['total_high_risk_meds'] = pd.DataFrame(high_risk_counts).sum(axis=1)
        
        # Add protective counts
        for category, counts in protective_counts.items():
            features[f'protective_{category}'] = counts
        
        # Total protective
        features['total_protective_meds'] = pd.DataFrame(protective_counts).sum(axis=1)
        
        # Other features
        features['drug_interaction_risk'] = interaction_risks
        features['total_med_count'] = total_meds
        features['unique_med_count'] = unique_meds
        
        # Polypharmacy
        features['polypharmacy'] = (np.array(unique_meds) > 10).astype(int)
        features['extreme_polypharmacy'] = (np.array(unique_meds) > 20).astype(int)
        
        # Low medication burden (protective)
        features['low_med_burden'] = (np.array(unique_meds) <= 5).astype(int)
        features['no_meds'] = (np.array(unique_meds) <= 1).astype(int)
        
        # Anticoagulation complexity
        features['on_anticoag'] = (np.array(high_risk_counts['anticoagulants']) > 0).astype(int)
        features['on_insulin'] = (np.array(high_risk_counts['insulin']) > 0).astype(int)
        features['on_opioids'] = (np.array(high_risk_counts['opioids']) > 0).astype(int)
        
        # Well-managed chronic disease indicators
        features['dm_managed'] = (np.array(protective_counts['dm_controlled']) > 0).astype(int)
        features['htn_managed'] = (np.array(protective_counts['htn_controlled']) > 0).astype(int)
        features['cv_prevention'] = (np.array(protective_counts['lipid_managed']) > 0).astype(int)
        
        return features
    
    def learn(
        self, 
        text_list: List[str], 
        context: pd.DataFrame, 
        y: np.ndarray
    ) -> None:
        """Learn WITHOUT context fusion - medications only."""
        print(f"   [PHARM] [{self.name}] Learning from Medications Only...")
        
        X_vec = self.vectorizer.fit_transform(text_list)
        risk_features = self.extract_medication_features(text_list)
        X_final = np.column_stack([X_vec.toarray(), risk_features.values])
        
        print(f"      Training on {X_final.shape[1]} MED-ONLY features...")
        self.model.fit(X_final, y)
        print(f"      ✅ Pharmacy specialist training complete")
    
    def give_opinion(
        self, 
        text_list: List[str], 
        context: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate risk opinion based on medications.
        
        Returns:
            probabilities: Risk probabilities for each patient
            has_data: Binary flags indicating if patient has medication data (1=has data, 0=missing)
        """
        # Detect missing medication lists: empty strings or very short
        has_data = np.array([len(str(text).strip()) > 5 for text in text_list], dtype=int)
        
        X_vec = self.vectorizer.transform(text_list)
        risk_features = self.extract_medication_features(text_list)
        X_final = np.column_stack([X_vec.toarray(), risk_features.values])
        probs = self.model.predict_proba(X_final)[:, 1]
        
        # Keep actual prediction - Doctor learns to weight by has_data
        return probs, has_data


# =============================================================================
# HISTORY SPECIALIST (unchanged - working well)
# =============================================================================
class HistorySpecialist:
    """
    Diagnosis history pattern specialist.
    """
    
    def __init__(self) -> None:
        """Initialize History Specialist with vectorizer and model."""
        self.name = "Spec_Hist"
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,  # Reduced from 2000
            ngram_range=(1, 2),
            min_df=5  # Increased from 3
        )
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        # High-risk chronic conditions
        self.high_risk_conditions = {
            'heart_failure': ['I50', 'I110', 'I130', 'I132', 'heart failure', 'chf', 'hfref', 'hfpef'],
            'copd': ['J44', 'J43', 'J41', 'J42', 'copd', 'emphysema', 'chronic bronchitis'],
            'ckd_advanced': ['N18[345]', 'N19', 'esrd', 'dialysis', 'stage 4', 'stage 5'],
            'cirrhosis': ['K74', 'K70', 'K71', 'cirrhosis', 'hepatic failure', 'meld'],
            'cancer': ['C[0-9]', 'malignancy', 'metasta', 'carcinoma', 'oncology'],
            'diabetes_complicated': ['E11[2-5]', 'E10[2-5]', 'diabetic nephropathy', 'diabetic retinopathy'],
            'dementia': ['F0[0-3]', 'G30', 'dementia', 'alzheimer', 'cognitive impairment'],
        }
        
        # Managed chronic conditions
        self.managed_conditions = {
            'hypertension': ['I10', 'hypertension', 'htn', 'high blood pressure'],
            'diabetes_uncomplicated': ['E119', 'E109', 'diabetes mellitus', 'dm2', 'type 2 diabetes'],
            'hyperlipidemia': ['E78', 'hyperlipidemia', 'dyslipidemia', 'high cholesterol'],
            'hypothyroidism': ['E03', 'hypothyroidism', 'hashimoto'],
            'gerd': ['K21', 'gerd', 'reflux', 'heartburn'],
            'osteoarthritis': ['M15', 'M16', 'M17', 'osteoarthritis', 'oa', 'degenerative joint'],
        }
        
        # High-risk combinations
        self.risky_combos = [
            ['heart_failure', 'ckd_advanced'],
            ['diabetes_complicated', 'ckd_advanced'],
            ['copd', 'heart_failure'],
            ['cirrhosis', 'ckd_advanced'],
        ]
    
    def extract_history_features(self, text_list: List[str]) -> pd.DataFrame:
        """Extract diagnosis history features."""
        features = pd.DataFrame()
        
        high_risk_counts = {cond: [] for cond in self.high_risk_conditions}
        managed_counts = {cond: [] for cond in self.managed_conditions}
        
        for text in text_list:
            text_lower = str(text).lower()
            text_upper = str(text).upper()
            
            # High-risk condition detection
            for condition, patterns in self.high_risk_conditions.items():
                count = sum(1 for p in patterns if re.search(p, text_lower, re.I) or p.upper() in text_upper)
                high_risk_counts[condition].append(min(count, 3))
            
            # Managed condition detection
            for condition, patterns in self.managed_conditions.items():
                count = sum(1 for p in patterns if re.search(p, text_lower, re.I) or p.upper() in text_upper)
                managed_counts[condition].append(min(count, 3))
        
        # Add counts
        for cond, counts in high_risk_counts.items():
            features[f'has_{cond}'] = (np.array(counts) > 0).astype(int)
        
        for cond, counts in managed_counts.items():
            features[f'has_{cond}'] = (np.array(counts) > 0).astype(int)
        
        # Aggregate scores
        features['n_high_risk_conditions'] = sum((np.array(c) > 0).astype(int) for c in high_risk_counts.values())
        features['n_managed_conditions'] = sum((np.array(c) > 0).astype(int) for c in managed_counts.values())
        
        # Multi-morbidity
        features['multimorbid'] = (features['n_high_risk_conditions'] >= 2).astype(int)
        features['severe_multimorbid'] = (features['n_high_risk_conditions'] >= 3).astype(int)
        
        # Check risky combinations
        combo_scores = np.zeros(len(text_list))
        for combo in self.risky_combos:
            if all(f'has_{c}' in features.columns for c in combo):
                has_combo = np.ones(len(text_list), dtype=int)
                for c in combo:
                    has_combo = has_combo & features[f'has_{c}'].values
                combo_scores += has_combo
        features['risky_combo_count'] = combo_scores
        
        # Risk ratio
        features['condition_risk_ratio'] = (
            features['n_high_risk_conditions'] / (features['n_managed_conditions'] + 1)
        )
        
        # History complexity
        features['history_complexity'] = features['n_high_risk_conditions'] + features['n_managed_conditions']
        
        # Clean patient
        features['clean_history'] = (features['history_complexity'] <= 2).astype(int)
        
        return features
    
    def learn(
        self, 
        text_list: List[str], 
        context: pd.DataFrame, 
        y: np.ndarray
    ) -> None:
        """Learn WITHOUT context fusion - diagnosis history only."""
        print(f"   [HIST] [{self.name}] Learning from Diagnosis History Only...")
        
        X_vec = self.vectorizer.fit_transform(text_list)
        history_features = self.extract_history_features(text_list)
        X_final = np.column_stack([X_vec.toarray(), history_features.values])
        
        print(f"      Training on {X_final.shape[1]} HISTORY-ONLY features...")
        self.model.fit(X_final, y)
        print(f"      ✅ History specialist training complete")
    
    def give_opinion(
        self, 
        text_list: List[str], 
        context: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate risk opinion based on diagnosis history.
        
        Returns:
            probabilities: Risk probabilities for each patient
            has_data: Binary flags indicating if patient has prior history (1=has history, 0=first admission)
        """
        # Detect missing history: empty strings or very short
        # CLINICALLY CRITICAL: Missing history = first admission = different risk profile!
        has_data = np.array([len(str(text).strip()) > 10 for text in text_list], dtype=int)
        
        X_vec = self.vectorizer.transform(text_list)
        history_features = self.extract_history_features(text_list)
        X_final = np.column_stack([X_vec.toarray(), history_features.values])
        probs = self.model.predict_proba(X_final)[:, 1]
        
        # Keep actual prediction - Doctor learns first admission pattern
        return probs, has_data


# =============================================================================
# NEW: MENTAL HEALTH SPECIALIST
# =============================================================================
class MentalSpecialist:
    """
    Mental Health and Psychiatric Specialist.
    
    INTELLIGENT INFERENCE:
    - Infers mental health status from psychiatric medications when diagnoses missing
    - Detects substance use from medication patterns (naltrexone, suboxone)
    - Uses note patterns to identify depression/anxiety even if not formally coded
    
    KEY FEATURES:
    - Psychiatric diagnoses (depression, anxiety, schizophrenia, bipolar, PTSD)
    - Substance use disorders (alcohol, opioids, stimulants)
    - Suicide/self-harm risk indicators
    - Cognitive impairments (dementia, delirium)
    - Psychiatric medication complexity
    """
    
    def __init__(self) -> None:
        """Initialize Mental Health Specialist."""
        self.name = "Spec_Mental"
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,  # Reduced from 1500
            ngram_range=(1, 2),  # Reduced from (1,3)
            min_df=5  # Increased from 2
        )
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        # === PSYCHIATRIC DIAGNOSES (ICD-10 patterns) ===
        self.psych_conditions = {
            # Mood Disorders (HIGH RISK)
            'major_depression': {
                'patterns': [r'F32', r'F33', r'major\s*depress', r'MDD', r'severe\s*depress'],
                'weight': 2.5
            },
            'bipolar': {
                'patterns': [r'F31', r'bipolar', r'manic', r'mania', r'hypomania'],
                'weight': 3.0
            },
            # Anxiety Disorders
            'anxiety_disorder': {
                'patterns': [r'F41', r'F40', r'anxiety', r'GAD', r'panic\s*disorder', r'phobia'],
                'weight': 1.5
            },
            'ptsd': {
                'patterns': [r'F43\.1', r'PTSD', r'post.?traumatic', r'trauma.?stress'],
                'weight': 2.0
            },
            # Psychotic Disorders (VERY HIGH RISK)
            'schizophrenia': {
                'patterns': [r'F20', r'schizophren', r'psychosis', r'psychotic'],
                'weight': 4.0
            },
            'schizoaffective': {
                'patterns': [r'F25', r'schizoaffective'],
                'weight': 3.5
            },
            # Personality Disorders
            'borderline': {
                'patterns': [r'F60\.3', r'BPD', r'borderline\s*personality'],
                'weight': 2.5
            },
            # Eating Disorders
            'eating_disorder': {
                'patterns': [r'F50', r'anorexia', r'bulimia', r'eating\s*disorder'],
                'weight': 2.0
            },
        }
        
        # === SUBSTANCE USE DISORDERS ===
        self.substance_patterns = {
            'alcohol_use': {
                'patterns': [r'F10', r'alcohol\s*(?:use|abuse|depend|withdrawal)', r'ETOH', 
                           r'alcoholic', r'alcoholism', r'DTs', r'delirium\s*tremens'],
                'weight': 3.0
            },
            'opioid_use': {
                'patterns': [r'F11', r'opioid\s*(?:use|abuse|depend|overdose)', r'heroin', 
                           r'OUD', r'IVDU', r'IV\s*drug\s*use', r'needle', r'track\s*marks'],
                'weight': 3.5
            },
            'stimulant_use': {
                'patterns': [r'F14', r'F15', r'cocaine', r'methamphetamine', r'meth', 
                           r'amphetamine', r'stimulant\s*(?:use|abuse)'],
                'weight': 3.0
            },
            'benzodiazepine': {
                'patterns': [r'F13', r'benzo', r'benzodiazepine\s*(?:use|abuse|depend)'],
                'weight': 2.5
            },
            'polysubstance': {
                'patterns': [r'polysubstance', r'multiple\s*substance', r'poly.?drug'],
                'weight': 4.0
            },
        }
        
        # === PSYCHIATRIC RISK INDICATORS ===
        self.psych_risk_patterns = {
            # Self-harm/Suicide (CRITICAL)
            'suicidal_ideation': {
                'patterns': [r'suicidal\s*(?:ideation|thought|SI)', r'wants?\s*to\s*(?:die|kill)', 
                           r'passive\s*SI', r'active\s*SI', r'death\s*wish'],
                'weight': 5.0
            },
            'suicide_attempt': {
                'patterns': [r'suicide\s*attempt', r'attempted\s*suicide', r'overdose.*intent', 
                           r'self.?inflicted', r'intentional\s*overdose'],
                'weight': 6.0
            },
            'self_harm': {
                'patterns': [r'self.?harm', r'self.?mutilat', r'cutting', r'self.?injur'],
                'weight': 4.0
            },
            # Psychiatric crises
            'psychiatric_admission': {
                'patterns': [r'psych(?:iatric)?\s*(?:admit|admission|unit|ward|floor|consult)',
                           r'transferred?\s*to\s*psych', r'1:1\s*sitter'],
                'weight': 3.0
            },
            'agitation': {
                'patterns': [r'agitat(?:ed|ion)', r'violent', r'combative', r'restraint', 
                           r'chemical\s*restraint', r'security\s*called'],
                'weight': 2.5
            },
        }
        
        # === COGNITIVE IMPAIRMENT ===
        self.cognitive_patterns = {
            'dementia': {
                'patterns': [r'F0[0-3]', r'G30', r'dementia', r'alzheimer', r'cognitive\s*impair',
                           r'memory\s*loss', r'confusion', r'disoriented'],
                'weight': 2.0
            },
            'delirium': {
                'patterns': [r'deliri', r'AMS', r'altered\s*mental', r'encephalopathy', 
                           r'acute\s*confusion', r'sundowning'],
                'weight': 2.5
            },
        }
        
        # === PSYCHIATRIC MEDICATIONS (for intelligent inference) ===
        self.psych_medications = {
            'antipsychotics': ['risperidone', 'risperdal', 'olanzapine', 'zyprexa', 'quetiapine',
                             'seroquel', 'aripiprazole', 'abilify', 'haloperidol', 'haldol',
                             'ziprasidone', 'geodon', 'clozapine', 'clozaril', 'paliperidone',
                             'invega', 'lurasidone', 'latuda'],
            'antidepressants': ['sertraline', 'zoloft', 'fluoxetine', 'prozac', 'escitalopram',
                              'lexapro', 'citalopram', 'celexa', 'paroxetine', 'paxil',
                              'venlafaxine', 'effexor', 'duloxetine', 'cymbalta', 'bupropion',
                              'wellbutrin', 'mirtazapine', 'remeron', 'trazodone'],
            'mood_stabilizers': ['lithium', 'valproic', 'depakote', 'lamotrigine', 'lamictal',
                                'carbamazepine', 'tegretol'],
            'anxiolytics': ['lorazepam', 'ativan', 'alprazolam', 'xanax', 'diazepam', 'valium',
                          'clonazepam', 'klonopin', 'buspirone', 'buspar'],
            'substance_treatment': ['naltrexone', 'vivitrol', 'buprenorphine', 'suboxone',
                                   'methadone', 'disulfiram', 'antabuse', 'acamprosate'],
            'adhd_meds': ['methylphenidate', 'ritalin', 'concerta', 'adderall', 'amphetamine',
                         'vyvanse', 'lisdexamfetamine', 'strattera', 'atomoxetine'],
        }
        
        # === PROTECTIVE FACTORS ===
        self.protective_patterns = {
            'psychiatry_followup': {
                'patterns': [r'psychiatry\s*follow.?up', r'mental\s*health\s*follow.?up',
                           r'outpatient\s*psych', r'see\s*psychiatr'],
                'weight': -3.0
            },
            'stable_on_meds': {
                'patterns': [r'stable\s*on\s*(?:psych|mental|medication)', r'controlled\s*on',
                           r'compliant\s*with\s*(?:psych|mental)', r'no\s*psych\s*symptoms'],
                'weight': -2.5
            },
            'social_support_mental': {
                'patterns': [r'therapy', r'therapist', r'counselor', r'support\s*group',
                           r'AA', r'NA', r'sober', r'recovery'],
                'weight': -2.0
            },
        }
    
    def extract_mental_features(
        self, 
        text_list: List[str],
        med_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract mental health features with intelligent inference.
        
        Args:
            text_list: List of clinical note texts
            med_list: Optional list of medication texts for inference
        """
        features = pd.DataFrame()
        
        psych_condition_flags = {cond: [] for cond in self.psych_conditions}
        substance_flags = {subst: [] for subst in self.substance_patterns}
        risk_flags = {risk: [] for risk in self.psych_risk_patterns}
        cognitive_flags = {cog: [] for cog in self.cognitive_patterns}
        protective_flags = {prot: [] for prot in self.protective_patterns}
        
        med_inferred = {med_cat: [] for med_cat in self.psych_medications}
        
        total_psych_scores = []
        total_substance_scores = []
        total_risk_scores = []
        
        for idx, text in enumerate(text_list):
            text_lower = str(text).lower()
            
            psych_score = 0.0
            subst_score = 0.0
            risk_score = 0.0
            
            # === PSYCHIATRIC CONDITIONS ===
            for condition, config in self.psych_conditions.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                psych_condition_flags[condition].append(1 if found else 0)
                if found:
                    psych_score += config['weight']
            
            # === SUBSTANCE USE ===
            for substance, config in self.substance_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                substance_flags[substance].append(1 if found else 0)
                if found:
                    subst_score += config['weight']
            
            # === RISK INDICATORS ===
            for risk, config in self.psych_risk_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                risk_flags[risk].append(1 if found else 0)
                if found:
                    risk_score += config['weight']
            
            # === COGNITIVE IMPAIRMENT ===
            for cognitive, config in self.cognitive_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                cognitive_flags[cognitive].append(1 if found else 0)
            
            # === PROTECTIVE FACTORS ===
            for protective, config in self.protective_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                protective_flags[protective].append(1 if found else 0)
                if found:
                    psych_score += config['weight']  # negative weight = protective
            
            # === INTELLIGENT INFERENCE FROM MEDICATIONS ===
            med_text = med_list[idx] if med_list and idx < len(med_list) else text_lower
            med_text_lower = str(med_text).lower()
            
            for med_cat, meds in self.psych_medications.items():
                found = any(med in med_text_lower for med in meds)
                med_inferred[med_cat].append(1 if found else 0)
                
                # Infer conditions from medications
                if found:
                    if med_cat == 'antipsychotics':
                        psych_score += 1.5  # Suggests psychotic or severe mood disorder
                    elif med_cat == 'mood_stabilizers':
                        psych_score += 1.0  # Suggests bipolar or mood instability
                    elif med_cat == 'substance_treatment':
                        subst_score += 2.0  # STRONG indicator of substance use
            
            total_psych_scores.append(psych_score)
            total_substance_scores.append(subst_score)
            total_risk_scores.append(risk_score)
        
        # Add condition flags
        for cond, flags in psych_condition_flags.items():
            features[f'mental_{cond}'] = flags
        
        for subst, flags in substance_flags.items():
            features[f'substance_{subst}'] = flags
        
        for risk, flags in risk_flags.items():
            features[f'risk_{risk}'] = flags
        
        for cog, flags in cognitive_flags.items():
            features[f'cognitive_{cog}'] = flags
        
        for prot, flags in protective_flags.items():
            features[f'protect_{prot}'] = flags
        
        # Add medication-inferred features
        for med_cat, flags in med_inferred.items():
            features[f'on_{med_cat}'] = flags
        
        # Aggregate scores
        features['total_psych_score'] = total_psych_scores
        features['total_substance_score'] = total_substance_scores
        features['total_risk_score'] = total_risk_scores
        features['combined_mental_score'] = (
            np.array(total_psych_scores) + 
            np.array(total_substance_scores) + 
            np.array(total_risk_scores)
        )
        
        # High-risk flags
        features['has_any_psych'] = (np.array(total_psych_scores) > 0).astype(int)
        features['has_substance_use'] = (np.array(total_substance_scores) > 0).astype(int)
        features['has_suicide_risk'] = [
            1 if (risk_flags['suicidal_ideation'][i] or risk_flags['suicide_attempt'][i] or 
                  risk_flags['self_harm'][i]) else 0
            for i in range(len(text_list))
        ]
        features['high_psych_risk'] = (np.array(total_psych_scores) >= 5).astype(int)
        features['dual_diagnosis'] = (
            (np.array(total_psych_scores) > 0) & 
            (np.array(total_substance_scores) > 0)
        ).astype(int)
        
        # Medication complexity
        total_psych_meds = sum(np.array(med_inferred[cat]) for cat in self.psych_medications)
        features['psych_med_complexity'] = total_psych_meds
        features['complex_psych_regimen'] = (total_psych_meds >= 3).astype(int)
        
        return features
    
    def learn(
        self, 
        text_list: List[str], 
        context: pd.DataFrame, 
        y: np.ndarray,
        med_list: Optional[List[str]] = None
    ) -> None:
        """Learn mental health risk patterns."""
        print(f"   [MENTAL] [{self.name}] Learning Mental Health Patterns...")
        
        X_vec = self.vectorizer.fit_transform(text_list)
        mental_features = self.extract_mental_features(text_list, med_list)
        X_final = np.column_stack([X_vec.toarray(), mental_features.values])
        
        print(f"      Training on {X_final.shape[1]} MENTAL-HEALTH features...")
        print(f"      Conditions detected: psych={mental_features['has_any_psych'].sum()}, "
              f"substance={mental_features['has_substance_use'].sum()}, "
              f"suicide_risk={mental_features['has_suicide_risk'].sum()}")
        self.model.fit(X_final, y)
        print(f"      ✅ Mental specialist training complete")
    
    def give_opinion(
        self, 
        text_list: List[str], 
        context: pd.DataFrame,
        med_list: Optional[List[str]] = None
    ) -> np.ndarray:
        """Generate risk opinion based on mental health status."""
        X_vec = self.vectorizer.transform(text_list)
        mental_features = self.extract_mental_features(text_list, med_list)
        X_final = np.column_stack([X_vec.toarray(), mental_features.values])
        return self.model.predict_proba(X_final)[:, 1]


# =============================================================================
# NEW: CARE SUPPORT SPECIALIST
# =============================================================================
class CareSupportSpecialist:
    """
    Care Coordination and Functional Status Specialist.
    
    INTELLIGENT INFERENCE:
    - Infers functional status from mobility aids in medication list
    - Detects frailty from nutritional supplements
    - Uses discharge planning notes for care coordination quality
    
    KEY FEATURES:
    - Functional status (ADLs, ambulation, mobility)
    - Frailty indicators (falls, malnutrition, weight loss)
    - Care coordination quality (discharge planning, follow-up)
    - Home care needs (equipment, services)
    - Caregiver availability and quality
    """
    
    def __init__(self) -> None:
        """Initialize Care Support Specialist."""
        self.name = "Spec_Care"
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,  # Reduced from 1500
            ngram_range=(1, 2),  # Reduced from (1,3)
            min_df=5  # Increased from 2
        )
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        # === FUNCTIONAL STATUS ===
        self.functional_patterns = {
            # Mobility Issues (HIGH RISK)
            'immobile': {
                'patterns': [r'bed.?bound', r'non.?ambulatory', r'unable\s*to\s*(?:walk|ambulate|stand)',
                           r'wheelchair.?bound', r'total\s*assist', r'dependent\s*for\s*mobility'],
                'weight': 3.5
            },
            'limited_mobility': {
                'patterns': [r'assist(?:ance)?\s*(?:with|for)\s*(?:ambulation|walking|mobility)',
                           r'walker', r'cane', r'rollator', r'gait\s*instab', r'unsteady\s*gait'],
                'weight': 2.0
            },
            # ADL Dependencies
            'adl_dependent': {
                'patterns': [r'ADL\s*(?:dependent|assistance|help)', r'(?:needs|requires)\s*help\s*with\s*(?:bathing|dressing|eating|toileting)',
                           r'unable\s*to\s*(?:bathe|dress|feed|toilet)', r'total\s*care'],
                'weight': 3.0
            },
            'toileting_issues': {
                'patterns': [r'incontinence', r'foley', r'catheter', r'colostomy', r'ileostomy',
                           r'ostomy', r'depends', r'briefs'],
                'weight': 2.0
            },
            # Cognitive-Functional
            'cognitive_functional': {
                'patterns': [r'wander(?:ing)?', r'safety\s*risk', r'impaired\s*judgment', 
                           r'unable\s*to\s*follow', r'24.?hour\s*supervision'],
                'weight': 2.5
            },
        }
        
        # === FRAILTY INDICATORS ===
        self.frailty_patterns = {
            'falls': {
                'patterns': [r'(?:history\s*of\s*)?fall(?:s|en)', r'fell', r'fall\s*risk', 
                           r'high\s*fall\s*risk', r'mechanical\s*fall'],
                'weight': 2.5
            },
            'malnutrition': {
                'patterns': [r'malnutri', r'failure\s*to\s*thrive', r'FTT', r'cachex',
                           r'weight\s*loss', r'poor\s*(?:oral\s*)?intake', r'decreased\s*appetite',
                           r'not\s*eating', r'poor\s*nutrition'],
                'weight': 2.5
            },
            'sarcopenia': {
                'patterns': [r'sarcopenia', r'muscle\s*(?:loss|wasting)', r'weakness', 
                           r'debilitat', r'decondition'],
                'weight': 2.0
            },
            'frailty_syndrome': {
                'patterns': [r'frail(?:ty)?', r'vulnerable', r'advanced\s*age', 
                           r'elderly\s*(?:patient|male|female)'],
                'weight': 2.0
            },
            'pressure_injury': {
                'patterns': [r'pressure\s*(?:ulcer|injury|sore)', r'decubitus', r'bedsore',
                           r'stage\s*[1234]\s*(?:ulcer|pressure)', r'wound\s*care'],
                'weight': 2.5
            },
        }
        
        # === CARE COORDINATION ===
        self.care_coordination = {
            # PROTECTIVE - Good coordination
            'case_management': {
                'patterns': [r'case\s*manag(?:er|ement)', r'care\s*coordinat', r'discharge\s*plann(?:er|ing)',
                           r'social\s*work(?:er)?', r'CM\s*consult'],
                'weight': -2.0
            },
            'pt_ot_involved': {
                'patterns': [r'PT\s*(?:consult|eval|recommend)', r'OT\s*(?:consult|eval|recommend)',
                           r'physical\s*therapy', r'occupational\s*therapy', r'rehab\s*consult'],
                'weight': -1.5
            },
            'home_care_arranged': {
                'patterns': [r'home\s*(?:health|care)\s*(?:arranged|ordered|set\s*up)',
                           r'VNA', r'visiting\s*nurse', r'home\s*PT', r'home\s*OT'],
                'weight': -2.5
            },
            'equipment_arranged': {
                'patterns': [r'DME\s*(?:ordered|arranged)', r'(?:walker|wheelchair|commode|hospital\s*bed)\s*(?:ordered|arranged)',
                           r'equipment\s*(?:ordered|arranged|provided)'],
                'weight': -1.5
            },
            # RISK - Poor coordination
            'care_gaps': {
                'patterns': [r'no\s*(?:follow.?up|PCP|primary\s*care)', r'unable\s*to\s*arrange',
                           r'patient\s*(?:declined|refused)\s*(?:services|home\s*care|PT|OT)',
                           r'no\s*transportation', r'coverage\s*(?:denied|issue)'],
                'weight': 2.5
            },
        }
        
        # === FOLLOW-UP QUALITY ===
        self.followup_patterns = {
            'pcp_followup': {
                'patterns': [r'PCP\s*(?:follow.?up|appointment)', r'primary\s*care\s*follow.?up',
                           r'see\s*(?:PCP|doctor)\s*(?:in|within)'],
                'weight': -2.0
            },
            'specialist_followup': {
                'patterns': [r'(?:cardiology|pulmonology|nephrology|neurology|oncology|GI)\s*(?:follow.?up|appointment)',
                           r'specialist\s*follow.?up'],
                'weight': -2.0
            },
            'close_followup': {
                'patterns': [r'(?:1|2|3)\s*(?:day|days)\s*follow.?up', r'close\s*follow.?up',
                           r'call\s*(?:back|tomorrow|in\s*\d+\s*days)'],
                'weight': -2.5
            },
            'no_followup': {
                'patterns': [r'no\s*follow.?up\s*(?:scheduled|arranged|needed)', r'lost\s*to\s*follow.?up',
                           r'patient\s*(?:declined|refused)\s*follow.?up'],
                'weight': 3.0
            },
        }
        
        # === CAREGIVER STATUS ===
        self.caregiver_patterns = {
            'has_caregiver': {
                'patterns': [r'caregiver', r'24.?hour\s*(?:care|supervision)', r'(?:family|spouse|daughter|son)\s*(?:at\s*home|present|available)',
                           r'support\s*(?:at\s*home|system)', r'lives\s*with\s*(?:family|spouse|children)'],
                'weight': -2.5
            },
            'caregiver_burden': {
                'patterns': [r'caregiver\s*(?:burden|stress|exhaust)', r'family\s*(?:overwhelmed|stressed|concern)',
                           r'difficult\s*(?:home|social)\s*situation'],
                'weight': 2.0
            },
            'lives_alone': {
                'patterns': [r'lives?\s*alone', r'no\s*(?:family|support|caregiver|one\s*at\s*home)',
                           r'isolated', r'poor\s*social\s*support'],
                'weight': 3.0
            },
        }
        
        # === INFERRED FROM EQUIPMENT/SUPPLIES ===
        self.equipment_proxy = {
            'mobility_aids': ['wheelchair', 'walker', 'cane', 'rollator', 'crutches', 'scooter'],
            'oxygen': ['oxygen', 'o2', 'concentrator', 'nasal cannula'],
            'nutrition_support': ['ensure', 'boost', 'glucerna', 'tube feeding', 'peg', 'ng tube'],
            'wound_care': ['wound vac', 'dressing', 'tegaderm', 'wound care'],
            'diabetes_supplies': ['glucometer', 'test strips', 'lancets', 'insulin syringes'],
        }
    
    def extract_care_features(
        self, 
        text_list: List[str],
        med_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract care support features with intelligent inference.
        
        Args:
            text_list: List of clinical note texts
            med_list: Optional medication texts for supply inference
        """
        features = pd.DataFrame()
        
        functional_flags = {func: [] for func in self.functional_patterns}
        frailty_flags = {frail: [] for frail in self.frailty_patterns}
        care_flags = {care: [] for care in self.care_coordination}
        followup_flags = {fu: [] for fu in self.followup_patterns}
        caregiver_flags = {cg: [] for cg in self.caregiver_patterns}
        equipment_flags = {eq: [] for eq in self.equipment_proxy}
        
        functional_scores = []
        frailty_scores = []
        care_scores = []
        
        for idx, text in enumerate(text_list):
            text_lower = str(text).lower()
            
            func_score = 0.0
            frail_score = 0.0
            care_score = 0.0
            
            # === FUNCTIONAL STATUS ===
            for func, config in self.functional_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                functional_flags[func].append(1 if found else 0)
                if found:
                    func_score += config['weight']
            
            # === FRAILTY ===
            for frail, config in self.frailty_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                frailty_flags[frail].append(1 if found else 0)
                if found:
                    frail_score += config['weight']
            
            # === CARE COORDINATION ===
            for care, config in self.care_coordination.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                care_flags[care].append(1 if found else 0)
                if found:
                    care_score += config['weight']
            
            # === FOLLOW-UP ===
            for fu, config in self.followup_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                followup_flags[fu].append(1 if found else 0)
                if found:
                    care_score += config['weight']
            
            # === CAREGIVER ===
            for cg, config in self.caregiver_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                caregiver_flags[cg].append(1 if found else 0)
                if found:
                    care_score += config['weight']
            
            # === EQUIPMENT INFERENCE ===
            med_text = med_list[idx] if med_list and idx < len(med_list) else ""
            combined_text = text_lower + " " + str(med_text).lower()
            
            for eq_cat, items in self.equipment_proxy.items():
                found = any(item in combined_text for item in items)
                equipment_flags[eq_cat].append(1 if found else 0)
                if found:
                    if eq_cat == 'mobility_aids':
                        func_score += 1.0  # Infer mobility limitation
                    elif eq_cat == 'oxygen':
                        func_score += 1.5  # Infer respiratory dependency
                    elif eq_cat == 'nutrition_support':
                        frail_score += 1.5  # Infer nutritional vulnerability
            
            functional_scores.append(func_score)
            frailty_scores.append(frail_score)
            care_scores.append(care_score)
        
        # Add flags
        for func, flags in functional_flags.items():
            features[f'functional_{func}'] = flags
        
        for frail, flags in frailty_flags.items():
            features[f'frailty_{frail}'] = flags
        
        for care, flags in care_flags.items():
            features[f'care_{care}'] = flags
        
        for fu, flags in followup_flags.items():
            features[f'followup_{fu}'] = flags
        
        for cg, flags in caregiver_flags.items():
            features[f'caregiver_{cg}'] = flags
        
        for eq, flags in equipment_flags.items():
            features[f'equipment_{eq}'] = flags
        
        # Aggregate scores
        features['functional_score'] = functional_scores
        features['frailty_score'] = frailty_scores
        features['care_score'] = care_scores
        
        # Composite measures
        features['total_care_risk'] = (
            np.array(functional_scores) + 
            np.array(frailty_scores) + 
            np.array(care_scores)
        )
        
        # High-risk flags
        features['high_functional_risk'] = (np.array(functional_scores) >= 5).astype(int)
        features['high_frailty_risk'] = (np.array(frailty_scores) >= 5).astype(int)
        features['care_gap_risk'] = (np.array(care_scores) >= 2).astype(int)
        
        # Protective flags
        features['good_care_coordination'] = (np.array(care_scores) <= -3).astype(int)
        features['has_support_system'] = [
            1 if caregiver_flags['has_caregiver'][i] and not caregiver_flags['lives_alone'][i] else 0
            for i in range(len(text_list))
        ]
        
        return features
    
    def learn(
        self, 
        text_list: List[str], 
        context: pd.DataFrame, 
        y: np.ndarray,
        med_list: Optional[List[str]] = None
    ) -> None:
        """Learn care support risk patterns."""
        print(f"   [CARE] [{self.name}] Learning Care Support Patterns...")
        
        X_vec = self.vectorizer.fit_transform(text_list)
        care_features = self.extract_care_features(text_list, med_list)
        X_final = np.column_stack([X_vec.toarray(), care_features.values])
        
        print(f"      Training on {X_final.shape[1]} CARE-SUPPORT features...")
        print(f"      High frailty={care_features['high_frailty_risk'].sum()}, "
              f"care_gaps={care_features['care_gap_risk'].sum()}, "
              f"good_coordination={care_features['good_care_coordination'].sum()}")
        self.model.fit(X_final, y)
        print(f"      ✅ Care Support specialist training complete")
    
    def give_opinion(
        self, 
        text_list: List[str], 
        context: pd.DataFrame,
        med_list: Optional[List[str]] = None
    ) -> np.ndarray:
        """Generate risk opinion based on care support status."""
        X_vec = self.vectorizer.transform(text_list)
        care_features = self.extract_care_features(text_list, med_list)
        X_final = np.column_stack([X_vec.toarray(), care_features.values])
        return self.model.predict_proba(X_final)[:, 1]


# =============================================================================
# NEW: SOCIAL SUPPORT SPECIALIST
# =============================================================================
class SocialSupportSpecialist:
    """
    Social Determinants of Health and Financial Barriers Specialist.
    
    INTELLIGENT INFERENCE:
    - Infers financial status from insurance type proxies (Medicaid → lower SES)
    - Uses medication complexity as proxy for healthcare engagement
    - Detects social barriers from discharge planning notes
    
    KEY FEATURES:
    - Housing status (homeless, unstable, institutional)
    - Financial barriers (insurance issues, can't afford medications)
    - Transportation access
    - Language/literacy barriers
    - Social isolation and support network
    - Employment/disability status
    """
    
    def __init__(self) -> None:
        """Initialize Social Support Specialist."""
        self.name = "Spec_Social"
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,  # Reduced from 1500
            ngram_range=(1, 2),  # Reduced from (1,3)
            min_df=5  # Increased from 2
        )
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        # === HOUSING STATUS ===
        self.housing_patterns = {
            'homeless': {
                'patterns': [r'homeless', r'undomiciled', r'shelter', r'no\s*(?:fixed\s*)?address',
                           r'lives\s*(?:on|in)\s*(?:street|car|vehicle)', r'housing\s*insecure'],
                'weight': 5.0
            },
            'unstable_housing': {
                'patterns': [r'unstable\s*housing', r'temporary\s*housing', r'staying\s*with',
                           r'couch\s*surfing', r'transitional\s*housing', r'evicted', r'eviction'],
                'weight': 3.5
            },
            'institutional': {
                'patterns': [r'nursing\s*home', r'assisted\s*living', r'group\s*home',
                           r'adult\s*foster', r'board\s*care', r'residential\s*facility'],
                'weight': 1.0  # Neutral - supervised but limited independence
            },
            'independent': {
                'patterns': [r'lives\s*(?:at\s*)?home', r'independent(?:ly)?', r'own\s*(?:home|apartment)',
                           r'stable\s*housing'],
                'weight': -1.5
            },
        }
        
        # === FINANCIAL BARRIERS ===
        self.financial_patterns = {
            'uninsured': {
                'patterns': [r'uninsured', r'no\s*(?:health\s*)?insurance', r'self.?pay',
                           r'no\s*coverage', r'insurance\s*(?:lapsed|expired)'],
                'weight': 4.0
            },
            'insurance_issues': {
                'patterns': [r'insurance\s*(?:issue|problem|denied|barrier)', r'coverage\s*(?:denied|issue|gap)',
                           r'prior\s*auth(?:orization)?\s*(?:needed|denied|pending)'],
                'weight': 2.5
            },
            'medication_cost': {
                'patterns': [r'(?:cannot|can\'t|unable\s*to)\s*afford', r'medication\s*(?:cost|expense)',
                           r'copay\s*(?:issue|barrier)', r'financial\s*(?:barrier|issue|hardship)',
                           r'unable\s*to\s*(?:pay|purchase|get)\s*medication'],
                'weight': 3.0
            },
            'disability': {
                'patterns': [r'on\s*disability', r'SSDI', r'SSI', r'disabled', r'disability\s*(?:income|benefit)'],
                'weight': 1.5  # Risk factor for limited resources
            },
            'unemployed': {
                'patterns': [r'unemployed', r'not\s*working', r'lost\s*job', r'laid\s*off',
                           r'between\s*jobs'],
                'weight': 2.0
            },
        }
        
        # === TRANSPORTATION ===
        self.transportation_patterns = {
            'no_transportation': {
                'patterns': [r'no\s*transportation', r'transportation\s*(?:issue|barrier|problem)',
                           r'cannot\s*(?:get\s*to|drive)', r'unable\s*to\s*(?:drive|travel)',
                           r'no\s*(?:car|ride|way\s*to\s*get)'],
                'weight': 3.0
            },
            'transportation_arranged': {
                'patterns': [r'transportation\s*(?:arranged|provided|scheduled)',
                           r'medical\s*transportation', r'medicaid\s*transport', r'uber\s*health'],
                'weight': -1.5
            },
        }
        
        # === LANGUAGE/LITERACY ===
        self.language_patterns = {
            'limited_english': {
                'patterns': [r'(?:limited|poor|no)\s*english', r'non.?english\s*speak',
                           r'interpreter\s*(?:needed|used|present)', r'language\s*barrier',
                           r'(?:spanish|chinese|vietnamese|russian)\s*(?:speak|only)'],
                'weight': 2.0
            },
            'low_literacy': {
                'patterns': [r'low\s*(?:health\s*)?literacy', r'cannot\s*read', r'unable\s*to\s*read',
                           r'limited\s*education', r'difficulty\s*understanding'],
                'weight': 2.5
            },
        }
        
        # === SOCIAL ISOLATION ===
        self.isolation_patterns = {
            'isolated': {
                'patterns': [r'(?:social(?:ly)?|emotional(?:ly)?)\s*isolated', r'lonely', 
                           r'no\s*(?:friends|contacts|visitors)', r'limited\s*social\s*(?:contact|support)',
                           r'widowed', r'lives?\s*alone'],
                'weight': 2.5
            },
            'no_family': {
                'patterns': [r'no\s*(?:family|relatives|next\s*of\s*kin)', r'estranged\s*from\s*family',
                           r'family\s*not\s*(?:available|involved)', r'no\s*emergency\s*contact'],
                'weight': 3.0
            },
            'good_support': {
                'patterns': [r'good\s*(?:family|social)\s*support', r'strong\s*support\s*(?:system|network)',
                           r'family\s*(?:very\s*)?(?:involved|supportive|present)',
                           r'(?:children|spouse|partner)\s*(?:involved|supportive|available)'],
                'weight': -3.0
            },
        }
        
        # === BEHAVIORAL/COMPLIANCE ===
        self.compliance_patterns = {
            'noncompliant': {
                'patterns': [r'non.?complian(?:t|ce)', r'medication\s*non.?adherence',
                           r'not\s*taking\s*medications?', r'missed\s*(?:doses|medications?|appointments?)',
                           r'poor\s*(?:compliance|adherence)'],
                'weight': 3.0
            },
            'ama': {
                'patterns': [r'left\s*(?:against|AMA)', r'against\s*medical\s*advice', 
                           r'signed\s*out\s*AMA', r'eloped'],
                'weight': 4.0
            },
            'frequent_ed': {
                'patterns': [r'frequent\s*(?:ED|ER|emergency)', r'multiple\s*ED\s*visits',
                           r'ED\s*(?:utiliz|frequent)', r'super.?utilizer'],
                'weight': 2.5
            },
        }
        
        # === INSURANCE TYPE PROXIES (intelligent inference) ===
        # These patterns help infer socioeconomic status
        self.insurance_proxy_patterns = {
            'medicaid': {
                'patterns': [r'medicaid', r'medical\s*assistance', r'medi.?cal',
                           r'state\s*(?:insurance|assistance)'],
                'inferred_risk': 1.5  # Higher social vulnerability proxy
            },
            'medicare': {
                'patterns': [r'medicare(?:\s*only)?', r'cms'],
                'inferred_risk': 0.5  # Moderate - usually elderly
            },
            'dual_eligible': {
                'patterns': [r'dual.?eligible', r'medicare.?medicaid', r'dually\s*eligible'],
                'inferred_risk': 2.0  # High vulnerability
            },
            'commercial': {
                'patterns': [r'(?:private|commercial)\s*insurance', r'employer\s*(?:sponsored|based)',
                           r'BCBS', r'aetna', r'cigna', r'united', r'humana'],
                'inferred_risk': -1.0  # Protective proxy
            },
        }
    
    def extract_social_features(
        self, 
        text_list: List[str],
        med_list: Optional[List[str]] = None,
        context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract social support features with intelligent inference.
        
        Args:
            text_list: List of clinical note texts
            med_list: Optional medication texts
            context: Optional context DataFrame with numeric features
        """
        features = pd.DataFrame()
        
        housing_flags = {h: [] for h in self.housing_patterns}
        financial_flags = {f: [] for f in self.financial_patterns}
        transport_flags = {t: [] for t in self.transportation_patterns}
        language_flags = {l: [] for l in self.language_patterns}
        isolation_flags = {i: [] for i in self.isolation_patterns}
        compliance_flags = {c: [] for c in self.compliance_patterns}
        insurance_proxy_flags = {ins: [] for ins in self.insurance_proxy_patterns}
        
        housing_scores = []
        financial_scores = []
        social_scores = []
        total_sdoh_scores = []
        
        for idx, text in enumerate(text_list):
            text_lower = str(text).lower()
            
            housing_score = 0.0
            financial_score = 0.0
            social_score = 0.0
            
            # === HOUSING ===
            for housing, config in self.housing_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                housing_flags[housing].append(1 if found else 0)
                if found:
                    housing_score += config['weight']
            
            # === FINANCIAL ===
            for fin, config in self.financial_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                financial_flags[fin].append(1 if found else 0)
                if found:
                    financial_score += config['weight']
            
            # === TRANSPORTATION ===
            for trans, config in self.transportation_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                transport_flags[trans].append(1 if found else 0)
                if found:
                    social_score += config['weight']
            
            # === LANGUAGE ===
            for lang, config in self.language_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                language_flags[lang].append(1 if found else 0)
                if found:
                    social_score += config['weight']
            
            # === ISOLATION ===
            for iso, config in self.isolation_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                isolation_flags[iso].append(1 if found else 0)
                if found:
                    social_score += config['weight']
            
            # === COMPLIANCE ===
            for comp, config in self.compliance_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                compliance_flags[comp].append(1 if found else 0)
                if found:
                    social_score += config['weight']
            
            # === INSURANCE PROXY INFERENCE ===
            for ins, config in self.insurance_proxy_patterns.items():
                found = any(re.search(p, text_lower, re.I) for p in config['patterns'])
                insurance_proxy_flags[ins].append(1 if found else 0)
                if found:
                    financial_score += config['inferred_risk']
            
            housing_scores.append(housing_score)
            financial_scores.append(financial_score)
            social_scores.append(social_score)
            total_sdoh_scores.append(housing_score + financial_score + social_score)
        
        # Add flags
        for h, flags in housing_flags.items():
            features[f'housing_{h}'] = flags
        
        for f, flags in financial_flags.items():
            features[f'financial_{f}'] = flags
        
        for t, flags in transport_flags.items():
            features[f'transport_{t}'] = flags
        
        for l, flags in language_flags.items():
            features[f'language_{l}'] = flags
        
        for i, flags in isolation_flags.items():
            features[f'isolation_{i}'] = flags
        
        for c, flags in compliance_flags.items():
            features[f'compliance_{c}'] = flags
        
        for ins, flags in insurance_proxy_flags.items():
            features[f'insurance_{ins}'] = flags
        
        # Aggregate scores
        features['housing_score'] = housing_scores
        features['financial_score'] = financial_scores
        features['social_score'] = social_scores
        features['total_sdoh_score'] = total_sdoh_scores
        
        # High-risk flags
        features['homeless_risk'] = [
            1 if housing_flags['homeless'][i] or housing_flags['unstable_housing'][i] else 0
            for i in range(len(text_list))
        ]
        features['high_financial_risk'] = (np.array(financial_scores) >= 4).astype(int)
        features['high_social_risk'] = (np.array(social_scores) >= 5).astype(int)
        features['high_sdoh_risk'] = (np.array(total_sdoh_scores) >= 8).astype(int)
        
        # Protective flags
        features['good_social_support'] = [
            1 if isolation_flags['good_support'][i] and not isolation_flags['isolated'][i] else 0
            for i in range(len(text_list))
        ]
        features['stable_housing'] = [
            1 if housing_flags['independent'][i] and not (housing_flags['homeless'][i] or housing_flags['unstable_housing'][i]) else 0
            for i in range(len(text_list))
        ]
        
        # Compliance risk
        features['high_compliance_risk'] = [
            1 if compliance_flags['noncompliant'][i] or compliance_flags['ama'][i] else 0
            for i in range(len(text_list))
        ]
        
        # Inferred SES from insurance (intelligent proxy)
        features['low_ses_proxy'] = [
            1 if (insurance_proxy_flags['medicaid'][i] or insurance_proxy_flags['dual_eligible'][i])
                 and not insurance_proxy_flags['commercial'][i] else 0
            for i in range(len(text_list))
        ]
        
        return features
    
    def learn(
        self, 
        text_list: List[str], 
        context: pd.DataFrame, 
        y: np.ndarray,
        med_list: Optional[List[str]] = None
    ) -> None:
        """Learn social support risk patterns."""
        print(f"   [SOCIAL] [{self.name}] Learning Social Determinants Patterns...")
        
        X_vec = self.vectorizer.fit_transform(text_list)
        social_features = self.extract_social_features(text_list, med_list, context)
        X_final = np.column_stack([X_vec.toarray(), social_features.values])
        
        print(f"      Training on {X_final.shape[1]} SOCIAL-SUPPORT features...")
        print(f"      Homeless={social_features['homeless_risk'].sum()}, "
              f"high_financial={social_features['high_financial_risk'].sum()}, "
              f"good_support={social_features['good_social_support'].sum()}, "
              f"low_SES_proxy={social_features['low_ses_proxy'].sum()}")
        self.model.fit(X_final, y)
        print(f"      ✅ Social Support specialist training complete")
    
    def give_opinion(
        self, 
        text_list: List[str], 
        context: pd.DataFrame,
        med_list: Optional[List[str]] = None
    ) -> np.ndarray:
        """Generate risk opinion based on social support status."""
        X_vec = self.vectorizer.transform(text_list)
        social_features = self.extract_social_features(text_list, med_list, context)
        X_final = np.column_stack([X_vec.toarray(), social_features.values])
        return self.model.predict_proba(X_final)[:, 1]


# =============================================================================
# UNIFIED PSYCHOSOCIAL SPECIALIST (Combines Mental, Care, Social)
# =============================================================================
class PsychosocialSpecialist:
    """
    Unified Psychosocial Specialist that combines:
    - MentalSpecialist (psychiatric, substance use)
    - CareSupportSpecialist (functional status, frailty, care coordination)
    - SocialSupportSpecialist (housing, financial, transportation, isolation)
    
    Returns unified probability + sub-scores for interpretability.
    """
    
    def __init__(self) -> None:
        """Initialize unified psychosocial specialist with three sub-specialists."""
        self.name = "Spec_Psychosocial"
        
        # Initialize sub-specialists (with regularized hyperparameters from global constants)
        self.mental = MentalSpecialist()
        self.care = CareSupportSpecialist()
        self.social = SocialSupportSpecialist()
        
        # Ensemble weights (can be tuned)
        self.weights = {
            'mental': 0.35,
            'care': 0.35,
            'social': 0.30
        }
        
        self._is_trained = False
    
    def learn(
        self,
        text_list: List[str],
        context: pd.DataFrame,
        y: np.ndarray,
        med_list: Optional[List[str]] = None
    ) -> None:
        """Train all three sub-specialists."""
        print(f"   [PSYCHOSOCIAL] [{self.name}] Training Unified Psychosocial Specialist...")
        print("-" * 50)
        
        # Train each sub-specialist
        self.mental.learn(text_list, context, y, med_list)
        self.care.learn(text_list, context, y, med_list)
        self.social.learn(text_list, context, y, med_list)
        
        self._is_trained = True
        print(f"      ✅ Unified Psychosocial specialist training complete")
    
    def give_opinion(
        self,
        text_list: List[str],
        context: pd.DataFrame,
        med_list: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate unified psychosocial risk opinion with sub-scores.
        
        Returns:
            Tuple of (unified_probability, has_data, sub_scores_dict)
            - unified_probability: Weighted ensemble of three specialists
            - has_data: Binary flags indicating if patient has psychosocial data
            - sub_scores_dict: {'mental': array, 'care': array, 'social': array}
        """
        if not self._is_trained:
            raise ValueError("PsychosocialSpecialist must be trained before giving opinions")
        
        # Detect missing data: check if notes or meds are available
        # Missing psychosocial indicators could mean data quality issues or minimal complexity
        has_note_data = np.array([len(str(text).strip()) > 10 for text in text_list], dtype=int)
        has_med_data = np.array([len(str(m).strip()) > 5 for m in (med_list if med_list else text_list)], dtype=int)
        has_data = np.maximum(has_note_data, has_med_data)  # Has data if either is available
        
        # Get opinions from each sub-specialist
        mental_prob = self.mental.give_opinion(text_list, context, med_list)
        care_prob = self.care.give_opinion(text_list, context, med_list)
        social_prob = self.social.give_opinion(text_list, context, med_list)
        
        # Weighted ensemble
        unified_prob = (
            self.weights['mental'] * mental_prob +
            self.weights['care'] * care_prob +
            self.weights['social'] * social_prob
        )
        
        # Keep actual prediction - Doctor learns to weight by has_data
        
        # Sub-scores for interpretability
        sub_scores = {
            'mental': mental_prob,
            'care': care_prob,
            'social': social_prob
        }
        
        return unified_prob, has_data, sub_scores
    
    def get_detailed_features(
        self,
        text_list: List[str],
        med_list: Optional[List[str]] = None,
        context: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract detailed psychosocial features from all three specialists.
        Useful for explainability and analysis.
        """
        all_features = pd.DataFrame()
        
        # Mental features
        mental_features = self.mental.extract_mental_features(text_list, med_list)
        for col in mental_features.columns:
            all_features[f'mental_{col}'] = mental_features[col]
        
        # Care features
        care_features = self.care.extract_care_features(text_list, med_list)
        for col in care_features.columns:
            all_features[f'care_{col}'] = care_features[col]
        
        # Social features
        social_features = self.social.extract_social_features(text_list, med_list, context)
        for col in social_features.columns:
            all_features[f'social_{col}'] = social_features[col]
        
        return all_features