"""
SPECIALIST AGENTS - STRICT MODE (NO FALLBACKS)
==============================================

IMPORTANT: This file should REPLACE your existing specialist_agents.py

This version requires all dependencies to be present.
No TF-IDF fallback - uses ClinicalBERT only.

Key Requirements:
1. torch and sentence-transformers must be installed
2. clinicalbert_cache.py must be present
3. GPU recommended (but CPU works)

If any dependency is missing, an error is raised immediately.

Specialists:
1. LabSpecialist - Lab values and organ dysfunction with trajectory features
2. NoteSpecialist - Clinical notes with ClinicalBERT embeddings + LDA topics
3. PharmacySpecialist - Medication-based risk
4. HistorySpecialist - Diagnosis history patterns
5. PsychosocialSpecialist - Mental, Care, Social sub-specialists

USAGE:
------
    # Rename this file to replace the original:
    mv specialist_agents_strict.py specialist_agents.py
    
    # Or copy over:
    cp specialist_agents_strict.py specialist_agents.py
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

# =============================================================================
# STRICT DEPENDENCY CHECKING - NO FALLBACKS
# =============================================================================

# Check for ClinicalBERT cache
try:
    from clinicalbert_cache import ClinicalBERTCache
    print("      ✅ ClinicalBERT cache loaded successfully")
except ImportError as e:
    raise ImportError(
        "❌ REQUIRED: clinicalbert_cache.py not found!\n"
        "   Please ensure clinicalbert_cache.py is in your project directory.\n"
        f"   Original error: {e}"
    )

# Check for PyTorch and SentenceTransformers
try:
    import torch
    from sentence_transformers import SentenceTransformer
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    print(f"      ✅ PyTorch loaded successfully (Device: {DEVICE})")
except ImportError as e:
    raise ImportError(
        "❌ REQUIRED: torch and sentence-transformers not installed!\n"
        "   Install with: pip install torch sentence-transformers\n"
        f"   Original error: {e}"
    )

# Verify GPU availability
if CUDA_AVAILABLE:
    print(f"      ✅ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print(f"      ⚠️ WARNING: Running on CPU - this will be slower")

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
        return df[col_name].fillna(default_value)
    return pd.Series([default_value] * len(df), index=df.index)


def get_note_keywords() -> Dict[str, List[str]]:
    """
    Return dictionaries of clinical keywords for note analysis.
    Updated to balance risk and protective indicators.
    """
    return {
        # Risk indicators
        'high_risk': [
            'unstable', 'critical', 'sepsis', 'intubated', 'icu', 'code',
            'deteriorating', 'unresponsive', 'emergent', 'acute', 'severe',
            'shock', 'failure', 'respiratory distress', 'cardiac arrest',
            'altered mental status', 'hypotensive', 'bleeding', 'infection'
        ],
        'readmission_risk': [
            'readmission', 'frequent flyer', 'multiple admissions', 'non-compliant',
            'poor follow-up', 'missed appointments', 'left ama', 'ama',
            'against medical advice', 'medication non-adherence', 'non-adherent',
            'social issues', 'no support', 'homeless', 'substance abuse',
            'alcohol', 'drug use', 'psychiatric', 'mental health crisis'
        ],
        'complexity': [
            'multiple comorbidities', 'complex', 'complicated', 'multiorgan',
            'polypharmacy', 'dialysis', 'transplant', 'immunocompromised',
            'cancer', 'malignancy', 'metastatic', 'palliative', 'hospice',
            'end stage', 'chronic', 'recurrent', 'resistant'
        ],
        
        # Protective indicators (ENHANCED)
        'protective': [
            'stable', 'improved', 'improving', 'good progress', 'recovered',
            'tolerating', 'ambulatory', 'independent', 'self-care',
            'family support', 'good support', 'caregiver', 'daughter',
            'son', 'wife', 'husband', 'spouse', 'lives with',
            'follow-up scheduled', 'appointment', 'outpatient',
            'discharge home', 'home health', 'visiting nurse',
            'rehabilitation', 'skilled nursing', 'snf',
            'understands', 'compliant', 'adherent', 'educated',
            'verbalized understanding', 'will follow', 'agrees to'
        ],
        'discharge_ready': [
            'ready for discharge', 'cleared for discharge', 'medically stable',
            'stable for discharge', 'discharge today', 'home today',
            'tolerating diet', 'pain controlled', 'afebrile',
            'ambulating well', 'no acute issues', 'at baseline'
        ]
    }


# =============================================================================
# LAB SPECIALIST
# =============================================================================
class LabSpecialist:
    """
    Laboratory values specialist with organ dysfunction detection.
    Analyzes lab panels for abnormalities, trends, and critical values.
    """
    
    def __init__(self) -> None:
        """Initialize Lab Specialist."""
        self.name = "Spec_Lab"
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
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def _extract_lab_features(self, X_labs: pd.DataFrame, X_context: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive lab-based features.
        
        Key features:
        - Raw lab values (mean, first, last, std)
        - Organ dysfunction scores
        - Lab trajectories (worsening/improving)
        - Critical value flags
        """
        features = pd.DataFrame(index=X_labs.index)
        
        # Core lab values
        lab_cols = [c for c in X_labs.columns if any(x in c.lower() for x in 
                    ['sodium', 'potassium', 'creatinine', 'bun', 'glucose', 
                     'hemoglobin', 'wbc', 'platelets', 'albumin', 'bilirubin',
                     'lactate', 'inr', 'troponin', 'bnp'])]
        
        for col in lab_cols:
            features[col] = X_labs[col].fillna(0)
        
        # Organ dysfunction: Renal
        if 'creatinine_mean' in X_labs.columns:
            features['renal_dysfunction'] = (X_labs['creatinine_mean'].fillna(1.0) > 1.5).astype(int)
            features['severe_aki'] = (X_labs['creatinine_mean'].fillna(1.0) > 3.0).astype(int)
        
        if 'bun_mean' in X_labs.columns and 'creatinine_mean' in X_labs.columns:
            creat = X_labs['creatinine_mean'].fillna(1.0).replace(0, 1.0)
            features['bun_creat_ratio'] = X_labs['bun_mean'].fillna(15) / creat
            features['prerenal_pattern'] = (features['bun_creat_ratio'] > 20).astype(int)
        
        # Organ dysfunction: Hepatic
        if 'bilirubin_mean' in X_labs.columns:
            features['liver_dysfunction'] = (X_labs['bilirubin_mean'].fillna(1.0) > 2.0).astype(int)
        
        if 'albumin_mean' in X_labs.columns:
            features['hypoalbuminemia'] = (X_labs['albumin_mean'].fillna(4.0) < 3.0).astype(int)
            features['severe_hypoalbuminemia'] = (X_labs['albumin_mean'].fillna(4.0) < 2.5).astype(int)
        
        # Organ dysfunction: Hematologic
        if 'hemoglobin_mean' in X_labs.columns:
            features['anemia'] = (X_labs['hemoglobin_mean'].fillna(12) < 10).astype(int)
            features['severe_anemia'] = (X_labs['hemoglobin_mean'].fillna(12) < 7).astype(int)
        
        if 'platelets_mean' in X_labs.columns:
            features['thrombocytopenia'] = (X_labs['platelets_mean'].fillna(200) < 100).astype(int)
        
        if 'wbc_mean' in X_labs.columns:
            features['leukocytosis'] = (X_labs['wbc_mean'].fillna(8) > 12).astype(int)
            features['leukopenia'] = (X_labs['wbc_mean'].fillna(8) < 4).astype(int)
        
        # Metabolic derangements
        if 'sodium_mean' in X_labs.columns:
            features['hyponatremia'] = (X_labs['sodium_mean'].fillna(140) < 130).astype(int)
            features['hypernatremia'] = (X_labs['sodium_mean'].fillna(140) > 150).astype(int)
        
        if 'potassium_mean' in X_labs.columns:
            features['hypokalemia'] = (X_labs['potassium_mean'].fillna(4.0) < 3.0).astype(int)
            features['hyperkalemia'] = (X_labs['potassium_mean'].fillna(4.0) > 5.5).astype(int)
        
        if 'glucose_mean' in X_labs.columns:
            features['hyperglycemia'] = (X_labs['glucose_mean'].fillna(100) > 200).astype(int)
            features['hypoglycemia'] = (X_labs['glucose_mean'].fillna(100) < 70).astype(int)
        
        # Critical markers
        if 'lactate_mean' in X_labs.columns:
            features['elevated_lactate'] = (X_labs['lactate_mean'].fillna(1.0) > 2.0).astype(int)
            features['severe_lactate'] = (X_labs['lactate_mean'].fillna(1.0) > 4.0).astype(int)
        
        if 'troponin_mean' in X_labs.columns:
            features['elevated_troponin'] = (X_labs['troponin_mean'].fillna(0) > 0.04).astype(int)
        
        if 'bnp_mean' in X_labs.columns:
            features['elevated_bnp'] = (X_labs['bnp_mean'].fillna(100) > 400).astype(int)
        
        # Coagulopathy
        if 'inr_mean' in X_labs.columns:
            features['coagulopathy'] = (X_labs['inr_mean'].fillna(1.0) > 1.5).astype(int)
        
        # Lab trajectory features (first vs last)
        trajectory_labs = ['creatinine', 'hemoglobin', 'wbc', 'platelets', 'sodium', 'potassium']
        for lab in trajectory_labs:
            first_col = f'{lab}_first'
            last_col = f'{lab}_last'
            if first_col in X_labs.columns and last_col in X_labs.columns:
                first_val = X_labs[first_col].fillna(0)
                last_val = X_labs[last_col].fillna(0)
                features[f'{lab}_delta'] = last_val - first_val
                features[f'{lab}_improving'] = (features[f'{lab}_delta'] < 0).astype(int) if lab in ['creatinine', 'wbc'] else (features[f'{lab}_delta'] > 0).astype(int)
        
        # Aggregate dysfunction score
        dysfunction_cols = [c for c in features.columns if 'dysfunction' in c or 'severe' in c]
        if dysfunction_cols:
            features['total_organ_dysfunction'] = features[dysfunction_cols].sum(axis=1)
        
        # Lab instability (high variance)
        std_cols = [c for c in X_labs.columns if '_std' in c]
        if std_cols:
            features['lab_instability'] = X_labs[std_cols].fillna(0).mean(axis=1)
        
        # Context features that help interpretation
        if 'had_icu_stay' in X_context.columns:
            features['icu_with_lab_issues'] = (
                (X_context['had_icu_stay'] == 1) & 
                (features.get('total_organ_dysfunction', pd.Series(0, index=features.index)) > 0)
            ).astype(int)
        
        return features
    
    def learn(self, X_labs: pd.DataFrame, X_context: pd.DataFrame, y: np.ndarray) -> None:
        """Train the lab specialist."""
        features = self._extract_lab_features(X_labs, X_context)
        self.feature_names = features.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(features.fillna(0))
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def give_opinion(self, X_labs: pd.DataFrame, X_context: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("LabSpecialist must be trained before giving opinions!")
        
        features = self._extract_lab_features(X_labs, X_context)
        
        # Add any missing columns from training
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        features = features[self.feature_names]
        
        X_scaled = self.scaler.transform(features.fillna(0))
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Determine if we have meaningful lab data
        has_data = (X_labs.notna().sum(axis=1) > 3).astype(float).values
        
        return probs, has_data


# =============================================================================
# NOTE SPECIALIST - CLINICALBERT ONLY (NO FALLBACK)
# =============================================================================
class NoteSpecialist:
    """
    Clinical notes specialist with ClinicalBERT embeddings.
    
    STRICT MODE: Requires SentenceTransformer - no TF-IDF fallback.
    This ensures maximum accuracy from clinical note analysis.
    """
    
    def __init__(self) -> None:
        """Initialize Note Specialist with ClinicalBERT encoder."""
        self.name = "Spec_Notes"
        
        # Load ClinicalBERT encoder (REQUIRED - no fallback)
        print(f"      Loading ClinicalBERT encoder...")
        self.encoder = SentenceTransformer(
            'pritamdeka/S-PubMedBert-MS-MARCO', 
            device=DEVICE
        )
        print(f"      ✅ NoteSpecialist: ClinicalBERT on {DEVICE.upper()}")
        
        # Initialize cache (REQUIRED)
        self.cache = ClinicalBERTCache()
        print(f"      ✅ NoteSpecialist: ClinicalBERT cache enabled")
        
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
        
        # Section patterns for extraction
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
        
        # Keyword dictionaries
        self.keywords = get_note_keywords()
        
        # Pre-compile keyword patterns for speed
        self._keyword_patterns = {}
        for category, words in self.keywords.items():
            pattern = '|'.join(re.escape(w) for w in words)
            self._keyword_patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        self.is_fitted = False
    
    def _clean_text_for_lda(self, text: str) -> str:
        """Clean text for LDA processing."""
        text = self._lda_clean_patterns['deident'].sub(' ', text)
        text = self._lda_clean_patterns['punctuation'].sub(' ', text)
        text = self._lda_clean_patterns['whitespace'].sub(' ', text.lower())
        return text.strip()
    
    def _fit_lda(self, texts: List[str]) -> None:
        """Fit LDA topic model on training texts."""
        print(f"      Fitting LDA topic model ({LDA_N_TOPICS} topics)...")
        
        cleaned_texts = [self._clean_text_for_lda(t) for t in texts]
        
        self.count_vectorizer = CountVectorizer(
            max_features=2000,
            min_df=5,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.count_vectorizer.fit_transform(cleaned_texts)
        
        self.lda = LatentDirichletAllocation(
            n_components=LDA_N_TOPICS,
            max_iter=LDA_MAX_ITER,
            learning_method='online',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.lda.fit(doc_term_matrix)
        self.lda_fitted = True
    
    def _get_lda_features(self, texts: List[str]) -> np.ndarray:
        """Get LDA topic distribution for texts."""
        if not self.lda_fitted:
            return np.zeros((len(texts), LDA_N_TOPICS))
        
        cleaned_texts = [self._clean_text_for_lda(t) for t in texts]
        doc_term_matrix = self.count_vectorizer.transform(cleaned_texts)
        topic_distributions = self.lda.transform(doc_term_matrix)
        
        return topic_distributions
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from clinical note."""
        if section_name not in self.section_patterns:
            return ""
        
        pattern = self.section_patterns[section_name]
        match = pattern.search(text)
        
        if not match:
            return ""
        
        start = match.end()
        next_section = len(text)
        
        for other_name, other_pattern in self.section_patterns.items():
            if other_name != section_name:
                other_match = other_pattern.search(text[start:])
                if other_match:
                    next_section = min(next_section, start + other_match.start())
        
        return text[start:next_section].strip()[:2000]
    
    def _extract_manual_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract interpretable manual features from text."""
        features_list = []
        
        for text in texts:
            text_lower = text.lower() if text else ""
            features = {}
            
            # Text statistics
            features['text_length'] = len(text_lower)
            features['word_count'] = len(text_lower.split())
            features['sentence_count'] = text_lower.count('.') + text_lower.count('!') + text_lower.count('?')
            features['unique_word_ratio'] = len(set(text_lower.split())) / max(1, len(text_lower.split()))
            
            # Keyword counts
            for category, pattern in self._keyword_patterns.items():
                matches = pattern.findall(text_lower)
                features[f'kw_{category}_count'] = len(matches)
                features[f'kw_{category}_present'] = 1 if matches else 0
            
            # Section presence
            for section_name in self.section_patterns.keys():
                section_text = self._extract_section(text, section_name)
                features[f'has_{section_name}'] = 1 if section_text else 0
                features[f'{section_name}_length'] = len(section_text)
            
            # Risk vs Protective balance
            risk_count = features.get('kw_high_risk_count', 0) + features.get('kw_readmission_risk_count', 0)
            protective_count = features.get('kw_protective_count', 0) + features.get('kw_discharge_ready_count', 0)
            features['risk_protective_ratio'] = risk_count / max(1, protective_count)
            features['net_risk_score'] = risk_count - protective_count
            
            # Specific clinical patterns
            features['mentions_family_support'] = 1 if re.search(r'family.*(support|present|bedside)', text_lower) else 0
            features['mentions_follow_up'] = 1 if re.search(r'follow.?up.*(appt|appointment|scheduled)', text_lower) else 0
            features['mentions_home_health'] = 1 if re.search(r'home health|vna|visiting nurse', text_lower) else 0
            features['mentions_snf_rehab'] = 1 if re.search(r'snf|skilled nursing|rehab', text_lower) else 0
            features['mentions_ama'] = 1 if re.search(r'against medical advice|ama|left ama', text_lower) else 0
            features['mentions_noncompliance'] = 1 if re.search(r'non.?compli|non.?adher', text_lower) else 0
            
            # Stability indicators
            features['is_stable'] = 1 if re.search(r'stable|at baseline|doing well', text_lower) else 0
            features['is_unstable'] = 1 if re.search(r'unstable|deteriorat|worsen', text_lower) else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _get_embeddings_cached(self, texts: List[str]) -> np.ndarray:
        """Get ClinicalBERT embeddings with caching."""
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_encode.append(text[:512])  # Truncate for BERT
                indices_to_encode.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self.encoder.encode(
                texts_to_encode,
                show_progress_bar=True,
                batch_size=32
            )
            
            for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                self.cache.set(text, emb)
                embeddings.append((idx, emb))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([e[1] for e in embeddings])
    
    def learn(self, texts: List[str], X_context: pd.DataFrame, y: np.ndarray) -> None:
        """Train the note specialist."""
        # Fit LDA on training texts
        self._fit_lda(texts)
        
        # Get ClinicalBERT embeddings
        print(f"      Encoding {len(texts):,} texts with ClinicalBERT...")
        embeddings = self._get_embeddings_cached(texts)
        
        # Get LDA features
        lda_features = self._get_lda_features(texts)
        
        # Get manual features
        manual_features = self._extract_manual_features(texts)
        self.manual_feature_names = manual_features.columns.tolist()
        
        # Combine all features
        X_combined = np.hstack([
            embeddings,
            lda_features,
            manual_features.fillna(0).values
        ])
        
        X_scaled = self.scaler.fit_transform(X_combined)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"      ✅ NoteSpecialist trained with {X_combined.shape[1]} features "
              f"({embeddings.shape[1]} BERT + {lda_features.shape[1]} LDA + {len(self.manual_feature_names)} manual)")
    
    def give_opinion(self, texts: List[str], X_context: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("NoteSpecialist must be trained before giving opinions!")
        
        # Get embeddings
        embeddings = self._get_embeddings_cached(texts)
        
        # Get LDA features
        lda_features = self._get_lda_features(texts)
        
        # Get manual features
        manual_features = self._extract_manual_features(texts)
        
        # Combine
        X_combined = np.hstack([
            embeddings,
            lda_features,
            manual_features.fillna(0).values
        ])
        
        X_scaled = self.scaler.transform(X_combined)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Has data if text is meaningful
        has_data = np.array([1.0 if len(t.strip()) > 50 else 0.0 for t in texts])
        
        return probs, has_data


# =============================================================================
# PHARMACY SPECIALIST
# =============================================================================
class PharmacySpecialist:
    """
    Medication-based risk assessment specialist.
    Analyzes medication lists for high-risk drugs, polypharmacy, and drug classes.
    """
    
    def __init__(self) -> None:
        """Initialize Pharmacy Specialist."""
        self.name = "Spec_Pharm"
        
        # High-risk medication categories
        self.high_risk_meds = {
            'anticoagulants': ['warfarin', 'coumadin', 'heparin', 'enoxaparin', 'lovenox', 
                              'rivaroxaban', 'xarelto', 'apixaban', 'eliquis', 'dabigatran'],
            'insulin': ['insulin', 'novolog', 'humalog', 'lantus', 'levemir', 'glargine'],
            'opioids': ['morphine', 'oxycodone', 'hydrocodone', 'fentanyl', 'dilaudid', 
                       'hydromorphone', 'methadone', 'tramadol'],
            'cardiac': ['digoxin', 'amiodarone', 'sotalol', 'dofetilide'],
            'immunosuppressants': ['tacrolimus', 'cyclosporine', 'mycophenolate', 'sirolimus',
                                   'prednisone', 'methylprednisolone'],
            'chemotherapy': ['methotrexate', 'cyclophosphamide', 'azathioprine'],
            'sedatives': ['lorazepam', 'diazepam', 'midazolam', 'alprazolam', 'clonazepam'],
            'antipsychotics': ['haloperidol', 'olanzapine', 'quetiapine', 'risperidone'],
            'diuretics': ['furosemide', 'lasix', 'bumetanide', 'torsemide', 'spironolactone'],
        }
        
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            lowercase=True
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
        self.is_fitted = False
    
    def _extract_med_features(self, med_texts: List[str]) -> pd.DataFrame:
        """Extract medication-based features."""
        features_list = []
        
        for text in med_texts:
            text_lower = text.lower() if text else ""
            features = {}
            
            # Medication count
            meds = [m.strip() for m in text_lower.split(',') if m.strip()]
            features['med_count'] = len(meds)
            features['polypharmacy'] = 1 if len(meds) >= 10 else 0
            features['extreme_polypharmacy'] = 1 if len(meds) >= 15 else 0
            
            # High-risk medication categories
            for category, med_list in self.high_risk_meds.items():
                count = sum(1 for med in med_list if med in text_lower)
                features[f'{category}_count'] = count
                features[f'has_{category}'] = 1 if count > 0 else 0
            
            # Total high-risk count
            features['total_high_risk'] = sum(
                features.get(f'{cat}_count', 0) for cat in self.high_risk_meds.keys()
            )
            
            # Multiple high-risk categories
            features['high_risk_categories'] = sum(
                features.get(f'has_{cat}', 0) for cat in self.high_risk_meds.keys()
            )
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def learn(self, med_texts: List[str], X_context: pd.DataFrame, y: np.ndarray) -> None:
        """Train the pharmacy specialist."""
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(med_texts).toarray()
        
        # Manual features
        manual_features = self._extract_med_features(med_texts)
        self.manual_feature_names = manual_features.columns.tolist()
        
        # Combine
        X_combined = np.hstack([tfidf_features, manual_features.fillna(0).values])
        X_scaled = self.scaler.fit_transform(X_combined)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def give_opinion(self, med_texts: List[str], X_context: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("PharmacySpecialist must be trained before giving opinions!")
        
        tfidf_features = self.tfidf.transform(med_texts).toarray()
        manual_features = self._extract_med_features(med_texts)
        
        X_combined = np.hstack([tfidf_features, manual_features.fillna(0).values])
        X_scaled = self.scaler.transform(X_combined)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        has_data = np.array([1.0 if len(t.strip()) > 5 else 0.0 for t in med_texts])
        
        return probs, has_data


# =============================================================================
# HISTORY SPECIALIST
# =============================================================================
class HistorySpecialist:
    """
    Diagnosis history specialist.
    Analyzes prior diagnoses, procedures, and medical history patterns.
    """
    
    def __init__(self) -> None:
        """Initialize History Specialist."""
        self.name = "Spec_Hist"
        
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            lowercase=True
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
        self.is_fitted = False
        
        # Chronic condition patterns
        self.chronic_patterns = {
            'heart_failure': r'heart failure|chf|hfref|hfpef|cardiomyopathy',
            'copd': r'copd|chronic obstructive|emphysema|chronic bronchitis',
            'diabetes': r'diabetes|dm type|diabetic|dm2|dm1|iddm|niddm',
            'ckd': r'chronic kidney|ckd|esrd|dialysis|renal failure',
            'liver_disease': r'cirrhosis|hepatic|liver disease|hepatitis',
            'cancer': r'cancer|carcinoma|malignancy|metastatic|neoplasm',
            'dementia': r'dementia|alzheimer|cognitive impairment',
            'depression': r'depression|depressive|major depressive',
            'substance_abuse': r'substance|alcohol|drug abuse|opioid use|addiction',
        }
        
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.chronic_patterns.items()
        }
    
    def _extract_history_features(self, history_texts: List[str]) -> pd.DataFrame:
        """Extract history-based features."""
        features_list = []
        
        for text in history_texts:
            text_lower = text.lower() if text else ""
            features = {}
            
            # Text statistics
            features['history_length'] = len(text_lower)
            features['history_word_count'] = len(text_lower.split())
            
            # Chronic conditions
            for condition, pattern in self._compiled_patterns.items():
                features[f'has_{condition}'] = 1 if pattern.search(text_lower) else 0
            
            # Total chronic conditions
            features['chronic_condition_count'] = sum(
                features.get(f'has_{cond}', 0) for cond in self.chronic_patterns.keys()
            )
            
            # Prior admissions mentioned
            features['prior_admissions_mentioned'] = 1 if re.search(
                r'prior admission|previous hospitalization|readmit', text_lower
            ) else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def learn(self, history_texts: List[str], X_context: pd.DataFrame, y: np.ndarray) -> None:
        """Train the history specialist."""
        tfidf_features = self.tfidf.fit_transform(history_texts).toarray()
        manual_features = self._extract_history_features(history_texts)
        
        X_combined = np.hstack([tfidf_features, manual_features.fillna(0).values])
        X_scaled = self.scaler.fit_transform(X_combined)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def give_opinion(self, history_texts: List[str], X_context: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("HistorySpecialist must be trained before giving opinions!")
        
        tfidf_features = self.tfidf.transform(history_texts).toarray()
        manual_features = self._extract_history_features(history_texts)
        
        X_combined = np.hstack([tfidf_features, manual_features.fillna(0).values])
        X_scaled = self.scaler.transform(X_combined)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        has_data = np.array([1.0 if len(t.strip()) > 10 else 0.0 for t in history_texts])
        
        return probs, has_data


# =============================================================================
# PSYCHOSOCIAL SPECIALIST
# =============================================================================
class PsychosocialSpecialist:
    """
    Psychosocial risk assessment specialist.
    Combines mental health, care access, and social support sub-specialists.
    """
    
    def __init__(self) -> None:
        """Initialize Psychosocial Specialist."""
        self.name = "Spec_Psychosocial"
        
        # Sub-specialist models
        self.mental_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        
        self.care_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        
        self.social_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        
        # Meta-model combines sub-specialists
        self.meta_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=3,  # Simpler meta-model
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        
        self.scalers = {
            'mental': StandardScaler(),
            'care': StandardScaler(),
            'social': StandardScaler(),
        }
        
        # Keyword patterns
        self.patterns = {
            'mental_health': {
                'risk': re.compile(r'depression|anxiety|psychiatric|bipolar|schizophren|suicid|self.?harm|psychosis|mental health', re.I),
                'protective': re.compile(r'psychiatr.*follow|mental health.*support|counseling|therapy', re.I)
            },
            'care_access': {
                'risk': re.compile(r'non.?compli|missed.*appoint|no.*pcp|no.*follow|lost to|ama|against medical', re.I),
                'protective': re.compile(r'follow.?up.*scheduled|appointment.*made|pcp.*follow|compliant|adherent', re.I)
            },
            'social': {
                'risk': re.compile(r'homeless|no.*support|lives.*alone|isolated|no.*family|shelter|substance|alcohol|drug', re.I),
                'protective': re.compile(r'family.*support|lives.*with|caregiver|spouse|daughter|son|home health|snf|rehab', re.I)
            }
        }
        
        self.is_fitted = False
    
    def _extract_mental_features(self, texts: List[str], med_texts: List[str]) -> pd.DataFrame:
        """Extract mental health related features."""
        features_list = []
        
        for text, meds in zip(texts, med_texts):
            text_lower = text.lower() if text else ""
            meds_lower = meds.lower() if meds else ""
            features = {}
            
            # Mental health keywords
            features['mental_risk_mentions'] = len(self.patterns['mental_health']['risk'].findall(text_lower))
            features['mental_protective_mentions'] = len(self.patterns['mental_health']['protective'].findall(text_lower))
            features['mental_net_risk'] = features['mental_risk_mentions'] - features['mental_protective_mentions']
            
            # Psychiatric medications
            psych_meds = ['sertraline', 'fluoxetine', 'escitalopram', 'citalopram', 'paroxetine',
                         'venlafaxine', 'duloxetine', 'bupropion', 'mirtazapine', 'trazodone',
                         'quetiapine', 'olanzapine', 'risperidone', 'aripiprazole', 'lithium',
                         'lamotrigine', 'valproate', 'lorazepam', 'clonazepam', 'alprazolam']
            
            features['psych_med_count'] = sum(1 for med in psych_meds if med in meds_lower)
            features['has_psych_meds'] = 1 if features['psych_med_count'] > 0 else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _extract_care_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract care access related features."""
        features_list = []
        
        for text in texts:
            text_lower = text.lower() if text else ""
            features = {}
            
            features['care_risk_mentions'] = len(self.patterns['care_access']['risk'].findall(text_lower))
            features['care_protective_mentions'] = len(self.patterns['care_access']['protective'].findall(text_lower))
            features['care_net_risk'] = features['care_risk_mentions'] - features['care_protective_mentions']
            
            # Specific patterns
            features['has_followup'] = 1 if re.search(r'follow.?up.*(appt|schedul)', text_lower) else 0
            features['has_pcp'] = 1 if re.search(r'pcp|primary care', text_lower) else 0
            features['ama_mentioned'] = 1 if re.search(r'against medical advice|ama', text_lower) else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _extract_social_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract social support related features."""
        features_list = []
        
        for text in texts:
            text_lower = text.lower() if text else ""
            features = {}
            
            features['social_risk_mentions'] = len(self.patterns['social']['risk'].findall(text_lower))
            features['social_protective_mentions'] = len(self.patterns['social']['protective'].findall(text_lower))
            features['social_net_risk'] = features['social_risk_mentions'] - features['social_protective_mentions']
            
            # Specific patterns
            features['has_family_support'] = 1 if re.search(r'family.*(support|present|bedside)', text_lower) else 0
            features['lives_alone'] = 1 if re.search(r'lives.*alone|no.*support', text_lower) else 0
            features['homeless'] = 1 if re.search(r'homeless|shelter', text_lower) else 0
            features['substance_issue'] = 1 if re.search(r'substance|alcohol|drug.*use|opioid', text_lower) else 0
            features['has_snf_rehab'] = 1 if re.search(r'snf|skilled nursing|rehab', text_lower) else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def learn(self, texts: List[str], X_context: pd.DataFrame, y: np.ndarray,
              med_list: Optional[List[str]] = None) -> None:
        """Train the psychosocial specialist."""
        if med_list is None:
            med_list = [''] * len(texts)
        
        # Extract features for each sub-specialist
        mental_features = self._extract_mental_features(texts, med_list)
        care_features = self._extract_care_features(texts)
        social_features = self._extract_social_features(texts)
        
        # Scale and train sub-specialists
        X_mental = self.scalers['mental'].fit_transform(mental_features.fillna(0))
        X_care = self.scalers['care'].fit_transform(care_features.fillna(0))
        X_social = self.scalers['social'].fit_transform(social_features.fillna(0))
        
        self.mental_model.fit(X_mental, y)
        self.care_model.fit(X_care, y)
        self.social_model.fit(X_social, y)
        
        # Get sub-specialist predictions
        mental_probs = self.mental_model.predict_proba(X_mental)[:, 1]
        care_probs = self.care_model.predict_proba(X_care)[:, 1]
        social_probs = self.social_model.predict_proba(X_social)[:, 1]
        
        # Train meta-model
        X_meta = np.column_stack([mental_probs, care_probs, social_probs])
        self.meta_model.fit(X_meta, y)
        
        self.is_fitted = True
    
    def give_opinion(self, texts: List[str], X_context: pd.DataFrame,
                     med_list: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Generate probability predictions with sub-specialist breakdown."""
        if not self.is_fitted:
            raise RuntimeError("PsychosocialSpecialist must be trained before giving opinions!")
        
        if med_list is None:
            med_list = [''] * len(texts)
        
        # Extract features
        mental_features = self._extract_mental_features(texts, med_list)
        care_features = self._extract_care_features(texts)
        social_features = self._extract_social_features(texts)
        
        # Scale
        X_mental = self.scalers['mental'].transform(mental_features.fillna(0))
        X_care = self.scalers['care'].transform(care_features.fillna(0))
        X_social = self.scalers['social'].transform(social_features.fillna(0))
        
        # Get sub-specialist predictions
        mental_probs = self.mental_model.predict_proba(X_mental)[:, 1]
        care_probs = self.care_model.predict_proba(X_care)[:, 1]
        social_probs = self.social_model.predict_proba(X_social)[:, 1]
        
        # Get meta-model prediction
        X_meta = np.column_stack([mental_probs, care_probs, social_probs])
        combined_probs = self.meta_model.predict_proba(X_meta)[:, 1]
        
        # Has data
        has_data = np.array([1.0 if len(t.strip()) > 20 else 0.0 for t in texts])
        
        # Sub-specialist breakdown
        sub_opinions = {
            'mental': mental_probs,
            'care': care_probs,
            'social': social_probs
        }
        
        return combined_probs, has_data, sub_opinions