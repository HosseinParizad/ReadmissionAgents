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
import gc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation

# Memory monitoring (optional - install psutil if needed)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def print_memory_usage(stage: str = ""):
    """Print current memory usage."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"      [MEM] {stage} - RAM: {mem.percent:.1f}% ({mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)")
    # Don't print if psutil not available to avoid clutter

# =============================================================================
# FIX WINDOWS CONSOLE ENCODING FOR UNICODE CHARACTERS
# =============================================================================
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
    
    # Enhanced GPU detection
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        # Verify GPU is actually accessible
        try:
            torch.cuda.device_count()
            torch.cuda.get_device_name(0)
            DEVICE = "cuda"
            print(f"      ✅ PyTorch loaded successfully (Device: {DEVICE})")
            print(f"      ✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"      ✅ CUDA version: {torch.version.cuda}")
        except Exception as e:
            print(f"      ⚠️ GPU detected but not accessible: {e}")
            DEVICE = "cpu"
            print(f"      ⚠️ Falling back to CPU")
    else:
        DEVICE = "cpu"
        print(f"      ✅ PyTorch loaded successfully (Device: {DEVICE})")
        print(f"      ⚠️ WARNING: No GPU detected - running on CPU (this will be slower)")
        print(f"      💡 To use GPU, ensure:")
        print(f"         1. NVIDIA GPU with CUDA support is installed")
        print(f"         2. CUDA drivers are installed")
        print(f"         3. PyTorch with CUDA is installed: pip install torch --index-url https://download.pytorch.org/whl/cu118")
except ImportError as e:
    raise ImportError(
        "❌ REQUIRED: torch and sentence-transformers not installed!\n"
        "   Install with: pip install torch sentence-transformers\n"
        f"   Original error: {e}"
    )

# Import centralized configuration
from config import RANDOM_STATE

# Constants - OPTIMIZED REGULARIZATION
DEFAULT_LEARNING_RATE = 0.02      # Slightly slower for better generalization
DEFAULT_MAX_ITER = 200            # Reduced for faster training (was 500, too slow on large datasets)
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
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
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
        # Build feature columns in a dict first to avoid DataFrame fragmentation
        feat: Dict[str, pd.Series] = {}

        # Core lab values
        lab_cols = [c for c in X_labs.columns if any(x in c.lower() for x in 
                    ['sodium', 'potassium', 'creatinine', 'bun', 'glucose', 
                     'hemoglobin', 'wbc', 'platelets', 'albumin', 'bilirubin',
                     'lactate', 'inr', 'troponin', 'bnp'])]
        for col in lab_cols:
            feat[col] = X_labs[col].fillna(0)

        # Organ dysfunction: Renal
        if 'creatinine_mean' in X_labs.columns:
            feat['renal_dysfunction'] = (X_labs['creatinine_mean'].fillna(1.0) > 1.5).astype(int)
            feat['severe_aki'] = (X_labs['creatinine_mean'].fillna(1.0) > 3.0).astype(int)

        if 'bun_mean' in X_labs.columns and 'creatinine_mean' in X_labs.columns:
            creat = X_labs['creatinine_mean'].fillna(1.0).replace(0, 1.0)
            feat['bun_creat_ratio'] = X_labs['bun_mean'].fillna(15) / creat
            feat['prerenal_pattern'] = (feat['bun_creat_ratio'] > 20).astype(int)

        # Organ dysfunction: Hepatic
        if 'bilirubin_mean' in X_labs.columns:
            feat['liver_dysfunction'] = (X_labs['bilirubin_mean'].fillna(1.0) > 2.0).astype(int)

        if 'albumin_mean' in X_labs.columns:
            feat['hypoalbuminemia'] = (X_labs['albumin_mean'].fillna(4.0) < 3.0).astype(int)
            feat['severe_hypoalbuminemia'] = (X_labs['albumin_mean'].fillna(4.0) < 2.5).astype(int)

        # Organ dysfunction: Hematologic
        if 'hemoglobin_mean' in X_labs.columns:
            feat['anemia'] = (X_labs['hemoglobin_mean'].fillna(12) < 10).astype(int)
            feat['severe_anemia'] = (X_labs['hemoglobin_mean'].fillna(12) < 7).astype(int)

        if 'platelets_mean' in X_labs.columns:
            feat['thrombocytopenia'] = (X_labs['platelets_mean'].fillna(200) < 100).astype(int)

        if 'wbc_mean' in X_labs.columns:
            feat['leukocytosis'] = (X_labs['wbc_mean'].fillna(8) > 12).astype(int)
            feat['leukopenia'] = (X_labs['wbc_mean'].fillna(8) < 4).astype(int)

        # Metabolic derangements
        if 'sodium_mean' in X_labs.columns:
            feat['hyponatremia'] = (X_labs['sodium_mean'].fillna(140) < 130).astype(int)
            feat['hypernatremia'] = (X_labs['sodium_mean'].fillna(140) > 150).astype(int)

        if 'potassium_mean' in X_labs.columns:
            feat['hypokalemia'] = (X_labs['potassium_mean'].fillna(4.0) < 3.0).astype(int)
            feat['hyperkalemia'] = (X_labs['potassium_mean'].fillna(4.0) > 5.5).astype(int)

        if 'glucose_mean' in X_labs.columns:
            feat['hyperglycemia'] = (X_labs['glucose_mean'].fillna(100) > 200).astype(int)
            feat['hypoglycemia'] = (X_labs['glucose_mean'].fillna(100) < 70).astype(int)

        # Critical markers
        if 'lactate_mean' in X_labs.columns:
            feat['elevated_lactate'] = (X_labs['lactate_mean'].fillna(1.0) > 2.0).astype(int)
            feat['severe_lactate'] = (X_labs['lactate_mean'].fillna(1.0) > 4.0).astype(int)

        if 'troponin_mean' in X_labs.columns:
            feat['elevated_troponin'] = (X_labs['troponin_mean'].fillna(0) > 0.04).astype(int)

        if 'bnp_mean' in X_labs.columns:
            feat['elevated_bnp'] = (X_labs['bnp_mean'].fillna(100) > 400).astype(int)

        # Coagulopathy
        if 'inr_mean' in X_labs.columns:
            feat['coagulopathy'] = (X_labs['inr_mean'].fillna(1.0) > 1.5).astype(int)

        # Lab trajectory features (first vs last)
        trajectory_labs = ['creatinine', 'hemoglobin', 'wbc', 'platelets', 'sodium', 'potassium']
        for lab in trajectory_labs:
            first_col = f'{lab}_first'
            last_col = f'{lab}_last'
            if first_col in X_labs.columns and last_col in X_labs.columns:
                first_val = X_labs[first_col].fillna(0)
                last_val = X_labs[last_col].fillna(0)
                feat[f'{lab}_delta'] = last_val - first_val
                if lab in ['creatinine', 'wbc']:
                    feat[f'{lab}_improving'] = (feat[f'{lab}_delta'] < 0).astype(int)
                else:
                    feat[f'{lab}_improving'] = (feat[f'{lab}_delta'] > 0).astype(int)

        # Aggregate dysfunction score
        dysfunction_cols = [k for k in feat.keys() if 'dysfunction' in k or 'severe' in k]
        if dysfunction_cols:
            # elementwise sum of Series
            feat['total_organ_dysfunction'] = sum([feat[c] for c in dysfunction_cols])

        # Lab instability (high variance)
        std_cols = [c for c in X_labs.columns if '_std' in c]
        if std_cols:
            feat['lab_instability'] = X_labs[std_cols].fillna(0).mean(axis=1)

        # Context features that help interpretation
        if 'had_icu_stay' in X_context.columns:
            # Ensure total_organ_dysfunction exists in feat (if not, treat as zeros)
            tod = feat.get('total_organ_dysfunction', pd.Series(0, index=X_labs.index))
            feat['icu_with_lab_issues'] = (
                (X_context['had_icu_stay'] == 1) & 
                (tod > 0)
            ).astype(int)

        # Create DataFrame once from the dict to avoid fragmentation
        features = pd.DataFrame(feat, index=X_labs.index)
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
    Enhanced Clinical Notes Specialist with negation detection,
    severity scoring, and context-aware predictions.
    """

    def __init__(self) -> None:
        self.name = "Spec_Notes_Enhanced"

        # Embedding model
        self.encoder = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO",
            device=DEVICE
        )
        self.cache = ClinicalBERTCache(max_cache_size=100000)

        # Predictive model
        self.model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        )

        self.scaler = StandardScaler()

        # Topic model
        self.vectorizer = None
        self.lda = None
        self.lda_fitted = False

        # Section detectors
        self.section_patterns = {
            "chief_complaint": re.compile(r"chief complaint:", re.I),
            "hpi": re.compile(r"history of present illness:", re.I),
            "pmh": re.compile(r"past medical history:", re.I),
            "social": re.compile(r"social history:", re.I),
            "assessment": re.compile(r"assessment and plan:", re.I),
            "discharge_dx": re.compile(r"discharge diagnosis:", re.I),
            "discharge_condition": re.compile(r"discharge condition:", re.I),
            "discharge_instructions": re.compile(r"discharge instructions:", re.I),
            "followup": re.compile(r"follow up:", re.I),
        }

        # Enhanced clinical patterns
        self.risk_patterns = {
            "sepsis": r"sepsis|septic",
            "shock": r"shock|hypotensive crisis",
            "respiratory_failure": r"respiratory failure|intubated|ventilator|mechanical ventilation",
            "cardiac_arrest": r"cardiac arrest|code blue|cpr|resuscitation",
            "icu_admission": r"icu|intensive care|critical care",
            "acute_decompensation": r"acute decompensation|acute exacerbation|acute on chronic",
            "unstable": r"unstable|deteriorating|worsening",
        }
        
        self.protective_patterns = {
            "stable": r"stable|improving|improved|good progress",
            "family_support": r"family.*(?:support|present|involved)|(?:daughter|son|spouse|wife|husband).*(?:support|care|help)",
            "followup_scheduled": r"follow(?:-|\s)up.*(?:scheduled|arranged|appointment)|appointment.*(?:made|scheduled)",
            "home_services": r"home health|visiting nurse|vna|home care",
            "patient_education": r"patient.*(?:understands|educated|verbalized understanding|demonstrates understanding)",
            "compliance": r"compliant|adherent|cooperative|agrees to.*plan",
            "supervised_discharge": r"(?:discharge|going).*(?:snf|rehab|skilled nursing|rehabilitation)",
        }

        # Negation patterns
        self.negation_pattern = re.compile(
            r"\b(no|not|without|denies|denied|negative for|ruled out|absence of|free of)\b",
            re.I
        )
        
        # Severity modifiers
        self.severity_high = re.compile(r"\b(severe|critical|acute|life-threatening|emergent)\b", re.I)
        self.severity_low = re.compile(r"\b(mild|minor|slight|minimal)\b", re.I)
        
        # Trajectory patterns
        self.improving_pattern = re.compile(r"\b(improving|improved|better|resolved|resolving|stable)\b", re.I)
        self.worsening_pattern = re.compile(r"\b(worsening|worse|declining|deteriorating|progressive)\b", re.I)

        self.is_fitted = False

    def _normalize_note(self, text: str) -> str:
        """Normalize note structure."""
        if not text:
            return ""
        
        text = text.strip()
        
        # If already structured, keep as-is
        if any(p.search(text) for p in self.section_patterns.values()):
            return text[:3500]
        
        # Otherwise, wrap raw text
        return (
            "Chief Complaint: UNCLEAR\n"
            f"History of Present Illness: {text}\n"
            "Past Medical History: UNCLEAR\n"
            "Social History: UNCLEAR\n"
            "Assessment and Plan: UNCLEAR\n"
            "Discharge Diagnosis: UNCLEAR\n"
            "Discharge Condition: UNCLEAR\n"
            "Discharge Instructions: UNCLEAR\n"
            "Follow Up: UNCLEAR"
        )[:3500]

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract individual sections from note."""
        sections = {}
        text_lower = text.lower()
        
        for section_name, pattern in self.section_patterns.items():
            match = pattern.search(text)
            if match:
                start = match.end()
                # Find next section or end of text
                next_starts = []
                for other_pattern in self.section_patterns.values():
                    if other_pattern != pattern:
                        next_match = other_pattern.search(text, start)
                        if next_match:
                            next_starts.append(next_match.start())
                
                end = min(next_starts) if next_starts else len(text)
                sections[section_name] = text[start:end].strip()
            else:
                sections[section_name] = ""
        
        return sections

    def _check_negation(self, text: str, pattern: str, window: int = 50) -> Tuple[int, int]:
        """
        Check for both presence and negated presence of a pattern.
        
        Returns:
            (affirmed_count, negated_count)
        """
        affirmed = 0
        negated = 0
        
        # Find all matches
        for match in re.finditer(pattern, text, re.I):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            context = text[start:end]
            
            # Check if negation appears before the match
            if self.negation_pattern.search(context[:match.start()-start]):
                negated += 1
            else:
                affirmed += 1
        
        return affirmed, negated

    def _score_severity(self, text: str) -> float:
        """Score clinical severity based on modifiers."""
        high_count = len(self.severity_high.findall(text))
        low_count = len(self.severity_low.findall(text))
        
        # Normalize to -1 (low) to +1 (high)
        total = high_count + low_count
        if total == 0:
            return 0.0
        
        return (high_count - low_count) / total

    def _get_trajectory(self, text: str) -> float:
        """Assess clinical trajectory (improving vs worsening)."""
        improving = len(self.improving_pattern.findall(text))
        worsening = len(self.worsening_pattern.findall(text))
        
        total = improving + worsening
        if total == 0:
            return 0.0
        
        # -1 (worsening) to +1 (improving)
        return (improving - worsening) / total

    def _embed_notes(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with caching."""
        embeddings = []
        for t in tqdm(texts, desc="      [Note] Embedding", unit="note", leave=False):
            cached = self.cache.get(t)
            if cached is not None:
                embeddings.append(cached)
            else:
                emb = self.encoder.encode(t[:512], convert_to_numpy=True)
                self.cache.set(t, emb)
                embeddings.append(emb)
        return np.vstack(embeddings)

    def _topic_features(self, texts: List[str]) -> np.ndarray:
        """Extract topic model features."""
        if not self.lda_fitted:
            return np.zeros((len(texts), LDA_N_TOPICS))

        cleaned = [re.sub(r"[^\w\s]", " ", t.lower()) for t in 
                   tqdm(texts, desc="      [Note] Cleaning for LDA", unit="note", leave=False)]
        dtm = self.vectorizer.transform(cleaned)
        return self.lda.transform(dtm)

    def _manual_features(self, texts: List[str], X_context: pd.DataFrame) -> pd.DataFrame:
        """
        Extract enhanced manual features with negation detection,
        severity scoring, and context awareness.
        """
        rows = []

        for idx, text in enumerate(tqdm(texts, desc="      [Note] Manual features", unit="note", leave=False)):
            text_lower = text.lower()
            f = {
                "length": len(text_lower),
                "word_count": len(text_lower.split()),
            }

            # Section presence
            for section_name, pattern in self.section_patterns.items():
                f[f"has_{section_name}"] = 1 if pattern.search(text) else 0

            # Risk patterns with negation awareness
            for risk_name, risk_pattern in self.risk_patterns.items():
                affirmed, negated = self._check_negation(text, risk_pattern)
                f[f"{risk_name}_affirmed"] = affirmed
                f[f"{risk_name}_negated"] = negated
                f[f"{risk_name}_net"] = affirmed - negated

            # Protective patterns with negation awareness
            for prot_name, prot_pattern in self.protective_patterns.items():
                affirmed, negated = self._check_negation(text, prot_pattern)
                f[f"{prot_name}_affirmed"] = affirmed
                f[f"{prot_name}_negated"] = negated
                f[f"{prot_name}_net"] = affirmed - negated

            # Severity and trajectory
            f["severity_score"] = self._score_severity(text)
            f["trajectory_score"] = self._get_trajectory(text)

            # Aggregate risk/protective scores
            f["total_risk_score"] = sum(f.get(f"{r}_net", 0) for r in self.risk_patterns.keys())
            f["total_protective_score"] = sum(f.get(f"{p}_net", 0) for p in self.protective_patterns.keys())
            f["net_clinical_risk"] = f["total_risk_score"] - f["total_protective_score"]

            # Context-aware modulation
            if idx < len(X_context):
                ctx = X_context.iloc[idx] if isinstance(X_context, pd.DataFrame) else X_context[idx]
                
                # If ICU patient, weight risk factors more
                if isinstance(ctx, (pd.Series, dict)) and ctx.get('had_icu_stay', 0):
                    f["risk_icu_interaction"] = f["total_risk_score"] * 1.5
                else:
                    f["risk_icu_interaction"] = f["total_risk_score"]
                
                # If elderly, weight certain risks more
                if isinstance(ctx, (pd.Series, dict)):
                    age = ctx.get('anchor_age', 0) or ctx.get('age', 0)
                    if age > 75:
                        f["risk_age_interaction"] = f["total_risk_score"] * 1.3
                    else:
                        f["risk_age_interaction"] = f["total_risk_score"]

            # Discharge quality indicators
            f["has_discharge_plan"] = 1 if re.search(
                r"discharge.*plan|plan.*discharge|discharge.*instructions", text_lower
            ) else 0
            
            f["has_medication_list"] = 1 if re.search(
                r"medications?.*(?:list|prescribed|discharge)|discharge.*medications?", text_lower
            ) else 0

            rows.append(f)

        return pd.DataFrame(rows).fillna(0)

    def learn(self, texts: List[str], X_context: pd.DataFrame, y: np.ndarray) -> None:
        """Train the note specialist with all enhancements."""
        # Normalize
        texts = [self._normalize_note(t) for t in 
                 tqdm(texts, desc="      [Note] Normalizing", unit="note", leave=False)]

        # Topic model with FIXED min_df
        cleaned = [re.sub(r"[^\w\s]", " ", t.lower()) for t in texts]
        self.vectorizer = CountVectorizer(
            max_features=1000,
            min_df=5,  # CHANGED from 50 - captures rare medical terms
            stop_words="english"
        )
        dtm = self.vectorizer.fit_transform(cleaned)

        self.lda = LatentDirichletAllocation(
            n_components=LDA_N_TOPICS,
            max_iter=LDA_MAX_ITER,
            learning_method="online",
            random_state=RANDOM_STATE
        )
        self.lda.fit(dtm)
        self.lda_fitted = True

        # Extract all feature types
        emb = self._embed_notes(texts)
        lda = self._topic_features(texts)
        man = self._manual_features(texts, X_context).values

        # IMPORTANT: Include basic context features
        context_features = X_context[['had_icu_stay', 'anchor_age', 'los_days']].fillna(0).values \
            if all(col in X_context.columns for col in ['had_icu_stay', 'anchor_age', 'los_days']) \
            else np.zeros((len(texts), 3))

        # Combine all features
        X = np.hstack([emb, lda, man, context_features])
        Xs = self.scaler.fit_transform(X)

        self.model.fit(Xs, y)
        self.cache.save()
        self.is_fitted = True

    def give_opinion(self, texts: List[str], X_context: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with all enhancements."""
        if not self.is_fitted:
            raise RuntimeError("NoteSpecialist must be trained first")

        # Normalize
        texts = [self._normalize_note(t) for t in 
                 tqdm(texts, desc="      [Note] Normalizing (Inference)", unit="note", leave=False)]

        # Extract features
        emb = self._embed_notes(texts)
        lda = self._topic_features(texts)
        man = self._manual_features(texts, X_context).values

        # Context features
        context_features = X_context[['had_icu_stay', 'anchor_age', 'los_days']].fillna(0).values \
            if all(col in X_context.columns for col in ['had_icu_stay', 'anchor_age', 'los_days']) \
            else np.zeros((len(texts), 3))

        # Combine
        X = np.hstack([emb, lda, man, context_features])
        Xs = self.scaler.transform(X)

        probs = self.model.predict_proba(Xs)[:, 1]
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
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
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
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
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
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
        )
        
        self.care_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
        )
        
        self.social_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
        )
        
        # Meta-model combines sub-specialists
        self.meta_model = HistGradientBoostingClassifier(
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            max_depth=3,  # Simpler meta-model
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-7
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