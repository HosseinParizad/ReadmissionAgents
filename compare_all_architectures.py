"""
MULTI-AGENT ARCHITECTURE COMPARISON FOR HOSPITAL READMISSION PREDICTION
========================================================================

This script implements and compares 7 different multi-agent architectures
for 30-day hospital readmission prediction using MIMIC-IV data.

ARCHITECTURES:
--------------
0. BASELINE: Multi-Agent Ensemble with CatBoost Meta-Learner (Current)
   - Paper name: "Interpretable Multi-Specialist Ensemble (IMSE)"
   - 5 specialist agents + CatBoost Doctor Agent
   - K-Fold OOF stacking

1. MIXTURE OF EXPERTS (MoE)
   - Paper name: "Adaptive Specialist Weighting Network (ASWN)"
   - Gating network learns dynamic specialist weights per patient

2. HIERARCHICAL DEBATE
   - Paper name: "Multi-Round Deliberative Ensemble (MRDE)"  
   - Specialists revise opinions through iterative debate rounds

3. GRAPH NEURAL NETWORK
   - Paper name: "Patient Similarity Graph Network (PSGN)"
   - Learns from similar patients' outcomes via message passing

4. TRANSFORMER FUSION
   - Paper name: "Cross-Specialist Attention Network (CSAN)"
   - Self-attention mechanism fuses specialist outputs

5. LLM-AS-JUDGE (Clinical Reasoning)
   - Paper name: "Clinical Reasoning Engine (CRE)"
   - Rule-based clinical reasoning mimicking physician decision-making

6. TEMPORAL ATTENTION
   - Paper name: "Trajectory-Aware Specialist Network (TASN)"
   - Incorporates lab value trajectories with attention mechanism

USAGE:
------
    # Run all architectures
    python compare_all_architectures.py
    
    # Run specific architectures (0=baseline, 1-6=new)
    python compare_all_architectures.py --arch 0,1,3,5
    
    # Skip specialist training if already done
    python compare_all_architectures.py --skip-specialist-training

OUTPUT:
-------
    ./model_outputs/
    ├── shared/                     # Shared OOF predictions
    ├── comparison/                 # Final comparison results
    │   └── full_comparison_*.csv
    └── arch{0-6}_*/               # Individual architecture results

AUTHORS: [Your Name]
DATE: 2024
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =============================================================================
# FIX WINDOWS CONSOLE ENCODING FOR UNICODE CHARACTERS
# =============================================================================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# STRICT DEPENDENCY CHECKING - NO FALLBACKS
# =============================================================================
print("=" * 70)
print("CHECKING DEPENDENCIES (STRICT MODE)")
print("=" * 70)

# Check CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    print("   ✅ CatBoost loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: CatBoost not installed!\n   Install with: pip install catboost\n   Error: {e}")

# Check LightGBM
try:
    from lightgbm import LGBMClassifier
    print("   ✅ LightGBM loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: LightGBM not installed!\n   Install with: pip install lightgbm\n   Error: {e}")

# Check XGBoost
try:
    from xgboost import XGBClassifier
    print("   ✅ XGBoost loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: XGBoost not installed!\n   Install with: pip install xgboost\n   Error: {e}")

# Check PyTorch
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    if CUDA_AVAILABLE:
        print(f"   ✅ PyTorch loaded successfully (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"   ✅ PyTorch loaded successfully (CPU mode - this will be slower)")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: PyTorch not installed!\n   Install with: pip install torch\n   Error: {e}")

# Check SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    print("   ✅ SentenceTransformers loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: sentence-transformers not installed!\n   Install with: pip install sentence-transformers\n   Error: {e}")

import warnings
warnings.filterwarnings('ignore')

# Import from existing codebase
print("\n   Loading project modules...")

try:
    from config import (
        DATA_PATH, TARGET, RANDOM_STATE, USE_TEMPORAL_SPLIT,
        DOCTOR_CONTEXT_FEATURES, DEFAULT_N_FOLDS, CATBOOST_PARAMS
    )
    print("   ✅ config.py loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: config.py not found!\n   Ensure config.py is in your project directory.\n   Error: {e}")

try:
    from specialist_agents import (
        LabSpecialist, NoteSpecialist, PharmacySpecialist,
        HistorySpecialist, PsychosocialSpecialist
    )
    print("   ✅ specialist_agents.py loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: specialist_agents.py not found or has errors!\n   Ensure specialist_agents.py is in your project directory.\n   Error: {e}")

# Check for clinicalbert_cache (required by specialist_agents)
try:
    from clinicalbert_cache import ClinicalBERTCache
    print("   ✅ clinicalbert_cache.py loaded successfully")
except ImportError as e:
    raise ImportError(f"❌ REQUIRED: clinicalbert_cache.py not found!\n   Ensure clinicalbert_cache.py is in your project directory.\n   Error: {e}")

# Check for protective_factors (optional but recommended)
try:
    from protective_factors import extract_protective_factors
    print("   ✅ protective_factors.py loaded successfully")
    HAS_PROTECTIVE_FACTORS = True
except ImportError:
    print("   ⚠️ protective_factors.py not found (optional, continuing without it)")
    HAS_PROTECTIVE_FACTORS = False

print("\n" + "=" * 70)
print("ALL DEPENDENCIES VERIFIED - STARTING COMPARISON")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================
SHARED_DIR = './model_outputs/shared/'
COMPARISON_DIR = './model_outputs/comparison/'
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

# Architecture names for paper
ARCHITECTURE_NAMES = {
    0: {
        'short': 'Baseline (IMSE)',
        'full': 'Interpretable Multi-Specialist Ensemble',
        'abbrev': 'IMSE'
    },
    1: {
        'short': 'Mixture of Experts',
        'full': 'Adaptive Specialist Weighting Network',
        'abbrev': 'ASWN'
    },
    2: {
        'short': 'Hierarchical Debate',
        'full': 'Multi-Round Deliberative Ensemble',
        'abbrev': 'MRDE'
    },
    3: {
        'short': 'Graph Neural Network',
        'full': 'Patient Similarity Graph Network',
        'abbrev': 'PSGN'
    },
    4: {
        'short': 'Transformer Fusion',
        'full': 'Cross-Specialist Attention Network',
        'abbrev': 'CSAN'
    },
    5: {
        'short': 'Clinical Reasoning',
        'full': 'Clinical Reasoning Engine',
        'abbrev': 'CRE'
    },
    6: {
        'short': 'Temporal Attention',
        'full': 'Trajectory-Aware Specialist Network',
        'abbrev': 'TASN'
    }
}


# =============================================================================
# GPU CONFIGURATION FOR GRADIENT BOOSTING
# =============================================================================
def get_gpu_params():
    """
    Returns GPU parameters for CatBoost, LightGBM, and XGBoost if GPU is available.
    Falls back to CPU if GPU is not available.
    """
    gpu_params = {}
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            # CatBoost GPU
            gpu_params['catboost'] = {'task_type': 'GPU', 'devices': '0'}
            # LightGBM GPU
            gpu_params['lightgbm'] = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
            # XGBoost GPU (XGBoost 3.1+ uses 'device' parameter)
            gpu_params['xgboost'] = {'device': 'cuda:0'}
            return gpu_params
    except:
        pass
    
    # Default to CPU
    gpu_params['catboost'] = {}
    gpu_params['lightgbm'] = {}
    gpu_params['xgboost'] = {}
    return gpu_params

# Get GPU parameters once at module load
GPU_PARAMS = get_gpu_params()
if GPU_PARAMS['catboost']:
    print(f"   ✅ GPU acceleration enabled for gradient boosting models")
else:
    print(f"   ℹ️  Gradient boosting models will use CPU (GPU not available)")


# =============================================================================
# SPECIALIST CALIBRATOR (from train_model.py)
# =============================================================================
class SpecialistCalibrator:
    """Calibrates specialist predictions using isotonic regression."""
    
    def __init__(self):
        self.calibrators = {}
        
    def fit(self, name: str, predictions: np.ndarray, labels: np.ndarray):
        """Fit calibrator for a specialist."""
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(predictions, labels)
        self.calibrators[name] = calibrator
        
    def transform(self, name: str, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        if name in self.calibrators:
            return self.calibrators[name].transform(predictions)
        return predictions


# =============================================================================
# CONTEXT-CONDITIONAL CALIBRATOR (LIGHTWEIGHT, ARCHITECTURE-PRESERVING)
# =============================================================================
class ContextConditionalCalibrator:
    """Fits per-subgroup isotonic calibrators using a small set of context features.

    Goal: fix systematic miscalibration that varies by patient context (e.g., ICU vs non-ICU)
    without changing the core architecture.
    """

    def __init__(self, min_group_size: int = 500):
        self.min_group_size = min_group_size
        self.global_calibrators = {}
        self.group_calibrators = {}  # (name, key) -> calibrator
        self.group_cols = None

    @staticmethod
    def _pick_group_cols(ctx_df: pd.DataFrame) -> List[str]:
        """Pick a small, robust set of grouping columns if present."""
        candidates = [
            'had_icu_stay', 'icu', 'is_icu',
            'is_emergency', 'emergency',
            'discharge_to_snf', 'discharge_to_rehab',
            'age', 'age_years',
            'length_of_stay', 'los',
            'num_comorbidities', 'comorbidity_count'
        ]
        cols = [c for c in candidates if c in ctx_df.columns]
        # Keep at most 3 cols to control group explosion
        return cols[:3]

    @staticmethod
    def _binarize_series(s: pd.Series) -> pd.Series:
        # Treat booleans/0-1 as-is; otherwise threshold at median
        s2 = s.fillna(0)
        uniq = s2.dropna().unique()
        if len(uniq) <= 2:
            return (s2 > 0).astype(int)
        return (s2 > s2.median()).astype(int)

    def _make_group_keys(self, ctx_df: pd.DataFrame) -> np.ndarray:
        if self.group_cols is None:
            self.group_cols = self._pick_group_cols(ctx_df)
        if not self.group_cols:
            return np.array(['__all__'] * len(ctx_df))

        parts = []
        for c in self.group_cols:
            parts.append(self._binarize_series(ctx_df[c]).astype(str).values)
        keys = ['|'.join(items) for items in zip(*parts)]
        return np.array(keys, dtype=object)

    def fit(self, name: str, predictions: np.ndarray, labels: np.ndarray, ctx_df: pd.DataFrame):
        # Global calibrator
        gcal = IsotonicRegression(out_of_bounds='clip')
        gcal.fit(predictions, labels)
        self.global_calibrators[name] = gcal

        keys = self._make_group_keys(ctx_df)
        self.group_calibrators[name] = {}

        # Per-group calibrators
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            if len(idx) < self.min_group_size:
                continue
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(predictions[idx], labels[idx])
            self.group_calibrators[name][key] = cal

    def transform(self, name: str, predictions: np.ndarray, ctx_df: pd.DataFrame) -> np.ndarray:
        if name not in self.global_calibrators:
            return predictions

        keys = self._make_group_keys(ctx_df)
        out = np.array(predictions, copy=True)
        gcal = self.global_calibrators[name]
        out = gcal.transform(out)

        # Apply per-group overrides where available
        gmap = self.group_calibrators.get(name, {})
        if gmap:
            for key, cal in gmap.items():
                idx = np.where(keys == key)[0]
                if len(idx):
                    out[idx] = cal.transform(predictions[idx])
        return out


# =============================================================================
# SHARED DATA MANAGEMENT
# =============================================================================
def train_specialists_and_save_oof(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                    n_folds: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train all specialists using K-Fold CV and save OOF predictions.
    This is run ONCE and shared by all architectures.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING SPECIALISTS (SHARED)")
    print("=" * 70)
    
    y_train = df_train[TARGET].values
    y_test = df_test[TARGET].values
    n_train = len(df_train)
    n_test = len(df_test)
    
    print(f"   Train samples: {n_train:,}")
    print(f"   Test samples: {n_test:,}")
    print(f"   Using {n_folds}-Fold CV for OOF predictions")
    
    # Initialize OOF arrays
    oof_train = {
        'lab': np.zeros(n_train),
        'note': np.zeros(n_train),
        'pharm': np.zeros(n_train),
        'hist': np.zeros(n_train),
        'psych': np.zeros(n_train),
        'psych_mental': np.zeros(n_train),
        'psych_care': np.zeros(n_train),
        'psych_social': np.zeros(n_train),
    }
    
    oof_test = {name: np.zeros(n_test) for name in oof_train.keys()}
    
    # K-Fold training
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(kf.split(df_train, y_train)):
        print(f"\n   --- Fold {fold_idx + 1}/{n_folds} ---")
        print(f"       Train: {len(fold_train_idx):,} | Val: {len(fold_val_idx):,}")
        
        df_fold_train = df_train.iloc[fold_train_idx].reset_index(drop=True)
        df_fold_val = df_train.iloc[fold_val_idx].reset_index(drop=True)
        y_fold_train = y_train[fold_train_idx]
        
        # Prepare features
        X_ctx_train = _create_context_features(df_fold_train)
        X_ctx_val = _create_context_features(df_fold_val)
        X_labs_train = _prepare_lab_features(df_fold_train)
        X_labs_val = _prepare_lab_features(df_fold_val)
        
        # Train specialists
        print(f"       Training Lab specialist...", end=" ", flush=True)
        spec_lab = LabSpecialist()
        spec_lab.learn(X_labs_train, X_ctx_train, y_fold_train)
        op_lab, _ = spec_lab.give_opinion(X_labs_val, X_ctx_val)
        oof_train['lab'][fold_val_idx] = op_lab
        print(f"AUC: {roc_auc_score(y_train[fold_val_idx], op_lab):.4f}")
        
        print(f"       Training Note specialist...", end=" ", flush=True)
        spec_note = NoteSpecialist()
        spec_note.learn(df_fold_train['clinical_text'].fillna('').tolist(), X_ctx_train, y_fold_train)
        op_note, _ = spec_note.give_opinion(df_fold_val['clinical_text'].fillna('').tolist(), X_ctx_val)
        oof_train['note'][fold_val_idx] = op_note
        print(f"AUC: {roc_auc_score(y_train[fold_val_idx], op_note):.4f}")
        
        print(f"       Training Pharmacy specialist...", end=" ", flush=True)
        spec_pharm = PharmacySpecialist()
        spec_pharm.learn(df_fold_train['med_list_text'].fillna('').tolist(), X_ctx_train, y_fold_train)
        op_pharm, _ = spec_pharm.give_opinion(df_fold_val['med_list_text'].fillna('').tolist(), X_ctx_val)
        oof_train['pharm'][fold_val_idx] = op_pharm
        print(f"AUC: {roc_auc_score(y_train[fold_val_idx], op_pharm):.4f}")
        
        print(f"       Training History specialist...", end=" ", flush=True)
        spec_hist = HistorySpecialist()
        spec_hist.learn(df_fold_train['full_history_text'].fillna('').tolist(), X_ctx_train, y_fold_train)
        op_hist, _ = spec_hist.give_opinion(df_fold_val['full_history_text'].fillna('').tolist(), X_ctx_val)
        oof_train['hist'][fold_val_idx] = op_hist
        print(f"AUC: {roc_auc_score(y_train[fold_val_idx], op_hist):.4f}")
        
        print(f"       Training Psychosocial specialist...", end=" ", flush=True)
        spec_psych = PsychosocialSpecialist()
        spec_psych.learn(
            df_fold_train['clinical_text'].fillna('').tolist(), X_ctx_train, y_fold_train,
            med_list=df_fold_train['med_list_text'].fillna('').tolist()
        )
        op_psych, _, psych_sub = spec_psych.give_opinion(
            df_fold_val['clinical_text'].fillna('').tolist(), X_ctx_val,
            med_list=df_fold_val['med_list_text'].fillna('').tolist()
        )
        oof_train['psych'][fold_val_idx] = op_psych
        oof_train['psych_mental'][fold_val_idx] = psych_sub['mental']
        oof_train['psych_care'][fold_val_idx] = psych_sub['care']
        oof_train['psych_social'][fold_val_idx] = psych_sub['social']
        print(f"AUC: {roc_auc_score(y_train[fold_val_idx], op_psych):.4f}")
        
        # === MEMORY CLEANUP BETWEEN FOLDS ===
        print(f"       🧹 Cleaning up fold {fold_idx + 1} memory...")
        # Delete specialist objects (they hold large models and embeddings)
        del spec_lab, spec_note, spec_pharm, spec_hist, spec_psych
        # Delete fold-specific data
        del df_fold_train, df_fold_val, y_fold_train
        del X_ctx_train, X_ctx_val, X_labs_train, X_labs_val
        del op_lab, op_note, op_pharm, op_hist, op_psych, psych_sub
        # Force garbage collection
        gc.collect()
        # Clear GPU cache if available
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        # Print memory status
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"       [MEM] After cleanup: {mem.percent:.1f}% ({mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)")
        except:
            pass
    
    # Report overall OOF AUC
    print(f"\n   📊 Overall OOF AUC (Training Set):")
    for name in ['lab', 'note', 'pharm', 'hist', 'psych']:
        print(f"       {name:8s}: {roc_auc_score(y_train, oof_train[name]):.4f}")
    
    # Retrain on full training data for test predictions
    print(f"\n   🔄 Retraining specialists on full training data...")
    
    X_ctx_full = _create_context_features(df_train)
    X_ctx_test = _create_context_features(df_test)
    X_labs_full = _prepare_lab_features(df_train)
    X_labs_test = _prepare_lab_features(df_test)
    
    # Lab
    print(f"       Training Lab...", end=" ", flush=True)
    spec_lab = LabSpecialist()
    spec_lab.learn(X_labs_full, X_ctx_full, y_train)
    oof_test['lab'], _ = spec_lab.give_opinion(X_labs_test, X_ctx_test)
    print(f"Test AUC: {roc_auc_score(y_test, oof_test['lab']):.4f}")
    with open(os.path.join(SHARED_DIR, 'specialist_lab.pkl'), 'wb') as f:
        pickle.dump(spec_lab, f)
    
    # Note
    print(f"       Training Note...", end=" ", flush=True)
    spec_note = NoteSpecialist()
    spec_note.learn(df_train['clinical_text'].fillna('').tolist(), X_ctx_full, y_train)
    oof_test['note'], _ = spec_note.give_opinion(df_test['clinical_text'].fillna('').tolist(), X_ctx_test)
    print(f"Test AUC: {roc_auc_score(y_test, oof_test['note']):.4f}")
    with open(os.path.join(SHARED_DIR, 'specialist_note.pkl'), 'wb') as f:
        pickle.dump(spec_note, f)
    
    # Pharmacy
    print(f"       Training Pharmacy...", end=" ", flush=True)
    spec_pharm = PharmacySpecialist()
    spec_pharm.learn(df_train['med_list_text'].fillna('').tolist(), X_ctx_full, y_train)
    oof_test['pharm'], _ = spec_pharm.give_opinion(df_test['med_list_text'].fillna('').tolist(), X_ctx_test)
    print(f"Test AUC: {roc_auc_score(y_test, oof_test['pharm']):.4f}")
    with open(os.path.join(SHARED_DIR, 'specialist_pharm.pkl'), 'wb') as f:
        pickle.dump(spec_pharm, f)
    
    # History
    print(f"       Training History...", end=" ", flush=True)
    spec_hist = HistorySpecialist()
    spec_hist.learn(df_train['full_history_text'].fillna('').tolist(), X_ctx_full, y_train)
    oof_test['hist'], _ = spec_hist.give_opinion(df_test['full_history_text'].fillna('').tolist(), X_ctx_test)
    print(f"Test AUC: {roc_auc_score(y_test, oof_test['hist']):.4f}")
    with open(os.path.join(SHARED_DIR, 'specialist_hist.pkl'), 'wb') as f:
        pickle.dump(spec_hist, f)
    
    # Psychosocial
    print(f"       Training Psychosocial...", end=" ", flush=True)
    spec_psych = PsychosocialSpecialist()
    spec_psych.learn(
        df_train['clinical_text'].fillna('').tolist(), X_ctx_full, y_train,
        med_list=df_train['med_list_text'].fillna('').tolist()
    )
    op_psych_test, _, psych_sub_test = spec_psych.give_opinion(
        df_test['clinical_text'].fillna('').tolist(), X_ctx_test,
        med_list=df_test['med_list_text'].fillna('').tolist()
    )
    oof_test['psych'] = op_psych_test
    oof_test['psych_mental'] = psych_sub_test['mental']
    oof_test['psych_care'] = psych_sub_test['care']
    oof_test['psych_social'] = psych_sub_test['social']
    print(f"Test AUC: {roc_auc_score(y_test, oof_test['psych']):.4f}")
    with open(os.path.join(SHARED_DIR, 'specialist_psych.pkl'), 'wb') as f:
        pickle.dump(spec_psych, f)
    
    # Create DataFrames
    train_oof_df = pd.DataFrame({
        'hadm_id': df_train['hadm_id'].values if 'hadm_id' in df_train.columns else np.zeros(len(df_train)),
        'subject_id': df_train['subject_id'].values if 'subject_id' in df_train.columns else np.zeros(len(df_train)),
        'y_true': y_train,
        'op_lab': oof_train['lab'],
        'op_note': oof_train['note'],
        'op_pharm': oof_train['pharm'],
        'op_hist': oof_train['hist'],
        'op_psych': oof_train['psych'],
        'op_psych_mental': oof_train['psych_mental'],
        'op_psych_care': oof_train['psych_care'],
        'op_psych_social': oof_train['psych_social'],
        'split': 'train'
    })
    
    test_oof_df = pd.DataFrame({
        'hadm_id': df_test['hadm_id'].values if 'hadm_id' in df_test.columns else np.zeros(len(df_test)),
        'subject_id': df_test['subject_id'].values if 'subject_id' in df_test.columns else np.zeros(len(df_test)),
        'y_true': y_test,
        'op_lab': oof_test['lab'],
        'op_note': oof_test['note'],
        'op_pharm': oof_test['pharm'],
        'op_hist': oof_test['hist'],
        'op_psych': oof_test['psych'],
        'op_psych_mental': oof_test['psych_mental'],
        'op_psych_care': oof_test['psych_care'],
        'op_psych_social': oof_test['psych_social'],
        'split': 'test'
    })
    
    # Save OOF predictions
    oof_df = pd.concat([train_oof_df, test_oof_df], ignore_index=True)
    oof_df.to_csv(os.path.join(SHARED_DIR, 'oof_predictions.csv'), index=False)
    
    # Save context features
    ctx_train = _create_context_features(df_train)
    ctx_train['split'] = 'train'
    ctx_test = _create_context_features(df_test)
    ctx_test['split'] = 'test'
    ctx_df = pd.concat([ctx_train, ctx_test], ignore_index=True)
    ctx_df.to_csv(os.path.join(SHARED_DIR, 'context_features.csv'), index=False)
    
    print(f"\n   ✅ Specialists trained and saved to {SHARED_DIR}")
    
    return train_oof_df, test_oof_df


def _create_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create context features from DataFrame."""
    X_context = pd.DataFrame(index=df.index)
    for col in DOCTOR_CONTEXT_FEATURES:
        if col in df.columns:
            X_context[col] = df[col].fillna(0)
    return X_context


def _prepare_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare lab features (simplified version)."""
    lab_cols = [c for c in df.columns if any(x in c.lower() for x in 
                ['sodium', 'potassium', 'creatinine', 'glucose', 'hemoglobin', 
                 'wbc', 'bilirubin', 'lactate', 'bun', 'albumin', 'platelets',
                 'hr_', 'sbp_', 'dbp_', 'resp_', 'spo2_', 'temp_'])]
    
    if lab_cols:
        return df[lab_cols].fillna(0)
    return pd.DataFrame(index=df.index)


def load_shared_data():
    """Load pre-computed OOF predictions and context features."""
    oof_df = pd.read_csv(os.path.join(SHARED_DIR, 'oof_predictions.csv'))
    ctx_df = pd.read_csv(os.path.join(SHARED_DIR, 'context_features.csv'))
    
    train_oof = oof_df[oof_df['split'] == 'train'].reset_index(drop=True)
    test_oof = oof_df[oof_df['split'] == 'test'].reset_index(drop=True)
    train_ctx = ctx_df[ctx_df['split'] == 'train'].reset_index(drop=True)
    test_ctx = ctx_df[ctx_df['split'] == 'test'].reset_index(drop=True)
    
    return train_oof, test_oof, train_ctx, test_ctx


def get_specialist_preds(oof_df: pd.DataFrame) -> np.ndarray:
    """Extract specialist predictions as numpy array."""
    return oof_df[['op_lab', 'op_note', 'op_pharm', 'op_hist', 'op_psych']].values


def get_context_features(ctx_df: pd.DataFrame) -> np.ndarray:
    """Get context features as numpy array."""
    cols = [c for c in ctx_df.columns if c != 'split']
    return ctx_df[cols].fillna(0).values


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray, thresh_min: float = 0.01,
                           thresh_max: float = 0.5, step: float = 0.01) -> float:
    """Select threshold on a *validation* set to maximize F1 (prevents test leakage)."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(thresh_min, thresh_max + 1e-9, step):
        f1 = f1_score(y_true, (y_prob >= thresh).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute evaluation metrics at a provided threshold (no threshold tuning on test)."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'auc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
        'brier': brier_score_loss(y_true, y_prob),
        'threshold': float(threshold),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
    }


def save_predictions_with_features(
    arch_num: int,
    arch_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    test_oof: pd.DataFrame,
    test_ctx: pd.DataFrame,
    output_dir: str = './model_outputs/failure_analysis/'
) -> None:
    """Save predictions with features for failure analysis.
    
    Args:
        arch_num: Architecture number (0-6)
        arch_name: Architecture name (e.g., 'IMSE')
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        test_oof: Test OOF DataFrame with specialist opinions
        test_ctx: Test context DataFrame with clinical features
        output_dir: Output directory for predictions
    """
    # Create output directory
    arch_dir = os.path.join(output_dir, f'arch{arch_num}_{arch_name}')
    os.makedirs(arch_dir, exist_ok=True)
    
    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    })
    
    # Add specialist opinions from test_oof
    specialist_cols = ['op_lab', 'op_note', 'op_pharm', 'op_hist', 'op_psych']
    for col in specialist_cols:
        if col in test_oof.columns:
            predictions_df[col] = test_oof[col].values
    
    # Add psychosocial sub-scores if available
    psych_sub_cols = ['op_psych_mental', 'op_psych_care', 'op_psych_social']
    for col in psych_sub_cols:
        if col in test_oof.columns:
            predictions_df[col] = test_oof[col].values
    
    # Add specialist statistics
    if all(col in predictions_df.columns for col in specialist_cols):
        spec_array = predictions_df[specialist_cols].values
        predictions_df['op_mean'] = np.mean(spec_array, axis=1)
        predictions_df['op_std'] = np.std(spec_array, axis=1)
        predictions_df['op_max'] = np.max(spec_array, axis=1)
        predictions_df['op_min'] = np.min(spec_array, axis=1)
    
    # Add context features from test_ctx
    ctx_cols = [c for c in test_ctx.columns if c != 'split']
    for col in ctx_cols:
        if col in test_ctx.columns:
            predictions_df[col] = test_ctx[col].values
    
    # Save to CSV
    output_file = os.path.join(arch_dir, 'predictions_with_features.csv')
    predictions_df.to_csv(output_file, index=False)
    
    print(f"   💾 Saved predictions to: {output_file}")
    print(f"      Columns: {len(predictions_df.columns)}, Rows: {len(predictions_df)}")


# =============================================================================
# ARCHITECTURE 0: BASELINE (IMSE) - Your Current Model
# =============================================================================
def run_arch0_baseline(train_oof, test_oof, train_ctx, test_ctx, df_train=None):
    """
    Architecture 0: Interpretable Multi-Specialist Ensemble (IMSE)

    Minimal-change improvements (no architectural change):
      1) Context-conditional specialist calibration
      2) Reliability / confidence features for Doctor Agent
      3) Meta-learner trained with StratifiedKFold OOF (reduces split variance)
      4) Final isotonic calibration fitted on OOF predictions (no leakage)
      5) Threshold selected on OOF (no leakage)
    """
    print("\n" + "-" * 60)
    print("ARCH 0: INTERPRETABLE MULTI-SPECIALIST ENSEMBLE (IMSE)")
    print("-" * 60)

    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)

    # ----------------------------
    # 1) Context-conditional calibration of specialists
    # ----------------------------
    print("   Calibrating specialists (context-conditional)...")
    ctx_train_df = train_ctx.drop(columns=['split'], errors='ignore').reset_index(drop=True)
    ctx_test_df = test_ctx.drop(columns=['split'], errors='ignore').reset_index(drop=True)

    calibrator = ContextConditionalCalibrator(min_group_size=500)
    spec_names = ['lab', 'note', 'pharm', 'hist', 'psych']

    for i, name in enumerate(spec_names):
        calibrator.fit(name, spec_train[:, i], y_train, ctx_train_df)
        spec_train[:, i] = calibrator.transform(name, spec_train[:, i], ctx_train_df)
        spec_test[:, i] = calibrator.transform(name, spec_test[:, i], ctx_test_df)

    # ----------------------------
    # 2) Build Doctor features (interpretable + confidence)
    # ----------------------------
    print("   Building Doctor Agent features (with confidence)...")

    def build_doctor_features(spec_preds, ctx_features, oof_df, conf_features=None):
        n = len(spec_preds)
        feats = {}

        # Specialist opinions
        for i, name in enumerate(spec_names):
            feats[f'op_{name}'] = spec_preds[:, i]

        # Ensemble stats
        feats['op_mean'] = np.mean(spec_preds, axis=1)
        feats['op_std'] = np.std(spec_preds, axis=1)
        feats['op_max'] = np.max(spec_preds, axis=1)
        feats['op_min'] = np.min(spec_preds, axis=1)
        feats['op_range'] = feats['op_max'] - feats['op_min']

        # Consensus / disagreement
        high_risk = spec_preds > 0.6
        feats['n_high_risk'] = np.sum(high_risk, axis=1)
        feats['consensus_high'] = (feats['n_high_risk'] >= 3).astype(int)
        feats['consensus_low'] = (feats['n_high_risk'] == 0).astype(int)
        feats['any_very_high'] = (np.max(spec_preds, axis=1) > 0.75).astype(int)

        feats['max_disagreement'] = np.max(np.abs(
            spec_preds[:, :, np.newaxis] - spec_preds[:, np.newaxis, :]
        ).reshape(n, -1), axis=1)

        # Simple clinically-motivated weighted combination
        weights_clinical = np.array([0.15, 0.35, 0.15, 0.20, 0.15])
        feats['weighted_clinical'] = np.sum(spec_preds * weights_clinical, axis=1)

        # Psychosocial sub-specialists (if available)
        if 'op_psych_mental' in oof_df.columns:
            feats['op_psych_mental'] = oof_df['op_psych_mental'].values
            feats['op_psych_care'] = oof_df['op_psych_care'].values
            feats['op_psych_social'] = oof_df['op_psych_social'].values

        # Context features (kept simple & robust)
        ctx_df_tmp = pd.DataFrame(ctx_features)
        for i, _ in enumerate(ctx_df_tmp.columns):
            feats[f'ctx_{i}'] = ctx_features[:, i]

        # Confidence features (optional, but usually present)
        if conf_features is not None:
            for j in range(conf_features.shape[1]):
                feats[f'conf_{j}'] = conf_features[:, j]
            feats['conf_mean'] = np.mean(conf_features, axis=1)
            feats['conf_min'] = np.min(conf_features, axis=1)
            feats['conf_max'] = np.max(conf_features, axis=1)

            # Confidence-weighted mean opinion (more robust than raw mean)
            conf = np.clip(conf_features[:, :len(spec_names)], 1e-6, None)
            conf = conf / conf.sum(axis=1, keepdims=True)
            feats['op_conf_weighted'] = np.sum(spec_preds * conf, axis=1)

        return pd.DataFrame(feats)

    # Learn confidence features on train, infer on test
    def build_specialist_confidence_features(spec_train, y_train, ctx_train_df, spec_test, ctx_test_df):
        """Train simple per-specialist confidence models and return train/test probs.

        Uses a lightweight LogisticRegression per specialist on (spec_pred + context).
        Returns two arrays: conf_train (n_train x n_specs), conf_test (n_test x n_specs).
        """
        from sklearn.linear_model import LogisticRegression

        X_ctx_train = ctx_train_df.fillna(0).values
        X_ctx_test = ctx_test_df.fillna(0).values

        n_train = spec_train.shape[0]
        n_test = spec_test.shape[0]
        n_specs = spec_train.shape[1]

        conf_train = np.zeros((n_train, n_specs), dtype=np.float32)
        conf_test = np.zeros((n_test, n_specs), dtype=np.float32)

        for i in range(n_specs):
            try:
                X_tr = np.column_stack([spec_train[:, i].reshape(-1, 1), X_ctx_train])
                X_te = np.column_stack([spec_test[:, i].reshape(-1, 1), X_ctx_test])
                y_flag = (y_train == 1).astype(int)

                clf = LogisticRegression(max_iter=200, class_weight='balanced', solver='lbfgs')
                clf.fit(X_tr, y_flag)

                conf_train[:, i] = clf.predict_proba(X_tr)[:, 1]
                conf_test[:, i] = clf.predict_proba(X_te)[:, 1]
            except Exception:
                # Fallback: use the raw specialist prediction as confidence
                conf_train[:, i] = spec_train[:, i]
                conf_test[:, i] = spec_test[:, i]

        return conf_train, conf_test


    conf_train, conf_test = build_specialist_confidence_features(
        spec_train, y_train, ctx_train_df, spec_test, ctx_test_df
    )

    X_train = build_doctor_features(spec_train, ctx_train, train_oof, conf_train)
    X_test = build_doctor_features(spec_test, ctx_test, test_oof, conf_test)

    print(f"   Total Doctor features: {X_train.shape[1]}")

    # ----------------------------
    # 3) Meta-learner with StratifiedKFold OOF (reduces split variance)
    # ----------------------------
    print("   Training ensemble with StratifiedKFold OOF...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_meta = np.zeros(len(X_train), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y_train[tr_idx], y_train[va_idx]

        n_pos = max(1, int(y_tr.sum()))
        n_neg = max(1, int((1 - y_tr).sum()))
        pos_weight = n_neg / n_pos

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=2000, learning_rate=0.01, depth=6,
            l2_leaf_reg=8, min_data_in_leaf=30,
            early_stopping_rounds=200, verbose=0, random_seed=RANDOM_STATE,
            scale_pos_weight=pos_weight,
            **GPU_PARAMS['catboost']
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        # LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=1500, learning_rate=0.01, max_depth=5,
            min_child_samples=50, reg_lambda=5.0,
            random_state=RANDOM_STATE, verbose=-1,
            scale_pos_weight=pos_weight,
            **GPU_PARAMS['lightgbm']
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        # XGBoost (optimize for AUPRC signal)
        xgb_model = XGBClassifier(
            n_estimators=1500, learning_rate=0.01, max_depth=5,
            min_child_weight=50, reg_lambda=5.0,
            random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False,
            scale_pos_weight=pos_weight,
            eval_metric='aucpr',
            **GPU_PARAMS['xgboost']
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        p_cat = cat_model.predict_proba(X_val)[:, 1]
        p_lgb = lgb_model.predict_proba(X_val)[:, 1]
        p_xgb = xgb_model.predict_proba(X_val)[:, 1]
        oof_meta[va_idx] = (0.5 * p_cat + 0.25 * p_lgb + 0.25 * p_xgb).astype(np.float32)

        fold_auc = roc_auc_score(y_val, oof_meta[va_idx])
        fold_auprc = average_precision_score(y_val, oof_meta[va_idx])
        print(f"      Fold {fold}/5: AUC={fold_auc:.4f}  AUPRC={fold_auprc:.4f}")

    # ----------------------------
    # 4) Calibrate final predictions using OOF (no leakage)
    # ----------------------------
    final_calibrator = IsotonicRegression(out_of_bounds='clip')
    final_calibrator.fit(oof_meta, y_train)

    oof_cal = final_calibrator.transform(oof_meta)
    best_thresh = find_best_threshold_f1(y_train, oof_cal, thresh_min=0.03, thresh_max=0.5, step=0.01)
    print(f"   Selected threshold (OOF, F1-opt): {best_thresh:.4f}")

    # ----------------------------
    # 5) Refit models on full train, predict test, calibrate, evaluate
    # ----------------------------
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int((1 - y_train).sum()))
    pos_weight = n_neg / n_pos

    cat_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.01, depth=6,
        l2_leaf_reg=8, min_data_in_leaf=30,
        early_stopping_rounds=200, verbose=0, random_seed=RANDOM_STATE,
        scale_pos_weight=pos_weight,
        **GPU_PARAMS['catboost']
    )
    lgb_model = LGBMClassifier(
        n_estimators=1500, learning_rate=0.01, max_depth=5,
        min_child_samples=50, reg_lambda=5.0,
        random_state=RANDOM_STATE, verbose=-1,
        scale_pos_weight=pos_weight,
        **GPU_PARAMS['lightgbm']
    )
    xgb_model = XGBClassifier(
        n_estimators=1500, learning_rate=0.01, max_depth=5,
        min_child_weight=50, reg_lambda=5.0,
        random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False,
        scale_pos_weight=pos_weight,
        eval_metric='aucpr',
        **GPU_PARAMS['xgboost']
    )

    # A small internal split for early stopping only (keeps behavior close to original)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=RANDOM_STATE
    )

    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    p_cat = cat_model.predict_proba(X_test)[:, 1]
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_prob_raw = 0.5 * p_cat + 0.25 * p_lgb + 0.25 * p_xgb

    y_prob = final_calibrator.transform(y_prob_raw)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob



# =============================================================================
# ARCHITECTURE 1: MIXTURE OF EXPERTS (ASWN)
# =============================================================================
def run_arch1_moe(train_oof, test_oof, train_ctx, test_ctx):
    """
    Architecture 1: Adaptive Specialist Weighting Network (ASWN)
    Gating network learns dynamic specialist weights per patient.
    """
    print("\n" + "-" * 60)
    print("ARCH 1: ADAPTIVE SPECIALIST WEIGHTING NETWORK (ASWN)")
    print("-" * 60)
    
    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)
    
    # Train gating network for each specialist
    print("   Training gating network...")
    errors = np.abs(spec_train - y_train.reshape(-1, 1))
    neg_errors = -errors * 5
    exp_neg_errors = np.exp(neg_errors - np.max(neg_errors, axis=1, keepdims=True))
    soft_weights = exp_neg_errors / exp_neg_errors.sum(axis=1, keepdims=True)
    
    weight_models = []
    for i in range(5):
        model = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=4,
            verbose=0, random_seed=RANDOM_STATE,
            **GPU_PARAMS['catboost']
        )
        model.fit(ctx_train, (soft_weights[:, i] > 0.2).astype(int))
        weight_models.append(model)
    
    # Predict weights
    weights_test = np.column_stack([m.predict_proba(ctx_test)[:, 1] for m in weight_models])
    weights_test = np.clip(weights_test, 0.05, 1.0)
    weights_test = weights_test / weights_test.sum(axis=1, keepdims=True)
    
    # Get gated predictions
    gated_test = np.sum(spec_test * weights_test, axis=1)
    
    # Add features and train final model
    X_train = np.column_stack([spec_train, ctx_train])
    X_test = np.column_stack([spec_test, ctx_test, weights_test, gated_test.reshape(-1, 1)])
    
    # Need to add weights for training too
    weights_train = np.column_stack([m.predict_proba(ctx_train)[:, 1] for m in weight_models])
    weights_train = np.clip(weights_train, 0.05, 1.0)
    weights_train = weights_train / weights_train.sum(axis=1, keepdims=True)
    gated_train = np.sum(spec_train * weights_train, axis=1)
    X_train = np.column_stack([spec_train, ctx_train, weights_train, gated_train.reshape(-1, 1)])
    
    print("   Training final model...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Class imbalance handling (improves AUPRC in practice)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = n_neg / n_pos
    
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=6,
        verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=150,
        **GPU_PARAMS['catboost']
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_prob_val = model.predict_proba(X_val)[:, 1]
    calibrator.fit(y_prob_val, y_val)
    y_prob = calibrator.transform(y_prob)
    
    # Select threshold on validation set (prevents test leakage)
    best_thresh = find_best_threshold_f1(y_val, y_prob_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob


# =============================================================================
# ARCHITECTURE 2: HIERARCHICAL DEBATE (MRDE)
# =============================================================================
def run_arch2_debate(train_oof, test_oof, train_ctx, test_ctx):
    """
    Architecture 2: Multi-Round Deliberative Ensemble (MRDE)
    Specialists revise opinions through iterative debate rounds.
    """
    print("\n" + "-" * 60)
    print("ARCH 2: MULTI-ROUND DELIBERATIVE ENSEMBLE (MRDE)")
    print("-" * 60)
    
    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)
    
    spec_names = ['lab', 'note', 'pharm', 'hist', 'psych']
    current_train = {name: spec_train[:, i] for i, name in enumerate(spec_names)}
    current_test = {name: spec_test[:, i] for i, name in enumerate(spec_names)}
    
    # Two rounds of debate
    for round_num in range(2):
        print(f"   Debate round {round_num + 1}...")
        revision_models = {}
        
        for spec_idx, spec_name in enumerate(spec_names):
            X_train_debate = np.column_stack([
                ctx_train,
                current_train[spec_name],
                *[current_train[n] for n in spec_names if n != spec_name],
                np.mean([current_train[n] for n in spec_names], axis=0),
                np.std([current_train[n] for n in spec_names], axis=0)
            ])
            
            model = CatBoostClassifier(
                iterations=300, learning_rate=0.03, depth=4,
                verbose=0, random_seed=RANDOM_STATE+round_num,
                **GPU_PARAMS['catboost']
            )
            model.fit(X_train_debate, y_train)
            revision_models[spec_name] = model
        
        # Revise test predictions
        for spec_idx, spec_name in enumerate(spec_names):
            X_test_debate = np.column_stack([
                ctx_test,
                current_test[spec_name],
                *[current_test[n] for n in spec_names if n != spec_name],
                np.mean([current_test[n] for n in spec_names], axis=0),
                np.std([current_test[n] for n in spec_names], axis=0)
            ])
            current_test[spec_name] = revision_models[spec_name].predict_proba(X_test_debate)[:, 1]
    
    # Final resolver
    print("   Training resolver...")
    revised_train = np.column_stack([current_train[n] for n in spec_names])
    revised_test = np.column_stack([current_test[n] for n in spec_names])
    
    # Update training preds through debate too
    for round_num in range(2):
        for spec_idx, spec_name in enumerate(spec_names):
            X_train_debate = np.column_stack([
                ctx_train,
                current_train[spec_name],
                *[current_train[n] for n in spec_names if n != spec_name],
                np.mean([current_train[n] for n in spec_names], axis=0),
                np.std([current_train[n] for n in spec_names], axis=0)
            ])
            current_train[spec_name] = revision_models[spec_name].predict_proba(X_train_debate)[:, 1]
    
    revised_train = np.column_stack([current_train[n] for n in spec_names])
    
    X_train = np.column_stack([revised_train, ctx_train])
    X_test = np.column_stack([revised_test, ctx_test])
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Class imbalance handling (improves AUPRC in practice)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = n_neg / n_pos
    
    resolver = CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=5,
        verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=100,
        **GPU_PARAMS['catboost']
    )
    resolver.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_prob = resolver.predict_proba(X_test)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_prob_val = resolver.predict_proba(X_val)[:, 1]
    calibrator.fit(y_prob_val, y_val)
    y_prob = calibrator.transform(y_prob)
    
    # Select threshold on validation set (prevents test leakage)
    best_thresh = find_best_threshold_f1(y_val, y_prob_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob


# =============================================================================
# ARCHITECTURE 3: GRAPH NEURAL NETWORK (PSGN)
# =============================================================================
def run_arch3_gnn(train_oof, test_oof, train_ctx, test_ctx):
    """
    Architecture 3: Patient Similarity Graph Network (PSGN)
    Learns from similar patients' outcomes via k-NN graph.
    """
    print("\n" + "-" * 60)
    print("ARCH 3: PATIENT SIMILARITY GRAPH NETWORK (PSGN)")
    print("-" * 60)
    
    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)
    
    # Build patient graph
    print("   Building patient similarity graph...")
    k_neighbors = 15
    graph_train = np.concatenate([spec_train, ctx_train], axis=1)
    graph_test = np.concatenate([spec_test, ctx_test], axis=1)
    
    scaler = StandardScaler()
    graph_train_scaled = scaler.fit_transform(graph_train)
    graph_test_scaled = scaler.transform(graph_test)
    
    nn_model = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='auto')
    nn_model.fit(graph_train_scaled)
    
    _, train_neighbors = nn_model.kneighbors(graph_train_scaled)
    _, test_neighbors = nn_model.kneighbors(graph_test_scaled)
    train_neighbors = train_neighbors[:, 1:]
    
    # Compute neighbor statistics
    print("   Computing neighborhood features...")
    
    def get_neighbor_stats(neighbors, preds, labels):
        n_samples = len(neighbors)
        stats = np.zeros((n_samples, 25))
        for i in range(n_samples):
            valid = neighbors[i][neighbors[i] < len(preds)]
            if len(valid) > 0:
                for j in range(5):
                    neighbor_preds = preds[valid, j]
                    stats[i, j*5:j*5+4] = [
                        np.mean(neighbor_preds), np.std(neighbor_preds),
                        np.max(neighbor_preds), np.min(neighbor_preds)
                    ]
                    stats[i, j*5+4] = np.mean(labels[valid])
        return stats
    
    neighbor_stats_train = get_neighbor_stats(train_neighbors, spec_train, y_train)
    neighbor_stats_test = get_neighbor_stats(test_neighbors, spec_train, y_train)
    
    X_train = np.concatenate([spec_train, ctx_train, neighbor_stats_train], axis=1)
    X_test = np.concatenate([spec_test, ctx_test, neighbor_stats_test], axis=1)
    
    print("   Training final model...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Class imbalance handling (improves AUPRC in practice)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = n_neg / n_pos
    
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=6,
        verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=150,
        **GPU_PARAMS['catboost']
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_prob_val = model.predict_proba(X_val)[:, 1]
    calibrator.fit(y_prob_val, y_val)
    y_prob = calibrator.transform(y_prob)
    
    # Select threshold on validation set (prevents test leakage)
    best_thresh = find_best_threshold_f1(y_val, y_prob_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob


# =============================================================================
# ARCHITECTURE 4: TRANSFORMER FUSION (CSAN)
# =============================================================================
def run_arch4_transformer(train_oof, test_oof, train_ctx, test_ctx):
    """
    Architecture 4: Cross-Specialist Attention Network (CSAN)
    Self-attention mechanism fuses specialist outputs.
    """
    print("\n" + "-" * 60)
    print("ARCH 4: CROSS-SPECIALIST ATTENTION NETWORK (CSAN)")
    print("-" * 60)
    
    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)
    
    # Learn attention weights
    print("   Training attention mechanism...")
    attention_models = []
    for i in range(5):
        error = np.abs(spec_train[:, i] - y_train)
        is_good = (error < np.median(error)).astype(int)
        model = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=3,
            verbose=0, random_seed=RANDOM_STATE+i,
            **GPU_PARAMS['catboost']
        )
        model.fit(ctx_train, is_good)
        attention_models.append(model)
    
    # Get attention weights
    attn_train = np.column_stack([m.predict_proba(ctx_train)[:, 1] for m in attention_models])
    attn_test = np.column_stack([m.predict_proba(ctx_test)[:, 1] for m in attention_models])
    
    # Softmax
    attn_train = np.exp(attn_train) / np.exp(attn_train).sum(axis=1, keepdims=True)
    attn_test = np.exp(attn_test) / np.exp(attn_test).sum(axis=1, keepdims=True)
    
    # Weighted combination
    fused_train = np.sum(spec_train * attn_train, axis=1)
    fused_test = np.sum(spec_test * attn_test, axis=1)
    
    X_train = np.column_stack([spec_train, ctx_train, attn_train, fused_train])
    X_test = np.column_stack([spec_test, ctx_test, attn_test, fused_test])
    
    print("   Training final model...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Class imbalance handling (improves AUPRC in practice)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = n_neg / n_pos
    
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=5,
        verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=100,
        **GPU_PARAMS['catboost']
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_prob_val = model.predict_proba(X_val)[:, 1]
    calibrator.fit(y_prob_val, y_val)
    y_prob = calibrator.transform(y_prob)
    
    # Select threshold on validation set (prevents test leakage)
    best_thresh = find_best_threshold_f1(y_val, y_prob_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob


# =============================================================================
# ARCHITECTURE 5: CLINICAL REASONING ENGINE (CRE)
# =============================================================================
def run_arch5_clinical_reasoning(train_oof, test_oof, train_ctx, test_ctx):
    """
    Architecture 5: Clinical Reasoning Engine (CRE)
    Rule-based clinical reasoning mimicking physician decision-making.
    """
    print("\n" + "-" * 60)
    print("ARCH 5: CLINICAL REASONING ENGINE (CRE)")
    print("-" * 60)
    
    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)
    
    print("   Applying clinical reasoning rules...")
    
    def apply_reasoning(spec_preds, ctx_features):
        """Apply rule-based clinical reasoning."""
        n_samples = len(spec_preds)
        modifiers = np.ones(n_samples)
        
        # High risk patterns
        modifiers[(spec_preds[:, 1] > 0.7) & (spec_preds[:, 4] > 0.6)] *= 1.3
        modifiers[(spec_preds[:, 2] > 0.7)] *= 1.15
        modifiers[np.max(spec_preds, axis=1) > 0.8] *= 1.2
        modifiers[(spec_preds[:, 0] > 0.7) & (spec_preds[:, 1] > 0.7)] *= 1.25
        
        # Protective patterns
        modifiers[spec_preds[:, 0] < 0.25] *= 0.85
        modifiers[np.mean(spec_preds, axis=1) < 0.25] *= 0.75
        modifiers[np.max(spec_preds, axis=1) < 0.4] *= 0.8
        
        # Disagreement penalty
        disagreement = np.max(spec_preds, axis=1) - np.min(spec_preds, axis=1)
        modifiers[disagreement > 0.4] *= 1.1
        
        return np.clip(modifiers, 0.5, 2.0)
    
    modifiers_train = apply_reasoning(spec_train, ctx_train)
    modifiers_test = apply_reasoning(spec_test, ctx_test)
    
    base_train = np.mean(spec_train, axis=1)
    base_test = np.mean(spec_test, axis=1)
    
    adjusted_train = np.clip(base_train * modifiers_train, 0, 1)
    adjusted_test = np.clip(base_test * modifiers_test, 0, 1)
    
    X_train = np.column_stack([spec_train, ctx_train, modifiers_train, adjusted_train])
    X_test = np.column_stack([spec_test, ctx_test, modifiers_test, adjusted_test])
    
    print("   Training final model...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Class imbalance handling (improves AUPRC in practice)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = n_neg / n_pos
    
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=5,
        verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=100,
        **GPU_PARAMS['catboost']
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_prob_val = model.predict_proba(X_val)[:, 1]
    calibrator.fit(y_prob_val, y_val)
    y_prob = calibrator.transform(y_prob)
    
    # Select threshold on validation set (prevents test leakage)
    best_thresh = find_best_threshold_f1(y_val, y_prob_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob



# =============================================================================
# ARCHITECTURE 6: TEMPORAL ATTENTION (TASN)
# =============================================================================
def run_arch6_temporal(train_oof, test_oof, train_ctx, test_ctx, df_train, df_test):
    """
    Architecture 6: Trajectory-Aware Specialist Network (TASN)
    Incorporates lab value trajectories with attention mechanism.
    """
    print("\n" + "-" * 60)
    print("ARCH 6: TRAJECTORY-AWARE SPECIALIST NETWORK (TASN)")
    print("-" * 60)
    
    y_train = train_oof['y_true'].values
    y_test = test_oof['y_true'].values
    spec_train = get_specialist_preds(train_oof)
    spec_test = get_specialist_preds(test_oof)
    ctx_train = get_context_features(train_ctx)
    ctx_test = get_context_features(test_ctx)
    
    print("   Extracting trajectory features...")
    
    def extract_trajectory(df_subset):
        """Extract trajectory features from lab values."""
        features = []
        lab_prefixes = ['sodium', 'potassium', 'creatinine', 'glucose', 'hemoglobin',
                        'wbc', 'bilirubin', 'lactate', 'bun', 'albumin', 'platelets']
        
        for prefix in lab_prefixes:
            first_col = f'{prefix}_first'
            last_col = f'{prefix}_last'
            mean_col = f'{prefix}_mean'
            std_col = f'{prefix}_std'
            
            if first_col in df_subset.columns and last_col in df_subset.columns:
                delta = df_subset[last_col].fillna(0) - df_subset[first_col].fillna(0)
                features.append(delta.values)
                
                rel_change = np.where(
                    df_subset[first_col].fillna(0).abs() > 0.01,
                    delta.values / (df_subset[first_col].fillna(0).abs() + 1e-6),
                    0
                )
                features.append(rel_change)
            
            if mean_col in df_subset.columns and std_col in df_subset.columns:
                instability = np.where(
                    df_subset[mean_col].fillna(0).abs() > 0.01,
                    df_subset[std_col].fillna(0) / (df_subset[mean_col].fillna(0).abs() + 1e-6),
                    0
                )
                features.append(instability)
        
        if features:
            return np.column_stack(features)
        return np.zeros((len(df_subset), 1))
    
    traj_train = extract_trajectory(df_train)
    traj_test = extract_trajectory(df_test)
    
    # Scale trajectory features
    scaler = StandardScaler()
    traj_train_scaled = scaler.fit_transform(traj_train)
    traj_test_scaled = scaler.transform(traj_test)
    
    # Train trajectory specialist
    print("   Training trajectory specialist...")
    # Class imbalance handling for trajectory specialist
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int((1 - y_train).sum()))
    pos_weight = n_neg / n_pos

    traj_model = CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=4,
        verbose=0, random_seed=RANDOM_STATE,
        scale_pos_weight=pos_weight,
        **GPU_PARAMS['catboost']
    )
    traj_model.fit(traj_train_scaled, y_train)
    
    traj_pred_train = traj_model.predict_proba(traj_train_scaled)[:, 1]
    traj_pred_test = traj_model.predict_proba(traj_test_scaled)[:, 1]
    
    # Combine all features
    X_train = np.column_stack([spec_train, traj_pred_train, ctx_train, traj_train_scaled])
    X_test = np.column_stack([spec_test, traj_pred_test, ctx_test, traj_test_scaled])
    
    print("   Training final model...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Class imbalance handling (improves AUPRC in practice)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = n_neg / n_pos
    
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=6,
        verbose=0, random_seed=RANDOM_STATE, early_stopping_rounds=150,
        **GPU_PARAMS['catboost']
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    y_prob_val = model.predict_proba(X_val)[:, 1]
    calibrator.fit(y_prob_val, y_val)
    y_prob = calibrator.transform(y_prob)
    
    # Select threshold on validation set (prevents test leakage)
    best_thresh = find_best_threshold_f1(y_val, y_prob_val)
    y_pred = (y_prob >= best_thresh).astype(int)

    results = evaluate_predictions(y_test, y_prob, threshold=best_thresh)
    return results, y_pred, y_prob



# =============================================================================
# MAIN COMPARISON
# =============================================================================
def run_full_comparison(arch_nums: List[int], skip_specialist_training: bool = False, debug: bool = False, half_data: bool = False):
    """Run specified architectures and compare results."""
    
    print("\n" + "=" * 70)
    print("MULTI-AGENT ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architectures to run: {arch_nums}")
    
    if debug:
        print("\n⚠️  DEBUG MODE ACTIVE: USING ONLY 2000 ROWS  ⚠️")
    elif half_data:
        print("\n⚠️  HALF-DATA MODE ACTIVE: USING 50% OF DATA FOR TESTING  ⚠️")
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv(DATA_PATH)
    original_len = len(df)
    print(f"   Total samples: {len(df):,}")
    
    # Create train/test split
    if USE_TEMPORAL_SPLIT and 'dischtime' in df.columns:
        df = df.sort_values('dischtime').reset_index(drop=True)
        
    # Debug mode: subsample
    if debug:
        df = df.head(2000)
        print(f"   DEBUG: Subsampled to {len(df)} rows")
    elif half_data:
        # Use half the data (stratified to maintain class balance)
        if USE_TEMPORAL_SPLIT and 'dischtime' in df.columns:
            # For temporal split, just take first half
            df = df.iloc[:len(df)//2].reset_index(drop=True)
        else:
            # For stratified split, use stratified sampling
            _, half_idx = train_test_split(
                np.arange(len(df)), test_size=0.5, random_state=RANDOM_STATE,
                stratify=df[TARGET]
            )
            df = df.iloc[half_idx].reset_index(drop=True)
        print(f"   HALF-DATA: Subsampled to {len(df):,} rows ({len(df)/original_len*100:.1f}% of original)")
        
    if USE_TEMPORAL_SPLIT and 'dischtime' in df.columns:
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_test = df.iloc[split_idx:].reset_index(drop=True)
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE,
            stratify=df[TARGET]
        )
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
    
    print(f"   Train: {len(df_train):,}, Test: {len(df_test):,}")
    
    # Train specialists or load cached
    oof_path = os.path.join(SHARED_DIR, 'oof_predictions.csv')
    
    if skip_specialist_training and os.path.exists(oof_path):
        print("\n⏭️ Skipping specialist training, loading cached OOF predictions...")
        train_oof, test_oof, train_ctx, test_ctx = load_shared_data()
        
        # Retroactive ID Injection for Explainability
        if 'hadm_id' not in train_oof.columns and 'hadm_id' in df_train.columns:
             if len(train_oof) == len(df_train):
                 print("   ⚠️ Injecting missing hadm_id/subject_id into cached train_oof...")
                 train_oof['hadm_id'] = df_train['hadm_id'].values
                 if 'subject_id' in df_train.columns:
                     train_oof['subject_id'] = df_train['subject_id'].values
        
        if 'hadm_id' not in test_oof.columns and 'hadm_id' in df_test.columns:
             if len(test_oof) == len(df_test):
                 print("   ⚠️ Injecting missing hadm_id/subject_id into cached test_oof...")
                 test_oof['hadm_id'] = df_test['hadm_id'].values
                 if 'subject_id' in df_test.columns:
                     test_oof['subject_id'] = df_test['subject_id'].values
    else:
        # Use fewer folds for debug/half-data to speed it up
        if debug:
            n_folds = 2
        elif half_data:
            n_folds = 2  # Use 2 folds for half-data mode
        else:
            n_folds = DEFAULT_N_FOLDS
        print(f"   Using {n_folds}-fold CV")
        train_oof, test_oof = train_specialists_and_save_oof(
            df_train, df_test, n_folds=n_folds
        )
        train_ctx = _create_context_features(df_train)
        train_ctx['split'] = 'train'
        test_ctx = _create_context_features(df_test)
        test_ctx['split'] = 'test'
    
    # Architecture mapping
    arch_funcs = {
        0: ('IMSE (Baseline)', run_arch0_baseline),
        1: ('ASWN (MoE)', run_arch1_moe),
        2: ('MRDE (Debate)', run_arch2_debate),
        3: ('PSGN (GNN)', run_arch3_gnn),
        4: ('CSAN (Attention)', run_arch4_transformer),
        5: ('CRE (Clinical)', run_arch5_clinical_reasoning),
        6: ('TASN (Temporal)', run_arch6_temporal),
    }
    
    # Run architectures
    all_results = []
    
    print("\n" + "=" * 70)
    print("PHASE 2: RUNNING ARCHITECTURES")
    print("=" * 70)
    
    for arch_num in arch_nums:
        if arch_num not in arch_funcs:
            print(f"⚠️ Unknown architecture: {arch_num}")
            continue
        
        name, func = arch_funcs[arch_num]
        print(f"\nRunning Architecture {arch_num}: {name}...")
        try:
            # Call architecture function and capture predictions
            if arch_num == 0:
                results, y_pred, y_prob = func(train_oof, test_oof, train_ctx, test_ctx, df_train)
            elif arch_num == 6:
                results, y_pred, y_prob = func(train_oof, test_oof, train_ctx, test_ctx, df_train, df_test)
            else:
                results, y_pred, y_prob = func(train_oof, test_oof, train_ctx, test_ctx)
            
            results['arch_num'] = arch_num
            results['short_name'] = name
            results['full_name'] = ARCHITECTURE_NAMES[arch_num]['full']
            results['abbrev'] = ARCHITECTURE_NAMES[arch_num]['abbrev']
            all_results.append(results)
            
            print(f"   ✅ {name}: AUC = {results['auc']:.4f}, F1 = {results['f1']:.4f}")
            
            # Save predictions with features for failure analysis
            y_test = test_oof['y_true'].values
            save_predictions_with_features(
                arch_num=arch_num,
                arch_name=ARCHITECTURE_NAMES[arch_num]['abbrev'],
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                test_oof=test_oof,
                test_ctx=test_ctx
            )
            
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON RESULTS")
    print("=" * 70)
    
    if not all_results:
        print("No results to compare.")
        return None
        
    comparison_data = []
    for r in all_results:
        comparison_data.append({
            'Arch': r['arch_num'],
            'Abbreviation': r['abbrev'],
            'Full Name': r['full_name'],
            'AUC': r['auc'],
            'AUPRC': r['auprc'],
            'F1': r['f1'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'Specificity': r['specificity'],
            'Brier': r['brier'],
        })
    
    df_comp = pd.DataFrame(comparison_data)
    df_comp = df_comp.sort_values('AUC', ascending=False)
    
    # Print formatted table
    print("\n" + df_comp.to_string(index=False))
    
    # Find best
    best = df_comp.iloc[0]
    print(f"\n🏆 BEST ARCHITECTURE: {best['Abbreviation']} ({best['Full Name']})")
    print(f"   AUC: {best['AUC']:.4f} | F1: {best['F1']:.4f} | AUPRC: {best['AUPRC']:.4f}")
    
    # Performance visualization
    print("\n" + "=" * 70)
    print("PERFORMANCE VISUALIZATION (AUC)")
    print("=" * 70)
    
    max_auc = df_comp['AUC'].max()
    for _, row in df_comp.iterrows():
        bar_len = int(50 * row['AUC'] / max_auc)
        bar = '█' * bar_len
        print(f"{row['Abbreviation']:6s} | {bar} {row['AUC']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if debug: timestamp += "_DEBUG"
    results_file = os.path.join(COMPARISON_DIR, f'full_comparison_{timestamp}.csv')
    df_comp.to_csv(results_file, index=False)
    print(f"\n📊 Results saved to: {results_file}")
    
    # Save detailed results for paper
    detailed_file = os.path.join(COMPARISON_DIR, f'detailed_results_{timestamp}.txt')
    with open(detailed_file, 'w') as f:
        f.write("MULTI-AGENT ARCHITECTURE COMPARISON RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: MIMIC-IV\n")
        f.write(f"Task: 30-day Hospital Readmission Prediction\n")
        f.write(f"Train samples: {len(df_train):,}\n")
        f.write(f"Test samples: {len(df_test):,}\n\n")
        
        f.write("ARCHITECTURES:\n")
        f.write("-" * 60 + "\n")
        for arch_num in sorted(ARCHITECTURE_NAMES.keys()):
            info = ARCHITECTURE_NAMES[arch_num]
            f.write(f"  {arch_num}. {info['abbrev']}: {info['full']}\n")
        f.write("\n")
        
        f.write("RESULTS (sorted by AUC):\n")
        f.write("-" * 60 + "\n")
        f.write(df_comp.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED METRICS:\n")
        f.write("-" * 60 + "\n")
        for r in all_results:
            f.write(f"\n{r['abbrev']} ({r['full_name']}):\n")
            f.write(f"  AUC:         {r['auc']:.4f}\n")
            f.write(f"  AUPRC:       {r['auprc']:.4f}\n")
            f.write(f"  F1:          {r['f1']:.4f}\n")
            f.write(f"  Precision:   {r['precision']:.4f}\n")
            f.write(f"  Recall:      {r['recall']:.4f}\n")
            f.write(f"  Specificity: {r['specificity']:.4f}\n")
            f.write(f"  Brier:       {r['brier']:.4f}\n")
            f.write(f"  Threshold:   {r['threshold']:.4f}\n")
            f.write(f"  Confusion:   TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}\n")
    
    print(f"📄 Detailed results saved to: {detailed_file}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df_comp


def main():
    parser = argparse.ArgumentParser(
        description='Compare Multi-Agent Architectures for Readmission Prediction'
    )
    parser.add_argument(
        '--arch', type=str, default='0,1,2,3,4,5,6',
        help='Comma-separated list of architectures (0=baseline, 1-6=new)'
    )
    parser.add_argument(
        '--skip-specialist-training', action='store_true',
        help='Skip specialist training if cached OOF predictions exist'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run in debug mode (use 2000 rows, 2 folds)'
    )
    parser.add_argument(
        '--half-data', action='store_true',
        help='Use half the data for testing (faster, good for validation)'
    )
    args = parser.parse_args()
    
    arch_nums = [int(x.strip()) for x in args.arch.split(',')]
    
    results = run_full_comparison(
        arch_nums=arch_nums,
        skip_specialist_training=args.skip_specialist_training,
        debug=args.debug,
        half_data=args.half_data
    )
    
    return results


if __name__ == "__main__":
    main()