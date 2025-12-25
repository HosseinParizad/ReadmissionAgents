"""
CENTRALIZED CONFIGURATION
==========================
All shared constants, paths, and feature lists should be defined here.
This is the single source of truth for the entire project.

Architecture:
- 5 Interpretable Specialist Agents (Lab, Note, Pharmacy, History, Psychosocial)
- K-Fold Out-of-Fold Stacking
- CatBoost/LightGBM/XGBoost Ensemble
- All features are clinically interpretable

Source Files:
- extract_data.py: Data extraction from MIMIC-IV
- specialist_agents.py: 5 domain specialist agents
- train_model.py: Doctor Agent training and evaluation
"""

import os

# =============================================================================
# DATA PATHS
# =============================================================================
BASE_PATH = 'D:/Parizad/PHD/Project/Data/mimic-iv-2.2/'
OUTPUT_FILE = './model_outputs/extracted_features.csv'
DATA_PATH = './model_outputs/extracted_features.csv'  # Alias for OUTPUT_FILE

# Model paths
MODEL_PATH = './model_outputs/doctor_agent.pkl'  # Default model save/load path

# MIMIC-IV File paths
FILE_PATHS = {
    'admissions': os.path.join(BASE_PATH, 'hosp/admissions.csv'),
    'patients': os.path.join(BASE_PATH, 'hosp/patients.csv'),
    'labevents': os.path.join(BASE_PATH, 'hosp/labevents.csv'),
    'chartevents': os.path.join(BASE_PATH, 'icu/chartevents.csv'),
    'discharge': os.path.join(BASE_PATH, 'hosp/discharge.csv'),
    'prescriptions': os.path.join(BASE_PATH, 'hosp/prescriptions.csv'),
    'diagnoses': os.path.join(BASE_PATH, 'hosp/diagnoses_icd.csv'),
    'procedures': os.path.join(BASE_PATH, 'hosp/procedures_icd.csv'),
    'services': os.path.join(BASE_PATH, 'hosp/services.csv'),
    'icustays': os.path.join(BASE_PATH, 'icu/icustays.csv'),
    'edstays': os.path.join(BASE_PATH, 'ed/edstays.csv'),
    'transfers': os.path.join(BASE_PATH, 'hosp/transfers.csv'),
}

# =============================================================================
# DATA EXTRACTION CONSTANTS
# =============================================================================
CHUNK_SIZE = 500000
READMISSION_WINDOW_DAYS = 30
MIN_LOS_DAYS = 1.0
MAX_LOS_DAYS = 100.0
MIN_OBSERVABLE_DAYS = 30

# =============================================================================
# MODEL TRAINING CONSTANTS
# =============================================================================
TARGET = 'readmitted_30d'
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_SIZE = 0.15  # 15% of training data for calibration
RANDOM_STATE = 42
USE_TEMPORAL_SPLIT = True  # Use temporal (time-based) train/test split

# =============================================================================
# K-FOLD CROSS-VALIDATION SETTINGS
# =============================================================================
DEFAULT_N_FOLDS = 5  # Default number of folds for OOF stacking

# =============================================================================
# CATBOOST HYPERPARAMETERS
# =============================================================================
CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.008,
    'depth': 6,
    'l2_leaf_reg': 8,
    'min_data_in_leaf': 30,
    'rsm': 0.8,
    'subsample': 0.8,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 200,
    'use_best_model': True,
    'border_count': 254,
    'grow_policy': 'Lossguide',
    'max_leaves': 64,
    'random_seed': RANDOM_STATE,
}

# =============================================================================
# SPECIALIST AGENT HYPERPARAMETERS
# =============================================================================
SPECIALIST_PARAMS = {
    'learning_rate': 0.02,
    'max_iter': 500,
    'max_depth': 4,
    'min_samples_leaf': 100,
    'l2_regularization': 3.0,
}

# TF-IDF settings for text specialists
TFIDF_MAX_FEATURES = 600

# LDA settings for NoteSpecialist
LDA_N_TOPICS = 30
LDA_MAX_ITER = 20

# =============================================================================
# FEATURE LISTS
# =============================================================================
# Context features passed to Doctor Agent
DOCTOR_CONTEXT_FEATURES = [
    # Demographics
    'anchor_age', 'gender_M', 'age_over_65', 'age_over_75', 'age_over_85',
    
    # Admission characteristics
    'los_days', 'emergency_admit', 'had_icu_stay', 'icu_days', 'came_from_ed',
    'is_surgery', 'high_risk_service',
    
    # Clinical complexity
    'charlson_score', 'n_diagnoses', 'n_procedures', 'n_medications',
    
    # Prior utilization (key readmission predictors)
    'n_prior_readmissions', 'prior_visits_count', 'ed_visits_6mo', 'ed_visits_30d',
    'ever_readmitted', 'frequent_flyer', 'very_frequent_flyer',
    'days_since_last_discharge', 'prev_was_readmitted',
    
    # Discharge disposition
    'dc_to_home', 'dc_to_snf', 'dc_to_rehab', 'dc_to_home_health', 'supervised_discharge',
    
    # Risk scores
    'lace_total', 'lace_plus_total',
    
    # First-timer flag
    'is_first_timer',
    
    # =========================================================================
    # PROTECTIVE FACTORS (reduces systematic false positives)
    # =========================================================================
    # These are extracted from discharge notes and disposition data.
    # All available at discharge time - NO data leakage.
    'pf_supervised_discharge',   # Going to SNF/rehab/home health
    'pf_followup_scheduled',     # Follow-up appointment mentioned in notes
    'pf_care_coordination',      # Case manager/social work/discharge planning
    'pf_family_support',         # Family present/support at home
    'pf_clinically_stable',      # Described as stable/improved at discharge
    'pf_patient_engaged',        # Patient educated/understands/compliant
    'pf_no_ama',                 # Not leaving against medical advice
    'protective_score',          # Sum of protective factors (0-6)
    'high_protection',           # protective_score >= 3
    'low_protection',            # protective_score <= 1
]

# Text columns (input to specialists)
TEXT_COLS = [
    'clinical_text',        # Discharge notes
    'med_list_text',        # Medication list
    'diagnosis_list_text',  # Diagnosis codes
    'procedure_list_text',  # Procedure codes
    'full_history_text'     # Prior admission history
]

# ID columns (not used as features)
ID_COLS = [
    'subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
    'next_admittime', 'days_to_next', 'curr_service', 'discharge_location',
    'admission_type', 'gender', 'dod', 'last_dischtime', 'discharge_dow',
    'admission_dow', 'edregtime', 'edouttime', 'primary_diagnosis'
]

# =============================================================================
# LAB/VITAL ITEM IDs (MIMIC-IV specific)
# =============================================================================
VITAL_ITEMIDS = {
    'hr': [220045],
    'sbp': [220179, 220050],
    'dbp': [220180, 220051],
    'resp_rate': [220210, 224690],
    'spo2': [220277],
    'temp': [223761, 223762],
    'map': [220052, 220181, 225312],
}

LAB_ITEMIDS = {
    # Basic metabolic panel
    'sodium': [50983],
    'potassium': [50971],
    'chloride': [50902],
    'bicarbonate': [50882],
    'bun': [51006],
    'creatinine': [50912],
    'glucose': [50931, 50809],
    
    # Complete blood count
    'wbc': [51300, 51301],
    'hemoglobin': [51222, 51221],
    'hematocrit': [51221],
    'platelets': [51265],
    
    # Liver function
    'albumin': [50862],
    'bilirubin': [50885],
    'alt': [50861],
    'ast': [50878],
    'alp': [50863],
    
    # Cardiac markers
    'bnp': [50963],
    'troponin': [51002, 51003],
    
    # Coagulation
    'inr': [51237],
    'ptt': [51275],
    
    # Other important labs
    'lactate': [50813],
    'magnesium': [50960],
    'phosphate': [50970],
    'calcium': [50893],
}

# =============================================================================
# CLINICAL REFERENCE RANGES
# =============================================================================
# Normal ranges for critical value detection
CRITICAL_RANGES = {
    'sodium': (136, 145),
    'potassium': (3.5, 5.0),
    'chloride': (96, 106),
    'bicarbonate': (22, 29),
    'bun': (7, 20),
    'creatinine': (0.6, 1.2),
    'glucose': (70, 180),
    'hemoglobin': (12.0, 17.5),
    'hematocrit': (36, 50),
    'platelets': (150, 400),
    'wbc': (4.0, 11.0),
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

# Life-threatening boundaries (panic values)
PANIC_RANGES = {
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

# =============================================================================
# INTERPRETABLE FEATURE THRESHOLDS
# =============================================================================
# Thresholds for creating interpretable interaction features
RISK_THRESHOLDS = {
    'high_risk': 0.6,       # Specialist opinion > 0.6 = high risk
    'very_high_risk': 0.75, # Specialist opinion > 0.75 = very high risk
    'low_risk': 0.3,        # Specialist opinion < 0.3 = low risk
    'disagreement': 0.3,    # |op1 - op2| > 0.3 = specialists disagree
    'consensus_count': 3,   # Need ≥3 specialists to agree for consensus
}

# Short stay threshold (potential premature discharge)
SHORT_STAY_DAYS = 3

# Polypharmacy threshold
POLYPHARMACY_THRESHOLD = 10  # ≥10 medications = polypharmacy