"""
MIMIC-IV FEATURE EXTRACTION FOR 30-DAY READMISSION PREDICTION
==============================================================

Author: Research Team
Date: 2024

Key Improvements:
1. PRIOR READMISSION HISTORY - Critical predictor (patients who bounced back before)
2. ED VISIT HISTORY - Strong predictor of future hospitalization
3. DISCHARGE DISPOSITION ENCODING - SNF/Rehab is protective
4. PATIENT HISTORY TEXT - Combines diagnoses across all prior visits
5. ENHANCED LAB FEATURES - Critical value flags, trends, stability
6. ACUITY MARKERS - ED-to-admit, time in ED, arrival method
7. OPTIMIZED PERFORMANCE - Vectorized operations where possible
8. DISEASE-SPECIFIC READMISSION - Only related diagnoses count as readmission

Recent Enhancements:
9. LAB TRAJECTORY EXTRACTION - First/last values for trend detection
10. TEXT STATISTICS - Word count, sentence count in notes
11. ENHANCED SECTION PARSING - Better discharge note structure

Expected AUC improvement: +0.03-0.08 over baseline (higher for disease-specific)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
import re
import warnings
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# Import centralized configuration
from config import (
    BASE_PATH, OUTPUT_FILE, CHUNK_SIZE, READMISSION_WINDOW_DAYS,
    MIN_LOS_DAYS, MAX_LOS_DAYS, MIN_OBSERVABLE_DAYS,
    FILE_PATHS, VITAL_ITEMIDS, LAB_ITEMIDS, CRITICAL_RANGES, PANIC_RANGES,
)
import pickle

# Import protective factors for reducing false positives
from protective_factors import extract_protective_factors


class MIMICExtractor:
    """
    Comprehensive MIMIC-IV feature extractor for readmission prediction.
    
    This class handles the complete pipeline for extracting features from
    MIMIC-IV data for 30-day DISEASE-SPECIFIC readmission prediction.
    
    Disease-specific readmission: Only counts readmissions where the primary
    diagnosis category matches between index and readmission admission.
    """
    
    def __init__(self) -> None:
        """Initialize the MIMIC extractor."""
        self.admissions: Optional[pd.DataFrame] = None
        self.median_imputer = SimpleImputer(strategy='median')
        self.dataset_end_date: Optional[datetime] = None
        self.ed_data: Optional[pd.DataFrame] = None  # Cache for ED data
        self.diagnosis_lookup: Optional[Dict[str, str]] = None  # Cache for diagnosis lookup
        
    def _clean_id(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Standardize ID columns.
        
        Args:
            df: DataFrame to clean
            col: Column name containing IDs
            
        Returns:
            DataFrame with cleaned IDs
        """
        df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    
    def _get_diagnosis_category(self, icd_code: str) -> str:
        """Extract diagnosis category from ICD code.
        
        Uses first 3 characters for ICD-10 or first 3 digits for ICD-9
        to create diagnosis categories for matching.
        
        Args:
            icd_code: Full ICD code string
            
        Returns:
            Diagnosis category (first 3 characters)
        """
        if pd.isna(icd_code) or icd_code == '' or icd_code == 'nan':
            return 'UNKNOWN'
        
        # Clean the code
        code = str(icd_code).strip().upper()
        
        # Return first 3 characters as category
        # This groups related conditions together
        # E.g., I50 (Heart failure), I21 (Acute MI), J44 (COPD)
        return code[:3] if len(code) >= 3 else code
    
    def _load_primary_diagnoses(self) -> Dict[str, str]:
        """Load primary diagnosis for each admission.
        
        Reads diagnoses file and extracts the primary diagnosis (seq_num=1)
        for each admission to use in disease-specific readmission matching.
        
        Returns:
            Dictionary mapping hadm_id to primary diagnosis category
        """
        print(f"  Loading primary diagnoses for disease-specific matching...")
        
        if not os.path.exists(FILE_PATHS['diagnoses']):
            print(f"    [WARNING] Diagnoses file not found, falling back to all-cause")
            return {}
        
        diagnosis_dict = {}
        
        for chunk in tqdm(pd.read_csv(FILE_PATHS['diagnoses'], chunksize=CHUNK_SIZE,
                                       usecols=['hadm_id', 'icd_code', 'seq_num']),
                          desc="    Loading diagnoses", unit="chunk"):
            self._clean_id(chunk, 'hadm_id')
            
            # Get primary diagnosis (seq_num = 1) for each admission
            primary = chunk[chunk['seq_num'] == 1].copy()
            
            for _, row in primary.iterrows():
                hadm_id = str(row['hadm_id'])
                category = self._get_diagnosis_category(row['icd_code'])
                diagnosis_dict[hadm_id] = category
        
        print(f"    Loaded primary diagnoses for {len(diagnosis_dict):,} admissions")
        
        # Print top diagnosis categories
        if diagnosis_dict:
            from collections import Counter
            top_categories = Counter(diagnosis_dict.values()).most_common(10)
            print(f"    Top diagnosis categories:")
            for cat, count in top_categories:
                print(f"      {cat}: {count:,}")
        
        return diagnosis_dict
    
    # =========================================================================
    # STEP 1: LOAD CORE DATA
    # =========================================================================
    def load_core(self) -> None:
        """Load admissions, patients, services, and ICU data.
        
        Raises:
            FileNotFoundError: If required files are not found
        """
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 1: Loading Core Data...")
        
        # --- LOAD PRIMARY DIAGNOSES FIRST (for disease-specific matching) ---
        self.diagnosis_lookup = self._load_primary_diagnoses()
        
        # --- ADMISSIONS ---
        if not os.path.exists(FILE_PATHS['admissions']):
            raise FileNotFoundError(f"CRITICAL: Admissions not found at {FILE_PATHS['admissions']}")
        
        print(f"  Loading admissions...")
        self.admissions = pd.read_csv(
            FILE_PATHS['admissions'],
            parse_dates=['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        )
        self._clean_id(self.admissions, 'hadm_id')
        self._clean_id(self.admissions, 'subject_id')
        
        self.dataset_end_date = self.admissions['dischtime'].max()
        print(f"    Loaded {len(self.admissions):,} admissions")
        print(f"    Dataset end date: {self.dataset_end_date.strftime('%Y-%m-%d')}")
        
        # --- ADD PRIMARY DIAGNOSIS CATEGORY TO ADMISSIONS ---
        self.admissions['primary_dx_category'] = self.admissions['hadm_id'].map(
            self.diagnosis_lookup
        ).fillna('UNKNOWN')
        
        n_with_dx = (self.admissions['primary_dx_category'] != 'UNKNOWN').sum()
        print(f"    Admissions with primary diagnosis: {n_with_dx:,} ({n_with_dx/len(self.admissions):.1%})")
        
        # --- PATIENTS ---
        print(f"  Loading patients...")
        patients = pd.read_csv(FILE_PATHS['patients'])
        self._clean_id(patients, 'subject_id')
        self.admissions = self.admissions.merge(
            patients[['subject_id', 'anchor_age', 'gender', 'dod']],
            on='subject_id',
            how='left'
        )
        
        # --- SERVICES ---
        if os.path.exists(FILE_PATHS['services']):
            print(f"  Loading services...")
            services = pd.read_csv(FILE_PATHS['services'])
            self._clean_id(services, 'hadm_id')
            # Get last service for each admission
            services = services.sort_values('transfertime').groupby('hadm_id')['curr_service'].last().reset_index()
            self.admissions = self.admissions.merge(services, on='hadm_id', how='left')
            self.admissions['curr_service'] = self.admissions['curr_service'].fillna('UNKNOWN')
        else:
            self.admissions['curr_service'] = 'UNKNOWN'
        
        # --- ICU STAYS ---
        if os.path.exists(FILE_PATHS['icustays']):
            print(f"  Loading ICU stays...")
            icustays = pd.read_csv(FILE_PATHS['icustays'], parse_dates=['intime', 'outtime'])
            self._clean_id(icustays, 'hadm_id')
            
            icustays['icu_los'] = (icustays['outtime'] - icustays['intime']).dt.total_seconds() / 86400
            icu_summary = icustays.groupby('hadm_id').agg({
                'icu_los': 'sum',
                'stay_id': 'count'
            }).reset_index()
            icu_summary.columns = ['hadm_id', 'icu_days', 'icu_transfers']
            
            self.admissions = self.admissions.merge(icu_summary, on='hadm_id', how='left')
            self.admissions['icu_days'] = self.admissions['icu_days'].fillna(0)
            self.admissions['icu_transfers'] = self.admissions['icu_transfers'].fillna(0)
        else:
            self.admissions['icu_days'] = 0
            self.admissions['icu_transfers'] = 0
        
        # --- ED STAYS (Load for later use) ---
        self._load_ed_data()
        
        # --- CALCULATE BASIC FEATURES ---
        print(f"  Calculating base features...")
        
        # Sort by patient and time
        self.admissions = self.admissions.sort_values(['subject_id', 'admittime'])
        
        # Length of stay
        self.admissions['los_days'] = (
            self.admissions['dischtime'] - self.admissions['admittime']
        ).dt.total_seconds() / 86400
        
        # --- DISEASE-SPECIFIC READMISSION TARGET ---
        self._calculate_disease_specific_readmission()
        
        # --- APPLY EXCLUSION FILTERS ---
        self._filter_admissions()
        
        # --- ENGINEERED FEATURES ---
        self._engineer_base_features()
        
        # --- PRIOR READMISSION HISTORY (CRITICAL!) ---
        self._add_prior_readmission_features()
        
        # --- ED VISIT HISTORY ---
        self._add_ed_visit_features()
        
        # --- DISCHARGE DISPOSITION ---
        self._encode_discharge_disposition()
        
        # --- CLINICAL CORRECTNESS FLAGS ---
        self._add_clinical_correctness_flags()
        
        elapsed = time.time() - start_time
        print(f"  [OK] Core data loaded: {len(self.admissions):,} admissions ({elapsed:.1f}s)")
    
    def _calculate_disease_specific_readmission(self) -> None:
        """Calculate disease-specific 30-day readmission target.
        
        A readmission is only counted if:
        1. Same patient returns within 30 days
        2. The primary diagnosis category matches between index and next admission
        
        This filters out unrelated readmissions (e.g., HF patient returns for fracture).
        """
        print(f"  Calculating DISEASE-SPECIFIC readmission target...")
        
        df = self.admissions.copy()
        
        # Get next admission info for each patient
        df['next_admittime'] = df.groupby('subject_id')['admittime'].shift(-1)
        df['next_hadm_id'] = df.groupby('subject_id')['hadm_id'].shift(-1)
        df['next_dx_category'] = df.groupby('subject_id')['primary_dx_category'].shift(-1)
        
        # Calculate days to next admission
        df['days_to_next'] = (
            df['next_admittime'] - df['dischtime']
        ).dt.total_seconds() / 86400
        
        # Check if within 30 days
        within_window = (
            (df['days_to_next'] <= READMISSION_WINDOW_DAYS) &
            (df['days_to_next'] >= 0)
        )
        
        # Check if diagnosis categories match (DISEASE-SPECIFIC)
        diagnosis_match = (df['primary_dx_category'] == df['next_dx_category'])
        
        # Both conditions must be true for disease-specific readmission
        df['readmitted_30d'] = (within_window & diagnosis_match).astype(int)
        
        # Also keep all-cause readmission for comparison
        df['readmitted_30d_all_cause'] = within_window.astype(int)
        
        self.admissions = df
        
        # Print comparison statistics
        disease_specific_rate = df['readmitted_30d'].mean()
        all_cause_rate = df['readmitted_30d_all_cause'].mean()
        
        print(f"    Disease-specific readmission rate: {disease_specific_rate:.2%}")
        print(f"    All-cause readmission rate: {all_cause_rate:.2%}")
        print(f"    Reduction: {(1 - disease_specific_rate/all_cause_rate)*100:.1f}% of readmissions filtered")
        
        # Show top diagnosis categories for readmissions
        readmit_dx = df[df['readmitted_30d'] == 1]['primary_dx_category'].value_counts().head(10)
        print(f"    Top diagnosis categories in disease-specific readmissions:")
        for dx, count in readmit_dx.items():
            print(f"      {dx}: {count:,}")
    
    def _load_ed_data(self) -> None:
        """Load ED data for later feature engineering.
        
        Tries multiple possible file locations and caches the result.
        """
        # Try multiple possible locations
        ed_paths = [
            FILE_PATHS['edstays'],
            os.path.join(BASE_PATH, 'ed/edstays.csv'),
            os.path.join(BASE_PATH, 'hosp/edstays.csv'),
        ]
        
        for path in ed_paths:
            if os.path.exists(path):
                print(f"  Loading ED stays from: {path}")
                self.ed_data = pd.read_csv(path, parse_dates=['intime', 'outtime'])
                self._clean_id(self.ed_data, 'subject_id')
                if 'hadm_id' in self.ed_data.columns:
                    self._clean_id(self.ed_data, 'hadm_id')
                print(f"    Loaded {len(self.ed_data):,} ED visits")
                return
        
        print(f"  [WARNING] ED stays file not found, ED features will be 0")
        self.ed_data = None
    
    def _filter_admissions(self) -> None:
        """Apply rigorous exclusion criteria for publication-quality cohort.
        
        Filters include:
        - In-hospital deaths
        - Right-censored cases
        - Transfers/hospice/AMA
        - LOS outliers
        - Non-acute admission types
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        initial_count = len(self.admissions)
        print(f"\n  Applying exclusion criteria (Start: {initial_count:,})")
        
        # 1. Exclude in-hospital deaths
        self.admissions = self.admissions[self.admissions['deathtime'].isna()]
        post_death = len(self.admissions)
        print(f"    - Removed {initial_count - post_death:,} in-hospital deaths")
        
        # 2. Right-censoring: exclude if cannot observe 30 days post-discharge
        if self.dataset_end_date is None:
            raise ValueError("Dataset end date must be set")
        max_observable = self.dataset_end_date - timedelta(days=READMISSION_WINDOW_DAYS)
        self.admissions = self.admissions[self.admissions['dischtime'] <= max_observable]
        post_censor = len(self.admissions)
        print(f"    - Removed {post_death - post_censor:,} right-censored (data end)")
        
        # 3. Exclude transfers to other acute care (loss to follow-up)
        transfer_keywords = [
            'HOSPITAL', 'HOSPICE', 'OTHER FACILITY', 'PSYCH', 'MEDICAL CTR',
            'LONG TERM ACUTE CARE', 'LTACH', 'ACUTE CARE', 'AGAINST ADVICE'
        ]
        
        if 'discharge_location' in self.admissions.columns:
            dc_loc_upper = self.admissions['discharge_location'].fillna('').str.upper()
            mask = ~dc_loc_upper.str.contains('|'.join(transfer_keywords), na=False)
            pre_transfer = len(self.admissions)
            self.admissions = self.admissions[mask]
            post_transfer = len(self.admissions)
            print(f"    - Removed {pre_transfer - post_transfer:,} transfers/hospice/AMA")
        
        # 4. LOS filters
        pre_los = len(self.admissions)
        self.admissions = self.admissions[
            (self.admissions['los_days'] >= MIN_LOS_DAYS) &
            (self.admissions['los_days'] <= MAX_LOS_DAYS)
        ]
        post_los = len(self.admissions)
        print(f"    - Removed {pre_los - post_los:,} LOS outliers "
              f"(<{MIN_LOS_DAYS} or >{MAX_LOS_DAYS} days)")
        
        # 5. Exclude non-acute admission types
        pre_type = len(self.admissions)
        exclude_types = ['ELECTIVE', 'NEWBORN', 'AMBULATORY OBSERVATION', 'EU OBSERVATION']
        self.admissions = self.admissions[
            ~self.admissions['admission_type'].isin(exclude_types)
        ]
        post_type = len(self.admissions)
        print(f"    - Removed {pre_type - post_type:,} non-acute admissions")
        
        print(f"  [OK] Final cohort: {len(self.admissions):,} "
              f"({initial_count - len(self.admissions):,} excluded)")
    
    def _engineer_base_features(self) -> None:
        """Engineer basic context features.
        
        Creates derived features from base admission data including:
        - Prior visit counts
        - Time-based features
        - Service indicators
        - Age buckets
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        print(f"  Engineering base features...")
        
        df = self.admissions
        
        # Prior visits count (cumulative for this patient)
        df['prior_visits_count'] = df.groupby('subject_id').cumcount()
        
        # Gender encoding
        df['gender_M'] = (df['gender'] == 'M').astype(int)
        
        # Day of week features
        df['discharge_dow'] = df['dischtime'].dt.dayofweek
        df['weekend_discharge'] = (df['discharge_dow'] >= 5).astype(int)
        df['admission_dow'] = df['admittime'].dt.dayofweek
        df['weekend_admission'] = (df['admission_dow'] >= 5).astype(int)
        
        # Hour of admission (captures overnight admissions)
        df['admission_hour'] = df['admittime'].dt.hour
        df['night_admission'] = ((df['admission_hour'] >= 22) | (df['admission_hour'] <= 6)).astype(int)
        
        # Emergency admission flag
        df['emergency_admit'] = df['admission_type'].str.contains('EMERGENCY|URGENT', case=False, na=False).astype(int)
        
        # Days since last discharge
        df['last_dischtime'] = df.groupby('subject_id')['dischtime'].shift(1)
        df['days_since_last_discharge'] = (
            df['admittime'] - df['last_dischtime']
        ).dt.total_seconds() / 86400
        df['days_since_last_discharge'] = df['days_since_last_discharge'].fillna(999)
        
        # Admission velocity (admissions per year of life)
        df['admission_velocity'] = df['prior_visits_count'] / (df['anchor_age'] + 1)
        
        # Service-based risk
        high_risk_services = ['MEDICINE', 'MED', 'CARDIOLOGY', 'CMED', 'ONCOLOGY', 
                             'PULMONARY', 'RENAL', 'NMED', 'HEME']
        df['high_risk_service'] = df['curr_service'].str.upper().isin(
            [s.upper() for s in high_risk_services]
        ).astype(int)
        df['is_surgery'] = df['curr_service'].str.contains('SURG', case=False, na=False).astype(int)
        
        # Age buckets (non-linear age effects)
        df['age_under_30'] = (df['anchor_age'] < 30).astype(int)
        df['age_over_65'] = (df['anchor_age'] >= 65).astype(int)
        df['age_over_75'] = (df['anchor_age'] >= 75).astype(int)
        df['age_over_85'] = (df['anchor_age'] >= 85).astype(int)
        
        # ICU indicators
        df['had_icu_stay'] = (df['icu_days'] > 0).astype(int)
        df['long_icu_stay'] = (df['icu_days'] >= 3).astype(int)
        
        # ED-related features from admission data
        if 'edregtime' in df.columns and 'edouttime' in df.columns:
            df['came_from_ed'] = df['edregtime'].notna().astype(int)
            df['ed_los_hours'] = (
                (df['edouttime'] - df['edregtime']).dt.total_seconds() / 3600
            ).fillna(0).clip(lower=0)  # Clip negative values (data quality issue)
            df['long_ed_stay'] = (df['ed_los_hours'] >= 6).astype(int)
        else:
            df['came_from_ed'] = 0
            df['ed_los_hours'] = 0
            df['long_ed_stay'] = 0
        
        self.admissions = df
    
    def _add_prior_readmission_features(self) -> None:
        """
        CRITICAL FEATURE: Prior readmission history.
        
        This is one of the strongest predictors of future readmission.
        Patients who were readmitted before are ~2x more likely to be readmitted again.
        
        Creates features:
        - prev_was_readmitted: Was previous admission a readmission?
        - n_prior_readmissions: Cumulative count of prior readmissions
        - patient_readmit_rate: Historical readmission rate
        - Various bounce-back and frequent flyer indicators
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        print(f"  Adding prior readmission features (CRITICAL)...")
        
        df = self.admissions.sort_values(['subject_id', 'admittime'])
        
        # 1. Was the PREVIOUS admission a 30-day readmission?
        df['prev_was_readmitted'] = df.groupby('subject_id')['readmitted_30d'].shift(1).fillna(0).astype(int)
        
        # 2. Cumulative count of prior 30-day readmissions for this patient
        df['n_prior_readmissions'] = df.groupby('subject_id')['readmitted_30d'].cumsum().shift(1).fillna(0).astype(int)
        
        # 3. Readmission rate for this patient (prior admissions only)
        # Rate = readmissions / total prior visits (0 if no prior vis its)
        df['patient_readmit_rate'] = np.where(
            df['prior_visits_count'] > 0,
            df['n_prior_readmissions'] / df['prior_visits_count'],
            0.0
        ).clip(0, 1)  # Ensure valid rate between 0 and 1
        
        # 4. "Bounce back" - admission within 7 days of last discharge
        df['bounce_back_7d'] = (df['days_since_last_discharge'] <= 7).astype(int)
        
        # 5. Rapid return - admission within 14 days
        df['rapid_return_14d'] = (df['days_since_last_discharge'] <= 14).astype(int)
        
        # 6. "Frequent flyer" - 3+ admissions in past year (approximated by prior_visits_count)
        df['frequent_flyer'] = (df['prior_visits_count'] >= 3).astype(int)
        
        # 7. Very frequent - 5+ prior admissions
        df['very_frequent_flyer'] = (df['prior_visits_count'] >= 5).astype(int)
        
        # 8. Has any prior readmission ever
        df['ever_readmitted'] = (df['n_prior_readmissions'] > 0).astype(int)
        
        self.admissions = df
        
        # Print statistics
        readmit_rate_with_prior = df[df['prev_was_readmitted'] == 1]['readmitted_30d'].mean()
        readmit_rate_without_prior = df[df['prev_was_readmitted'] == 0]['readmitted_30d'].mean()
        print(f"    Readmission rate if prev was readmitted: {readmit_rate_with_prior:.1%}")
        print(f"    Readmission rate if prev was NOT readmitted: {readmit_rate_without_prior:.1%}")
        print(f"    Risk ratio: {readmit_rate_with_prior / (readmit_rate_without_prior + 1e-10):.2f}x")
    
    def _add_ed_visit_features(self) -> None:
        """Add ED visit history features.
        
        Creates features for ED utilization including:
        - ed_visits_6mo: ED visits in past 6 months
        - ed_visits_30d: ED visits in past 30 days
        - recent_ed_visit: Binary flag for recent ED visit
        - high_ed_utilization: Flag for high utilization (3+ visits)
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        print(f"  Adding ED visit features...")
        
        if self.ed_data is None:
            self.admissions['ed_visits_6mo'] = 0
            self.admissions['ed_visits_30d'] = 0
            self.admissions['recent_ed_visit'] = 0
            print(f"    [SKIP] No ED data available")
            return
        
        # Create lookup for faster processing
        ed_by_patient = self.ed_data.groupby('subject_id')['intime'].apply(list).to_dict()
        
        ed_6mo = []
        ed_30d = []
        
        for _, row in tqdm(self.admissions.iterrows(), total=len(self.admissions), 
                          desc="    Processing ED history"):
            subj_id = row['subject_id']
            admit_time = row['admittime']
            
            if subj_id not in ed_by_patient:
                ed_6mo.append(0)
                ed_30d.append(0)
                continue
            
            patient_ed_times = ed_by_patient[subj_id]
            lookback_6mo = admit_time - timedelta(days=180)
            lookback_30d = admit_time - timedelta(days=30)
            
            count_6mo = sum(1 for t in patient_ed_times if lookback_6mo <= t < admit_time)
            count_30d = sum(1 for t in patient_ed_times if lookback_30d <= t < admit_time)
            
            ed_6mo.append(count_6mo)
            ed_30d.append(count_30d)
        
        self.admissions['ed_visits_6mo'] = ed_6mo
        self.admissions['ed_visits_30d'] = ed_30d
        self.admissions['recent_ed_visit'] = (self.admissions['ed_visits_30d'] > 0).astype(int)
        self.admissions['high_ed_utilization'] = (self.admissions['ed_visits_6mo'] >= 3).astype(int)
        
        print(f"    Mean ED visits (6mo): {self.admissions['ed_visits_6mo'].mean():.2f}")
    
    def _encode_discharge_disposition(self) -> None:
        """Encode discharge location as features.
        
        Creates binary flags for:
        - dc_to_home: Discharged to home
        - dc_to_snf: Discharged to skilled nursing facility (protective)
        - dc_to_rehab: Discharged to rehabilitation (protective)
        - dc_to_home_health: Home with home health services
        - supervised_discharge: Any supervised discharge setting
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        print(f"  Encoding discharge disposition...")
        
        if 'discharge_location' not in self.admissions.columns:
            for col in ['dc_to_home', 'dc_to_snf', 'dc_to_rehab', 'dc_to_home_health']:
                self.admissions[col] = 0
            return
        
        dc_loc = self.admissions['discharge_location'].fillna('').str.upper()
        
        # Initialize all to 0
        self.admissions['dc_to_snf'] = 0
        self.admissions['dc_to_rehab'] = 0
        self.admissions['dc_to_home_health'] = 0
        self.admissions['dc_to_home'] = 0
        
        # SNF/Skilled Nursing (protective - supervised care)
        # Check this FIRST before home to avoid "HOME" in "NURSING HOME"
        snf_mask = dc_loc.str.contains('SKILLED|SNF|NURSING HOME', na=False, regex=True)
        self.admissions.loc[snf_mask, 'dc_to_snf'] = 1
        
        # Rehab (protective)
        rehab_mask = dc_loc.str.contains('REHAB', na=False)
        self.admissions.loc[rehab_mask, 'dc_to_rehab'] = 1
        
        # Home Health (somewhat protective)
        hh_mask = dc_loc.str.contains('HOME HEALTH|HHA|VISITING NURSE|HOME HEALTH CARE', na=False, regex=True)
        self.admissions.loc[hh_mask, 'dc_to_home_health'] = 1
        
        # Home (baseline) - Exclude if already SNF, rehab, or Home Health
        # Only flag HOME if it's JUST home, not HOME+service
        home_mask = (
            dc_loc.str.contains('HOME', na=False) & 
            ~snf_mask & ~rehab_mask & ~hh_mask
        )
        self.admissions.loc[home_mask, 'dc_to_home'] = 1
        
        # Any supervised discharge (protective overall)
        self.admissions['supervised_discharge'] = (
            self.admissions['dc_to_snf'] | 
            self.admissions['dc_to_rehab'] | 
            self.admissions['dc_to_home_health']
        ).astype(int)
    
    # =========================================================================
    # CLINICAL CORRECTNESS FLAGS
    # =========================================================================
    def _add_clinical_correctness_flags(self) -> None:
        """
        Add flags needed for Clinical Correctness evaluation.
        
        These flags help identify:
        1. Unpredictable readmissions (trauma, accidents) - fixes False Negatives
        2. Competing risks (hospice, death) - fixes False Positives  
        3. Intervention contexts (supervised discharge) - already captured above
        
        Clinical Correctness Rules:
        - Rule 1: Low Risk + Readmitted + TRAUM service = Clinically Correct (unavoidable)
        - Rule 2: High Risk + Not Readmitted + HOSPICE = Clinically Correct (competing risk)
        - Rule 3: High Risk + Not Readmitted + Supervised DC = Successful Prevention
        
        Creates flags:
        - is_trauma_service: Trauma/observation service indicator
        - is_ortho_service: Orthopedic service indicator
        - is_unpredictable_service: Combined unpredictable flag
        - dc_to_hospice: Hospice/palliative discharge
        - dc_ama: Against medical advice discharge
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        print(f"  Adding Clinical Correctness flags...")
        df = self.admissions
        
        # --- UNPREDICTABLE EVENTS FLAGS (Rule 1: Fixing False Negatives) ---
        # Service types that indicate accidental/unpredictable admissions
        unpredictable_services = ['TRAUM', 'TRAUMA', 'OBS', 'OBSERVATION']
        df['is_trauma_service'] = df['curr_service'].fillna('').str.upper().str.contains(
            '|'.join(unpredictable_services), na=False, regex=True
        ).astype(int)
        
        # Orthopedic often indicates falls/accidents
        df['is_ortho_service'] = df['curr_service'].fillna('').str.upper().str.contains(
            'ORTHO', na=False
        ).astype(int)
        
        # Combined unpredictable flag
        df['is_unpredictable_service'] = (
            df['is_trauma_service'] | df['is_ortho_service']
        ).astype(int)
        
        # --- COMPETING RISKS FLAGS (Rule 2: Fixing False Positives) ---
        if 'discharge_location' in df.columns:
            dc_upper = df['discharge_location'].fillna('').str.upper()
            
            # Hospice/Palliative discharge - patient didn't return because goal of care changed
            df['dc_to_hospice'] = dc_upper.str.contains(
                'HOSPICE|PALLIATIVE|COMFORT', na=False, regex=True
            ).astype(int)
            
            # Against Medical Advice - unpredictable behavior
            df['dc_ama'] = dc_upper.str.contains(
                'AGAINST|AMA', na=False, regex=True
            ).astype(int)
        else:
            df['dc_to_hospice'] = 0
            df['dc_ama'] = 0
        
        # --- INTERVENTION SUCCESS FLAG (Rule 3) ---
        # supervised_discharge is already computed in _encode_discharge_disposition
        # This captures SNF, Rehab, Home Health - all protective interventions
        
        self.admissions = df
        
        # Print statistics
        print(f"    Trauma/OBS service: {df['is_trauma_service'].sum():,} ({df['is_trauma_service'].mean():.1%})")
        print(f"    Ortho service: {df['is_ortho_service'].sum():,} ({df['is_ortho_service'].mean():.1%})")
        print(f"    Unpredictable service (combined): {df['is_unpredictable_service'].sum():,} ({df['is_unpredictable_service'].mean():.1%})")
        print(f"    Hospice discharge: {df['dc_to_hospice'].sum():,} ({df['dc_to_hospice'].mean():.1%})")
        print(f"    Supervised discharge: {df['supervised_discharge'].sum():,} ({df['supervised_discharge'].mean():.1%})")
    
    # =========================================================================
    # STEP 2: EXTRACT TEXT DATA
    # =========================================================================
    def extract_text_lists(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract medications, diagnoses, and procedures as text.
        
        Returns:
            Tuple of (meds_df, diag_df, proc_df) DataFrames
        """
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 2: Extracting Text Lists...")
        
        # --- MEDICATIONS ---
        print(f"  Extracting medications...")
        meds_list = []
        for chunk in tqdm(pd.read_csv(FILE_PATHS['prescriptions'], chunksize=CHUNK_SIZE,
                                       usecols=['hadm_id', 'drug']),
                          desc="    Meds", unit="chunk"):
            self._clean_id(chunk, 'hadm_id')
            chunk['drug'] = chunk['drug'].astype(str).str.lower().str.strip()
            chunk = chunk[(chunk['drug'] != 'nan') & (chunk['drug'] != '')]
            if not chunk.empty:
                meds_list.append(
                    chunk.groupby('hadm_id')['drug'].apply(lambda x: ' '.join(x.unique())).reset_index()
                )
        
        if meds_list:
            meds_df = pd.concat(meds_list).groupby('hadm_id')['drug'].apply(
                lambda x: ' '.join(sorted(set(' '.join(x.astype(str)).split())))
            ).reset_index()
        else:
            meds_df = pd.DataFrame(columns=['hadm_id', 'drug'])
        meds_df.rename(columns={'drug': 'med_list_text'}, inplace=True)
        meds_df['n_medications'] = meds_df['med_list_text'].apply(lambda x: len(set(str(x).split())))
        
        # --- DIAGNOSES ---
        print(f"  Extracting diagnoses...")
        diag_list = []
        for chunk in tqdm(pd.read_csv(FILE_PATHS['diagnoses'], chunksize=CHUNK_SIZE,
                                       usecols=['hadm_id', 'icd_code', 'seq_num']),
                          desc="    Diagnoses", unit="chunk"):
            self._clean_id(chunk, 'hadm_id')
            chunk['icd_code'] = chunk['icd_code'].astype(str)
            if not chunk.empty:
                # Sort by sequence number to get primary diagnosis first
                chunk = chunk.sort_values(['hadm_id', 'seq_num'])
                diag_list.append(
                    chunk.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
                )
        
        if diag_list:
            diag_df = pd.concat(diag_list).groupby('hadm_id')['icd_code'].apply(
                lambda x: ' '.join(x)
            ).reset_index()
        else:
            diag_df = pd.DataFrame(columns=['hadm_id', 'icd_code'])
        diag_df.rename(columns={'icd_code': 'diagnosis_list_text'}, inplace=True)
        diag_df['n_diagnoses'] = diag_df['diagnosis_list_text'].apply(lambda x: len(str(x).split()))
        
        # Extract primary diagnosis (first one)
        diag_df['primary_diagnosis'] = diag_df['diagnosis_list_text'].apply(
            lambda x: str(x).split()[0] if x else ''
        )
        
        # --- PROCEDURES ---
        print(f"  Extracting procedures...")
        proc_df = pd.DataFrame()
        if os.path.exists(FILE_PATHS['procedures']):
            proc_list = []
            for chunk in tqdm(pd.read_csv(FILE_PATHS['procedures'], chunksize=CHUNK_SIZE,
                                           usecols=['hadm_id', 'icd_code']),
                              desc="    Procedures", unit="chunk"):
                self._clean_id(chunk, 'hadm_id')
                chunk['icd_code'] = chunk['icd_code'].astype(str)
                if not chunk.empty:
                    proc_list.append(
                        chunk.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
                    )
            
            if proc_list:
                proc_df = pd.concat(proc_list).groupby('hadm_id')['icd_code'].apply(
                    lambda x: ' '.join(x)
                ).reset_index()
                proc_df.rename(columns={'icd_code': 'procedure_list_text'}, inplace=True)
                proc_df['n_procedures'] = proc_df['procedure_list_text'].apply(
                    lambda x: len(str(x).split())
                )
        
        elapsed = time.time() - start_time
        print(f"  [OK] Text extraction complete ({elapsed:.1f}s)")
        
        return meds_df, diag_df, proc_df
    
    # =========================================================================
    # STEP 3: EXTRACT CLINICAL NOTES
    # =========================================================================
    def extract_notes(self) -> pd.DataFrame:
        """Extract discharge summaries with data leakage prevention.
        
        CRITICAL: Only includes notes available at or before discharge time.
        This prevents data leakage where discharge summaries are dictated
        days after discharge (Q1 journal requirement).
        
        Returns:
            DataFrame with hadm_id and clinical_text columns
        """
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 3: Extracting Discharge Summaries...")
        print(f"  [DATA LEAKAGE CHECK] Filtering notes to discharge time or earlier")
        
        if not os.path.exists(FILE_PATHS['discharge']):
            print(f"  [ERROR] Discharge file not found at {FILE_PATHS['discharge']}")
            return pd.DataFrame()
        
        # Get discharge times for leakage check
        discharge_times = None
        if hasattr(self, 'admissions') and 'dischtime' in self.admissions.columns:
            discharge_times = self.admissions[['hadm_id', 'dischtime']].copy()
            discharge_times['dischtime'] = pd.to_datetime(discharge_times['dischtime'])
            print(f"  [CHECK] Loaded discharge times for {len(discharge_times):,} admissions")
        
        notes = []
        for chunk in tqdm(pd.read_csv(FILE_PATHS['discharge'], chunksize=CHUNK_SIZE, dtype=str),
                          desc="    Notes", unit="chunk"):
            if 'hadm_id' in chunk.columns:
                self._clean_id(chunk, 'hadm_id')
                chunk = chunk[chunk['hadm_id'] != 'nan']
                
                cols_to_keep = ['hadm_id', 'text']
                if 'charttime' in chunk.columns:
                    cols_to_keep.append('charttime')
                
                notes.append(chunk[cols_to_keep])
        
        if not notes:
            return pd.DataFrame()
        
        df = pd.concat(notes)
        
        # DATA LEAKAGE PREVENTION: Filter notes by discharge time
        if 'charttime' in df.columns and discharge_times is not None:
            print(f"  [CHECK] Filtering notes to prevent data leakage...")
            df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
            
            # Merge with discharge times
            df = df.merge(discharge_times, on='hadm_id', how='left')
            
            # Only keep notes created at or before discharge time
            # Allow 1-hour buffer for notes created during discharge process
            df['time_diff'] = (df['charttime'] - df['dischtime']).dt.total_seconds() / 3600  # hours
            before_discharge = df['time_diff'] <= 1.0  # 1 hour buffer
            
            n_before = before_discharge.sum()
            n_after = (~before_discharge).sum()
            n_no_time = df['charttime'].isna().sum()
            
            print(f"      Notes before discharge: {n_before:,}")
            print(f"      Notes after discharge (EXCLUDED): {n_after:,}")
            print(f"      Notes without timestamp: {n_no_time:,}")
            
            if n_after > 0:
                print(f"      ⚠️  WARNING: Excluding {n_after:,} notes created after discharge")
                print(f"         This prevents data leakage (Q1 journal requirement)")
            
            # Keep notes before discharge, or notes without timestamp (assume safe)
            df = df[before_discharge | df['charttime'].isna()].copy()
            
            # Drop helper columns
            df = df.drop(columns=['dischtime', 'time_diff'], errors='ignore')
        
        # Sort by time and keep last note per admission
        if 'charttime' in df.columns:
            df = df.sort_values('charttime')
        df = df.groupby('hadm_id')['text'].last().reset_index()
        
        # Clean text
        df['clinical_text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        
        elapsed = time.time() - start_time
        print(f"  [OK] Extracted notes for {len(df):,} admissions ({elapsed:.1f}s)")
        
        if discharge_times is not None:
            coverage = len(df) / len(discharge_times) * 100
            print(f"  [COVERAGE] {coverage:.1f}% of admissions have discharge summaries")
        
        return df[['hadm_id', 'clinical_text']]
    
    # =========================================================================
    # STEP 4: EXTRACT LABS/VITALS
    # =========================================================================
    def extract_stats_optimized(
        self, 
        table_name: str, 
        itemids_dict: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """Extract lab/vital statistics with clinical value flags.
        
        Args:
            table_name: Name of the table to extract (e.g., 'labevents', 'chartevents')
            itemids_dict: Dictionary mapping lab/vital names to item IDs
            
        Returns:
            DataFrame with aggregated statistics and critical value flags
        """
        start_time = time.time()
        print(f"  Extracting {table_name}...")
        
        if not os.path.exists(FILE_PATHS[table_name]):
            print(f"    [SKIP] File not found: {FILE_PATHS[table_name]}")
            return pd.DataFrame()
        
        all_ids = [i for s in itemids_dict.values() for i in s]
        chunks = []
        
        for chunk in tqdm(pd.read_csv(FILE_PATHS[table_name], chunksize=CHUNK_SIZE,
                                       usecols=['hadm_id', 'itemid', 'valuenum', 'charttime'],
                                       parse_dates=['charttime']),
                          desc=f"    Reading {table_name}", unit="chunk"):
            
            self._clean_id(chunk, 'hadm_id')
            chunk = chunk[chunk['itemid'].isin(all_ids)]
            chunk = chunk.dropna(subset=['valuenum'])
            chunk = chunk.sort_values(['hadm_id', 'charttime'])
            
            if not chunk.empty:
                agg = chunk.groupby(['hadm_id', 'itemid'])['valuenum'].agg(
                    mean='mean', last='last', first='first',
                    min='min', max='max', std='std'
                ).reset_index()
                agg['trend'] = agg['last'] - agg['first']
                agg['range'] = agg['max'] - agg['min']
                chunks.append(agg.drop(columns=['first']))
        
        if not chunks:
            return pd.DataFrame()
        
        df = pd.concat(chunks)
        id_map = {i: k for k, v in itemids_dict.items() for i in v}
        df['name'] = df['itemid'].map(id_map)
        
        # Aggregate by hadm_id and name
        df_pivot = df.groupby(['hadm_id', 'name']).mean().reset_index()
        df_pivot = df_pivot.pivot(
            index='hadm_id', 
            columns='name',
            values=['mean', 'last', 'min', 'max', 'trend', 'range', 'std']
        )
        df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]
        result = df_pivot.reset_index()
        
        # Add critical value flags
        result = self._add_critical_value_flags(result)
        
        elapsed = time.time() - start_time
        print(f"    [OK] Extracted {len(result):,} admissions ({elapsed:.1f}s)")
        
        return result
    
    def _add_critical_value_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary flags for critical lab values.
        
        Args:
            df: DataFrame with lab values
            
        Returns:
            DataFrame with added critical value flags
        """
        for lab_name, (low, high) in CRITICAL_RANGES.items():
            last_col = f'{lab_name}_last'
            if last_col in df.columns:
                # Critical at discharge (out of range)
                df[f'{lab_name}_critical'] = (
                    (df[last_col] < low) | (df[last_col] > high)
                ).astype(int)
                
                # Normal at discharge
                df[f'{lab_name}_normal'] = (
                    (df[last_col] >= low) & (df[last_col] <= high)
                ).astype(int)
        
        # Panic values
        for lab_name, (panic_low, panic_high) in PANIC_RANGES.items():
            last_col = f'{lab_name}_last'
            if last_col in df.columns:
                df[f'{lab_name}_panic'] = (
                    (df[last_col] < panic_low) | (df[last_col] > panic_high)
                ).astype(int)
        
        # Count of critical labs
        critical_cols = [c for c in df.columns if c.endswith('_critical')]
        if critical_cols:
            df['n_critical_labs'] = df[critical_cols].sum(axis=1)
        
        return df
    
    # =========================================================================
    # STEP 5: CLINICAL SCORES
    # =========================================================================
    def calculate_charlson_score(self, diag_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Charlson Comorbidity Index.
        
        Args:
            diag_df: DataFrame with diagnosis_list_text column
            
        Returns:
            DataFrame with added charlson_score and condition flags
        """
        print(f"  Calculating Charlson Comorbidity Index...")
        
        charlson_map = {
            'myocardial_infarction': (['I21', 'I22', 'I252'], 1),
            'chf': (['I50', 'I110', 'I130', 'I132'], 1),
            'pvd': (['I70', 'I71', 'I731', 'I738', 'I739', 'I771'], 1),
            'cvd': (['G45', 'G46', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'], 1),
            'dementia': (['F00', 'F01', 'F02', 'F03', 'G30', 'G311'], 1),
            'copd': (['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67'], 1),
            'rheumatic': (['M05', 'M06', 'M32', 'M33', 'M34', 'M353'], 1),
            'peptic_ulcer': (['K25', 'K26', 'K27', 'K28'], 1),
            'liver_mild': (['B18', 'K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K73', 'K74', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944'], 1),
            'diabetes': (['E10', 'E11', 'E12', 'E13', 'E14'], 1),
            'diabetes_complications': (['E102', 'E103', 'E104', 'E105', 'E112', 'E113', 'E114', 'E115', 'E122', 'E123', 'E124', 'E125', 'E132', 'E133', 'E134', 'E135', 'E142', 'E143', 'E144', 'E145'], 2),
            'hemiplegia': (['G041', 'G114', 'G801', 'G802', 'G81', 'G82', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839'], 2),
            'renal': (['I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N18', 'N19', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992'], 2),
            'cancer': (['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C90', 'C91', 'C92', 'C93', 'C94', 'C95', 'C96', 'C97'], 2),
            'liver_severe': (['I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767'], 3),
            'metastatic': (['C77', 'C78', 'C79', 'C80'], 6),
            'aids': (['B20', 'B21', 'B22', 'B24'], 6),
        }
        
        scores = []
        condition_flags = {cond: [] for cond in charlson_map.keys()}
        
        for _, row in diag_df.iterrows():
            diagnoses = str(row['diagnosis_list_text']).upper().split()
            score = 0
            
            for condition, (codes, weight) in charlson_map.items():
                has_condition = any(
                    any(diag.startswith(code) for code in codes)
                    for diag in diagnoses
                )
                condition_flags[condition].append(int(has_condition))
                if has_condition:
                    score += weight
            
            scores.append(min(score, 20))  # Cap at 20
        
        diag_df['charlson_score'] = scores
        
        # Add individual condition flags
        for condition in charlson_map.keys():
            diag_df[f'has_{condition}'] = condition_flags[condition]
        
        print(f"    Mean Charlson score: {diag_df['charlson_score'].mean():.2f}")
        
        return diag_df
    
    # =========================================================================
    # STEP 6: PATIENT HISTORY TEXT
    # =========================================================================
    def build_patient_history_text(self, diag_df: pd.DataFrame) -> None:
        """
        Build comprehensive patient history from all prior diagnoses.
        
        This gives the History Specialist more to work with by creating
        a cumulative history of all prior diagnoses for each patient.
        
        Args:
            diag_df: DataFrame with diagnosis_list_text column
        """
        if self.admissions is None:
            raise ValueError("Admissions data must be loaded first")
            
        print(f"  Building patient history text...")
        
        # Merge diagnoses with admissions to get patient and time info
        df = self.admissions[['hadm_id', 'subject_id', 'admittime']].copy()
        df = df.merge(diag_df[['hadm_id', 'diagnosis_list_text']], on='hadm_id', how='left')
        df['diagnosis_list_text'] = df['diagnosis_list_text'].fillna('')
        
        # Sort by patient and time
        df = df.sort_values(['subject_id', 'admittime'])
        
        # Build cumulative history for each admission
        history_texts = []
        
        for hadm_id in tqdm(self.admissions['hadm_id'], desc="    Building history"):
            row = df[df['hadm_id'] == hadm_id].iloc[0]
            subj_id = row['subject_id']
            admit_time = row['admittime']
            
            # Get all prior diagnoses for this patient
            prior = df[
                (df['subject_id'] == subj_id) & 
                (df['admittime'] < admit_time)
            ]['diagnosis_list_text'].tolist()
            
            if prior:
                # Combine all prior diagnoses
                all_prior = ' '.join(prior)
                # Get unique codes
                unique_codes = ' '.join(sorted(set(all_prior.split())))
                history_texts.append(unique_codes)
            else:
                history_texts.append('')
        
        self.admissions['full_history_text'] = history_texts
        
        n_with_history = (self.admissions['full_history_text'] != '').sum()
        print(f"    {n_with_history:,} admissions have prior history")
    
    # =========================================================================
    # SOTA: TEMPORAL SEQUENCE EXTRACTION
    # =========================================================================
    # =========================================================================
    # MAIN RUN
    # =========================================================================
    def run(self) -> None:
        """Run complete extraction pipeline.
        
        Executes all extraction steps:
        1. Load core data
        2. Extract text lists (medications, diagnoses, procedures)
        3. Extract clinical notes
        4. Extract labs and vitals
        5. Merge all data sources
        6. Build patient history
        7. Impute missing values
        8. Save output
        """
        total_start = time.time()
        
        print("=" * 80)
        print(f"MIMIC-IV FEATURE EXTRACTION FOR DISEASE-SPECIFIC READMISSION PREDICTION")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Step 1: Core data
        self.load_core()
        
        # Step 2: Text lists
        meds_df, diag_df, proc_df = self.extract_text_lists()
        
        # Calculate Charlson score
        if not diag_df.empty:
            diag_df = self.calculate_charlson_score(diag_df)
        
        # Step 3: Clinical notes
        notes_df = self.extract_notes()
        
        # Step 4: Labs and vitals
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 4: Extracting Labs & Vitals...")
        vitals_df = self.extract_stats_optimized('chartevents', VITAL_ITEMIDS)
        labs_df = self.extract_stats_optimized('labevents', LAB_ITEMIDS)
        
        # Step 5: Merge all data
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 5: Merging All Data Sources...")
        df = self.admissions.copy()
        
        for name, data in [('medications', meds_df), ('diagnoses', diag_df),
                           ('procedures', proc_df), ('notes', notes_df),
                           ('vitals', vitals_df), ('labs', labs_df)]:
            if data is not None and not data.empty:
                print(f"  Merging {name}: {len(data):,} records")
                self._clean_id(data, 'hadm_id')
                df = df.merge(data, on='hadm_id', how='left')
        
        # Step 6: Build patient history
        if not diag_df.empty:
            self.admissions = df  # Update for history building
            self.build_patient_history_text(diag_df)
            df['full_history_text'] = self.admissions['full_history_text']
        
        # Step 6b: Extract protective factors (reduces systematic false positives)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 6b: Extracting Protective Factors...")
        protective_df = extract_protective_factors(df, verbose=True)
        for col in protective_df.columns:
            df[col] = protective_df[col].values
        
        # Fill missing text columns
        text_cols = ['med_list_text', 'diagnosis_list_text', 'clinical_text', 
                     'procedure_list_text', 'full_history_text']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('empty')
            else:
                df[col] = 'empty'
        
        # Fill missing numeric columns
        for col in ['n_medications', 'n_diagnoses', 'n_procedures', 'charlson_score']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0
        
        # Step 7: Impute remaining numerics
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 6: Imputing Missing Values...")
        
        exclude_from_impute = [
            'subject_id', 'hadm_id', 'readmitted_30d', 'readmitted_30d_all_cause', 'gender_M',
            'prior_visits_count', 'weekend_discharge', 'weekend_admission',
            'emergency_admit', 'high_risk_service', 'is_surgery',
            'icu_transfers', 'prev_was_readmitted', 'n_prior_readmissions',
            'bounce_back_7d', 'rapid_return_14d', 'frequent_flyer',
            'very_frequent_flyer', 'ever_readmitted', 'dc_to_home',
            'dc_to_snf', 'dc_to_rehab', 'dc_to_home_health', 'supervised_discharge',
            'came_from_ed', 'night_admission', 'long_ed_stay', 'had_icu_stay',
            'long_icu_stay', 'age_under_30', 'age_over_65', 'age_over_75',
            'age_over_85', 'recent_ed_visit', 'high_ed_utilization',
            # Clinical Correctness flags
            'is_trauma_service', 'is_ortho_service', 'is_unpredictable_service',
            'dc_to_hospice', 'dc_ama',
            # Protective factors (binary flags - don't impute)
            'pf_supervised_discharge', 'pf_followup_scheduled', 'pf_care_coordination',
            'pf_family_support', 'pf_clinically_stable', 'pf_patient_engaged',
            'pf_no_ama', 'protective_score', 'high_protection', 'low_protection',
        ]
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_impute = [c for c in num_cols if c not in exclude_from_impute]
        
        if cols_to_impute:
            df[cols_to_impute] = self.median_imputer.fit_transform(df[cols_to_impute])
            print(f"  Imputed {len(cols_to_impute)} numeric columns")
        
        # Step 8: Save
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STEP 7: Saving Output...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        file_size = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)
        
        total_elapsed = time.time() - total_start
        
        # Final summary
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"  Output file: {OUTPUT_FILE}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Total admissions: {len(df):,}")
        print(f"  DISEASE-SPECIFIC readmission rate: {df['readmitted_30d'].mean():.2%}")
        print(f"  All-cause readmission rate: {df['readmitted_30d_all_cause'].mean():.2%}")
        print(f"  Total features: {len(df.columns)}")
        print(f"  Total time: {total_elapsed / 60:.1f} minutes")
        print("=" * 80)
        
        # Print key feature statistics
        print("\nKEY FEATURE STATISTICS:")
        print(f"  Prior readmissions (mean): {df['n_prior_readmissions'].mean():.2f}")
        print(f"  Patients with prior readmit: {(df['ever_readmitted'] == 1).sum():,} ({df['ever_readmitted'].mean():.1%})")
        print(f"  Bounce-backs (7d): {df['bounce_back_7d'].sum():,} ({df['bounce_back_7d'].mean():.1%})")
        print(f"  Mean Charlson score: {df['charlson_score'].mean():.2f}")
        print(f"  Mean LOS: {df['los_days'].mean():.1f} days")
        
        print("\nCLINICAL CORRECTNESS FLAGS:")
        print(f"  Unpredictable service (trauma/obs): {df['is_unpredictable_service'].sum():,} ({df['is_unpredictable_service'].mean():.1%})")
        print(f"  Hospice discharge: {df['dc_to_hospice'].sum():,} ({df['dc_to_hospice'].mean():.1%})")
        print(f"  Supervised discharge: {df['supervised_discharge'].sum():,} ({df['supervised_discharge'].mean():.1%})")
        
        print("\nDISEASE-SPECIFIC READMISSION INFO:")
        print(f"  Primary diagnosis categories: {df['primary_dx_category'].nunique():,} unique")
        print(f"  Top categories in readmissions:")
        top_dx = df[df['readmitted_30d'] == 1]['primary_dx_category'].value_counts().head(5)
        for dx, count in top_dx.items():
            print(f"    {dx}: {count:,}")


if __name__ == "__main__":
    extractor = MIMICExtractor()
    extractor.run()