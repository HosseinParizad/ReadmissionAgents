import pandas as pd
import numpy as np
import os
import re
import warnings
import time
from datetime import datetime
from sklearn.impute import SimpleImputer 

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_PATH = 'D:/Parizad/PHD/Project/Data/mimic-iv-2.2/'
OUTPUT_FILE = './model_outputs/features_4_agents.csv'
CHUNK_SIZE = 500000 

FILE_PATHS = {
    'admissions': os.path.join(BASE_PATH, 'hosp/admissions.csv'),
    'patients': os.path.join(BASE_PATH, 'hosp/patients.csv'),
    'labevents': os.path.join(BASE_PATH, 'hosp/labevents.csv'),
    'chartevents': os.path.join(BASE_PATH, 'icu/chartevents.csv'),
    'noteevents': os.path.join(BASE_PATH, 'note/noteevents.csv'),
    'prescriptions': os.path.join(BASE_PATH, 'hosp/prescriptions.csv'),
    'diagnoses': os.path.join(BASE_PATH, 'hosp/diagnoses_icd.csv'),
    # NEW FILES
    'procedures': os.path.join(BASE_PATH, 'hosp/procedures_icd.csv'),
    'services': os.path.join(BASE_PATH, 'hosp/services.csv')
}

# Optimized Labs/Vitals (Keep this strict to save RAM)
VITAL_ITEMIDS = {
    'hr': [220045], 'sbp': [220179, 220050], 'dbp': [220180, 220051], 
    'resp_rate': [220210, 224690], 'spo2': [220277], 'temp': [223761, 223762]
}
LAB_ITEMIDS = {
    'wbc': [51300, 51301], 'creat': [50912], 'gluc': [50931, 50809], 
    'hgb': [51222, 51221], 'plt': [51265], 'na': [50983], 'k': [50971], 'bnp': [50963]
}

class MIMICExtractor:
    def __init__(self):
        self.admissions = None
        self.median_imputer = SimpleImputer(strategy='median')

    def load_core(self):
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 1: Loading Core Context & Services...")
        print(f"  Reading admissions from: {FILE_PATHS['admissions']}")
        self.admissions = pd.read_csv(FILE_PATHS['admissions'], 
                                      parse_dates=['admittime', 'dischtime', 'deathtime'])
        print(f"  Loaded {len(self.admissions):,} admissions")
        
        print(f"  Reading patients from: {FILE_PATHS['patients']}")
        patients = pd.read_csv(FILE_PATHS['patients'])
        print(f"  Loaded {len(patients):,} patients")
        
        print("  Merging admissions with patient data...")
        self.admissions = self.admissions.merge(patients[['subject_id', 'anchor_age', 'gender']], on='subject_id')
        
        # Load Services (NEW: What department was the patient in?)
        if os.path.exists(FILE_PATHS['services']):
            print(f"  Loading Services from: {FILE_PATHS['services']}")
            services = pd.read_csv(FILE_PATHS['services'])
            print(f"  Loaded {len(services):,} service records")
            # Keep the LAST service (discharge service)
            services = services.sort_values('transfertime').groupby('hadm_id')['curr_service'].last().reset_index()
            self.admissions = self.admissions.merge(services, on='hadm_id', how='left')
            self.admissions['curr_service'] = self.admissions['curr_service'].fillna("UNKNOWN")
            print(f"  Merged services: {self.admissions['curr_service'].value_counts().sum():,} admissions have service info")
        else:
            print(f"  Warning: Services file not found at {FILE_PATHS['services']}")
            self.admissions['curr_service'] = "UNKNOWN"

        print("  Calculating readmission targets...")
        # Target Calculation
        self.admissions = self.admissions.sort_values(['subject_id', 'admittime'])
        self.admissions['next_admittime'] = self.admissions.groupby('subject_id')['admittime'].shift(-1)
        self.admissions['days_to_next'] = (self.admissions['next_admittime'] - self.admissions['dischtime']).dt.total_seconds() / 86400
        self.admissions['readmitted_30d'] = ((self.admissions['days_to_next'] <= 30) & (self.admissions['days_to_next'] >= 0)).astype(int)
        
        before_death_filter = len(self.admissions)
        self.admissions = self.admissions[self.admissions['deathtime'].isna()]
        print(f"  Filtered out {before_death_filter - len(self.admissions):,} admissions with death records")
        
        # Basic Context Features
        print("  Computing context features...")
        self.admissions['prior_visits_count'] = self.admissions.groupby('subject_id').cumcount()
        self.admissions['los_days'] = (self.admissions['dischtime'] - self.admissions['admittime']).dt.total_seconds() / 86400
        self.admissions['gender_M'] = (self.admissions['gender'] == 'M').astype(int)
        
        readmitted_count = self.admissions['readmitted_30d'].sum()
        readmission_rate = (readmitted_count / len(self.admissions)) * 100
        elapsed = time.time() - start_time
        print(f"  [OK] Core Data: {len(self.admissions):,} admissions ({readmitted_count:,} readmissions, {readmission_rate:.2f}%)")
        print(f"  Time elapsed: {elapsed:.2f} seconds")

    def extract_stats_optimized(self, table_name, itemids_dict):
        start_time = time.time()
        print(f"  Extracting {table_name}...")
        if not os.path.exists(FILE_PATHS[table_name]): 
            print(f"    Warning: File not found at {FILE_PATHS[table_name]}")
            return pd.DataFrame()
        
        file_size = os.path.getsize(FILE_PATHS[table_name]) / (1024**3)  # GB
        print(f"    File size: {file_size:.2f} GB")
        print(f"    Reading in chunks of {CHUNK_SIZE:,} rows...")
        
        all_ids = [i for s in itemids_dict.values() for i in s]
        print(f"    Filtering for {len(all_ids)} item IDs")
        chunks = []
        chunk_count = 0
        total_rows = 0
        filtered_rows = 0
        
        for chunk in pd.read_csv(FILE_PATHS[table_name], chunksize=CHUNK_SIZE, 
                                 usecols=['hadm_id', 'itemid', 'valuenum', 'charttime'], parse_dates=['charttime']):
            chunk_count += 1
            total_rows += len(chunk)
            if chunk_count % 10 == 0:
                print(f"    Processed {chunk_count} chunks ({total_rows:,} rows, {filtered_rows:,} matched)...")
            
            chunk = chunk[chunk['itemid'].isin(all_ids)].sort_values(['hadm_id', 'charttime'])
            filtered_rows += len(chunk)
            if not chunk.empty:
                # Key Optimization: Only Mean, Last, Trend
                agg = chunk.groupby(['hadm_id', 'itemid'])['valuenum'].agg(
                    mean='mean', last='last', first='first'
                ).reset_index()
                agg['trend'] = agg['last'] - agg['first']
                chunks.append(agg.drop(columns=['first']))
        
        print(f"    Total: {chunk_count} chunks, {total_rows:,} rows read, {filtered_rows:,} rows matched")
        
        if not chunks: 
            print(f"    No matching data found")
            return pd.DataFrame()
        
        print(f"    Aggregating {len(chunks)} chunk results...")
        df = pd.concat(chunks)
        print(f"    Aggregated to {len(df):,} hadm_id-itemid pairs")
        
        print(f"    Mapping item IDs to names...")
        id_map = {i: k for k, v in itemids_dict.items() for i in v}
        df['name'] = df['itemid'].map(id_map)
        
        print(f"    Pivoting data...")
        df_pivot = df.groupby(['hadm_id', 'name']).mean().reset_index()
        df_pivot = df_pivot.pivot(index='hadm_id', columns='name', values=['mean', 'last', 'trend'])
        df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]
        result = df_pivot.reset_index()
        
        elapsed = time.time() - start_time
        print(f"    [OK] Extracted {len(result):,} admissions with {len(result.columns)-1} features in {elapsed:.2f} seconds")
        return result

    def extract_text_lists(self):
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 2: Extracting Text Lists (Meds, Hx, & Procedures)...")
        
        # 1. Meds
        med_list = []
        if os.path.exists(FILE_PATHS['prescriptions']):
            print(f"  Extracting medications from: {FILE_PATHS['prescriptions']}")
            chunk_count = 0
            for chunk in pd.read_csv(FILE_PATHS['prescriptions'], chunksize=CHUNK_SIZE, usecols=['hadm_id', 'drug']):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"    Processed {chunk_count} chunks...")
                chunk['drug'] = chunk['drug'].astype(str).str.lower()
                med_list.append(chunk.groupby('hadm_id')['drug'].apply(lambda x: ' '.join(set(x))).reset_index())
            print(f"    Aggregating {len(med_list)} chunks...")
            meds_df = pd.concat(med_list).groupby('hadm_id')['drug'].apply(lambda x: ' '.join(x)).reset_index()
            meds_df.rename(columns={'drug': 'med_list_text'}, inplace=True)
            print(f"    [OK] Extracted medications for {len(meds_df):,} admissions")
        else: 
            print(f"    Warning: Prescriptions file not found at {FILE_PATHS['prescriptions']}")
            meds_df = pd.DataFrame()

        # 2. Diagnoses
        diag_list = []
        if os.path.exists(FILE_PATHS['diagnoses']):
            print(f"  Extracting diagnoses from: {FILE_PATHS['diagnoses']}")
            chunk_count = 0
            for chunk in pd.read_csv(FILE_PATHS['diagnoses'], chunksize=CHUNK_SIZE, usecols=['hadm_id', 'icd_code']):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"    Processed {chunk_count} chunks...")
                chunk['icd_code'] = chunk['icd_code'].astype(str)
                diag_list.append(chunk.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index())
            print(f"    Aggregating {len(diag_list)} chunks...")
            diag_df = pd.concat(diag_list).groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
            diag_df.rename(columns={'icd_code': 'diagnosis_list_text'}, inplace=True)
            print(f"    [OK] Extracted diagnoses for {len(diag_df):,} admissions")
        else: 
            print(f"    Warning: Diagnoses file not found at {FILE_PATHS['diagnoses']}")
            diag_df = pd.DataFrame()

        # 3. Procedures (NEW)
        proc_list = []
        if os.path.exists(FILE_PATHS['procedures']):
            print(f"  Extracting procedures from: {FILE_PATHS['procedures']}")
            chunk_count = 0
            for chunk in pd.read_csv(FILE_PATHS['procedures'], chunksize=CHUNK_SIZE, usecols=['hadm_id', 'icd_code']):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"    Processed {chunk_count} chunks...")
                chunk['icd_code'] = chunk['icd_code'].astype(str)
                proc_list.append(chunk.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index())
            if proc_list:
                print(f"    Aggregating {len(proc_list)} chunks...")
                proc_df = pd.concat(proc_list).groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
                proc_df.rename(columns={'icd_code': 'procedure_list_text'}, inplace=True)
                print(f"    [OK] Extracted procedures for {len(proc_df):,} admissions")
            else: proc_df = pd.DataFrame()
        else: 
            print(f"    Warning: Procedures file not found at {FILE_PATHS['procedures']}")
            proc_df = pd.DataFrame()
        
        elapsed = time.time() - start_time
        print(f"  [OK] Text extraction completed in {elapsed:.2f} seconds")
        return meds_df, diag_df, proc_df

    def extract_notes(self):
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 3: Extracting Discharge Summaries...")
        if not os.path.exists(FILE_PATHS['noteevents']): 
            print(f"  Warning: Notes file not found at {FILE_PATHS['noteevents']}")
            return pd.DataFrame()
        
        file_size = os.path.getsize(FILE_PATHS['noteevents']) / (1024**3)  # GB
        print(f"  File size: {file_size:.2f} GB")
        print(f"  Reading in chunks of {CHUNK_SIZE:,} rows, filtering for 'Discharge summary'...")
            
        notes = []
        chunk_count = 0
        total_rows = 0
        discharge_rows = 0
        # Use 'str' dtype to avoid mixed type warnings on large files
        for chunk in pd.read_csv(FILE_PATHS['noteevents'], chunksize=CHUNK_SIZE, dtype=str):
            chunk_count += 1
            total_rows += len(chunk)
            if chunk_count % 10 == 0:
                print(f"    Processed {chunk_count} chunks ({total_rows:,} rows, {discharge_rows:,} discharge summaries)...")
            # Filter for Discharge Summaries
            chunk = chunk[chunk['category'] == 'Discharge summary']
            discharge_rows += len(chunk)
            if not chunk.empty: 
                notes.append(chunk[['hadm_id', 'text', 'charttime']])
        
        print(f"    Total: {chunk_count} chunks, {total_rows:,} rows read, {discharge_rows:,} discharge summaries found")
        
        if not notes: 
            print(f"    No discharge summaries found")
            return pd.DataFrame()
        
        print(f"    Concatenating {len(notes)} note chunks...")
        df = pd.concat(notes).sort_values('charttime')
        print(f"    Keeping last note per admission from {len(df):,} notes...")
        # Keep last note per admission
        df = df.groupby('hadm_id')['text'].last().reset_index()
        print(f"    Cleaning text (removing extra whitespace)...")
        # Basic cleaning
        df['clinical_text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        
        elapsed = time.time() - start_time
        print(f"  [OK] Extracted notes for {len(df):,} admissions in {elapsed:.2f} seconds")
        return df[['hadm_id', 'clinical_text']]

    def run(self):
        total_start = time.time()
        print("=" * 80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting MIMIC-IV Data Extraction")
        print("=" * 80)
        
        self.load_core()
        
        # Extract
        meds_df, diag_df, proc_df = self.extract_text_lists()
        notes_df = self.extract_notes()
        vitals_df = self.extract_stats_optimized('chartevents', VITAL_ITEMIDS)
        labs_df = self.extract_stats_optimized('labevents', LAB_ITEMIDS)
        
        # Merge
        merge_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 4: Merging all data sources...")
        df = self.admissions
        merge_count = 0
        for name, d in [('medications', meds_df), ('diagnoses', diag_df), ('procedures', proc_df), 
                        ('notes', notes_df), ('vitals', vitals_df), ('labs', labs_df)]:
            if not d.empty:
                merge_count += 1
                print(f"  Merging {name} ({len(d):,} records)...")
                # Ensure ID key is correct type
                if 'hadm_id' in d.columns:
                    d['hadm_id'] = pd.to_numeric(d['hadm_id'], errors='coerce')
                before_merge = len(df)
                df = df.merge(d, on='hadm_id', how='left')
                matched = len(df[df[d.columns[-1]].notna()]) if len(d.columns) > 1 else len(df)
                print(f"    Matched {matched:,} admissions (kept {len(df):,} total)")
            else:
                print(f"  Skipping {name} (empty dataframe)")
        
        print(f"  [OK] Merged {merge_count} data sources, final dataset: {len(df):,} admissions, {len(df.columns)} columns")
        print(f"  Merge time: {time.time() - merge_start:.2f} seconds")
        
        # Fill Text NaNs (Critical for Specialists)
        print("  Filling missing text fields...")
        for col in ['med_list_text', 'diagnosis_list_text', 'clinical_text', 'procedure_list_text']:
            if col in df.columns: 
                missing = df[col].isna().sum()
                df[col] = df[col].fillna("empty")
                if missing > 0:
                    print(f"    Filled {missing:,} missing values in {col}")
            else: 
                df[col] = "empty"
                print(f"    Created {col} column (all empty)")
            
        # Impute numeric (Critical for Structure Specialist)
        impute_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STEP 5: Imputing Numerics...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        # Don't impute IDs or Targets
        cols_to_impute = [c for c in num_cols if c not in ['subject_id', 'hadm_id', 'readmitted_30d', 'gender_M', 'prior_visits_count']]
        if cols_to_impute:
            missing_counts = df[cols_to_impute].isna().sum()
            total_missing = missing_counts.sum()
            print(f"  Imputing {len(cols_to_impute)} numeric columns ({total_missing:,} missing values total)...")
            df[cols_to_impute] = self.median_imputer.fit_transform(df[cols_to_impute])
            print(f"  [OK] Imputation completed")
        else:
            print(f"  No numeric columns to impute")
        print(f"  Imputation time: {time.time() - impute_start:.2f} seconds")
        
        # Save
        save_start = time.time()
        if not os.path.exists('./model_outputs'):
            os.makedirs('./model_outputs')
            print(f"  Created model_outputs directory")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving to {OUTPUT_FILE}...")
        print(f"  Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        df.to_csv(OUTPUT_FILE, index=False)
        file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)  # MB
        print(f"  [OK] Saved {file_size:.2f} MB to {OUTPUT_FILE}")
        print(f"  Save time: {time.time() - save_start:.2f} seconds")
        
        total_elapsed = time.time() - total_start
        print("=" * 80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extraction Complete!")
        print(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        print("=" * 80)

if __name__ == "__main__":
    MIMICExtractor().run()