import pandas as pd
import numpy as np
import os
import re
import warnings
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
        print("STEP 1: Loading Core Context & Services...")
        self.admissions = pd.read_csv(FILE_PATHS['admissions'], 
                                      parse_dates=['admittime', 'dischtime', 'deathtime'])
        patients = pd.read_csv(FILE_PATHS['patients'])
        self.admissions = self.admissions.merge(patients[['subject_id', 'anchor_age', 'gender']], on='subject_id')
        
        # Load Services (NEW: What department was the patient in?)
        if os.path.exists(FILE_PATHS['services']):
            print("  Loading Services...")
            services = pd.read_csv(FILE_PATHS['services'])
            # Keep the LAST service (discharge service)
            services = services.sort_values('transfertime').groupby('hadm_id')['curr_service'].last().reset_index()
            self.admissions = self.admissions.merge(services, on='hadm_id', how='left')
            self.admissions['curr_service'] = self.admissions['curr_service'].fillna("UNKNOWN")
        else:
            self.admissions['curr_service'] = "UNKNOWN"

        # Target Calculation
        self.admissions = self.admissions.sort_values(['subject_id', 'admittime'])
        self.admissions['next_admittime'] = self.admissions.groupby('subject_id')['admittime'].shift(-1)
        self.admissions['days_to_next'] = (self.admissions['next_admittime'] - self.admissions['dischtime']).dt.total_seconds() / 86400
        self.admissions['readmitted_30d'] = ((self.admissions['days_to_next'] <= 30) & (self.admissions['days_to_next'] >= 0)).astype(int)
        self.admissions = self.admissions[self.admissions['deathtime'].isna()]
        
        # Basic Context Features
        self.admissions['prior_visits_count'] = self.admissions.groupby('subject_id').cumcount()
        self.admissions['los_days'] = (self.admissions['dischtime'] - self.admissions['admittime']).dt.total_seconds() / 86400
        self.admissions['gender_M'] = (self.admissions['gender'] == 'M').astype(int)
        
        print(f"  [OK] Core Data: {len(self.admissions)} admissions")

    def extract_stats_optimized(self, table_name, itemids_dict):
        print(f"  Extracting {table_name}...")
        if not os.path.exists(FILE_PATHS[table_name]): return pd.DataFrame()
        
        all_ids = [i for s in itemids_dict.values() for i in s]
        chunks = []
        
        for chunk in pd.read_csv(FILE_PATHS[table_name], chunksize=CHUNK_SIZE, 
                                 usecols=['hadm_id', 'itemid', 'valuenum', 'charttime'], parse_dates=['charttime']):
            chunk = chunk[chunk['itemid'].isin(all_ids)].sort_values(['hadm_id', 'charttime'])
            if not chunk.empty:
                # Key Optimization: Only Mean, Last, Trend
                agg = chunk.groupby(['hadm_id', 'itemid'])['valuenum'].agg(
                    mean='mean', last='last', first='first'
                ).reset_index()
                agg['trend'] = agg['last'] - agg['first']
                chunks.append(agg.drop(columns=['first']))
                
        if not chunks: return pd.DataFrame()
        df = pd.concat(chunks)
        
        id_map = {i: k for k, v in itemids_dict.items() for i in v}
        df['name'] = df['itemid'].map(id_map)
        
        df_pivot = df.groupby(['hadm_id', 'name']).mean().reset_index()
        df_pivot = df_pivot.pivot(index='hadm_id', columns='name', values=['mean', 'last', 'trend'])
        df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]
        return df_pivot.reset_index()

    def extract_text_lists(self):
        print("STEP 2: Extracting Text Lists (Meds, Hx, & Procedures)...")
        
        # 1. Meds
        med_list = []
        if os.path.exists(FILE_PATHS['prescriptions']):
            for chunk in pd.read_csv(FILE_PATHS['prescriptions'], chunksize=CHUNK_SIZE, usecols=['hadm_id', 'drug']):
                chunk['drug'] = chunk['drug'].astype(str).str.lower()
                med_list.append(chunk.groupby('hadm_id')['drug'].apply(lambda x: ' '.join(set(x))).reset_index())
            meds_df = pd.concat(med_list).groupby('hadm_id')['drug'].apply(lambda x: ' '.join(x)).reset_index()
            meds_df.rename(columns={'drug': 'med_list_text'}, inplace=True)
        else: meds_df = pd.DataFrame()

        # 2. Diagnoses
        diag_list = []
        if os.path.exists(FILE_PATHS['diagnoses']):
            for chunk in pd.read_csv(FILE_PATHS['diagnoses'], chunksize=CHUNK_SIZE, usecols=['hadm_id', 'icd_code']):
                chunk['icd_code'] = chunk['icd_code'].astype(str)
                diag_list.append(chunk.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index())
            diag_df = pd.concat(diag_list).groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
            diag_df.rename(columns={'icd_code': 'diagnosis_list_text'}, inplace=True)
        else: diag_df = pd.DataFrame()

        # 3. Procedures (NEW)
        proc_list = []
        if os.path.exists(FILE_PATHS['procedures']):
            print("  Extracting Procedures...")
            for chunk in pd.read_csv(FILE_PATHS['procedures'], chunksize=CHUNK_SIZE, usecols=['hadm_id', 'icd_code']):
                chunk['icd_code'] = chunk['icd_code'].astype(str)
                proc_list.append(chunk.groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index())
            if proc_list:
                proc_df = pd.concat(proc_list).groupby('hadm_id')['icd_code'].apply(lambda x: ' '.join(x)).reset_index()
                proc_df.rename(columns={'icd_code': 'procedure_list_text'}, inplace=True)
            else: proc_df = pd.DataFrame()
        else: proc_df = pd.DataFrame()
            
        return meds_df, diag_df, proc_df

    def extract_notes(self):
        print("STEP 3: Extracting Discharge Summaries...")
        if not os.path.exists(FILE_PATHS['noteevents']): 
            print("  ⚠ Notes file not found.")
            return pd.DataFrame()
            
        notes = []
        # Use 'str' dtype to avoid mixed type warnings on large files
        for chunk in pd.read_csv(FILE_PATHS['noteevents'], chunksize=CHUNK_SIZE, dtype=str):
            # Filter for Discharge Summaries
            chunk = chunk[chunk['category'] == 'Discharge summary']
            if not chunk.empty: 
                notes.append(chunk[['hadm_id', 'text', 'charttime']])
            
        if not notes: return pd.DataFrame()
        
        df = pd.concat(notes).sort_values('charttime')
        # Keep last note per admission
        df = df.groupby('hadm_id')['text'].last().reset_index()
        # Basic cleaning
        df['clinical_text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        return df[['hadm_id', 'clinical_text']]

    def run(self):
        self.load_core()
        
        # Extract
        meds_df, diag_df, proc_df = self.extract_text_lists()
        notes_df = self.extract_notes()
        vitals_df = self.extract_stats_optimized('chartevents', VITAL_ITEMIDS)
        labs_df = self.extract_stats_optimized('labevents', LAB_ITEMIDS)
        
        # Merge
        print("STEP 4: Merging...")
        df = self.admissions
        for d in [meds_df, diag_df, proc_df, notes_df, vitals_df, labs_df]:
            if not d.empty:
                # Ensure ID key is correct type
                if 'hadm_id' in d.columns:
                    d['hadm_id'] = pd.to_numeric(d['hadm_id'], errors='coerce')
                df = df.merge(d, on='hadm_id', how='left')
        
        # Fill Text NaNs (Critical for Specialists)
        for col in ['med_list_text', 'diagnosis_list_text', 'clinical_text', 'procedure_list_text']:
            if col in df.columns: df[col] = df[col].fillna("empty")
            else: df[col] = "empty" # Handle missing files gracefully
            
        # Impute numeric (Critical for Structure Specialist)
        print("STEP 5: Imputing Numerics...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        # Don't impute IDs or Targets
        cols_to_impute = [c for c in num_cols if c not in ['subject_id', 'hadm_id', 'readmitted_30d', 'gender_M', 'prior_visits_count']]
        if cols_to_impute:
            df[cols_to_impute] = self.median_imputer.fit_transform(df[cols_to_impute])
        
        # Save
        if not os.path.exists('./model_outputs'):
            os.makedirs('./model_outputs')
            
        print(f"Saving to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        print("[OK] Done.")

if __name__ == "__main__":
    MIMICExtractor().run()