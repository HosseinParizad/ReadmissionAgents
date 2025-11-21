import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix

from specialist_agents import LabSpecialist, NoteSpecialist, PharmacySpecialist, HistorySpecialist

# --- CONFIG ---
DATA_PATH = './model_outputs/features_4_agents_augmented.csv'
TARGET = 'readmitted_30d'

# BASE CONTEXT (We will add more dynamic ones below)
BASE_CONTEXT = ['anchor_age', 'gender_M', 'prior_visits_count', 'los_days']
TEXT_COLS = ['clinical_text', 'med_list_text', 'diagnosis_list_text', 'proc_list_text', 'full_history_text']
ID_COLS = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'next_admittime', 'days_to_next', 'curr_service']

class DoctorAgent:
    def __init__(self):
        print("\n🏥 INITIALIZING CLINICAL-BERT MEDICAL BOARD (V2: Regex + Rich Context)")
        self.spec_lab = LabSpecialist()
        self.spec_note = NoteSpecialist()
        self.spec_pharm = PharmacySpecialist()
        self.spec_hist = HistorySpecialist()
        
        # Slower learning rate for better generalization
        self.brain = GradientBoostingClassifier(
            n_estimators=300, 
            learning_rate=0.02, 
            max_depth=5,
            random_state=42
        )
        self.optimal_threshold = 0.5
        self.context_cols = [] # Will be populated in train

    def enrich_context(self, df):
        """
        SUGGESTION IMPLEMENTED: Feature Engineering
        We derive quantitative metrics from the text lists.
        """
        df = df.copy()
        
        # 1. Polypharmacy (Count of drugs)
        # We count spaces + 1 as a rough proxy for number of items
        df['n_meds'] = df['med_list_text'].fillna("").apply(lambda x: str(x).count(' ') + 1 if len(str(x)) > 0 else 0)
        
        # 2. Complexity (Count of diagnoses and procedures)
        df['n_diagnoses'] = df['diagnosis_list_text'].fillna("").apply(lambda x: str(x).count(' ') + 1 if len(str(x)) > 0 else 0)
        
        # Check if proc_list_text exists (it might be missing in some extracts)
        if 'proc_list_text' in df.columns:
             df['n_procs'] = df['proc_list_text'].fillna("").apply(lambda x: str(x).count(' ') + 1 if len(str(x)) > 0 else 0)
        else:
             df['n_procs'] = 0

        # 3. Service Risk (Is it a surgical or medical admission?)
        # Simple encoding: 1 if Surgery, 0 if Medicine/Other
        if 'curr_service' in df.columns:
            df['is_surgery'] = df['curr_service'].astype(str).str.contains('SURG', case=False, na=False).astype(int)
        else:
            df['is_surgery'] = 0

        # 4. Interaction Terms (Age * Complexity)
        df['age_x_meds'] = df['anchor_age'] * df['n_meds']
        
        # Define final context columns
        new_features = ['n_meds', 'n_diagnoses', 'n_procs', 'is_surgery', 'age_x_meds']
        return df, BASE_CONTEXT + new_features

    def train_team(self, df):
        y = df[TARGET]
        
        print("   ✨ Engineering Rich Context Features...")
        df_enriched, self.context_cols = self.enrich_context(df)
        X_context = df_enriched[self.context_cols].fillna(0)
        
        print("\n--- PHASE 1: SPECIALIST TRAINING ---")
        
        # --- THE FALLBACK MANEUVER ---
        # We combine Clinical Text + Diagnosis + Procedures.
        # If the note is "empty", BERT will still see the diagnoses!
        print("   📄 Constructing Hybrid Text (Notes + Diagnoses)...")
        # Check if proc_list_text exists
        proc_text = df['proc_list_text'].fillna("") if 'proc_list_text' in df.columns else pd.Series([""] * len(df))
        hybrid_text = (
            df['clinical_text'].replace("empty", "") + " \n " + 
            "DIAGNOSES: " + df['diagnosis_list_text'].fillna("") + " \n " +
            "PROCEDURES: " + proc_text
        )
        
        drop_cols = TEXT_COLS + BASE_CONTEXT + ID_COLS + [TARGET] + ['n_meds', 'n_diagnoses', 'n_procs', 'is_surgery', 'age_x_meds']
        X_lab = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        
        print(f"  Lab features: {X_lab.shape[1]} numeric columns")
        self.spec_lab.learn(X_lab, X_context, y)
        
        # FEED HYBRID TEXT HERE
        self.spec_note.learn(hybrid_text, X_context, y)
        
        self.spec_pharm.learn(df['med_list_text'].fillna(""), X_context, y)
        hist_col = 'full_history_text' if 'full_history_text' in df.columns else 'diagnosis_list_text'
        self.spec_hist.learn(df[hist_col].fillna(""), X_context, y)
        
        print("\n--- PHASE 2: DOCTOR TRAINING ---")
        op_lab = self.spec_lab.give_opinion(X_lab, X_context)
        
        # FEED HYBRID TEXT HERE
        op_note = self.spec_note.give_opinion(hybrid_text, X_context)
        
        op_pharm = self.spec_pharm.give_opinion(df['med_list_text'].fillna(""), X_context)
        op_hist = self.spec_hist.give_opinion(df[hist_col].fillna(""), X_context)
        
        X_doctor = pd.DataFrame({
            'op_lab': op_lab, 'op_note': op_note, 
            'op_pharm': op_pharm, 'op_hist': op_hist
        })
        X_doctor = pd.concat([X_doctor, X_context.reset_index(drop=True)], axis=1)
        
        self.brain.fit(X_doctor, y)
        
        # --- CALIBRATION ---
        train_probs = self.brain.predict_proba(X_doctor)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, train_probs)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        
        safe_indices = np.where(precision[:-1] > 0.40)[0]
        if len(safe_indices) > 0:
            best_idx = safe_indices[np.argmax(f1_scores[safe_indices])]
            self.optimal_threshold = thresholds[best_idx]
            print(f"✅ Team Trained. Threshold: {self.optimal_threshold:.4f} (Precision-Enforced)")
        else:
            self.optimal_threshold = thresholds[np.argmax(f1_scores)]
            print(f"⚠️ Team Trained (Fallback). Threshold: {self.optimal_threshold:.4f}")

    def diagnose_and_audit(self, df):
        print("\n--- DIAGNOSTIC ROUND ---")
        df_enriched, _ = self.enrich_context(df)
        X_context = df_enriched[self.context_cols].fillna(0)
        
        # --- THE FALLBACK MANEUVER (Repeat for Test Set) ---
        # Check if proc_list_text exists
        proc_text = df['proc_list_text'].fillna("") if 'proc_list_text' in df.columns else pd.Series([""] * len(df))
        hybrid_text = (
            df['clinical_text'].replace("empty", "") + " \n " + 
            "DIAGNOSES: " + df['diagnosis_list_text'].fillna("") + " \n " +
            "PROCEDURES: " + proc_text
        )
        
        drop_cols = TEXT_COLS + BASE_CONTEXT + ID_COLS + [TARGET] + ['n_meds', 'n_diagnoses', 'n_procs', 'is_surgery', 'age_x_meds']
        X_lab = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        
        op_lab = self.spec_lab.give_opinion(X_lab, X_context)
        
        # FEED HYBRID TEXT HERE
        op_note = self.spec_note.give_opinion(hybrid_text, X_context)
        
        op_pharm = self.spec_pharm.give_opinion(df['med_list_text'].fillna(""), X_context)
        hist_col = 'full_history_text' if 'full_history_text' in df.columns else 'diagnosis_list_text'
        op_hist = self.spec_hist.give_opinion(df[hist_col].fillna(""), X_context)
        
        X_doctor = pd.DataFrame({
            'op_lab': op_lab, 'op_note': op_note, 
            'op_pharm': op_pharm, 'op_hist': op_hist
        })
        X_doctor = pd.concat([X_doctor, X_context.reset_index(drop=True)], axis=1)
        
        final_probs = self.brain.predict_proba(X_doctor)[:, 1]
        self.perform_autopsy(df[TARGET].values, final_probs, op_lab, op_note, op_pharm, op_hist)
        return final_probs

    def perform_autopsy(self, y_true, y_pred_prob, op_lab, op_note, op_pharm, op_hist):
        print("\n🔎 FINAL PERFORMANCE REPORT (Rich Context + Regex)...")
        
        y_pred = (y_pred_prob > self.optimal_threshold).astype(int)
        
        auc = roc_auc_score(y_true, y_pred_prob)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"   🏆 AUC-ROC:   {auc:.4f}")
        print(f"   🎯 Recall:    {recall:.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   ⚖️ F1-Score:  {f1:.4f}")
        
        print("\n   🕵️ AGENT CORRELATION:")
        print(f"      Lab Agent:     {np.corrcoef(op_lab, y_true)[0,1]:.4f}")
        print(f"      Note Agent:    {np.corrcoef(op_note, y_true)[0,1]:.4f}")
        print(f"      Pharm Agent:   {np.corrcoef(op_pharm, y_true)[0,1]:.4f}")
        print(f"      History Agent: {np.corrcoef(op_hist, y_true)[0,1]:.4f}")

if __name__ == "__main__":
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Run ExtractMimicData.py first!")
        exit()

    df = df.dropna(subset=[TARGET])
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    dr = DoctorAgent()
    dr.train_team(train_df)
    dr.diagnose_and_audit(test_df)