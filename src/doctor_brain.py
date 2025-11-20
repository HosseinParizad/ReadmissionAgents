import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

from specialist_agents import LabSpecialist, NoteSpecialist, PharmacySpecialist, HistorySpecialist

# --- CONFIG ---
DATA_PATH = './model_outputs/features_4_agents_augmented.csv'
TARGET = 'readmitted_30d'

# Universal Context 
CONTEXT_COLS = ['anchor_age', 'gender_M', 'prior_visits_count', 'los_days']
TEXT_COLS = ['clinical_text', 'med_list_text', 'diagnosis_list_text', 'proc_list_text', 'full_history_text']
ID_COLS = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'next_admittime', 'days_to_next', 'curr_service']

class DoctorAgent:
    def __init__(self):
        print("\n🏥 INITIALIZING CONTEXT-AWARE MEDICAL BOARD")
        self.spec_lab = LabSpecialist()
        self.spec_note = NoteSpecialist()
        self.spec_pharm = PharmacySpecialist()
        self.spec_hist = HistorySpecialist()
        
        # Doctor's Brain
        self.brain = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05)
        self.optimal_threshold = 0.5

    def train_team(self, df):
        y = df[TARGET]
        X_context = df[CONTEXT_COLS].fillna(0)
        
        print("\n--- PHASE 1: SPECIALIST TRAINING ---")
        # Labs
        drop_cols = TEXT_COLS + CONTEXT_COLS + ID_COLS + [TARGET]
        X_lab = df.drop(columns=drop_cols, errors='ignore').fillna(0)
        self.spec_lab.learn(X_lab, X_context, y)
        
        # Others
        self.spec_note.learn(df['clinical_text'].fillna(""), X_context, y)
        self.spec_pharm.learn(df['med_list_text'].fillna(""), X_context, y)
        # Check if full_history exists (from augmented extractor), else use diagnosis
        hist_col = 'full_history_text' if 'full_history_text' in df.columns else 'diagnosis_list_text'
        self.spec_hist.learn(df[hist_col].fillna(""), X_context, y)
        
        print("\n--- PHASE 2: DOCTOR TRAINING ---")
        # Generate opinions
        op_lab = self.spec_lab.give_opinion(X_lab, X_context)
        op_note = self.spec_note.give_opinion(df['clinical_text'].fillna(""), X_context)
        op_pharm = self.spec_pharm.give_opinion(df['med_list_text'].fillna(""), X_context)
        op_hist = self.spec_hist.give_opinion(df[hist_col].fillna(""), X_context)
        
        X_doctor = pd.DataFrame({
            'op_lab': op_lab, 'op_note': op_note, 
            'op_pharm': op_pharm, 'op_hist': op_hist
        })
        X_doctor = pd.concat([X_doctor, X_context.reset_index(drop=True)], axis=1)
        
        self.brain.fit(X_doctor, y)
        
        # --- CALIBRATE THRESHOLD ---
        # Find the best threshold that balances Precision/Recall (F1 Score)
        train_probs = self.brain.predict_proba(X_doctor)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, train_probs)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        self.optimal_threshold = thresholds[np.argmax(f1_scores)]
        print(f"✅ Team Trained. Optimal Decision Threshold: {self.optimal_threshold:.4f}")

    def diagnose_and_audit(self, df):
        print("\n--- DIAGNOSTIC ROUND ---")
        X_context = df[CONTEXT_COLS].fillna(0)
        
        # Get Opinions
        drop_cols = TEXT_COLS + CONTEXT_COLS + ID_COLS + [TARGET]
        X_lab = df.drop(columns=drop_cols, errors='ignore').fillna(0)
        
        op_lab = self.spec_lab.give_opinion(X_lab, X_context)
        op_note = self.spec_note.give_opinion(df['clinical_text'].fillna(""), X_context)
        op_pharm = self.spec_pharm.give_opinion(df['med_list_text'].fillna(""), X_context)
        
        hist_col = 'full_history_text' if 'full_history_text' in df.columns else 'diagnosis_list_text'
        op_hist = self.spec_hist.give_opinion(df[hist_col].fillna(""), X_context)
        
        # Doctor Decides
        X_doctor = pd.DataFrame({
            'op_lab': op_lab, 'op_note': op_note, 
            'op_pharm': op_pharm, 'op_hist': op_hist
        })
        X_doctor = pd.concat([X_doctor, X_context.reset_index(drop=True)], axis=1)
        
        final_probs = self.brain.predict_proba(X_doctor)[:, 1]
        
        self.perform_autopsy(df[TARGET].values, final_probs, op_lab, op_note, op_pharm, op_hist)
        
        return final_probs

    def perform_autopsy(self, y_true, y_pred_prob, op_lab, op_note, op_pharm, op_hist):
        print("\n🔎 PERFORMING AUTOPSY ON PREDICTIONS...")
        print(f"   Using Calibrated Threshold: {self.optimal_threshold:.4f}")
        
        # Use the SMART threshold, not 0.5
        y_pred = (y_pred_prob > self.optimal_threshold).astype(int)
        
        errors = np.where(y_pred != y_true)[0]
        print(f"   Found {len(errors)} errors out of {len(y_true)} patients.")
        
        if len(errors) == 0: return

        print(f"   Analyzing first 5 failures:")
        for idx in errors[:5]:
            truth = "Readmitted" if y_true[idx] == 1 else "Healthy"
            doc_guess = "Readmitted" if y_pred[idx] == 1 else "Healthy"
            
            print(f"\n   ⚠️ Patient #{idx} | Truth: {truth} | Doctor Said: {doc_guess} (Risk: {y_pred_prob[idx]:.2f})")
            # We check against the threshold, not 0.5
            print(f"      - Lab Agent:  {op_lab[idx]:.2f}")
            print(f"      - Note Agent: {op_note[idx]:.2f}")
            print(f"      - Pharm Agent:{op_pharm[idx]:.2f}")
            print(f"      - Hist Agent: {op_hist[idx]:.2f}")
            
        print("\n   🏆 DOCTOR'S TRUST MATRIX (What matters most?):")
        imps = self.brain.feature_importances_
        # Match features to importance
        feats = ['Lab_Opinion', 'Note_Opinion', 'Pharm_Opinion', 'Hist_Opinion', 'Age', 'Gender', 'Prior_Visits', 'LOS']
        for f, i in zip(feats, imps):
            print(f"      - {f}: {i:.4f}")

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