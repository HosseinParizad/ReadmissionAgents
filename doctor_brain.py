import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

from specialist_agents import LabSpecialist, NoteSpecialist, PharmacySpecialist, HistorySpecialist

# --- CONFIG ---
DATA_PATH = './model_outputs/features_4_agents.csv'
TARGET = 'readmitted_30d'

# Universal Context 
CONTEXT_COLS = ['anchor_age', 'gender_M', 'prior_visits_count', 'los_days']
TEXT_COLS = ['clinical_text', 'med_list_text', 'diagnosis_list_text']
ID_COLS = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'next_admittime', 'days_to_next']

class DoctorAgent:
    def __init__(self):
        print("\n🏥 INITIALIZING CONTEXT-AWARE MEDICAL BOARD")
        self.spec_lab = LabSpecialist()
        self.spec_note = NoteSpecialist()
        self.spec_pharm = PharmacySpecialist()
        self.spec_hist = HistorySpecialist()
        self.brain = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

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
        #(df['diagnosis_list_text'].fillna(""), X_context, y)
        self.spec_hist.learn(df['full_history_text'], X_context, y)
        
        print("\n--- PHASE 2: DOCTOR TRAINING ---")
        # Generate opinions on training data
        op_lab = self.spec_lab.give_opinion(X_lab, X_context)
        op_note = self.spec_note.give_opinion(df['clinical_text'].fillna(""), X_context)
        op_pharm = self.spec_pharm.give_opinion(df['med_list_text'].fillna(""), X_context)
        op_hist = self.spec_hist.give_opinion(df['diagnosis_list_text'].fillna(""), X_context)
        
        X_doctor = pd.DataFrame({
            'op_lab': op_lab, 'op_note': op_note, 
            'op_pharm': op_pharm, 'op_hist': op_hist
        })
        # Add context to Doctor too
        X_doctor = pd.concat([X_doctor, X_context.reset_index(drop=True)], axis=1)
        
        self.brain.fit(X_doctor, y)
        print("✅ Team Trained.")

    def diagnose_and_audit(self, df):
        print("\n--- DIAGNOSTIC ROUND ---")
        X_context = df[CONTEXT_COLS].fillna(0)
        
        # Get Opinions
        drop_cols = TEXT_COLS + CONTEXT_COLS + ID_COLS + [TARGET]
        X_lab = df.drop(columns=drop_cols, errors='ignore').fillna(0)
        
        op_lab = self.spec_lab.give_opinion(X_lab, X_context)
        op_note = self.spec_note.give_opinion(df['clinical_text'].fillna(""), X_context)
        op_pharm = self.spec_pharm.give_opinion(df['med_list_text'].fillna(""), X_context)
        op_hist = self.spec_hist.give_opinion(df['diagnosis_list_text'].fillna(""), X_context)
        
        # Prepare Data for Doctor
        X_doctor = pd.DataFrame({
            'op_lab': op_lab, 'op_note': op_note, 
            'op_pharm': op_pharm, 'op_hist': op_hist
        })
        X_doctor = pd.concat([X_doctor, X_context.reset_index(drop=True)], axis=1)
        
        final_probs = self.brain.predict_proba(X_doctor)[:, 1]
        
        # --- PERFORM AUTOPSY (Debug the Agents) ---
        self.perform_autopsy(df[TARGET].values, final_probs, op_lab, op_note, op_pharm, op_hist)
        
        return final_probs

    def perform_autopsy(self, y_true, y_pred_prob, op_lab, op_note, op_pharm, op_hist):
        """
        Analyzes who is to blame for errors.
        """
        print("\n🔎 PERFORMING AUTOPSY ON PREDICTIONS...")
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Find indices where the Doctor was WRONG
        errors = np.where(y_pred != y_true)[0]
        print(f"   Found {len(errors)} errors out of {len(y_true)} patients.")
        
        if len(errors) == 0: return

        # Audit the first 5 errors
        print(f"   Analyzing first 5 failures:")
        for idx in errors[:5]:
            truth = "Readmitted" if y_true[idx] == 1 else "Healthy"
            doc_guess = "Readmitted" if y_pred[idx] == 1 else "Healthy"
            
            print(f"\n   ⚠️ Patient #{idx} | Truth: {truth} | Doctor Said: {doc_guess}")
            print(f"      - Lab Agent:  {op_lab[idx]:.2f}  ({'✅' if (op_lab[idx]>0.5) == y_true[idx] else '❌'})")
            print(f"      - Note Agent: {op_note[idx]:.2f} ({'✅' if (op_note[idx]>0.5) == y_true[idx] else '❌'})")
            print(f"      - Pharm Agent:{op_pharm[idx]:.2f} ({'✅' if (op_pharm[idx]>0.5) == y_true[idx] else '❌'})")
            print(f"      - Hist Agent: {op_hist[idx]:.2f} ({'✅' if (op_hist[idx]>0.5) == y_true[idx] else '❌'})")
            
        # Global Agent Performance
        print("\n   🏆 AGENT RELIABILITY REPORT (Accuracy on this Batch):")
        acc_lab = np.mean((op_lab > 0.5) == y_true)
        acc_note = np.mean((op_note > 0.5) == y_true)
        acc_pharm = np.mean((op_pharm > 0.5) == y_true)
        acc_hist = np.mean((op_hist > 0.5) == y_true)
        
        print(f"      Labs:     {acc_lab:.2%}")
        print(f"      Notes:    {acc_note:.2%}")
        print(f"      Meds:     {acc_pharm:.2%}")
        print(f"      History:  {acc_hist:.2%}")

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