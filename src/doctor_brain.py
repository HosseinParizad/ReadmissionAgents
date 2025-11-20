import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix

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
        # Labs - only numeric columns (exclude text, context, IDs, and any string columns)
        drop_cols = TEXT_COLS + CONTEXT_COLS + ID_COLS + [TARGET]
        X_lab = df.drop(columns=drop_cols, errors='ignore')
        # Select only numeric columns to avoid string columns like 'curr_service'
        X_lab = X_lab.select_dtypes(include=[np.number]).fillna(0)
        print(f"  Lab features: {X_lab.shape[1]} numeric columns")
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
        X_lab = df.drop(columns=drop_cols, errors='ignore')
        # Select only numeric columns to avoid string columns like 'curr_service'
        X_lab = X_lab.select_dtypes(include=[np.number]).fillna(0)
        
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
        print("\n🔎 FINAL PERFORMANCE REPORT...")
        
        # Use the optimized threshold
        y_pred = (y_pred_prob > self.optimal_threshold).astype(int)
        
        # 1. METRICS THAT MATTER
        auc = roc_auc_score(y_true, y_pred_prob)
        recall = recall_score(y_true, y_pred)     # How many readmissions did we catch?
        precision = precision_score(y_true, y_pred) # When we flagged risk, were we right?
        f1 = f1_score(y_true, y_pred)             # Balance of both
        
        print(f"   🏆 AUC-ROC:   {auc:.4f} (Primary Metric)")
        print(f"   🎯 Recall:    {recall:.4f} (Did we catch the sick patients?)")
        print(f"   🎯 Precision: {precision:.4f} (Did we cry wolf too often?)")
        print(f"   ⚖️ F1-Score:  {f1:.4f}")
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\n   🏥 Clinical Impact Matrix:")
        print(f"      [ True Neg (Safe) | False Pos (Over-alert) ]")
        print(f"      [ {cm[0,0]:<13} | {cm[0,1]:<18} ]")
        print(f"      ------------------------------------------")
        print(f"      [ {cm[1,0]:<13} | {cm[1,1]:<18} ]")
        print(f"      [ False Neg (Missed!) | True Pos (Caught!) ]")
        
        # 3. Agent Reliability (Correlation with Truth)
        # Which agent is most correlated with the actual outcome?
        print("\n   🕵️ AGENT CORRELATION (Who knows the truth?):")
        # Pearson correlation
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