import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# --- HELPER: Safe Context Fusion ---
class ContextFusion:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fit = False
        
    def fit(self, context_df):
        self.scaler.fit(context_df)
        self.is_fit = True
        
    def merge(self, features, context_df):
        if not self.is_fit: self.fit(context_df)
        context_scaled = self.scaler.transform(context_df)
        if hasattr(features, "toarray"): features = features.toarray()
        return np.column_stack((features, context_scaled))

# ==========================================
# 1. LAB SPECIALIST
# ==========================================
class LabSpecialist:
    def __init__(self):
        self.name = "Spec_Labs"
        self.model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300)
        self.fusion = ContextFusion()

    def learn(self, X_labs, context, y):
        print(f"   🧪 [{self.name}] Learning from Labs + Context...")
        self.fusion.fit(context)
        X_final = self.fusion.merge(X_labs, context)
        self.model.fit(X_final, y)

    def give_opinion(self, X_labs, context):
        X_final = self.fusion.merge(X_labs, context)
        return self.model.predict_proba(X_final)[:, 1]

# ==========================================
# 2. NOTE SPECIALIST (Risk Highlighter + Hybrid)
# ==========================================
class NoteSpecialist:
    def __init__(self):
        self.name = "Spec_Notes"
        # ClinicalBERT is good, but we need to force it to see "Risk"
        self.encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        self.model = LogisticRegression(class_weight='balanced', max_iter=2000)
        self.fusion = ContextFusion()
        
        # HIGH-RISK KEYWORDS (The "Red Flags")
        self.risk_lexicon = [
            r'non.?complian', r'refus', r'against medical advice', r' ama ', 
            r'homeless', r'shelter', r'undomiciled', r'substance', r'alcohol', r'etoh',
            r'unstable', r'frail', r'dementia', r'fall', r'metast', r'hospice', 
            r'palliative', r'poor prognosis', r'severe', r'critical'
        ]

    def _preprocess(self, text_list, train_mode=False):
        # Ensure text_list is a list/array, not a pandas Series
        if hasattr(text_list, 'tolist'):
            text_list = text_list.tolist()
        elif hasattr(text_list, 'values'):
            text_list = text_list.values.tolist()
        
        cleaned_text = []
        risk_scores = []
        
        for i, t in enumerate(text_list):
            s = str(t).lower()
            
            # 1. Calculate Explicit Risk Score (Count the Red Flags)
            score = 0
            for pattern in self.risk_lexicon:
                if re.search(pattern, s):
                    score += 1
            risk_scores.append(score)

            # 2. Smart Extraction: Grab the "Assessment" OR "Discharge" OR "Plan"
            # If Regex fails, we take the 'Risk Sentences' + Last 1000 chars
            match = re.search(r'(?:assessment|impression|plan|discharge instructions)([\s\S]*?)(?:signed|dictated|\Z)', s)
            
            if match and len(match.group(1)) > 50:
                extracted = match.group(1).strip()
            else:
                # FALLBACK: Take the last 1000 chars
                extracted = s[-1000:]
            
            # Prepend the risk score so BERT "sees" it immediately
            final_text = f"[RISK_SCORE: {score}] {extracted[:1500]}" 
            cleaned_text.append(final_text)
            
            # DEBUG: Print the first one so we can see what it found
            if train_mode and i == 0:
                print(f"\n   🕵️ [DEBUG] Note Agent is reading this (Sample 1):\n   '{final_text[:200]}...'\n")

        return cleaned_text, np.array(risk_scores).reshape(-1, 1)

    def learn(self, text_list, context, y):
        print(f"   📖 [{self.name}] Learning (ClinicalBERT + Risk Lexicon)...")
        self.fusion.fit(context)
        
        processed_text, risk_scores = self._preprocess(text_list, train_mode=True)
        embeds = self.encoder.encode(processed_text, batch_size=32, show_progress_bar=True)
        
        # Merge: BERT Embeddings + Risk Score + Context
        # We inject the Risk Score as an explicit feature alongside the embedding
        X_augmented = np.column_stack([embeds, risk_scores])
        X_final = self.fusion.merge(X_augmented, context)
        
        self.model.fit(X_final, y)

    def give_opinion(self, text_list, context):
        processed_text, risk_scores = self._preprocess(text_list)
        # Show progress bar when encoding (this is the slow part)
        embeds = self.encoder.encode(processed_text, batch_size=32, show_progress_bar=True)
        
        X_augmented = np.column_stack([embeds, risk_scores])
        X_final = self.fusion.merge(X_augmented, context)
        
        return self.model.predict_proba(X_final)[:, 1]

# ==========================================
# 3. PHARMACY SPECIALIST
# ==========================================
class PharmacySpecialist:
    def __init__(self):
        self.name = "Spec_Pharm"
        self.vectorizer = TfidfVectorizer(max_features=800, stop_words='english')
        self.model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
        self.fusion = ContextFusion()

    def learn(self, text_list, context, y):
        print(f"   💊 [{self.name}] Learning from Meds + Context...")
        self.fusion.fit(context)
        X_vec = self.vectorizer.fit_transform(text_list)
        X_final = self.fusion.merge(X_vec, context)
        self.model.fit(X_final, y)

    def give_opinion(self, text_list, context):
        X_vec = self.vectorizer.transform(text_list)
        X_final = self.fusion.merge(X_vec, context)
        return self.model.predict_proba(X_final)[:, 1]

# ==========================================
# 4. HISTORY SPECIALIST
# ==========================================
class HistorySpecialist:
    def __init__(self):
        self.name = "Spec_Hist"
        self.vectorizer = TfidfVectorizer(max_features=1500)
        self.model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
        self.fusion = ContextFusion()

    def learn(self, text_list, context, y):
        print(f"   📜 [{self.name}] Learning from History + Context...")
        self.fusion.fit(context)
        X_vec = self.vectorizer.fit_transform(text_list)
        X_final = self.fusion.merge(X_vec, context)
        self.model.fit(X_final, y)

    def give_opinion(self, text_list, context):
        X_vec = self.vectorizer.transform(text_list)
        X_final = self.fusion.merge(X_vec, context)
        return self.model.predict_proba(X_final)[:, 1]