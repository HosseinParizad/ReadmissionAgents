import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  # CHANGED from CountVectorizer
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
# 1. LAB SPECIALIST (Numeric Expert)
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
# 2. NOTE SPECIALIST (Semantic Expert)
# ==========================================
class NoteSpecialist:
    def __init__(self):
        self.name = "Spec_Notes"
        # Use a slightly larger model if GPU available, otherwise keep MiniLM
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.fusion = ContextFusion()

    def _preprocess(self, text_list):
        # FIX: Take the LAST 2000 chars (Discharge instructions/Plan), not the first.
        # If text is shorter than 2000, it takes the whole thing.
        return [str(t)[-2000:] for t in text_list]

    def learn(self, text_list, context, y):
        print(f"   📖 [{self.name}] Learning from Notes (Discharge Plan Focus)...")
        self.fusion.fit(context)
        
        processed_text = self._preprocess(text_list)
        embeds = self.encoder.encode(processed_text, batch_size=64, show_progress_bar=True)
        
        X_final = self.fusion.merge(embeds, context)
        self.model.fit(X_final, y)

    def give_opinion(self, text_list, context):
        processed_text = self._preprocess(text_list)
        embeds = self.encoder.encode(processed_text, batch_size=64)
        X_final = self.fusion.merge(embeds, context)
        return self.model.predict_proba(X_final)[:, 1]

# ==========================================
# 3. PHARMACY SPECIALIST (Keyword Expert)
# ==========================================
class PharmacySpecialist:
    def __init__(self):
        self.name = "Spec_Pharm"
        # CHANGE: TF-IDF captures "rare" drugs (often for severe conditions) better than counts
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
# 4. HISTORY SPECIALIST (Keyword Expert)
# ==========================================
class HistorySpecialist:
    def __init__(self):
        self.name = "Spec_Hist"
        # CHANGE: Increased features to 1500 to capture more specific ICD codes
        # CHANGE: TF-IDF downweights common codes (e.g., "Hypertension") to focus on specific comorbidities
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