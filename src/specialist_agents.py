import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
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
        # Gradient Boosting is best for dense numeric data (Labs/Vitals)
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
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # INCREASED MAX_ITER to 1000 to prevent warnings
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.fusion = ContextFusion()

    def learn(self, text_list, context, y):
        print(f"   📖 [{self.name}] Learning from Notes + Context...")
        self.fusion.fit(context)
        short_text = [str(t)[:1000] for t in text_list]
        embeds = self.encoder.encode(short_text, batch_size=64, show_progress_bar=True)
        X_final = self.fusion.merge(embeds, context)
        self.model.fit(X_final, y)

    def give_opinion(self, text_list, context):
        short_text = [str(t)[:1000] for t in text_list]
        embeds = self.encoder.encode(short_text, batch_size=64)
        X_final = self.fusion.merge(embeds, context)
        return self.model.predict_proba(X_final)[:, 1]

# ==========================================
# 3. PHARMACY SPECIALIST (Keyword Expert)
# ==========================================
class PharmacySpecialist:
    def __init__(self):
        self.name = "Spec_Pharm"
        self.vectorizer = CountVectorizer(max_features=500, stop_words='english')
        # SOLVER CHANGED TO 'liblinear' (Faster for sparse data)
        # MAX_ITER INCREASED
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
        # Increased features slightly to capture more ICD codes
        self.vectorizer = CountVectorizer(max_features=800)
        # SOLVER CHANGED TO 'liblinear'
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