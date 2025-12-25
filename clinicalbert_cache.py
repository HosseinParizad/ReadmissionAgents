import os
import pickle
import hashlib
from typing import Optional
import numpy as np

CACHE_DIR = './model_outputs/cache/'
CACHE_FILE = os.path.join(CACHE_DIR, 'clinicalbert_embeddings.pkl')

class ClinicalBERTCache:
    """
    Simple persistent cache for ClinicalBERT embeddings.
    """
    def __init__(self):
        self.cache = {}
        self._ensure_cache_dir()
        self._load_cache()
        
    def _ensure_cache_dir(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"      [CACHE] Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"      [CACHE] Error loading cache: {e}")
                self.cache = {}
        else:
            print("      [CACHE] No existing cache found, starting fresh")
            
    def _save_cache(self):
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"      [CACHE] Warning: Failed to save cache: {e}")

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._hash_text(text)
        return self.cache.get(key)
        
    def set(self, text: str, embedding: np.ndarray):
        key = self._hash_text(text)
        self.cache[key] = embedding
        # Auto-save every 1000 new items could be added here, 
        # but for now we rely on OS buffer or explicit save if needed.
        # Ideally, we should save periodically.
        if len(self.cache) % 1000 == 0:
            self._save_cache()

    def save(self):
        self._save_cache()
