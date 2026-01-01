import os
import pickle
import hashlib
from typing import Optional
from collections import deque
import numpy as np

CACHE_DIR = './model_outputs/cache/'
CACHE_FILE = os.path.join(CACHE_DIR, 'clinicalbert_embeddings.pkl')

class ClinicalBERTCache:
    """
    Simple persistent cache for ClinicalBERT embeddings.
    """
    def __init__(self, max_cache_size: Optional[int] = None):
        """
        Args:
            max_cache_size: Maximum number of embeddings to keep in memory.
                           If None, no limit. If exceeded, oldest entries are removed.
        """
        self.cache = {}
        self.max_cache_size = max_cache_size
        self._access_order = deque()  # Use deque for O(1) pop from left
        self._access_set = set()  # Use set for O(1) lookup
        self._ensure_cache_dir()
        self._load_cache()
        
    def _ensure_cache_dir(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                # Check file size - if too small, likely corrupted
                file_size = os.path.getsize(CACHE_FILE)
                if file_size < 100:  # Less than 100 bytes is definitely corrupted
                    print(f"      [CACHE] Cache file too small ({file_size} bytes), likely corrupted. Starting fresh.")
                    os.remove(CACHE_FILE)
                    self.cache = {}
                    return
                
                with open(CACHE_FILE, 'rb') as f:
                    loaded_cache = pickle.load(f)
                
                # Validate cache structure
                if not isinstance(loaded_cache, dict):
                    raise ValueError("Cache is not a dictionary")
                
                # Check a few entries to ensure they're valid
                sample_keys = list(loaded_cache.keys())[:10]
                for key in sample_keys:
                    if not isinstance(loaded_cache[key], np.ndarray):
                        raise ValueError(f"Invalid cache entry for key {key}")
                
                # Apply max_cache_size limit if set (keep most recent entries)
                if self.max_cache_size and len(loaded_cache) > self.max_cache_size:
                    # Keep only the most recent entries (simple approach: keep last N)
                    all_keys = list(loaded_cache.keys())
                    keys_to_keep = all_keys[-self.max_cache_size:]
                    self.cache = {k: loaded_cache[k] for k in keys_to_keep}
                    self._access_order = deque(keys_to_keep)  # Use deque
                    self._access_set = set(keys_to_keep)  # Use set
                    print(f"      [CACHE] Loaded {len(loaded_cache)} cached embeddings, keeping {len(self.cache)} in memory (max {self.max_cache_size})")
                else:
                    self.cache = loaded_cache
                    if self.max_cache_size:
                        self._access_order = deque(self.cache.keys())  # Use deque
                        self._access_set = set(self.cache.keys())  # Use set
                    print(f"      [CACHE] Loaded {len(self.cache)} cached embeddings")
            except (EOFError, pickle.UnpicklingError, ValueError) as e:
                print(f"      [CACHE] Error loading cache: {e}")
                print(f"      [CACHE] Deleting corrupted cache file and starting fresh...")
                try:
                    os.remove(CACHE_FILE)
                except:
                    pass
                self.cache = {}
            except Exception as e:
                print(f"      [CACHE] Unexpected error loading cache: {e}")
                print(f"      [CACHE] Deleting cache file and starting fresh...")
                try:
                    os.remove(CACHE_FILE)
                except:
                    pass
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
        result = self.cache.get(key)
        # Update access order for LRU - optimized with set
        if result is not None and self.max_cache_size:
            if key in self._access_set:
                # Move to end - remove from deque and re-add
                # Note: deque doesn't support remove efficiently, so we rebuild if needed
                # But for get operations, we can skip this optimization
                pass  # Skip reordering on get for performance
        return result
        
    def set(self, text: str, embedding: np.ndarray):
        key = self._hash_text(text)
        
        # If cache is full, remove oldest entry (LRU eviction) - O(1) with deque
        if self.max_cache_size and len(self.cache) >= self.max_cache_size:
            if key not in self.cache:  # Only evict if adding new entry
                oldest_key = self._access_order.popleft()  # O(1) instead of O(n)
                self._access_set.discard(oldest_key)  # O(1)
                del self.cache[oldest_key]
        
        self.cache[key] = embedding
        
        # Update access order - O(1) operations
        if self.max_cache_size:
            if key in self._access_set:
                # Key already exists - we'll just append (duplicate in deque is OK for now)
                # For true LRU, we'd need to remove first, but that's O(n)
                # For bulk operations, we optimize by not removing duplicates
                pass
            self._access_order.append(key)  # O(1)
            self._access_set.add(key)  # O(1)
        
        # Auto-save less frequently during bulk operations
        cache_size = len(self.cache)
        if cache_size % 5000 == 0 and cache_size < 50000:  # Less frequent saves
            try:
                self._save_cache()
            except Exception as e:
                pass  # Silent during bulk operations

    def save(self):
        self._save_cache()
