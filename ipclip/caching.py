import os, hashlib, numpy as np
def _cache_dir() -> str:
    return os.environ.get("PC_CACHE_FOLDER", ".cache")

def _hashed_path(name: str) -> str:
    os.makedirs(_cache_dir(), exist_ok=True)
    m = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return os.path.join(_cache_dir(), m + ".npy")

def load(name: str):
    path = _hashed_path(name)
    if os.path.exists(path):
        return np.load(path)
    return None

def save(arr, name: str):
    path = _hashed_path(name)
    with open(path, "wb") as f:
        np.save(f, arr)
    return path
