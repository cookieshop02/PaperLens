from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np

# Load once at module level — avoids reloading on every call
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[Embedder] Loading all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Embedder] Model loaded.")
    return _model


def embed_text(text: Union[str, list[str]]) -> np.ndarray:
    """
    Embed a single string or a list of strings.
    Returns a numpy array of shape (n, 384).
    """
    model = get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Returns a flat list of floats (for Qdrant search).
    """
    vector = embed_text(query)
    return vector.tolist()
