"""Semantic embedding generation module using Sentence Transformers."""

import logging
from typing import List, Optional, Callable
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Module-level cache for standalone/test execution
_GLOBAL_MODEL_CACHE: dict = {}


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model.

    Uses Streamlit's cache_resource if in a Streamlit context, or module-level
    dictionary cache otherwise.
    """
    try:
        import streamlit as st  # type: ignore
        # If running inside Streamlit, use st.cache_resource
        if hasattr(st, "cache_resource"):
            @st.cache_resource(show_spinner=False)
            def _load_st_model(name: str) -> SentenceTransformer:
                logger.info(f"Loading embedding model: {name}")
                return SentenceTransformer(name)
            return _load_st_model(model_name)
    except Exception:
        pass

    # Fallback to module cache
    if model_name not in _GLOBAL_MODEL_CACHE:
        logger.info(f"Loading embedding model (standalone): {model_name}")
        _GLOBAL_MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    return _GLOBAL_MODEL_CACHE[model_name]


def generate_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """Generate normalized float32 embeddings for a list of text strings.

    Args:
        texts: List of text passages to embed.
        model: Optional pre-loaded SentenceTransformer instance.
        model_name: Name of model to load if model is None.
        batch_size: Batch size for encoding large document collections.
        progress_callback: Optional callback receiving (progress_pct: float, status_msg: str).

    Returns:
        np.ndarray of shape (N, dimension) with dtype float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    active_model = model or get_embedding_model(model_name)

    total_texts = len(texts)
    all_embeddings: List[np.ndarray] = []

    for i in range(0, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = active_model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=True,  # Crucial for Cosine Similarity with FAISS IndexFlatIP
            convert_to_numpy=True,
        )
        all_embeddings.append(batch_embeddings)

        if progress_callback:
            progress = min(1.0, (i + len(batch)) / total_texts)
            progress_callback(
                progress,
                f"Generated embeddings for {min(i + len(batch), total_texts)} / {total_texts} chunks...",
            )

    combined = np.vstack(all_embeddings).astype(np.float32)

    # Double check L2 normalization for float32 precision
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = (combined / norms).astype(np.float32)

    # Dimension validation
    if normalized.shape[1] != EMBEDDING_DIMENSION:
        logger.warning(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {normalized.shape[1]}"
        )

    return normalized


def generate_query_embedding(
    query: str,
    model: Optional[SentenceTransformer] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Generate a single normalized float32 embedding vector for a user query.

    Args:
        query: User's question or search query.
        model: Optional pre-loaded model.
        model_name: Model identifier.

    Returns:
        np.ndarray of shape (1, dimension) with dtype float32, L2-normalized.
    """
    if not query or not query.strip():
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    active_model = model or get_embedding_model(model_name)
    vector = active_model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Re-normalize for numerical stability
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    if norm[0, 0] > 0:
        vector = vector / norm

    return vector
