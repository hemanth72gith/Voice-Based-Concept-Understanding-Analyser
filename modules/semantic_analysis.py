"""Sentence Transformers based comparison with a user-provided reference."""

from __future__ import annotations

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticAnalysisError(ValueError):
    """Raised when an embedding model cannot compare the provided text."""


def load_sentence_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load a local Sentence Transformer, downloading its weights if necessary."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise SemanticAnalysisError(
            f"Could not load semantic model '{model_name}'. Check the model name and "
            f"internet access for its first download. Details: {exc}"
        ) from exc


def calculate_semantic_similarity(
    transcript: str,
    reference: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model=None,
) -> dict:
    """Return cosine similarity and a 0-100 display value clamped at zero."""
    if not transcript or not transcript.strip():
        raise SemanticAnalysisError("The transcript is empty, so it cannot be compared.")
    if not reference or not reference.strip():
        raise SemanticAnalysisError("Enter a reference explanation to compare against.")
    model = model or load_sentence_model(model_name)
    try:
        embeddings = model.encode(
            [reference.strip(), transcript.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.shape[0] != 2 or embeddings.ndim != 2:
            raise ValueError("The embedding model returned an unexpected result shape.")
        cosine = float(np.dot(embeddings[0], embeddings[1]))
        cosine = float(np.clip(cosine, -1.0, 1.0))
    except Exception as exc:
        raise SemanticAnalysisError(f"Could not compare the two explanations: {exc}") from exc
    return {
        "cosine_similarity": cosine,
        "similarity_percent": float(np.clip(cosine, 0.0, 1.0) * 100.0),
        "model_name": model_name,
    }
