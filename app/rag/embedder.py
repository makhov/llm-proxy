from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings

_model: SentenceTransformer | None = None


def init_embedder(settings: Settings) -> SentenceTransformer:
    global _model
    _model = SentenceTransformer(settings.rag_embedding_model)
    return _model


def get_embedder() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Embedder not initialized")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
