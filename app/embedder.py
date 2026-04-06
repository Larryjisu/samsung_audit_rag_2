from __future__ import annotations

import sys

from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    _MODEL_CACHE: dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        cached = self._MODEL_CACHE.get(self.model_name)
        if cached is None:
            cached = SentenceTransformer(self.model_name)
            self._MODEL_CACHE[self.model_name] = cached
        self.model = cached

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1 and sys.stdout.isatty(),
        )
        return vectors.tolist()
