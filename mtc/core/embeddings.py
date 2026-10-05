"""
Embedding seam
==============

A small protocol for text embedders so that modules which compare texts
(the affect estimator, the drift sampler) take an injectable embedder and
never import a model library at import time.

Two implementations ship here:

- ``SentenceTransformerEmbedder`` wraps a sentence-transformers model. The
  package is the optional ``embeddings`` extra (``pip install
  multi-theory-consciousness[embeddings]``) and is imported lazily on the
  first ``encode`` call; the model weights download on first use. A missing
  package raises an ``ImportError`` that names the extra.
- ``HashEmbedder`` is deterministic, offline and built on ``hashlib``. It
  hashes word tokens and character trigrams into a fixed number of buckets.
  It is NOT semantic: similar meanings do not land near each other, only
  shared tokens do. Use it for tests and offline runs, never as evidence of
  semantic similarity. A hash seeded from Python's built-in ``hash`` would
  differ across processes, which is why ``hashlib`` is used.

Every embedder returns an array of shape ``(n, dim)`` with L2-normalized
rows. No config keys and no persisted state.
"""

import hashlib
import logging
import re
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384

_TOKEN = re.compile(r"[a-z0-9']+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into unit-norm vectors of a fixed width."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) array with L2-normalized rows."""
        ...


class SentenceTransformerEmbedder:
    """Sentence-transformers model, loaded on the first ``encode`` call."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.dim = DEFAULT_DIM
        self._model: Optional[Any] = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed; install the "
                    "'embeddings' extra: pip install "
                    "multi-theory-consciousness[embeddings]"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            reported = self._model.get_sentence_embedding_dimension()
            if reported:
                self.dim = int(reported)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=float).reshape(len(texts), self.dim)


class HashEmbedder:
    """Deterministic offline embedder; lexical overlap only, not semantic."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = int(dim)

    def _features(self, text: str) -> list:
        tokens = _TOKEN.findall(text.lower())
        features = list(tokens)
        for token in tokens:
            padded = f"#{token}#"
            features.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return features or ["<empty>"]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=float)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dim
                sign = 1.0 if digest[8] & 1 else -1.0
                out[row, bucket] += sign
            norm = np.linalg.norm(out[row])
            if norm == 0.0:
                out[row, 0] = 1.0
            else:
                out[row] /= norm
        return out
