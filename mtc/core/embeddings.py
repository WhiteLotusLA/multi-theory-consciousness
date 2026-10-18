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
rows and carries a ``semantic`` flag: True for a trained model, False for the
hash embedder. A store that compares meanings refuses a non-semantic embedder
unless the caller opts in explicitly.

Config keys read at load time: ``EMBEDDING_MODEL`` (the model name when none is
passed), ``MTC_DEVICE`` through ``mtc.core.device`` (the encoder device) and
``VECTOR_DIMENSIONS`` (the width the loaded model must report). A load failure,
an unavailable device or a width mismatch raises ``EmbeddingUnavailableError``
chained to its cause, is logged once at ERROR with a pre-download hint (a
missing package is not logged: its error names the extra) and is not retried
before ``retry_after_s``; ``status()`` reports the state. No
embedding is ever invented in place of a missing model. No persisted state.
"""

import hashlib
import logging
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from mtc.core.config import get_settings

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


class EmbeddingUnavailableError(RuntimeError):
    """The embedder cannot produce vectors; nothing was embedded or stored."""


class EmbeddingExtraMissingError(EmbeddingUnavailableError, ImportError):
    """The ``embeddings`` extra is not installed (also an ImportError)."""


def _embedding_dimension(model: Any) -> Any:
    """The model's output width, through whichever accessor it has.

    Newer sentence-transformers releases name it ``get_embedding_dimension``
    and flag the older ``get_sentence_embedding_dimension`` as deprecated; the
    dependency has no upper bound, so the new name is used when present and
    the old one otherwise (no version check, no warning).
    """
    accessor = getattr(model, "get_embedding_dimension", None)
    if accessor is None:
        accessor = model.get_sentence_embedding_dimension
    return accessor()


class SentenceTransformerEmbedder:
    """Sentence-transformers model, loaded on the first ``encode`` call."""

    semantic = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        expected_dim: Optional[int] = None,
        retry_after_s: float = 300.0,
    ) -> None:
        self._model_name = model_name
        self._device_argument = device
        self._expected_dim = expected_dim
        self.retry_after_s = float(retry_after_s)
        self.dim = int(expected_dim) if expected_dim else DEFAULT_DIM
        self._model: Optional[Any] = None
        self._device: Optional[str] = None
        self._last_error: Optional[str] = None
        self._failed_at: Optional[float] = None
        self._load_attempts = 0
        self._clock: Callable[[], float] = time.monotonic

    @property
    def model_name(self) -> str:
        """The explicit name, else ``EMBEDDING_MODEL`` read at call time."""
        if self._model_name:
            return self._model_name
        return get_settings().EMBEDDING_MODEL

    def status(self) -> Dict[str, Any]:
        """Return the load state; ``available`` is False after a failed load."""
        return {
            "backend": "sentence-transformers",
            "available": self._last_error is None,
            "loaded": self._model is not None,
            "model_name": self.model_name,
            "dim": self.dim,
            "device": self._device,
            "last_error": self._last_error,
            "load_attempts": self._load_attempts,
        }

    def _fail(
        self, message: str, cause: Optional[BaseException], extra: bool = False
    ) -> EmbeddingUnavailableError:
        self._model = None
        self._last_error = message
        self._failed_at = self._clock()
        if not extra:
            # A missing package is reported by the raised error that names the
            # extra; the caller owns the one warning about it.
            logger.error(
                "embedder unavailable: %s (pre-download the model with "
                "sentence_transformers.SentenceTransformer(%r) while online, or "
                "check MTC_DEVICE and VECTOR_DIMENSIONS)",
                message,
                self.model_name,
            )
        cls = EmbeddingExtraMissingError if extra else EmbeddingUnavailableError
        error = cls(message)
        error.__cause__ = cause
        return error

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._failed_at is not None:
            waited = self._clock() - self._failed_at
            if waited < self.retry_after_s:
                raise EmbeddingUnavailableError(
                    f"embedder unavailable ({self._last_error}); will retry "
                    f"loading in {self.retry_after_s - waited:.0f} s"
                )
        self._load_attempts += 1
        name = self.model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise self._fail(
                "sentence-transformers is not installed; install the "
                "'embeddings' extra: pip install "
                "multi-theory-consciousness[embeddings]",
                exc,
                extra=True,
            ) from exc
        from mtc.core.device import DeviceUnavailableError, resolve_device_name

        try:
            device = resolve_device_name(self._device_argument)
        except DeviceUnavailableError as exc:
            raise self._fail(str(exc), exc) from exc
        try:
            model = SentenceTransformer(name, device=device)
            reported = _embedding_dimension(model)
        except Exception as exc:
            raise self._fail(f"model {name!r} failed to load: {exc}", exc) from exc
        expected = (
            int(self._expected_dim)
            if self._expected_dim
            else get_settings().VECTOR_DIMENSIONS
        )
        if reported and int(reported) != expected:
            raise self._fail(
                f"model {name!r} (EMBEDDING_MODEL) produces {int(reported)} "
                f"dimensions but VECTOR_DIMENSIONS is {expected}; change one "
                "of the two keys",
                None,
            )
        self._model = model
        self._device = device
        self._last_error = None
        self._failed_at = None
        if reported:
            self.dim = int(reported)
        return model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        try:
            vectors = model.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception as exc:
            raise self._fail(f"encoding failed: {exc}", exc) from exc
        return np.asarray(vectors, dtype=float).reshape(len(texts), self.dim)


class HashEmbedder:
    """Deterministic offline embedder; lexical overlap only, not semantic."""

    semantic = False

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
