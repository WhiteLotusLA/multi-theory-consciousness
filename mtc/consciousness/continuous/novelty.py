"""
Thought Novelty
===============

Measures how new a background thought is against the most recent thoughts, so
the continuous thought machine can (1) store a near-duplicate as an access of
the earlier thought instead of a new document and (2) let a thought surface
only when it is novel enough.

``NoveltyFilter`` keeps a ring buffer of the last ``cache_size`` thought
embeddings in numpy and compares each new thought by cosine similarity:

- a similarity at or above ``dedup_threshold`` (0.85) makes the thought a
  near-duplicate; the result names the cached text it matched and the thought
  is not added to the cache;
- otherwise the thought is unique and joins the cache (the oldest entry is
  overwritten once the buffer is full);
- ``express_allowed`` is True only for a unique thought whose novelty
  (one minus the similarity) is at least ``express_threshold`` (0.5).

The embedder is injected (``mtc.core.embeddings``). Its ``dim`` sets the
buffer width; a lazily loaded model whose first encoding reports another width
resets it while nothing is cached yet, and a later mismatch raises. Without an embedder nothing can be measured:
``find_near_duplicate`` returns None and ``express_allowed`` is False, so every
thought stays quiet and ``status()["embedder"]`` reads ``none``. An embedder
whose ``semantic`` flag is missing or false (the hash embedder) measures
lexical overlap only; it still detects near-identical texts, but its novelty is
not evidence of new meaning, so expression stays off unless the caller passes
``allow_non_semantic=True``, and ``status()`` reports ``non_semantic`` either
way. An embedder failure propagates to the caller and leaves the cache as it
was.

``AcceleratedNoveltyFilter`` holds the buffer as a torch tensor on the device
``mtc.core.device`` resolves from ``MTC_DEVICE`` (or the ``device`` argument);
torch is imported when the filter is built, never at module import. Results
match the numpy path within float32 precision.

The cached texts are private thought content at runtime and live only in
process memory; nothing is persisted. No config keys of its own: the machine
passes the ``novelty`` group of ``config/ctm/ctm.yaml``.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["AcceleratedNoveltyFilter", "NoveltyFilter", "NoveltyResult"]


@dataclass(frozen=True)
class NoveltyResult:
    """One measurement: unique or not, the best similarity and its match.

    ``similarity`` is the highest cosine similarity to a cached thought (0.0
    when the cache was empty); ``matched_text`` is the cached text of a
    near-duplicate and None for a unique thought.
    """

    is_unique: bool
    similarity: float
    matched_text: Optional[str] = None

    @property
    def novelty(self) -> float:
        """One minus the similarity."""
        return 1.0 - self.similarity


def _check_threshold(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1]")
    return number


class NoveltyFilter:
    """Cosine near-duplicate check over a ring buffer of recent thoughts."""

    backend = "numpy"

    def __init__(
        self,
        embedder: Optional[Any],
        cache_size: int = 50,
        dedup_threshold: float = 0.85,
        express_threshold: float = 0.5,
        *,
        allow_non_semantic: bool = False,
    ) -> None:
        if int(cache_size) < 1:
            raise ValueError("cache_size must be at least 1")
        self.embedder = embedder
        self.cache_size = int(cache_size)
        self.dedup_threshold = _check_threshold("dedup_threshold", dedup_threshold)
        self.express_threshold = _check_threshold(
            "express_threshold", express_threshold
        )
        self.allow_non_semantic = bool(allow_non_semantic)
        self.dim: Optional[int] = None
        if embedder is not None:
            dim = getattr(embedder, "dim", None)
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
                raise ValueError(
                    "the embedder must report a positive integer dim; the novelty "
                    "cache width comes from it"
                )
            self.dim = dim
        self._texts: List[str] = []
        self._size = 0
        self._write = 0
        self.checked = 0
        self.duplicates = 0
        self._allocate()

    # -- buffer -------------------------------------------------------------

    def _allocate(self) -> None:
        width = self.dim or 1
        self._buffer = np.zeros((self.cache_size, width), dtype=float)

    def _store(self, row: int, vector: Any) -> None:
        self._buffer[row] = vector

    def _similarities(self, vector: Any) -> np.ndarray:
        return self._buffer[: self._size] @ vector

    def _to_vector(self, raw: Any) -> Any:
        vector = np.asarray(raw, dtype=float).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"the embedder returned {vector.shape[0]} values but reports "
                f"dim {self.dim}"
            )
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("the embedder returned a zero or non-finite vector")
        return vector / norm

    def _add(self, text: str, vector: Any) -> None:
        self._store(self._write, vector)
        if self._size < self.cache_size:
            self._texts.append(text)
            self._size += 1
        else:
            self._texts[self._write] = text
        self._write = (self._write + 1) % self.cache_size

    # -- public -------------------------------------------------------------

    @property
    def embedder_kind(self) -> str:
        """``none``, ``semantic`` or ``non_semantic``."""
        if self.embedder is None:
            return "none"
        if getattr(self.embedder, "semantic", False):
            return "semantic"
        return "non_semantic"

    @property
    def expression_possible(self) -> bool:
        """Whether any measurement could allow a thought to surface."""
        kind = self.embedder_kind
        return kind == "semantic" or (
            kind == "non_semantic" and self.allow_non_semantic
        )

    def find_near_duplicate(self, text: str) -> Optional[NoveltyResult]:
        """Measure ``text`` against the cache; None without an embedder.

        A unique text joins the cache; a near-duplicate does not. An embedder
        failure propagates and the cache is unchanged.
        """
        if self.embedder is None:
            return None
        encoded = np.asarray(self.embedder.encode([text]), dtype=float)
        if encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError("the embedder must return one row per text")
        if encoded.shape[1] != self.dim and self._size == 0:
            # a lazily loaded model reports its true width on first use
            self.dim = int(encoded.shape[1])
            self._allocate()
        vector = self._to_vector(encoded[0])
        self.checked += 1
        if self._size == 0:
            self._add(text, vector)
            return NoveltyResult(is_unique=True, similarity=0.0, matched_text=None)
        similarities = np.asarray(self._similarities(vector), dtype=float)
        index = int(np.argmax(similarities))
        similarity = float(np.clip(similarities[index], -1.0, 1.0))
        if similarity >= self.dedup_threshold:
            self.duplicates += 1
            return NoveltyResult(
                is_unique=False,
                similarity=similarity,
                matched_text=self._texts[index],
            )
        self._add(text, vector)
        return NoveltyResult(is_unique=True, similarity=similarity, matched_text=None)

    def express_allowed(self, result: Optional[NoveltyResult]) -> bool:
        """True for a unique, novel enough thought from a usable embedder."""
        if result is None or not self.expression_possible:
            return False
        if not result.is_unique:
            return False
        return result.novelty >= self.express_threshold

    def cached_texts(self) -> List[str]:
        """The cached texts in slot order (private thought content)."""
        return list(self._texts[: self._size])

    def clear(self) -> None:
        """Empty the cache (the buffer stays allocated)."""
        self._texts.clear()
        self._size = 0
        self._write = 0

    def __len__(self) -> int:
        return self._size

    def status(self) -> Dict[str, Any]:
        """Return the embedder kind, thresholds, cache use and counts."""
        return {
            "embedder": self.embedder_kind,
            "non_semantic_opt_in": self.allow_non_semantic,
            "expression_possible": self.expression_possible,
            "backend": self.backend,
            "device": self._device_name(),
            "dim": self.dim,
            "cache_size": self.cache_size,
            "cached": self._size,
            "dedup_threshold": self.dedup_threshold,
            "express_threshold": self.express_threshold,
            "checked": self.checked,
            "duplicates": self.duplicates,
        }

    def _device_name(self) -> str:
        return "cpu"


class AcceleratedNoveltyFilter(NoveltyFilter):
    """The same filter with its buffer on the device ``MTC_DEVICE`` selects."""

    backend = "torch"

    def __init__(
        self,
        embedder: Optional[Any],
        cache_size: int = 50,
        dedup_threshold: float = 0.85,
        express_threshold: float = 0.5,
        *,
        allow_non_semantic: bool = False,
        device: Optional[str] = None,
        settings: Any = None,
    ) -> None:
        from mtc.core.device import resolve_device
        from mtc.memory.clients import require_client

        self._torch = require_client("torch", "neural")
        self.device = resolve_device(device, settings=settings)
        super().__init__(
            embedder,
            cache_size=cache_size,
            dedup_threshold=dedup_threshold,
            express_threshold=express_threshold,
            allow_non_semantic=allow_non_semantic,
        )

    def _allocate(self) -> None:
        width = self.dim or 1
        self._buffer = self._torch.zeros(
            (self.cache_size, width), dtype=self._torch.float32, device=self.device
        )

    def _to_vector(self, raw: Any) -> Any:
        vector = super()._to_vector(raw)
        return self._torch.as_tensor(
            vector, dtype=self._torch.float32, device=self.device
        )

    def _store(self, row: int, vector: Any) -> None:
        self._buffer[row] = vector

    def _similarities(self, vector: Any) -> np.ndarray:
        scores = self._buffer[: self._size] @ vector
        return scores.detach().to("cpu").numpy()

    def _device_name(self) -> str:
        return str(self.device)
