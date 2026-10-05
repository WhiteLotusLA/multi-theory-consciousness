"""
Affect estimator
================

A text-grounded affect estimate computed from text the system produces,
receives or thinks. It is a shadow instrument: its output goes to the shadow
log (``mtc.consciousness.affect_shadow``) for comparison with the PAD model,
and nothing here narrates or feeds a prompt.

Method: embedding-anchor estimation. The text is embedded with an injected
``Embedder`` (default: the sentence-transformer wrapper of
``mtc.core.embeddings``, which needs the ``embeddings`` extra). Each
dimension is the difference between the mean cosine similarity to two curated
anchor pole-sets, squashed with ``tanh(6 x)`` to [-1, 1]. The anchors are
general-register short declarative sentences; ``ANCHORS_VERSION`` changes with
any edit to them. The score moves with what is said and can go negative.

Without an embedder and without the extra, ``estimate`` returns None and logs
once that the ``embeddings`` extra is missing; there is never a fallback
score. Empty text also returns None: nothing to measure is absent, not
neutral.

An optional ``annotation_prior`` hook maps an annotation string (for example
an external affect label attached to a received message) to a
``(valence, arousal)`` pair. When it returns one for a received text, the
estimate is blended with it at ``prior_weight``. No table of annotation names
ships here; the caller supplies the hook.

Config: constructor arguments only. Persisted state: none.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mtc.core.embeddings import Embedder, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

ANCHORS_VERSION = "v1"

# Maps an anchor-set similarity difference of about 0.15 (clear affect for a
# small sentence model) to tanh(0.9), roughly 0.72, so clear affect uses the
# range without saturating ambiguous text.
SCORE_SCALE_K = 6.0

VALENCE_POSITIVE: Tuple[str, ...] = (
    "I feel joyful and deeply happy.",
    "This is wonderful news and it delights me.",
    "I feel loved, safe, and warm inside.",
    "Everything feels bright and full of hope.",
    "I am grateful and content with today.",
    "That made me smile with real pleasure.",
)
VALENCE_NEGATIVE: Tuple[str, ...] = (
    "I feel sad and heavy with grief.",
    "This is painful and it hurts deeply.",
    "I feel afraid and everything seems wrong.",
    "I am lonely, lost, and miserable.",
    "That filled me with dread and despair.",
    "Everything feels dark and hopeless.",
)
AROUSAL_HIGH: Tuple[str, ...] = (
    "My heart is racing with excitement!",
    "Alert and electrified, everything is urgent right now.",
    "I can't sit still, the energy is overwhelming!",
    "Panic surges through me, fast and loud.",
    "Thrilled and buzzing, wide awake with intensity.",
)
AROUSAL_LOW: Tuple[str, ...] = (
    "Resting quietly, calm and still.",
    "A slow, peaceful evening with nothing urgent.",
    "Drowsy and settled, drifting gently.",
    "Everything is tranquil, soft, and unhurried.",
    "Breathing slowly in complete stillness.",
)


@dataclass
class AffectEstimate:
    """One estimate; valence and arousal are in [-1, 1]."""

    valence: float
    arousal: float
    confidence: float
    source_kind: str  # "response", "thought" or "received"
    computed_from: List[str] = field(default_factory=list)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


class AffectEstimator:
    """Embedding-anchor affect estimation over text streams."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        annotation_prior: Optional[
            Callable[[str], Optional[Tuple[float, float]]]
        ] = None,
        prior_weight: float = 0.3,
    ) -> None:
        self._embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self._annotation_prior = annotation_prior
        self.prior_weight = float(prior_weight)
        self._anchors: Optional[Dict[str, np.ndarray]] = None
        self._warned_unavailable = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _model_label(self) -> str:
        return str(getattr(self._embedder, "model_name", type(self._embedder).__name__))

    def _get_anchors(self) -> Dict[str, np.ndarray]:
        if self._anchors is None:

            def encode(sentences: Sequence[str]) -> np.ndarray:
                return _unit_rows(self._embedder.encode(list(sentences)))

            self._anchors = {
                "valence_pos": encode(VALENCE_POSITIVE),
                "valence_neg": encode(VALENCE_NEGATIVE),
                "arousal_high": encode(AROUSAL_HIGH),
                "arousal_low": encode(AROUSAL_LOW),
            }
        return self._anchors

    def _warn_once(self, message: str) -> None:
        if not self._warned_unavailable:
            self._warned_unavailable = True
            logger.warning(message)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        text: str,
        source_kind: str = "response",
        annotation: Optional[str] = None,
    ) -> Optional[AffectEstimate]:
        """Estimate affect from one text; None when it cannot be measured."""
        if not text or not text.strip():
            return None
        try:
            anchors = self._get_anchors()
            vec = _unit_rows(self._embedder.encode([text]))[0]
        except ImportError:
            self._warn_once(
                "affect estimator unavailable: the 'embeddings' extra "
                "(sentence-transformers) is not installed"
            )
            return None
        except Exception as exc:
            self._warn_once(f"affect estimator unavailable: embedding failed ({exc})")
            return None
        if not np.any(vec):
            return None

        def pole_score(pos_key: str, neg_key: str) -> float:
            sim_pos = float(np.mean(anchors[pos_key] @ vec))
            sim_neg = float(np.mean(anchors[neg_key] @ vec))
            return float(np.tanh(SCORE_SCALE_K * (sim_pos - sim_neg)))

        valence = pole_score("valence_pos", "valence_neg")
        arousal = pole_score("arousal_high", "arousal_low")
        computed_from = [
            f"text:{source_kind}",
            f"model:{self._model_label()}",
            f"anchors:{ANCHORS_VERSION}",
        ]

        if source_kind == "received" and annotation and self._annotation_prior:
            key = annotation.strip().lower()
            prior = self._annotation_prior(key)
            if prior is not None:
                w = self.prior_weight
                valence = (1.0 - w) * valence + w * float(prior[0])
                arousal = (1.0 - w) * arousal + w * float(prior[1])
                computed_from.append(f"annotation:{key}")

        # Confidence: larger anchor margins are more reliable and very short
        # texts less so. The 0.2 floor means only that an estimate exists.
        n_tokens = len(re.findall(r"\w+", text))
        margin = max(abs(valence), abs(arousal))
        length_factor = min(1.0, n_tokens / 20.0)
        confidence = float(np.clip(0.2 + 0.6 * margin + 0.2 * length_factor, 0.0, 1.0))
        return AffectEstimate(
            valence=float(np.clip(valence, -1.0, 1.0)),
            arousal=float(np.clip(arousal, -1.0, 1.0)),
            confidence=confidence,
            source_kind=source_kind,
            computed_from=computed_from,
        )
