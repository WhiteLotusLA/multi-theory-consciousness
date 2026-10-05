"""
Thalamic Gate - pre-competition salience filter
===============================================

Sits between candidate generation and workspace competition and filters out
noise, so only potentially relevant candidates enter the attention
bottleneck. Three gating mechanisms, loosely modelled on thalamic roles:

1. Arousal gate (reticular nucleus): the activation threshold falls as
   arousal rises, so an alert system lets more through.
2. Habituation filter (pulvinar): content seen recently loses activation
   with every repeat, up to a cap.
3. Relevance filter (mediodorsal nucleus): goal-aligned content gains a
   small boost; content still below a floor is dropped.

The gate is permissive by default, and safety content always passes.
"""

import hashlib
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from mtc.consciousness.enhanced_global_workspace import WorkspaceCandidate

logger = logging.getLogger(__name__)

SAFETY_SOURCE = "safety"


@dataclass
class ThalamicGateConfig:
    """Tuned constants of the thalamic gate.

    The activation threshold is ``arousal_threshold_base +
    arousal_threshold_range * (1 - arousal)``: 0.35 at arousal 0, 0.20 at
    arousal 0.5 and 0.05 at arousal 1 with the defaults.
    """

    arousal_threshold_base: float = 0.05
    arousal_threshold_range: float = 0.30
    # Number of recent content hashes the habituation filter remembers
    habituation_window: int = 50
    # Activation penalty per repeat within the window, and its cap
    habituation_penalty: float = 0.15
    habituation_max_penalty: float = 0.40
    # Boost per unit of goal relevance
    relevance_boost: float = 0.10
    # Effective activation below this floor is gated as irrelevant
    irrelevance_threshold: float = 0.08

    def __post_init__(self):
        if int(self.habituation_window) < 1:
            raise ValueError(
                f"habituation_window must be at least 1, got {self.habituation_window}"
            )
        self.habituation_window = int(self.habituation_window)
        for name in (
            "arousal_threshold_base",
            "arousal_threshold_range",
            "habituation_penalty",
            "habituation_max_penalty",
            "relevance_boost",
            "irrelevance_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
            setattr(self, name, value)


@dataclass
class GatingStats:
    """Cumulative gating statistics for research analysis."""

    total_candidates: int = 0
    total_gated: int = 0
    gated_by_arousal: int = 0
    gated_by_habituation: int = 0
    gated_by_irrelevance: int = 0
    total_safety_bypassed: int = 0

    @property
    def gate_rate(self) -> float:
        """Fraction of all candidates seen that the gate filtered."""
        if self.total_candidates <= 0:
            return 0.0
        return self.total_gated / self.total_candidates

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "total_gated": self.total_gated,
            "gated_by_arousal": self.gated_by_arousal,
            "gated_by_habituation": self.gated_by_habituation,
            "gated_by_irrelevance": self.gated_by_irrelevance,
            "total_safety_bypassed": self.total_safety_bypassed,
            "gate_rate": self.gate_rate,
        }


_STAT_FIELDS = (
    "total_candidates",
    "total_gated",
    "gated_by_arousal",
    "gated_by_habituation",
    "gated_by_irrelevance",
    "total_safety_bypassed",
)


class ThalamicGate:
    """Pre-competition salience filter for the global workspace."""

    def __init__(self, config: Optional[ThalamicGateConfig] = None) -> None:
        self.config = config or ThalamicGateConfig()
        self._recent_hashes: deque = deque(maxlen=self.config.habituation_window)
        self.stats = GatingStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gate(
        self,
        candidates: List["WorkspaceCandidate"],
        arousal: float = 0.5,
        goals: Optional[List[str]] = None,
    ) -> List["WorkspaceCandidate"]:
        """Return the candidates that pass all three filters, in input order.

        Safety candidates always pass and never enter the habituation window.
        """
        if not candidates:
            return []

        goals = goals or []
        cfg = self.config
        threshold = self.arousal_threshold(arousal)
        passed: List["WorkspaceCandidate"] = []
        arousal_gated = habituation_gated = irrelevance_gated = 0
        safety_bypassed = 0

        for candidate in candidates:
            if _source_value(candidate) == SAFETY_SOURCE:
                passed.append(candidate)
                safety_bypassed += 1
                continue

            # 1. Arousal gate
            if candidate.activation_level < threshold:
                arousal_gated += 1
                continue

            # 2. Habituation filter
            content_hash = self._content_hash(candidate)
            penalty = min(
                self._recent_hashes.count(content_hash) * cfg.habituation_penalty,
                cfg.habituation_max_penalty,
            )
            effective = candidate.activation_level - penalty
            if effective < threshold:
                habituation_gated += 1
                continue

            # 3. Relevance filter
            if goals:
                effective += self._goal_relevance(candidate, goals) * (
                    cfg.relevance_boost
                )
            if effective < cfg.irrelevance_threshold:
                irrelevance_gated += 1
                continue

            passed.append(candidate)
            self._recent_hashes.append(content_hash)

        total_gated = arousal_gated + habituation_gated + irrelevance_gated
        self.stats.total_candidates += len(candidates)
        self.stats.total_gated += total_gated
        self.stats.gated_by_arousal += arousal_gated
        self.stats.gated_by_habituation += habituation_gated
        self.stats.gated_by_irrelevance += irrelevance_gated
        self.stats.total_safety_bypassed += safety_bypassed

        if total_gated:
            logger.debug(
                "Thalamic gate: %d/%d filtered (arousal=%d, habituation=%d, "
                "irrelevance=%d) at arousal %.2f, threshold %.3f",
                total_gated,
                len(candidates),
                arousal_gated,
                habituation_gated,
                irrelevance_gated,
                arousal,
                threshold,
            )
        return passed

    def arousal_threshold(self, arousal: float) -> float:
        """Activation threshold for an arousal level (clamped to 0 to 1).

        High arousal gives a low threshold (more gets through); low arousal
        gives a high one (only strong signals pass).
        """
        try:
            value = float(arousal)
        except (TypeError, ValueError):
            value = 0.5
        if not math.isfinite(value):
            value = 0.5
        clamped = max(0.0, min(1.0, value))
        cfg = self.config
        return cfg.arousal_threshold_base + cfg.arousal_threshold_range * (
            1.0 - clamped
        )

    def repeat_count(self, candidate: "WorkspaceCandidate") -> int:
        """How often this candidate's content sits in the habituation window."""
        return self._recent_hashes.count(self._content_hash(candidate))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Statistics as scalars and the habituation window as a history."""
        return {
            "scalars": {name: int(getattr(self.stats, name)) for name in _STAT_FIELDS},
            "arrays": {},
            "histories": {"recent_hashes": list(self._recent_hashes)},
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore statistics and the habituation window."""
        state = state or {}
        s = state.get("scalars", {}) or {}
        h = state.get("histories", {}) or {}
        for name in _STAT_FIELDS:
            if name in s:
                try:
                    setattr(self.stats, name, max(0, int(s[name])))
                except (TypeError, ValueError):
                    logger.warning("Thalamic gate restore: bad %s skipped", name)
        if "recent_hashes" in h:
            self._recent_hashes = deque(
                (str(item) for item in h["recent_hashes"] if isinstance(item, str)),
                maxlen=self.config.habituation_window,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(candidate: "WorkspaceCandidate") -> str:
        """Deterministic hash of a candidate's type and summary."""
        key = f"{candidate.content_type}:{(candidate.summary or '')[:60]}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]

    @staticmethod
    def _goal_relevance(candidate: "WorkspaceCandidate", goals: List[str]) -> float:
        """Goal relevance in [0, 1].

        Trusts a goal relevance already scored upstream; otherwise the share
        of goals whose text appears in the candidate's summary.
        """
        if candidate.goal_relevance > 0.0:
            return min(1.0, float(candidate.goal_relevance))
        if not goals or not candidate.summary:
            return 0.0
        summary_lower = candidate.summary.lower()
        matches = sum(1 for goal in goals if goal.lower() in summary_lower)
        return min(1.0, matches / max(len(goals), 1))


def _source_value(candidate: Any) -> Any:
    source = getattr(candidate, "source", None)
    return getattr(source, "value", source)
