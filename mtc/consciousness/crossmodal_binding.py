"""
Cross-modal binding
===================

Integrates signals from different modalities before the global workspace
competition. The binder scans a candidate list for the best candidate of
each registered modality inside a short time window, checks emotional
congruence, and appends one bound candidate that represents the integrated
percept. The originals stay in the list, so both the parts and the whole
compete (feature integration, Treisman and Gelade 1980; organized percepts,
the recurrent processing indicator of the 2025 update to the consciousness
indicator review).

The pipeline's cycle augmenter (``mtc.pipeline.cycle_augmenter``) builds a
binder and calls ``bind`` on the candidate list before each workspace cycle
when its ``crossmodal_binding`` switch is on (the default).

Modality comes from two registries on ``CrossModalConfig``:
``modality_by_source_module`` (checked first, empty by default) and
``modality_by_content_type``. A candidate whose content type and source
module are in neither registry is unclassified and is never bound. Fewer
than two modalities never bind.

The bound candidate has source ``WorkspaceCandidateSource.INTERNAL`` (an
internal integrative process, not language-model reasoning), hierarchy level
1 (integrative), source module ``crossmodal_binder`` and content type
``crossmodal_bound``; its metadata names the bound candidate ids and
modalities. Strength is a modality bonus plus a congruence bonus plus a
proximity bonus, each a field of the config.

Config: ``CrossModalConfig`` (window, registries, valence word lists and the
strength weights). Persisted state (``to_state_dict``): the bind counters and
the bounded binding history, which holds candidate summaries and is private
data.
"""

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, FrozenSet, List, Optional

from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z']+")

DEFAULT_MODALITY_BY_CONTENT_TYPE: Dict[str, str] = {
    "text": "text",
    "message": "text",
    "thought": "text",
    "audio": "audio",
    "visual": "vision",
}

# Generic English word lists used for a rough valence estimate; documented
# defaults that a deployment can replace.
DEFAULT_POSITIVE_WORDS: FrozenSet[str] = frozenset(
    {
        "happy",
        "joy",
        "laughter",
        "smiling",
        "smile",
        "laughing",
        "excited",
        "cheerful",
        "amused",
        "playful",
        "love",
        "funny",
        "applause",
        "warm",
        "bright",
        "pleasant",
    }
)
DEFAULT_NEGATIVE_WORDS: FrozenSet[str] = frozenset(
    {
        "sad",
        "crying",
        "angry",
        "fear",
        "pain",
        "distress",
        "anxious",
        "frustrated",
        "dark",
        "gloomy",
        "upset",
        "worried",
        "tense",
        "scared",
    }
)


@dataclass
class CrossModalConfig:
    """Registries, window, valence words and strength weights."""

    modality_by_content_type: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MODALITY_BY_CONTENT_TYPE)
    )
    modality_by_source_module: Dict[str, str] = field(default_factory=dict)
    window_seconds: float = 5.0
    positive_words: FrozenSet[str] = DEFAULT_POSITIVE_WORDS
    negative_words: FrozenSet[str] = DEFAULT_NEGATIVE_WORDS
    # Valence spread allowed between candidates of opposite sign is
    # 1 - congruence_threshold.
    congruence_threshold: float = 0.3
    min_binding_strength: float = 0.4
    pair_bonus: float = 0.5
    triple_bonus: float = 0.7
    congruent_bonus: float = 0.2
    incongruent_bonus: float = 0.05
    proximity_weight: float = 0.15
    activation_bonus: float = 0.15
    max_activation: float = 0.95
    novelty_bonus: float = 0.15
    history_size: int = 50


class CrossModalBinder:
    """Adds bound candidates alongside the originals before competition."""

    def __init__(self, config: Optional[CrossModalConfig] = None) -> None:
        self.config = config or CrossModalConfig()
        self._binding_history: Deque[Dict[str, Any]] = deque(
            maxlen=self.config.history_size
        )
        self._total_calls = 0
        self._total_bound = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind(
        self, candidates: List[WorkspaceCandidate], now: Optional[float] = None
    ) -> List[WorkspaceCandidate]:
        """Return the originals plus any bound candidates.

        Only candidates no older than ``window_seconds`` before ``now``
        (default: the wall clock) are considered.
        """
        self._total_calls += 1
        reference = time.time() if now is None else float(now)
        window = self.config.window_seconds
        eligible = [c for c in candidates if reference - c.timestamp <= window]
        bound: List[WorkspaceCandidate] = []
        for group in self._find_bindable_groups(eligible):
            strength = self._binding_strength(group)
            if strength < self.config.min_binding_strength:
                continue
            candidate = self._create_bound_candidate(group, strength)
            bound.append(candidate)
            self._total_bound += 1
            self._binding_history.append(
                {
                    "timestamp": reference,
                    "modalities": [self._modality(c) for c in group],
                    "strength": strength,
                    "summary": candidate.summary,
                }
            )
            logger.debug("Cross-modal bind (%.2f)", strength)
        return list(candidates) + bound

    def get_statistics(self) -> Dict[str, Any]:
        """Counters and a summary of the retained binding history."""
        strengths = [h["strength"] for h in self._binding_history]
        return {
            "total_calls": self._total_calls,
            "total_bound": self._total_bound,
            "history_length": len(self._binding_history),
            "mean_strength": (sum(strengths) / len(strengths) if strengths else None),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for the persistence hub."""
        return {
            "scalars": {
                "total_calls": self._total_calls,
                "total_bound": self._total_bound,
            },
            "arrays": {},
            "histories": {
                "binding_history": [
                    {
                        "timestamp": h["timestamp"],
                        "modalities": list(h["modalities"]),
                        "strength": h["strength"],
                        "summary": h["summary"],
                    }
                    for h in self._binding_history
                ]
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore counters and the bounded history."""
        scalars = state.get("scalars", {})
        self._total_calls = int(scalars.get("total_calls", 0))
        self._total_bound = int(scalars.get("total_bound", 0))
        self._binding_history = deque(
            (
                {
                    "timestamp": float(h["timestamp"]),
                    "modalities": list(h["modalities"]),
                    "strength": float(h["strength"]),
                    "summary": str(h["summary"]),
                }
                for h in state.get("histories", {}).get("binding_history", [])
            ),
            maxlen=self.config.history_size,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _modality(self, candidate: WorkspaceCandidate) -> Optional[str]:
        by_module = self.config.modality_by_source_module.get(candidate.source_module)
        if by_module:
            return by_module
        return self.config.modality_by_content_type.get(candidate.content_type)

    def _valence(self, candidate: WorkspaceCandidate) -> float:
        """Rough valence in [-1, 1] from the summary words and salience."""
        words = _WORD.findall((candidate.summary or "").lower())
        pos = sum(1 for w in words if w in self.config.positive_words)
        neg = sum(1 for w in words if w in self.config.negative_words)
        if pos + neg == 0:
            return 0.0
        raw = (pos - neg) / (pos + neg)
        # Scaling by salience keeps neutral text near zero.
        return raw * max(0.3, candidate.emotional_salience)

    def _temporally_close(self, group: List[WorkspaceCandidate]) -> bool:
        stamps = [c.timestamp for c in group]
        return (max(stamps) - min(stamps)) <= self.config.window_seconds

    def _congruent(self, group: List[WorkspaceCandidate]) -> bool:
        valences = [self._valence(c) for c in group]
        if all(v >= 0 for v in valences) or all(v <= 0 for v in valences):
            return True
        spread = max(valences) - min(valences)
        return spread <= (1.0 - self.config.congruence_threshold)

    def _find_bindable_groups(
        self, candidates: List[WorkspaceCandidate]
    ) -> List[List[WorkspaceCandidate]]:
        buckets: Dict[str, List[WorkspaceCandidate]] = {}
        for candidate in candidates:
            modality = self._modality(candidate)
            if modality is not None:
                buckets.setdefault(modality, []).append(candidate)
        if len(buckets) < 2:
            return []

        def best(modality: str) -> WorkspaceCandidate:
            return max(
                buckets[modality],
                key=lambda c: c.activation_level + c.emotional_salience,
            )

        modalities = sorted(buckets)
        if len(modalities) >= 3:
            triple = [best(m) for m in modalities[:3]]
            if self._temporally_close(triple) and self._congruent(triple):
                return [triple]
        groups: List[List[WorkspaceCandidate]] = []
        for i, first in enumerate(modalities):
            for second in modalities[i + 1 :]:
                pair = [best(first), best(second)]
                if self._temporally_close(pair) and self._congruent(pair):
                    groups.append(pair)
        return groups

    def _binding_strength(self, group: List[WorkspaceCandidate]) -> float:
        cfg = self.config
        modality_bonus = cfg.pair_bonus if len(group) == 2 else cfg.triple_bonus
        signs = {self._valence(c) >= 0 for c in group}
        emotion_bonus = (
            cfg.congruent_bonus if len(signs) == 1 else cfg.incongruent_bonus
        )
        stamps = [c.timestamp for c in group]
        span = max(stamps) - min(stamps)
        proximity = 1.0 - span / cfg.window_seconds if cfg.window_seconds > 0 else 1.0
        proximity_bonus = max(0.0, proximity) * cfg.proximity_weight
        return min(1.0, modality_bonus + emotion_bonus + proximity_bonus)

    def _create_bound_candidate(
        self, group: List[WorkspaceCandidate], strength: float
    ) -> WorkspaceCandidate:
        modalities = [self._modality(c) for c in group]
        parts = [f"{self._modality(c)}: {c.summary[:60]}" for c in group]
        summary = "Bound: " + " + ".join(parts)
        avg_activation = sum(c.activation_level for c in group) / len(group)
        activation = min(
            self.config.max_activation,
            avg_activation + strength * self.config.activation_bonus,
        )
        avg_novelty = sum(c.novelty_score for c in group) / len(group)
        return WorkspaceCandidate(
            content={
                "bound_components": [
                    {
                        "id": c.id,
                        "modality": self._modality(c),
                        "summary": c.summary,
                    }
                    for c in group
                ],
                "binding_strength": strength,
                "modalities": modalities,
            },
            content_type="crossmodal_bound",
            summary=summary[:200],
            source=WorkspaceCandidateSource.INTERNAL,
            source_module="crossmodal_binder",
            activation_level=activation,
            emotional_salience=max(c.emotional_salience for c in group),
            novelty_score=min(1.0, avg_novelty + self.config.novelty_bonus),
            goal_relevance=max(c.goal_relevance for c in group),
            recency_bonus=max(c.recency_bonus for c in group),
            hierarchy_level=1,
            metadata={
                "crossmodal_bound": True,
                "binding_strength": strength,
                "modalities": modalities,
                "component_ids": [c.id for c in group],
            },
        )
