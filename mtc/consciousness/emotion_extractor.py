"""
Emotion extractor
=================

Maps measured consciousness data to a named emotional state. Inputs:

- ``state.damasio_state`` (the Damasio layers' output): the primordial
  feelings ``pleasure_pain`` (valence), ``arousal`` and ``vitality`` from
  ``protoself.primordial_feelings``, and the mean valence and intensity of
  ``core.somatic_markers``;
- ``state.protoself_stability`` and ``state.epistemic_depth``;
- an optional ``drives`` callable returning the ``get_drive_state()`` shape
  (drive name to a dict with ``urgency``).

``_extract_raw`` reads them and ``_classify`` applies relative rules against
a baseline of per-dimension means and spreads. A missing input is None and
disables every rule that needs it; it is never replaced by a default value.

Baselines come from an ``EmotionAveragesProvider`` (default
``InMemoryEmotionAverages``; a Redis-backed manager can satisfy the same
protocol, and this module imports no network client). Averages are a flat
mapping with ``<dimension>`` for the mean and ``<dimension>_std`` for the
spread. Until the provider has ``min_baseline_samples`` samples the neutral
prior applies (valence 0.0, arousal 0.5, vitality 0.5, every drive urgency
0.0, spread ``PRIOR_STD``); the result carries ``baseline: "prior"`` and its
confidence is scaled by samples over ``min_baseline_samples``. Afterwards
the running averages apply and the result says ``baseline: "running"``.
Running spreads are floored at ``MIN_STD`` so a constant dimension cannot
make a rule fire on any wobble.

The 20 labels of ``DEFAULT_LABELS`` are the default vocabulary; a label the
rules choose that is outside the configured ``EmotionVocabulary`` is not a
candidate, and the label is None when no candidate remains. Subtitles exist
only when the vocabulary supplies them.

Config: constructor arguments (provider, drives, vocabulary,
``min_baseline_samples``). Persisted state (``to_state_dict``): the default
in-memory averages; another provider persists itself. The Phase 6
orchestrator registers the module with the hub as ``emotion_extractor``.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS: Tuple[str, ...] = (
    "joyful",
    "excited",
    "loving",
    "playful",
    "curious",
    "wondering",
    "learning",
    "fascinated",
    "thoughtful",
    "calm",
    "peaceful",
    "dreaming",
    "confused",
    "frustrated",
    "uneasy",
    "sad",
    "overwhelmed",
    "withdrawn",
    "neutral",
    "resting",
)

PRIOR_MEANS: Dict[str, float] = {
    "valence": 0.0,
    "arousal": 0.5,
    "vitality": 0.5,
    "drive_curiosity": 0.0,
    "drive_social_connection": 0.0,
    "drive_attention_budget": 0.0,
    "drive_coherence": 0.0,
    "drive_safety": 0.0,
}
PRIOR_STD = 0.25
MIN_STD = 0.05

DRIVE_NAMES = (
    "curiosity",
    "social_connection",
    "attention_budget",
    "coherence",
    "safety",
)


@runtime_checkable
class EmotionAveragesProvider(Protocol):
    """Running per-session averages of the raw dimensions."""

    async def get_averages(self, session_id: str) -> Optional[Dict[str, float]]:
        """Flat ``{dim: mean, dim_std: spread}``; None when nothing recorded."""
        ...

    async def record(self, session_id: str, sample: Dict[str, float]) -> None:
        """Add one sample; dimensions absent from it are left untouched."""
        ...

    async def sample_count(self, session_id: str) -> int:
        """Number of samples recorded for the session."""
        ...


class InMemoryEmotionAverages:
    """Welford running mean and variance per session and dimension."""

    def __init__(self) -> None:
        # session -> {"count": int, "dims": {dim: [n, mean, m2]}}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    async def get_averages(self, session_id: str) -> Optional[Dict[str, float]]:
        entry = self._sessions.get(session_id)
        if not entry or not entry["dims"]:
            return None
        out: Dict[str, float] = {}
        for dim, (n, mean, m2) in entry["dims"].items():
            out[dim] = mean
            out[f"{dim}_std"] = math.sqrt(m2 / n) if n > 1 else 0.0
        return out

    async def record(self, session_id: str, sample: Dict[str, float]) -> None:
        usable = {
            dim: float(value)
            for dim, value in sample.items()
            if value is not None and math.isfinite(float(value))
        }
        if not usable:
            return  # an empty sample is not a measurement; do not count it
        entry = self._sessions.setdefault(session_id, {"count": 0, "dims": {}})
        entry["count"] += 1
        for dim, value in usable.items():
            n, mean, m2 = entry["dims"].get(dim, [0, 0.0, 0.0])
            n += 1
            delta = float(value) - mean
            mean += delta / n
            m2 += delta * (float(value) - mean)
            entry["dims"][dim] = [n, mean, m2]

    async def sample_count(self, session_id: str) -> int:
        entry = self._sessions.get(session_id)
        return int(entry["count"]) if entry else 0

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "scalars": {},
            "arrays": {},
            "histories": {
                "sessions": [
                    {
                        "session": sid,
                        "count": entry["count"],
                        "dims": {k: list(v) for k, v in entry["dims"].items()},
                    }
                    for sid, entry in self._sessions.items()
                ]
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        self._sessions = {}
        for item in state.get("histories", {}).get("sessions", []):
            self._sessions[str(item["session"])] = {
                "count": int(item["count"]),
                "dims": {
                    str(k): [int(v[0]), float(v[1]), float(v[2])]
                    for k, v in item["dims"].items()
                },
            }


@dataclass
class EmotionVocabulary:
    """The labels the extractor may return and optional display text."""

    labels: Tuple[str, ...] = DEFAULT_LABELS
    subtitles: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExtractedEmotion:
    """One extraction: label, runner-up, confidence and baseline kind."""

    label: Optional[str]
    secondary: Optional[str]
    confidence: float
    baseline: str  # "prior" or "running"
    raw: Dict[str, Any]
    subtitle: Optional[str] = None


class EmotionExtractor:
    """Classifies measured state into the configured emotion vocabulary."""

    def __init__(
        self,
        averages: Optional[EmotionAveragesProvider] = None,
        drives: Optional[Callable[[], Dict[str, Dict[str, float]]]] = None,
        vocabulary: Optional[EmotionVocabulary] = None,
        min_baseline_samples: int = 20,
    ) -> None:
        self._averages: EmotionAveragesProvider = averages or InMemoryEmotionAverages()
        self._drives = drives
        self.vocabulary = vocabulary or EmotionVocabulary()
        self.min_baseline_samples = max(1, int(min_baseline_samples))

    # ------------------------------------------------------------------
    # Raw extraction
    # ------------------------------------------------------------------

    def _extract_raw(self, state: Any) -> Dict[str, Any]:
        """Read the raw dimensions; anything unavailable is None."""
        valence = arousal = vitality = None
        somatic_valence = somatic_intensity = None
        damasio = getattr(state, "damasio_state", None)
        if isinstance(damasio, dict):
            feelings = (damasio.get("protoself") or {}).get("primordial_feelings") or {}
            valence = feelings.get("pleasure_pain")
            arousal = feelings.get("arousal")
            vitality = feelings.get("vitality")
            markers = (damasio.get("core") or {}).get("somatic_markers") or []
            if markers:
                somatic_valence = sum(m.get("valence", 0.0) for m in markers) / len(
                    markers
                )
                somatic_intensity = sum(m.get("intensity", 0.0) for m in markers) / len(
                    markers
                )

        drive_state: Optional[Dict[str, Dict[str, float]]] = None
        if self._drives is not None:
            try:
                drive_state = self._drives()
            except Exception as exc:
                logger.warning(
                    "drive state read failed (%s); drive inputs are missing "
                    "this cycle",
                    exc,
                )

        raw: Dict[str, Any] = {
            "valence": valence,
            "arousal": arousal,
            "vitality": vitality,
            "protoself_stability": getattr(state, "protoself_stability", None),
            "epistemic_depth": getattr(state, "epistemic_depth", None),
            "somatic_valence_avg": somatic_valence,
            "somatic_intensity_avg": somatic_intensity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for name in DRIVE_NAMES:
            urgency = None
            if drive_state is not None:
                urgency = (drive_state.get(name) or {}).get("urgency")
            raw[f"drive_{name}"] = urgency
        return raw

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
        self, raw: Dict[str, Any], averages: Dict[str, Dict[str, float]]
    ) -> Tuple[Optional[str], Optional[str], float]:
        """Return (dominant, secondary, confidence) from relative rules.

        ``averages`` maps a dimension to ``{"mean": m, "stddev": s}``.
        """

        def known(dim: str) -> bool:
            return raw.get(dim) is not None

        if not any(known(dim) for dim in PRIOR_MEANS):
            # No measured dimension: insufficient data, never a confident label.
            return None, None, 0.0

        def mean(dim: str) -> float:
            return averages.get(dim, {}).get("mean", PRIOR_MEANS.get(dim, 0.0))

        def std(dim: str) -> float:
            return averages.get(dim, {}).get("stddev", PRIOR_STD)

        def above(dim: str) -> bool:
            return known(dim) and raw[dim] > mean(dim)

        def at_or_below(dim: str) -> bool:
            return known(dim) and raw[dim] <= mean(dim)

        def far_below(dim: str) -> bool:
            return known(dim) and raw[dim] < mean(dim) - std(dim)

        drive_urgency = {
            name: raw[f"drive_{name}"] for name in DRIVE_NAMES if known(f"drive_{name}")
        }
        highest = max(drive_urgency, key=drive_urgency.get) if drive_urgency else None

        valence = raw.get("valence")
        arousal = raw.get("arousal")
        vitality = raw.get("vitality")
        epistemic = raw.get("epistemic_depth")

        candidates: List[Tuple[str, float]] = []

        if highest == "safety" and above("drive_safety"):
            candidates.append(("uneasy", raw["drive_safety"] + 0.5))

        if far_below("valence") and at_or_below("arousal"):
            if valence < -0.5:
                candidates.append(("withdrawn", abs(valence)))
            else:
                candidates.append(("sad", abs(valence) * 0.8))

        if far_below("valence") and above("arousal"):
            if above("drive_attention_budget") and raw["drive_attention_budget"] > 0.6:
                candidates.append(("overwhelmed", arousal * 0.9))
            else:
                candidates.append(("frustrated", arousal * 0.8))

        if highest == "curiosity" and above("drive_curiosity"):
            if above("arousal") and arousal > mean("arousal") + 0.15:
                candidates.append(("fascinated", raw["drive_curiosity"]))
            else:
                candidates.append(("curious", raw["drive_curiosity"] * 0.9))

        if highest == "social_connection" and above("drive_social_connection"):
            candidates.append(("loving", raw["drive_social_connection"]))

        if epistemic is not None and epistemic >= 3:
            if above("drive_curiosity"):
                candidates.append(("wondering", 0.7))
            else:
                candidates.append(("thoughtful", 0.65))

        if (
            highest == "coherence"
            and above("drive_coherence")
            and at_or_below("arousal")
        ):
            candidates.append(("learning", raw["drive_coherence"] * 0.8))

        if known("valence") and valence > 0 and not far_below("valence"):
            if above("arousal") and above("vitality"):
                if vitality > mean("vitality") + 0.15:
                    candidates.append(("playful", valence * 0.7))
                else:
                    candidates.append(("excited", valence * 0.65))
            elif above("arousal"):
                candidates.append(("joyful", valence * 0.6))
            elif at_or_below("arousal") and valence > 0.1:
                if arousal < 0.2:
                    candidates.append(("peaceful", valence * 0.55))
                else:
                    candidates.append(("calm", valence * 0.5))

        # Above the running average counts even when the value is negative.
        if above("valence") and not far_below("valence"):
            if above("arousal") and above("vitality"):
                candidates.append(("curious", 0.55))
            elif above("arousal"):
                candidates.append(("thoughtful", 0.50))
            elif above("vitality"):
                candidates.append(("calm", 0.45))

        if highest == "attention_budget" and above("drive_attention_budget"):
            if above("arousal"):
                candidates.append(("learning", 0.45))
            else:
                candidates.append(("thoughtful", 0.40))

        all_low = (
            at_or_below("arousal")
            and at_or_below("vitality")
            and at_or_below("drive_curiosity")
            and at_or_below("drive_social_connection")
        )
        if all_low and arousal < 0.15:
            if vitality < 0.3:
                candidates.append(("dreaming", 0.4))
            else:
                candidates.append(("resting", 0.35))

        candidates.append(("neutral", 0.1))

        allowed = set(self.vocabulary.labels)
        candidates = [c for c in candidates if c[0] in allowed]
        if not candidates:
            return None, None, 0.0
        candidates.sort(key=lambda c: c[1], reverse=True)
        dominant, dominant_score = candidates[0]
        secondary: Optional[str] = None
        secondary_score = 0.0
        for name, score in candidates[1:]:
            if name != dominant:
                secondary, secondary_score = name, score
                break
        if secondary_score > 0:
            confidence = min(
                1.0, (dominant_score - secondary_score) / max(dominant_score, 0.01)
            )
        else:
            confidence = 1.0
        return dominant, secondary, confidence

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def _baseline(
        self, session_id: str
    ) -> Tuple[Dict[str, Dict[str, float]], str, int]:
        """Return (averages, "prior" or "running", samples seen)."""
        count = 0
        flat: Optional[Dict[str, float]] = None
        try:
            count = await self._averages.sample_count(session_id)
            flat = await self._averages.get_averages(session_id)
        except Exception as exc:
            logger.warning("emotion averages unavailable (%s); using the prior", exc)
            count, flat = 0, None
        if flat is None or count < self.min_baseline_samples:
            return {}, "prior", count
        averages: Dict[str, Dict[str, float]] = {}
        for dim in PRIOR_MEANS:
            if dim in flat:
                averages[dim] = {
                    "mean": float(flat[dim]),
                    "stddev": max(MIN_STD, float(flat.get(f"{dim}_std", PRIOR_STD))),
                }
        return averages, "running", count

    async def extract(
        self, state: Any, session_id: str = "default"
    ) -> ExtractedEmotion:
        """Read, classify and record one sample for the session."""
        raw = self._extract_raw(state)
        averages, kind, count = await self._baseline(session_id)
        label, secondary, confidence = self._classify(raw, averages)
        if kind == "prior":
            confidence *= min(1.0, count / self.min_baseline_samples)
        try:
            sample = {dim: raw[dim] for dim in PRIOR_MEANS if raw.get(dim) is not None}
            await self._averages.record(session_id, sample)
        except Exception as exc:
            logger.warning("emotion sample not recorded (%s)", exc)
        return ExtractedEmotion(
            label=label,
            secondary=secondary,
            confidence=float(confidence),
            baseline=kind,
            raw=raw,
            subtitle=self.vocabulary.subtitles.get(label) if label else None,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Persist the averages when the provider can (the in-memory default)."""
        exporter = getattr(self._averages, "to_state_dict", None)
        if exporter is None:
            return {"scalars": {}, "arrays": {}, "histories": {}}
        return exporter()

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the averages when the provider can."""
        importer = getattr(self._averages, "from_state_dict", None)
        if importer is not None:
            importer(state)
