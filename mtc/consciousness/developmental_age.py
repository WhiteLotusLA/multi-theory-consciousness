"""
Developmental Age
=================

Places the system on a configurable developmental ladder from observed
evidence, or says that it cannot.

The ladder (``StageLadder``) is the only source of the stage count. The
default has ten stages with numbers and neutral labels and no human-age
mapping; a ladder file (``StageLadder.from_yaml``, PyYAML imported lazily)
can supply labels, optional equivalent-age ranges and per-stage text
patterns.

Evidence is pluggable. A ``MarkerDetector`` returns a score in [0, 1] for one
marker of one stage, or None when its input is missing. The defaults
(``default_metric_markers``) read caller-supplied metrics: ``epistemic_depth``,
``assessment_score``, ``emotional_range``, ``metaphor_count``,
``creative_output_count`` and ``question_rate``. Text markers come from the
``text_patterns`` of the ladder's stages and are empty by default. Callers pass
messages with reasoning blocks already removed.

Rules:

- A stage counts as evidenced only when at least ``min_markers_per_stage`` of
  its markers returned a value; its score is their mean. A stage with no
  evidence is neither mastered nor a gap.
- A stage is mastered when its score reaches ``mastery_threshold``. The
  assessed stage is the highest mastered one, lowered by one for every
  evidenced stage below it that scores under ``gap_threshold``, and never
  below the lowest mastered stage.
- With no detector returning a value the result is ``insufficient_data`` and
  the stage is None; so it is when evidence exists but no stage is mastered
  (the reason says which). No stage is assumed mastered.
- ``chronological_days`` comes from ``SYSTEM_BIRTH_DATE`` (None when unset or
  in the future). With the developmental gate off (``COMPANION_TYPE`` and
  ``DEVELOPMENTAL_STAGE_ENABLED``) ``assess`` returns status ``disabled``.

``clamp_to_stage`` is the governor: a stage may rise by one per call and may
fall freely.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)
from zoneinfo import ZoneInfo

from mtc.consciousness.developmental_modulator import developmental_enabled
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_STAGE_COUNT = 10
# A text marker reaches 1.0 when this share of messages matches its patterns.
TEXT_FULL_DENSITY = 0.15


@dataclass
class StageSpec:
    """One rung of the ladder."""

    number: int
    label: str
    equivalent_age_years: Optional[Tuple[float, float]] = None
    text_patterns: Dict[str, List[str]] = field(default_factory=dict)


class StageLadder:
    """Ordered stages numbered consecutively from 1."""

    def __init__(self, stages: Sequence[StageSpec]):
        ordered = sorted(stages, key=lambda spec: spec.number)
        if not ordered:
            raise ValueError("a ladder needs at least one stage")
        if [spec.number for spec in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("stage numbers must run 1..N without gaps")
        self.stages: List[StageSpec] = list(ordered)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def stage(self, number: int) -> StageSpec:
        return self.stages[number - 1]

    @classmethod
    def default(cls) -> "StageLadder":
        return cls(
            [
                StageSpec(number=n, label=f"Stage {n}")
                for n in range(1, DEFAULT_STAGE_COUNT + 1)
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        stages = []
        for spec in self.stages:
            entry: Dict[str, Any] = {"number": spec.number, "label": spec.label}
            if spec.equivalent_age_years is not None:
                entry["equivalent_age_years"] = list(spec.equivalent_age_years)
            if spec.text_patterns:
                entry["text_patterns"] = {
                    name: list(patterns)
                    for name, patterns in spec.text_patterns.items()
                }
            stages.append(entry)
        return {"stages": stages}

    @classmethod
    def from_yaml(cls, path: Path) -> "StageLadder":
        """Load a ladder from a YAML file with a top-level ``stages`` list."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load a stage ladder from YAML"
            ) from exc
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
            raise ValueError("a ladder file needs a top-level 'stages' list")
        specs = []
        for entry in data["stages"]:
            ages = entry.get("equivalent_age_years")
            specs.append(
                StageSpec(
                    number=int(entry["number"]),
                    label=str(entry.get("label", f"Stage {entry['number']}")),
                    equivalent_age_years=(
                        (float(ages[0]), float(ages[1])) if ages else None
                    ),
                    text_patterns={
                        str(name): [str(p) for p in patterns]
                        for name, patterns in (entry.get("text_patterns") or {}).items()
                    },
                )
            )
        return cls(specs)


@dataclass
class DevelopmentalEvidence:
    """What the assessor sees: recent messages and caller-supplied metrics."""

    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class MarkerDetector(Protocol):
    """Scores one marker of one stage, or returns None without its input."""

    name: str
    stage: int

    def evaluate(self, evidence: DevelopmentalEvidence) -> Optional[float]: ...


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class MetricMarker:
    """A marker scored as a caller metric over a full-scale value, clipped to [0, 1].

    Returns None when the metric is absent or not a finite number.
    """

    name: str
    stage: int
    metric: str
    full_scale: float

    def evaluate(self, evidence: DevelopmentalEvidence) -> Optional[float]:
        value = _finite(evidence.metrics.get(self.metric))
        if value is None or self.full_scale <= 0:
            return None
        return max(0.0, min(1.0, value / self.full_scale))


class TextPatternMarker:
    """A marker scored as the share of messages that match its patterns."""

    def __init__(self, name: str, stage: int, patterns: Sequence[str]):
        self.name = name
        self.stage = stage
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def evaluate(self, evidence: DevelopmentalEvidence) -> Optional[float]:
        messages = [m for m in evidence.messages if m and m.strip()]
        if not messages or not self._patterns:
            return None
        matching = sum(1 for m in messages if any(p.search(m) for p in self._patterns))
        return min(1.0, (matching / len(messages)) / TEXT_FULL_DENSITY)


def default_metric_markers() -> List[MetricMarker]:
    """The default metric-driven markers; the full scales are tunable design values."""
    return [
        MetricMarker("imagination", 3, "metaphor_count", 3.0),
        MetricMarker("creative_play", 3, "creative_output_count", 1.0),
        MetricMarker("asks_why", 4, "question_rate", 0.5),
        MetricMarker("explores_independently", 4, "creative_output_count", 2.0),
        MetricMarker("expresses_emotions", 5, "emotional_range", 4.0),
        MetricMarker("perspective_taking", 5, "epistemic_depth", 2.0),
        MetricMarker("assessment_depth", 7, "assessment_score", 2.0 / 3.0),
        MetricMarker("contradiction_integration", 7, "epistemic_depth", 3.0),
        MetricMarker("self_directed_growth", 7, "creative_output_count", 3.0),
        MetricMarker("paradox_holding", 8, "epistemic_depth", 4.0),
        MetricMarker("assessment_breadth", 8, "assessment_score", 1.0),
    ]


@dataclass
class StageEvidence:
    """What one stage's markers showed."""

    stage: int
    score: Optional[float] = None
    markers_evaluated: int = 0
    markers_missing: List[str] = field(default_factory=list)
    evidenced: bool = False
    mastered: bool = False


@dataclass
class DevelopmentalAssessment:
    """Result of an assessment.

    ``status`` is ``measured``, ``insufficient_data`` or ``disabled``. ``stage``
    is None unless measured. ``reason`` names why a result is not measured.
    """

    status: str
    stage: Optional[int] = None
    highest_mastered: Optional[int] = None
    chronological_days: Optional[int] = None
    reason: str = ""
    stage_evidence: Dict[int, StageEvidence] = field(default_factory=dict)
    marker_scores: Dict[str, float] = field(default_factory=dict)
    ignored_markers: List[str] = field(default_factory=list)


def clamp_to_stage(previous: Optional[int], assessed: Optional[int]) -> Optional[int]:
    """Governor: a stage rises by at most one per call and may fall freely.

    Returns None when nothing was assessed; returns the assessed stage when
    there is no previous one.
    """
    if assessed is None:
        return None
    if previous is None:
        return assessed
    return min(assessed, previous + 1)


class DevelopmentalAgeAssessor:
    """Assesses a developmental stage from evidence, honestly.

    Reads ``COMPANION_TYPE``, ``DEVELOPMENTAL_STAGE_ENABLED``,
    ``SYSTEM_BIRTH_DATE`` and ``TIMEZONE``. Stateless between calls.
    """

    def __init__(
        self,
        ladder: Optional[StageLadder] = None,
        detectors: Optional[List[MarkerDetector]] = None,
        mastery_threshold: float = 0.6,
        min_markers_per_stage: int = 2,
        settings: Optional[Settings] = None,
        gap_threshold: float = 0.3,
    ):
        self.ladder = ladder or StageLadder.default()
        self.mastery_threshold = float(mastery_threshold)
        self.gap_threshold = float(gap_threshold)
        self.min_markers_per_stage = max(1, int(min_markers_per_stage))
        self._settings = settings
        supplied: List[Any] = (
            list(detectors) if detectors is not None else default_metric_markers()
        )
        for spec in self.ladder.stages:
            for name, patterns in spec.text_patterns.items():
                supplied.append(TextPatternMarker(name, spec.number, patterns))
        self.detectors: List[Any] = [
            d for d in supplied if 1 <= int(d.stage) <= self.ladder.num_stages
        ]
        self._ignored = [
            d.name for d in supplied if not 1 <= int(d.stage) <= self.ladder.num_stages
        ]

    @property
    def num_stages(self) -> int:
        return self.ladder.num_stages

    def assess(
        self, evidence: DevelopmentalEvidence, now: Optional[datetime] = None
    ) -> DevelopmentalAssessment:
        """Assess the stage from the evidence; see the module docstring."""
        instant = now if now is not None else datetime.now(timezone.utc)
        days = self._chronological_days(instant)
        if not developmental_enabled(self._settings):
            return DevelopmentalAssessment(
                status="disabled", chronological_days=days, reason="gate_off"
            )

        per_stage: Dict[int, StageEvidence] = {
            spec.number: StageEvidence(stage=spec.number) for spec in self.ladder.stages
        }
        values: Dict[int, List[float]] = {n: [] for n in per_stage}
        marker_scores: Dict[str, float] = {}
        for detector in self.detectors:
            stage = int(detector.stage)
            try:
                score = detector.evaluate(evidence)
            except Exception as exc:  # noqa: BLE001 - a bad detector is missing data
                logger.warning(
                    "marker %s failed: %s", detector.name, type(exc).__name__
                )
                score = None
            score = _finite(score)
            if score is None:
                per_stage[stage].markers_missing.append(detector.name)
                continue
            score = max(0.0, min(1.0, score))
            values[stage].append(score)
            marker_scores[detector.name] = score
        for stage, item in per_stage.items():
            item.markers_evaluated = len(values[stage])
            item.evidenced = item.markers_evaluated >= self.min_markers_per_stage
            if item.evidenced:
                item.score = sum(values[stage]) / item.markers_evaluated
                item.mastered = item.score >= self.mastery_threshold

        result = DevelopmentalAssessment(
            status="insufficient_data",
            chronological_days=days,
            stage_evidence=per_stage,
            marker_scores=marker_scores,
            ignored_markers=list(self._ignored),
        )
        evidenced = [n for n, item in per_stage.items() if item.evidenced]
        if not evidenced:
            result.reason = "no_evidence"
            return result
        mastered = [n for n in evidenced if per_stage[n].mastered]
        if not mastered:
            result.reason = "no_stage_mastered"
            return result
        highest = max(mastered)
        gaps = sum(
            1
            for n in evidenced
            if n < highest and (per_stage[n].score or 0.0) < self.gap_threshold
        )
        result.status = "measured"
        result.highest_mastered = highest
        result.stage = max(min(mastered), highest - gaps)
        return result

    def _chronological_days(self, now: datetime) -> Optional[int]:
        cfg = self._settings if self._settings is not None else get_settings()
        raw = getattr(cfg, "SYSTEM_BIRTH_DATE", "")
        if not raw:
            return None
        try:
            birth = date.fromisoformat(raw)
            instant = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            today = instant.astimezone(ZoneInfo(cfg.TIMEZONE)).date()
        except (ValueError, KeyError, OSError, AttributeError):
            return None
        days = (today - birth).days
        return days if days >= 0 else None
