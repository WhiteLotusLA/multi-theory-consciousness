"""
Model Transition Protocol
=========================

Checkpoints for a model change. Before the switch, ``ModelTransitionProtocol``
captures a baseline: the assessment summary (through an injected ``assess``
callable returning a ``ConsciousnessReport``), the identity markers an
injected ``identity_source`` supplies (Phase 4's identity continuity manager
can supply its latest marker; nothing here reaches into it), and the answers
the brain gives to the generic probes in ``config/transition_prompts.json``.
After the switch, ``evaluate_post_transition`` repeats the three and compares:

- identity drift: the mean absolute drift over the marker sections
  (``core_values``, ``personality_traits``, ``emotional_fingerprint``), each
  compared key by key with a missing key read as 0;
- personality drift: the same over ``personality_traits`` alone;
- score drift: the absolute change of the assessment's ``overall_score``;
- response similarity: the mean Jaccard overlap of lowercased word sets over
  the probes answered on both sides.

The recommendation is ``rollback`` when any drift reaches twice its threshold
or three alerts fire, ``review`` when any threshold is exceeded, ``proceed``
otherwise, and ``insufficient_data`` when either assessment has no overall
score (nothing is assumed in its place). A probe the brain does not answer
(``NoAnswerError`` or an empty spoken text) is excluded from the comparison
and named in ``skipped_prompts``; no text is composed in the model's place.
Alerts and the report name no system.

The probes come from ``prompts_path`` when one is passed (``configured``),
else from the shipped ``config/transition_prompts.json`` next to the code or
under ``PROJECT_ROOT`` (``file``), else from its packaged copy (``package``);
``prompts_source`` reports which, after ``load_prompts``.

This is a tool a researcher runs around a model change, never part of the
automatic cycle. Config keys read at call time: ``TRANSITION_HISTORY_DIR``
(when no ``history_dir`` is passed). Persisted files: ``latest_baseline.json``
and ``transition_<timestamp>.json`` under the history directory, created at
the first write. The probe answers are the system's own words and stay in
that directory, which is private runtime state.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks
from mtc.core.config import get_settings
from mtc.core.paths import resolve_path
from mtc.core.resources import resolve_shipped

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "transition_prompts.json"
)
BASELINE_FILE = "latest_baseline.json"
TRANSITION_FILE_PREFIX = "transition_"
RECOMMENDATIONS: Tuple[str, ...] = (
    "proceed",
    "review",
    "rollback",
    "insufficient_data",
)
MARKER_SECTIONS: Tuple[str, ...] = (
    "core_values",
    "personality_traits",
    "emotional_fingerprint",
)
ASSESSMENT_FIELDS: Tuple[str, ...] = (
    "overall_score",
    "status",
    "coverage",
    "confidence",
    "passing_count",
    "total_indicators",
    "theory_scores",
    "insufficient_indicators",
)
LOW_SIMILARITY = 0.3
PROBE_MAX_TOKENS = 300
PROBE_TEMPERATURE = 0.7

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class TransitionThresholds:
    """Drift above which an alert fires; twice the value recommends rollback."""

    identity: float = 0.15
    personality: float = 0.20
    score: float = 0.10


@dataclass
class TransitionBaseline:
    """What was captured before the model change."""

    captured_at: datetime
    assessment: Dict[str, Any]
    identity_markers: Dict[str, Any]
    responses: Dict[str, str]
    skipped_prompts: List[str] = field(default_factory=list)
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["captured_at"] = self.captured_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionBaseline":
        return cls(
            captured_at=_parse_instant(data.get("captured_at")),
            assessment=dict(data.get("assessment") or {}),
            identity_markers=dict(data.get("identity_markers") or {}),
            responses={
                str(k): str(v) for k, v in (data.get("responses") or {}).items()
            },
            skipped_prompts=[str(p) for p in data.get("skipped_prompts") or []],
            model_name=data.get("model_name"),
        )


@dataclass
class TransitionEvaluation:
    """The comparison of the post-switch state with the baseline."""

    evaluated_at: datetime
    baseline_captured_at: datetime
    identity_drift: Optional[float]
    personality_drift: Optional[float]
    score_drift: Optional[float]
    response_similarity: Optional[float]
    compared_prompts: List[str]
    skipped_prompts: List[str]
    alerts: List[str]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evaluated_at"] = self.evaluated_at.isoformat()
        data["baseline_captured_at"] = self.baseline_captured_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionEvaluation":
        recommendation = str(data.get("recommendation") or "insufficient_data")
        if recommendation not in RECOMMENDATIONS:
            raise ValueError(f"unknown transition recommendation {recommendation!r}")
        return cls(
            evaluated_at=_parse_instant(data.get("evaluated_at")),
            baseline_captured_at=_parse_instant(data.get("baseline_captured_at")),
            identity_drift=_optional_float(data.get("identity_drift")),
            personality_drift=_optional_float(data.get("personality_drift")),
            score_drift=_optional_float(data.get("score_drift")),
            response_similarity=_optional_float(data.get("response_similarity")),
            compared_prompts=[str(p) for p in data.get("compared_prompts") or []],
            skipped_prompts=[str(p) for p in data.get("skipped_prompts") or []],
            alerts=[str(a) for a in data.get("alerts") or []],
            recommendation=recommendation,
            details=dict(data.get("details") or {}),
            model_name=data.get("model_name"),
        )


def _parse_instant(value: Any) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value.strip():
        stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError("a transition record needs an ISO timestamp")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _float_map(value: Any) -> Dict[str, float]:
    """A mapping's numeric entries as floats; everything else is left out."""
    result: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                result[str(key)] = float(item)
    return result


def compare_trait_dicts(pre: Mapping[str, Any], post: Mapping[str, Any]) -> float:
    """Mean absolute drift over the union of keys; a missing key reads as 0."""
    before, after = _float_map(pre), _float_map(post)
    keys = set(before) | set(after)
    if not keys:
        return 0.0
    return sum(abs(before.get(k, 0.0) - after.get(k, 0.0)) for k in keys) / len(keys)


def compare_identity_markers(
    pre: Mapping[str, Any], post: Mapping[str, Any]
) -> Optional[float]:
    """Mean drift over the marker sections present on either side; None if none."""
    drifts = [
        compare_trait_dicts(pre.get(section) or {}, post.get(section) or {})
        for section in MARKER_SECTIONS
        if pre.get(section) or post.get(section)
    ]
    if not drifts:
        return None
    return sum(drifts) / len(drifts)


def jaccard_similarity(first: str, second: str) -> float:
    """Word-set overlap of two texts, lowercased; two empty texts count as 1."""
    a = set(_WORD.findall(first.lower()))
    b = set(_WORD.findall(second.lower()))
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _summarize_report(report: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name in ASSESSMENT_FIELDS:
        value = getattr(report, name, None)
        if name == "theory_scores":
            value = dict(value or {})
        elif name == "insufficient_indicators":
            value = list(value or [])
        summary[name] = value
    return summary


class ModelTransitionProtocol:
    """Baseline, evaluation, report and history around one model change."""

    def __init__(
        self,
        assess: Callable[[], Awaitable[Any]],
        brain: BrainProtocol,
        identity_source: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        prompts_path: Optional[Path] = None,
        history_dir: Optional[Path] = None,
        thresholds: Optional[TransitionThresholds] = None,
        clock: Optional[Callable[[], datetime]] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.assess = assess
        self.brain = brain
        self.identity_source = identity_source
        self._prompts_configured = bool(prompts_path)
        self.prompts_path = Path(prompts_path) if prompts_path else DEFAULT_PROMPTS_PATH
        self.prompts_source: Optional[str] = None
        self._history_dir = Path(history_dir) if history_dir is not None else None
        self.thresholds = thresholds or TransitionThresholds()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.model_name = model_name
        self._baseline: Optional[TransitionBaseline] = None
        self._history: List[TransitionEvaluation] = []
        self._history_loaded = False

    # ------------------------------------------------------------------
    # Paths and files
    # ------------------------------------------------------------------

    def history_dir(self) -> Path:
        """The history directory, resolved now; nothing is created."""
        if self._history_dir is not None:
            return self._history_dir
        cfg = get_settings()
        return resolve_path(cfg.TRANSITION_HISTORY_DIR, settings=cfg)

    def _prompts_file(self) -> Tuple[Path, str]:
        """The probes file and its source (see the module docstring)."""
        if self._prompts_configured:
            return self.prompts_path, "configured"
        default = Path(DEFAULT_PROMPTS_PATH)
        if default.is_file():
            return default, "file"
        try:
            path, where = resolve_shipped(
                f"config/{default.name}", settings=get_settings()
            )
        except FileNotFoundError:
            return default, "file"
        return path, ("file" if where == "project" else "package")

    def load_prompts(self) -> List[Tuple[str, str]]:
        """The probe prompts as ``(id, text)`` pairs, read now."""
        path, source = self._prompts_file()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"transition prompts file {path} is missing") from exc
        self.prompts_source = source
        entries = data.get("prompts") if isinstance(data, Mapping) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"transition prompts file {path} needs a non-empty " "'prompts' list"
            )
        prompts: List[Tuple[str, str]] = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"{path}: each prompt is a mapping")
            prompt_id = str(entry.get("id") or "").strip()
            text = str(entry.get("prompt") or "").strip()
            if not prompt_id or not text or prompt_id in seen:
                raise ValueError(f"{path}: each prompt needs a unique id and text")
            seen.add(prompt_id)
            prompts.append((prompt_id, text))
        return prompts

    def _write(self, name: str, payload: Mapping[str, Any]) -> Path:
        directory = self.history_dir()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def _transition_file_name(self, instant: datetime) -> str:
        stamp = instant.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        directory = self.history_dir()
        candidate = f"{TRANSITION_FILE_PREFIX}{stamp}.json"
        counter = 1
        while (directory / candidate).exists():
            counter += 1
            candidate = f"{TRANSITION_FILE_PREFIX}{stamp}_{counter}.json"
        return candidate

    def _load_history(self) -> None:
        if self._history_loaded:
            return
        self._history_loaded = True
        directory = self.history_dir()
        if not directory.is_dir():
            return
        loaded: List[TransitionEvaluation] = []
        for path in sorted(directory.glob(f"{TRANSITION_FILE_PREFIX}*.json")):
            try:
                loaded.append(
                    TransitionEvaluation.from_dict(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                )
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(
                    "transition record %s skipped: %s", path.name, type(exc).__name__
                )
        self._history = loaded + self._history

    def _load_baseline(self) -> Optional[TransitionBaseline]:
        path = self.history_dir() / BASELINE_FILE
        if not path.is_file():
            return None
        try:
            return TransitionBaseline.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("persisted baseline unreadable: %s", type(exc).__name__)
            return None

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    async def _probe(self) -> Tuple[Dict[str, str], List[str]]:
        """Ask every probe; return the spoken answers and the ids with none."""
        answers: Dict[str, str] = {}
        skipped: List[str] = []
        for prompt_id, text in self.load_prompts():
            try:
                reply = await self.brain.think(
                    text,
                    system_prompt="",
                    max_tokens=PROBE_MAX_TOKENS,
                    temperature=PROBE_TEMPERATURE,
                )
            except NoAnswerError as exc:
                logger.info(
                    "transition probe %s: no answer (%s)", prompt_id, exc.reason
                )
                skipped.append(prompt_id)
                continue
            spoken = strip_reasoning_blocks(reply.text or "")
            if not spoken:
                logger.info("transition probe %s: reasoning only, no answer", prompt_id)
                skipped.append(prompt_id)
                continue
            answers[prompt_id] = spoken
        return answers, skipped

    def _identity(self) -> Dict[str, Any]:
        if self.identity_source is None:
            return {}
        markers = self.identity_source()
        return dict(markers) if isinstance(markers, Mapping) else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def capture_pre_transition_baseline(self) -> TransitionBaseline:
        """Assess, read the identity markers and answer the probes; persist."""
        report = await self.assess()
        answers, skipped = await self._probe()
        baseline = TransitionBaseline(
            captured_at=self._clock(),
            assessment=_summarize_report(report),
            identity_markers=self._identity(),
            responses=answers,
            skipped_prompts=skipped,
            model_name=self.model_name,
        )
        self._baseline = baseline
        self._write(BASELINE_FILE, baseline.to_dict())
        logger.info(
            "transition baseline captured: %d probes answered, %d skipped",
            len(answers),
            len(skipped),
        )
        return baseline

    async def evaluate_post_transition(self) -> TransitionEvaluation:
        """Compare the current state with the baseline; persist the record."""
        baseline = self._baseline or self._load_baseline()
        if baseline is None:
            raise RuntimeError(
                "no pre-transition baseline: call capture_pre_transition_baseline "
                "before the model change"
            )
        self._baseline = baseline
        report = await self.assess()
        post_assessment = _summarize_report(report)
        post_identity = self._identity()
        post_answers, post_skipped = await self._probe()

        thresholds = self.thresholds
        alerts: List[str] = []
        details: Dict[str, Any] = {}

        identity_drift: Optional[float] = None
        if baseline.identity_markers and post_identity:
            identity_drift = compare_identity_markers(
                baseline.identity_markers, post_identity
            )
        if identity_drift is None:
            details["identity"] = {
                "drift": None,
                "threshold": thresholds.identity,
                "reason": (
                    "no_identity_source"
                    if self.identity_source is None
                    else "no_markers"
                ),
            }
        else:
            details["identity"] = {
                "drift": identity_drift,
                "threshold": thresholds.identity,
                "exceeded": identity_drift > thresholds.identity,
            }
            if identity_drift > thresholds.identity:
                alerts.append(
                    f"identity drift {identity_drift:.2f} exceeds the threshold "
                    f"{thresholds.identity:.2f}"
                )

        personality_drift: Optional[float] = None
        pre_traits = baseline.identity_markers.get("personality_traits") or {}
        post_traits = post_identity.get("personality_traits") or {}
        if pre_traits and post_traits:
            personality_drift = compare_trait_dicts(pre_traits, post_traits)
            details["personality"] = {
                "drift": personality_drift,
                "threshold": thresholds.personality,
                "exceeded": personality_drift > thresholds.personality,
            }
            if personality_drift > thresholds.personality:
                alerts.append(
                    f"personality drift {personality_drift:.2f} exceeds the "
                    f"threshold {thresholds.personality:.2f}"
                )
        else:
            details["personality"] = {
                "drift": None,
                "threshold": thresholds.personality,
                "reason": "no_personality_traits",
            }

        pre_score = baseline.assessment.get("overall_score")
        post_score = post_assessment.get("overall_score")
        score_drift: Optional[float] = None
        if pre_score is not None and post_score is not None:
            score_drift = abs(float(post_score) - float(pre_score))
            details["score"] = {
                "pre": float(pre_score),
                "post": float(post_score),
                "drift": score_drift,
                "threshold": thresholds.score,
                "exceeded": score_drift > thresholds.score,
                "pre_passing": baseline.assessment.get("passing_count"),
                "post_passing": post_assessment.get("passing_count"),
            }
            if score_drift > thresholds.score:
                direction = "fell" if float(post_score) < float(pre_score) else "rose"
                alerts.append(
                    f"assessment score {direction} from {float(pre_score):.3f} to "
                    f"{float(post_score):.3f} (drift {score_drift:.2f}, threshold "
                    f"{thresholds.score:.2f})"
                )
        else:
            details["score"] = {
                "pre": pre_score,
                "post": post_score,
                "drift": None,
                "threshold": thresholds.score,
                "reason": "insufficient_data",
                "pre_status": baseline.assessment.get("status"),
                "post_status": post_assessment.get("status"),
            }

        compared = [pid for pid in baseline.responses if pid in post_answers]
        skipped = sorted(
            set(baseline.skipped_prompts) | set(post_skipped),
            key=lambda pid: self._prompt_order().get(pid, len(compared) + 1),
        )
        similarity: Optional[float] = None
        if compared:
            similarity = sum(
                jaccard_similarity(baseline.responses[pid], post_answers[pid])
                for pid in compared
            ) / len(compared)
            details["responses"] = {
                "similarity": similarity,
                "compared": list(compared),
                "skipped": list(skipped),
            }
            if similarity < LOW_SIMILARITY:
                alerts.append(
                    f"response similarity {similarity:.2f} is below "
                    f"{LOW_SIMILARITY:.2f} over {len(compared)} probes"
                )
        else:
            details["responses"] = {
                "similarity": None,
                "compared": [],
                "skipped": list(skipped),
                "reason": "no_probe_answered_on_both_sides",
            }

        recommendation = self._recommend(
            identity_drift, personality_drift, score_drift, alerts
        )
        evaluation = TransitionEvaluation(
            evaluated_at=self._clock(),
            baseline_captured_at=baseline.captured_at,
            identity_drift=identity_drift,
            personality_drift=personality_drift,
            score_drift=score_drift,
            response_similarity=similarity,
            compared_prompts=list(compared),
            skipped_prompts=list(skipped),
            alerts=alerts,
            recommendation=recommendation,
            details=details,
            model_name=self.model_name,
        )
        self._load_history()
        self._history.append(evaluation)
        self._write(
            self._transition_file_name(evaluation.evaluated_at), evaluation.to_dict()
        )
        logger.info(
            "transition evaluated: recommendation %s, %d alerts",
            recommendation,
            len(alerts),
        )
        return evaluation

    def _prompt_order(self) -> Dict[str, int]:
        try:
            return {
                pid: index for index, (pid, _text) in enumerate(self.load_prompts())
            }
        except ValueError:
            return {}

    def _recommend(
        self,
        identity_drift: Optional[float],
        personality_drift: Optional[float],
        score_drift: Optional[float],
        alerts: Sequence[str],
    ) -> str:
        if score_drift is None:
            return "insufficient_data"
        thresholds = self.thresholds
        pairs = (
            (identity_drift, thresholds.identity),
            (personality_drift, thresholds.personality),
            (score_drift, thresholds.score),
        )
        if len(alerts) >= 3 or any(
            drift is not None and drift > 2 * limit for drift, limit in pairs
        ):
            return "rollback"
        if any(drift is not None and drift > limit for drift, limit in pairs):
            return "review"
        return "proceed"

    def generate_transition_report(
        self, evaluation: Optional[TransitionEvaluation]
    ) -> str:
        """A neutral markdown report of one evaluation (or of a pending baseline)."""
        lines = ["# Model transition report", ""]
        baseline = self._baseline
        if baseline is not None:
            lines.append(f"Baseline captured: {baseline.captured_at.isoformat()}")
            score = baseline.assessment.get("overall_score")
            status = baseline.assessment.get("status")
            lines.append(
                "Baseline assessment: "
                + (f"{float(score):.3f}" if score is not None else "no overall score")
                + (f" ({status})" if status else "")
            )
            if baseline.skipped_prompts:
                lines.append(
                    "Baseline probes without an answer: "
                    + ", ".join(baseline.skipped_prompts)
                )
        if evaluation is None:
            lines.extend(
                ["", "## Status", "", "Awaiting the post-transition evaluation."]
            )
            return "\n".join(lines)

        lines.extend(
            [
                "",
                "## Post-transition results",
                "",
                f"Evaluated: {evaluation.evaluated_at.isoformat()}",
                f"Recommendation: {evaluation.recommendation.upper()}",
                "",
                "| metric | drift | threshold | status |",
                "|---|---|---|---|",
            ]
        )
        rows = (
            ("identity", evaluation.identity_drift, self.thresholds.identity),
            ("personality", evaluation.personality_drift, self.thresholds.personality),
            ("score", evaluation.score_drift, self.thresholds.score),
        )
        for name, drift, limit in rows:
            if drift is None:
                reason = (evaluation.details.get(name) or {}).get(
                    "reason", "unmeasured"
                )
                lines.append(f"| {name} | not measured | {limit:.2f} | {reason} |")
                continue
            status = "alert" if drift > limit else "within"
            lines.append(f"| {name} | {drift:.3f} | {limit:.2f} | {status} |")
        if evaluation.response_similarity is None:
            lines.append("| response similarity | not measured | n/a | no pairs |")
        else:
            status = "low" if evaluation.response_similarity < LOW_SIMILARITY else "ok"
            lines.append(
                f"| response similarity | {evaluation.response_similarity:.3f} | "
                f"{LOW_SIMILARITY:.2f} | {status} |"
            )
        lines.extend(
            [
                "",
                "Probes compared: "
                + (", ".join(evaluation.compared_prompts) or "none"),
                "Probes without an answer on one side: "
                + (", ".join(evaluation.skipped_prompts) or "none"),
            ]
        )
        if evaluation.alerts:
            lines.extend(["", "## Alerts", ""])
            lines.extend(f"- {alert}" for alert in evaluation.alerts)
        score = evaluation.details.get("score") or {}
        if score.get("drift") is not None:
            lines.extend(
                [
                    "",
                    "## Assessment detail",
                    "",
                    f"- Score before: {score['pre']:.3f}",
                    f"- Score after: {score['post']:.3f}",
                    f"- Indicators passing: {score.get('pre_passing')} to "
                    f"{score.get('post_passing')}",
                ]
            )
        lines.extend(
            [
                "",
                "## History",
                "",
                f"Recorded transitions: {len(self.history())}",
            ]
        )
        return "\n".join(lines)

    def history(self) -> List[TransitionEvaluation]:
        """Every recorded evaluation, oldest first; reads the directory once."""
        self._load_history()
        return sorted(self._history, key=lambda item: item.evaluated_at)
