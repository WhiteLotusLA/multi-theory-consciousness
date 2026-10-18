"""
Consciousness Orchestrator
==========================

Runs one consciousness cycle per conversation turn and shapes the brain's
input from the result. The pipeline hands it a ``RawConversationContext``
(the message, the verified speaker identity and the gathered material); it
builds workspace candidates, lets the ``CycleAugmenter`` add its per-cycle
material, runs ``EnhancedGlobalWorkspace.process_consciousness_cycle`` with the
neural orchestrator's latest signals, fills the cadence-bound measurements
(RPT and causal emergence every N cycles, PAD on the state through the
augmenter, reality monitoring), checks the baseline heartbeat and event
triggers, and returns a ``ModulatedContext`` built by four layers:

1. the narrative (workspace focus and ignition are always reported; every
   other line is gated by a ``NarrationFlags`` field),
2. the generation temperature,
3. the behavioural directives (permissions in neutral second person),
4. the self-awareness block (gated by the inner-state flag; an unmeasured
   baseline is reported as unmeasured, never assumed).

The five theory sections of the prompt carry a computed workspace line and
the module reports their flags allow: the attention schema's and the
metacognition module's first-person reports under the self-model flag (a
neutral measured summary of this cycle's attention schema and higher-order
thoughts otherwise), the active inference report under the free-energy or
body-drive flag, and the substrate block in ``neural_state`` under the
binding flag. The flags reach the modules once, at construction, through
``apply_narration``; a PAD model the augmenter builds later receives them on
first sight. Phi appears in no line of any layer and no section: the
integration proxy is telemetry, read through the API only.

Measurement honesty: the workspace records into the PhiTracker during the
cycle and the orchestrator never records again; the theory interaction
manager is updated by the workspace alone and ``status()`` lists its six
modulations with ``consumed`` true for the two the workspace applies; the
baseline's overall score stays None while the assessment reports insufficient
data, and the default assessment scales its bounded histories by
``OrchestratorConfig.assessment_warmup_cycles``, never by the running cycle
count; every Optional ``ConsciousnessState`` field left None is named in
``absent_signals`` with a reason from ``ABSENT_REASONS`` (``failed`` when the
step that fills it failed in that cycle); the PAD inputs that were measured
are listed in ``pad_sources_measured``. A cached assessment report is reused
only while no cycle has run since it was taken, and is applied to the
baseline once. The orchestrator and the
augmenter share one PAD model and one prospective simulator, and the neural
orchestrator is reached through its public surface only
(``publish_retrieved_memories``, ``verification_neural_data``, the latest
signals and substrate states, the binding stats and context).

Soft failure is loud: every step (the idle Phi sample, a direct assessment
and the baseline update included) counts its failures by name, logs the
first and every hundredth at WARNING (type name only, never the text), raises
instead when ``DEBUG`` is true, and a failed cycle returns
``ModulatedContext.unavailable`` with an empty narrative.

Config keys read: ``DEBUG`` (through ``settings`` or ``get_settings()``).
Persisted state (``to_state_dict``): the baseline, the trigger maxima, the
cycle and failure counters. The message, the retrieved material, the speaker
name and the narrative are private conversation data at runtime.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from mtc.consciousness.developmental_modulator import developmental_enabled
from mtc.consciousness.enhanced_global_workspace import (
    COMPETITION_HIERARCHICAL,
    ConsciousnessState,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.core.config import Settings, get_settings
from mtc.core.narration import NarrationFlags
from mtc.core.profiler import StageProfiler
from mtc.pipeline.context_types import (
    ModulatedContext,
    RawConversationContext,
    TheorySections,
    cycle_winner,
)
from mtc.pipeline.cycle_augmenter import AugmentRecord, CycleAugmenter, CycleInputs
from mtc.pipeline.speaker_trust import SpeakerTrust

logger = logging.getLogger(__name__)

# The eight theories the assessment scores (its ConsciousnessTheory names).
THEORIES: Tuple[str, ...] = ("GWT", "IIT", "AST", "HOT", "FEP", "RPT", "BLT", "AE")

BASELINE_UNMEASURED = "unmeasured"
# The assessment's report status when no indicator was measured.
INSUFFICIENT_STATUS = "insufficient_data"
MESSAGE_CONTENT_TYPE = "message"

# The two theory modulations the workspace applies as the next cycle's
# salience bias; the other four are computed and reported only.
CONSUMED_MODULATIONS: Tuple[str, ...] = (
    "somatic_salience_bias",
    "phi_competition_bias",
)

# Generic English emotion words for the message candidate's salience.
EMOTION_WORDS = frozenset(
    {
        "love",
        "hate",
        "fear",
        "joy",
        "sad",
        "angry",
        "happy",
        "scared",
        "worried",
        "excited",
        "lonely",
        "proud",
        "ashamed",
        "curious",
        "grateful",
        "hurt",
        "anxious",
        "peaceful",
        "frustrated",
        "hopeful",
        "afraid",
        "glad",
        "upset",
        "calm",
    }
)
_EMOTION_WORD_WEIGHT = 0.2
_LENGTH_REFERENCE_WORDS = 50.0
_IGNITION_REFERENCE = 5.0
_DEPTH_REFERENCE = 4.0

# Lines that name Phi or the integration proxy never reach any layer.
_PHI_MARK = re.compile(r"(?<![a-z])phi(?![a-z])|integration[ _-]?proxy", re.IGNORECASE)

_OPTIONAL_STATE_FIELDS: Tuple[str, ...] = tuple(
    item.name for item in fields(ConsciousnessState) if item.default is None
)
# Optional state fields the default scan cannot see (they have no default or
# a non-None one), named explicitly whenever the cycle leaves them None.
_EXPLICIT_OPTIONAL_FIELDS: Tuple[str, ...] = (
    "integration_level",
    "broadcast_coverage",
    "binding_quality",
    "protoself_stability",
    "feeling_of_knowing",
    "autobiographic_continuity",
)
# Every state field the cycle names in ``absent_signals`` when it is None.
NAMED_STATE_FIELDS: Tuple[str, ...] = _OPTIONAL_STATE_FIELDS + tuple(
    name for name in _EXPLICIT_OPTIONAL_FIELDS if name not in _OPTIONAL_STATE_FIELDS
)
# (state field prefix, the orchestrator step that fills it): a field left None
# in a cycle whose step failed is named ``failed``. The PAD fields are judged
# apart: their step also runs the gate and prospective sub-steps, so a failure
# of that step names them ``failed`` only when PAD from the state is on and
# the PAD computation itself did not finish.
_PRODUCER_STEPS: Tuple[Tuple[str, str], ...] = (
    ("rpt_", "rpt"),
    ("causal_emergence", "causal_emergence"),
    ("prm_discrimination", "reality_monitor"),
)
_PAD_STEP = "augment_after"


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _read(item: Any, key: str, default: Any = None) -> Any:
    """A field from a mapping or an attribute from an object."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _template_label(candidate: Any, metadata: Mapping[str, Any]) -> str:
    """The kind of an authored-template candidate, never its words: for a
    prospective simulation 'a prospective-simulation template (kind:
    anticipatory)'."""
    content_type = _text(getattr(candidate, "content_type", "")) or "authored"
    kind = _text(metadata.get("sim_type")) or content_type
    return f"a {content_type.replace('_', '-')} template (kind: {kind})"


def _theory_name(theory: Any) -> Optional[str]:
    """The theory key of an indicator result: an enum member's name (the
    assessment's ``ConsciousnessTheory``) or a plain string; None when the
    result names no theory."""
    if theory is None:
        return None
    if isinstance(theory, str):
        return theory or None
    name = getattr(theory, "name", None)
    return name if isinstance(name, str) and name else None


def _names_phi(text: Any) -> bool:
    return bool(_PHI_MARK.search(str(text)))


def _phi_free(text: Any) -> Optional[str]:
    """The report with every line naming Phi removed; None when nothing is left."""
    kept = [line for line in _text(text).splitlines() if not _names_phi(line)]
    joined = "\n".join(line for line in kept if line.strip())
    return joined or None


_MISSING = object()


def _attention_summary(schema: Any) -> Optional[str]:
    """This cycle's measured attention-schema values in neutral third person:
    the attention state, the focus's source module, the capacity used and,
    once a shift has been classified, the voluntary ratio (its 0.5 before
    then is a default, not a measurement). None without a schema."""
    if schema is None:
        return None
    parts: List[str] = []
    mode = getattr(getattr(schema, "attention_state", None), "value", None)
    if isinstance(mode, str) and mode:
        parts.append(f"state {mode}")
    focus = getattr(schema, "current_focus", _MISSING)
    if focus is None:
        parts.append("no focus")
    elif focus is not _MISSING:
        source = _text(getattr(focus, "source_module", ""))
        if source:
            parts.append(f"focus from {source}")
    capacity = _finite(getattr(schema, "attention_capacity_used", None))
    if capacity is not None:
        parts.append(f"capacity used {capacity:.2f}")
    voluntary = getattr(schema, "voluntary_shifts", None)
    captured = getattr(schema, "captured_shifts", None)
    ratio = _finite(getattr(schema, "voluntary_ratio", None))
    if (
        isinstance(voluntary, int)
        and isinstance(captured, int)
        and voluntary + captured > 0
        and ratio is not None
    ):
        parts.append(f"voluntary ratio {ratio:.2f}")
    if not parts:
        return None
    return "Attention schema: " + "; ".join(parts) + "."


def _mean(values: List[Optional[float]]) -> Optional[float]:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _metacognition_summary(state: Any) -> Optional[str]:
    """This cycle's higher-order thoughts in neutral third person: how many the
    cycle generated, with their mean confidence and clarity. The module's
    standing confidence and doubt are left out: under the hot_mean source (the
    default) the standing confidence is the mean over the buffered higher-order
    thoughts (None before the first), and nothing measures doubt. None when
    the cycle recorded no metacognitive state."""
    if getattr(state, "metacognitive_state", None) is None:
        return None
    thoughts = list(getattr(state, "higher_order_thoughts", None) or [])
    if not thoughts:
        return "Metacognition: no higher-order thought this cycle."
    noun = "thought" if len(thoughts) == 1 else "thoughts"
    text = f"Metacognition: {len(thoughts)} higher-order {noun} this cycle"
    details: List[str] = []
    confidence = _mean([_finite(getattr(t, "confidence", None)) for t in thoughts])
    if confidence is not None:
        details.append(f"mean confidence {confidence:.2f}")
    clarity = _mean([_finite(getattr(t, "clarity", None)) for t in thoughts])
    if clarity is not None:
        details.append(f"mean clarity {clarity:.2f}")
    if details:
        text += " (" + ", ".join(details) + ")"
    return text + "."


# ----------------------------------------------------------------------------
# Baseline and expression profile
# ----------------------------------------------------------------------------


@dataclass
class ConsciousnessBaseline:
    """The cached assessment the modulation layers read.

    ``overall_score`` and every theory score stay None until an assessment
    measures them (an ``insufficient_data`` report measures nothing, and a
    theory with no measured indicator result keeps None); ``status`` is
    ``unmeasured`` before the first report and the report's own status
    afterwards. ``stretch_ceiling`` is 1.2 times the
    measured overall (capped at 1.0) and ``trend`` needs five scores.
    """

    overall_score: Optional[float] = None
    theory_scores: Dict[str, Optional[float]] = field(
        default_factory=lambda: {theory: None for theory in THEORIES}
    )
    indicator_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    status: str = BASELINE_UNMEASURED
    coverage: Optional[float] = None
    stretch_ceiling: Optional[float] = None
    score_history: List[float] = field(default_factory=list)
    trend: Optional[str] = None
    last_updated: Optional[datetime] = None

    MAX_HISTORY = 20
    TREND_WINDOW = 5
    TREND_SLOPE = 0.01

    @property
    def measured(self) -> bool:
        return self.overall_score is not None

    def update_from_report(self, report: Any, now: datetime) -> None:
        """Take a ``ConsciousnessReport``; an insufficient overall stays None.

        A report whose status is ``insufficient_data`` measured nothing, so its
        overall is stored as None whatever number it carries: no stretch
        ceiling and no score history entry.
        """
        status = str(getattr(report, "status", None) or "measured")
        overall = _finite(getattr(report, "overall_score", None))
        if status == INSUFFICIENT_STATUS:
            overall = None
        scores = dict(getattr(report, "theory_scores", None) or {})
        results = getattr(report, "indicator_results", None) or {}
        # A theory score is kept only for a theory with at least one measured
        # indicator result; one whose included results are all absent or
        # insufficient is not a measurement. A report whose results name no
        # theory (an injected assessor's plain report) keeps its scores.
        named, measured_theories = False, set()
        for result in results.values():
            theory = _theory_name(getattr(result, "theory", None))
            if theory is None:
                continue
            named = True
            if getattr(result, "status", "measured") == "measured":
                measured_theories.add(theory)

        def kept(theory: str, value: Any) -> Optional[float]:
            if named and theory not in measured_theories:
                return None
            return _finite(value)

        self.theory_scores = {
            theory: kept(theory, scores.get(theory)) for theory in THEORIES
        }
        for theory, value in scores.items():
            if theory not in self.theory_scores:
                self.theory_scores[theory] = kept(str(theory), value)
        self.indicator_scores = {
            name: (
                _finite(getattr(result, "score", None))
                if getattr(result, "status", "measured") == "measured"
                else None
            )
            for name, result in results.items()
        }
        self.status = status
        self.coverage = _finite(getattr(report, "coverage", None))
        self.overall_score = overall
        if overall is not None:
            self.stretch_ceiling = min(1.0, overall * 1.2)
            self.add_score(overall)
            self.trend = self.compute_trend()
        else:
            self.stretch_ceiling = None
        self.last_updated = now

    def add_score(self, score: float) -> None:
        self.score_history.append(float(score))
        if len(self.score_history) > self.MAX_HISTORY:
            self.score_history = self.score_history[-self.MAX_HISTORY :]

    def compute_trend(self) -> Optional[str]:
        """Slope sign of the last five scores; None with fewer than five."""
        if len(self.score_history) < self.TREND_WINDOW:
            return None
        recent = self.score_history[-self.TREND_WINDOW :]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0
        if slope > self.TREND_SLOPE:
            return "rising"
        if slope < -self.TREND_SLOPE:
            return "falling"
        return "stable"

    def age_s(self, now: datetime) -> Optional[float]:
        if self.last_updated is None:
            return None
        return (now - self.last_updated).total_seconds()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "theory_scores": dict(self.theory_scores),
            "status": self.status,
            "coverage": self.coverage,
            "stretch_ceiling": self.stretch_ceiling,
            "trend": self.trend,
            "score_history": list(self.score_history),
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "measured": self.measured,
        }


@dataclass(frozen=True)
class ExpressionProfile:
    """Five expressive gradients in [0, 1] derived from a measured baseline."""

    introspective_depth: float
    emotional_range: float
    abstract_thinking: float
    self_initiated_topics: float
    narrative_coherence: float

    GRADIENT_NAMES = {
        "introspective_depth": "introspective depth",
        "emotional_range": "emotional range",
        "abstract_thinking": "abstract thinking",
        "self_initiated_topics": "self-initiated topics",
        "narrative_coherence": "narrative coherence",
    }

    def average(self) -> float:
        return sum(self.as_dict().values()) / 5.0

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.GRADIENT_NAMES}

    def scaled(self, factor: float) -> "ExpressionProfile":
        return ExpressionProfile(
            **{name: _clip(value * factor) for name, value in self.as_dict().items()}
        )

    @classmethod
    def from_baseline(
        cls,
        baseline: ConsciousnessBaseline,
        state: Any = None,
        drives: Optional[Mapping[str, Any]] = None,
    ) -> Optional["ExpressionProfile"]:
        """The profile for a measured baseline; None while it is unmeasured.

        Each of the eight theory scores moves one gradient; a theory the
        assessment did not score contributes nothing (not a stand-in 0.5).
        """
        overall = _finite(baseline.overall_score)
        if overall is None:
            return None

        def term(theory: str, weight: float) -> float:
            score = _finite(baseline.theory_scores.get(theory))
            return 0.0 if score is None else (score - 0.5) * weight

        def secondary(attr: str) -> float:
            value = _finite(getattr(state, attr, None)) if state is not None else None
            return 0.0 if value is None else (value - 0.5) * 0.1

        curiosity = 0.0
        drive = drives.get("curiosity") if isinstance(drives, Mapping) else None
        if drive is not None and callable(getattr(drive, "get_deviation", None)):
            deviation = _finite(drive.get_deviation())
            if deviation is not None:
                curiosity = (_clip(deviation) - 0.5) * 0.1

        return cls(
            introspective_depth=_clip(overall + term("HOT", 0.3)),
            emotional_range=_clip(
                overall + term("AST", 0.3) + secondary("feeling_of_knowing")
            ),
            abstract_thinking=_clip(
                overall + term("BLT", 0.3) + term("IIT", 0.1) + term("FEP", 0.1)
            ),
            self_initiated_topics=_clip(
                overall + term("GWT", 0.3) + term("AE", 0.1) + curiosity
            ),
            narrative_coherence=_clip(
                overall + term("RPT", 0.3) + secondary("autobiographic_continuity")
            ),
        )


# ----------------------------------------------------------------------------
# Configuration and narration wiring
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorConfig:
    """Cadences, triggers and temperature bounds.

    ``min_temperature`` and ``max_temperature`` bound every temperature the
    orchestrator returns, the expression profile's band included.
    ``assessment_warmup_cycles`` is the ``warmup_cycles`` the default
    assessment receives: the scale its bounded histories are read against
    (the assessment's own default, 25), never the running cycle count.
    """

    baseline_temperature: float = 0.7
    min_temperature: float = 0.5
    max_temperature: float = 1.2
    heartbeat_s: float = 900.0
    rpt_every_n_cycles: int = 50
    causal_emergence_every_n_cycles: int = 50
    ignition_spike_factor: float = 1.5
    high_integration_threshold: float = 0.8
    high_integration_streak: int = 5
    loop_quality_trigger: float = 0.85
    prm_internal_thoughts: int = 3
    assessment_cache_s: float = 10.0
    assessment_normalize: bool = False
    assessment_warmup_cycles: int = 25
    goal_count: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.min_temperature <= self.max_temperature:
            raise ValueError("temperature bounds must satisfy 0 < min <= max")
        if (
            not self.min_temperature
            <= self.baseline_temperature
            <= self.max_temperature
        ):
            raise ValueError("baseline_temperature must lie within the bounds")
        for name in (
            "rpt_every_n_cycles",
            "causal_emergence_every_n_cycles",
            "high_integration_streak",
            "assessment_warmup_cycles",
            "goal_count",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if int(self.prm_internal_thoughts) < 0:
            raise ValueError("prm_internal_thoughts must be 0 or more")
        for name in ("heartbeat_s", "assessment_cache_s"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be 0 or more")
        if float(self.ignition_spike_factor) <= 1.0:
            raise ValueError("ignition_spike_factor must be above 1")


def apply_narration(
    workspace: Any,
    pad_model: Any = None,
    theory_of_mind: Any = None,
    flags: Optional[NarrationFlags] = None,
) -> Dict[str, bool]:
    """Set every module narration attribute from the Settings-backed flags.

    Returns the attribute names that were set. The modules keep computing
    whatever is narrated or not; only their context strings change.
    """
    flags = flags if flags is not None else NarrationFlags.from_settings()
    applied: Dict[str, bool] = {}

    def put(target: Any, attribute: str, value: bool, label: str) -> None:
        if target is None or not hasattr(target, attribute):
            return
        setattr(target, attribute, bool(value))
        applied[label] = bool(value)

    damasio = getattr(workspace, "damasio", None)
    put(damasio, "narrate_body_drives", flags.body_drives, "damasio.body_drives")
    put(damasio, "narrate_somatic", flags.somatic, "damasio.somatic")
    put(
        damasio,
        "narrate_narrative_identity",
        flags.narrative_identity,
        "damasio.narrative_identity",
    )
    ai_config = getattr(getattr(workspace, "active_inference", None), "config", None)
    put(ai_config, "narrate_body_drives", flags.body_drives, "fep.body_drives")
    put(ai_config, "narrate_free_energy", flags.free_energy, "fep.free_energy")
    put(
        getattr(workspace, "metacognition", None),
        "narrate_self_model",
        flags.self_model,
        "hot.self_model",
    )
    put(
        getattr(workspace, "beautiful_loop", None),
        "narrate_self_model",
        flags.self_model,
        "blt.self_model",
    )
    put(pad_model, "narrate_affect_line", flags.affect_line, "pad.affect_line")
    put(
        theory_of_mind,
        "narrate_speaker_model",
        flags.speaker_model,
        "tom.speaker_model",
    )
    return applied


# ----------------------------------------------------------------------------
# The orchestrator
# ----------------------------------------------------------------------------


class ConsciousnessOrchestrator:
    """One consciousness cycle per turn, then four modulation layers.

    Every collaborator except the workspace, the augmenter, the narration
    flags and the speaker trust source is optional; a missing producer leaves
    its signals named in ``absent_signals``. ``pad_model`` and ``prospective``
    are handed to the augmenter, so one of each is in use (a different one
    already in the augmenter raises ``ValueError``). The narration flags are
    applied to the workspace's modules and the PAD model at construction.
    ``assessor`` is an async callable returning a ``ConsciousnessReport``;
    None runs ``run_full_assessment`` over the workspace's own modules with
    the workspace's PhiTracker, the RPT and causal emergence analyzers and
    the PAD model.
    """

    def __init__(
        self,
        workspace: Any,
        *,
        augmenter: CycleAugmenter,
        narration: NarrationFlags,
        speaker_trust: SpeakerTrust,
        neural_orchestrator: Any = None,
        rpt_measurement: Any = None,
        causal_emergence: Any = None,
        pad_model: Any = None,
        assessor: Optional[Callable[[], Awaitable[Any]]] = None,
        snapshot_service: Any = None,
        state_manager: Any = None,
        reality_monitor: Any = None,
        prospective: Any = None,
        goal_source: Any = None,
        attention_observer: Any = None,
        developmental: Any = None,
        profiler: Optional[StageProfiler] = None,
        config: Optional[OrchestratorConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not isinstance(augmenter, CycleAugmenter):
            raise TypeError("augmenter must be a CycleAugmenter")
        if not isinstance(narration, NarrationFlags):
            raise TypeError("narration must be a NarrationFlags")
        if not isinstance(speaker_trust, SpeakerTrust):
            raise TypeError("speaker_trust must be a SpeakerTrust")
        self.workspace = workspace
        self.augmenter = augmenter
        self.narration = narration
        self.speaker_trust = speaker_trust
        self.neural_orchestrator = neural_orchestrator
        self.rpt_measurement = rpt_measurement
        self.causal_emergence = causal_emergence
        # One PAD model and one prospective simulator, shared with the
        # augmenter: an injected one is handed over (a different one already
        # in the augmenter raises), otherwise the augmenter's is read.
        if pad_model is not None:
            augmenter.adopt_pad_model(pad_model)
        if prospective is not None:
            augmenter.adopt_prospective(prospective)
        self._assessor = assessor
        self.snapshot_service = snapshot_service
        self.state_manager = state_manager
        self.reality_monitor = reality_monitor
        self.goal_source = goal_source
        self.attention_observer = attention_observer
        self.developmental = developmental
        self.profiler = profiler if profiler is not None else StageProfiler()
        self.config = config or OrchestratorConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._settings = settings

        self.baseline = ConsciousnessBaseline()
        self._baseline_due = True
        self._baseline_due_reason: Optional[str] = "never_measured"
        self._last_report: Any = None
        self._last_report_at: Optional[datetime] = None
        self._last_report_cycle: Optional[int] = None
        # Serial of the stored report (one per assessment run) and of the one
        # the baseline last took, so a reused cached report is not re-applied
        self._report_serial = 0
        self._applied_serial: Optional[int] = None

        self._cycle_count = 0
        self._last_state: Optional[Any] = None
        self._last_absent: Dict[str, str] = {}
        self._last_pad_sources: Tuple[str, ...] = ()
        self._last_record: Optional[AugmentRecord] = None
        self._failures: Dict[str, int] = {}
        # The steps that failed in the cycle being processed (cleared when a
        # turn starts); the absent reasons consult it first
        self._cycle_failures: Set[str] = set()

        self._max_ignition = 0
        self._max_epistemic_depth: Optional[int] = None
        self._high_integration_streak = 0

        self._developmental_stage: Optional[float] = None
        self._applied_stage: Optional[float] = None
        self._pending_self_assessment: Optional[Dict[str, Any]] = None
        self._last_self_assessment: Optional[Dict[str, Any]] = None

        # The narration flags reach the modules once, here (applying the same
        # flags again, as the composition root may, changes nothing); a PAD
        # model the augmenter builds later receives them on first sight.
        self._narration_applied: Dict[str, bool] = apply_narration(
            workspace, self.pad_model, None, narration
        )
        self._narrated_pad: Any = self.pad_model

    # ------------------------------------------------------------------
    # Settings, time and failure accounting
    # ------------------------------------------------------------------

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _debug(self) -> bool:
        return bool(getattr(self._cfg(), "DEBUG", False))

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment

    def _record_failure(self, step: str, exc: BaseException) -> None:
        count = self._failures.get(step, 0) + 1
        self._failures[step] = count
        self._cycle_failures.add(step)
        if count == 1 or count % 100 == 0:
            logger.warning(
                "orchestrator step %s failed (%d so far): %s",
                step,
                count,
                type(exc).__name__,
            )
        if self._debug():
            raise exc

    @contextmanager
    def _step(self, name: str) -> Iterator[None]:
        """Count a step's failure by name; re-raise under DEBUG."""
        try:
            with self.profiler.stage(name):
                yield
        except Exception as exc:  # noqa: BLE001 - every step is accounted for
            self._record_failure(name, exc)

    # ------------------------------------------------------------------
    # The turn
    # ------------------------------------------------------------------

    async def process(self, raw: RawConversationContext) -> ModulatedContext:
        """Run the cycle for one turn and return the modulated context."""
        if not isinstance(raw, RawConversationContext):
            raise TypeError("process takes a RawConversationContext")
        self._cycle_failures = set()
        with self.profiler.stage("process"):
            with self._step("publish_retrieval"):
                self._publish_retrieval(raw)
            with self._step("heartbeat"):
                self._check_heartbeat()
            goals: List[str] = []
            with self._step("goals"):
                goals = self._read_goals()
            inputs = CycleInputs(
                recent_messages=self._recent_messages(raw),
                goals=goals,
                now=raw.now,
                neural_affect=self._neural_affect(),
            )

            candidates: List[WorkspaceCandidate] = []
            with self._step("candidates"):
                candidates = self._build_candidates(raw)
            if not candidates:
                with self._step("message_candidate"):
                    candidates = [self._message_to_candidate(raw)]
            # The message candidate is always first; the focus line compares
            # the cycle's primary content with this object by identity.
            message_candidate = candidates[0] if candidates else None
            with self._step("augment_before"):
                candidates = self.augmenter.before_cycle(candidates, inputs)
            competed = len(candidates)

            state: Optional[Any] = None
            with self._step("cycle"):
                state = await self._run_cycle(candidates)
            if state is None:
                return ModulatedContext.unavailable(
                    "cycle_failed", self.config.baseline_temperature
                )
            self._cycle_count += 1
            self._last_state = state
            self._mark_due_on_first_data()

            with self._step("rpt"):
                self._run_rpt(state)
            with self._step("causal_emergence"):
                self._run_causal_emergence(state)
            record = AugmentRecord()
            with self._step("augment_after"):
                record = self.augmenter.after_cycle(state, inputs)
            with self._step("narration"):
                self._narrate_new_pad_model()
            self._last_record = record
            self._last_pad_sources = tuple(record.pad_sources_measured)
            with self._step("reality_monitor"):
                self._record_reality_monitoring(raw, state)
            with self._step("event_triggers"):
                self._check_event_triggers(state)
            with self._step("attention_observer"):
                await self._observe_attention()
            with self._step("developmental"):
                self._apply_developmental()
            with self._step("persistence"):
                if self.state_manager is not None:
                    self.state_manager.on_cycle_complete()

            context: Optional[ModulatedContext] = None
            with self._step("modulation"):
                context = self._modulate(
                    raw, state, record, competed, message_candidate
                )
            if context is None:
                context = ModulatedContext.unavailable(
                    "modulation_failed", self.config.baseline_temperature
                )
                context.consciousness_state = state
            return context

    @property
    def pad_model(self) -> Any:
        """The one PAD model, shared with the augmenter that computes PAD
        through it: the injected one, else the one the augmenter builds on its
        first ``after_cycle`` (so this is None until then, never a copy)."""
        return self.augmenter.pad_model

    @property
    def prospective(self) -> Any:
        """The one prospective simulator, shared with the augmenter that
        generates through it: the injected one, else the augmenter's own
        (None until the augmenter builds it)."""
        return self.augmenter.prospective

    def _narrate_new_pad_model(self) -> None:
        """Apply the narration flags to a PAD model not seen before."""
        pad = self.pad_model
        if pad is None or pad is self._narrated_pad:
            return
        self._narration_applied.update(apply_narration(None, pad, None, self.narration))
        self._narrated_pad = pad

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _publish_retrieval(self, raw: RawConversationContext) -> None:
        """Hand the turn's retrieved memories to the measurement layer through
        the neural orchestrator's public ``publish_retrieved_memories``; a
        neural orchestrator without it is a counted failure, never a write
        into its private members."""
        if self.neural_orchestrator is None:
            return
        self.neural_orchestrator.publish_retrieved_memories(
            list(raw.retrieved_memories or [])
        )

    def _check_heartbeat(self) -> None:
        if self._baseline_due:
            return
        age = self.baseline.age_s(self._now())
        if age is None or age >= self.config.heartbeat_s:
            self._mark_due("heartbeat")

    def _mark_due(self, reason: str) -> None:
        self._baseline_due = True
        self._baseline_due_reason = reason

    def _mark_due_on_first_data(self) -> None:
        """After a successful cycle an insufficient baseline that is not due
        becomes due (reason ``first_data``), so the next refresh measures the
        data this cycle produced instead of waiting for the heartbeat."""
        if self.baseline.status == INSUFFICIENT_STATUS and not self._baseline_due:
            self._mark_due("first_data")

    def _read_goals(self) -> List[str]:
        """Top goals from the goal source, handed to the prospective simulator."""
        if self.goal_source is None:
            return []
        top = self.goal_source.get_top_goals(self.config.goal_count) or []
        goals = [_text(getattr(goal, "description", goal)) for goal in top]
        goals = [goal for goal in goals if goal]
        simulator = self.prospective
        if simulator is not None and callable(getattr(simulator, "set_goals", None)):
            simulator.set_goals(goals)
        return goals

    @staticmethod
    def _recent_messages(raw: RawConversationContext) -> List[Dict[str, Any]]:
        if raw.recent_messages:
            return list(raw.recent_messages)
        return [{"role": "user", "content": raw.message}]

    def _neural_affect(self) -> Optional[Dict[str, float]]:
        """The substrate's measured PAD inputs from the latest experience.

        Reads real substrate output only: ``snn_valence`` is the SNN's decoded
        output emotions (joy minus sadness) or, when the SNN decoded none, the
        reservoir's intuitive valence from a trained readout; ``snn_spike_rate``
        is the mean of the SNN's output firing rates on a known scale. The
        experience's own ``emotions`` are what the caller fed the substrate,
        not what it produced, and are never read. Returns None when nothing
        was measured, so both sources keep their ``PADSources`` defaults and
        stay out of ``pad_sources_measured``.
        """
        experience = getattr(
            getattr(self.neural_orchestrator, "state", None), "current_experience", None
        )
        if experience is None:
            return None
        snn = getattr(experience, "snn_result", None)
        snn = snn if isinstance(snn, Mapping) else {}
        affect: Dict[str, float] = {}

        valence: Optional[float] = None
        decoded = snn.get("output_emotions")
        if isinstance(decoded, Mapping):
            joy = _finite(decoded.get("joy"))
            sadness = _finite(decoded.get("sadness"))
            if joy is not None or sadness is not None:
                valence = (joy or 0.0) - (sadness or 0.0)
        if valence is None:
            subconscious = getattr(experience, "subconscious_state", None)
            if getattr(subconscious, "readout_trained", False) is True:
                valence = _finite(getattr(subconscious, "intuitive_valence", None))
        if valence is not None:
            affect["snn_valence"] = _clip(valence, -1.0, 1.0)

        rate = self._snn_spike_rate(snn)
        if rate is not None:
            affect["snn_spike_rate"] = rate
        return affect or None

    @staticmethod
    def _snn_spike_rate(snn: Mapping[str, Any]) -> Optional[float]:
        """Mean output activity of the SNN in [0, 1], or None without a known
        scale: ``spike_fraction`` output is already in [0, 1]; ``hz`` output is
        read through the encoder's decoded emotions, which carry the same rates
        scaled to [0, 1]. An unknown unit is not measured."""
        unit = snn.get("output_unit")
        if unit == "spike_fraction":
            rates = snn.get("firing_rates")
            values: Any = rates.get("output") if isinstance(rates, Mapping) else None
        elif unit == "hz":
            decoded = snn.get("output_emotions")
            values = list(decoded.values()) if isinstance(decoded, Mapping) else None
        else:
            return None
        if values is None:
            return None
        try:
            items = list(values)
        except TypeError:
            return None
        finite = [v for v in (_finite(item) for item in items) if v is not None]
        if not finite:
            return None
        return _clip(sum(finite) / len(finite), 0.0, 1.0)

    async def _run_cycle(self, candidates: List[WorkspaceCandidate]) -> Any:
        signals = getattr(self.neural_orchestrator, "latest_neural_signals", None)
        return await self.workspace.process_consciousness_cycle(
            candidates, neural_signals=signals
        )

    def _run_rpt(self, state: Any) -> None:
        rpt = self.rpt_measurement
        if rpt is None:
            return
        neural = self.neural_orchestrator
        if neural is not None and callable(getattr(neural, "get_binding_stats", None)):
            stats = neural.get_binding_stats()
            if stats and callable(getattr(rpt, "set_oscillatory_binding_stats", None)):
                rpt.set_oscillatory_binding_stats(stats)
        if self._cycle_count % self.config.rpt_every_n_cycles != 0:
            return
        substrate = getattr(neural, "latest_substrate_states", None) or {}
        metrics = rpt.measure_full(
            snn_state=substrate.get("snn_state"),
            lsm_state=substrate.get("lsm_state"),
            workspace_state=self.workspace.get_statistics(),
        )
        if metrics is not None:
            state.rpt_deep_recurrence = _finite(metrics.global_recurrence_score)
            state.rpt_superficial_recurrence = _finite(metrics.local_recurrence_score)

    def _run_causal_emergence(self, state: Any) -> None:
        analyzer = self.causal_emergence
        tracker = getattr(self.workspace, "phi_tracker", None)
        if analyzer is None or tracker is None:
            return
        if self._cycle_count % self.config.causal_emergence_every_n_cycles != 0:
            return
        result = analyzer.analyze_modules(tracker)
        if result is None:
            state.causal_emergence_warming_up = True
            return
        state.causal_emergence = _finite(result.causal_emergence)
        state.causal_emergence_warming_up = bool(getattr(result, "warming_up", False))

    def _record_reality_monitoring(
        self, raw: RawConversationContext, state: Any
    ) -> None:
        monitor = self.reality_monitor
        if monitor is None:
            return
        monitor.record_signal("user_message", summary=raw.message[:200])
        for thought in (raw.ctm_thoughts or [])[: self.config.prm_internal_thoughts]:
            monitor.record_signal(
                "thought_generator", summary=_text(_read(thought, "content", ""))[:200]
            )
        state.prm_discrimination = monitor.discrimination()

    def _check_event_triggers(self, state: Any) -> None:
        """Mark the baseline due on the four consciousness events."""
        cfg = self.config
        ignition = getattr(state, "ignition_events", None)
        if isinstance(ignition, int) and not isinstance(ignition, bool):
            if (
                self._max_ignition > 0
                and ignition > self._max_ignition * cfg.ignition_spike_factor
            ):
                self._mark_due("ignition_spike")
            self._max_ignition = max(self._max_ignition, ignition)

        depth = getattr(state, "epistemic_depth", None)
        if isinstance(depth, int) and not isinstance(depth, bool):
            if self._max_epistemic_depth is None:
                self._max_epistemic_depth = depth
            elif depth > self._max_epistemic_depth:
                self._max_epistemic_depth = depth
                self._mark_due("new_max_depth")

        integration = _finite(getattr(state, "integration_level", None))
        if integration is not None and integration > cfg.high_integration_threshold:
            self._high_integration_streak += 1
            if self._high_integration_streak >= cfg.high_integration_streak:
                self._mark_due("sustained_integration")
                self._high_integration_streak = 0
        else:
            self._high_integration_streak = 0

        loop_quality = _finite(getattr(state, "beautiful_loop_quality", None))
        if loop_quality is not None and loop_quality > cfg.loop_quality_trigger:
            self._mark_due("loop_quality")

    async def _observe_attention(self) -> None:
        observer = self.attention_observer
        if observer is None or not bool(getattr(observer, "enabled", True)):
            return
        schema = getattr(
            getattr(self.workspace, "attention_schema", None), "schema", None
        )
        await observer.observe(getattr(schema, "current_focus", None))

    def set_developmental_stage(self, stage: Optional[float]) -> None:
        """The stage an assessor measured; None means no stage is known."""
        self._developmental_stage = None if stage is None else float(stage)

    def _apply_developmental(self) -> None:
        modulator = self.developmental
        stage = self._developmental_stage
        if modulator is None or stage is None:
            return
        if not developmental_enabled(self._settings):
            return
        if self._applied_stage == stage:
            return
        modulator.apply_to_workspace(self.workspace, stage, self.pad_model)
        self._applied_stage = stage

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def _build_candidates(
        self, raw: RawConversationContext
    ) -> List[WorkspaceCandidate]:
        candidates = [self._message_to_candidate(raw)]
        candidates.extend(self._memory_to_candidate(m) for m in raw.retrieved_memories)
        candidates.extend(self._thought_to_candidate(t) for t in raw.ctm_thoughts)
        candidates.extend(self._concept_to_candidate(c) for c in raw.retrieved_concepts)
        candidates.extend(self._gist_to_candidate(g) for g in raw.thought_gists)
        pending = self._pending_self_assessment
        if pending is not None:
            self._pending_self_assessment = None
            candidates.append(self._self_assessment_candidate(pending))
        return candidates

    def _message_to_candidate(self, raw: RawConversationContext) -> WorkspaceCandidate:
        """The incoming message: question urgency, emotion words, length, and
        the trust bonus of a verified trusted speaker (a name grants none)."""
        message = raw.message or ""
        words = message.lower().split()
        is_question = "?" in message
        urgency = 0.7 if is_question else 0.5
        emotion_count = sum(1 for w in words if w.strip(".,!?;:'\"") in EMOTION_WORDS)
        emotional_salience = _clip(emotion_count * _EMOTION_WORD_WEIGHT)
        length_factor = _clip(len(words) / _LENGTH_REFERENCE_WORDS)
        activation = urgency * (0.7 + 0.3 * length_factor)
        bonus = self.speaker_trust.trust_bonus(raw.identity)
        return WorkspaceCandidate(
            content=message,
            content_type=MESSAGE_CONTENT_TYPE,
            summary=message[:200],
            source=WorkspaceCandidateSource.SENSORY,
            source_module="conversation",
            activation_level=_clip(activation + bonus),
            emotional_salience=emotional_salience,
            novelty_score=0.3,
            goal_relevance=0.5 if is_question else 0.3,
            recency_bonus=1.0,
            metadata={
                "is_question": is_question,
                "word_count": len(words),
                "emotion_words": emotion_count,
                "trust_bonus": bonus,
            },
        )

    @staticmethod
    def _memory_to_candidate(memory: Any) -> WorkspaceCandidate:
        """A retrieved memory; an unscored one competes at the neutral 0.5.

        A memory without a measured emotional weight competes with no
        emotional salience (0.0, a competition input) and is flagged
        ``emotional_weight_measured`` False in the metadata."""
        score = None
        for key in ("score", "similarity_score", "relevance"):
            score = _finite(_read(memory, key))
            if score is not None:
                break
        text = _text(_read(memory, "content") or _read(memory, "text"))
        measured_weight = _finite(_read(memory, "emotional_weight"))
        weight = 0.0 if measured_weight is None else measured_weight
        return WorkspaceCandidate(
            content=memory,
            content_type="memory",
            summary=text[:200] if text else "retrieved memory",
            source=WorkspaceCandidateSource.MEMORY,
            source_module="memory_retrieval",
            activation_level=_clip(0.5 if score is None else score),
            emotional_salience=_clip(weight),
            novelty_score=0.1,
            recency_bonus=0.2,
            metadata={
                "scored": score is not None,
                "emotional_weight_measured": measured_weight is not None,
            },
        )

    @staticmethod
    def _thought_to_candidate(thought: Any) -> WorkspaceCandidate:
        content = _text(_read(thought, "content"))
        salience = _finite(_read(thought, "salience_score"))
        if salience is None:
            salience = _finite(_read(thought, "salience"))
        intensity = _finite(_read(thought, "intensity"))
        tone = _text(_read(thought, "emotional_tone"))
        if intensity is not None:
            emotional = _clip(intensity * 0.5)
        else:
            emotional = 0.3 if tone else 0.0
        return WorkspaceCandidate(
            content=thought,
            content_type="thought",
            summary=content[:200] if content else "background thought",
            source=WorkspaceCandidateSource.CTM,
            source_module="continuous_thought",
            activation_level=_clip(0.5 if salience is None else salience),
            emotional_salience=emotional,
            novelty_score=0.4,
            metadata={"thought_type": _text(_read(thought, "thought_type"))},
        )

    @staticmethod
    def _concept_to_candidate(concept: Any) -> WorkspaceCandidate:
        name = _text(_read(concept, "name")) or "concept"
        importance = _finite(_read(concept, "importance"))
        importance = 0.5 if importance is None else _clip(importance)
        connections = _finite(_read(concept, "connections"))
        if connections is None:
            connections = _finite(_read(concept, "connection_count")) or 0.0
        connection_factor = min(0.3, connections * 0.05)
        return WorkspaceCandidate(
            content=concept,
            content_type="concept",
            summary=name[:200],
            source=WorkspaceCandidateSource.CONCEPT,
            source_module="knowledge_graph",
            activation_level=_clip(importance * 0.8 + connection_factor),
            emotional_salience=0.1,
            novelty_score=0.2,
            goal_relevance=importance,
            metadata={"connections": connections},
        )

    @staticmethod
    def _gist_to_candidate(gist: Any) -> WorkspaceCandidate:
        """A thought gist, read through the ThoughtGist field names.

        An unmeasured importance competes at the neutral 0.5 and an unmeasured
        retention at full strength (competition inputs, not measurements); the
        metadata flags ``importance_measured`` and ``retention_measured`` say
        which were measured."""
        summary = _text(_read(gist, "summary"))
        importance = _finite(_read(gist, "importance_score"))
        retention = _finite(_read(gist, "retention_strength"))
        emotion = _text(_read(gist, "dominant_emotion"))
        count = _read(gist, "source_count")
        count = (
            int(count) if isinstance(count, int) and not isinstance(count, bool) else 0
        )
        activation = (0.5 if importance is None else _clip(importance)) * (
            1.0 if retention is None else _clip(retention)
        )
        return WorkspaceCandidate(
            content=gist,
            content_type="gist",
            summary=summary[:200] if summary else "thought pattern",
            source=WorkspaceCandidateSource.GIST,
            source_module="thought_consolidation",
            activation_level=_clip(activation),
            emotional_salience=0.3 if emotion else 0.0,
            novelty_score=0.15,
            metadata={
                "source_count": count,
                "dominant_emotion": emotion or None,
                "importance_measured": importance is not None,
                "retention_measured": retention is not None,
            },
        )

    @staticmethod
    def _self_assessment_candidate(pending: Dict[str, Any]) -> WorkspaceCandidate:
        comparison = pending.get("comparison") or {}
        divergences = comparison.get("dimension_divergences") or {}
        notable = comparison.get("notable_divergences") or []
        summary = (
            f"Self-assessment compared with the measured indicators: "
            f"{len(divergences)} dimension(s) compared, {len(notable)} notable "
            f"divergence(s)"
        )
        return WorkspaceCandidate(
            content=pending,
            content_type="self_assessment",
            summary=summary,
            source=WorkspaceCandidateSource.INTERNAL,
            source_module="self_assessment",
            activation_level=0.6,
            emotional_salience=0.1,
            novelty_score=0.5,
            metadata={"notable_divergences": len(notable)},
        )

    def record_self_assessment(self, self_report: Any, comparison: Any) -> None:
        """Keep the latest self-report comparison and inject it once as an
        INTERNAL candidate in the next cycle."""
        entry = {"report": self_report, "comparison": comparison}
        self._last_self_assessment = entry
        self._pending_self_assessment = entry

    # ------------------------------------------------------------------
    # Baseline, assessment and snapshots
    # ------------------------------------------------------------------

    @property
    def baseline_due(self) -> bool:
        """True until the first measurement, after the heartbeat, or after an
        event trigger; ``refresh_baseline`` clears it."""
        if not self._baseline_due:
            self._check_heartbeat()
        return self._baseline_due

    @property
    def baseline_due_reason(self) -> Optional[str]:
        return self._baseline_due_reason if self._baseline_due else None

    async def _default_assess(self) -> Any:
        from mtc.assessment.assessment import ConsciousnessAssessment

        ws = self.workspace
        assessment = ConsciousnessAssessment()
        return await assessment.run_full_assessment(
            global_workspace=ws,
            attention_schema=getattr(ws, "attention_schema", None),
            metacognition=getattr(ws, "metacognition", None),
            active_inference=getattr(ws, "active_inference", None),
            beautiful_loop=getattr(ws, "beautiful_loop", None),
            rpt_measurement=self.rpt_measurement,
            pad_affect_model=self.pad_model,
            causal_emergence_analyzer=self.causal_emergence,
            phi_tracker=getattr(ws, "phi_tracker", None),
            warmup_cycles=self.config.assessment_warmup_cycles,
            normalize=self.config.assessment_normalize,
        )

    async def assess_now(self) -> Optional[Any]:
        """Run the assessment, reusing a report younger than the cache window
        that was taken after the latest cycle (a cycle since makes it stale).

        Returns the report, or None when the assessment failed: the failure is
        counted under ``assessment`` (raised instead under ``DEBUG``), whether
        this is called directly or through ``refresh_baseline``.
        """
        now = self._now()
        if (
            self._last_report is not None
            and self._last_report_at is not None
            and self._last_report_cycle == self._cycle_count
            and (now - self._last_report_at).total_seconds()
            <= self.config.assessment_cache_s
        ):
            return self._last_report
        report = None
        with self._step("assessment"):
            assess = self._assessor or self._default_assess
            report = await assess()
        if report is None:
            return None
        self._last_report = report
        self._last_report_at = self._now()
        self._last_report_cycle = self._cycle_count
        self._report_serial += 1
        return report

    async def refresh_baseline(self, trigger: str = "heartbeat") -> Optional[Any]:
        """Assess, update the baseline and take a snapshot with the same report.

        Returns the report, or None when the assessment or the baseline update
        failed (counted under ``assessment`` or ``baseline_update``, raised
        under ``DEBUG``); after a failed update the baseline is exactly as it
        was and stays due. A snapshot failure is counted under ``snapshot``.
        """
        report = await self.assess_now()
        if report is None:
            return None
        if self._applied_serial != self._report_serial:
            updated: Optional[ConsciousnessBaseline] = None
            with self._step("baseline_update"):
                # Update a copy and swap it in, so a report that breaks the
                # update halfway never leaves a partly updated baseline behind
                candidate = copy.deepcopy(self.baseline)
                candidate.update_from_report(report, self._now())
                updated = candidate
            if updated is None:
                return None
            self.baseline = updated
            self._applied_serial = self._report_serial
        # The same cached report applied again adds no second score; the due
        # flag is still cleared and the snapshot still taken under this trigger
        self._baseline_due = False
        self._baseline_due_reason = None
        if self.snapshot_service is not None:
            with self._step("snapshot"):
                await self.snapshot_service.take_snapshot(trigger, report=report)
        return report

    def sample_phi_state(self) -> bool:
        """Record the last cycle's state into the workspace's PhiTracker (for
        idle sampling between turns). False when there is nothing to record
        or the recording failed (counted under ``phi_sample``, raised under
        ``DEBUG``)."""
        tracker = getattr(self.workspace, "phi_tracker", None)
        if tracker is None or self._last_state is None:
            return False
        recorded = False
        with self._step("phi_sample"):
            tracker.record_module_states(self._last_state)
            recorded = True
        return recorded

    # ------------------------------------------------------------------
    # Modulation
    # ------------------------------------------------------------------

    @staticmethod
    def _ignited(state: Any) -> bool:
        ignition = getattr(state, "ignition_events", 0)
        return isinstance(ignition, int) and ignition > 0

    @staticmethod
    def _is_message_winner(state: Any, message_candidate: Any) -> bool:
        """True only when this cycle ignited and its primary content is this
        turn's message candidate, compared by identity.

        The workspace keeps its primary content across cycles that do not
        ignite, so a content-type check alone would credit a previous turn's
        message; a cycle with no ignition claims no focus at all."""
        if message_candidate is None:
            return False
        return cycle_winner(state) is message_candidate

    @staticmethod
    def _winner_summary(state: Any) -> str:
        """This cycle's winner's summary; empty when nothing ignited.

        A winner whose candidate is flagged ``authored_template`` (authored
        sentences, such as a prospective simulation) is named by its kind,
        never by its words: the words were written, not produced.
        """
        candidate = cycle_winner(state)
        if candidate is None:
            return ""
        if ConsciousnessOrchestrator._template_winner(state):
            return _template_label(candidate, candidate.metadata)
        return _text(getattr(candidate, "summary", ""))

    @staticmethod
    def _template_winner(state: Any) -> bool:
        """True when this cycle's winner is flagged ``authored_template``
        (any truthy value, the same test the workspace applies when it
        withholds the higher-order thought)."""
        candidate = cycle_winner(state)
        metadata = getattr(candidate, "metadata", None)
        return isinstance(metadata, Mapping) and bool(metadata.get("authored_template"))

    def _momentary_consciousness(self, state: Any) -> Optional[float]:
        integration = _finite(getattr(state, "integration_level", None))
        ignition = getattr(state, "ignition_events", None)
        depth = getattr(state, "epistemic_depth", None)
        loop_quality = _finite(getattr(state, "beautiful_loop_quality", None))
        if integration is None or loop_quality is None:
            return None
        if not isinstance(ignition, int) or not isinstance(depth, int):
            return None
        return (
            integration * 0.3
            + _clip(ignition / _IGNITION_REFERENCE) * 0.25
            + _clip(depth / _DEPTH_REFERENCE) * 0.25
            + loop_quality * 0.2
        )

    def _elastic(
        self, profile: Optional[ExpressionProfile], momentary: Optional[float]
    ) -> Optional[ExpressionProfile]:
        """Scale the profile by the momentary measure against the baseline,
        symmetrically, capped at the stretch ceiling."""
        if profile is None or momentary is None:
            return profile
        baseline = _finite(self.baseline.overall_score)
        ceiling = _finite(self.baseline.stretch_ceiling)
        if baseline is None or baseline <= 0.0 or ceiling is None:
            return profile
        effective = min(momentary, ceiling)
        return profile.scaled(effective / baseline)

    def _modulate(
        self,
        raw: RawConversationContext,
        state: Any,
        record: AugmentRecord,
        competed: int,
        message_candidate: Any = None,
    ) -> ModulatedContext:
        ignited = self._ignited(state)
        winner_is_message = self._is_message_winner(state, message_candidate)
        # The winner is this cycle's only when something ignited this cycle
        winner = self._winner_summary(state)
        drives = getattr(
            getattr(
                getattr(self.workspace, "active_inference", None),
                "homeostatic_drives",
                None,
            ),
            "drives",
            None,
        )
        profile = ExpressionProfile.from_baseline(self.baseline, state, drives)
        momentary = self._momentary_consciousness(state)
        profile = self._elastic(profile, momentary)

        template = self._template_winner(state)
        narrative = self._narrative(
            raw, state, winner_is_message, winner, ignited, template
        )
        sections = self._theory_sections(state, competed)
        temperature = self._temperature(state, profile)
        directives = self._directives(
            state, winner_is_message, profile, other_item_won=ignited
        )
        self_awareness = self._self_awareness(profile, momentary)
        absent = self._absent_signals(state)
        self._last_absent = dict(absent)

        return ModulatedContext(
            prompt_context=narrative,
            theory_sections=sections,
            generation_params={"temperature": temperature},
            behavioral_directives=directives,
            consciousness_self_awareness=self_awareness,
            consciousness_state=state,
            workspace_winner="incoming message" if winner_is_message else winner,
            winner_is_message=winner_is_message,
            cycle_ok=True,
            failure=None,
            absent_signals=absent,
            pad_sources_measured=tuple(record.pad_sources_measured),
            extras={
                "cycle": self._cycle_count,
                "candidates_competed": competed,
                "momentary_consciousness": momentary,
                "expression_profile": profile.as_dict() if profile else None,
                "baseline_status": self.baseline.status,
                "augment": {
                    "drive_candidates": record.drive_candidates,
                    "prospective_injected": record.prospective_injected,
                    "bound": record.bound,
                    "capped": record.capped,
                    "gate_arousal_value": record.gate_arousal_value,
                },
            },
        )

    # Layer 1 ------------------------------------------------------------

    def _narrative(
        self,
        raw: RawConversationContext,
        state: Any,
        winner_is_message: bool,
        winner: str,
        ignited: bool = True,
        template: bool = False,
    ) -> str:
        """``template`` is whether this cycle's winner is an authored
        template, whose ``winner`` text is then its kind, not its words."""
        flags = self.narration
        lines: List[str] = []

        if winner_is_message:
            lines.append("The workspace is focused on the incoming message.")
        elif ignited and template:
            lines.append(
                f"The workspace is holding {winner}. "
                "It ranks above the incoming message this cycle."
            )
        elif ignited:
            lines.append(
                f"The workspace is holding another item: {winner or 'unnamed content'}. "
                "It ranks above the incoming message this cycle."
            )
        else:
            lines.append(
                "Nothing ignited in the workspace this cycle; no content was "
                "broadcast."
            )
        ignition = getattr(state, "ignition_events", 0)
        if isinstance(ignition, int) and ignition > 0:
            coverage = _finite(getattr(state, "broadcast_coverage", None))
            coverage_text = f"{coverage:.0%}" if coverage is not None else "unmeasured"
            lines.append(
                f"Ignition this cycle: {ignition} event(s); broadcast coverage "
                f"{coverage_text}."
            )

        if flags.free_energy:
            rank = _finite(getattr(state, "prediction_error_rank", None))
            updates = getattr(
                getattr(state, "active_inference_state", None), "model_updates", 0
            )
            if (
                rank is not None
                and rank > 0.90
                and isinstance(updates, int)
                and updates >= 50
            ):
                error = _finite(getattr(state, "prediction_error", None)) or 0.0
                lines.append(
                    "The generative model's predictions are not matching the "
                    f"current input (prediction error {error:.2f}); the input may "
                    "be unexpected."
                )

        if flags.body_drives:
            urgency = getattr(state, "homeostatic_urgency", None)
            if isinstance(urgency, tuple) and len(urgency) == 2:
                name, level = urgency
                level = _finite(level)
                if level is not None:
                    lines.append(f"The {name} drive is urgent ({level:.2f}).")
            stability = _finite(getattr(state, "protoself_stability", None))
            binding = _finite(getattr(state, "self_world_binding", None))
            knowing = _finite(getattr(state, "feeling_of_knowing", None))
            if None not in (stability, binding, knowing):
                lines.append(
                    f"Body-state stability {stability:.2f}; self-world binding "
                    f"{binding:.2f}; feeling of knowing {knowing:.2f}."
                )
            circadian = self._circadian_line()
            if circadian:
                lines.append(circadian)

        if flags.narrative_identity:
            narrative = _text(getattr(state, "narrative_context", ""))
            if narrative:
                lines.append(narrative)
        if flags.body_drives or flags.somatic or flags.narrative_identity:
            report = _text(getattr(state, "damasio_report", ""))
            if report:
                lines.append(report)

        if flags.self_model:
            depth = getattr(state, "epistemic_depth", 0)
            if isinstance(depth, int) and depth > 1:
                deeply = " deeply" if depth >= 3 else ""
                lines.append(
                    f"Epistemic depth {depth}: the self-model is recursing{deeply} "
                    "this cycle."
                )
            for attr in ("beautiful_loop_report", "self_model_report"):
                report = _text(getattr(state, attr, ""))
                if report:
                    lines.append(report)

        if flags.binding and self.neural_orchestrator is not None:
            context = getattr(self.neural_orchestrator, "get_binding_context", None)
            binding_line = _text(context()) if callable(context) else ""
            if binding_line:
                lines.append(binding_line)

        if flags.affect_line and self.pad_model is not None:
            affect = _text(self.pad_model.generate_context())
            if affect:
                lines.append(affect)

        if flags.speaker_model:
            tom = _text(raw.tom_context)
            if tom:
                lines.append(tom)

        return "\n".join(line for line in lines if line and not _names_phi(line))

    def _circadian_line(self) -> str:
        """Time of day and alertness from the workspace's wall-clock protoself."""
        protoself = getattr(getattr(self.workspace, "damasio", None), "protoself", None)
        if protoself is None:
            return ""
        period = getattr(protoself, "time_of_day", None)
        alertness = _finite(getattr(protoself, "alertness", None))
        if period is None or alertness is None:
            return ""
        return f"Local time of day: {period}; circadian alertness {alertness:.0%}."

    # Theory sections -----------------------------------------------------

    def _theory_sections(self, state: Any, competed: int) -> TheorySections:
        admitted = len(getattr(state, "workspace_contents", None) or [])
        ignited = getattr(state, "ignition_events", 0)
        integration = _finite(getattr(state, "integration_level", None))
        focus = _text(getattr(state, "attention_focus", "")) or "none"
        line = (
            f"Cycle {self._cycle_count}: {competed} candidates competed, "
            f"{admitted} admitted, {ignited} ignited; integration level "
            f"{'unmeasured' if integration is None else f'{integration:.2f}'}; "
            f"attention focus {focus}."
        )
        level = getattr(state, "dominant_hierarchy_level", None)
        if isinstance(level, int):
            line += f" Dominant hierarchy level {level}."
        flags = self.narration
        # The attention schema's and the metacognition module's own reports
        # speak in the first person, so they reach their sections only under
        # the self-model flag; otherwise a neutral measured summary does. The
        # active inference report carries only the lines its module's free
        # energy and body-drive flags allow.
        if flags.self_model:
            attention = _phi_free(getattr(state, "attention_report", ""))
            metacognitive = _phi_free(getattr(state, "metacognitive_report", ""))
        else:
            attention = _phi_free(
                _attention_summary(getattr(state, "attention_schema", None))
            )
            metacognitive = _phi_free(_metacognition_summary(state))
        active_inference = (
            _phi_free(getattr(state, "active_inference_report", ""))
            if flags.free_energy or flags.body_drives
            else None
        )
        return TheorySections(
            neural_state=self._substrate_block() if flags.binding else None,
            attention_guidance=attention,
            metacognitive_insights=metacognitive,
            active_inference_context=active_inference,
            global_workspace_context=_phi_free(line),
        )

    def _substrate_block(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Finite metrics of the substrates whose latest output is real, read
        through the neural orchestrator's public ``verification_neural_data``;
        a metric or block named after Phi is dropped. None otherwise."""
        neural = self.neural_orchestrator
        builder = getattr(neural, "verification_neural_data", None)
        if not callable(builder):
            return None
        blocks = builder()
        if not isinstance(blocks, Mapping):
            return None
        real: Dict[str, Dict[str, Any]] = {}
        for name, block in blocks.items():
            if not isinstance(block, Mapping) or not block.get("is_real"):
                continue
            if _names_phi(name):
                continue
            metrics = {
                key: value
                for key, value in block.items()
                if key != "is_real"
                and not _names_phi(key)
                and _finite(value) is not None
            }
            if metrics:
                real[name] = metrics
        return real or None

    # Layer 2 ------------------------------------------------------------

    def _temperature(self, state: Any, profile: Optional[ExpressionProfile]) -> float:
        cfg = self.config
        temperature = cfg.baseline_temperature
        rank = _finite(getattr(state, "prediction_error_rank", None))
        if rank is not None and rank > 0.92:
            temperature += 0.1
        urgency = getattr(state, "homeostatic_urgency", None)
        if isinstance(urgency, tuple) and len(urgency) == 2:
            name, level = urgency
            level = _finite(level)
            if name == "curiosity" and level is not None and level > 0.5:
                temperature += 0.05
        capacity = _finite(
            getattr(
                getattr(state, "attention_schema", None),
                "attention_capacity_used",
                None,
            )
        )
        if capacity is not None and capacity > 0.9:
            temperature -= 0.1
        arousal = _finite(getattr(state, "pad_arousal", None))
        if arousal is not None and arousal > 0.6:
            temperature += (arousal - 0.6) * 0.3
        floor, ceiling = cfg.min_temperature, cfg.max_temperature
        if profile is not None:
            average = profile.average()
            if average < 0.45:
                band = (0.5, 0.9)
            elif average > 0.65:
                band = (0.6, 1.2)
            else:
                band = (0.5, 1.1)
            # The profile's band narrows the configured bounds and never
            # widens them; a band entirely outside them yields to them
            low, high = max(band[0], floor), min(band[1], ceiling)
            if low <= high:
                floor, ceiling = low, high
        return _clip(temperature, floor, ceiling)

    # Layer 3 ------------------------------------------------------------

    def _directives(
        self,
        state: Any,
        winner_is_message: bool,
        profile: Optional[ExpressionProfile],
        *,
        other_item_won: bool = True,
    ) -> List[str]:
        """``other_item_won`` is whether this cycle ignited at all: a cycle
        with no ignition holds nothing, so the inner-state permission to lead
        with another item is not granted."""
        flags = self.narration
        directives: List[str] = []

        if flags.body_drives:
            urgency = getattr(state, "homeostatic_urgency", None)
            if isinstance(urgency, tuple) and len(urgency) == 2:
                name, level = urgency
                level = _finite(level) or 0.0
                drive_lines = {
                    "attention_budget": (
                        0.7,
                        "The attention budget is depleted; you may keep the reply "
                        "brief and focused.",
                    ),
                    "curiosity": (
                        0.7,
                        "The curiosity drive is urgent; you may ask questions and "
                        "follow tangents.",
                    ),
                    "social_connection": (
                        0.8,
                        "The social-connection drive is urgent; you may be warm and "
                        "ask about the other person.",
                    ),
                    "coherence": (
                        0.7,
                        "The coherence drive is urgent; you may slow down and "
                        "clarify before adding new material.",
                    ),
                    "safety": (
                        0.7,
                        "The safety drive is urgent; you may be cautious and name "
                        "your uncertainty.",
                    ),
                }
                threshold_and_line = drive_lines.get(name)
                if threshold_and_line is not None and level > threshold_and_line[0]:
                    directives.append(threshold_and_line[1])

        if flags.inner_state and other_item_won and not winner_is_message:
            directives.append(
                "Attention is held by something other than the message; you may "
                "lead with it before addressing what was said."
            )

        if flags.free_energy:
            rank = _finite(getattr(state, "prediction_error_rank", None))
            if rank is not None and rank > 0.97:
                directives.append(
                    "This input is surprising relative to recent predictions; you "
                    "may say so."
                )

        if flags.self_model:
            depth = getattr(state, "epistemic_depth", 0)
            if isinstance(depth, int) and depth >= 3:
                directives.append(
                    "The self-model is in a deep recursive state; you may share "
                    "observations about your own processing."
                )
            if bool(getattr(state, "is_field_evidencing", False)):
                directives.append(
                    "A recursive pattern in the processing was detected; you may "
                    "comment on it if it is relevant."
                )

        if flags.somatic:
            core = (getattr(state, "damasio_state", None) or {}).get("core")
            valence = (
                _finite(core.get("somatic_valence"))
                if isinstance(core, Mapping)
                else None
            )
            if valence is not None and valence < -0.3:
                directives.append(
                    "The somatic markers for this content are negative; you may "
                    "express caution."
                )
            elif valence is not None and valence > 0.7:
                directives.append(
                    "The somatic markers for this content are strongly positive; "
                    "you may let that come through."
                )

        if flags.inner_state and profile is not None:
            if profile.introspective_depth > 0.6:
                directives.append("You may articulate your own thought process.")
            elif profile.introspective_depth < 0.45:
                directives.append("You may keep to the present and speak directly.")
            if profile.emotional_range > 0.6:
                directives.append("You may express nuanced or mixed emotions.")
            elif profile.emotional_range < 0.45:
                directives.append("You may express emotions simply.")

        return [line for line in directives if not _names_phi(line)]

    # Layer 4 ------------------------------------------------------------

    _DEPTH_WORDS = (
        (0.30, "a faint"),
        (0.40, "a quiet"),
        (0.50, "a gentle"),
        (0.60, "a moderate"),
        (0.70, "a clear"),
        (0.80, "a vivid"),
        (float("inf"), "a strong"),
    )

    def _self_awareness(
        self, profile: Optional[ExpressionProfile], momentary: Optional[float]
    ) -> str:
        if not self.narration.inner_state:
            return ""
        baseline = self.baseline
        overall = _finite(baseline.overall_score)
        if overall is None or profile is None:
            return (
                "Inner state: no consciousness baseline has been measured yet; "
                "nothing is assumed about its level."
            )
        depth_word = "a moderate"
        for threshold, word in self._DEPTH_WORDS:
            if overall < threshold:
                depth_word = word
                break
        ranked = sorted(profile.as_dict().items(), key=lambda kv: kv[1], reverse=True)
        names = ExpressionProfile.GRADIENT_NAMES
        lines = [
            "Inner state:",
            f"The measured baseline places the current level of conscious processing "
            f"at {depth_word} level.",
            f"Strongest gradients: {names[ranked[0][0]]} and {names[ranked[1][0]]}.",
        ]
        weakest_name, weakest_value = ranked[-1]
        if weakest_value < 0.45:
            lines.append(f"The weakest gradient is {names[weakest_name]}.")
        trend = baseline.trend
        if trend == "rising":
            lines.append("The baseline has been rising over recent assessments.")
        elif trend == "falling":
            lines.append("The baseline has been falling over recent assessments.")
        elif trend == "stable":
            lines.append("The baseline has been stable over recent assessments.")
        else:
            lines.append(
                "The baseline trend is not yet measurable (fewer than five "
                "assessments)."
            )
        if momentary is not None:
            if momentary > overall + 0.05:
                lines.append("This cycle's momentary measure is above the baseline.")
            elif momentary < overall - 0.1:
                lines.append("This cycle's momentary measure is below the baseline.")
        return "\n".join(line for line in lines if not _names_phi(line))

    # Absent signals -------------------------------------------------------

    def _absent_reason(self, name: str, state: Any) -> str:
        ws = self.workspace
        for prefix, step in _PRODUCER_STEPS:
            if name.startswith(prefix) and step in self._cycle_failures:
                return "failed"
        if name in (
            "dominant_hierarchy_level",
            "level_winner_counts",
            "cross_level_promotions",
        ):
            hierarchical = (
                getattr(ws, "competition_mode", None) == COMPETITION_HIERARCHICAL
            )
            return "no_input" if hierarchical else "disabled"
        if name in ("thalamic_gate_rate", "thalamic_gated_count", "arousal_source"):
            gate = getattr(getattr(ws, "bottleneck", None), "thalamic_gate", None)
            return "no_input" if gate is not None else "disabled"
        if name == "architectural_phi":
            if getattr(ws, "phi_tracker", None) is None:
                return "no_producer"
            return "warming_up"
        if name == "phi_warming_up":
            return "no_producer"
        if name.startswith("rpt_"):
            if self.rpt_measurement is None:
                return "no_producer"
            # On a cadence cycle the step ran without error and measured nothing
            due = self._cycle_count % self.config.rpt_every_n_cycles == 0
            return "no_input" if due else "cadence_not_reached"
        if name == "prm_discrimination":
            if self.reality_monitor is None:
                return "no_producer"
            return self._prm_reason()
        if name.startswith("causal_emergence"):
            if (
                self.causal_emergence is None
                or getattr(ws, "phi_tracker", None) is None
            ):
                return "no_producer"
            due = self._cycle_count % self.config.causal_emergence_every_n_cycles == 0
            return "warming_up" if due else "cadence_not_reached"
        if name.startswith("pad_"):
            if not self.augmenter.config.pad_from_state:
                return "disabled"
            if _PAD_STEP in self._cycle_failures and not getattr(
                self.augmenter, "pad_computed", False
            ):
                return "failed"
            # The augmenter computed PAD and a dimension had no measured input
            return "no_input"
        if name == "prediction_error_rank":
            return "warming_up"
        if name == "primary_content":
            return "no_input"
        if name == "integration_level":
            # Measured from the neural orchestrator's signals only; no stand-in
            return "no_producer" if self.neural_orchestrator is None else "no_input"
        if name in (
            "broadcast_coverage",
            "binding_quality",
            "autobiographic_continuity",
        ):
            return "no_input"
        if name in ("protoself_stability", "feeling_of_knowing"):
            # The first body update has no previous body state to compare with
            return "warming_up" if self._cycle_count <= 1 else "no_input"
        return "no_producer"

    def _prm_reason(self) -> str:
        """``no_input`` once both kinds were seen and every classified signal
        came without an attribution confidence (nothing supplies one, and no
        prior stands in for it); ``warming_up`` before both kinds were seen."""
        reader = getattr(self.reality_monitor, "get_statistics", None)
        stats = reader() if callable(reader) else None
        if isinstance(stats, Mapping):
            both = (stats.get("external_count") or 0) > 0 and (
                stats.get("internal_count") or 0
            ) > 0
            if both and stats.get("default_attribution_fraction") == 1.0:
                return "no_input"
        return "warming_up"

    def _absent_signals(self, state: Any) -> Dict[str, str]:
        absent: Dict[str, str] = {}
        for name in NAMED_STATE_FIELDS:
            if getattr(state, name, None) is None:
                absent[name] = self._absent_reason(name, state)
        return absent

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _modulation_status(self) -> Dict[str, Any]:
        manager = getattr(self.workspace, "theory_interactions", None)
        if manager is None:
            return {"enabled": False, "modulations": []}
        status = manager.get_status()
        values = status.get("modulations", {})
        return {
            "enabled": True,
            "cycle_count": status.get("cycle_count"),
            "modulations": [
                {"name": name, "value": value, "consumed": name in CONSUMED_MODULATIONS}
                for name, value in values.items()
            ],
        }

    def first_run_report(self) -> Dict[str, Any]:
        """What still warms up: Phi transitions, the prediction-error baseline,
        the RPT and causal emergence cadences and the baseline's age."""
        cfg = self.config
        tracker = getattr(self.workspace, "phi_tracker", None)
        cycles = self._cycle_count

        def until(every: int) -> int:
            return every - (cycles % every) if cycles % every else every

        rank = getattr(self._last_state, "prediction_error_rank", None)
        return {
            "cycles": cycles,
            "phi": {
                "available": tracker is not None,
                "transitions_recorded": getattr(tracker, "transitions_recorded", None),
                "min_transitions": getattr(tracker, "min_transitions", None),
                "warming_up": (
                    bool(getattr(tracker, "warming_up", True)) if tracker else None
                ),
            },
            "prediction_error_baseline": {
                "ranked": rank is not None,
                "last_rank": _finite(rank),
            },
            "rpt": {
                "available": self.rpt_measurement is not None,
                "every_n_cycles": cfg.rpt_every_n_cycles,
                "cycles_until_next": until(cfg.rpt_every_n_cycles),
                "measurements": getattr(
                    self.rpt_measurement, "measurement_count", None
                ),
            },
            "causal_emergence": {
                "available": self.causal_emergence is not None and tracker is not None,
                "every_n_cycles": cfg.causal_emergence_every_n_cycles,
                "cycles_until_next": until(cfg.causal_emergence_every_n_cycles),
            },
            "baseline": {
                "status": self.baseline.status,
                "measured": self.baseline.measured,
                "age_s": self.baseline.age_s(self._now()),
                "due": self.baseline_due,
                "due_reason": self.baseline_due_reason,
            },
        }

    def status(self) -> Dict[str, Any]:
        return {
            "cycles": self._cycle_count,
            "failures": dict(self._failures),
            "timings": self.profiler.summary(),
            "baseline": self.baseline.as_dict(),
            "baseline_due": self.baseline_due,
            "baseline_due_reason": self.baseline_due_reason,
            "theory_modulations": self._modulation_status(),
            "absent_signals": dict(self._last_absent),
            "pad_sources_measured": list(self._last_pad_sources),
            "narration": self.narration.as_dict(),
            "narration_applied": dict(self._narration_applied),
            "augmenter": self.augmenter.status(),
            "config": asdict(self.config),
            "developmental_stage": self._developmental_stage,
            "self_assessment_pending": self._pending_self_assessment is not None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        baseline = self.baseline
        return {
            "scalars": {
                "cycle_count": self._cycle_count,
                "baseline_overall_score": baseline.overall_score,
                "baseline_theory_scores": dict(baseline.theory_scores),
                "baseline_indicator_scores": dict(baseline.indicator_scores),
                "baseline_status": baseline.status,
                "baseline_coverage": baseline.coverage,
                "baseline_stretch_ceiling": baseline.stretch_ceiling,
                "baseline_trend": baseline.trend,
                "baseline_last_updated": (
                    baseline.last_updated.isoformat() if baseline.last_updated else None
                ),
                "baseline_score_history": list(baseline.score_history),
                "max_ignition": self._max_ignition,
                "max_epistemic_depth": self._max_epistemic_depth,
                "high_integration_streak": self._high_integration_streak,
                "failures": dict(self._failures),
                "developmental_stage": self._developmental_stage,
            },
            "arrays": {},
            "histories": {},
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        scalars = (state or {}).get("scalars", {}) or {}
        self._cycle_count = int(scalars.get("cycle_count", 0))
        baseline = ConsciousnessBaseline()
        baseline.overall_score = _finite(scalars.get("baseline_overall_score"))
        theories = scalars.get("baseline_theory_scores") or {}
        baseline.theory_scores = {theory: None for theory in THEORIES}
        for theory, value in theories.items():
            baseline.theory_scores[str(theory)] = _finite(value)
        baseline.indicator_scores = {
            str(name): _finite(value)
            for name, value in (scalars.get("baseline_indicator_scores") or {}).items()
        }
        baseline.status = str(scalars.get("baseline_status") or BASELINE_UNMEASURED)
        baseline.coverage = _finite(scalars.get("baseline_coverage"))
        baseline.stretch_ceiling = _finite(scalars.get("baseline_stretch_ceiling"))
        trend = scalars.get("baseline_trend")
        baseline.trend = trend if trend in ("rising", "falling", "stable") else None
        baseline.score_history = [
            value
            for value in (
                _finite(item) for item in (scalars.get("baseline_score_history") or [])
            )
            if value is not None
        ]
        stamp = scalars.get("baseline_last_updated")
        if isinstance(stamp, str) and stamp:
            try:
                parsed = datetime.fromisoformat(stamp)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                baseline.last_updated = parsed
            except ValueError:
                baseline.last_updated = None
        if baseline.status == INSUFFICIENT_STATUS:
            # A state written before the insufficient guard may carry a number
            # for a report that measured nothing; it is restored unmeasured.
            baseline.overall_score = None
            baseline.stretch_ceiling = None
            baseline.score_history = []
            baseline.trend = None
        self.baseline = baseline
        # A report cached before the restore describes the replaced state
        self._last_report = None
        self._last_report_at = None
        self._last_report_cycle = None
        self._applied_serial = None
        self._baseline_due = not baseline.measured
        self._baseline_due_reason = "never_measured" if self._baseline_due else None
        self._max_ignition = int(scalars.get("max_ignition", 0) or 0)
        depth = scalars.get("max_epistemic_depth")
        self._max_epistemic_depth = int(depth) if isinstance(depth, int) else None
        self._high_integration_streak = int(scalars.get("high_integration_streak", 0))
        failures = scalars.get("failures") or {}
        self._failures = {str(k): int(v) for k, v in failures.items()}
        stage = scalars.get("developmental_stage")
        self._developmental_stage = _finite(stage)
        self._applied_stage = None


__all__ = [
    "THEORIES",
    "NAMED_STATE_FIELDS",
    "CONSUMED_MODULATIONS",
    "EMOTION_WORDS",
    "ConsciousnessBaseline",
    "ExpressionProfile",
    "OrchestratorConfig",
    "apply_narration",
    "ConsciousnessOrchestrator",
]
