"""
Cycle Augmenter
===============

The per-cycle additions around ``EnhancedGlobalWorkspace.process_consciousness_cycle``
that need no conversation input, gathered in one place so the live pipeline and
the research runners share them:

- ``before_cycle``: urgent homeostatic drives become HOMEOSTATIC candidates
  (urgency as activation, above ``drive_urgency_threshold``); the prospective
  simulation generated after the previous cycle is injected as an INTERNAL
  candidate; cross-modal binding adds bound percepts; a pre-attentive cap keeps
  the ``candidate_cap`` candidates with the highest activation plus emotional
  salience.
- ``after_cycle``: the PAD coordinate is computed from the cycle's state (and
  the neural affect of ``CycleInputs``) and recorded on the state; the thalamic
  gate's arousal is fed from the protoself (``set_arousal``), left to the PAD
  arousal recorded on the state (the workspace reads it next cycle), or left to
  the workspace; the prospective simulator may generate one candidate for the
  next cycle.

Every switch is a ``CycleAugmenterConfig`` field, mirrored by the ``cycle``
group of ``experiments/param_schema.toml``. The defaults describe the live
pipeline; ``mtc.core.runner_support.build_cycle_augmenter`` pins every switch
off for the runners unless the layered overrides set it, and ``all_off`` is an
identity: the candidates come back unchanged and the state is not touched, so
a seeded run stays bit-identical.

Honest measurement: a PAD input absent in a cycle is left out of
``AugmentRecord.pad_sources_measured`` and contributes nothing: each PAD
dimension is the weighted mean over its measured inputs, and when a dimension
has none no coordinate is recorded (the state's PAD fields stay None and
``AugmentRecord.pad_state`` is None; the measured list is still kept). One PAD
model and one prospective simulator: a caller that also reads them hands its
own over with ``adopt_pad_model`` and ``adopt_prospective`` (a different one
already in use raises), or reads the augmenter's. No Settings are read here.
Persisted state (``to_state_dict``): the counters and the binder's and
simulator's own state; the simulator's topic words are conversation fragments
and therefore private.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from mtc.consciousness.crossmodal_binding import CrossModalBinder
from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.pad_affect import (
    AROUSAL_WEIGHTS,
    DOMINANCE_WEIGHTS,
    PLEASURE_WEIGHTS,
    PADAffectModel,
    PADSources,
    PADState,
)
from mtc.consciousness.prospective_memory import (
    ProspectiveContext,
    ProspectiveSimulator,
)

logger = logging.getLogger(__name__)

GATE_AROUSAL_SOURCES: Tuple[str, ...] = ("protoself", "pad", "workspace")

# Ignition events at which the PAD ignition rate saturates, and the epistemic
# depth that maps to 1.0 (the reference rules the PAD model was tuned with).
_IGNITION_SATURATION = 5.0
_DEPTH_REFERENCE = 5.0


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class CycleAugmenterConfig:
    """The cycle switches; the defaults describe the live pipeline."""

    drive_candidates: bool = True
    drive_urgency_threshold: float = 0.5
    prospective_candidates: bool = True
    crossmodal_binding: bool = True
    gate_arousal: str = "protoself"
    pad_from_state: bool = True
    candidate_cap: int = 20

    def __post_init__(self) -> None:
        if self.gate_arousal not in GATE_AROUSAL_SOURCES:
            raise ValueError(
                f"gate_arousal must be one of {GATE_AROUSAL_SOURCES}, "
                f"got {self.gate_arousal!r}"
            )
        threshold = _finite(self.drive_urgency_threshold)
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "drive_urgency_threshold must be in [0, 1], "
                f"got {self.drive_urgency_threshold!r}"
            )
        if isinstance(self.candidate_cap, bool) or int(self.candidate_cap) < 0:
            raise ValueError(
                f"candidate_cap must be 0 (no cap) or positive, got {self.candidate_cap!r}"
            )
        object.__setattr__(self, "drive_urgency_threshold", float(threshold))
        object.__setattr__(self, "candidate_cap", int(self.candidate_cap))
        for name in (
            "drive_candidates",
            "prospective_candidates",
            "crossmodal_binding",
            "pad_from_state",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))

    @classmethod
    def all_off(cls) -> "CycleAugmenterConfig":
        """Every addition off: the v0.4.0 runner behaviour, an identity."""
        return cls(
            drive_candidates=False,
            prospective_candidates=False,
            crossmodal_binding=False,
            gate_arousal="workspace",
            pad_from_state=False,
            candidate_cap=0,
        )

    @classmethod
    def from_overrides(
        cls, overrides: Optional[Mapping[str, Any]], *, pinned_off: bool = False
    ) -> "CycleAugmenterConfig":
        """Build from the ``cycle`` override table.

        With ``pinned_off`` the base is ``all_off`` and only the keys the
        table sets move; otherwise the base is the live defaults. An unknown
        key raises ``ValueError``.
        """
        base = cls.all_off() if pinned_off else cls()
        table = dict(overrides or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(table) - known)
        if unknown:
            raise ValueError(f"unknown cycle switches: {unknown}")
        return replace(base, **table)

    def as_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass
class CycleInputs:
    """What a caller may hand the augmenter for one cycle.

    ``neural_affect`` carries the substrate's measured PAD inputs when a neural
    orchestrator produced real output: ``snn_valence`` in [-1, 1] and
    ``snn_spike_rate`` in [0, 1], each present only when it was measured. None
    means no SNN source. The caller hands over substrate output only, never the
    emotions it fed the substrate; any other key is ignored.
    """

    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    now: Optional[datetime] = None
    neural_affect: Optional[Dict[str, float]] = None


@dataclass
class AugmentRecord:
    """What one cycle's hooks did."""

    drive_candidates: int = 0
    prospective_injected: int = 0
    bound: int = 0
    capped: int = 0
    gate_arousal_value: Optional[float] = None
    pad_sources_measured: Tuple[str, ...] = ()
    pad_state: Optional[PADState] = None


_PAD_DIMENSION_WEIGHTS: Tuple[Tuple[str, Mapping[str, float]], ...] = (
    ("pleasure", PLEASURE_WEIGHTS),
    ("arousal", AROUSAL_WEIGHTS),
    ("dominance", DOMINANCE_WEIGHTS),
)


def _pad_gate_cause(record: AugmentRecord) -> str:
    """Why the PAD gate found no arousal with PAD from the state on.

    Names the PAD dimensions that had no measured input this cycle (any one
    of them leaves the coordinate unrecorded); when every dimension had one,
    says only that no finite PAD arousal reached the state.
    """
    measured = set(record.pad_sources_measured)
    missing = [
        dimension
        for dimension, weights in _PAD_DIMENSION_WEIGHTS
        if not any(name in measured for name in weights)
    ]
    if missing:
        return "no measured input this cycle for " + ", ".join(missing)
    return "no finite PAD arousal on the state this cycle"


def extract_pad_sources(
    state: Any,
    workspace: Any = None,
    neural_affect: Optional[Mapping[str, Any]] = None,
) -> Tuple[PADSources, Tuple[str, ...]]:
    """PAD inputs from a cycle's state and its producers.

    Returns the sources and the names of the inputs that were measured; an
    absent input keeps its declared ``PADSources`` default and is not named,
    so a caller that passes the names to ``PADAffectModel.compute`` leaves it
    out of the coordinate.
    """
    sources = PADSources()
    measured: List[str] = []

    damasio = getattr(state, "damasio_state", None)
    damasio = damasio if isinstance(damasio, Mapping) else {}
    core = damasio.get("core")
    if isinstance(core, Mapping):
        somatic = _finite(core.get("somatic_valence"))
        if somatic is not None:
            sources.somatic_valence = _clip(somatic, -1.0, 1.0)
            measured.append("somatic_valence")

    drives = getattr(
        getattr(
            getattr(workspace, "active_inference", None), "homeostatic_drives", None
        ),
        "drives",
        None,
    )
    if isinstance(drives, Mapping) and drives:
        deviations = [
            _finite(drive.get_deviation())
            for drive in drives.values()
            if callable(getattr(drive, "get_deviation", None))
        ]
        deviations = [value for value in deviations if value is not None]
        if deviations:
            satisfaction = 1.0 - sum(deviations) / len(deviations)
            sources.homeostatic_satisfaction = _clip(
                2.0 * satisfaction - 1.0, -1.0, 1.0
            )
            measured.append("homeostatic_satisfaction")

    if isinstance(neural_affect, Mapping):
        valence = _finite(neural_affect.get("snn_valence"))
        if valence is not None:
            sources.snn_valence = _clip(valence, -1.0, 1.0)
            measured.append("snn_valence")
        rate = _finite(neural_affect.get("snn_spike_rate"))
        if rate is not None:
            sources.snn_spike_rate = _clip(rate, 0.0, 1.0)
            measured.append("snn_spike_rate")

    protoself = damasio.get("protoself")
    primordial = (
        protoself.get("primordial_feelings") if isinstance(protoself, Mapping) else None
    )
    if isinstance(primordial, Mapping):
        arousal = _finite(primordial.get("arousal"))
        if arousal is not None:
            sources.protoself_arousal = _clip(arousal, 0.0, 1.0)
            measured.append("protoself_arousal")

    ignition = getattr(state, "ignition_events", None)
    if isinstance(ignition, int) and not isinstance(ignition, bool):
        sources.ignition_rate = _clip(ignition / _IGNITION_SATURATION, 0.0, 1.0)
        measured.append("ignition_rate")

    if getattr(state, "inference_result", None) is not None:
        error = _finite(getattr(state, "prediction_error", None))
        if error is not None:
            sources.prediction_error = _clip(error, 0.0, 1.0)
            measured.append("prediction_error")

    ai_state = getattr(state, "active_inference_state", None)
    confidence = _finite(getattr(ai_state, "meta_confidence", None))
    if confidence is not None:
        sources.agency_score = _clip(confidence, 0.0, 1.0)
        measured.append("agency_score")

    schema = getattr(state, "attention_schema", None)
    voluntary = _finite(getattr(schema, "voluntary_ratio", None))
    if voluntary is not None:
        sources.voluntary_attention_ratio = _clip(voluntary, 0.0, 1.0)
        measured.append("voluntary_attention_ratio")

    if getattr(state, "beautiful_loop_moment", None) is not None:
        depth = _finite(getattr(state, "epistemic_depth", None))
        if depth is not None:
            sources.epistemic_depth_normalized = _clip(
                depth / _DEPTH_REFERENCE, 0.0, 1.0
            )
            measured.append("epistemic_depth_normalized")

    return sources, tuple(measured)


def protoself_arousal(state: Any) -> Optional[float]:
    """The protoself's primordial arousal recorded on a state, if any."""
    damasio = getattr(state, "damasio_state", None)
    if not isinstance(damasio, Mapping):
        return None
    protoself = damasio.get("protoself")
    if not isinstance(protoself, Mapping):
        return None
    primordial = protoself.get("primordial_feelings")
    if not isinstance(primordial, Mapping):
        return None
    value = _finite(primordial.get("arousal"))
    return None if value is None else _clip(value, 0.0, 1.0)


class CycleAugmenter:
    """The before and after hooks around one workspace cycle.

    Args:
        config: The switches.
        workspace: The ``EnhancedGlobalWorkspace`` whose drives, gate and
            state the hooks read and feed.
        pad_model: PAD model fed when ``pad_from_state`` is on; None builds a
            default ``PADAffectModel`` the first time it is needed.
        prospective: Prospective simulator; None builds one when
            ``prospective_candidates`` is on.
        binder: Cross-modal binder; None builds one when ``crossmodal_binding``
            is on.
        clock: Returns the current aware UTC time.
    """

    def __init__(
        self,
        config: CycleAugmenterConfig,
        workspace: Any,
        pad_model: Optional[PADAffectModel] = None,
        prospective: Optional[ProspectiveSimulator] = None,
        binder: Optional[CrossModalBinder] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not isinstance(config, CycleAugmenterConfig):
            raise TypeError("config must be a CycleAugmenterConfig")
        self.config = config
        self.workspace = workspace
        self._pad_model = pad_model
        self._prospective = prospective
        self._binder = binder
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending_prospective: List[WorkspaceCandidate] = []
        self._pending = AugmentRecord()
        self._before_calls = 0
        self._after_calls = 0
        self._totals = {
            "drive_candidates": 0,
            "prospective": 0,
            "bound": 0,
            "capped": 0,
        }
        self._pad_gate_warned = False
        self._pad_computed = False

    # ------------------------------------------------------------------
    # Components, built on first use when their switch is on
    # ------------------------------------------------------------------

    @property
    def pad_model(self) -> Optional[PADAffectModel]:
        """The PAD model in use (built on first use when ``pad_from_state``)."""
        return self._pad_model

    @property
    def prospective(self) -> Optional[ProspectiveSimulator]:
        return self._prospective

    @property
    def binder(self) -> Optional[CrossModalBinder]:
        return self._binder

    def adopt_pad_model(self, pad_model: Any) -> Any:
        """Use ``pad_model`` as the one PAD model and return it.

        A caller that reads PAD (the orchestrator's affect line and the
        assessment) and this augmenter, which computes it, must share one
        model; a different model already in use raises ``ValueError``.
        """
        if pad_model is None:
            raise ValueError("adopt_pad_model needs a PAD model")
        if self._pad_model is not None and self._pad_model is not pad_model:
            raise ValueError(
                "the augmenter already computes PAD through a different PAD model"
            )
        self._pad_model = pad_model
        return pad_model

    def adopt_prospective(self, simulator: Any) -> Any:
        """Use ``simulator`` as the one prospective simulator and return it.

        The goals a caller sets and the simulations this augmenter generates
        must meet in one simulator; a different one already in use raises
        ``ValueError``.
        """
        if simulator is None:
            raise ValueError("adopt_prospective needs a prospective simulator")
        if self._prospective is not None and self._prospective is not simulator:
            raise ValueError(
                "the augmenter already uses a different prospective simulator"
            )
        self._prospective = simulator
        return simulator

    def _ensure_pad_model(self) -> PADAffectModel:
        if self._pad_model is None:
            self._pad_model = PADAffectModel()
        return self._pad_model

    def _ensure_prospective(self) -> ProspectiveSimulator:
        if self._prospective is None:
            self._prospective = ProspectiveSimulator(clock=self._clock)
        return self._prospective

    def _ensure_binder(self) -> CrossModalBinder:
        if self._binder is None:
            self._binder = CrossModalBinder()
        return self._binder

    def _now(self, inputs: Optional[CycleInputs]) -> datetime:
        moment = inputs.now if inputs is not None and inputs.now is not None else None
        if moment is None:
            moment = self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def drive_candidates(self) -> List[WorkspaceCandidate]:
        """HOMEOSTATIC candidates for the drives above the urgency threshold."""
        drives = getattr(
            getattr(self.workspace, "active_inference", None),
            "homeostatic_drives",
            None,
        )
        if drives is None or not callable(getattr(drives, "get_urgent_drives", None)):
            return []
        urgent = drives.get_urgent_drives(threshold=self.config.drive_urgency_threshold)
        candidates: List[WorkspaceCandidate] = []
        for name, urgency in urgent:
            level = _finite(urgency)
            if level is None:
                continue
            level = _clip(level, 0.0, 1.0)
            candidates.append(
                WorkspaceCandidate(
                    content={"drive": name, "urgency": level},
                    content_type="homeostatic_drive",
                    summary=f"Urgent drive: {name} ({level:.2f})",
                    source=WorkspaceCandidateSource.HOMEOSTATIC,
                    source_module="active_inference",
                    activation_level=level,
                    emotional_salience=_clip(level * 0.7, 0.0, 1.0),
                    novelty_score=0.1,
                    goal_relevance=level,
                    metadata={"drive": name, "urgency": level},
                )
            )
        return candidates

    def before_cycle(
        self,
        candidates: List[WorkspaceCandidate],
        inputs: Optional[CycleInputs] = None,
    ) -> List[WorkspaceCandidate]:
        """Return the candidates the cycle should compete over."""
        self._before_calls += 1
        record = AugmentRecord()
        out = list(candidates)
        cfg = self.config

        if cfg.drive_candidates:
            drives = self.drive_candidates()
            out.extend(drives)
            record.drive_candidates = len(drives)

        if cfg.prospective_candidates and self._pending_prospective:
            out.extend(self._pending_prospective)
            record.prospective_injected = len(self._pending_prospective)
            self._pending_prospective = []

        if cfg.crossmodal_binding:
            before = len(out)
            out = list(
                self._ensure_binder().bind(out, now=self._now(inputs).timestamp())
            )
            record.bound = max(0, len(out) - before)

        if cfg.candidate_cap > 0 and len(out) > cfg.candidate_cap:
            ranked = sorted(
                out,
                key=lambda c: c.activation_level + c.emotional_salience,
                reverse=True,
            )
            record.capped = len(out) - cfg.candidate_cap
            out = ranked[: cfg.candidate_cap]

        self._totals["drive_candidates"] += record.drive_candidates
        self._totals["prospective"] += record.prospective_injected
        self._totals["bound"] += record.bound
        self._totals["capped"] += record.capped
        self._pending = record
        return out

    @property
    def pad_computed(self) -> bool:
        """True when the latest ``after_cycle`` call ran the PAD computation
        to its end (with or without a coordinate); False when PAD from the
        state is off, before the first call, or when the computation raised.
        A failure later in the same call leaves it True."""
        return self._pad_computed

    def after_cycle(
        self, state: Any, inputs: Optional[CycleInputs] = None
    ) -> AugmentRecord:
        """Record PAD on the state, feed the gate and prepare the next cycle."""
        self._after_calls += 1
        self._pad_computed = False
        record = self._pending
        self._pending = AugmentRecord()
        cfg = self.config

        if cfg.pad_from_state:
            affect = inputs.neural_affect if inputs is not None else None
            sources, measured = extract_pad_sources(state, self.workspace, affect)
            # Each dimension is computed over its measured inputs only; a
            # dimension with none leaves the coordinate unrecorded (None on
            # the state, no PAD state), while the measured list is still kept
            pad_state = self._ensure_pad_model().compute(sources, measured=measured)
            self._pad_computed = True
            record.pad_sources_measured = measured
            record.pad_state = pad_state
            if pad_state is not None:
                state.pad_pleasure = pad_state.pleasure
                state.pad_arousal = pad_state.arousal
                state.pad_dominance = pad_state.dominance
                state.pad_label = pad_state.label

        if cfg.gate_arousal == "protoself":
            arousal = protoself_arousal(state)
            if arousal is not None and callable(
                getattr(self.workspace, "set_arousal", None)
            ):
                self.workspace.set_arousal(arousal)
                record.gate_arousal_value = arousal
        elif cfg.gate_arousal == "pad":
            # The workspace reads the PAD arousal recorded on this state when it
            # derives the next cycle's arousal; nothing to set here.
            record.gate_arousal_value = _finite(getattr(state, "pad_arousal", None))
            if record.gate_arousal_value is None and not self._pad_gate_warned:
                self._pad_gate_warned = True
                cause = (
                    _pad_gate_cause(record)
                    if cfg.pad_from_state
                    else "pad_from_state off"
                )
                logger.warning(
                    "gate_arousal is pad but no PAD arousal was recorded on the "
                    "state (%s); the gate keeps its own source",
                    cause,
                )

        if cfg.prospective_candidates:
            simulator = self._ensure_prospective()
            messages = list(inputs.recent_messages) if inputs is not None else []
            goals = list(inputs.goals) if inputs is not None else []
            if messages:
                last = messages[-1]
                text = str(last.get("content", last.get("message", "")) or "")
                if text:
                    simulator.update_topics(text)
            context = ProspectiveContext(
                recent_messages=messages,
                goals=goals,
                arousal=_finite(getattr(state, "pad_arousal", None)),
                now=self._now(inputs),
            )
            self._pending_prospective = list(simulator.maybe_generate(context))

        return record

    # ------------------------------------------------------------------
    # Status and persistence
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "before_calls": self._before_calls,
            "after_calls": self._after_calls,
            "totals": dict(self._totals),
            "pending_prospective": len(self._pending_prospective),
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Counters plus the binder's and simulator's own state."""
        scalars: Dict[str, Any] = {
            "before_calls": self._before_calls,
            "after_calls": self._after_calls,
        }
        scalars.update({f"total_{name}": value for name, value in self._totals.items()})
        histories: Dict[str, Any] = {}
        if self._binder is not None:
            histories["binder"] = self._binder.to_state_dict()
        if self._prospective is not None:
            histories["prospective"] = self._prospective.to_state_dict()
        return {"scalars": scalars, "arrays": {}, "histories": histories}

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        scalars = (state or {}).get("scalars", {}) or {}
        self._before_calls = int(scalars.get("before_calls", 0))
        self._after_calls = int(scalars.get("after_calls", 0))
        for name in self._totals:
            self._totals[name] = int(scalars.get(f"total_{name}", 0))
        histories = (state or {}).get("histories", {}) or {}
        if "binder" in histories:
            self._ensure_binder().from_state_dict(histories["binder"])
        if "prospective" in histories:
            self._ensure_prospective().from_state_dict(histories["prospective"])
        self._pending_prospective = []
