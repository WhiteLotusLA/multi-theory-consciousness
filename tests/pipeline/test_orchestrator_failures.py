"""Orchestrator failure accounting: counted by step, raised under DEBUG,
honest unavailability, status, named absences and persistence."""

import asyncio
import logging
from dataclasses import fields
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    ConsciousnessState,
    EnhancedGlobalWorkspace,
)
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.core.config import Settings
from mtc.core.narration import NarrationFlags
from mtc.pipeline.consciousness_orchestrator import ConsciousnessOrchestrator
from mtc.pipeline.context_types import ABSENT_REASONS, RawConversationContext
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig
from mtc.pipeline.speaker_trust import SpeakerTrust

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


class _FailingManager:
    def __init__(self):
        self.calls = 0

    def on_cycle_complete(self):
        self.calls += 1
        raise RuntimeError("disk full")


class _FailingMonitor:
    def record_signal(self, *args, **kwargs):
        raise ValueError("bad signal")

    def discrimination(self):
        return None


class _BrokenWorkspace(EnhancedGlobalWorkspace):
    async def process_consciousness_cycle(self, candidates, neural_signals=None, **kw):
        raise RuntimeError("workspace exploded")


def _build(workspace=None, settings=None, augmenter_config=None, **kwargs):
    cfg = settings or Settings(DEBUG=False)
    workspace = workspace or EnhancedGlobalWorkspace(
        phi_tracker=PhiTracker(min_transitions=50), thalamic_gate_enabled=True
    )
    augmenter = CycleAugmenter(
        augmenter_config or CycleAugmenterConfig(),
        workspace,
        pad_model=PADAffectModel(),
    )
    return ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=NarrationFlags(),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
        clock=lambda: T0,
        **kwargs,
    )


def _turn(orchestrator, message="hello"):
    raw = RawConversationContext(
        message=message, identity=orchestrator.speaker_trust.verify("Alice", None)
    )
    return asyncio.run(orchestrator.process(raw))


def test_step_failure_counted_and_cycle_continues(caplog):
    manager = _FailingManager()
    orchestrator = _build(state_manager=manager, reality_monitor=_FailingMonitor())
    with caplog.at_level(
        logging.WARNING, logger="mtc.pipeline.consciousness_orchestrator"
    ):
        first = _turn(orchestrator)
        second = _turn(orchestrator)
    assert first.cycle_ok and second.cycle_ok
    assert first.prompt_context and second.prompt_context
    failures = orchestrator.status()["failures"]
    assert failures == {"persistence": 2, "reality_monitor": 2}
    assert manager.calls == 2
    # The first failure of each step is logged once, by type name only
    warnings = [r for r in caplog.records if "failed" in r.getMessage()]
    assert len(warnings) == 2
    assert all(
        "RuntimeError" in r.getMessage() or "ValueError" in r.getMessage()
        for r in warnings
    )
    assert not any("disk full" in r.getMessage() for r in warnings)


def test_debug_setting_raises():
    orchestrator = _build(
        state_manager=_FailingManager(), settings=Settings(DEBUG=True)
    )
    with pytest.raises(RuntimeError, match="disk full"):
        _turn(orchestrator)
    assert orchestrator.status()["failures"] == {"persistence": 1}
    broken = _build(workspace=_BrokenWorkspace(), settings=Settings(DEBUG=True))
    with pytest.raises(RuntimeError, match="workspace exploded"):
        _turn(broken)


def _broken_record(state):
    raise RuntimeError("tracker broke")


def test_sample_phi_state_failure_counted_and_raised_under_debug():
    orchestrator = _build()
    _turn(orchestrator)
    orchestrator.workspace.phi_tracker.record_module_states = _broken_record
    assert orchestrator.sample_phi_state() is False
    assert orchestrator.status()["failures"] == {"phi_sample": 1}

    loud = _build(settings=Settings(DEBUG=True))
    _turn(loud)
    loud.workspace.phi_tracker.record_module_states = _broken_record
    with pytest.raises(RuntimeError, match="tracker broke"):
        loud.sample_phi_state()
    assert loud.status()["failures"] == {"phi_sample": 1}


def test_direct_assessment_failure_counted_and_raised_under_debug():
    async def failing():
        raise RuntimeError("assessment broke")

    orchestrator = _build(assessor=failing)
    assert asyncio.run(orchestrator.assess_now()) is None
    assert orchestrator.status()["failures"] == {"assessment": 1}
    assert asyncio.run(orchestrator.refresh_baseline()) is None
    assert orchestrator.status()["failures"] == {"assessment": 2}
    assert orchestrator.baseline_due is True

    loud = _build(assessor=failing, settings=Settings(DEBUG=True))
    with pytest.raises(RuntimeError, match="assessment broke"):
        asyncio.run(loud.assess_now())
    assert loud.status()["failures"] == {"assessment": 1}


def test_baseline_update_failure_counted_and_baseline_untouched():
    malformed = SimpleNamespace(
        overall_score=0.6,
        theory_scores={"GWT": 0.7},
        indicator_results=["not", "a", "mapping"],
        status="measured",
        coverage=0.9,
    )

    async def assessor():
        return malformed

    orchestrator = _build(assessor=assessor)
    assert asyncio.run(orchestrator.refresh_baseline()) is None
    assert orchestrator.status()["failures"] == {"baseline_update": 1}
    baseline = orchestrator.baseline
    assert baseline.measured is False
    assert baseline.status == "unmeasured"
    # No partial update: the theory scores were never touched either
    assert set(baseline.theory_scores.values()) == {None}
    assert orchestrator.baseline_due is True

    loud = _build(assessor=assessor, settings=Settings(DEBUG=True))
    with pytest.raises(AttributeError):
        asyncio.run(loud.refresh_baseline())
    assert loud.status()["failures"] == {"baseline_update": 1}


def test_total_failure_returns_unavailable_without_narrative():
    orchestrator = _build(workspace=_BrokenWorkspace())
    context = _turn(orchestrator)
    assert context.cycle_ok is False
    assert context.failure == "cycle_failed"
    assert context.prompt_context == ""
    assert context.behavioral_directives == []
    assert context.consciousness_self_awareness == ""
    assert context.consciousness_state is None
    assert context.theory_sections.as_dict() == {}
    assert context.generation_params == {"temperature": 0.7}
    assert orchestrator.status()["failures"] == {"cycle": 1}
    assert orchestrator.status()["cycles"] == 0


def test_status_exposes_counters_and_timings():
    orchestrator = _build(state_manager=_FailingManager())
    _turn(orchestrator)
    _turn(orchestrator)
    status = orchestrator.status()
    assert status["cycles"] == 2
    assert status["failures"] == {"persistence": 2}
    timings = status["timings"]
    assert timings["process"]["count"] == 2
    assert timings["cycle"]["count"] == 2
    assert timings["process"]["p95"] >= timings["cycle"]["mean"] > 0.0
    assert timings["modulation"]["count"] == 2
    assert status["baseline"]["measured"] is False
    assert status["baseline_due"] is True
    assert status["narration"] == NarrationFlags().as_dict()
    assert status["augmenter"]["before_calls"] == 2
    assert status["pad_sources_measured"]
    assert "absent_signals" in status
    assert status["config"]["heartbeat_s"] == 900.0


def test_absent_signals_named_after_first_cycle():
    rpt = SimpleNamespace(
        measurement_count=0,
        set_oscillatory_binding_stats=lambda stats: None,
        measure_full=lambda **k: None,
    )
    analyzer = SimpleNamespace(analyze_modules=lambda tracker: None)
    orchestrator = _build(rpt_measurement=rpt, causal_emergence=analyzer)
    context = _turn(orchestrator)
    state = context.consciousness_state
    optional = [
        item.name for item in fields(ConsciousnessState) if item.default is None
    ]
    for name in optional:
        value = getattr(state, name)
        if value is None:
            assert name in context.absent_signals, name
            assert context.absent_signals[name] in ABSENT_REASONS, name
        else:
            assert name not in context.absent_signals, name
    absent = context.absent_signals
    assert absent["architectural_phi"] == "warming_up"
    assert absent["rpt_deep_recurrence"] == "cadence_not_reached"
    assert absent["causal_emergence"] == "cadence_not_reached"
    assert absent["prm_discrimination"] == "no_producer"
    assert absent["prediction_error_rank"] == "warming_up"
    assert "pad_arousal" not in absent  # measured through the augmenter
    assert (
        "thalamic_gate_rate" not in absent or absent["thalamic_gate_rate"] == "no_input"
    )
    assert orchestrator.status()["absent_signals"] == absent

    # A gate-less, flat, tracker-less workspace names its absences as disabled
    bare = _build(
        workspace=EnhancedGlobalWorkspace(
            thalamic_gate_enabled=False, competition_mode="flat"
        ),
        augmenter_config=CycleAugmenterConfig.all_off(),
    )
    context = _turn(bare)
    absent = context.absent_signals
    assert absent["thalamic_gate_rate"] == "disabled"
    assert absent["arousal_source"] == "disabled"
    assert absent["dominant_hierarchy_level"] == "disabled"
    assert absent["architectural_phi"] == "no_producer"
    assert absent["pad_arousal"] == "disabled"
    assert absent["rpt_deep_recurrence"] == "no_producer"


class _FlakyRPT:
    """Raises on the cycles listed in ``fail_on``; otherwise returns the
    scripted metrics (None: ran cleanly and measured nothing)."""

    def __init__(self, fail_on, metrics=None):
        self.fail_on = set(fail_on)
        self.metrics = metrics
        self.measurement_count = 0
        self.calls = 0

    def set_oscillatory_binding_stats(self, stats):
        pass

    def measure_full(self, **kwargs):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("rpt broke")
        return self.metrics


class _FlakyMonitor:
    def __init__(self, fail_on):
        self.fail_on = set(fail_on)
        self.calls = 0

    def record_signal(self, *args, **kwargs):
        pass

    def discrimination(self):
        self.calls += 1
        if self.calls in self.fail_on:
            raise ValueError("monitor broke")
        return None


class _FlakyAnalyzer:
    def __init__(self, fail_on):
        self.fail_on = set(fail_on)
        self.calls = 0

    def analyze_modules(self, tracker):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("analyzer broke")
        return None


def test_failed_producer_is_named_failed_not_warming_up():
    from mtc.pipeline.consciousness_orchestrator import OrchestratorConfig

    rpt = _FlakyRPT(fail_on={1})
    monitor = _FlakyMonitor(fail_on={1})
    analyzer = _FlakyAnalyzer(fail_on={1})
    orchestrator = _build(
        rpt_measurement=rpt,
        reality_monitor=monitor,
        causal_emergence=analyzer,
        config=OrchestratorConfig(
            rpt_every_n_cycles=1, causal_emergence_every_n_cycles=1
        ),
    )
    failed = _turn(orchestrator)
    assert failed.cycle_ok
    absent = failed.absent_signals
    assert absent["rpt_deep_recurrence"] == "failed"
    assert absent["rpt_superficial_recurrence"] == "failed"
    assert absent["prm_discrimination"] == "failed"
    assert absent["causal_emergence"] == "failed"
    assert set(orchestrator.status()["failures"]) == {
        "rpt",
        "reality_monitor",
        "causal_emergence",
    }
    # The next clean cycle returns to the configured reasons: the cadence
    # step ran and measured nothing (no input), the monitor and the analyzer
    # are warming up
    clean = _turn(orchestrator)
    absent = clean.absent_signals
    assert absent["rpt_deep_recurrence"] == "no_input"
    assert absent["prm_discrimination"] == "warming_up"
    assert absent["causal_emergence"] == "warming_up"
    assert "failed" not in absent.values()

    # A failed PAD computation is named failed, then measured again
    class _FlakyPAD(PADAffectModel):
        calls = 0

        def compute(self, sources, measured=None):
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("pad broke")
            return super().compute(sources, measured)

    pad = _build(augmenter_config=CycleAugmenterConfig())
    pad.augmenter._pad_model = _FlakyPAD()
    first = _turn(pad)
    assert first.absent_signals["pad_arousal"] == "failed"
    assert pad.status()["failures"] == {"augment_after": 1}
    second = _turn(pad)
    assert "pad_arousal" not in second.absent_signals


def test_cadence_cycle_that_measured_nothing_reads_no_input():
    from mtc.pipeline.consciousness_orchestrator import OrchestratorConfig

    orchestrator = _build(
        rpt_measurement=_FlakyRPT(fail_on=set()),
        config=OrchestratorConfig(rpt_every_n_cycles=2),
    )
    off_cadence, on_cadence = _turn(orchestrator), _turn(orchestrator)
    assert off_cadence.absent_signals["rpt_deep_recurrence"] == "cadence_not_reached"
    assert on_cadence.absent_signals["rpt_deep_recurrence"] == "no_input"


class _StateWorkspace(EnhancedGlobalWorkspace):
    """A real workspace whose cycle returns a state with no dominance input:
    no meta reading, no classified attention shift, no loop moment."""

    async def process_consciousness_cycle(self, candidates, neural_signals=None, **kw):
        state = await super().process_consciousness_cycle(
            candidates, neural_signals=neural_signals, **kw
        )
        state.active_inference_state = None
        state.attention_schema = None
        state.beautiful_loop_moment = None
        return state


def test_unmeasured_pad_dimension_is_named_no_input():
    orchestrator = _build(workspace=_StateWorkspace(thalamic_gate_enabled=False))
    context = _turn(orchestrator)
    state = context.consciousness_state
    assert state.pad_dominance is None and state.pad_label is None
    assert "ignition_rate" in context.pad_sources_measured
    for name in ("pad_pleasure", "pad_arousal", "pad_dominance", "pad_label"):
        assert context.absent_signals[name] == "no_input", name
    assert "augment_after" not in orchestrator.status()["failures"]


def test_state_dict_round_trip():
    orchestrator = _build(state_manager=_FailingManager())
    _turn(orchestrator)
    _turn(orchestrator)
    report = SimpleNamespace(
        overall_score=0.55,
        theory_scores={"GWT": 0.6, "HOT": 0.5},
        indicator_results={
            "global_broadcast": SimpleNamespace(score=0.6, status="measured")
        },
        status="partial",
        coverage=0.5,
    )
    orchestrator.baseline.update_from_report(report, T0)
    orchestrator.set_developmental_stage(3)
    saved = orchestrator.to_state_dict()
    assert set(saved) == {"scalars", "arrays", "histories"}
    scalars = saved["scalars"]
    assert scalars["cycle_count"] == 2
    assert scalars["failures"] == {"persistence": 2}
    assert scalars["baseline_overall_score"] == pytest.approx(0.55)
    assert scalars["baseline_last_updated"] == T0.isoformat()

    restored = _build()
    restored.from_state_dict(saved)
    assert restored.status()["cycles"] == 2
    assert restored.status()["failures"] == {"persistence": 2}
    baseline = restored.baseline
    assert baseline.overall_score == pytest.approx(0.55)
    assert baseline.theory_scores["GWT"] == pytest.approx(0.6)
    assert baseline.theory_scores["RPT"] is None
    assert baseline.status == "partial"
    assert baseline.coverage == pytest.approx(0.5)
    assert baseline.stretch_ceiling == pytest.approx(0.66)
    assert baseline.last_updated == T0
    assert baseline.score_history == [0.55]
    assert baseline.indicator_scores == {"global_broadcast": 0.6}
    assert restored.baseline_due is False
    assert restored.status()["developmental_stage"] == 3.0
    assert restored.to_state_dict() == saved

    empty = _build()
    empty.from_state_dict({"scalars": {}, "arrays": {}, "histories": {}})
    assert empty.baseline.measured is False
    assert empty.baseline_due is True


class _UnmeasuredPAD(PADAffectModel):
    """A PAD model whose dimensions have no measured input this cycle."""

    def compute(self, sources, measured=None):
        return None


def _break_gate_arousal(monkeypatch):
    """Makes the augmenter's gate arousal sub-step raise; it runs after the
    PAD computation inside the same orchestrator step."""
    import mtc.pipeline.cycle_augmenter as augmenter_module

    def _raise(state):
        raise RuntimeError("gate arousal broke")

    monkeypatch.setattr(augmenter_module, "protoself_arousal", _raise)


def test_disabled_pad_is_not_named_failed_by_another_augment_failure(monkeypatch):
    _break_gate_arousal(monkeypatch)
    orchestrator = _build(augmenter_config=CycleAugmenterConfig(pad_from_state=False))
    context = _turn(orchestrator)
    assert orchestrator.status()["failures"] == {"augment_after": 1}
    assert context.absent_signals["pad_arousal"] == "disabled"
    assert context.absent_signals["pad_label"] == "disabled"


def test_unmeasured_pad_before_a_later_augment_failure_reads_no_input(monkeypatch):
    _break_gate_arousal(monkeypatch)
    orchestrator = _build()
    orchestrator.augmenter._pad_model = _UnmeasuredPAD()
    context = _turn(orchestrator)
    assert orchestrator.status()["failures"] == {"augment_after": 1}
    # The PAD computation itself returned (no measured input); the failure
    # belongs to the gate arousal sub-step that ran after it
    assert context.absent_signals["pad_arousal"] == "no_input"
    assert context.absent_signals["pad_pleasure"] == "no_input"
