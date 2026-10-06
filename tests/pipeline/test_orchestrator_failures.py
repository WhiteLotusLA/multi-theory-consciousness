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
