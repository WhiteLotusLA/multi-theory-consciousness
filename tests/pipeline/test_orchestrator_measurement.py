"""Orchestrator measurement: the heartbeat, the assessment surface, the event
triggers, snapshots, honest None and the first-run report."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    COMPETITION_HIERARCHICAL,
    ConsciousnessState,
    EnhancedGlobalWorkspace,
)
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.core.config import Settings
from mtc.core.narration import NarrationFlags
from mtc.pipeline.consciousness_orchestrator import (
    ConsciousnessOrchestrator,
    OrchestratorConfig,
)
from mtc.pipeline.context_types import RawConversationContext
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig
from mtc.pipeline.speaker_trust import SpeakerTrust

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


class _Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


def _report(overall=0.62, status="measured", **theories):
    scores = {"GWT": 0.7, "IIT": 0.4, "AST": 0.6, "HOT": 0.65, "FEP": 0.5}
    scores.update(theories)
    indicators = {
        "global_broadcast": SimpleNamespace(score=0.7, status="measured"),
        "agency": SimpleNamespace(score=0.0, status="insufficient_data"),
    }
    return SimpleNamespace(
        overall_score=overall,
        theory_scores=scores,
        indicator_results=indicators,
        status=status,
        coverage=0.9,
    )


class _Assessor:
    def __init__(self, report=None):
        self.calls = 0
        self.report = report if report is not None else _report()

    async def __call__(self):
        self.calls += 1
        return self.report


class _Snapshots:
    def __init__(self):
        self.taken = []

    async def take_snapshot(self, trigger, report=None):
        self.taken.append((trigger, report))
        return SimpleNamespace(id=len(self.taken), trigger=trigger)

    async def maybe_snapshot(self, *args, **kwargs):
        raise AssertionError("maybe_snapshot must never be called")


class _ScriptedWorkspace:
    """Returns the next scripted state on every cycle."""

    def __init__(self, states):
        self.states = list(states)
        self.competition_mode = COMPETITION_HIERARCHICAL
        self.bottleneck = SimpleNamespace(thalamic_gate=None)
        self.phi_tracker = None
        self.theory_interactions = None
        self.damasio = SimpleNamespace(protoself=None)
        self.active_inference = SimpleNamespace(
            homeostatic_drives=SimpleNamespace(
                drives={}, get_urgent_drives=lambda **k: []
            )
        )
        self.attention_schema = SimpleNamespace(
            schema=SimpleNamespace(current_focus=None)
        )
        self.cycle_count = 0

    async def process_consciousness_cycle(self, candidates, neural_signals=None):
        self.cycle_count += 1
        return self.states.pop(0)

    def set_arousal(self, value):
        pass

    def get_statistics(self):
        return {"cycle_count": self.cycle_count}


def _state(**overrides):
    values = dict(
        is_conscious=False,
        primary_content=None,
        workspace_contents=[],
        ignition_events=0,
        integration_level=0.2,
        broadcast_coverage=0.0,
        attention_focus="none",
        attention_distribution={},
        stream_position=0,
        epistemic_depth=1,
        beautiful_loop_quality=0.3,
    )
    values.update(overrides)
    return ConsciousnessState(**values)


def _build(workspace=None, clock=None, **kwargs):
    cfg = Settings(DEBUG=False)
    workspace = workspace or EnhancedGlobalWorkspace(
        phi_tracker=PhiTracker(min_transitions=3), thalamic_gate_enabled=False
    )
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.all_off(), workspace, pad_model=PADAffectModel()
    )
    return ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=NarrationFlags(),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
        clock=clock or _Clock(),
        **kwargs,
    )


def _turn(orchestrator, message="hello"):
    raw = RawConversationContext(
        message=message, identity=orchestrator.speaker_trust.verify("Alice", None)
    )
    return asyncio.run(orchestrator.process(raw))


def test_heartbeat_marks_baseline_due_not_inline():
    clock = _Clock()
    assessor = _Assessor()
    # Identical states every cycle, so no event trigger can mark the baseline
    # due; only the heartbeat does.
    orchestrator = _build(
        workspace=_ScriptedWorkspace([_state() for _ in range(10)]),
        clock=clock,
        assessor=assessor,
        config=OrchestratorConfig(heartbeat_s=600),
    )
    assert orchestrator.baseline_due is True
    assert orchestrator.baseline_due_reason == "never_measured"
    _turn(orchestrator)
    assert assessor.calls == 0  # never inline
    report = asyncio.run(orchestrator.refresh_baseline())
    assert report is assessor.report
    assert assessor.calls == 1
    assert orchestrator.baseline_due is False
    assert orchestrator.baseline.overall_score == pytest.approx(0.62)
    clock.advance(300)
    _turn(orchestrator)
    assert orchestrator.baseline_due is False
    clock.advance(301)
    _turn(orchestrator)
    assert orchestrator.baseline_due is True
    assert orchestrator.baseline_due_reason == "heartbeat"
    assert assessor.calls == 1  # the turn still runs no assessment
    asyncio.run(orchestrator.refresh_baseline())
    assert assessor.calls == 2
    assert orchestrator.baseline_due is False
    # Inside the cache window the same report comes back without a new run
    clock.advance(5)
    asyncio.run(orchestrator.refresh_baseline())
    assert assessor.calls == 2
    clock.advance(20)
    asyncio.run(orchestrator.refresh_baseline())
    assert assessor.calls == 3


def test_refresh_baseline_runs_full_assessment_with_phi_tracker(monkeypatch):
    import mtc.assessment.assessment as assessment_module

    seen = {}

    class _FakeAssessment:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        async def run_full_assessment(self, **kwargs):
            seen["kwargs"] = kwargs
            return _report()

    monkeypatch.setattr(assessment_module, "ConsciousnessAssessment", _FakeAssessment)
    rpt = object()
    analyzer = object()
    orchestrator = _build(rpt_measurement=rpt, causal_emergence=analyzer)
    _turn(orchestrator)
    report = asyncio.run(orchestrator.refresh_baseline())
    assert report is not None
    kwargs = seen["kwargs"]
    workspace = orchestrator.workspace
    assert kwargs["phi_tracker"] is workspace.phi_tracker
    assert kwargs["global_workspace"] is workspace
    assert kwargs["attention_schema"] is workspace.attention_schema
    assert kwargs["metacognition"] is workspace.metacognition
    assert kwargs["active_inference"] is workspace.active_inference
    assert kwargs["beautiful_loop"] is workspace.beautiful_loop
    assert kwargs["rpt_measurement"] is rpt
    assert kwargs["causal_emergence_analyzer"] is analyzer
    assert kwargs["pad_affect_model"] is orchestrator.pad_model
    # The configured warmup, the assessment's own default of 25
    assert kwargs["warmup_cycles"] == orchestrator.config.assessment_warmup_cycles
    assert kwargs["warmup_cycles"] == 25
    assert kwargs["normalize"] is False
    assert orchestrator.baseline.status == "measured"


def test_assessment_warmup_is_configured_not_the_cycle_count(monkeypatch):
    import mtc.assessment.assessment as assessment_module

    seen = []

    class _FakeAssessment:
        async def run_full_assessment(self, **kwargs):
            seen.append(kwargs["warmup_cycles"])
            return _report()

    monkeypatch.setattr(assessment_module, "ConsciousnessAssessment", _FakeAssessment)
    clock = _Clock()
    orchestrator = _build(clock=clock)
    for _ in range(3):
        _turn(orchestrator)
    asyncio.run(orchestrator.refresh_baseline())
    clock.advance(60)  # past the assessment cache
    for _ in range(4):
        _turn(orchestrator)
    asyncio.run(orchestrator.refresh_baseline())
    assert orchestrator.status()["cycles"] == 7
    # The bounded histories are scaled by the configured warmup, never by the
    # running cycle count
    assert seen == [25, 25]

    tuned = _build(
        clock=_Clock(), config=OrchestratorConfig(assessment_warmup_cycles=7)
    )
    for _ in range(2):
        _turn(tuned)
    asyncio.run(tuned.refresh_baseline())
    assert seen[-1] == 7
    with pytest.raises(ValueError, match="assessment_warmup_cycles"):
        OrchestratorConfig(assessment_warmup_cycles=0)


def test_event_triggers_mark_due():
    states = [
        _state(ignition_events=2),
        _state(ignition_events=4),  # over 1.5 times the running maximum
        _state(ignition_events=1, epistemic_depth=3),  # a new maximum depth
        *[_state(integration_level=0.9, epistemic_depth=1) for _ in range(5)],
        _state(beautiful_loop_quality=0.9, epistemic_depth=1),
        _state(),
    ]
    assessor = _Assessor()
    orchestrator = _build(workspace=_ScriptedWorkspace(states), assessor=assessor)
    asyncio.run(orchestrator.refresh_baseline())
    assert orchestrator.baseline_due is False

    def turn_and_reason():
        _turn(orchestrator)
        reason = orchestrator.baseline_due_reason
        if orchestrator.baseline_due:
            asyncio.run(orchestrator.refresh_baseline())
        return reason

    assert turn_and_reason() is None  # the first maximum is not a spike
    assert turn_and_reason() == "ignition_spike"
    assert turn_and_reason() == "new_max_depth"
    for _ in range(4):
        assert turn_and_reason() is None
    assert turn_and_reason() == "sustained_integration"
    assert turn_and_reason() == "loop_quality"
    assert turn_and_reason() is None


def test_snapshot_taken_with_refreshed_report():
    snapshots = _Snapshots()
    assessor = _Assessor()
    orchestrator = _build(assessor=assessor, snapshot_service=snapshots)
    report = asyncio.run(orchestrator.refresh_baseline("heartbeat"))
    assert snapshots.taken == [("heartbeat", report)]
    assert assessor.calls == 1
    clock_report = asyncio.run(orchestrator.refresh_baseline("event"))
    assert snapshots.taken[-1] == ("event", clock_report)
    assert assessor.calls == 1  # cached report, no second assessment


def test_insufficient_overall_kept_as_none():
    assessor = _Assessor(_report(overall=None, status="insufficient_data"))
    orchestrator = _build(assessor=assessor)
    report = asyncio.run(orchestrator.refresh_baseline())
    assert report is not None
    baseline = orchestrator.baseline
    assert baseline.overall_score is None
    assert baseline.measured is False
    assert baseline.status == "insufficient_data"
    assert baseline.stretch_ceiling is None
    assert baseline.score_history == []
    assert baseline.trend is None
    assert baseline.theory_scores["GWT"] == pytest.approx(0.7)
    assert baseline.theory_scores["RPT"] is None  # never scored
    assert baseline.indicator_scores == {"global_broadcast": 0.7, "agency": None}
    assert orchestrator.baseline_due is False
    context = _turn(orchestrator)
    assert context.extras["expression_profile"] is None
    assert context.extras["baseline_status"] == "insufficient_data"


def test_sample_phi_state_records_last_state():
    orchestrator = _build()
    tracker = orchestrator.workspace.phi_tracker
    assert orchestrator.sample_phi_state() is False
    _turn(orchestrator)
    before = tracker.transitions_recorded
    assert orchestrator.sample_phi_state() is True
    assert tracker.transitions_recorded == before + 1
    assert orchestrator.sample_phi_state() is True
    assert tracker.transitions_recorded == before + 2
    assert orchestrator.status()["timings"]["phi_sample"]["count"] == 2
    no_tracker = _build(workspace=EnhancedGlobalWorkspace(thalamic_gate_enabled=False))
    _turn(no_tracker)
    assert no_tracker.sample_phi_state() is False


def test_one_assessment_records_one_timing_sample():
    assessor = _Assessor()
    orchestrator = _build(assessor=assessor)
    asyncio.run(orchestrator.refresh_baseline())
    assert assessor.calls == 1
    assert orchestrator.status()["timings"]["assessment"]["count"] == 1


def test_first_run_report_lists_warmups():
    clock = _Clock()
    assessor = _Assessor()
    orchestrator = _build(
        clock=clock,
        assessor=assessor,
        rpt_measurement=SimpleNamespace(
            measurement_count=0,
            set_oscillatory_binding_stats=lambda stats: None,
            measure_full=lambda **k: None,
        ),
        causal_emergence=SimpleNamespace(analyze_modules=lambda tracker: None),
        config=OrchestratorConfig(
            rpt_every_n_cycles=5, causal_emergence_every_n_cycles=4
        ),
    )
    report = orchestrator.first_run_report()
    assert report["cycles"] == 0
    assert report["phi"] == {
        "available": True,
        "transitions_recorded": 0,
        "min_transitions": 3,
        "warming_up": True,
    }
    assert report["prediction_error_baseline"] == {"ranked": False, "last_rank": None}
    assert (
        report["rpt"]["every_n_cycles"] == 5 and report["rpt"]["cycles_until_next"] == 5
    )
    assert report["causal_emergence"]["cycles_until_next"] == 4
    assert report["baseline"]["age_s"] is None
    assert report["baseline"]["measured"] is False
    assert report["baseline"]["due"] is True

    _turn(orchestrator)
    _turn(orchestrator)
    asyncio.run(orchestrator.refresh_baseline())
    clock.advance(42)
    report = orchestrator.first_run_report()
    assert report["cycles"] == 2
    assert report["phi"]["transitions_recorded"] == 1
    assert report["rpt"]["cycles_until_next"] == 3
    assert report["causal_emergence"]["cycles_until_next"] == 2
    assert report["baseline"]["age_s"] == pytest.approx(42.0)
    assert report["baseline"]["measured"] is True
    assert report["baseline"]["due"] is False
