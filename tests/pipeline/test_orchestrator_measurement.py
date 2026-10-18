"""Orchestrator measurement: the heartbeat, the assessment surface, the event
triggers, snapshots, honest None and the first-run report."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


def test_zero_overall_with_insufficient_status_is_not_measured():
    """The real shape of an unmeasured report (a number with the status
    insufficient_data) is not taken as a measurement."""
    assessor = _Assessor(_report(overall=0.0, status="insufficient_data"))
    orchestrator = _build(assessor=assessor)
    report = asyncio.run(orchestrator.refresh_baseline())
    assert report is assessor.report
    baseline = orchestrator.baseline
    assert baseline.overall_score is None
    assert baseline.measured is False
    assert baseline.status == "insufficient_data"
    assert baseline.stretch_ceiling is None
    assert baseline.score_history == []
    assert baseline.trend is None
    assert baseline.as_dict()["measured"] is False
    context = _turn(orchestrator)
    assert context.extras["expression_profile"] is None
    assert context.extras["baseline_status"] == "insufficient_data"
    # The same number under a measured status is a measurement
    measured = _build(assessor=_Assessor(_report(overall=0.0)))
    asyncio.run(measured.refresh_baseline())
    assert measured.baseline.overall_score == 0.0
    assert measured.baseline.score_history == [0.0]


def test_insufficient_baseline_is_due_again_after_the_next_cycle():
    clock = _Clock()
    assessor = _Assessor(_report(overall=0.0, status="insufficient_data"))
    orchestrator = _build(
        workspace=_ScriptedWorkspace([_state() for _ in range(6)]),
        clock=clock,
        assessor=assessor,
    )
    asyncio.run(orchestrator.refresh_baseline())
    # Right after the insufficient refresh the baseline is not due
    assert orchestrator.baseline_due is False
    assert orchestrator.baseline_due_reason is None
    _turn(orchestrator)
    # A successful cycle brings data: the baseline is due again
    assert orchestrator.baseline_due is True
    assert orchestrator.baseline_due_reason == "first_data"
    assert assessor.calls == 1  # the turn itself runs no assessment
    # A measured refresh clears it, and further cycles leave it clear
    assessor.report = _report()
    clock.advance(60)
    asyncio.run(orchestrator.refresh_baseline())
    assert orchestrator.baseline.measured is True
    assert orchestrator.baseline_due is False
    _turn(orchestrator)
    assert orchestrator.baseline_due is False
    # A failed cycle marks nothing
    failing = _build(
        workspace=_ScriptedWorkspace([]),
        assessor=_Assessor(_report(overall=0.0, status="insufficient_data")),
    )
    asyncio.run(failing.refresh_baseline())
    context = _turn(failing)
    assert context.cycle_ok is False
    assert failing.baseline_due is False


def test_restored_insufficient_baseline_is_unmeasured_and_due():
    saved = _build(assessor=_Assessor(_report(overall=0.0))).to_state_dict()
    scalars = saved["scalars"]
    # A state file written before the insufficient guard existed
    scalars.update(
        baseline_overall_score=0.0,
        baseline_status="insufficient_data",
        baseline_stretch_ceiling=0.0,
        baseline_trend="stable",
        baseline_score_history=[0.0, 0.0, 0.0, 0.0, 0.0],
        baseline_last_updated=T0.isoformat(),
    )
    restored = _build()
    restored.from_state_dict(saved)
    baseline = restored.baseline
    assert baseline.overall_score is None
    assert baseline.measured is False
    assert baseline.status == "insufficient_data"
    assert baseline.stretch_ceiling is None
    assert baseline.score_history == []
    assert baseline.trend is None
    assert restored.baseline_due is True
    # A measured state restores as measured and not due
    scalars.update(baseline_overall_score=0.4, baseline_status="partial")
    measured = _build()
    measured.from_state_dict(saved)
    assert measured.baseline.overall_score == pytest.approx(0.4)
    assert measured.baseline.score_history == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert measured.baseline_due is False


def test_refresh_after_a_cycle_never_reuses_an_older_report():
    clock = _Clock()
    reports = [_report(overall=0.3), _report(overall=0.55)]

    class _Sequence(_Assessor):
        async def __call__(self):
            self.calls += 1
            return reports[self.calls - 1]

    assessor = _Sequence()
    orchestrator = _build(
        workspace=_ScriptedWorkspace([_state() for _ in range(3)]),
        clock=clock,
        assessor=assessor,
    )
    first = asyncio.run(orchestrator.refresh_baseline())
    assert first is reports[0]
    assert assessor.calls == 1
    _turn(orchestrator)
    clock.advance(5)  # inside the cache window, but a cycle ran since
    second = asyncio.run(orchestrator.refresh_baseline())
    assert assessor.calls == 2
    assert second is reports[1]
    assert orchestrator.baseline.overall_score == pytest.approx(0.55)
    assert orchestrator.baseline.score_history == [
        pytest.approx(0.3),
        pytest.approx(0.55),
    ]
    # assess_now follows the same rule
    assert asyncio.run(orchestrator.assess_now()) is reports[1]
    assert assessor.calls == 2


def test_reapplying_the_same_report_does_not_duplicate_the_score():
    clock = _Clock()
    snapshots = _Snapshots()
    assessor = _Assessor()
    orchestrator = _build(clock=clock, assessor=assessor, snapshot_service=snapshots)
    first = asyncio.run(orchestrator.refresh_baseline("heartbeat"))
    clock.advance(5)
    second = asyncio.run(orchestrator.refresh_baseline("loop_quality"))
    assert second is first
    assert assessor.calls == 1
    assert orchestrator.baseline.score_history == [pytest.approx(0.62)]
    assert snapshots.taken == [("heartbeat", first), ("loop_quality", first)]
    assert orchestrator.baseline_due is False
    # A new report is applied again
    clock.advance(20)
    asyncio.run(orchestrator.refresh_baseline())
    assert assessor.calls == 2
    assert len(orchestrator.baseline.score_history) == 2


def test_theory_without_a_measured_indicator_reads_none():
    """A theory whose only included results are absent or insufficient is
    scored 0.0 by the assessment; the baseline does not take that as a
    measurement, and the expression profile loses the stand-in's term."""
    from mtc.assessment.assessment import ConsciousnessTheory
    from mtc.pipeline.consciousness_orchestrator import (
        ConsciousnessBaseline,
        ExpressionProfile,
    )

    def result(theory, status, score=0.0):
        return SimpleNamespace(theory=theory, status=status, score=score)

    report = SimpleNamespace(
        overall_score=0.5,
        theory_scores={"GWT": 0.7, "IIT": 0.0, "XT": 0.3, "YT": 0.6},
        indicator_results={
            "global_broadcast": result(ConsciousnessTheory.GWT, "measured", 0.7),
            "integrated_information": result(ConsciousnessTheory.IIT, "absent"),
            "phi_structure": result(ConsciousnessTheory.IIT, "insufficient_data"),
            "extra_absent": result("XT", "absent"),
            "extra_measured": result("YT", "measured", 0.6),
        },
        status="partial",
        coverage=0.4,
    )
    baseline = ConsciousnessBaseline()
    baseline.update_from_report(report, T0)
    assert baseline.theory_scores["IIT"] is None
    assert baseline.theory_scores["GWT"] == pytest.approx(0.7)
    # Theories outside the eight follow the same rule
    assert baseline.theory_scores["XT"] is None
    assert baseline.theory_scores["YT"] == pytest.approx(0.6)
    assert baseline.overall_score == pytest.approx(0.5)

    expected = ConsciousnessBaseline()
    expected.update_from_report(report, T0)
    expected.theory_scores = dict(expected.theory_scores, IIT=None)
    stand_in = ConsciousnessBaseline()
    stand_in.update_from_report(report, T0)
    stand_in.theory_scores = dict(stand_in.theory_scores, IIT=0.0)
    profile = ExpressionProfile.from_baseline(baseline)
    assert profile == ExpressionProfile.from_baseline(expected)
    assert profile.abstract_thinking == pytest.approx(
        ExpressionProfile.from_baseline(stand_in).abstract_thinking + 0.05
    )

    # A plain report whose results name no theory keeps its scores as given
    plain = ConsciousnessBaseline()
    plain.update_from_report(_report(IIT=0.0), T0)
    assert plain.theory_scores["IIT"] == 0.0
    assert plain.theory_scores["GWT"] == pytest.approx(0.7)


@pytest.fixture
def _runtime_side_effects(monkeypatch):
    """build_runtime configures the mtc logger and records its Settings in the
    PyPhi bridge; both are given back to the suite."""
    from mtc.consciousness import pyphi_bridge
    from mtc.core.logger import reset_logging

    monkeypatch.setattr(pyphi_bridge, "_home_settings", None)
    yield
    reset_logging()


def _runtime_settings(tmp_path):
    return Settings(
        _env_file=None,
        PROJECT_ROOT=str(Path(__file__).resolve().parents[2]),
        DATA_DIRECTORY=str(tmp_path / "data"),
        STATE_DIR=str(tmp_path / "state"),
        LOG_DIRECTORY=str(tmp_path / "logs"),
        RESULTS_DIR=str(tmp_path / "results"),
        AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
        SPEAKERS_PATH=str(tmp_path / "absent_speakers.yaml"),
        LLM_PROVIDER="mock",
        CTM_PROFILE="demo",
        SYSTEM_NAME="Testagent",
        DEVICE="cpu",
        DEBUG=True,
    )


@pytest.mark.asyncio
async def test_fresh_runtime_heartbeat_leaves_the_baseline_unmeasured(
    tmp_path, _runtime_side_effects
):
    from mtc.core.embeddings import HashEmbedder
    from mtc.pipeline.runtime import build_runtime

    runtime = await build_runtime(
        _runtime_settings(tmp_path),
        embedder=HashEmbedder(dim=128),
        allow_non_semantic=True,
        substrates=False,
    )
    try:
        orchestrator = runtime.orchestrator
        assert orchestrator.baseline_due is True
        (heartbeat,) = [
            task for task in runtime.background if task.name == "baseline_heartbeat"
        ]
        report = await heartbeat.tick()
        assert report is not None
        assert report.status == "insufficient_data"
        assert orchestrator.baseline.measured is False
        assert orchestrator.baseline.overall_score is None
        assert orchestrator.baseline.score_history == []
        assert runtime.level_cache is not None
        assert await runtime.level_cache.level() is None
    finally:
        await runtime.stop()


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
