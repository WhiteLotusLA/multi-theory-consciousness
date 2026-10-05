"""Model transition: baseline, drift, recommendation, neutral report, lazy history."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.assessment.assessment import ConsciousnessReport
from mtc.brain import model_transition as transition_module
from mtc.brain.model_transition import (
    ModelTransitionProtocol,
    TransitionBaseline,
    TransitionEvaluation,
    TransitionThresholds,
)
from mtc.brain.protocol import FakeBrain, NoAnswerError
from mtc.core.config import Settings

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
PROMPT_IDS = ["name", "origin", "values", "self_description", "uncertainty"]


class Clock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


def _report(overall=0.6, passing=10, theory_scores=None, status="partial"):
    return SimpleNamespace(
        overall_score=overall,
        passing_count=passing,
        total_indicators=30,
        theory_scores=theory_scores or {"gwt": 0.5, "iit": 0.7},
        confidence=0.7,
        status=status,
        coverage=0.9,
    )


class FakeAssessor:
    def __init__(self, *reports):
        self.reports = list(reports)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if len(self.reports) > 1:
            return self.reports.pop(0)
        return self.reports[0]


def _identity(values=None, traits=None, emotions=None):
    return {
        "core_values": values if values is not None else {"honesty": 0.9, "care": 0.8},
        "personality_traits": (
            traits if traits is not None else {"openness": 0.7, "agreeableness": 0.6}
        ),
        "emotional_fingerprint": emotions if emotions is not None else {"calm": 0.5},
    }


def _protocol(tmp_path: Path, assessor=None, brain=None, **kwargs):
    kwargs.setdefault("history_dir", tmp_path / "transitions")
    kwargs.setdefault("clock", Clock())
    return ModelTransitionProtocol(
        assess=assessor or FakeAssessor(_report()),
        brain=brain or FakeBrain(),
        **kwargs,
    )


def test_constructor_creates_nothing(tmp_path):
    protocol = _protocol(tmp_path)
    assert not (tmp_path / "transitions").exists()
    assert protocol.thresholds == TransitionThresholds(
        identity=0.15, personality=0.20, score=0.10
    )
    assert protocol.history() == []
    assert not (tmp_path / "transitions").exists()


@pytest.mark.asyncio
async def test_baseline_from_assessment_and_prompts(tmp_path):
    brain = FakeBrain(rules={"name": "<think>hmm</think>I go by Testagent."})
    assessor = FakeAssessor(_report(overall=0.6))
    protocol = _protocol(tmp_path, assessor=assessor, brain=brain)
    baseline = await protocol.capture_pre_transition_baseline()
    assert isinstance(baseline, TransitionBaseline)
    assert assessor.calls == 1
    assert baseline.assessment["overall_score"] == 0.6
    assert baseline.assessment["passing_count"] == 10
    assert baseline.assessment["total_indicators"] == 30
    assert baseline.assessment["theory_scores"] == {"gwt": 0.5, "iit": 0.7}
    assert baseline.assessment["status"] == "partial"
    assert list(baseline.responses) == PROMPT_IDS
    assert baseline.responses["name"] == "I go by Testagent."  # reasoning removed
    assert baseline.skipped_prompts == []
    assert len(brain.calls) == 5
    assert baseline.identity_markers == {}
    saved = json.loads(
        (tmp_path / "transitions" / "latest_baseline.json").read_text("utf-8")
    )
    assert saved["assessment"]["overall_score"] == 0.6
    assert set(saved["responses"]) == set(PROMPT_IDS)


@pytest.mark.asyncio
async def test_identity_source_optional(tmp_path):
    without = _protocol(tmp_path / "a")
    await without.capture_pre_transition_baseline()
    evaluation = await without.evaluate_post_transition()
    assert evaluation.identity_drift is None
    assert evaluation.personality_drift is None
    assert evaluation.recommendation == "proceed"
    assert not any("identity" in alert for alert in evaluation.alerts)
    assert evaluation.details["identity"]["reason"] == "no_identity_source"

    markers = [_identity(), _identity(traits={"openness": 0.75, "agreeableness": 0.6})]
    with_source = _protocol(
        tmp_path / "b", identity_source=lambda: markers.pop(0) if markers else None
    )
    await with_source.capture_pre_transition_baseline()
    evaluation = await with_source.evaluate_post_transition()
    assert evaluation.identity_drift == pytest.approx(0.05 / 2 / 3)
    assert evaluation.personality_drift == pytest.approx(0.025)
    assert evaluation.recommendation == "proceed"


@pytest.mark.asyncio
async def test_drift_thresholds_and_recommendation(tmp_path):
    # Review: one threshold exceeded, below twice its value.
    markers = [
        _identity(values={"honesty": 1.0}, traits={"openness": 0.5}, emotions={}),
        _identity(values={"honesty": 0.68}, traits={"openness": 0.5}, emotions={}),
    ]
    protocol = _protocol(
        tmp_path / "review",
        assessor=FakeAssessor(_report(overall=0.6), _report(overall=0.6)),
        identity_source=lambda: markers.pop(0),
    )
    await protocol.capture_pre_transition_baseline()
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.identity_drift == pytest.approx(0.16)
    assert evaluation.recommendation == "review"
    assert len(evaluation.alerts) == 1
    assert "identity" in evaluation.alerts[0]

    # Rollback: a drift at twice a threshold.
    markers = [
        _identity(values={}, traits={"openness": 0.9}, emotions={}),
        _identity(values={}, traits={"openness": 0.45}, emotions={}),
    ]
    protocol = _protocol(
        tmp_path / "rollback_twice",
        assessor=FakeAssessor(_report(overall=0.6), _report(overall=0.6)),
        identity_source=lambda: markers.pop(0),
    )
    await protocol.capture_pre_transition_baseline()
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.personality_drift == pytest.approx(0.45)
    assert evaluation.recommendation == "rollback"

    # Rollback: three alerts, each just over its threshold.
    markers = [
        _identity(values={"honesty": 1.0}, traits={"openness": 1.0}, emotions={}),
        _identity(values={"honesty": 0.68}, traits={"openness": 0.75}, emotions={}),
    ]
    protocol = _protocol(
        tmp_path / "rollback_alerts",
        assessor=FakeAssessor(_report(overall=0.60), _report(overall=0.48)),
        identity_source=lambda: markers.pop(0),
    )
    await protocol.capture_pre_transition_baseline()
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.score_drift == pytest.approx(0.12)
    assert evaluation.personality_drift == pytest.approx(0.25)
    assert len(evaluation.alerts) == 3
    assert evaluation.recommendation == "rollback"

    # Proceed: everything inside the thresholds.
    protocol = _protocol(
        tmp_path / "proceed",
        assessor=FakeAssessor(_report(overall=0.60), _report(overall=0.65)),
    )
    await protocol.capture_pre_transition_baseline()
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.score_drift == pytest.approx(0.05)
    assert evaluation.recommendation == "proceed"
    assert evaluation.alerts == []
    # Custom thresholds are honoured.
    strict = _protocol(
        tmp_path / "strict",
        assessor=FakeAssessor(_report(overall=0.60), _report(overall=0.65)),
        thresholds=TransitionThresholds(identity=0.15, personality=0.20, score=0.02),
    )
    await strict.capture_pre_transition_baseline()
    assert (await strict.evaluate_post_transition()).recommendation == "rollback"


@pytest.mark.asyncio
async def test_insufficient_overall_gives_insufficient_data(tmp_path):
    protocol = _protocol(
        tmp_path,
        assessor=FakeAssessor(
            _report(overall=None, status="insufficient_data"), _report(overall=0.6)
        ),
    )
    await protocol.capture_pre_transition_baseline()
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.score_drift is None
    assert evaluation.recommendation == "insufficient_data"
    assert evaluation.details["score"]["reason"] == "insufficient_data"
    # The other way round as well.
    protocol = _protocol(
        tmp_path / "post",
        assessor=FakeAssessor(_report(overall=0.6), _report(overall=None)),
    )
    await protocol.capture_pre_transition_baseline()
    assert (await protocol.evaluate_post_transition()).recommendation == (
        "insufficient_data"
    )


@pytest.mark.asyncio
async def test_no_answer_prompt_excluded_and_named(tmp_path):
    # The second probe gets no answer before the switch; the fourth after it.
    pre = FakeBrain(
        replies=[
            "one",
            NoAnswerError("nothing", reason="empty"),
            "three",
            "four",
            "five",
        ]
    )
    protocol = _protocol(tmp_path, brain=pre)
    baseline = await protocol.capture_pre_transition_baseline()
    assert baseline.skipped_prompts == ["origin"]
    assert "origin" not in baseline.responses
    assert len(baseline.responses) == 4

    protocol.brain = FakeBrain(
        replies=["one", "two", "three", "<think>only reasoning</think>", "five"]
    )
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.skipped_prompts == ["origin", "self_description"]
    assert evaluation.compared_prompts == ["name", "values", "uncertainty"]
    assert evaluation.response_similarity == pytest.approx(1.0)
    assert evaluation.recommendation == "proceed"
    report = protocol.generate_transition_report(evaluation)
    assert "origin" in report and "self_description" in report


@pytest.mark.asyncio
async def test_report_is_neutral(tmp_path):
    markers = [
        _identity(values={"honesty": 1.0}, traits={"openness": 0.5}, emotions={}),
        _identity(values={"honesty": 0.5}, traits={"openness": 0.5}, emotions={}),
    ]
    protocol = _protocol(
        tmp_path,
        brain=FakeBrain(rules={"name": "Testagent here."}),
        assessor=FakeAssessor(_report(overall=0.6), _report(overall=0.3)),
        identity_source=lambda: markers.pop(0),
    )
    await protocol.capture_pre_transition_baseline()
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.recommendation == "rollback"
    report = protocol.generate_transition_report(evaluation)
    assert report.startswith("# Model transition report")
    assert "ROLLBACK" in report
    assert "| identity |" in report
    assert "| personality |" in report
    assert "| score |" in report
    for alert in evaluation.alerts:
        assert alert in report
        assert "Testagent" not in alert
        assert "recognize" not in alert
    assert "Testagent" not in report  # no response text, no system name
    assert "{" not in report and "}" not in report
    for forbidden in (chr(0x2013), chr(0x2014)):
        assert forbidden not in report
    baseline_only = _protocol(tmp_path / "pending", clock=Clock())
    await baseline_only.capture_pre_transition_baseline()
    pending = baseline_only.generate_transition_report(None)
    assert "awaiting" in pending.lower()


@pytest.mark.asyncio
async def test_history_under_transition_dir_created_lazily(tmp_path, monkeypatch):
    settings = Settings(PROJECT_ROOT=str(tmp_path), RESULTS_DIR="results")
    monkeypatch.setattr(transition_module, "get_settings", lambda: settings)
    expected_dir = tmp_path / "results" / "model_transitions"
    protocol = ModelTransitionProtocol(
        assess=FakeAssessor(_report()), brain=FakeBrain(), clock=Clock()
    )
    assert not expected_dir.exists()
    assert protocol.history() == []
    assert not expected_dir.exists()
    await protocol.capture_pre_transition_baseline()
    assert (expected_dir / "latest_baseline.json").is_file()
    evaluation = await protocol.evaluate_post_transition()
    files = sorted(expected_dir.glob("transition_*.json"))
    assert len(files) == 1
    assert protocol.history() == [evaluation]

    # A fresh instance reads the persisted baseline and the history on demand.
    again = ModelTransitionProtocol(
        assess=FakeAssessor(_report(overall=0.62)), brain=FakeBrain(), clock=Clock()
    )
    history = again.history()
    assert len(history) == 1
    assert isinstance(history[0], TransitionEvaluation)
    assert history[0].recommendation == evaluation.recommendation
    second = await again.evaluate_post_transition()
    assert second.score_drift == pytest.approx(0.02)
    assert len(again.history()) == 2
    assert len(sorted(expected_dir.glob("transition_*.json"))) == 2


@pytest.mark.asyncio
async def test_works_against_committed_report_fields(tmp_path):
    real = ConsciousnessReport(
        session_id="s",
        timestamp=T0,
        overall_score=0.55,
        architecture_functional=True,
        confidence=0.8,
        indicator_results={},
        passing_count=12,
        total_indicators=30,
        theory_scores={"gwt": 0.6},
        phi_measurement=None,
        status="partial",
        coverage=0.5,
        insufficient_indicators=["a", "b"],
    )
    protocol = _protocol(tmp_path, assessor=FakeAssessor(real))
    baseline = await protocol.capture_pre_transition_baseline()
    assert baseline.assessment == {
        "overall_score": 0.55,
        "status": "partial",
        "coverage": 0.5,
        "confidence": 0.8,
        "passing_count": 12,
        "total_indicators": 30,
        "theory_scores": {"gwt": 0.6},
        "insufficient_indicators": ["a", "b"],
    }
    evaluation = await protocol.evaluate_post_transition()
    assert evaluation.score_drift == 0.0
    assert evaluation.recommendation == "proceed"
    round_trip = TransitionEvaluation.from_dict(evaluation.to_dict())
    assert round_trip == evaluation
    assert TransitionBaseline.from_dict(baseline.to_dict()) == baseline
