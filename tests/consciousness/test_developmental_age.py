"""Developmental age: evidence-driven stages, no assumed mastery, a governor."""

import os
from datetime import datetime, timezone

import pytest

from mtc.consciousness.developmental_age import (
    DevelopmentalAgeAssessor,
    DevelopmentalEvidence,
    MetricMarker,
    StageLadder,
    StageSpec,
    clamp_to_stage,
    default_metric_markers,
)
from mtc.core.config import Settings

NOW = datetime(2000, 3, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    def build(**overrides):
        overrides.setdefault("COMPANION_TYPE", "child")
        return Settings(**overrides)

    return build


class FixedMarker:
    """Detector returning a fixed score (or None) for one stage."""

    def __init__(self, name, stage, value):
        self.name = name
        self.stage = stage
        self._value = value

    def evaluate(self, evidence):
        return self._value


STRONG = {
    "epistemic_depth": 4.0,
    "assessment_score": 1.0,
    "emotional_range": 5,
    "metaphor_count": 5,
    "creative_output_count": 5,
    "question_rate": 1.0,
}


def test_disabled_status_when_gate_off(cfg):
    assessor = DevelopmentalAgeAssessor(settings=cfg(COMPANION_TYPE="research_subject"))
    result = assessor.assess(DevelopmentalEvidence(messages=[], metrics=STRONG), NOW)
    assert result.status == "disabled"
    assert result.stage is None
    enabled = DevelopmentalAgeAssessor(
        settings=cfg(
            COMPANION_TYPE="research_subject", DEVELOPMENTAL_STAGE_ENABLED=True
        )
    )
    assert enabled.assess(DevelopmentalEvidence([], STRONG), NOW).status == "measured"


def test_no_evidence_is_insufficient_not_a_stage(cfg):
    assessor = DevelopmentalAgeAssessor(settings=cfg())
    result = assessor.assess(DevelopmentalEvidence(messages=[], metrics={}), NOW)
    assert result.status == "insufficient_data"
    assert result.stage is None
    assert result.reason == "no_evidence"
    assert not any(item.evidenced for item in result.stage_evidence.values())
    # Messages alone give no stage either: the defaults have no text markers.
    chatty = assessor.assess(DevelopmentalEvidence(["hello there"] * 30, {}), NOW)
    assert chatty.status == "insufficient_data" and chatty.stage is None
    # A single metric is below the evidence floor of two markers per stage.
    thin = assessor.assess(DevelopmentalEvidence([], {"emotional_range": 5}), NOW)
    assert thin.status == "insufficient_data" and thin.stage is None


def test_measured_zero_is_not_a_stage(cfg):
    assessor = DevelopmentalAgeAssessor(settings=cfg())
    zeros = {key: 0.0 for key in STRONG}
    result = assessor.assess(DevelopmentalEvidence([], zeros), NOW)
    assert result.status == "insufficient_data"
    assert result.reason == "no_stage_mastered"
    assert result.stage is None


def test_metric_markers_master_stages(cfg):
    assessor = DevelopmentalAgeAssessor(settings=cfg())
    result = assessor.assess(DevelopmentalEvidence([], STRONG), NOW)
    assert result.status == "measured"
    assert result.stage == 8
    evidence = result.stage_evidence
    assert evidence[8].mastered and evidence[8].score == 1.0
    assert not evidence[6].evidenced  # no marker covers stage 6
    assert not evidence[9].mastered
    assert result.marker_scores["expresses_emotions"] == 1.0
    mid = assessor.assess(
        DevelopmentalEvidence(
            [], {"emotional_range": 4, "epistemic_depth": 2.0, "metaphor_count": 3}
        ),
        NOW,
    )
    assert mid.stage == 5


def test_missing_metric_leaves_marker_unevaluated(cfg):
    assessor = DevelopmentalAgeAssessor(settings=cfg())
    result = assessor.assess(
        DevelopmentalEvidence([], {"emotional_range": 4, "metaphor_count": 3}), NOW
    )
    assert "epistemic_depth" not in result.marker_scores
    assert "perspective_taking" in result.stage_evidence[5].markers_missing


def test_gap_penalty_applied(cfg):
    detectors = [
        FixedMarker("a3", 3, 0.9),
        FixedMarker("b3", 3, 0.9),
        FixedMarker("a4", 4, 0.1),
        FixedMarker("b4", 4, 0.1),
        FixedMarker("a5", 5, 0.2),
        FixedMarker("b5", 5, 0.2),
        FixedMarker("a6", 6, 0.8),
        FixedMarker("b6", 6, 0.8),
    ]
    assessor = DevelopmentalAgeAssessor(detectors=detectors, settings=cfg())
    result = assessor.assess(DevelopmentalEvidence([], {}), NOW)
    assert result.highest_mastered == 6
    assert result.stage == 4  # two gapped stages below pull the stage down by two
    clean = [d for d in detectors if d.stage in (3, 6)]
    plain = DevelopmentalAgeAssessor(detectors=clean, settings=cfg())
    assert plain.assess(DevelopmentalEvidence([], {}), NOW).stage == 6
    floor = [
        FixedMarker("a3", 3, 0.9),
        FixedMarker("b3", 3, 0.9),
        FixedMarker("a4", 4, 0.0),
        FixedMarker("b4", 4, 0.0),
        FixedMarker("a5", 5, 0.0),
        FixedMarker("b5", 5, 0.0),
        FixedMarker("a6", 6, 0.0),
        FixedMarker("b6", 6, 0.0),
        FixedMarker("a7", 7, 0.9),
        FixedMarker("b7", 7, 0.9),
    ]
    low = DevelopmentalAgeAssessor(detectors=floor, settings=cfg())
    assert low.assess(DevelopmentalEvidence([], {}), NOW).stage == 4


def test_governor_limits_rise_to_one_stage():
    assert clamp_to_stage(3, 8) == 4
    assert clamp_to_stage(3, 4) == 4
    assert clamp_to_stage(3, 3) == 3
    assert clamp_to_stage(5, 2) == 2
    assert clamp_to_stage(None, 7) == 7
    assert clamp_to_stage(4, None) is None
    assert clamp_to_stage(None, None) is None


def test_chronological_days_none_without_birth_date(cfg):
    assessor = DevelopmentalAgeAssessor(settings=cfg())
    result = assessor.assess(DevelopmentalEvidence([], STRONG), NOW)
    assert result.chronological_days is None


def test_chronological_days_from_birth_date(cfg):
    settings = cfg(SYSTEM_BIRTH_DATE="2000-01-01")
    assessor = DevelopmentalAgeAssessor(settings=settings)
    result = assessor.assess(DevelopmentalEvidence([], STRONG), NOW)
    assert result.chronological_days == 60
    future = DevelopmentalAgeAssessor(settings=cfg(SYSTEM_BIRTH_DATE="2001-01-01"))
    assert (
        future.assess(DevelopmentalEvidence([], STRONG), NOW).chronological_days is None
    )


def test_ladder_is_single_source_of_stage_count(cfg):
    assert StageLadder.default().num_stages == 10
    four = StageLadder([StageSpec(number=n, label=f"Stage {n}") for n in range(1, 5)])
    assessor = DevelopmentalAgeAssessor(ladder=four, settings=cfg())
    assert assessor.num_stages == 4
    result = assessor.assess(DevelopmentalEvidence([], STRONG), NOW)
    assert set(result.stage_evidence) == {1, 2, 3, 4}
    assert result.stage is not None and result.stage <= 4
    assert any(name == "paradox_holding" for name in result.ignored_markers)
    with pytest.raises(ValueError):
        StageLadder([StageSpec(number=1, label="a"), StageSpec(number=3, label="c")])
    with pytest.raises(ValueError):
        StageLadder([])


def test_default_ladder_has_no_names_or_ages(cfg):
    ladder = StageLadder.default()
    for spec in ladder.stages:
        assert spec.equivalent_age_years is None
        assert spec.label == f"Stage {spec.number}"


def test_text_patterns_empty_by_default(cfg):
    ladder = StageLadder.default()
    assert all(spec.text_patterns == {} for spec in ladder.stages)
    assert all(isinstance(m, MetricMarker) for m in default_metric_markers())


def test_text_pattern_markers_read_the_ladder(cfg):
    stages = [
        StageSpec(number=1, label="Stage 1"),
        StageSpec(
            number=2,
            label="Stage 2",
            text_patterns={"greets": [r"\bhello\b"], "thanks": [r"\bthank"]},
        ),
    ]
    assessor = DevelopmentalAgeAssessor(
        ladder=StageLadder(stages), detectors=[], settings=cfg()
    )
    result = assessor.assess(
        DevelopmentalEvidence(["hello", "thank you", "hello again"], {}), NOW
    )
    assert result.status == "measured" and result.stage == 2
    assert set(result.marker_scores) == {"greets", "thanks"}
    nothing = assessor.assess(DevelopmentalEvidence([], {}), NOW)
    assert nothing.status == "insufficient_data"


def test_yaml_ladder_round_trip(tmp_path):
    yaml = pytest.importorskip("yaml")
    ladder = StageLadder(
        [
            StageSpec(number=1, label="First"),
            StageSpec(
                number=2,
                label="Second",
                equivalent_age_years=(1.0, 2.0),
                text_patterns={"m": [r"\bword\b"]},
            ),
        ]
    )
    path = tmp_path / "ladder.yaml"
    path.write_text(yaml.safe_dump(ladder.to_dict()), encoding="utf-8")
    loaded = StageLadder.from_yaml(path)
    assert loaded.to_dict() == ladder.to_dict()
    assert loaded.stages[1].equivalent_age_years == (1.0, 2.0)


def test_from_yaml_names_the_missing_package(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="PyYAML"):
        StageLadder.from_yaml(tmp_path / "missing.yaml")


def test_metric_marker_scales_and_ignores_bad_values():
    marker = MetricMarker("m", 3, "metric_x", full_scale=4.0)
    assert marker.evaluate(DevelopmentalEvidence([], {"metric_x": 2.0})) == 0.5
    assert marker.evaluate(DevelopmentalEvidence([], {"metric_x": 9.0})) == 1.0
    assert marker.evaluate(DevelopmentalEvidence([], {"metric_x": -1.0})) == 0.0
    assert marker.evaluate(DevelopmentalEvidence([], {})) is None
    assert (
        marker.evaluate(DevelopmentalEvidence([], {"metric_x": float("nan")})) is None
    )
