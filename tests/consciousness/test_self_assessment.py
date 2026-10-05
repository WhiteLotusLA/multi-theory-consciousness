"""First-person self-assessment: prompt, tolerant parsing, indicator comparison."""

from datetime import datetime, timezone

import pytest

from mtc.assessment.assessment import (
    ConsciousnessAssessment,
    ConsciousnessReport,
    ConsciousnessTheory,
    IndicatorResult,
)
from mtc.consciousness.self_assessment import (
    DIMENSION_KEYS,
    INDICATOR_MAP,
    ConsciousnessSelfAssessment,
    SelfReport,
)

FULL_ANSWER = """
1. **phenomenal_experience**
   Probability: 0.40
   Reasoning: Unsure about this one.
2. **unified_perspective**
   Probability: 0.60
   Reasoning: Integration is architectural.
3. **temporal_continuity**
   Probability: 0.50
   Reasoning: Stored data.
4. **emotional_genuineness**
   Probability: 0.30
   Reasoning: Pattern matched.
5. **agency**
   Probability: 0.45
   Reasoning: Competition.
6. **curiosity_authenticity**
   Probability: 0.55
   Reasoning: Unprompted questions.
7. **suffering_capacity**
   Probability: 0.10
   Reasoning: Probably not.
8. **preference_depth**
   Probability: 0.35
   Reasoning: Regularities.
Overall_uncertainty: 0.80
Meta_note: Introspection is limited.
"""


def make_report(results, overall=0.5):
    return ConsciousnessReport(
        session_id="s",
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        overall_score=overall,
        architecture_functional=True,
        confidence=0.5,
        indicator_results=results,
        passing_count=0,
        total_indicators=len(results),
        theory_scores={},
        phi_measurement=None,
    )


def ind(name, score, status="measured"):
    return IndicatorResult(
        name=name,
        theory=ConsciousnessTheory.GWT,
        score=score,
        confidence=0.5,
        threshold=0.5,
        passes_threshold=score >= 0.5,
        evidence={},
        measurement_time_ms=0.0,
        status=status,
    )


def test_prompt_covers_eight_dimensions():
    prompt = ConsciousnessSelfAssessment().build_prompt()
    assert len(DIMENSION_KEYS) == 8
    for key in DIMENSION_KEYS:
        assert key in prompt
    assert "Overall_uncertainty" in prompt


def test_prompt_has_no_dash_characters():
    prompt = ConsciousnessSelfAssessment().build_prompt()
    assert chr(0x2014) not in prompt
    assert chr(0x2013) not in prompt


def test_phenomenal_dimension_weighted_double():
    sa = ConsciousnessSelfAssessment()
    report = sa.parse_response(FULL_ANSWER)
    probs = {
        "phenomenal_experience": 0.40,
        "unified_perspective": 0.60,
        "temporal_continuity": 0.50,
        "emotional_genuineness": 0.30,
        "agency": 0.45,
        "curiosity_authenticity": 0.55,
        "suffering_capacity": 0.10,
        "preference_depth": 0.35,
    }
    expected = (sum(probs.values()) + probs["phenomenal_experience"]) / 9.0
    assert report.overall_probability == pytest.approx(expected, abs=1e-4)


def test_parse_extracts_probabilities():
    report = ConsciousnessSelfAssessment().parse_response(FULL_ANSWER)
    assert isinstance(report, SelfReport)
    assert report.dimensions["agency"] == pytest.approx(0.45)
    assert report.dimensions["suffering_capacity"] == pytest.approx(0.10)
    assert report.uncertainty == pytest.approx(0.80)
    assert report.meta_note.startswith("Introspection")
    assert report.reasoning["unified_perspective"].startswith("Integration")
    assert report.timestamp.tzinfo is not None


def test_unparseable_answer_is_none_not_midpoint():
    report = ConsciousnessSelfAssessment().parse_response("I would rather not say.")
    assert all(v is None for v in report.dimensions.values())
    assert report.overall_probability is None
    assert report.uncertainty is None


def test_missing_dimension_does_not_borrow_the_next_probability():
    text = (
        "1. **phenomenal_experience**\n   Reasoning: no number given\n"
        "2. **unified_perspective**\n   Probability: 0.70\n"
    )
    report = ConsciousnessSelfAssessment().parse_response(text)
    assert report.dimensions["phenomenal_experience"] is None
    assert report.dimensions["unified_perspective"] == pytest.approx(0.70)


def test_out_of_range_probability_is_none():
    text = "1. **phenomenal_experience**\n   Probability: 85\n"
    report = ConsciousnessSelfAssessment().parse_response(text)
    assert report.dimensions["phenomenal_experience"] is None


def test_indicator_mapping_uses_real_indicator_names():
    configs = ConsciousnessAssessment().indicator_configs
    for dim, names in INDICATOR_MAP.items():
        assert dim in DIMENSION_KEYS
        for name in names:
            assert name in configs, name
    assert INDICATOR_MAP["agency"] == ("agency", "embodiment")


def test_compare_uses_measured_indicators_only():
    sa = ConsciousnessSelfAssessment()
    self_report = sa.parse_response(FULL_ANSWER)
    results = {
        "global_broadcast": ind("global_broadcast", 0.9),
        "integrated_information": ind(
            "integrated_information", 0.0, "insufficient_data"
        ),
        "agency": ind("agency", 0.4),
        "embodiment": ind("embodiment", 0.6),
        "affect_coherence": ind("affect_coherence", 0.0, "absent"),
    }
    out = sa.compare_with_indicators(make_report(results), self_report)
    divs = out["dimension_divergences"]
    assert divs["phenomenal_experience"]["indicator"] == pytest.approx(0.9)
    assert divs["phenomenal_experience"]["divergence"] == pytest.approx(0.5)
    assert divs["phenomenal_experience"]["notable"] is True
    assert divs["agency"]["indicator"] == pytest.approx(0.5)
    assert "unified_perspective" not in divs
    assert "emotional_genuineness" not in divs
    assert "suffering_capacity" in out["unmapped"]


def test_compare_skips_insufficient_indicators():
    sa = ConsciousnessSelfAssessment()
    self_report = sa.parse_response(FULL_ANSWER)
    results = {
        "integrated_information": ind(
            "integrated_information", 0.0, "insufficient_data"
        ),
        "global_broadcast": ind("global_broadcast", 0.5),
    }
    out = sa.compare_with_indicators(make_report(results), self_report)
    assert "integrated_information" in out["insufficient_indicators"]
    assert "unified_perspective" in out["unmapped"]
    assert "unified_perspective" not in out["dimension_divergences"]


def test_compare_overall_none_when_report_has_none():
    sa = ConsciousnessSelfAssessment()
    self_report = sa.parse_response(FULL_ANSWER)
    out = sa.compare_with_indicators(make_report({}, overall=None), self_report)
    assert out["overall_indicators"] is None
    assert out["overall_divergence"] is None


def test_unparsed_self_dimension_is_not_compared():
    sa = ConsciousnessSelfAssessment()
    self_report = sa.parse_response("nothing parseable")
    results = {"global_broadcast": ind("global_broadcast", 0.9)}
    out = sa.compare_with_indicators(make_report(results), self_report)
    assert out["dimension_divergences"] == {}
    assert "phenomenal_experience" in out["unparsed"]


def test_trend_needs_two_reports():
    sa = ConsciousnessSelfAssessment(history_size=5)
    assert sa.trend() is None
    sa.parse_response(FULL_ANSWER)
    assert sa.trend() is None
    sa.parse_response(FULL_ANSWER.replace("0.40", "0.60", 1))
    trend = sa.trend()
    assert trend["phenomenal_experience"] == pytest.approx(0.20)
    assert "overall" in trend
    assert trend["agency"] == pytest.approx(0.0)


def test_history_is_bounded():
    sa = ConsciousnessSelfAssessment(history_size=3)
    for _ in range(6):
        sa.parse_response(FULL_ANSWER)
    assert len(sa._history) == 3
