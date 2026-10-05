"""Attention appraisal: transparent dwell-versus-release scoring."""

import pytest

from mtc.consciousness.attention_appraisal import (
    AppraisalInput,
    AppraisalWeights,
    EmotionLexicon,
    appraise,
    valence_from_emotions,
)


def item(**kwargs):
    kwargs.setdefault("novelty", 0.5)
    kwargs.setdefault("valence", 0.2)
    kwargs.setdefault("recurrence", 0.3)
    return AppraisalInput(**kwargs)


def test_decision_is_deterministic():
    first = appraise(item())
    for _ in range(5):
        assert appraise(item()) == first
    assert first.decision in ("dwell", "release")
    assert "mechanism" in first.rationale


def test_contributions_sum_to_score():
    for values_match in (None, 0.0, 0.7, 1.0):
        for valence in (None, -0.8, 0.0, 0.9):
            result = appraise(
                item(novelty=0.8, valence=valence, values_match=values_match)
            )
            assert sum(result.contributions.values()) == pytest.approx(result.score)
            assert -1.0 <= result.score <= 1.0
            assert result.dwell_share == pytest.approx((result.score + 1.0) / 2.0)
    assert appraise(item(novelty=1.0, valence=1.0, recurrence=0.0)).decision == "dwell"
    rehash = appraise(item(novelty=0.0, valence=0.0, recurrence=1.0))
    assert rehash.decision == "release"
    assert rehash.contributions["recurrence"] < 0
    assert rehash.contributions["baseline_release"] < 0


def test_negative_salience_never_forces_release():
    neutral = appraise(item(valence=0.0))
    for strength in (0.3, 0.6, 1.0):
        negative = appraise(item(valence=-strength))
        positive = appraise(item(valence=strength))
        assert negative.score == pytest.approx(positive.score)
        assert negative.score >= neutral.score
        assert negative.contributions["salience"] > 0
    heavy = appraise(item(novelty=0.2, valence=-1.0, recurrence=0.0))
    assert heavy.decision == "dwell"
    assert neutral.decision == appraise(item(valence=0.0)).decision


def test_values_match_is_optional():
    absent = appraise(item())
    assert "values_dwell" not in absent.contributions
    assert "values_release" not in absent.contributions
    assert AppraisalInput(novelty=0.5, valence=0.0, recurrence=0.0).values_match is None
    high = appraise(item(values_match=1.0))
    low = appraise(item(values_match=0.0))
    assert high.score > low.score
    assert high.contributions["values_dwell"] > 0
    assert low.contributions["values_release"] < 0


def test_weights_are_transparent_and_tunable():
    heavy_novelty = AppraisalWeights(novelty=4.0)
    assert (
        appraise(item(novelty=1.0), heavy_novelty).score
        > appraise(item(novelty=1.0)).score
    )
    strict = AppraisalWeights(dwell_threshold=0.9)
    assert appraise(item(novelty=1.0, valence=1.0), strict).decision == "release"


def test_inputs_are_clamped_and_validated():
    clamped = appraise(item(novelty=5.0, valence=-9.0, recurrence=-2.0))
    in_range = appraise(item(novelty=1.0, valence=-1.0, recurrence=0.0))
    assert clamped.score == pytest.approx(in_range.score)
    with pytest.raises(ValueError):
        appraise(item(novelty=float("nan")))


def test_lexicon_configurable():
    lexicon = EmotionLexicon(
        positive=frozenset({"zest"}), negative=frozenset({"gloom"})
    )
    assert valence_from_emotions({"zest": 0.6, "gloom": 0.2}, lexicon) == pytest.approx(
        0.4
    )
    assert valence_from_emotions({"joy": 0.9}, lexicon) is None
    assert valence_from_emotions({"joy": 0.9}) == pytest.approx(0.9)
    assert valence_from_emotions({"sadness": 0.7}) == pytest.approx(-0.7)
    assert valence_from_emotions({"joy": 0.9, "sadness": 0.7}) == pytest.approx(0.2)
    assert valence_from_emotions({"JOY": 5.0}) == 1.0  # clamped, case-insensitive


def test_unknown_emotions_ignored():
    mixed = valence_from_emotions({"joy": 0.5, "flibbertigibbet": 0.9})
    assert mixed == pytest.approx(0.5)
    assert valence_from_emotions({"joy": "not a number", "sadness": 0.2}) == (
        pytest.approx(-0.2)
    )


def test_no_input_valence_is_none():
    assert valence_from_emotions(None) is None
    assert valence_from_emotions({}) is None
    assert valence_from_emotions({"flibbertigibbet": 0.9}) is None
    absent = appraise(item(valence=None))
    assert "salience" not in absent.contributions
    assert sum(absent.contributions.values()) == pytest.approx(absent.score)
