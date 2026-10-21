"""
Tests for activity-normalized consciousness indicators.

The consciousness assessment must not conflate usage volume with
consciousness quality. These tests verify that the noise-baseline
normalization correctly separates signal from activity-driven inflation.

Task 7 of the consciousness expansion plan.
"""

import inspect
import random

import numpy as np
import pytest

from mtc.assessment.assessment import (
    ConsciousnessAssessment,
    NormalizedAssessmentResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def assessment():
    """Create a fresh ConsciousnessAssessment instance for each test."""
    return ConsciousnessAssessment()


# ---------------------------------------------------------------------------
# Unit tests for get_normalized_score (pure math, no async)
# ---------------------------------------------------------------------------


class TestGetNormalizedScore:
    """Tests for the static get_normalized_score method."""

    def test_noise_baseline_method_exists(self, assessment):
        """The ConsciousnessAssessment class must expose run_noise_baseline."""
        assert hasattr(
            assessment, "run_noise_baseline"
        ), "ConsciousnessAssessment is missing the run_noise_baseline method"
        assert inspect.iscoroutinefunction(
            assessment.run_noise_baseline
        ), "run_noise_baseline must be an async method"

    def test_get_normalized_score_subtracts_baseline(self):
        """
        raw=0.8, baseline=0.3
        normalized = (0.8 - 0.3) / (1.0 - 0.3) = 0.5 / 0.7 ~= 0.714
        """
        result = ConsciousnessAssessment.get_normalized_score(0.8, 0.3)
        assert abs(result - 0.7142857142857143) < 1e-6, f"Expected ~0.714, got {result}"

    def test_get_normalized_score_floors_at_zero(self):
        """
        When raw < baseline the normalized score must not go negative.
        raw=0.2, baseline=0.5 -> clamped to 0.0
        """
        result = ConsciousnessAssessment.get_normalized_score(0.2, 0.5)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_get_normalized_score_handles_broken_indicator(self):
        """
        If noise alone produces a perfect score (baseline >= 1.0), the
        indicator is fundamentally broken and must return 0.0.
        """
        assert ConsciousnessAssessment.get_normalized_score(0.9, 1.0) == 0.0
        assert ConsciousnessAssessment.get_normalized_score(1.0, 1.0) == 0.0
        assert ConsciousnessAssessment.get_normalized_score(0.5, 1.5) == 0.0

    def test_get_normalized_score_zero_baseline(self):
        """With zero baseline the raw score passes through unchanged."""
        result = ConsciousnessAssessment.get_normalized_score(0.6, 0.0)
        assert abs(result - 0.6) < 1e-6

    def test_get_normalized_score_perfect_raw(self):
        """A perfect raw score (1.0) always normalizes to 1.0 unless broken."""
        result = ConsciousnessAssessment.get_normalized_score(1.0, 0.4)
        assert abs(result - 1.0) < 1e-6

    def test_get_normalized_score_equal_raw_and_baseline(self):
        """When raw == baseline the normalized score is exactly 0."""
        result = ConsciousnessAssessment.get_normalized_score(0.5, 0.5)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Async tests for noise baseline
# ---------------------------------------------------------------------------


class TestNoiseBaseline:
    """Tests that verify the noise baseline machinery."""

    @pytest.mark.asyncio
    async def test_run_noise_baseline_produces_low_scores(self, assessment):
        """
        Noise baselines should be moderate-to-low for most indicators.
        Individual indicators MAY have high noise baselines (that is exactly
        the problem this normalization is designed to catch), but the
        *majority* of indicators should not score near-perfect on noise.

        We also verify that every baseline is in [0, 1] and that the
        normalization correctly zeroes out any indicator whose noise
        baseline reaches 1.0 (a broken indicator).
        """
        baselines = await assessment.run_noise_baseline(iterations=3)

        # Must return a score for every configured indicator
        assert set(baselines.keys()) == set(
            assessment.indicator_configs.keys()
        ), "Baseline keys must match indicator config keys"

        high_baseline_count = 0
        for name, score in baselines.items():
            assert 0.0 <= score <= 1.0, f"Baseline for '{name}' out of range: {score}"
            if score >= 0.85:
                high_baseline_count += 1
                # A noise-saturated indicator must normalize to 0
                normalized = ConsciousnessAssessment.get_normalized_score(score, score)
                assert normalized == 0.0, (
                    f"Broken indicator '{name}' (baseline={score:.3f}) "
                    f"should normalize to 0.0, got {normalized}"
                )

        # At most a third of indicators may have high noise baselines;
        # if more are broken, the scoring framework itself needs rework.
        max_broken = len(assessment.indicator_configs) // 3
        assert high_baseline_count <= max_broken, (
            f"{high_baseline_count} indicators have noise baselines >= 0.85 "
            f"(max allowed: {max_broken}). The scoring framework may be "
            f"measuring activity rather than consciousness."
        )

    @pytest.mark.asyncio
    async def test_run_noise_baseline_returns_all_indicators(self, assessment):
        """Every indicator must have a baseline entry."""
        baselines = await assessment.run_noise_baseline(iterations=2)
        assert len(baselines) == len(assessment.indicator_configs)

    @pytest.mark.asyncio
    async def test_run_noise_baseline_deterministic_enough(self, assessment):
        """
        Two runs with the same iteration count should produce baselines
        in the same ballpark (within 0.3 of each other), confirming the
        averaging smooths out randomness.
        """
        # Seeded so the comparison is reproducible: unseeded, two 5-iteration
        # baselines differ by 0.3 or more about a third of the time (mostly
        # higher_order_representations), before and after v0.5.0.
        random.seed(42)
        np.random.seed(42)
        b1 = await assessment.run_noise_baseline(iterations=5)
        b2 = await assessment.run_noise_baseline(iterations=5)

        for name in b1:
            diff = abs(b1[name] - b2[name])
            assert diff < 0.3, (
                f"Baseline for '{name}' varied too much between runs: "
                f"{b1[name]:.3f} vs {b2[name]:.3f} (diff={diff:.3f})"
            )


# ---------------------------------------------------------------------------
# Async tests for the full normalized assessment pipeline
# ---------------------------------------------------------------------------


class TestNormalizedAssessment:
    """Tests for run_normalized_assessment end-to-end."""

    @pytest.mark.asyncio
    async def test_normalized_assessment_returns_correct_type(self, assessment):
        """run_normalized_assessment must return a NormalizedAssessmentResult."""
        result = await assessment.run_normalized_assessment(
            noise_iterations=2,
        )
        assert isinstance(result, NormalizedAssessmentResult)

    @pytest.mark.asyncio
    async def test_normalized_scores_leq_raw_scores(self, assessment):
        """
        Normalized scores should never exceed raw scores because we are
        subtracting a non-negative baseline and then rescaling into [0, 1].
        In practice: normalized = (raw - baseline) / (1 - baseline).
        When baseline > 0, normalized < raw for any raw < 1.
        When baseline == 0, normalized == raw.
        Either way normalized <= raw (with floating-point tolerance).
        """
        result = await assessment.run_normalized_assessment(
            noise_iterations=3,
        )

        for name in result.raw_scores:
            raw = result.raw_scores[name]
            norm = result.normalized_scores[name]
            assert norm <= raw + 1e-9, (
                f"Normalized score for '{name}' ({norm:.4f}) exceeds "
                f"raw score ({raw:.4f})"
            )

    @pytest.mark.asyncio
    async def test_flagged_indicators_have_high_baselines(self, assessment):
        """
        Every indicator in flagged_indicators must have a noise baseline
        strictly above 0.3.
        """
        result = await assessment.run_normalized_assessment(
            noise_iterations=3,
        )

        for name in result.flagged_indicators:
            assert result.noise_baselines[name] > 0.3, (
                f"'{name}' is flagged but baseline is only "
                f"{result.noise_baselines[name]:.3f}"
            )

    @pytest.mark.asyncio
    async def test_normalized_overall_score_is_nonnegative(self, assessment):
        """The normalized overall is >= 0 when measured, None with nothing measured."""
        from tests.assessment.test_missing_signals import fixture_modules

        nothing = await assessment.run_normalized_assessment(noise_iterations=2)
        assert nothing.normalized_overall_score is None
        result = await assessment.run_normalized_assessment(
            noise_iterations=2, **fixture_modules()
        )
        assert result.normalized_overall_score >= 0.0

    @pytest.mark.asyncio
    async def test_normalized_overall_matches_full_assessment(self, assessment):
        """The normalized overall is the overall run_full_assessment computes
        from the same baselines: the same weights, penalty and min range."""
        from tests.assessment.test_missing_signals import fixture_modules

        random.seed(3)
        np.random.seed(3)
        result = await assessment.run_normalized_assessment(
            noise_iterations=3, **fixture_modules()
        )
        full = await assessment.run_full_assessment(
            **fixture_modules(),
            normalize=True,
            noise_baselines=result.noise_baselines,
        )
        assert result.normalized_overall_score == pytest.approx(full.overall_score)
        for name, score in result.normalized_scores.items():
            assert score == pytest.approx(full.indicator_results[name].score), name
        assert set(result.normalized_scores) == {
            name
            for name, r in full.indicator_results.items()
            if r.status != "insufficient_data"
        }
        # The raw report is the un-normalized one
        assert result.raw_report.overall_score != full.overall_score

    @pytest.mark.asyncio
    async def test_indicator_without_baseline_keeps_raw(self, assessment, monkeypatch):
        from tests.assessment.test_missing_signals import fixture_modules

        async def partial_baselines(iterations=10):
            return {"global_broadcast": 0.2}

        monkeypatch.setattr(assessment, "run_noise_baseline", partial_baselines)
        result = await assessment.run_normalized_assessment(
            noise_iterations=2, **fixture_modules()
        )
        assert result.normalized_scores["global_broadcast"] == pytest.approx(
            (0.55 - 0.2) / 0.8
        )
        for name, raw in result.raw_scores.items():
            if name != "global_broadcast":
                assert result.normalized_scores[name] == raw, name
        assert result.flagged_indicators == []

    @pytest.mark.asyncio
    async def test_legacy_min_range_respected(self, monkeypatch):
        from tests.assessment.test_missing_signals import fixture_modules

        async def saturated_baselines(iterations=10):
            return {"genuine_implementation": 0.95}

        legacy = ConsciousnessAssessment(saturated_baseline_min_range=0.0)
        monkeypatch.setattr(legacy, "run_noise_baseline", saturated_baselines)
        result = await legacy.run_normalized_assessment(
            noise_iterations=2, **fixture_modules()
        )
        # raw 1.0 against a 0.95 floor: (1.0 - 0.95) / 0.05, not zeroed
        assert result.raw_scores["genuine_implementation"] == pytest.approx(1.0)
        assert result.normalized_scores["genuine_implementation"] == pytest.approx(1.0)
        assert result.flagged_indicators == ["genuine_implementation"]

        guarded = ConsciousnessAssessment()
        monkeypatch.setattr(guarded, "run_noise_baseline", saturated_baselines)
        zeroed = await guarded.run_normalized_assessment(
            noise_iterations=2, **fixture_modules()
        )
        assert zeroed.normalized_scores["genuine_implementation"] == 0.0

    @pytest.mark.asyncio
    async def test_forwards_rpt_measurement(self, assessment):
        from tests.assessment.test_missing_signals import FixtureRPT

        result = await assessment.run_normalized_assessment(
            noise_iterations=2, rpt_measurement=FixtureRPT()
        )
        local = result.raw_report.indicator_results["local_recurrence"]
        assert local.status == "measured"
        assert local.score == pytest.approx(0.42)
        assert "local_recurrence" in result.normalized_scores

    @pytest.mark.asyncio
    async def test_raw_report_is_present(self, assessment):
        """The result must contain the original un-normalized report."""
        result = await assessment.run_normalized_assessment(
            noise_iterations=2,
        )
        assert result.raw_report is not None
        # No module was supplied: nothing measured, so no overall score
        assert result.raw_report.overall_score is None
        assert result.raw_report.status == "insufficient_data"
        assert result.raw_report.total_indicators == len(assessment.indicator_configs)
