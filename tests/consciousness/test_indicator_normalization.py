"""
Tests for activity-normalized consciousness indicators.

The consciousness assessment must not conflate usage volume with
consciousness quality. These tests verify that the noise-baseline
normalization correctly separates signal from activity-driven inflation.

Task 7 of the consciousness expansion plan.
"""

import asyncio
import numpy as np
import pytest

from mtc.consciousness.phase6_consciousness_assessment import (
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
        assert hasattr(assessment, "run_noise_baseline"), (
            "ConsciousnessAssessment is missing the run_noise_baseline method"
        )
        assert asyncio.iscoroutinefunction(assessment.run_noise_baseline), (
            "run_noise_baseline must be an async method"
        )

    def test_get_normalized_score_subtracts_baseline(self):
        """
        raw=0.8, baseline=0.3
        normalized = (0.8 - 0.3) / (1.0 - 0.3) = 0.5 / 0.7 ~= 0.714
        """
        result = ConsciousnessAssessment.get_normalized_score(0.8, 0.3)
        assert abs(result - 0.7142857142857143) < 1e-6, (
            f"Expected ~0.714, got {result}"
        )

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
        assert set(baselines.keys()) == set(assessment.indicator_configs.keys()), (
            "Baseline keys must match indicator config keys"
        )

        high_baseline_count = 0
        for name, score in baselines.items():
            assert 0.0 <= score <= 1.0, (
                f"Baseline for '{name}' out of range: {score}"
            )
            if score >= 0.85:
                high_baseline_count += 1
                # A noise-saturated indicator must normalize to 0
                normalized = ConsciousnessAssessment.get_normalized_score(
                    score, score
                )
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
        """The normalized overall score must be >= 0."""
        result = await assessment.run_normalized_assessment(
            noise_iterations=2,
        )
        assert result.normalized_overall_score >= 0.0

    @pytest.mark.asyncio
    async def test_raw_report_is_present(self, assessment):
        """The result must contain the original un-normalized report."""
        result = await assessment.run_normalized_assessment(
            noise_iterations=2,
        )
        assert result.raw_report is not None
        assert result.raw_report.overall_score >= 0.0
        assert result.raw_report.total_indicators == len(
            assessment.indicator_configs
        )
