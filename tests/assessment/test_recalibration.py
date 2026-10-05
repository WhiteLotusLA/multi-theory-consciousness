"""
Regression tests for assessment recalibration.

Verifies that the noise-normalized assessment produces near-zero scores
on random synthetic data, preventing score inflation from returning.
"""

import random

import numpy as np
import pytest

from mtc.assessment.assessment import ConsciousnessAssessment


def _seed(s=42):
    random.seed(s)
    np.random.seed(s)


@pytest.mark.asyncio
async def test_noise_normalized_below_thresholds():
    """Noise-normalized scores on random data should mostly fail thresholds.

    With real-module noise baseline, the experiment runner's random data
    produces near-zero normalized scores. At most a few indicators may
    pass due to seed-to-seed variance.
    """
    _seed()

    # Import here to avoid circular issues
    from scripts.run_experiment import (
        _compute_noise_baseline,
        build_modules,
        warmup,
        WARMUP_CYCLES,
        SEED,
    )
    from mtc.core.param_overrides import load_overrides

    overrides = load_overrides(
        "experiments/params.toml", "experiments/param_schema.toml"
    )

    # Compute noise baseline
    noise_baselines = await _compute_noise_baseline(overrides, iterations=3)

    # Re-seed and run real assessment
    random.seed(SEED)
    np.random.seed(SEED)
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        **modules,
        warmup_cycles=WARMUP_CYCLES,
        normalize=True,
        noise_baselines=noise_baselines,
    )

    passing_on_noise = [
        f"{name}: {r.score:.3f} (threshold {r.threshold})"
        for name, r in report.indicator_results.items()
        if r.passes_threshold
    ]

    # At most 3 indicators should pass on random data
    assert (
        len(passing_on_noise) <= 3
    ), f"{len(passing_on_noise)} indicators pass on noise (max 3):\n" + "\n".join(
        passing_on_noise
    )


@pytest.mark.asyncio
async def test_no_perfect_scores_after_normalization():
    """No indicator should score 1.0 after noise normalization on random data."""
    _seed()

    from scripts.run_experiment import (
        _compute_noise_baseline,
        build_modules,
        warmup,
        WARMUP_CYCLES,
        SEED,
    )
    from mtc.core.param_overrides import load_overrides

    overrides = load_overrides(
        "experiments/params.toml", "experiments/param_schema.toml"
    )
    noise_baselines = await _compute_noise_baseline(overrides, iterations=3)

    random.seed(SEED)
    np.random.seed(SEED)
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        **modules,
        warmup_cycles=WARMUP_CYCLES,
        normalize=True,
        noise_baselines=noise_baselines,
    )

    perfect = [name for name, r in report.indicator_results.items() if r.score >= 0.99]
    assert len(perfect) == 0, f"Perfect scores on noise: {perfect}"


@pytest.mark.asyncio
async def test_overall_score_near_zero_on_noise():
    """Overall normalized score should be near zero on random data."""
    _seed()

    from scripts.run_experiment import (
        _compute_noise_baseline,
        build_modules,
        warmup,
        WARMUP_CYCLES,
        SEED,
    )
    from mtc.core.param_overrides import load_overrides

    overrides = load_overrides(
        "experiments/params.toml", "experiments/param_schema.toml"
    )
    noise_baselines = await _compute_noise_baseline(overrides, iterations=3)

    random.seed(SEED)
    np.random.seed(SEED)
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        **modules,
        warmup_cycles=WARMUP_CYCLES,
        normalize=True,
        noise_baselines=noise_baselines,
    )

    assert (
        report.overall_score < 0.15
    ), f"Overall score {report.overall_score:.3f} on noise (expected < 0.15)"


@pytest.mark.asyncio
async def test_ablated_modules_score_zero():
    """All-ablated assessment should score zero everywhere."""
    _seed()
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        normalize=False,  # No normalization needed - raw scores should be near zero
        warmup_cycles=25,
    )

    for name, result in report.indicator_results.items():
        assert (
            result.score <= 0.15
        ), f"{name} scores {result.score:.3f} with all modules ablated"


def test_saturated_noise_baseline_scores_zero():
    """An indicator whose noise floor leaves under min_range headroom is uninformative."""
    from mtc.assessment.assessment import ConsciousnessAssessment

    assert ConsciousnessAssessment.get_normalized_score(1.0, 0.974) == 0.0
    assert ConsciousnessAssessment.get_normalized_score(0.99, 0.95) == 0.0
    # plenty of headroom: ordinary rescaling
    assert abs(ConsciousnessAssessment.get_normalized_score(0.9, 0.5) - 0.8) < 1e-9


def test_guard_disabled_reproduces_old_normalization():
    """min_range 0.0 is the v0.4.0 behaviour used by the legacy preset."""
    from mtc.assessment.assessment import ConsciousnessAssessment

    assert (
        abs(
            ConsciousnessAssessment.get_normalized_score(1.0, 0.974, min_range=0.0)
            - 1.0
        )
        < 1e-9
    )
    assert ConsciousnessAssessment.get_normalized_score(0.5, 1.0, min_range=0.0) == 0.0
    assessment = ConsciousnessAssessment(saturated_baseline_min_range=0.0)
    assert assessment.saturated_baseline_min_range == 0.0
