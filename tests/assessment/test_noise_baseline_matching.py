"""A Phi value is normalised only against a noise baseline its own producer made."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from mtc.assessment.assessment import MEASURED, ConsciousnessAssessment

PROXY = "calculator_spectral_proxy"
TRACKER = "phi_tracker_iit4"


class FakeTracker:
    """A PhiTracker stand-in that reports one exact Phi."""

    N_NODES = 5

    def __init__(self, phi):
        self.phi = phi

    async def compute_architectural_phi(self):
        return SimpleNamespace(
            phi=self.phi,
            phi_normalized=None,
            distinctions_count=None,
            relations_count=None,
            partition_info="cut",
            is_exact=True,
            computation_time_ms=1.0,
            tpm_coverage=0.5,
            transitions_recorded=200,
            warming_up=False,
            error=None,
        )


def _seed():
    random.seed(0)
    np.random.seed(0)


def _neural_states():
    rng = np.random.default_rng(5)
    raw = rng.random((8, 8))
    return {"combined_state": rng.random(8), "connectivity": (raw + raw.T) / 2.0}


@pytest.mark.asyncio
async def test_tracker_phi_not_normalized_against_proxy_baseline():
    _seed()
    report = await ConsciousnessAssessment().run_full_assessment(
        phi_tracker=FakeTracker(2.0), normalize=True
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == MEASURED
    assert result.evidence["source"] == TRACKER
    # Phi 2.0 bits over the fixed reference 5.0: the raw score is kept
    assert result.score == pytest.approx(0.4)
    assert result.evidence["raw_score"] == pytest.approx(0.4)
    assert result.evidence["noise_baseline"] is None
    assert result.evidence["noise_saturated"] is False
    skipped = result.evidence["noise_baseline_skipped"]
    assert skipped["baseline_source"] == PROXY
    assert skipped["result_source"] == TRACKER


@pytest.mark.asyncio
async def test_calculator_value_still_normalized():
    _seed()
    report = await ConsciousnessAssessment().run_full_assessment(
        neural_states=_neural_states(), normalize=True
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == MEASURED
    assert result.evidence["source"] == PROXY
    baseline = result.evidence["noise_baseline"]
    assert isinstance(baseline, float)
    assert "noise_baseline_skipped" not in result.evidence
    assert result.score == pytest.approx(
        ConsciousnessAssessment.get_normalized_score(
            result.evidence["raw_score"], baseline, 0.1
        )
    )


@pytest.mark.asyncio
async def test_v04_any_keeps_old_behaviour():
    _seed()
    legacy = ConsciousnessAssessment(noise_baseline_matching="v04_any")
    report = await legacy.run_full_assessment(
        phi_tracker=FakeTracker(2.0), normalize=True
    )
    result = report.indicator_results["integrated_information"]
    assert result.evidence["source"] == TRACKER
    baseline = result.evidence["noise_baseline"]
    assert isinstance(baseline, float) and baseline > 0.4
    assert result.score == 0.0
    assert "noise_baseline_skipped" not in result.evidence


@pytest.mark.asyncio
async def test_caller_baselines_always_apply():
    _seed()
    report = await ConsciousnessAssessment().run_full_assessment(
        phi_tracker=FakeTracker(2.0),
        normalize=True,
        noise_baselines={"integrated_information": 0.2},
    )
    result = report.indicator_results["integrated_information"]
    assert result.evidence["noise_baseline"] == pytest.approx(0.2)
    assert result.score == pytest.approx((0.4 - 0.2) / 0.8)
    assert "noise_baseline_skipped" not in result.evidence


@pytest.mark.asyncio
async def test_noise_pass_records_its_phi_source():
    assessment = ConsciousnessAssessment()
    _seed()
    await assessment.run_noise_baseline(iterations=2)
    assert assessment._noise_baseline_sources == {"integrated_information": PROXY}
    legacy = ConsciousnessAssessment(indicator_formulas="v04")
    _seed()
    await legacy.run_noise_baseline(iterations=2)
    assert legacy._noise_baseline_sources == {"integrated_information": None}


def test_invalid_matching_names_the_field():
    with pytest.raises(ValueError, match="noise_baseline_matching"):
        ConsciousnessAssessment(noise_baseline_matching="always")
    assert ConsciousnessAssessment().noise_baseline_matching == "producer"
