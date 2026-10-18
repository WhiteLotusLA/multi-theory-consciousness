"""The spectral Phi proxy rejects bad connectivity instead of substituting 0.5."""

import numpy as np
import pytest

from mtc.assessment.assessment import (
    INSUFFICIENT_DATA,
    ConsciousnessAssessment,
    PhiCalculator,
)

STATE = np.array([0.2, 0.9, 0.4, 0.7])
CONNECTIVITY = np.array(
    [
        [0.0, 0.8, 0.1, 0.3],
        [0.8, 0.0, 0.5, 0.2],
        [0.1, 0.5, 0.0, 0.6],
        [0.3, 0.2, 0.6, 0.0],
    ]
)
# The proxy's value for STATE and CONNECTIVITY on the tree before the input
# checks were added; the formula for valid input does not move.
PHI_BEFORE = 1.886381682600977


@pytest.mark.asyncio
async def test_non_square_connectivity_is_insufficient():
    report = await ConsciousnessAssessment().run_full_assessment(
        neural_states={"combined_state": STATE, "connectivity": np.ones((4, 3)) * 0.4},
        normalize=False,
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == INSUFFICIENT_DATA
    assert "(4, 3)" in result.evidence["phi_error"]
    assert report.phi_measurement is None
    # A square matrix of the wrong size is rejected too
    mismatched = await ConsciousnessAssessment().run_full_assessment(
        neural_states={"combined_state": STATE, "connectivity": np.eye(5)},
        normalize=False,
    )
    assert mismatched.indicator_results["integrated_information"].status == (
        INSUFFICIENT_DATA
    )


@pytest.mark.asyncio
async def test_calculate_phi_raises_on_non_finite_connectivity():
    calculator = PhiCalculator()
    bad = CONNECTIVITY.copy()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        await calculator._calculate_phi_approximation(STATE, bad)
    with pytest.raises(ValueError, match="square"):
        await calculator._calculate_phi_approximation(STATE, np.ones((4, 3)))
    with pytest.raises(ValueError, match="square"):
        await calculator._calculate_phi_approximation(STATE, np.ones(4))
    # The indicator reports the error instead of a number
    report = await ConsciousnessAssessment().run_full_assessment(
        neural_states={"combined_state": STATE, "connectivity": bad}, normalize=False
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == INSUFFICIENT_DATA
    assert "finite" in result.evidence["phi_error"]


@pytest.mark.asyncio
async def test_valid_input_unchanged():
    phi = await PhiCalculator()._calculate_phi_approximation(STATE, CONNECTIVITY)
    assert phi == pytest.approx(PHI_BEFORE, abs=1e-12)
    measurement = await PhiCalculator().calculate_phi(STATE, CONNECTIVITY)
    assert measurement.method == "spectral_proxy"
    assert measurement.phi == pytest.approx(PHI_BEFORE, abs=1e-12)
    assert measurement.phi_normalized == pytest.approx(PHI_BEFORE / 5.0)
    # No 0.5 stands in anywhere on the proxy path
    source = open(
        PhiCalculator._calculate_phi_approximation.__code__.co_filename
    ).read()
    body = source[source.index("async def _calculate_phi_approximation") :]
    body = body[: body.index("class ConsciousnessAssessment")]
    assert "integration = 0.5" not in body
    assert "irreducibility = connectivity_density" not in body
