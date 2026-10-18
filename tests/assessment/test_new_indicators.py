"""The newer indicators: fresh modules measure nothing, driven modules measure.

Each test fails when its indicator body raises (the status is then
insufficient_data with an error) or returns a constant (the score no longer
follows the evidence), and when an unmeasured report reads 0.0 overall.
"""

import numpy as np
import pytest

from mtc.assessment.assessment import (
    ABSENT,
    INSUFFICIENT_DATA,
    MEASURED,
    ConsciousnessAssessment,
)
from mtc.assessment.rpt_measurement import RPTMeasurement
from tests.assessment.test_missing_signals import (
    FixtureInference,
    FixtureRPT,
    FixtureWorkspace,
    _fresh_modules,
    _weighted_mean,
)
from tests.assessment.test_rpt_workspace_only import WORKSPACE_STATE, _runner_states
from tests.assessment.test_stats_contract import build_and_run_modules


async def _fresh_report():
    return await ConsciousnessAssessment().run_full_assessment(
        **_fresh_modules(), normalize=False
    )


async def _driven_report():
    modules = await build_and_run_modules()
    return await ConsciousnessAssessment().run_full_assessment(
        **modules, normalize=False
    )


def _insufficient_without_error(report, name, missing):
    result = report.indicator_results[name]
    assert result.status == INSUFFICIENT_DATA
    assert result.evidence["missing"] == missing
    assert "error" not in result.evidence
    assert report.to_dict()["indicator_results"][name]["score"] is None
    assert result.threshold == pytest.approx(0.3)


def _measured_in_unit_interval(report, name):
    result = report.indicator_results[name]
    assert result.status == MEASURED, result.evidence
    assert "error" not in result.evidence
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert result.threshold == pytest.approx(0.3)
    return result


@pytest.mark.asyncio
async def test_temporal_prediction_depth_indicator():
    name = "temporal_prediction_depth"
    _insufficient_without_error(
        await _fresh_report(), name, "active_inference.total_inferences"
    )
    result = _measured_in_unit_interval(await _driven_report(), name)
    evidence = result.evidence
    assert evidence["total_inferences"] > 0
    horizon_quality = min(1.0, evidence["planning_horizon"] / 10.0) * max(
        0.0, 1.0 - evidence["avg_error"]
    )
    expected = (
        0.6 * horizon_quality
        + 0.2 * min(1.0, evidence["belief_history_depth"] / 20.0)
        + 0.2 * min(1.0, evidence["free_energy_history_depth"] / 20.0)
    )
    assert result.score == pytest.approx(expected)
    assert result.score > 0.0


@pytest.mark.asyncio
async def test_agent_environment_coupling_indicator():
    name = "agent_environment_coupling"
    _insufficient_without_error(
        await _fresh_report(), name, "active_inference.total_inferences"
    )
    result = _measured_in_unit_interval(await _driven_report(), name)
    actions = result.evidence["actions"]
    observations = result.evidence["observations"]
    assert actions > 0
    expected = (min(1.0, actions / 25) * min(1.0, observations / 25)) ** 0.5
    assert result.score == pytest.approx(expected)
    # The v05 formula on a known stub: 15 inferences and 7 updates over 25
    # warmup cycles give sqrt(0.6 x 0.28)
    stub = await ConsciousnessAssessment().run_full_assessment(
        active_inference=FixtureInference(), normalize=False
    )
    assert stub.indicator_results[name].status == MEASURED
    assert stub.indicator_results[name].score == pytest.approx(
        0.40987803063838396, abs=1e-12
    )


@pytest.mark.asyncio
async def test_contrast_theory_weighting():
    assessment = ConsciousnessAssessment()
    assert assessment.theory_empirical_weights["GWT"] == 1.0
    assert assessment.theory_empirical_weights["HOT"] == 0.50
    assert assessment.theory_empirical_weights["AE"] == 0.70
    # Fresh modules measure nothing: no overall at all
    fresh = await assessment.run_full_assessment(**_fresh_modules(), normalize=False)
    assert fresh.overall_score is None
    assert fresh.to_dict()["overall_score"] is None
    # A known stub mix: GWT and RPT measured, every other theory absent; the
    # overall is the indicator x theory weighted mean of the included results
    # minus the negative-indicator penalty
    report = await assessment.run_full_assessment(
        global_workspace=FixtureWorkspace(),
        rpt_measurement=FixtureRPT(),
        normalize=False,
    )
    included = [
        name
        for name, r in report.indicator_results.items()
        if r.status in (MEASURED, ABSENT)
    ]
    assert len(included) == len(report.indicator_results)
    assert {report.indicator_results[n].status for n in included} == {
        MEASURED,
        ABSENT,
    }
    penalty = sum(v["penalty"] for v in report.negative_indicators.values())
    weighted = _weighted_mean(assessment, report, included)
    assert report.overall_score == pytest.approx(max(0.0, weighted - penalty))
    unweighted = float(np.mean([report.indicator_results[n].score for n in included]))
    assert weighted != pytest.approx(unweighted)
    assert set(report.theory_scores) == {"GWT", "RPT"}


@pytest.mark.asyncio
async def test_local_recurrence_uses_rpt():
    rng = np.random.default_rng(7)
    rpt = RPTMeasurement()
    for _ in range(3):
        snn_state, lsm_state = _runner_states(rng)
        rpt.measure_full(
            snn_state=snn_state, lsm_state=lsm_state, workspace_state=WORKSPACE_STATE
        )
    average_local = rpt.get_statistics()["average_local"]
    assert average_local is not None and average_local > 0.0
    report = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=rpt, normalize=False
    )
    result = report.indicator_results["local_recurrence"]
    assert result.status == MEASURED
    assert result.score == pytest.approx(average_local)
    assert result.evidence["avg_local_recurrence"] == pytest.approx(average_local)
    assert result.evidence["measurements"] == 3

    fresh = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=RPTMeasurement(), normalize=False
    )
    _insufficient = fresh.indicator_results["local_recurrence"]
    assert _insufficient.status == INSUFFICIENT_DATA
    assert _insufficient.evidence["missing"] == "rpt_measurement.measurement_count"
