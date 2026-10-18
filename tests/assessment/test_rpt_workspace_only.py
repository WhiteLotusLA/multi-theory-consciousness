"""RPT measurement without a substrate state: local recurrence is unmeasured."""

import numpy as np
import pytest

from mtc.assessment.assessment import (
    INSUFFICIENT_DATA,
    MEASURED,
    ConsciousnessAssessment,
)
from mtc.assessment.rpt_measurement import RPTMeasurement

# coverage 0.8, cycles 5 of 10, no re-entry over 5 broadcasts, 2 ignitions of
# 5: (0.8 + 0.5 + 0.0 + 0.4) / 4
WORKSPACE_STATE = {
    "broadcast": {"coverage_ratio": 0.8, "total_broadcasts": 5},
    "cycle_count": 5,
    "ignition": {"total_ignitions": 2},
}
EXPECTED_GLOBAL = 0.425


class _RanWorkspace:
    """A workspace that ran: 12 cycles, 24 perceptual candidates through
    the bottleneck."""

    def get_statistics(self):
        return {
            "cycle_count": 12,
            "broadcast": {"total_broadcasts": 9, "coverage_ratio": 0.55},
            "ignition": {"total_ignitions": 6, "sustain_rate": 0.5},
            "bottleneck": {
                "total_competitions": 12,
                "avg_candidates": 6.0,
                "source_distribution": {"sensory": 20, "memory": 8, "concept": 4},
            },
        }


def _workspace_only_rpt(calls=3):
    rpt = RPTMeasurement()
    metrics = None
    for _ in range(calls):
        metrics = rpt.measure_full(
            snn_state=None, lsm_state=None, workspace_state=WORKSPACE_STATE
        )
    return rpt, metrics


def _runner_states(rng):
    """The SNN and LSM dicts the experiment runner builds each cycle."""
    num_neurons = 100
    input_spikes = rng.integers(1, 20, size=num_neurons)
    hidden_spikes = rng.integers(1, 20, size=num_neurons)
    snn_state = {
        "total_spikes": int(input_spikes.sum() + hidden_spikes.sum()),
        "num_neurons": num_neurons,
        "spike_counts": [input_spikes, hidden_spikes],
        "activation_pattern": rng.random(num_neurons),
    }
    lsm_state = {
        "spectral_radius": float(rng.uniform(0.3, 1.1)),
        "reservoir_state": rng.random(50),
        "active_neuron_ratio": float(rng.uniform(0.1, 0.9)),
        "edge_of_chaos_metric": float(rng.uniform(0.2, 1.0)),
    }
    return snn_state, lsm_state


def test_workspace_only_measurement_has_no_local_score():
    rpt, metrics = _workspace_only_rpt(calls=3)
    assert metrics.local_recurrence_score is None
    assert metrics.snn_feedback_strength is None
    assert metrics.lsm_reservoir_recurrence is None
    assert metrics.local_spike_reentry is None
    assert metrics.global_recurrence_score == pytest.approx(EXPECTED_GLOBAL)
    assert rpt.measure_local_recurrence(None, None) is None
    stats = rpt.get_statistics()
    assert stats["average_local"] is None
    assert stats["local_measurement_count"] == 0
    assert stats["measurement_count"] == 3
    assert stats["average_global"] == pytest.approx(EXPECTED_GLOBAL)
    assert len(rpt._local_scores) == 0
    # The report formats the unmeasured value instead of raising
    assert "Measurements: 3" in rpt.generate_report()


@pytest.mark.asyncio
async def test_workspace_only_indicators_report_insufficient():
    rpt, _metrics = _workspace_only_rpt()
    report = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=rpt, normalize=False
    )
    for name in ("local_recurrence", "algorithmic_recurrence"):
        result = report.indicator_results[name]
        assert result.status == INSUFFICIENT_DATA, (name, result.evidence)
        assert "rpt_measurement.average_local" in result.evidence["missing"]
        assert name in report.insufficient_indicators
    # Without a workspace either, the perceptual indicator has nothing
    percepts = report.indicator_results["organized_percepts"]
    assert percepts.status == INSUFFICIENT_DATA
    assert "rpt_measurement.average_local" in percepts.evidence["missing"]


@pytest.mark.asyncio
async def test_v04_constants_keeps_zero():
    rpt, _metrics = _workspace_only_rpt()
    report = await ConsciousnessAssessment(
        missing_signal_policy="v04_constants"
    ).run_full_assessment(rpt_measurement=rpt, normalize=False)
    local = report.indicator_results["local_recurrence"]
    assert local.status == MEASURED
    assert local.score == 0.0
    algorithmic = report.indicator_results["algorithmic_recurrence"]
    assert algorithmic.status == MEASURED
    assert algorithmic.score == pytest.approx(0.4 * EXPECTED_GLOBAL)


def test_substrate_states_still_measure():
    rng = np.random.default_rng(11)
    rpt = RPTMeasurement()
    locals_seen = []
    for _ in range(4):
        snn_state, lsm_state = _runner_states(rng)
        metrics = rpt.measure_full(
            snn_state=snn_state, lsm_state=lsm_state, workspace_state=WORKSPACE_STATE
        )
        assert metrics.local_recurrence_score is not None
        assert metrics.local_recurrence_score == pytest.approx(
            rpt.measure_local_recurrence(snn_state, lsm_state)
        )
        locals_seen.append(metrics.local_recurrence_score)
    stats = rpt.get_statistics()
    assert stats["local_measurement_count"] == 4
    assert stats["measurement_count"] == 4
    assert stats["average_local"] == pytest.approx(float(np.mean(locals_seen)))
    assert len(set(locals_seen)) > 1


def test_one_substrate_state_leaves_the_other_component_unmeasured():
    rpt = RPTMeasurement()
    metrics = rpt.measure_full(
        snn_state={"total_spikes": 200, "num_neurons": 100},
        workspace_state=WORKSPACE_STATE,
    )
    assert metrics.local_recurrence_score is not None
    assert metrics.snn_feedback_strength is not None
    assert metrics.local_spike_reentry == 0.0
    assert metrics.lsm_reservoir_recurrence is None
    assert rpt.get_statistics()["local_measurement_count"] == 1


@pytest.mark.asyncio
async def test_no_rpt_records_no_local_recurrence():
    """Without an RPT the perceptual indicator's recurrence term is unmeasured:
    the evidence records None, the score carries the perceptual term alone."""
    honest = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=_RanWorkspace(), normalize=False
    )
    percepts = honest.indicator_results["organized_percepts"]
    assert percepts.status == MEASURED
    assert percepts.evidence["avg_local_recurrence"] is None
    candidates = percepts.evidence["perceptual_candidates"]
    assert candidates > 0
    assert percepts.score == pytest.approx(0.5 * min(1.0, candidates / 5))
    # v04_constants keeps the constant 0.0 the v0.5.0 code recorded
    constants = await ConsciousnessAssessment(
        missing_signal_policy="v04_constants"
    ).run_full_assessment(global_workspace=_RanWorkspace(), normalize=False)
    legacy = constants.indicator_results["organized_percepts"]
    assert legacy.status == MEASURED
    assert legacy.evidence["avg_local_recurrence"] == 0.0
    assert legacy.score == percepts.score
    assert legacy.evidence["perceptual_candidates"] == candidates
