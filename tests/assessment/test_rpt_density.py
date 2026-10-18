"""The SNN spike density of the RPT measurement takes its neuron count from the state."""

import numpy as np
import pytest

from mtc.assessment.rpt_measurement import RPTMeasurement


def test_density_uses_layer_sizes():
    # 200 spikes over 5,000 neurons (the summed per-layer firing-rate
    # lengths): a density of 200 / (2 x 5000) = 0.02, not a saturated 1.0
    # against an invented 100 neurons.
    state = {
        "total_spikes": 200,
        "firing_rates": {"input": [0.0] * 4000, "output": [0.0] * 1000},
    }
    assert RPTMeasurement()._measure_snn_recurrence(state) == pytest.approx(0.02)


def test_density_skipped_without_a_count():
    # Neither num_neurons nor firing_rates: the density component is left
    # out and the score is the mean of the remaining two components
    # (amplification 15 / 10 -> 0.75; activation std 0.1 -> 0.2).
    state = {
        "total_spikes": 200,
        "spike_counts": [10, 15],
        "activation_pattern": np.array([0.4, 0.6, 0.4, 0.6]),
    }
    assert RPTMeasurement()._measure_snn_recurrence(state) == pytest.approx(
        (0.75 + 0.2) / 2
    )
    # With no other component either, nothing is measured from the state
    assert RPTMeasurement()._measure_snn_recurrence({"total_spikes": 200}) == 0.0


def test_explicit_count_unchanged():
    # The experiment runner's dict (num_neurons supplied): the value the
    # present tree gives, so no runner number moves.
    runner_state = {
        "total_spikes": 150,
        "num_neurons": 100,
        "spike_counts": [np.array([10]), np.array([25]), np.array([8])],
        "activation_pattern": np.linspace(0.0, 1.0, 100),
    }
    score = RPTMeasurement()._measure_snn_recurrence(runner_state)
    assert score == pytest.approx(0.7777176434190042, abs=1e-12)
    # An explicit count wins over the firing-rate lengths
    both = dict(runner_state, firing_rates={"input": [0.0] * 5000})
    assert RPTMeasurement()._measure_snn_recurrence(both) == pytest.approx(score)
    # A non-positive count is no count
    zero = dict(runner_state, num_neurons=0)
    assert RPTMeasurement()._measure_snn_recurrence(zero) != pytest.approx(score)
