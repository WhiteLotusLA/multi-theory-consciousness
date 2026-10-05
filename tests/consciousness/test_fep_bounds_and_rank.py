"""
Bounded histories and prediction-error ranks for the active inference module.

A process that runs for days must keep every history bounded, and a
percentile rank is only answerable once enough real errors exist.
"""

import numpy as np
import pytest

from mtc.consciousness.active_inference import (
    HISTORY_KEEP,
    HISTORY_LIMIT,
    ActiveInferenceConfig,
    ActiveInferenceModule,
    HomeostaticDrives,
    MetaState,
)


@pytest.fixture
def small_module():
    """A fast module: 4 states, 2 actions, horizon 2 (4 policies)."""
    return ActiveInferenceModule(
        ActiveInferenceConfig(num_hidden_states=4, num_actions=2, planning_horizon=2)
    )


def _observation():
    return np.array([0.4, 0.3, 0.1, 0.1, 0.1])


@pytest.mark.asyncio
async def test_fep_histories_trim_at_1000(small_module):
    # At the limit nothing is trimmed yet.
    small_module.prediction_errors = [0.1] * (HISTORY_LIMIT - 1)
    small_module.free_energy_history = [1.0] * (HISTORY_LIMIT - 1)
    await small_module.infer_and_act(_observation())
    assert len(small_module.prediction_errors) == HISTORY_LIMIT
    assert len(small_module.free_energy_history) == HISTORY_LIMIT

    # One past the limit trims to the most recent HISTORY_KEEP entries.
    await small_module.infer_and_act(_observation())
    assert len(small_module.prediction_errors) == HISTORY_KEEP
    assert len(small_module.free_energy_history) == HISTORY_KEEP
    # The newest entries survive the trim.
    assert small_module.prediction_errors[-1] != 0.1
    assert small_module.free_energy_history[-1] != 1.0


@pytest.mark.asyncio
async def test_belief_and_inference_histories_bounded(small_module):
    small_module.belief_history = [np.full(4, 0.25)] * HISTORY_LIMIT
    small_module.inference_history = []
    await small_module.infer_and_act(_observation())
    assert len(small_module.belief_history) == HISTORY_KEEP

    for _ in range(105):
        await small_module.infer_and_act(_observation())
    assert len(small_module.inference_history) <= 100


def test_meta_state_histories_trim():
    ms = MetaState()
    for i in range(HISTORY_LIMIT):
        ms.update(prediction_error=0.2, was_accurate=bool(i % 2))
    assert len(ms._accuracy_history) == HISTORY_LIMIT
    assert len(ms._confidence_history) == HISTORY_LIMIT

    ms.update(prediction_error=0.2, was_accurate=True)
    assert len(ms._accuracy_history) == HISTORY_KEEP
    assert len(ms._confidence_history) == HISTORY_KEEP

    for _ in range(3 * HISTORY_LIMIT):
        ms.update(prediction_error=0.2, was_accurate=False)
    assert len(ms._accuracy_history) <= HISTORY_LIMIT
    assert len(ms._confidence_history) <= HISTORY_LIMIT
    assert len(ms._recent_outcomes) <= ms._window_size


@pytest.mark.asyncio
async def test_homeostatic_histories_bounded():
    drives = HomeostaticDrives()
    for _ in range(HISTORY_LIMIT + 1):
        await drives.update_drives({"type": "idle", "intensity": 0.1})
        drives.get_free_energy()
    assert len(drives.valence_history) == HISTORY_KEEP
    assert len(drives.free_energy_history) == HISTORY_KEEP


def test_prediction_error_rank_none_until_100(small_module):
    baseline = ActiveInferenceModule.BASELINE_MIN_SAMPLES
    assert baseline == 100

    assert small_module.prediction_error_rank(0.5) is None
    small_module.prediction_errors = [0.1] * (baseline - 1)
    assert small_module.prediction_error_rank(0.5) is None

    small_module.prediction_errors.append(0.1)
    assert small_module.prediction_error_rank(0.5) == 1.0


def test_prediction_error_rank_is_percentile(small_module):
    small_module.prediction_errors = [float(x) for x in range(100)]

    assert small_module.prediction_error_rank(-1.0) == 0.0
    assert small_module.prediction_error_rank(0.0) == 0.0
    assert small_module.prediction_error_rank(50.0) == 0.5
    assert small_module.prediction_error_rank(99.5) == 1.0
    assert small_module.prediction_error_rank(1e6) == 1.0

    # The rank is computed against the module's own (bounded) history.
    small_module.prediction_errors = [float(x) for x in range(200)]
    assert small_module.prediction_error_rank(100.0) == 0.5
