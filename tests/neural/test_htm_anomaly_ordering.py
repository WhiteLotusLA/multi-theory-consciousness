"""Anomaly ordering of the production HTM.

The anomaly of a step is the share of its active columns that the previous
step predicted. v0.4.0 scored the active columns against the predictions the
temporal memory had just computed from those same columns, which reads about
1.0 on every step, learned or not. These tests pin the prior-prediction
ordering. The anomaly never feeds back into the HTM dynamics, so only the
reported score changes.
"""

import numpy as np
import pytest

from mtc.neural.htm.production_htm import HTMConfig, ProductionHTM, TemporalMemory


def _seq_config(**overrides) -> HTMConfig:
    """A small configuration that learns short sequences quickly."""
    defaults = dict(
        num_columns=256,
        cells_per_column=8,
        input_bits=512,
        input_sparsity=0.05,
        potential_radius=512,
        global_inhibition=True,
        local_area_density=0.05,
        activation_threshold=2,
        max_segments_per_cell=16,
        max_synapses_per_segment=16,
        learning_enabled=True,
    )
    defaults.update(overrides)
    return HTMConfig(**defaults)


def _block(bits: int, start: int, width: int = 25) -> np.ndarray:
    pattern = np.zeros(bits)
    pattern[start : start + width] = 1
    return pattern


def _sequence(bits: int, length: int = 5):
    return [_block(bits, 90 * i) for i in range(length)]


def test_first_presentation_is_fully_anomalous():
    np.random.seed(3)
    cfg = _seq_config()
    htm = ProductionHTM(cfg)

    result = htm.process(_block(cfg.input_bits, 0), learn=True)

    assert result["anomaly_score"] == pytest.approx(1.0)


def test_anomaly_falls_on_repeated_sequence():
    np.random.seed(0)
    cfg = _seq_config()
    htm = ProductionHTM(cfg)
    sequence = _sequence(cfg.input_bits)

    per_repetition = []
    for _ in range(3):
        scores = [htm.process(p, learn=True)["anomaly_score"] for p in sequence]
        per_repetition.append(float(np.mean(scores)))

    # The first pass is novel; by the third the sequence is anticipated
    assert per_repetition[0] >= 0.9, per_repetition
    assert per_repetition[2] < 0.5, per_repetition
    assert per_repetition[2] < per_repetition[0]


def test_anomaly_uses_prior_predictions():
    np.random.seed(5)
    cfg = _seq_config()
    htm = ProductionHTM(cfg)
    sequence = _sequence(cfg.input_bits)
    for _ in range(3):
        for pattern in sequence:
            htm.process(pattern, learn=True)

    prior_columns = {
        c // cfg.cells_per_column for c in htm.temporal_memory.predicted_cells
    }
    assert prior_columns, "the learned sequence should leave a prediction behind"

    result = htm.process(sequence[0], learn=False)
    active = {int(c) for c in np.where(result["active_columns"])[0]}
    expected = len(active - prior_columns) / max(1, len(active))

    assert result["anomaly_score"] == pytest.approx(expected)
    assert htm.anomaly_history[-1] == pytest.approx(expected)


def test_temporal_memory_reports_anomaly_against_prior_predictions():
    cfg = _seq_config(num_columns=8, cells_per_column=2)
    tm = TemporalMemory(cfg)
    columns = np.zeros(cfg.num_columns, dtype=bool)
    columns[[1, 2]] = True
    # Column 1 was predicted by the previous step, column 2 was not
    tm.predicted_cells = {1 * cfg.cells_per_column}

    state = tm.compute(columns, learn=False)

    assert state["anomaly_score"] == pytest.approx(0.5)
