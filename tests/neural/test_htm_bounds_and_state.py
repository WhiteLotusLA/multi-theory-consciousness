"""Bounded anomaly history, lightweight state dict and honest metrics of the
production HTM."""

import json

import numpy as np
import pytest

from mtc.neural.htm.production_htm import (
    ANOMALY_HISTORY_LIMIT,
    HTMConfig,
    ProductionHTM,
)


def _tiny_config(**overrides) -> HTMConfig:
    defaults = dict(
        num_columns=32,
        cells_per_column=4,
        input_bits=64,
        potential_radius=64,
        local_area_density=0.1,
        activation_threshold=2,
        max_synapses_per_segment=8,
    )
    defaults.update(overrides)
    return HTMConfig(**defaults)


def _pattern(bits: int, start: int, width: int = 6) -> np.ndarray:
    pattern = np.zeros(bits)
    pattern[start : start + width] = 1
    return pattern


def test_anomaly_history_trimmed_at_1000():
    htm = ProductionHTM.__new__(ProductionHTM)
    htm.anomaly_history = [float(i) for i in range(1500)]

    htm._trim_anomaly_history()

    assert ANOMALY_HISTORY_LIMIT == 1000
    assert len(htm.anomaly_history) == 1000
    assert htm.anomaly_history[0] == 500.0
    assert htm.anomaly_history[-1] == 1499.0


def test_process_keeps_history_bounded():
    np.random.seed(0)
    cfg = _tiny_config()
    htm = ProductionHTM(cfg)
    pattern = _pattern(cfg.input_bits, 0)

    for _ in range(ANOMALY_HISTORY_LIMIT + 5):
        htm.process(pattern, learn=False)

    assert len(htm.anomaly_history) == ANOMALY_HISTORY_LIMIT


def test_state_dict_round_trip_keeps_tail():
    np.random.seed(1)
    cfg = _tiny_config()
    source = ProductionHTM(cfg)
    for i in range(30):
        source.process(_pattern(cfg.input_bits, (i % 5) * 10), learn=True)

    state = source.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    assert state["scalars"]["anomaly_history_len"] == 30
    assert state["scalars"]["anomaly_score"] == pytest.approx(
        source.anomaly_history[-1]
    )
    tail = state["histories"]["anomaly_history_tail"]
    assert tail == pytest.approx(source.anomaly_history[-20:])
    json.dumps(state["scalars"])  # the persistence hub writes scalars as JSON

    target = ProductionHTM(cfg)
    target.from_state_dict(state)
    assert target.anomaly_history == pytest.approx(source.anomaly_history[-20:])
    assert target.get_metrics()["anomaly_score"] == pytest.approx(tail[-1])


def test_restore_drops_nonfinite_tail_values_and_keeps_live_history():
    cfg = _tiny_config()
    state = {
        "scalars": {},
        "arrays": {},
        "histories": {"anomaly_history_tail": [0.2, float("nan"), 0.4]},
    }

    fresh = ProductionHTM(cfg)
    fresh.from_state_dict(state)
    assert fresh.anomaly_history == [0.2, 0.4]

    live = ProductionHTM(cfg)
    live.process(_pattern(cfg.input_bits, 0), learn=False)
    before = list(live.anomaly_history)
    live.from_state_dict(state)
    assert live.anomaly_history == before


def test_metrics_anomaly_none_before_input():
    np.random.seed(2)
    cfg = _tiny_config()
    htm = ProductionHTM(cfg)

    metrics = htm.get_metrics()
    assert metrics["anomaly_score"] is None
    assert metrics["active_columns"] is None
    assert metrics["column_density"] is None
    assert metrics["avg_permanence"] is None
    assert htm.to_state_dict()["scalars"]["anomaly_score"] is None

    htm.process(_pattern(cfg.input_bits, 0), learn=True)
    htm.process(_pattern(cfg.input_bits, 20), learn=True)
    metrics = htm.get_metrics()
    assert metrics["anomaly_score"] == pytest.approx(htm.anomaly_history[-1])
    assert metrics["active_columns"] == int(cfg.num_columns * cfg.local_area_density)
    assert metrics["column_density"] == pytest.approx(
        metrics["active_columns"] / cfg.num_columns
    )
    # The second step grew segments at the initial permanence
    assert metrics["avg_permanence"] == pytest.approx(cfg.initial_permanence)


def test_reset_clears_measured_metrics():
    cfg = _tiny_config()
    htm = ProductionHTM(cfg)
    htm.process(_pattern(cfg.input_bits, 0), learn=True)

    htm.reset()

    metrics = htm.get_metrics()
    assert metrics["anomaly_score"] is None
    assert metrics["active_columns"] is None
