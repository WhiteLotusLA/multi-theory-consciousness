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

    # Learning passes fill the anomaly history; inference-only passes keep
    # their own (test_live_history_is_bounded).
    for _ in range(ANOMALY_HISTORY_LIMIT + 5):
        htm.process(pattern, learn=True)

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
    live.process(_pattern(cfg.input_bits, 0), learn=True)
    before = list(live.anomaly_history)
    assert before
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


def _segments(htm: ProductionHTM):
    return {
        cell: [
            (
                sorted(int(s) for s in seg["synapses"]),
                sorted(seg["permanences"].items()),
            )
            for seg in segments
        ]
        for cell, segments in htm.temporal_memory.segments.items()
    }


def _trained(seed: int, steps: int = 5) -> ProductionHTM:
    np.random.seed(seed)
    cfg = _tiny_config()
    htm = ProductionHTM(cfg)
    for i in range(steps):
        htm.process(_pattern(cfg.input_bits, (i % 3) * 20), learn=True)
    return htm


def test_inference_pass_leaves_learning_state():
    htm = _trained(seed=7)
    tm = htm.temporal_memory
    assert tm.active_cells and tm.winner_cells
    before = (
        set(tm.active_cells),
        set(tm.winner_cells),
        set(tm.predicted_cells),
        list(htm.anomaly_history),
        htm._last_active_columns.copy(),
    )

    result = htm.process(_pattern(htm.config.input_bits, 40), learn=False)

    assert tm.active_cells == before[0]
    assert tm.winner_cells == before[1]
    assert tm.predicted_cells == before[2]
    assert htm.anomaly_history == before[3]
    assert np.array_equal(htm._last_active_columns, before[4])
    assert htm.live_anomaly_history == [result["anomaly_score"]]
    assert np.array_equal(htm._last_live_active_columns, result["active_columns"])
    metrics = htm.get_metrics()
    assert metrics["anomaly_score"] == pytest.approx(before[3][-1])
    assert metrics["live_anomaly_score"] == pytest.approx(result["anomaly_score"])


def test_live_anomaly_none_before_a_live_pass():
    htm = _trained(seed=8)
    assert htm.get_metrics()["live_anomaly_score"] is None
    assert htm.live_anomaly_history == []


def test_live_context_carries_between_live_passes():
    np.random.seed(9)
    cfg = _tiny_config()
    htm = ProductionHTM(cfg)
    first, second = _pattern(cfg.input_bits, 0), _pattern(cfg.input_bits, 30)
    for _ in range(6):
        htm.process(first, learn=True)
        htm.process(second, learn=True)
    learned = htm.temporal_memory.predicted_cells

    live_first = htm.process(first, learn=False)["anomaly_score"]
    live_second = htm.process(second, learn=False)["anomaly_score"]

    # The live context starts empty, so its first pass has nothing predicted;
    # the second pass is scored against what the first live pass predicted.
    assert live_first == 1.0
    assert live_second < live_first
    assert htm.temporal_memory.predicted_cells == learned
    assert htm.live_anomaly_history == [live_first, live_second]


def test_live_pass_does_not_change_what_is_learned():
    control = _trained(seed=11)
    probed = _trained(seed=11)
    assert _segments(control) == _segments(probed)
    bits = control.config.input_bits

    np.random.seed(99)
    probed.process(_pattern(bits, 50), learn=False)

    np.random.seed(12)
    control.process(_pattern(bits, 10), learn=True)
    np.random.seed(12)
    probed.process(_pattern(bits, 10), learn=True)

    assert _segments(probed) == _segments(control)
    assert probed.anomaly_history == control.anomaly_history


def test_live_history_is_bounded():
    np.random.seed(0)
    cfg = _tiny_config()
    htm = ProductionHTM(cfg)
    pattern = _pattern(cfg.input_bits, 0)

    for _ in range(ANOMALY_HISTORY_LIMIT + 5):
        htm.process(pattern, learn=False)

    assert len(htm.live_anomaly_history) == ANOMALY_HISTORY_LIMIT
    assert htm.anomaly_history == []
