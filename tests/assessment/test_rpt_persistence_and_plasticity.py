"""RPT measurement: STDP plasticity component, binding stats, persistence."""

from collections import deque

import numpy as np
import pytest

from mtc.assessment.rpt_measurement import (
    RecurrenceMetrics,
    RecurrenceType,
    RPTMeasurement,
)


def _snn_state(plasticity_active: bool = True) -> dict:
    return {
        "total_spikes": 120,
        "num_neurons": 100,
        "spike_counts": [np.array([10]), np.array([25]), np.array([8])],
        "activation_pattern": np.linspace(0.0, 1.0, 100),
        "stdp_statistics": {
            "plasticity_active": plasticity_active,
            "stdp_enabled": True,
            "num_stdp_layers": 2,
            "mean_weight_magnitude": 0.4,
            "mean_weight_std": 0.15,
            "weight_diversity": 0.2,
        },
    }


def _workspace_state() -> dict:
    return {
        "broadcast": {"total_broadcasts": 8, "coverage_ratio": 0.6},
        "cycle_count": 6,
        "reentry_count": 4,
        "source_module_diversity": 3,
        "ignition": {"total_ignitions": 2},
    }


class _FakeBinding:
    def __init__(self, history, cross, count):
        self._order_history = deque(history, maxlen=100)
        self._last_cross_coherence = cross
        self._binding_count = count


def test_plasticity_component_when_stdp_active():
    rpt = RPTMeasurement()
    with_stdp = rpt._measure_snn_recurrence(_snn_state(plasticity_active=True))
    without = rpt._measure_snn_recurrence(_snn_state(plasticity_active=False))
    # Three components without plasticity, four with; the fourth is
    # min(1, 0.15 * 2 + 0.2) = 0.5.
    state = _snn_state(plasticity_active=False)
    base_sum = without * 3
    assert with_stdp == pytest.approx((base_sum + 0.5) / 4)
    assert with_stdp != pytest.approx(without)
    metrics = rpt.measure_full(snn_state=_snn_state(True))
    assert metrics.stdp_plasticity_score == pytest.approx(0.3)
    assert state["stdp_statistics"]["plasticity_active"] is False


def test_plasticity_ignored_when_disabled():
    enabled = RPTMeasurement()
    disabled = RPTMeasurement(stdp_plasticity_enabled=False)
    plain = _snn_state(plasticity_active=False)
    active = _snn_state(plasticity_active=True)
    assert disabled._measure_snn_recurrence(active) == pytest.approx(
        enabled._measure_snn_recurrence(plain)
    )
    metrics = disabled.measure_full(snn_state=active)
    assert metrics.stdp_plasticity_score == 0.0


def test_plasticity_counted_once():
    rpt = RPTMeasurement()
    metrics = rpt.measure_full(
        snn_state=_snn_state(True), workspace_state=_workspace_state()
    )
    assert metrics.stdp_plasticity_score > 0.0
    assert metrics.phenomenal_consciousness == pytest.approx(
        metrics.local_recurrence_score
    )


def test_state_dict_round_trip_rehydrates_metrics():
    rpt = RPTMeasurement(history_window=4)
    for _ in range(6):
        rpt.measure_full(snn_state=_snn_state(True), workspace_state=_workspace_state())
    state = rpt.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}

    restored = RPTMeasurement(history_window=4)
    restored.from_state_dict(state)
    assert restored.measurement_count == 6
    assert isinstance(restored._metrics_history, deque)
    assert restored._metrics_history.maxlen == 4
    assert restored._local_scores.maxlen == 4
    assert all(isinstance(m, RecurrenceMetrics) for m in restored._metrics_history)
    assert all(
        isinstance(m.recurrence_type, RecurrenceType) for m in restored._metrics_history
    )
    stats = restored.get_statistics()
    assert stats["measurement_count"] == 6
    assert stats["average_local"] == pytest.approx(
        rpt.get_statistics()["average_local"]
    )
    report = restored.generate_report()
    assert "Measurements: 6" in report


def test_stdp_score_serialized():
    rpt = RPTMeasurement()
    rpt.measure_full(snn_state=_snn_state(True))
    state = rpt.to_state_dict()
    record = state["histories"]["metrics_history"][0]
    assert record["stdp_plasticity_score"] == pytest.approx(0.3)
    assert record["recurrence_type"] in {t.value for t in RecurrenceType}
    restored = RPTMeasurement()
    restored.from_state_dict(state)
    assert restored._metrics_history[0].stdp_plasticity_score == pytest.approx(0.3)


def test_restore_skips_malformed_and_nonfinite_records():
    rpt = RPTMeasurement()
    rpt.measure_full(snn_state=_snn_state(True))
    state = rpt.to_state_dict()
    state["histories"]["metrics_history"].append({"recurrence_type": "bogus"})
    state["histories"]["local_scores"].append(float("nan"))
    restored = RPTMeasurement()
    restored.from_state_dict(state)
    assert len(restored._metrics_history) == 1
    assert len(restored._local_scores) == 1


def test_pushed_stats_preferred_over_attached():
    rpt = RPTMeasurement()
    rpt.attach_oscillatory_binding(_FakeBinding([0.2, 0.4], {"snn_lsm": 0.1}, 2))
    pushed = {
        "global_order_parameter": 0.9,
        "cross_coherence": {"snn_lsm": 0.7},
        "binding_count": 11,
    }
    rpt.set_oscillatory_binding_stats(pushed)
    stats = rpt.get_oscillatory_binding_stats()
    assert stats["global_order_parameter"] == pytest.approx(0.9)
    assert stats["binding_count"] == 11


def test_attached_binding_reports_mean_order_parameter():
    rpt = RPTMeasurement()
    assert rpt.get_oscillatory_binding_stats() == {}
    binding = _FakeBinding([0.2, 0.4, 0.6], {"snn_lsm": 0.3}, 3)
    rpt.attach_oscillatory_binding(binding)
    assert rpt.oscillatory_binding is binding
    stats = rpt.get_oscillatory_binding_stats()
    assert stats["global_order_parameter"] == pytest.approx(0.4)
    assert stats["cross_coherence"] == {"snn_lsm": 0.3}
    assert stats["binding_count"] == 3


def test_private_attribute_wiring_still_reads():
    """The runner's current ``rpt._oscillatory = osc`` wiring keeps working."""
    rpt = RPTMeasurement()
    rpt._oscillatory = _FakeBinding([0.5], {}, 1)
    assert rpt.get_oscillatory_binding_stats()["binding_count"] == 1
