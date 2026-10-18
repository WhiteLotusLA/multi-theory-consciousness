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
    assert metrics.local_recurrence_index == pytest.approx(
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


def test_none_local_fields_round_trip():
    rpt = RPTMeasurement(history_window=4)
    rpt.measure_full(workspace_state=_workspace_state())
    rpt.measure_full(snn_state=_snn_state(True), workspace_state=_workspace_state())
    state = rpt.to_state_dict()
    first, second = state["histories"]["metrics_history"]
    assert first["local_recurrence_score"] is None
    assert first["snn_feedback_strength"] is None
    assert first["lsm_reservoir_recurrence"] is None
    assert first["local_spike_reentry"] is None
    assert second["local_recurrence_score"] is not None
    assert second["lsm_reservoir_recurrence"] is None
    assert state["scalars"]["local_measurement_count"] == 1
    assert state["histories"]["local_scores"] == [second["local_recurrence_score"]]

    restored = RPTMeasurement(history_window=4)
    restored.from_state_dict(state)
    assert restored.measurement_count == 2
    assert restored.local_measurement_count == 1
    assert len(restored._metrics_history) == 2
    assert restored._metrics_history[0].local_recurrence_score is None
    assert restored._metrics_history[0].lsm_reservoir_recurrence is None
    assert restored._metrics_history[1].local_recurrence_score == pytest.approx(
        second["local_recurrence_score"]
    )
    assert restored.get_statistics()["average_local"] == pytest.approx(
        rpt.get_statistics()["average_local"]
    )
    # A state written before the counter existed falls back to the scores kept
    del state["scalars"]["local_measurement_count"]
    older = RPTMeasurement(history_window=4)
    older.from_state_dict(state)
    assert older.local_measurement_count == 1


def test_recurrence_indices_are_recurrence_statistics():
    rpt = RPTMeasurement()
    metrics = rpt.measure_full(
        snn_state=_snn_state(True), workspace_state=_workspace_state()
    )
    local = metrics.local_recurrence_score
    global_ = metrics.global_recurrence_score
    assert metrics.local_recurrence_index == pytest.approx(local)
    assert metrics.combined_recurrence_index == pytest.approx(
        0.5 * min(local, global_) + 0.5 * global_
    )
    latest = rpt.get_statistics()["latest"]
    assert latest["local_index"] == pytest.approx(local)
    assert latest["combined_index"] == pytest.approx(metrics.combined_recurrence_index)
    assert "phenomenal" not in latest and "access" not in latest
    report = rpt.generate_report()
    assert "local_index=" in report and "combined_index=" in report
    assert "phenomenal" not in report and "access=" not in report
    record = rpt.to_state_dict()["histories"]["metrics_history"][0]
    assert record["local_recurrence_index"] == pytest.approx(local)
    assert record["combined_recurrence_index"] == pytest.approx(
        metrics.combined_recurrence_index
    )


def test_old_record_keys_restore_into_new_fields():
    rpt = RPTMeasurement()
    rpt.measure_full(snn_state=_snn_state(True), workspace_state=_workspace_state())
    state = rpt.to_state_dict()
    record = state["histories"]["metrics_history"][0]
    # A record written before the rename carries the old keys only
    old = dict(record)
    old["phenomenal_consciousness"] = old.pop("local_recurrence_index")
    old["access_consciousness"] = old.pop("combined_recurrence_index")
    state["histories"]["metrics_history"] = [old]
    restored = RPTMeasurement()
    restored.from_state_dict(state)
    assert len(restored._metrics_history) == 1
    metrics = restored._metrics_history[0]
    assert metrics.local_recurrence_index == pytest.approx(
        record["local_recurrence_index"]
    )
    assert metrics.combined_recurrence_index == pytest.approx(
        record["combined_recurrence_index"]
    )
    assert not hasattr(metrics, "phenomenal_consciousness")
    # Without either key the record is malformed and skipped
    neither = dict(record)
    del neither["local_recurrence_index"]
    state["histories"]["metrics_history"] = [neither]
    skipped = RPTMeasurement()
    skipped.from_state_dict(state)
    assert len(skipped._metrics_history) == 0


def test_old_identifiers_survive_only_in_the_restore_fallback():
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "mtc"
        / "assessment"
        / "rpt_measurement.py"
    ).read_text()
    block = re.search(r"_RENAMED_RECORD_KEYS\s*=\s*\{[^}]*\}", source)
    assert block is not None
    assert "phenomenal_consciousness" in block.group(0)
    assert "access_consciousness" in block.group(0)
    rest = source.replace(block.group(0), "")
    assert "phenomenal_consciousness" not in rest
    assert "access_consciousness" not in rest
    # The module says what it measures: recurrence statistics, not either
    # kind of consciousness.
    assert "measures recurrence statistics only" in source


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
