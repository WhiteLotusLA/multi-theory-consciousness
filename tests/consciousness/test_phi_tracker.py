"""
Tests for PhiTracker: module-state binarization, the empirical TPM, the
latest_phi integration proxy, phi normalization, exact IIT 4.0 Phi through
the PyPhi bridge, spectral neural integration, and persistence.
"""

import asyncio
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from mtc.consciousness import phi_tracker as phi_tracker_module
from mtc.consciousness import pyphi_bridge
from mtc.consciousness.phi_tracker import PhiResult, PhiTracker


def _state(**overrides):
    """A consciousness-state stand-in with every module signal inactive."""
    fields = dict(
        integration_level=0.0,
        is_conscious=False,
        ignition_events=0,
        broadcast_coverage=0.0,
        protoself_stability=0.0,
        attention_schema=None,
        feeling_of_knowing=0.0,
        higher_order_thoughts=[],
        homeostatic_urgency=None,
        prediction_error=0.0,
        binding_quality=0.0,
        beautiful_loop_quality=0.0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mid_state():
    """Every module at signal 0.5, so each bit is ON about half the time."""
    return _state(
        integration_level=0.5,
        protoself_stability=0.5,
        feeling_of_knowing=0.5,
        homeostatic_urgency=("energy", 0.5),
        binding_quality=0.5,
    )


def _warm_tracker(cycles: int = 200, **kwargs) -> PhiTracker:
    kwargs.setdefault("min_transitions", 32)
    tracker = PhiTracker(**kwargs)
    state = _mid_state()
    for _ in range(cycles):
        tracker.record_module_states(state)
    return tracker


# ============================================================================
# Construction
# ============================================================================


class TestConstruction:
    def test_defaults_track_five_modules(self):
        tracker = PhiTracker()
        assert tracker.NODE_LABELS == ["GWT", "AST", "HOT", "FEP", "BLT"]
        assert tracker.N_NODES == 5
        assert tracker.N_STATES == 32
        assert tracker.transition_counts.shape == (32, 5)
        assert tracker.state_visit_counts.shape == (32,)
        assert tracker.phi_scale == 1.0
        assert tracker.phi_clamp is None

    def test_default_connectivity_is_target_by_source(self):
        tracker = PhiTracker()
        cm = tracker.connectivity
        assert cm.shape == (5, 5)
        # GWT receives from every module.
        assert list(cm[0]) == [1, 1, 1, 1, 1]
        # AST reads GWT and itself only.
        assert list(cm[1]) == [1, 1, 0, 0, 0]
        # BLT reads AST, HOT, FEP and itself, not GWT.
        assert list(cm[4]) == [0, 1, 1, 1, 1]

    def test_num_modules_selects_a_prefix_of_the_default_labels(self):
        tracker = PhiTracker(num_modules=3)
        assert tracker.NODE_LABELS == ["GWT", "AST", "HOT"]
        assert tracker.N_STATES == 8
        assert tracker.connectivity.shape == (3, 3)
        np.testing.assert_array_equal(
            tracker.connectivity, PhiTracker().connectivity[:3, :3]
        )

    def test_custom_labels_default_to_full_connectivity(self):
        tracker = PhiTracker(module_labels=["X", "Y"])
        assert tracker.N_NODES == 2
        np.testing.assert_array_equal(tracker.connectivity, np.ones((2, 2)))

    def test_mismatched_label_count_raises(self):
        with pytest.raises(ValueError):
            PhiTracker(num_modules=4, module_labels=["A", "B"])

    def test_bad_connectivity_shape_raises(self):
        with pytest.raises(ValueError):
            PhiTracker(num_modules=3, connectivity=np.ones((5, 5)))


# ============================================================================
# Binarization
# ============================================================================


class TestBinarization:
    def test_extremes_are_deterministic(self):
        tracker = PhiTracker()
        assert tracker.extract_module_states(_state()) == (0, 0, 0, 0, 0)
        full = _state(
            integration_level=1.0,
            protoself_stability=1.0,
            feeling_of_knowing=1.0,
            homeostatic_urgency=("energy", 1.0),
            binding_quality=1.0,
        )
        for _ in range(20):
            assert tracker.extract_module_states(full) == (1, 1, 1, 1, 1)
            tracker.record_module_states(full)

    def test_extract_is_idempotent_between_records(self):
        tracker = PhiTracker()
        state = _mid_state()
        first = tracker.extract_module_states(state)
        assert tracker.extract_module_states(state) == first
        assert tracker._cycle_counter == 0

    def test_binarization_is_diverse_after_history_saturates(self):
        """Hash thresholds keep changing after the bounded history fills."""
        history_size = 16
        tracker = PhiTracker(history_size=history_size)
        state = _mid_state()
        late_states = []
        for cycle in range(10 * history_size):
            tracker.record_module_states(state)
            if cycle >= 8 * history_size:
                late_states.append(tracker.state_history[-1])

        assert len(tracker.state_history) == history_size
        assert tracker._cycle_counter == 10 * history_size
        # A frozen threshold would repeat one state; 32 draws at p = 0.5 per
        # bit visit many states.
        assert len(set(late_states)) > 8

    def test_gwt_bit_is_zero_when_workspace_idle(self):
        """An idle workspace keeps GWT off even when broadcast coverage reads 1.0."""
        tracker = PhiTracker()
        idle = _state(broadcast_coverage=1.0)
        for _ in range(200):
            assert tracker.extract_module_states(idle)[0] == 0
            tracker.record_module_states(idle)

    def test_gwt_bit_follows_ignition_when_integration_unwired(self):
        tracker = PhiTracker()
        active = _state(is_conscious=True, ignition_events=3)
        bits = []
        for _ in range(200):
            bits.append(tracker.extract_module_states(active)[0])
            tracker.record_module_states(active)
        # Signal 1.0 (0.3 + 0.4 + 0.3) keeps the bit on every cycle.
        assert all(bits)

    def test_gwt_prefers_integration_level(self):
        tracker = PhiTracker()
        on = _state(integration_level=1.0)
        assert tracker.extract_module_states(on)[0] == 1

    def test_missing_attributes_read_as_inactive(self):
        tracker = PhiTracker()
        assert tracker.extract_module_states(SimpleNamespace()) == (0, 0, 0, 0, 0)

    def test_state_index_is_big_endian(self):
        tracker = PhiTracker()
        assert tracker._state_to_index((1, 0, 0, 0, 0)) == 16
        assert tracker._state_to_index((0, 0, 0, 0, 1)) == 1
        assert tracker._state_to_index((1, 1, 1, 1, 1)) == 31

    def test_state_index_matches_pyphi_big_endian(self):
        pytest.importorskip("pyphi")
        from pyphi import convert

        tracker = PhiTracker()
        for index in range(32):
            state = convert.be_index2state(index, 5)
            assert tracker._state_to_index(state) == convert.state2be_index(state)


# ============================================================================
# TPM
# ============================================================================


class TestTPM:
    def test_tpm_coverage_grows(self):
        tracker = PhiTracker()
        assert tracker.get_tpm_coverage() == 0.0
        state = _mid_state()
        coverage = []
        for _ in range(4):
            for _ in range(25):
                tracker.record_module_states(state)
            coverage.append(tracker.get_tpm_coverage())
        assert coverage[0] > 0.0
        assert coverage == sorted(coverage)
        assert coverage[-1] > coverage[0]
        assert tracker.tpm_coverage == tracker.get_tpm_coverage()

    def test_build_tpm_shape_and_range(self):
        tracker = _warm_tracker(100)
        tpm = tracker.build_tpm()
        assert tpm.shape == (32, 5)
        assert np.all(tpm > 0.0) and np.all(tpm < 1.0)

    def test_unvisited_rows_are_uniform(self):
        tracker = PhiTracker()
        np.testing.assert_allclose(tracker.build_tpm(), 0.5)

    def test_zero_alpha_does_not_divide_by_zero(self):
        tracker = PhiTracker(laplace_alpha=0.0)
        tpm = tracker.build_tpm()
        assert np.all(np.isfinite(tpm))
        np.testing.assert_allclose(tpm, 0.5)

    def test_observed_row_reflects_counts(self):
        tracker = PhiTracker(laplace_alpha=0.0)
        tracker.transition_counts[3] = [4, 0, 2, 4, 1]
        tracker.state_visit_counts[3] = 4
        np.testing.assert_allclose(tracker.build_tpm()[3], [1.0, 0.0, 0.5, 1.0, 0.25])

    def test_transitions_counted_from_second_record(self):
        tracker = PhiTracker()
        tracker.record_module_states(_state())
        assert tracker.transitions_recorded == 0
        tracker.record_module_states(_state())
        assert tracker.transitions_recorded == 1
        assert tracker.state_visit_counts[0] == 1


# ============================================================================
# latest_phi (integration proxy)
# ============================================================================


def _proxy_by_hand(tracker: PhiTracker) -> float:
    tpm = tracker.build_tpm()
    visits = tracker.state_visit_counts
    weights = visits / visits.sum()
    marginals = np.clip(weights @ tpm, 0.01, 0.99)
    total = 0.0
    for s in np.flatnonzero(visits > 0):
        for i in range(tracker.N_NODES):
            p = min(max(tpm[s, i], 0.01), 0.99)
            q = marginals[i]
            total += weights[s] * (
                p * math.log2(p / q) + (1 - p) * math.log2((1 - p) / (1 - q))
            )
    return total


class TestLatestPhi:
    def test_latest_phi_none_before_warmup(self):
        tracker = PhiTracker(min_transitions=50)
        state = _mid_state()
        for _ in range(10):
            tracker.record_module_states(state)
        assert tracker.warming_up
        assert tracker.latest_phi is None

        for _ in range(60):
            tracker.record_module_states(state)
        assert not tracker.warming_up
        value = tracker.latest_phi
        assert value is not None
        assert value >= 0.0

    def test_latest_phi_is_raw_bits_by_default(self):
        tracker = _warm_tracker(150)
        assert tracker.latest_phi == pytest.approx(_proxy_by_hand(tracker))

    def test_scale_and_clamp(self):
        raw = _warm_tracker(150)
        scaled = _warm_tracker(150, phi_scale=2.0)
        assert raw.latest_phi > 0.0
        assert scaled.latest_phi == pytest.approx(2.0 * raw.latest_phi)

        clamp = raw.latest_phi / 2.0
        clamped = _warm_tracker(150, phi_clamp=clamp)
        assert clamped.latest_phi == pytest.approx(clamp)

    def test_latest_phi_none_and_logged_on_failure(self, monkeypatch, caplog):
        tracker = _warm_tracker(100)

        def explode():
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(tracker, "build_tpm", explode)
        with caplog.at_level("WARNING"):
            assert tracker.latest_phi is None
        assert any("synthetic failure" in r.getMessage() for r in caplog.records)


# ============================================================================
# record_phi and normalize_phi
# ============================================================================


class TestNormalization:
    def test_record_and_normalize_phi(self):
        tracker = PhiTracker()
        # Missing input stays missing.
        assert tracker.normalize_phi(None) is None
        assert tracker.normalize_phi(float("nan")) is None
        # Warm-up: divide by the display calibration reference (5.0).
        assert tracker.normalize_phi(2.5) == pytest.approx(0.5)
        assert tracker.normalize_phi(50.0) == 1.0
        assert tracker.normalize_phi(-1.0) == 0.0

        # Invalid values never enter the history.
        tracker.record_phi(None)
        tracker.record_phi(float("nan"))
        tracker.record_phi(-0.5)
        tracker.record_phi("not a number")
        assert len(tracker._phi_history) == 0

        for value in np.linspace(0.1, 2.0, 40):
            tracker.record_phi(float(value))
        reference = float(np.percentile(np.linspace(0.1, 2.0, 40), 95))
        assert tracker.normalize_phi(1.0) == pytest.approx(1.0 / reference)
        assert tracker.normalize_phi(10.0) == 1.0

    def test_warmup_reference_is_configurable(self):
        tracker = PhiTracker(phi_warmup_reference=2.0)
        assert tracker.normalize_phi(1.0) == pytest.approx(0.5)

    def test_zero_reference_normalizes_to_zero(self):
        tracker = PhiTracker()
        for _ in range(30):
            tracker.record_phi(0.0)
        assert tracker.normalize_phi(1.0) == 0.0


# ============================================================================
# Exact Phi through the bridge
# ============================================================================


def _synthetic_three_module_tracker() -> PhiTracker:
    tracker = PhiTracker(num_modules=3, min_transitions=16)
    patterns = [
        _state(integration_level=0.9, protoself_stability=0.2, feeling_of_knowing=0.7),
        _state(integration_level=0.3, protoself_stability=0.8, feeling_of_knowing=0.4),
        _state(integration_level=0.6, protoself_stability=0.5, feeling_of_knowing=0.1),
    ]
    for cycle in range(120):
        tracker.record_module_states(patterns[cycle % len(patterns)])
    return tracker


class TestArchitecturalPhi:
    def test_compute_architectural_phi_smoke(self):
        pytest.importorskip("pyphi")
        tracker = _synthetic_three_module_tracker()
        result = asyncio.run(tracker.compute_architectural_phi())

        assert isinstance(result, PhiResult)
        assert result.error is None
        assert result.warming_up is False
        assert result.is_exact is True
        assert result.phi is not None
        assert math.isfinite(result.phi)
        assert result.phi >= 0.0
        assert result.computation_time_ms > 0.0
        assert isinstance(result.partition_info, str)
        assert result.transitions_recorded == tracker.transitions_recorded
        assert 0.0 < result.tpm_coverage <= 1.0

    def test_explicit_state_is_used(self):
        pytest.importorskip("pyphi")
        tracker = _synthetic_three_module_tracker()
        for state in [(0, 0, 0), (1, 1, 1), (1, 0, 1)]:
            result = asyncio.run(tracker.compute_architectural_phi(state))
            assert result.error is None
            assert result.phi >= 0.0

    def test_wrong_state_length_is_a_structured_error(self):
        tracker = _synthetic_three_module_tracker()
        result = asyncio.run(tracker.compute_architectural_phi((0, 1)))
        assert result.phi is None
        assert result.error is not None
        assert result.is_exact is False

    def test_warming_up_result(self):
        tracker = PhiTracker(min_transitions=128)
        for _ in range(10):
            tracker.record_module_states(_mid_state())
        result = asyncio.run(tracker.compute_architectural_phi())
        assert result.warming_up is True
        assert result.phi is None
        assert result.error is None
        assert result.transitions_recorded == 9

    def test_missing_pyphi_is_a_structured_error(self, monkeypatch):
        tracker = _synthetic_three_module_tracker()
        monkeypatch.setattr(pyphi_bridge, "pyphi_available", lambda: False)
        result = asyncio.run(tracker.compute_architectural_phi())
        assert result.phi is None
        assert result.warming_up is False
        assert result.error == "PyPhi is not installed"

    def test_exception_is_a_structured_error(self, monkeypatch):
        pytest.importorskip("pyphi")
        tracker = _synthetic_three_module_tracker()

        def explode(subsystem):
            raise RuntimeError("synthetic explosion")

        monkeypatch.setattr(pyphi_bridge, "system_irreducibility", explode)
        result = asyncio.run(tracker.compute_architectural_phi())
        assert result.phi is None
        assert result.is_exact is False
        assert result.error.startswith("RuntimeError")
        assert "synthetic explosion" in result.error

    def test_negative_phi_is_clamped_at_zero(self, monkeypatch):
        pytest.importorskip("pyphi")
        tracker = _synthetic_three_module_tracker()
        fake = SimpleNamespace(phi=-1e-4, normalized_phi=-1e-5, partition=None)
        monkeypatch.setattr(pyphi_bridge, "system_irreducibility", lambda s: fake)
        result = asyncio.run(tracker.compute_architectural_phi())
        assert result.phi == 0.0
        assert result.phi_normalized == 0.0
        assert result.partition_info is None

    def test_network_is_built_through_the_bridge(self, monkeypatch):
        pytest.importorskip("pyphi")
        tracker = _synthetic_three_module_tracker()
        seen = {}
        original = pyphi_bridge.build_network

        def spy(tpm, cm, labels):
            seen["tpm"] = np.array(tpm)
            seen["cm"] = np.array(cm)
            seen["labels"] = list(labels)
            return original(tpm, cm, labels)

        monkeypatch.setattr(pyphi_bridge, "build_network", spy)
        asyncio.run(tracker.compute_architectural_phi())
        # The tracker hands over its own big-endian TPM and target-by-source cm.
        np.testing.assert_array_equal(seen["tpm"], tracker.build_tpm())
        np.testing.assert_array_equal(seen["cm"], tracker.connectivity)
        assert seen["labels"] == tracker.NODE_LABELS

    def test_phi_tracker_module_never_imports_pyphi_directly(self):
        with open(phi_tracker_module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        assert not re.search(r"^\s*(import pyphi\b|from pyphi\b)", source, re.M)
        assert "pyphi.Network" not in source


# ============================================================================
# Neural integration
# ============================================================================


class TestNeuralIntegration:
    def test_neural_integration_zero_on_constant_input(self):
        tracker = PhiTracker()
        constant = np.ones(50)
        assert tracker.get_neural_integration(constant, constant, constant) == 0.0

    def test_correlated_is_more_integrated_than_independent(self):
        tracker = PhiTracker()
        rng = np.random.default_rng(0)
        base = rng.random(50)
        correlated = tracker.get_neural_integration(
            base, base + 0.01 * rng.random(50), base + 0.01 * rng.random(50)
        )
        independent = tracker.get_neural_integration(
            rng.random(50), rng.random(50), rng.random(50)
        )
        assert correlated > independent
        # Identical signals concentrate all variance in one eigenvalue.
        assert tracker.get_neural_integration(base, base, base) == pytest.approx(
            math.log(3)
        )

    def test_short_or_empty_input_is_zero(self):
        tracker = PhiTracker()
        assert tracker.get_neural_integration([1.0], [2.0], [3.0]) == 0.0
        assert tracker.get_neural_integration([], [], []) == 0.0

    def test_uses_common_prefix_of_first_fifty(self):
        tracker = PhiTracker()
        rng = np.random.default_rng(1)
        a, b, c = rng.random(80), rng.random(30), rng.random(60)
        expected = tracker.get_neural_integration(a[:30], b, c[:30])
        assert tracker.get_neural_integration(a, b, c) == pytest.approx(expected)


# ============================================================================
# Persistence
# ============================================================================


class TestPersistence:
    def test_state_dict_round_trip_restores_counter(self):
        tracker = _warm_tracker(70)
        for value in (0.2, 0.4, 0.6):
            tracker.record_phi(value)
        state = tracker.to_state_dict()
        assert set(state) == {"scalars", "arrays", "histories"}

        restored = PhiTracker(min_transitions=32)
        restored.from_state_dict(state)

        assert restored._cycle_counter == tracker._cycle_counter == 70
        np.testing.assert_array_equal(
            restored.transition_counts, tracker.transition_counts
        )
        np.testing.assert_array_equal(
            restored.state_visit_counts, tracker.state_visit_counts
        )
        assert list(restored.state_history) == list(tracker.state_history)
        assert all(isinstance(s, tuple) for s in restored.state_history)
        assert list(restored._phi_history) == [0.2, 0.4, 0.6]
        assert restored.latest_phi == pytest.approx(tracker.latest_phi)

        # The hash thresholds continue where they left off.
        probe = _mid_state()
        assert restored.extract_module_states(probe) == tracker.extract_module_states(
            probe
        )

    def test_shape_mismatch_is_skipped(self):
        tracker = PhiTracker()
        state = PhiTracker(num_modules=3).to_state_dict()
        state["arrays"]["transition_counts"] = np.ones((8, 3))
        state["arrays"]["state_visit_counts"] = np.ones(8)
        state["histories"]["state_history"] = [[0, 1, 0]]
        tracker.from_state_dict(state)
        assert tracker.transition_counts.shape == (32, 5)
        assert tracker.state_visit_counts.sum() == 0
        assert len(tracker.state_history) == 0

    def test_restore_from_lists(self):
        tracker = _warm_tracker(40)
        state = tracker.to_state_dict()
        state["arrays"] = {k: v.tolist() for k, v in state["arrays"].items()}
        restored = PhiTracker()
        restored.from_state_dict(state)
        np.testing.assert_array_equal(
            restored.transition_counts, tracker.transition_counts
        )
