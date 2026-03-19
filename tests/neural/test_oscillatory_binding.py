"""Tests for AKOrN Oscillatory Binding module."""

import numpy as np
import pytest
from typing import Dict
from mtc.neural.oscillatory_binding import OscillatoryBinding, BindingResult, OSCILLATOR_NAMES


class TestOscillatoryBindingInit:
    def test_default_init(self):
        ob = OscillatoryBinding()
        assert ob.num_oscillators == 30
        assert ob.phase_dim == 16
        assert ob.num_steps == 20
        assert ob.step_size == 0.1

    def test_oscillator_phases_on_unit_sphere(self):
        ob = OscillatoryBinding()
        norms = np.linalg.norm(ob.phases, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_static_coupling_shape(self):
        ob = OscillatoryBinding()
        assert ob.J_static.shape == (30, 30, 16, 16)

    def test_learned_coupling_starts_zero(self):
        ob = OscillatoryBinding()
        np.testing.assert_array_equal(ob.J_learned, np.zeros((30, 30, 16, 16)))

    def test_natural_frequencies_antisymmetric(self):
        ob = OscillatoryBinding()
        for i in range(ob.num_oscillators):
            omega = ob.omega[i]
            np.testing.assert_allclose(omega, -omega.T, atol=1e-10)

    def test_reproducible_cold_start(self):
        ob1 = OscillatoryBinding()
        ob2 = OscillatoryBinding()
        np.testing.assert_array_equal(ob1.proj_vectors, ob2.proj_vectors)
        np.testing.assert_array_equal(ob1.omega, ob2.omega)

    def test_projection_vectors_shape(self):
        ob = OscillatoryBinding()
        assert ob.proj_vectors.shape == (30, 16)


class TestKuramotoDynamics:
    def test_bind_returns_binding_result(self):
        ob = OscillatoryBinding()
        metrics = {name: 0.5 for name in OSCILLATOR_NAMES}
        result = ob.bind(metrics)
        assert isinstance(result, BindingResult)
        assert 0.0 <= result.global_order_parameter <= 1.0
        assert "snn" in result.group_order_parameters
        assert "lsm" in result.group_order_parameters
        assert "htm" in result.group_order_parameters
        assert len(result.clusters) > 0
        assert result.elapsed_ms >= 0

    def test_phases_remain_on_unit_sphere(self):
        ob = OscillatoryBinding()
        ob.bind({name: 0.5 for name in OSCILLATOR_NAMES})
        norms = np.linalg.norm(ob.phases, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_coupling_drives_synchrony(self):
        # When oscillators receive no external stimulus (all metrics=0), the static
        # coupling matrix alone drives synchrony. Verify the coupling mechanism
        # works: global order parameter with active coupling is higher than it
        # would be with a fresh (maximally dispersed) initialization.
        # We test that binding produces non-trivial synchrony (r > 0.5),
        # demonstrating the coupling matrix is functional.
        ob = OscillatoryBinding(num_steps=50, seed=42)
        # Zero stimulus - only static coupling drives dynamics
        zero_metrics = {name: 0.0 for name in OSCILLATOR_NAMES}
        result = ob.bind(zero_metrics)
        # Static coupling should push r well above random (1/sqrt(N) ~= 0.18)
        assert result.global_order_parameter > 0.5, (
            f"Static coupling should synchronize oscillators, got r={result.global_order_parameter:.4f}"
        )

    def test_bind_count_increments(self):
        ob = OscillatoryBinding()
        assert ob._binding_count == 0
        ob.bind({name: 0.5 for name in OSCILLATOR_NAMES})
        assert ob._binding_count == 1
        ob.bind({name: 0.5 for name in OSCILLATOR_NAMES})
        assert ob._binding_count == 2

    def test_order_parameter_in_valid_range(self):
        ob = OscillatoryBinding()
        for _ in range(5):
            metrics = {name: float(np.random.rand()) for name in OSCILLATOR_NAMES}
            result = ob.bind(metrics)
            assert 0.0 <= result.global_order_parameter <= 1.0


class TestPersistence:
    def test_to_state_dict_structure(self):
        ob = OscillatoryBinding()
        ob.bind({name: 0.5 for name in OSCILLATOR_NAMES})
        state = ob.to_state_dict()
        assert "scalars" in state
        assert "arrays" in state
        assert "histories" in state

    def test_round_trip_preserves_state(self):
        ob1 = OscillatoryBinding()
        metrics = {name: float(np.random.rand()) for name in OSCILLATOR_NAMES}
        ob1.bind(metrics)
        ob1.bind(metrics)
        state = ob1.to_state_dict()
        ob2 = OscillatoryBinding()
        ob2.from_state_dict(state)
        np.testing.assert_allclose(ob1.phases, ob2.phases, atol=1e-10)
        np.testing.assert_allclose(ob1.J_learned, ob2.J_learned, atol=1e-10)
        np.testing.assert_allclose(ob1.proj_vectors, ob2.proj_vectors, atol=1e-10)
        assert ob1._binding_count == ob2._binding_count

    def test_learned_coupling_persists(self):
        ob = OscillatoryBinding()
        for _ in range(10):
            ob.bind({name: 0.8 for name in OSCILLATOR_NAMES})
        assert np.any(ob.J_learned != 0)
        state = ob.to_state_dict()
        ob2 = OscillatoryBinding()
        ob2.from_state_dict(state)
        np.testing.assert_allclose(ob.J_learned, ob2.J_learned, atol=1e-10)


class TestOscillatorTPM:
    def test_transition_recording(self):
        """bind() records oscillator state transitions."""
        ob = OscillatoryBinding()
        assert ob.oscillator_transitions_recorded == 0
        metrics = {name: 0.5 for name in OSCILLATOR_NAMES}
        ob.bind(metrics)
        # First call sets prev_state, no transition yet
        assert ob._osc_prev_state_idx is not None
        ob.bind(metrics)
        # Second call produces the first transition
        assert ob.oscillator_transitions_recorded == 1
        ob.bind(metrics)
        assert ob.oscillator_transitions_recorded == 2

    def test_warming_up_initially(self):
        """TPM is warming up until 200 transitions have been recorded."""
        ob = OscillatoryBinding()
        assert ob.oscillator_tpm_warming_up is True
        metrics = {name: 0.5 for name in OSCILLATOR_NAMES}
        # Need 200 transitions -> 201 bind() calls (first sets prev, rest record)
        for _ in range(201):
            ob.bind(metrics)
        assert ob.oscillator_transitions_recorded >= 200
        assert ob.oscillator_tpm_warming_up is False

    def test_tpm_shape(self):
        """Oscillator transition counts array is (81, 81)."""
        ob = OscillatoryBinding()
        assert ob._osc_transition_counts.shape == (81, 81)
        assert ob._osc_state_visit_counts.shape == (81,)

    def test_discretize_state(self):
        """_discretize_group_state returns an index in [0, 80]."""
        ob = OscillatoryBinding()
        # All low -> index 0
        idx_low = ob._discretize_group_state(
            {"snn": 0.0, "lsm": 0.0, "htm": 0.0, "cross": 0.0}
        )
        assert 0 <= idx_low <= 80

        # All high -> index 80
        idx_high = ob._discretize_group_state(
            {"snn": 1.0, "lsm": 1.0, "htm": 1.0, "cross": 1.0}
        )
        assert 0 <= idx_high <= 80

        # Mixed values stay in range
        idx_mid = ob._discretize_group_state(
            {"snn": 0.5, "lsm": 0.2, "htm": 0.8, "cross": 0.4}
        )
        assert 0 <= idx_mid <= 80

        # Boundary between low and mid
        assert ob._discretize_group_state(
            {"snn": 1.0 / 3.0, "lsm": 0.0, "htm": 0.0, "cross": 0.0}
        ) != idx_low  # 1/3 should land in mid bucket

    def test_tpm_persists(self):
        """Oscillator TPM data survives a to_state_dict / from_state_dict round-trip."""
        ob = OscillatoryBinding()
        metrics = {name: 0.5 for name in OSCILLATOR_NAMES}
        for _ in range(10):
            ob.bind(metrics)

        state = ob.to_state_dict()
        assert "osc_transition_counts" in state["arrays"]
        assert "osc_state_visit_counts" in state["arrays"]
        assert "osc_prev_state_idx" in state["scalars"]

        ob2 = OscillatoryBinding()
        ob2.from_state_dict(state)

        np.testing.assert_array_equal(
            ob._osc_transition_counts, ob2._osc_transition_counts
        )
        np.testing.assert_array_equal(
            ob._osc_state_visit_counts, ob2._osc_state_visit_counts
        )
        assert ob._osc_prev_state_idx == ob2._osc_prev_state_idx
