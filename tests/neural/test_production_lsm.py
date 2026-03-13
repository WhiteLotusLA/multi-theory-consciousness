"""
Tests for ProductionLSM - the system's Liquid State Machine implementation.
ReservoirPy components are mocked so no heavy computation is needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — build mocks that behave like reservoirpy nodes
# ---------------------------------------------------------------------------

def _make_mock_reservoir(units: int = 100):
    """Return a mock Reservoir node whose .run() produces plausible states."""
    reservoir = MagicMock()

    def run_side_effect(data):
        timesteps = data.shape[0]
        return np.random.randn(timesteps, units).astype(np.float32) * 0.5

    reservoir.run = MagicMock(side_effect=run_side_effect)
    reservoir.__rshift__ = MagicMock(return_value=reservoir)
    return reservoir


def _make_mock_readout(output_dim: int = 50):
    """Return a mock RLS/Ridge readout node."""
    readout = MagicMock()

    def run_side_effect(states):
        timesteps = states.shape[0]
        return np.random.randn(timesteps, output_dim).astype(np.float32) * 0.1

    readout.run = MagicMock(side_effect=run_side_effect)
    readout.fit = MagicMock()
    readout.train = MagicMock()
    readout.__rshift__ = MagicMock(return_value=readout)
    return readout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lsm():
    """Create a ProductionLSM with all reservoirpy internals mocked."""
    with patch("src.neural.liquid.production_lsm.rpy") as mock_rpy, \
         patch("src.neural.liquid.production_lsm.Reservoir") as MockReservoir, \
         patch("src.neural.liquid.production_lsm.RLS") as MockRLS, \
         patch("src.neural.liquid.production_lsm.Ridge") as MockRidge, \
         patch("src.neural.liquid.production_lsm.Input") as MockInput:

        mock_reservoir = _make_mock_reservoir(units=100)
        MockReservoir.return_value = mock_reservoir

        mock_rls = _make_mock_readout(output_dim=50)
        MockRLS.return_value = mock_rls

        mock_ridge = _make_mock_readout(output_dim=50)
        MockRidge.return_value = mock_ridge

        mock_input = MagicMock()
        mock_input.__rshift__ = MagicMock(return_value=mock_reservoir)
        MockInput.return_value = mock_input

        from mtc.neural.liquid.production_lsm import ProductionLSM, LSMConfig

        config = LSMConfig(
            reservoir_size=100,
            input_dim=50,
            output_dim=50,
            spectral_radius=0.9,
            connectivity=0.1,
            washout=5,
        )
        model = ProductionLSM(config=config, fast_mode=True)

    return model


@pytest.fixture
def lsm_slow():
    """Create a ProductionLSM with fast_mode=False for metric tests."""
    with patch("src.neural.liquid.production_lsm.rpy"), \
         patch("src.neural.liquid.production_lsm.Reservoir") as MockReservoir, \
         patch("src.neural.liquid.production_lsm.RLS") as MockRLS, \
         patch("src.neural.liquid.production_lsm.Ridge") as MockRidge, \
         patch("src.neural.liquid.production_lsm.Input") as MockInput:

        mock_reservoir = _make_mock_reservoir(units=100)
        MockReservoir.return_value = mock_reservoir

        MockRLS.return_value = _make_mock_readout(output_dim=50)
        MockRidge.return_value = _make_mock_readout(output_dim=50)

        mock_input = MagicMock()
        mock_input.__rshift__ = MagicMock(return_value=mock_reservoir)
        MockInput.return_value = mock_input

        from mtc.neural.liquid.production_lsm import ProductionLSM, LSMConfig

        config = LSMConfig(
            reservoir_size=100,
            input_dim=50,
            output_dim=50,
            washout=5,
        )
        model = ProductionLSM(config=config, fast_mode=False)

    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLSMConfig:
    def test_default_config_values(self):
        from mtc.neural.liquid.production_lsm import LSMConfig

        cfg = LSMConfig()
        assert cfg.reservoir_size == 10000
        assert cfg.input_dim == 100
        assert cfg.output_dim == 50
        assert cfg.spectral_radius == 0.9
        assert cfg.connectivity == 0.1

    def test_custom_config(self):
        from mtc.neural.liquid.production_lsm import LSMConfig

        cfg = LSMConfig(reservoir_size=500, input_dim=20, output_dim=10)
        assert cfg.reservoir_size == 500
        assert cfg.input_dim == 20
        assert cfg.output_dim == 10


class TestProcessInput:
    def test_2d_input_returns_output_and_state(self, lsm):
        data = np.random.randn(20, 50).astype(np.float32)
        output, state = lsm.process_input(data)

        assert output.shape[0] == 20
        assert output.shape[1] == 50
        assert "processing_time_ms" in state
        assert "mean_activity" in state
        assert "sparsity" in state

    def test_1d_input_is_reshaped(self, lsm):
        data = np.random.randn(50).astype(np.float32)
        output, state = lsm.process_input(data)
        assert output is not None
        assert "processing_time_ms" in state

    def test_3d_batch_input(self, lsm):
        data = np.random.randn(3, 10, 50).astype(np.float32)
        outputs, states = lsm.process_input(data)

        assert outputs.shape[0] == 3
        assert len(states) == 3

    def test_fast_mode_skips_expensive_metrics(self, lsm):
        data = np.random.randn(10, 50).astype(np.float32)
        _, state = lsm.process_input(data)

        assert state["lyapunov_estimate"] is None
        assert state["edge_of_chaos_score"] is None
        assert state["entropy"] is None
        assert state["correlation_time"] is None


class TestProcessSubconscious:
    def test_1d_sensory_input(self, lsm):
        sensory = np.random.randn(50).astype(np.float32)
        result = lsm.process_subconscious(sensory)

        assert "emotional_valence" in result
        assert "emotional_arousal" in result
        assert "creative_emergence" in result
        assert "stability" in result
        assert "processing_depth" in result

    def test_2d_sensory_input(self, lsm):
        sensory = np.random.randn(10, 50).astype(np.float32)
        result = lsm.process_subconscious(sensory)

        assert isinstance(result["emotional_valence"], float)
        assert isinstance(result["emotional_arousal"], float)


class TestAnalyzeReservoirState:
    def test_basic_metrics_always_present(self, lsm):
        states = np.random.randn(20, 100).astype(np.float32)
        metrics = lsm._analyze_reservoir_state(states)

        assert "mean_activity" in metrics
        assert "std_activity" in metrics
        assert "sparsity" in metrics

    def test_slow_mode_computes_all_metrics(self, lsm_slow):
        states = np.random.randn(20, 100).astype(np.float32)
        metrics = lsm_slow._analyze_reservoir_state(states)

        assert metrics["lyapunov_estimate"] is not None
        assert metrics["edge_of_chaos_score"] is not None
        assert metrics["entropy"] is not None
        assert metrics["correlation_time"] is not None


class TestDynamicsMetrics:
    def test_estimate_lyapunov_returns_float(self, lsm):
        states = np.random.randn(20, 100).astype(np.float32)
        result = lsm._estimate_lyapunov(states)
        assert isinstance(result, float)

    def test_estimate_lyapunov_short_input(self, lsm):
        states = np.random.randn(5, 100).astype(np.float32)
        result = lsm._estimate_lyapunov(states)
        assert result == 0.0

    def test_edge_of_chaos_score_bounded(self, lsm):
        states = np.random.randn(20, 100).astype(np.float32)
        score = lsm._edge_of_chaos_score(states)
        assert 0.0 <= score <= 1.0

    def test_entropy_positive(self, lsm):
        states = np.random.randn(50, 100).astype(np.float32)
        entropy = lsm._calculate_entropy(states)
        assert entropy > 0.0

    def test_correlation_time_bounded(self, lsm):
        states = np.random.randn(50, 100).astype(np.float32)
        ct = lsm._correlation_time(states)
        assert 0.0 <= ct <= 1.0
