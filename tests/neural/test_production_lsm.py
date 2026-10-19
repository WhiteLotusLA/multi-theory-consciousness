"""
Tests for ProductionLSM - the system's Liquid State Machine implementation.
ReservoirPy components are mocked so no heavy computation is needed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: build mocks that behave like reservoirpy nodes
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
    readout.__rshift__ = MagicMock(return_value=readout)
    return readout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lsm():
    """Create a ProductionLSM with all reservoirpy internals mocked."""
    with (
        patch("mtc.neural.liquid.production_lsm.rpy"),
        patch("mtc.neural.liquid.production_lsm.Reservoir") as MockReservoir,
        patch("mtc.neural.liquid.production_lsm.RLS") as MockRLS,
        patch("mtc.neural.liquid.production_lsm.Ridge") as MockRidge,
        patch("mtc.neural.liquid.production_lsm.Input") as MockInput,
    ):

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
    with (
        patch("mtc.neural.liquid.production_lsm.rpy"),
        patch("mtc.neural.liquid.production_lsm.Reservoir") as MockReservoir,
        patch("mtc.neural.liquid.production_lsm.RLS") as MockRLS,
        patch("mtc.neural.liquid.production_lsm.Ridge") as MockRidge,
        patch("mtc.neural.liquid.production_lsm.Input") as MockInput,
    ):

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
        assert (
            cfg.reservoir_size == 5000
        )  # Scaled from 10K: 5K is latency ceiling for CPU-bound ReservoirPy
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

        assert isinstance(result["readout_mean"], float)
        assert isinstance(result["readout_std"], float)

    def test_untrained_readout_reports_no_affect(self, lsm):
        result = lsm.process_subconscious(np.random.randn(10, 50).astype(np.float32))

        assert result["emotional_valence"] is None
        assert result["emotional_arousal"] is None
        output = np.asarray(result["raw_output"])
        assert result["readout_mean"] == pytest.approx(float(np.mean(output)))
        assert result["readout_std"] == pytest.approx(float(np.std(output)))

    def test_trained_readout_reports_affect(self, lsm):
        lsm.train_rls(
            np.random.randn(2, 12, 50).astype(np.float32),
            np.random.randn(2, 12, 50).astype(np.float32),
        )
        result = lsm.process_subconscious(np.random.randn(10, 50).astype(np.float32))

        assert result["emotional_valence"] == pytest.approx(result["readout_mean"])
        assert result["emotional_arousal"] == pytest.approx(result["readout_std"])

    def test_fast_mode_leaves_unmeasured_dynamics_unset(self, lsm):
        result = lsm.process_subconscious(np.random.randn(10, 50).astype(np.float32))

        # No 0.5 stand-ins for the metrics fast mode skips
        assert result["creative_emergence"] is None
        assert result["stability"] is None
        assert result["processing_depth"] is None

    def test_slow_mode_derives_dynamics_from_measurements(self, lsm_slow):
        result = lsm_slow.process_subconscious(
            np.random.randn(20, 50).astype(np.float32)
        )

        assert result["creative_emergence"] == result["entropy"]
        assert result["stability"] == pytest.approx(1.0 - result["edge_of_chaos_score"])
        assert result["processing_depth"] == result["correlation_time"]

    def test_short_input_leaves_edge_and_stability_unset(self, lsm_slow):
        result = lsm_slow.process_subconscious(
            np.random.randn(5, 50).astype(np.float32)
        )

        assert result["lyapunov_estimate"] is None
        assert result["edge_of_chaos_score"] is None
        assert result["stability"] is None
        assert result["dynamics_reason"] == "too_few_states"
        # Entropy and the lag-1 correlation still have enough states
        assert result["creative_emergence"] == result["entropy"]
        assert result["processing_depth"] == result["correlation_time"]
        assert result["correlation_time"] is not None


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
        assert result is None

    def test_edge_of_chaos_none_for_short_trajectory(self, lsm):
        states = np.random.randn(9, 100).astype(np.float32)
        assert lsm._edge_of_chaos_score(states) is None
        longer = np.random.randn(10, 100).astype(np.float32)
        assert 0.0 <= lsm._edge_of_chaos_score(longer) <= 1.0

    def test_correlation_time_none_for_short_trajectory(self, lsm):
        assert lsm._correlation_time(np.random.randn(1, 100)) is None
        assert lsm._correlation_time(np.random.randn(2, 100)) is None
        assert lsm._correlation_time(np.ones((6, 100))) is None
        assert 0.0 <= lsm._correlation_time(np.random.randn(3, 100)) <= 1.0

    def test_fast_mode_reason(self, lsm):
        metrics = lsm._analyze_reservoir_state(np.random.randn(20, 100))
        assert metrics["dynamics_reason"] == "fast_mode"

    def test_slow_mode_full_trajectory_has_no_reason(self, lsm_slow):
        metrics = lsm_slow._analyze_reservoir_state(np.random.randn(20, 100))
        assert metrics["dynamics_reason"] is None

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


def test_import_does_not_seed_the_library(monkeypatch):
    """Loading the module seeds nothing; a construction seeds the library
    once with the configured seed."""
    import importlib.util
    import sys

    import reservoirpy

    calls = []
    monkeypatch.setattr(reservoirpy, "set_seed", lambda seed: calls.append(seed))
    import mtc.neural.liquid.production_lsm as shipped

    name = "_lsm_import_probe"
    spec = importlib.util.spec_from_file_location(name, shipped.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        assert calls == []

        module.ProductionLSM(
            module.LSMConfig(reservoir_size=12, input_dim=3, output_dim=2, seed=7)
        )
        assert calls == [7]

        module.ProductionLSM(
            module.LSMConfig(reservoir_size=12, input_dim=3, output_dim=2, seed=None)
        )
        assert calls == [7]
    finally:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# The offline ridge readout on a real (small) reservoir
# ---------------------------------------------------------------------------

RIDGE_INPUT_DIM = 4
RIDGE_OUTPUT_DIM = 2


def _real_lsm(**overrides):
    """A real ProductionLSM small enough for a unit test (40 units)."""
    pytest.importorskip("reservoirpy")
    from mtc.neural.liquid.production_lsm import LSMConfig, ProductionLSM

    values = dict(
        reservoir_size=40,
        input_dim=RIDGE_INPUT_DIM,
        output_dim=RIDGE_OUTPUT_DIM,
        washout=5,
    )
    values.update(overrides)
    return ProductionLSM(LSMConfig(**values))


def _sequences(seed: int, samples: int = 2, timesteps: int = 30):
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((samples, timesteps, RIDGE_INPUT_DIM))
    targets = rng.standard_normal((samples, timesteps, RIDGE_OUTPUT_DIM))
    return inputs, targets


def test_train_ridge_alone_reports_affect_from_the_ridge_readout():
    """After train_ridge and no other training, the live output is the ridge
    readout's and the affect fields carry its statistics."""
    lsm = _real_lsm()
    inputs, targets = _sequences(0)

    loss = lsm.train_ridge(inputs, targets)

    assert isinstance(loss, float) and np.isfinite(loss)
    assert lsm.readout_trained is True
    assert lsm.active_readout == "ridge"

    probe = np.random.default_rng(1).standard_normal((8, RIDGE_INPUT_DIM))
    lsm.reservoir.reset()
    result = lsm.process_subconscious(probe)
    output = np.asarray(result["raw_output"])
    lsm.reservoir.reset()
    expected = (
        lsm.reservoir.run(probe) @ lsm.ridge_readout.Wout + lsm.ridge_readout.bias
    )
    assert output.shape == (8, RIDGE_OUTPUT_DIM)
    np.testing.assert_allclose(output, expected, rtol=1e-10, atol=1e-12)
    assert isinstance(result["emotional_valence"], float)
    assert result["emotional_valence"] == pytest.approx(float(np.mean(output)))
    assert result["emotional_arousal"] == pytest.approx(float(np.std(output)))


def test_train_ridge_refuses_sequences_no_longer_than_the_washout():
    """With nothing left after the washout there is nothing to fit: the call
    raises and the readout stays untrained."""
    lsm = _real_lsm(washout=10)
    inputs, targets = _sequences(2, timesteps=10)

    with pytest.raises(ValueError, match="washout"):
        lsm.train_ridge(inputs, targets)

    assert lsm.readout_trained is False
    assert lsm.active_readout == "rls"
    result = lsm.process_subconscious(np.zeros((4, RIDGE_INPUT_DIM)))
    assert result["emotional_valence"] is None


def test_the_latest_training_selects_the_live_readout():
    """train_rls after train_ridge puts the RLS readout back on the live path,
    and train_ridge after train_rls takes it again."""
    lsm = _real_lsm()
    inputs, targets = _sequences(3)
    probe = np.random.default_rng(4).standard_normal((6, RIDGE_INPUT_DIM))

    lsm.train_ridge(inputs, targets)
    lsm.train_rls(inputs, targets)
    assert lsm.active_readout == "rls"
    lsm.reservoir.reset()
    output, _ = lsm.process_input(probe)
    lsm.reservoir.reset()
    rls_expected = (
        lsm.reservoir.run(probe) @ lsm.rls_readout.Wout + lsm.rls_readout.bias
    )
    np.testing.assert_allclose(output, rls_expected, rtol=1e-10, atol=1e-12)

    lsm.train_ridge(inputs, targets)
    assert lsm.active_readout == "ridge"
    lsm.reservoir.reset()
    output, _ = lsm.process_input(probe)
    lsm.reservoir.reset()
    ridge_expected = (
        lsm.reservoir.run(probe) @ lsm.ridge_readout.Wout + lsm.ridge_readout.bias
    )
    np.testing.assert_allclose(output, ridge_expected, rtol=1e-10, atol=1e-12)
    assert not np.allclose(ridge_expected, rls_expected)
