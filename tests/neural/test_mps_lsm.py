"""Torch liquid state machine: seeded, normalized once, batch equals sequence."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mtc.core import device as device_module  # noqa: E402
from mtc.core.config import Settings  # noqa: E402
from mtc.neural.liquid import mps_lsm  # noqa: E402
from mtc.neural.liquid.mps_lsm import MPSLiquidStateMachine  # noqa: E402
from mtc.neural.mps_utils import MPSManager  # noqa: E402


@pytest.fixture
def cpu_manager(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    device_module._LOGGED.clear()
    return MPSManager(settings=Settings())


def _lsm(manager, seed, **kwargs):
    return MPSLiquidStateMachine(
        4,
        n_reservoir=kwargs.pop("n_reservoir", 60),
        n_output=3,
        seed=seed,
        manager=manager,
        **kwargs,
    )


def test_seeded_reservoir_is_reproducible(cpu_manager):
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    first, second = _lsm(cpu_manager, 7), _lsm(cpu_manager, 7)
    for name in ("W_in", "W_res", "W_out", "tau"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    other = _lsm(cpu_manager, 8)
    assert not torch.equal(first.W_res, other.W_res)
    _lsm(cpu_manager, None)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)  # the global generator is never touched


def test_spectral_radius_normalized_once(cpu_manager, monkeypatch):
    real = np.linalg.eigvals
    calls = []

    def counting(matrix):
        calls.append(matrix.shape)
        return real(matrix)

    monkeypatch.setattr(mps_lsm.np.linalg, "eigvals", counting)
    lsm = _lsm(cpu_manager, 11, spectral_radius=0.8)
    assert len(calls) == 1
    monkeypatch.setattr(mps_lsm.np.linalg, "eigvals", real)
    radius = float(np.max(np.abs(real(lsm.W_res.numpy().astype(np.float64)))))
    assert radius == pytest.approx(0.8, rel=1e-4)
    assert lsm.get_metrics()["spectral_radius"] == pytest.approx(0.8)


def test_batch_step_matches_sequential_steps(cpu_manager):
    inputs = np.random.default_rng(5).normal(size=(6, 4))
    batch_lsm, step_lsm = _lsm(cpu_manager, 3), _lsm(cpu_manager, 3)
    states, readouts = batch_lsm.batch_step(inputs)
    sequential = [step_lsm.step(row) for row in inputs]
    np.testing.assert_allclose(states, np.stack([s for s, _ in sequential]))
    np.testing.assert_allclose(readouts, np.stack([r for _, r in sequential]))
    assert states.shape == (6, 60) and readouts.shape == (6, 3)
    assert batch_lsm.get_metrics()["total_steps"] == 6
    assert batch_lsm.get_metrics()["gpu_steps"] == 0
    with pytest.raises(ValueError):
        batch_lsm.step(np.ones(3))


def test_state_and_readout_training(cpu_manager):
    lsm = _lsm(cpu_manager, 2)
    assert lsm.get_metrics()["readout_trained"] is False
    states, _ = lsm.batch_step(np.random.default_rng(1).normal(size=(20, 4)))
    targets = np.random.default_rng(2).normal(size=(20, 3))
    lsm.train_readout(states, targets)
    assert lsm.get_metrics()["readout_trained"] is True
    saved = lsm.get_state()
    lsm.reset_state()
    assert not lsm.get_state().any() and lsm.current_time == 0.0
    lsm.set_state(saved)
    np.testing.assert_allclose(lsm.get_state(), saved)


@pytest.mark.gpu
def test_runs_on_accelerator(monkeypatch, tmp_path):
    if torch.backends.mps.is_available():
        kind = "mps"
    elif torch.cuda.is_available():
        kind = "cuda"
    else:
        pytest.skip("needs MPS or CUDA")
    manager = MPSManager(device=kind)
    lsm = MPSLiquidStateMachine(4, n_reservoir=64, n_output=2, seed=1, manager=manager)
    assert lsm.device.type == kind and lsm.use_gpu is True
    state, readout = lsm.step(np.ones(4))
    assert np.all(np.isfinite(state)) and readout.shape == (2,)
    assert lsm.get_metrics()["gpu_steps"] == 1


def test_lsm_status_reports_device_and_failed_readout(cpu_manager, monkeypatch):
    lsm = _lsm(cpu_manager, 3)
    report = lsm.status()
    assert report["backend"] == "torch" and report["available"] is True
    assert report["device"] == "cpu" and report["last_error"] is None
    assert report["readout_trained"] is False

    def singular(*args, **kwargs):
        raise RuntimeError("singular matrix")

    monkeypatch.setattr(torch.linalg, "solve", singular)
    states = np.zeros((5, lsm.n_reservoir))
    assert lsm.train_readout(states, np.zeros((5, 3))) is False
    report = lsm.status()
    assert report["available"] is True and report["readout_trained"] is False
    assert "readout training failed" in report["last_error"]
