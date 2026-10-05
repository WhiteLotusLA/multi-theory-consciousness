"""Membrane and STDP trace units of the production SNN.

``tau_mem``, ``tau_plus`` and ``tau_minus`` are in seconds; ``dt`` is the
duration of one simulation step in seconds. The per-step membrane decay is
``exp(-dt / tau_mem)`` (about 0.951 at the defaults) and the STDP trace decay
is ``exp(-dt / tau)``. v0.4.0 used ``exp(-1 / tau)``, which treats tau as a
step count: beta collapses to about 1.9e-22 and the network loses all
membrane memory between steps. ``dt = 1.0`` reproduces that value.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
snntorch = pytest.importorskip("snntorch")
if not hasattr(snntorch, "__version__"):
    pytest.skip("snntorch is replaced by a test double", allow_module_level=True)

from mtc.neural.spiking.production_snn import (  # noqa: E402
    ProductionSNN,
    SNNConfig,
    STDPLayer,
)

CONFIG_DERIVED_SUFFIXES = (
    "_beta",
    "_threshold",
    "_graded_spikes_factor",
    "_reset_mechanism_val",
)


def _config(**overrides) -> SNNConfig:
    """Small CPU config: 20 inputs, two hidden layers of 32, 10 outputs."""
    defaults = dict(
        input_dim=20,
        hidden_layers=[32, 32],
        output_dim=10,
        timesteps=10,
        use_gpu=False,
        stdp_enabled=False,
    )
    defaults.update(overrides)
    return SNNConfig(**defaults)


def _leaky_betas(model: ProductionSNN):
    return [float(m.beta) for m in model.network if isinstance(m, snntorch.Leaky)]


def _total_spikes(model: ProductionSNN, drive=None) -> int:
    if drive is None:
        drive = torch.ones(model.config.input_dim)
    model.train(False)
    with torch.no_grad():
        _, metrics = model.process_input(drive)
    return metrics["total_spikes"]


def _one_hot_drive(dim: int) -> "torch.Tensor":
    """A constant 1.0 on one channel: the per-step current through the first
    layer stays below the threshold of 1.0 (Linear init bounds weight and bias
    by 1/sqrt(fan_in)), so only a membrane that integrates across steps can
    reach it."""
    drive = torch.zeros(dim)
    drive[0] = 1.0
    return drive


def test_beta_uses_dt_over_tau():
    model = ProductionSNN(_config())
    betas = _leaky_betas(model)

    assert len(betas) == 3
    expected = math.exp(-1e-3 / 20e-3)
    for beta in betas:
        assert beta == pytest.approx(expected, rel=1e-6)
        assert beta == pytest.approx(0.951, abs=1e-3)


def test_default_snn_emits_spikes():
    torch.manual_seed(0)
    model = ProductionSNN(_config())
    assert _total_spikes(model) > 0

    # A sub-threshold constant drive separates the two betas: the default
    # membrane integrates it into spikes, the v0.4.0 membrane never does.
    torch.manual_seed(0)
    default = ProductionSNN(_config())
    torch.manual_seed(0)
    legacy = ProductionSNN(_config(dt=1.0))
    assert _total_spikes(default, _one_hot_drive(20)) > 0
    assert _total_spikes(legacy, _one_hot_drive(20)) == 0


def test_dt_one_reproduces_v04_beta():
    model = ProductionSNN(_config(dt=1.0))
    expected = float(np.exp(-1 / 20e-3))

    for beta in _leaky_betas(model):
        assert beta == pytest.approx(expected, rel=1e-6)
        assert beta < 1e-20


def test_stdp_trace_decay_uses_dt():
    cfg = _config(stdp_enabled=True)
    layer = STDPLayer(4, 3, cfg)
    layer.train()
    layer.pre_trace = torch.full((4,), 2.0)
    layer.post_trace = torch.full((3,), 0.5)

    layer(torch.zeros(1, 4), torch.zeros(1, 3))

    decay_plus = math.exp(-cfg.dt / cfg.tau_plus)
    decay_minus = math.exp(-cfg.dt / cfg.tau_minus)
    torch.testing.assert_close(layer.pre_trace, torch.full((4,), 2.0 * decay_plus))
    torch.testing.assert_close(layer.post_trace, torch.full((3,), 0.5 * decay_minus))
    assert decay_plus == pytest.approx(0.951, abs=1e-3)


def test_stdp_trace_decay_dt_one_matches_v04():
    cfg = _config(stdp_enabled=True, dt=1.0)
    layer = STDPLayer(2, 2, cfg)

    assert layer.pre_trace_decay == pytest.approx(float(np.exp(-1 / cfg.tau_plus)))
    assert layer.post_trace_decay == pytest.approx(float(np.exp(-1 / cfg.tau_minus)))


def test_restore_does_not_resilence_network():
    torch.manual_seed(1)
    source = ProductionSNN(_config())
    state = source.to_state_dict()

    corrupted = 0
    for key in list(state["arrays"]):
        if key.endswith(CONFIG_DERIVED_SUFFIXES):
            state["arrays"][key] = np.zeros_like(np.asarray(state["arrays"][key]))
            corrupted += 1
    assert corrupted == 12  # four config-derived buffers on each of three layers

    torch.manual_seed(1)
    restored = ProductionSNN(_config())
    restored.from_state_dict(state)

    for beta in _leaky_betas(restored):
        assert beta == pytest.approx(math.exp(-0.05), rel=1e-6)
    graded = [
        float(m.graded_spikes_factor)
        for m in restored.network
        if isinstance(m, snntorch.Leaky)
    ]
    assert graded == [1.0, 1.0, 1.0]
    assert _total_spikes(restored) > 0
