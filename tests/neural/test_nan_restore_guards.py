"""Restore guards on the SNN and oscillatory binding.

A non-finite array is never a legitimate learned state: the restore discards
it with a warning and keeps the module's fresh finite initialization, so the
next save writes clean state. Shape mismatches are skipped. The SNN never
restores the Leaky buffers that its config derives (beta, threshold,
graded_spikes_factor, reset_mechanism_val).
"""

import logging

import numpy as np
import pytest

from mtc.neural.oscillatory_binding import OSCILLATOR_NAMES, OscillatoryBinding


def _snn_modules():
    torch = pytest.importorskip("torch")
    snntorch = pytest.importorskip("snntorch")
    if not hasattr(snntorch, "__version__"):
        pytest.skip("snntorch is replaced by a test double")
    from mtc.neural.spiking.production_snn import ProductionSNN, SNNConfig

    return torch, snntorch, ProductionSNN, SNNConfig


def _small_snn(ProductionSNN, SNNConfig, **overrides):
    defaults = dict(
        input_dim=12,
        hidden_layers=[16],
        output_dim=6,
        timesteps=6,
        use_gpu=False,
        stdp_enabled=True,
    )
    defaults.update(overrides)
    return ProductionSNN(SNNConfig(**defaults))


def test_snn_restore_skips_nonfinite_and_shape_mismatch(caplog):
    torch, _, ProductionSNN, SNNConfig = _snn_modules()
    torch.manual_seed(0)
    source = _small_snn(ProductionSNN, SNNConfig)
    state = source.to_state_dict()

    params = [k for k in state["arrays"] if not k.startswith("snn_buf_")]
    buffers = [k for k in state["arrays"] if k.startswith("snn_buf_")]
    assert "snn_network_0_weight" in params
    assert "snn_buf_stdp_layers_0_pre_trace" in buffers

    poisoned = np.array(state["arrays"]["snn_network_0_weight"], dtype=np.float32)
    poisoned[0, 0] = np.nan
    state["arrays"]["snn_network_0_weight"] = poisoned
    state["arrays"]["snn_network_0_bias"] = np.zeros(99, dtype=np.float32)
    trace = np.array(state["arrays"]["snn_buf_stdp_layers_0_pre_trace"])
    trace[:] = np.inf
    state["arrays"]["snn_buf_stdp_layers_0_pre_trace"] = trace
    # A finite, shape-matched parameter restores normally
    state["arrays"]["snn_network_3_bias"] = np.full(6, 0.25, dtype=np.float32)

    torch.manual_seed(1)
    target = _small_snn(ProductionSNN, SNNConfig)
    fresh_weight = target.network[0].weight.detach().clone()
    fresh_bias = target.network[0].bias.detach().clone()
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    torch.testing.assert_close(target.network[0].weight.detach(), fresh_weight)
    torch.testing.assert_close(target.network[0].bias.detach(), fresh_bias)
    assert torch.all(torch.isfinite(target.stdp_layers[0].pre_trace))
    torch.testing.assert_close(target.network[3].bias.detach(), torch.full((6,), 0.25))
    for _, param in target.named_parameters():
        assert torch.all(torch.isfinite(param))
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "non-finite" in messages
    assert "shape" in messages


def test_snn_restore_never_restores_leaky_config_buffers():
    torch, snntorch, ProductionSNN, SNNConfig = _snn_modules()
    torch.manual_seed(2)
    source = _small_snn(ProductionSNN, SNNConfig, stdp_enabled=False)
    state = source.to_state_dict()
    # Zero every config-derived buffer, as a stale file on disk would hold
    # them (a reset_mechanism_val of 0 selects "subtract"; construction chose
    # "zero")
    for key in list(state["arrays"]):
        if key.endswith(
            ("_beta", "_threshold", "_graded_spikes_factor", "_reset_mechanism_val")
        ):
            state["arrays"][key] = np.zeros_like(np.asarray(state["arrays"][key]))

    torch.manual_seed(2)
    target = _small_snn(ProductionSNN, SNNConfig, stdp_enabled=False)
    expected = {
        name: buf.detach().clone()
        for name, buf in target.named_buffers()
        if name.endswith(
            ("beta", "threshold", "graded_spikes_factor", "reset_mechanism_val")
        )
    }
    assert len(expected) == 8  # four buffers on each of two Leaky layers

    target.from_state_dict(state)

    live = dict(target.named_buffers())
    for name, value in expected.items():
        torch.testing.assert_close(live[name], value)
    graded = [
        float(m.graded_spikes_factor)
        for m in target.network
        if isinstance(m, snntorch.Leaky)
    ]
    assert graded == [1.0, 1.0]

    target.train(False)
    with torch.no_grad():
        _, metrics = target.process_input(torch.ones(12))
    assert metrics["total_spikes"] > 0


def test_snn_round_trip_restores_learned_weights_and_update_counts():
    torch, _, ProductionSNN, SNNConfig = _snn_modules()
    torch.manual_seed(4)
    source = _small_snn(ProductionSNN, SNNConfig, a_plus=0.05, a_minus=0.0)
    layer = source.stdp_layers[0]
    layer.train()
    layer(torch.ones(2, 12), torch.ones(2, 16))
    assert source.get_stdp_statistics()["plasticity_active"] is True

    torch.manual_seed(5)
    target = _small_snn(ProductionSNN, SNNConfig, a_plus=0.05, a_minus=0.0)
    assert target.get_stdp_statistics()["plasticity_active"] is False
    target.from_state_dict(source.to_state_dict())

    torch.testing.assert_close(target.stdp_layers[0].weight, layer.weight)
    torch.testing.assert_close(target.network[0].weight, source.network[0].weight)
    assert target.get_stdp_statistics()["plasticity_active"] is True


def test_oscillatory_restore_rejects_nan_phases(caplog):
    source = OscillatoryBinding()
    source.bind({name: 0.5 for name in OSCILLATOR_NAMES})
    state = source.to_state_dict()
    phases = np.array(state["arrays"]["phases"])
    phases[2, :] = np.nan
    state["arrays"]["phases"] = phases

    target = OscillatoryBinding()
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert np.all(np.isfinite(target.phases))
    np.testing.assert_allclose(np.linalg.norm(target.phases, axis=1), 1.0, atol=1e-12)
    assert any("non-finite" in r.getMessage() for r in caplog.records)
    result = target.bind({name: 0.5 for name in OSCILLATOR_NAMES})
    assert 0.0 <= result.global_order_parameter <= 1.0


def test_oscillatory_restore_discards_nonfinite_coupling(caplog):
    source = OscillatoryBinding()
    for _ in range(3):
        source.bind({name: 0.8 for name in OSCILLATOR_NAMES})
    state = source.to_state_dict()
    coupling = np.array(state["arrays"]["J_learned"])
    coupling[0, 1, 0, 0] = np.inf
    coupling[4] = np.nan
    state["arrays"]["J_learned"] = coupling

    target = OscillatoryBinding()
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert np.all(np.isfinite(target.J_learned))
    np.testing.assert_array_equal(target.J_learned, np.zeros_like(coupling))
    assert any("J_learned" in r.getMessage() for r in caplog.records)


def test_oscillatory_finite_coupling_restores_normally():
    source = OscillatoryBinding()
    source.J_learned = np.full_like(source.J_learned, 0.25)

    target = OscillatoryBinding()
    target.from_state_dict(source.to_state_dict())

    np.testing.assert_allclose(target.J_learned, 0.25)
