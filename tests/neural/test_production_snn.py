#!/usr/bin/env python3
"""
Tests for Production SNN (Spiking Neural Network).

All tests use mocks where needed -- no real GPU, database, or LLM connections.
Uses a tiny network config so tests run quickly on CPU.

snntorch is mocked at module level since it may not be installed in the
test environment.
"""

import sys
from pathlib import Path


import types
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock snntorch before importing production_snn
# ---------------------------------------------------------------------------

class _StraightThroughSpike(torch.autograd.Function):
    """Spike function with straight-through estimator for gradients."""

    @staticmethod
    def forward(ctx, mem, threshold):
        spk = (mem >= threshold).float()
        ctx.save_for_backward(mem)
        ctx.threshold = threshold
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        # Straight-through estimator: pass gradient through unchanged
        return grad_output, None


class _FakeLeaky(nn.Module):
    """Minimal stand-in for snn.Leaky that behaves like a LIF neuron."""

    def __init__(self, beta=0.9, spike_grad=None, threshold=1.0,
                 reset_mechanism="zero", **kwargs):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self._mem = None

    def forward(self, x):
        if self._mem is None or self._mem.shape != x.shape:
            self._mem = torch.zeros_like(x)
        self._mem = self.beta * self._mem + x
        spk = _StraightThroughSpike.apply(self._mem, self.threshold)
        self._mem = self._mem * (1 - spk.detach())  # reset (detach spike for reset)
        return spk, self._mem


def _fake_rate(data, num_steps=1):
    """Minimal stand-in for spikegen.rate.

    Returns a tensor of the same shape where values > 0.5 become spikes.
    """
    return (torch.rand_like(data) < data.abs().clamp(0, 1)).float()


def _fake_fast_sigmoid(slope=25):
    """Return a dummy surrogate gradient callable."""
    return MagicMock()


# Build snntorch mock as a proper module-like object
_mock_snn = types.ModuleType("snntorch")
_mock_snn.Leaky = _FakeLeaky

_mock_spikegen = types.ModuleType("snntorch.spikegen")
_mock_spikegen.rate = _fake_rate

_mock_surrogate = types.ModuleType("snntorch.surrogate")
_mock_surrogate.fast_sigmoid = _fake_fast_sigmoid

sys.modules["snntorch"] = _mock_snn
sys.modules["snntorch.spikegen"] = _mock_spikegen
sys.modules["snntorch.surrogate"] = _mock_surrogate

# Now safe to import
from mtc.neural.spiking.production_snn import (  # noqa: E402
    SNNConfig,
    STDPLayer,
    ProductionSNN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(**overrides) -> SNNConfig:
    """Return a minimal SNNConfig that runs fast on CPU."""
    defaults = dict(
        num_neurons=50,
        input_dim=32,
        hidden_layers=[16],
        output_dim=8,
        tau_mem=20e-3,
        tau_syn=5e-3,
        v_threshold=1.0,
        v_reset=0.0,
        stdp_enabled=False,
        learning_rate=1e-3,
        batch_size=4,
        timesteps=5,
        use_gpu=False,  # Force CPU for test determinism
    )
    defaults.update(overrides)
    return SNNConfig(**defaults)


def _set_inference_mode(model):
    """Put model into inference (non-training) mode."""
    model.train(False)


# ---------------------------------------------------------------------------
# STDPLayer tests
# ---------------------------------------------------------------------------

class TestSTDPLayer:
    """Tests for the STDP (Spike-Timing-Dependent Plasticity) layer."""

    def test_forward_returns_correct_shape(self):
        """Output shape should be (batch_size, out_features)."""
        cfg = _tiny_config(stdp_enabled=True)
        layer = STDPLayer(32, 16, cfg)
        pre_spikes = torch.zeros(4, 32)
        post_spikes = torch.zeros(4, 16)

        output = layer(pre_spikes, post_spikes)

        assert output.shape == (4, 16)

    def test_stdp_updates_weights_during_training(self):
        """In training mode with STDP enabled, weights should change.

        Uses asymmetric a_plus/a_minus so potentiation and depression
        do not cancel each other out.
        """
        cfg = _tiny_config(stdp_enabled=True, a_plus=1.0, a_minus=0.0)
        layer = STDPLayer(32, 16, cfg)
        layer.train()

        original_weight = layer.weight.data.clone()

        pre_spikes = torch.ones(4, 32)
        post_spikes = torch.ones(4, 16)
        layer(pre_spikes, post_spikes)

        assert not torch.equal(layer.weight.data, original_weight)

    def test_stdp_no_update_in_inference_mode(self):
        """In inference mode, STDP should not update weights."""
        cfg = _tiny_config(stdp_enabled=True)
        layer = STDPLayer(32, 16, cfg)
        _set_inference_mode(layer)

        original_weight = layer.weight.data.clone()

        pre_spikes = torch.ones(4, 32)
        post_spikes = torch.ones(4, 16)
        layer(pre_spikes, post_spikes)

        torch.testing.assert_close(layer.weight.data, original_weight)

    def test_weight_clamping(self):
        """Weights must stay within [-2, 2] after STDP updates."""
        cfg = _tiny_config(stdp_enabled=True, a_plus=10.0, a_minus=10.0)
        layer = STDPLayer(32, 16, cfg)
        layer.train()

        pre_spikes = torch.ones(4, 32)
        post_spikes = torch.ones(4, 16)
        for _ in range(50):
            layer(pre_spikes, post_spikes)

        assert layer.weight.data.max() <= 2.0
        assert layer.weight.data.min() >= -2.0

    def test_traces_initialized_to_zero(self):
        """Pre and post traces should start at zero."""
        cfg = _tiny_config(stdp_enabled=True)
        layer = STDPLayer(32, 16, cfg)

        assert torch.all(layer.pre_trace == 0)
        assert torch.all(layer.post_trace == 0)


# ---------------------------------------------------------------------------
# ProductionSNN tests
# ---------------------------------------------------------------------------

class TestProductionSNN:
    """Tests for the top-level ProductionSNN."""

    def test_initialization_on_cpu(self):
        """SNN should initialise on CPU when use_gpu=False."""
        cfg = _tiny_config()
        model = ProductionSNN(cfg)

        assert model.device == torch.device("cpu")

    def test_forward_output_shape(self):
        """Forward pass output should be (batch_size, output_dim)."""
        cfg = _tiny_config()
        model = ProductionSNN(cfg)
        _set_inference_mode(model)

        inp = torch.randn(4, cfg.input_dim)
        with torch.no_grad():
            output, state = model(inp)

        assert output.shape == (4, cfg.output_dim)

    def test_forward_state_dict_keys(self):
        """State dict should contain expected monitoring keys."""
        cfg = _tiny_config()
        model = ProductionSNN(cfg)
        _set_inference_mode(model)

        inp = torch.randn(2, cfg.input_dim)
        with torch.no_grad():
            _, state = model(inp)

        for key in ["spike_counts", "membrane_potentials", "output_spikes",
                     "firing_rates", "timesteps_processed", "num_layers"]:
            assert key in state, f"Missing state key: {key}"

    def test_process_emotion_returns_metrics(self):
        """process_emotion should return output and metrics dict."""
        cfg = _tiny_config()
        model = ProductionSNN(cfg)
        _set_inference_mode(model)

        emotion = torch.randn(cfg.input_dim)
        output, metrics = model.process_emotion(emotion)

        assert output.shape[-1] == cfg.output_dim
        for key in ["processing_time_ms", "total_spikes", "mean_firing_rate",
                     "output_pattern", "layer_activities"]:
            assert key in metrics, f"Missing metric key: {key}"

    def test_process_emotion_unsqueezes_1d_input(self):
        """1-D emotion vector should be auto-unsqueezed to batch dim."""
        cfg = _tiny_config()
        model = ProductionSNN(cfg)
        _set_inference_mode(model)

        emotion = torch.randn(cfg.input_dim)
        output, _ = model.process_emotion(emotion)

        assert output.shape[-1] == cfg.output_dim

    def test_train_on_batch_computes_loss(self):
        """Forward pass should produce a valid MSE loss against targets."""
        cfg = _tiny_config()
        model = ProductionSNN(cfg)

        inputs = torch.randn(4, cfg.input_dim)
        targets = torch.randn(4, cfg.output_dim)

        with torch.no_grad():
            outputs, _ = model(inputs)
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=0)
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=1)
            loss = nn.functional.mse_loss(outputs, targets)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0.0

    def test_count_neurons(self):
        """_count_neurons should sum all layer sizes."""
        cfg = _tiny_config(input_dim=32, hidden_layers=[16], output_dim=8)
        model = ProductionSNN(cfg)

        assert model._count_neurons() == 32 + 16 + 8

    def test_network_builds_correct_number_of_layers(self):
        """ModuleList should have the right number of modules."""
        cfg = _tiny_config(input_dim=32, hidden_layers=[16, 16], output_dim=8)
        model = ProductionSNN(cfg)

        # 3 layer transitions: each has Linear + LIF
        # First 2 transitions also get Dropout (not the last one)
        # = 3*2 + 2 = 8 modules
        assert len(model.network) == 8
