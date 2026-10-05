"""Population synchrony (chi) and STDP statistics of the production SNN."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
snntorch = pytest.importorskip("snntorch")
if not hasattr(snntorch, "__version__"):
    pytest.skip("snntorch is replaced by a test double", allow_module_level=True)

from mtc.neural.spiking.production_snn import (  # noqa: E402
    ProductionSNN,
    SNNConfig,
    population_synchrony,
)

STDP_KEYS = {
    "plasticity_active",
    "stdp_enabled",
    "num_stdp_layers",
    "mean_weight_magnitude",
    "mean_weight_std",
    "weight_diversity",
}


def _config(**overrides) -> SNNConfig:
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


def test_synchrony_zero_when_silent():
    assert population_synchrony(torch.zeros(10, 8)) == 0.0
    # A single step has no temporal variance to measure
    assert population_synchrony(torch.ones(1, 8)) == 0.0
    # Every neuron firing on every step is constant, not synchronous
    assert population_synchrony(torch.ones(10, 8)) == 0.0


def test_synchrony_one_for_identical_trains():
    train = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    spikes = torch.stack([train] * 6, dim=1)  # [timesteps, neurons]

    assert population_synchrony(spikes) == pytest.approx(1.0)


def test_synchrony_in_unit_interval():
    generator = torch.Generator().manual_seed(0)
    independent = (torch.rand(400, 16, generator=generator) < 0.3).float()
    chi_independent = population_synchrony(independent)

    # Independent trains: chi near 1 / sqrt(neurons)
    assert 0.0 <= chi_independent < 0.5
    assert chi_independent == pytest.approx(1.0 / np.sqrt(16), abs=0.1)

    # Half the population shares a train: in between
    shared = (torch.rand(400, 1, generator=generator) < 0.3).float()
    mixed = torch.cat([shared.repeat(1, 8), independent[:, :8]], dim=1)
    chi_mixed = population_synchrony(mixed)
    assert chi_independent < chi_mixed < 1.0


def test_process_input_reports_synchrony():
    torch.manual_seed(0)
    model = ProductionSNN(_config())
    model.train(False)

    with torch.no_grad():
        output, state = model(torch.ones(1, 20))
        _, metrics = model.process_input(torch.ones(20))

    assert 0.0 <= state["synchrony"] <= 1.0
    assert isinstance(metrics["synchrony"], float)
    assert 0.0 <= metrics["synchrony"] <= 1.0
    assert metrics["total_spikes"] > 0


def test_process_emotion_is_alias():
    torch.manual_seed(3)
    first = ProductionSNN(_config())
    first.train(False)
    torch.manual_seed(3)
    second = ProductionSNN(_config())
    second.train(False)
    vector = torch.zeros(20)
    vector[[0, 4, 9]] = torch.tensor([0.65, 0.8, 0.3])

    torch.manual_seed(7)
    out_a, metrics_a = first.process_emotion(vector)
    torch.manual_seed(7)
    out_b, metrics_b = second.process_input(vector)

    torch.testing.assert_close(out_a, out_b)
    assert set(metrics_a) == set(metrics_b)
    for key in ("total_spikes", "mean_firing_rate", "synchrony", "layer_activities"):
        assert metrics_a[key] == metrics_b[key]


def test_stdp_statistics_keys_and_activity():
    disabled = ProductionSNN(_config(stdp_enabled=False)).get_stdp_statistics()
    assert set(disabled) == STDP_KEYS
    assert disabled["plasticity_active"] is False
    assert disabled["stdp_enabled"] is False
    assert disabled["num_stdp_layers"] == 0
    assert disabled["mean_weight_std"] is None

    model = ProductionSNN(_config(stdp_enabled=True, a_plus=0.05, a_minus=0.0))
    fresh = model.get_stdp_statistics()
    assert set(fresh) == STDP_KEYS
    assert fresh["stdp_enabled"] is True
    assert fresh["num_stdp_layers"] == 3
    # Enabled but no update applied yet: the weights are still their init
    assert fresh["plasticity_active"] is False
    assert fresh["mean_weight_std"] > 0.0

    layer = model.stdp_layers[0]
    layer.train()
    layer(torch.ones(2, 20), torch.ones(2, 32))

    active = model.get_stdp_statistics()
    assert active["plasticity_active"] is True
    assert active["mean_weight_magnitude"] > 0.0
    assert active["weight_diversity"] >= 0.0
