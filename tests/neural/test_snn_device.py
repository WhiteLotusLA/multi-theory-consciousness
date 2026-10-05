"""The production SNN takes its device from the MTC_DEVICE switch.

Every network here is tiny and explicitly or by setting on CPU: no test places a
tensor on an accelerator.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("snntorch")

from mtc.core import config as config_module  # noqa: E402
from mtc.core.config import Settings  # noqa: E402
from mtc.core.device import DeviceUnavailableError  # noqa: E402
from mtc.neural.spiking.production_snn import ProductionSNN, SNNConfig  # noqa: E402


def _config(**changes) -> SNNConfig:
    values = dict(
        input_dim=8,
        hidden_layers=[8],
        output_dim=4,
        timesteps=3,
        stdp_enabled=False,
    )
    values.update(changes)
    return SNNConfig(**values)


@pytest.fixture
def settings_device(monkeypatch):
    def install(name: str) -> None:
        monkeypatch.setattr(config_module, "_settings", Settings(DEVICE=name))

    return install


def test_config_device_defaults_to_none():
    assert SNNConfig().device is None


def test_snn_device_follows_setting(settings_device):
    settings_device("cpu")
    model = ProductionSNN(_config(use_gpu=True))
    assert model.device == torch.device("cpu")
    output, metrics = model.process_input(torch.rand(8))
    assert output.device.type == "cpu"
    assert metrics["total_spikes"] >= 0


def test_explicit_device_wins(settings_device):
    # the setting names a backend this machine cannot have, so the network only
    # builds when the explicit device in the config is what decides
    settings_device("cuda")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present, so the setting would not raise")
    with pytest.raises(DeviceUnavailableError):
        ProductionSNN(_config(use_gpu=True))
    model = ProductionSNN(_config(use_gpu=True, device="cpu"))
    assert model.device == torch.device("cpu")


def test_use_gpu_false_still_forces_cpu(settings_device):
    settings_device("cuda")  # never consulted: use_gpu=False means CPU
    model = ProductionSNN(_config(use_gpu=False))
    assert model.device == torch.device("cpu")
    forced = ProductionSNN(_config(use_gpu=False, device="mps"))
    assert forced.device == torch.device("cpu")


def test_unavailable_explicit_device_names_the_setting(settings_device):
    settings_device("cpu")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present")
    with pytest.raises(DeviceUnavailableError) as caught:
        ProductionSNN(_config(use_gpu=True, device="cuda"))
    assert "MTC_DEVICE" in str(caught.value)


def test_benchmark_and_cache_clear_go_through_the_device_manager(
    settings_device, monkeypatch
):
    import mtc.neural.mps_utils as mps_utils
    from mtc.neural.spiking import production_snn

    calls = []

    class Recorder:
        def synchronize(self):
            calls.append("synchronize")

        def empty_cache(self):
            calls.append("empty_cache")

    monkeypatch.setattr(mps_utils, "mps_manager", Recorder())
    settings_device("cpu")
    model = ProductionSNN(_config(use_gpu=True))
    model.benchmark_performance()
    assert calls == []  # a CPU network has nothing to synchronize
    production_snn._device_manager().empty_cache()
    assert calls == ["empty_cache"]


def test_module_names_no_accelerator_backend():
    import inspect

    from mtc.neural.spiking import production_snn

    source = inspect.getsource(production_snn)
    for needle in ("torch.mps", "torch.cuda", "backends.mps", 'torch.device("mps")'):
        assert needle not in source, needle
