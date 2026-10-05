"""The MTC_DEVICE switch: resolution order, overrides and honest errors."""

import logging
import os
import sys
import types

import pytest

from mtc.core import device
from mtc.core.config import Settings
from mtc.core.device import DeviceUnavailableError


@pytest.fixture(autouse=True)
def reset_log_memory():
    device._LOGGED.clear()
    yield
    device._LOGGED.clear()


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_torch(mps: bool, cuda: bool, mps_built: bool = True):
    module = types.ModuleType("torch")
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps, is_built=lambda: mps_built)
    )
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    module.device = lambda name: ("torch.device", name)
    return module


def _install(monkeypatch, **flags):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(**flags))


def test_auto_prefers_mps_then_cuda_then_cpu(clean_env, monkeypatch):
    settings = Settings()
    _install(monkeypatch, mps=True, cuda=True)
    assert device.resolve_device_name(settings=settings) == "mps"
    _install(monkeypatch, mps=False, cuda=True)
    assert device.resolve_device_name(settings=settings) == "cuda"
    _install(monkeypatch, mps=False, cuda=False)
    assert device.resolve_device_name(settings=settings) == "cpu"
    _install(monkeypatch, mps=True, cuda=False, mps_built=False)
    assert device.resolve_device_name(settings=settings) == "cpu"


def test_cpu_setting_forces_cpu(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    _install(monkeypatch, mps=True, cuda=True)
    assert device.resolve_device_name(settings=Settings()) == "cpu"
    assert device.describe_device(Settings()) == {"name": "cpu", "source": "setting"}


def test_argument_overrides_setting(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    _install(monkeypatch, mps=True, cuda=False)
    settings = Settings()
    assert device.resolve_device_name("mps", settings=settings) == "mps"
    assert device.resolve_device_name("auto", settings=settings) == "mps"
    with pytest.raises(ValueError):
        device.resolve_device_name("tpu", settings=settings)


def test_unavailable_explicit_device_raises_naming_the_setting(clean_env, monkeypatch):
    _install(monkeypatch, mps=False, cuda=False)
    for choice in ("mps", "cuda"):
        monkeypatch.setenv("MTC_DEVICE", choice)
        with pytest.raises(DeviceUnavailableError) as info:
            device.resolve_device_name(settings=Settings())
        assert "MTC_DEVICE" in str(info.value)
        assert choice in str(info.value)


def test_resolver_without_torch(clean_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    settings = Settings()
    assert device.resolve_device_name(settings=settings) == "cpu"
    monkeypatch.setenv("MTC_DEVICE", "mps")
    with pytest.raises(DeviceUnavailableError) as info:
        device.resolve_device_name(settings=Settings())
    assert "torch is not installed" in str(info.value)
    with pytest.raises(DeviceUnavailableError):
        device.resolve_device("cpu", settings=settings)


def test_resolve_device_builds_a_torch_device(clean_env, monkeypatch):
    _install(monkeypatch, mps=False, cuda=True)
    assert device.resolve_device(settings=Settings()) == ("torch.device", "cuda")


def test_resolution_logged_once_with_source(clean_env, monkeypatch, caplog):
    _install(monkeypatch, mps=False, cuda=False)
    settings = Settings()
    with caplog.at_level(logging.INFO, logger="mtc.core.device"):
        device.resolve_device_name(settings=settings)
        device.resolve_device_name(settings=settings)
        device.resolve_device_name("cpu", settings=settings)
    lines = [r.getMessage() for r in caplog.records if r.name == "mtc.core.device"]
    assert len(lines) == 2
    assert "cpu" in lines[0] and "auto" in lines[0]
    assert "cpu" in lines[1] and "argument" in lines[1]
