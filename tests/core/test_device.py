"""The MTC_DEVICE switch: resolution order, overrides and honest errors."""

import importlib.util
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from mtc.core import device
from mtc.core.config import Settings
from mtc.core.device import DeviceUnavailableError

ROOT = Path(__file__).resolve().parents[2]


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


def test_cpu_choice_never_asks_for_torch(clean_env, monkeypatch):
    def refuse():
        raise AssertionError("the cpu choice asked for torch")

    monkeypatch.setattr(device, "_import_torch", refuse)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    settings = Settings()
    assert device.resolve_device_name(settings=settings) == "cpu"
    assert device.resolve_device_name("cpu", settings=settings) == "cpu"
    assert device.describe_device(settings) == {"name": "cpu", "source": "setting"}


CPU_CHILD = """
import sys
import mtc.core.device as device
before = "torch" in sys.modules
name = device.resolve_device_name()
described = device.describe_device()
after = "torch" in sys.modules
print(f"{before} {name} {described['name']} {after}")
"""


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed, so nothing could load it",
)
def test_cpu_never_imports_torch(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MTC_") and key != "ANTHROPIC_API_KEY"
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_API_PORT": "0",
            "MTC_POSTGRES_ENABLED": "false",
            "MTC_MONGODB_ENABLED": "false",
            "MTC_REDIS_ENABLED": "false",
            "MTC_QDRANT_ENABLED": "false",
            "MTC_NEO4J_ENABLED": "false",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", CPU_CHILD],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    before, name, described, after = result.stdout.split()
    assert (name, described) == ("cpu", "cpu")
    assert before == "False", "importing mtc.core.device loaded torch"
    assert after == "False", "resolving the cpu device loaded torch"


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
