"""Provider registry: registration, the active choice and its persisted record."""

import json
import logging
import os

import pytest

from mtc.brain.providers import registry as registry_module
from mtc.brain.providers.mock_provider import MockProvider
from mtc.brain.providers.registry import ProviderRegistry
from mtc.core.config import Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def settings(clean_env):
    return Settings(PROJECT_ROOT=str(clean_env), DATA_DIRECTORY=str(clean_env / "data"))


def test_set_active_persists_under_data_directory(settings, clean_env):
    registry = ProviderRegistry(settings=settings)
    registry.register(MockProvider())
    registry.register(MockProvider(provider_id="mock-two"))
    registry.set_active("mock-two")
    path = clean_env / "data" / "provider_state.json"
    assert path.exists()
    assert json.loads(path.read_text())["active_provider"] == "mock-two"
    assert registry.active_id == "mock-two"
    assert registry.active().model_info().id == "mock-two"
    with pytest.raises(KeyError):
        registry.set_active("not-registered")
    assert registry.active_id == "mock-two"


def test_state_path_from_settings_at_call_time(clean_env, monkeypatch):
    current = {"settings": Settings(PROJECT_ROOT=str(clean_env), DATA_DIRECTORY="a")}
    monkeypatch.setattr(registry_module, "get_settings", lambda: current["settings"])
    registry = ProviderRegistry()
    registry.register(MockProvider())
    assert registry.save() == clean_env / "a" / "provider_state.json"
    current["settings"] = Settings(PROJECT_ROOT=str(clean_env), DATA_DIRECTORY="b")
    assert registry.save() == clean_env / "b" / "provider_state.json"
    explicit = ProviderRegistry(state_path=clean_env / "fixed.json")
    assert explicit.save() == clean_env / "fixed.json"


def test_load_unknown_id_returns_none_and_warns(settings, clean_env, caplog):
    path = clean_env / "data" / "provider_state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"active_provider": "gone"}))
    registry = ProviderRegistry(settings=settings)
    registry.register(MockProvider())
    with caplog.at_level(logging.WARNING, logger=registry_module.__name__):
        assert registry.load() is None
    assert "gone" in caplog.text
    path.write_text(json.dumps({"active_provider": "mock"}))
    assert registry.load() == "mock"
    # Loading reports the record; it never changes the active provider.
    assert registry.active_id is None
    path.write_text("{broken")
    assert registry.load() is None
    path.unlink()
    assert registry.load() is None


def test_active_unset_raises(settings):
    registry = ProviderRegistry(settings=settings)
    with pytest.raises(RuntimeError, match="active"):
        registry.active()
    registry.register(MockProvider())
    with pytest.raises(RuntimeError):
        registry.active()
    assert registry.get("mock").model_info().id == "mock"
    with pytest.raises(KeyError):
        registry.get("other")


def test_save_creates_directory_lazily(settings, clean_env):
    registry = ProviderRegistry(settings=settings)
    registry.register(MockProvider())
    assert not (clean_env / "data").exists()
    registry.unregister("mock")
    assert not (clean_env / "data").exists()
    path = registry.save()
    assert path.exists() and path.parent == clean_env / "data"
    assert json.loads(path.read_text())["active_provider"] is None


def test_list_available_infos(settings):
    registry = ProviderRegistry(settings=settings)
    assert registry.list_available() == []
    registry.register(MockProvider())
    registry.register(MockProvider(provider_id="mock-two", reasoning=False))
    infos = registry.list_available()
    assert [info.id for info in infos] == ["mock", "mock-two"]
    assert infos[1].supports_thinking is False
    registry.set_active("mock")
    registry.unregister("mock")
    assert registry.active_id is None
    assert [info.id for info in registry.list_available()] == ["mock-two"]


def test_no_module_singleton():
    assert not hasattr(registry_module, "get_registry")
    assert not hasattr(registry_module, "_registry")


def test_unreadable_state_file_is_logged_by_name_only(tmp_path, caplog):
    state_dir = tmp_path / "private-home" / "data"
    state_dir.mkdir(parents=True)
    path = state_dir / "provider_state.json"
    path.write_text("{not json", encoding="utf-8")
    registry = ProviderRegistry(state_path=path)
    registry.register(MockProvider())
    with caplog.at_level(logging.WARNING, logger=registry_module.logger.name):
        assert registry.load() is None
    messages = [record.getMessage() for record in caplog.records]
    assert messages, "the unreadable file was not reported"
    text = "\n".join(messages)
    assert "provider_state.json" in text
    assert "PROVIDER_STATE_PATH" in text
    assert str(tmp_path) not in text
    assert "private-home" not in text
