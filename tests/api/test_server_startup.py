"""Server lifecycle and the entry point: the lifespan builds, starts and stops
the runtime; ``main`` serves one process on loopback by default and refuses
an unauthenticated non-loopback bind. No port is ever bound here."""

import secrets
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.routing import APIWebSocketRoute
from fastapi.testclient import TestClient

from mtc.api import server as server_module
from mtc.api.auth import API_KEY_HEADER
from mtc.api.schemas import MAX_FRAME_BYTES
from mtc.api.server import create_app, main
from mtc.brain.brain import LLMBrain
from mtc.brain.providers.mock_provider import MockProvider
from mtc.brain.providers.registry import ProviderRegistry
from mtc.core.embeddings import HashEmbedder
from mtc.pipeline.runtime import build_runtime
from tests.api.test_auth import FakeRuntime, api_settings, isolated_env


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


pytestmark = pytest.mark.timeout(600)


@pytest.fixture
def key():
    return secrets.token_hex(24)


def test_lifespan_builds_runtime_from_settings_in_memory(clean_env, key):
    cfg = api_settings(clean_env, API_KEY=key, METRICS_UPDATE_INTERVAL=0)
    built: List[Any] = []

    async def builder(settings):
        runtime = await build_runtime(
            settings,
            embedder=HashEmbedder(),
            allow_non_semantic=True,
            substrates=False,
        )
        built.append(runtime)
        return runtime

    app = create_app(settings=cfg, runtime_builder=builder)
    assert app.state.runtime is None
    with TestClient(app) as client:
        runtime = app.state.runtime
        assert built == [runtime]
        assert runtime.settings is cfg
        health = client.get("/health").json()
        assert health["ready"] is True and health["status"] == "ok"
        status = client.get(
            "/api/consciousness/status", headers={API_KEY_HEADER: key}
        ).json()
        stores = status["stores"]
        assert stores, "the runtime reports its stores"
        backends = repr(stores).lower()
        for client_name in ("postgres_pool", "mongo_client", "redis_client"):
            assert client_name not in backends
        models = client.get("/api/models", headers={API_KEY_HEADER: key}).json()
        assert models["active"] == "mock"
    assert app.state.started is False


def test_setup_logging_called_once(clean_env, key, monkeypatch):
    calls: List[Any] = []
    monkeypatch.setattr(server_module, "setup_logging", lambda s: calls.append(s))
    cfg = api_settings(clean_env, API_KEY=key, METRICS_UPDATE_INTERVAL=0)
    runtime = FakeRuntime(cfg)
    with TestClient(create_app(settings=cfg, runtime=runtime)) as client:
        client.get("/health")
        client.get("/api/models", headers={API_KEY_HEADER: key})
    assert calls == [cfg]


def _unhealthy_brain(cfg) -> LLMBrain:
    """A brain whose only provider fails its health check."""
    registry = ProviderRegistry(
        state_path=Path(cfg.DATA_DIRECTORY) / "provider_state.json", settings=cfg
    )
    registry.register(MockProvider(healthy=False))
    registry.set_active("mock")
    return LLMBrain(registry=registry, settings=cfg)


def test_unreachable_brain_starts_degraded(clean_env, key):
    cfg = api_settings(clean_env, API_KEY=key, METRICS_UPDATE_INTERVAL=0)

    async def builder(settings):
        return await build_runtime(
            settings,
            brain=_unhealthy_brain(settings),
            embedder=HashEmbedder(),
            allow_non_semantic=True,
            substrates=False,
        )

    app = create_app(settings=cfg, runtime_builder=builder)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health == {**health, "status": "degraded", "ready": True}
        response = client.post(
            "/api/chat", json={"message": "hello"}, headers={API_KEY_HEADER: key}
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == "brain_unavailable"
        assert body["response"] is None and body["success"] is False


def test_shutdown_stops_runtime(clean_env, key):
    cfg = api_settings(clean_env, API_KEY=key, METRICS_UPDATE_INTERVAL=0)
    runtime = FakeRuntime(cfg)
    app = create_app(settings=cfg, runtime=runtime)
    with TestClient(app) as client:
        assert runtime.started == 1 and runtime.stopped == 0
        assert client.get("/health").json()["ready"] is True
    assert runtime.stopped == 1
    assert app.state.started is False


class _FakeUvicorn:
    """Records the server configuration; ``run`` binds nothing."""

    def __init__(self) -> None:
        self.configs: List[Dict[str, Any]] = []
        self.runs = 0
        owner = self

        class Config:
            def __init__(self, app, **kwargs):
                self.app = app
                self.kwargs = kwargs
                owner.configs.append({"app": app, **kwargs})

        class Server:
            def __init__(self, config):
                self.config = config

            def run(self):
                owner.runs += 1

        self.module = types.SimpleNamespace(Config=Config, Server=Server)


def test_main_single_process_and_loopback_default(clean_env, key, monkeypatch):
    fake = _FakeUvicorn()
    monkeypatch.setitem(sys.modules, "uvicorn", fake.module)
    cfg = api_settings(
        clean_env, API_KEY=key, WEBSOCKET_PING_INTERVAL=25, WEBSOCKET_PING_TIMEOUT=7
    )
    monkeypatch.setattr(server_module, "get_settings", lambda: cfg)
    assert main([]) == 0
    assert fake.runs == 1 and len(fake.configs) == 1
    config = fake.configs[0]
    assert config["host"] == "127.0.0.1" == cfg.API_HOST
    assert config["port"] == cfg.API_PORT
    assert config["workers"] == 1
    assert config["access_log"] is False
    assert config["ws_ping_interval"] == 25.0
    assert config["ws_ping_timeout"] == 7.0
    assert config["ws_max_size"] == MAX_FRAME_BYTES
    assert config["app"].state.settings is cfg


def test_main_refuses_unsafe_bind(clean_env, monkeypatch, capsys):
    fake = _FakeUvicorn()
    monkeypatch.setitem(sys.modules, "uvicorn", fake.module)
    cfg = api_settings(clean_env, API_AUTH_ENABLED=False, API_HOST="host.invalid")
    monkeypatch.setattr(server_module, "get_settings", lambda: cfg)
    assert main([]) == 2
    assert fake.configs == [] and fake.runs == 0
    err = capsys.readouterr().err
    assert "API_HOST" in err and "API_AUTH_ENABLED" in err


def test_ws_route_absent_when_disabled(clean_env, key):
    on = create_app(settings=api_settings(clean_env, API_KEY=key))
    off = create_app(
        settings=api_settings(clean_env, API_KEY=key, WEBSOCKET_ENABLED=False)
    )
    paths_on = {r.path for r in on.routes if isinstance(r, APIWebSocketRoute)}
    paths_off = {r.path for r in off.routes if isinstance(r, APIWebSocketRoute)}
    assert paths_on == {"/ws"}
    assert paths_off == set()
