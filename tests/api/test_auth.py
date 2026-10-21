"""API key auth: one header carrier, fail-closed, compared in constant time,
never logged; an unauthenticated server binds to loopback only.

``FakeRuntime`` and ``api_settings`` here are shared by the other API test
modules that need an application without building the whole runtime.
"""

import logging
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from mtc.api import auth as auth_module
from mtc.api.auth import (
    API_KEY_HEADER,
    UnsafeBindError,
    assert_bind_allowed,
    require_api_key,
)
from mtc.api.server import create_app
from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.core.logger import reset_logging
from mtc.pipeline.context_types import TurnResult

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def api_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings for an API test: every path under ``tmp_path``, no .env file."""
    values: Dict[str, Any] = {
        "PROJECT_ROOT": str(tmp_path),
        "DATA_DIRECTORY": str(tmp_path / "data"),
        "LOG_DIRECTORY": str(tmp_path / "logs"),
        "STATE_DIR": str(tmp_path / "state"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "AGENT_SPACE_DIR": str(tmp_path / "agent_space"),
        "SPEAKERS_PATH": str(tmp_path / "speakers.yaml"),
        "LLM_PROVIDER": "mock",
        "DEVICE": "cpu",
        "SLEEP_SCHEDULER_ENABLED": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def isolated_env(monkeypatch, tmp_path):
    """No MTC_ variable and a temporary working directory; the logging an
    application lifespan configured is reset afterwards, so later tests see
    the framework's default logger state. Each API test module wraps this
    generator in its own ``clean_env`` fixture."""
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    # The accelerator may be shared with a live system: stay on the CPU.
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    # A component that reads the process Settings during the test builds its
    # own from this environment; the earlier singleton is restored afterwards.
    monkeypatch.setattr(config_module, "_settings", None)
    yield tmp_path
    reset_logging()


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


class FakeBrain:
    available = True

    def status(self) -> Dict[str, Any]:
        return {
            "provider": {"id": "mock", "name": "mock provider"},
            "active_provider": "mock",
            "providers": ["local", "mock"],
            "available": True,
            "health": {"ok": True, "detail": "mock provider"},
        }


class FakePipeline:
    """Records every turn; replies with a fixed labelled text."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def send_message(
        self,
        message: str,
        speaker: Optional[str] = None,
        speaker_token: Optional[str] = None,
        *,
        stream_callback=None,
        request_id=None,
        timeout_s=None,
        on_generated=None,
    ) -> TurnResult:
        self.calls.append(
            {"message": message, "speaker": speaker, "speaker_token": speaker_token}
        )
        if stream_callback is not None:
            await stream_callback("[fake] reply")
        if on_generated is not None:
            on_generated()
        return TurnResult(
            success=True,
            response="[fake] reply",
            thinking=None,
            conversation_id=None,
            message_count=2,
            timestamp=T0,
            cycle_ok=True,
            speaker_label="",
            provider_id="fake",
        )

    def status(self) -> Dict[str, Any]:
        return {"turns": len(self.calls)}


class FakeRuntime:
    """The runtime surface the API reads, without building any component."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.brain = FakeBrain()
        self.pipeline = FakePipeline()
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    def status(self) -> Dict[str, Any]:
        return {"started": self.started > 0}


def make_app(settings: Settings, runtime: Any = None):
    return create_app(settings=settings, runtime=runtime or FakeRuntime(settings))


# ---------------------------------------------------------------------------
# HTTP carrier
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return secrets.token_hex(24)


def test_missing_key_401(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    response = client.get("/api/models")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_wrong_key_401(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    response = client.get(
        "/api/models", headers={API_KEY_HEADER: secrets.token_hex(24)}
    )
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    # A key that differs only in its last character is still refused.
    near = key[:-1] + ("0" if key[-1] != "0" else "1")
    assert client.get("/api/models", headers={API_KEY_HEADER: near}).status_code == 401


def test_right_key_200(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    response = client.get("/api/models", headers={API_KEY_HEADER: key})
    assert response.status_code == 200


def test_empty_configured_key_503_fail_closed(clean_env):
    app = make_app(api_settings(clean_env, API_KEY="", API_AUTH_ENABLED=True))
    client = TestClient(app)
    for headers in ({}, {API_KEY_HEADER: ""}, {API_KEY_HEADER: secrets.token_hex(8)}):
        response = client.get("/api/models", headers=headers)
        assert response.status_code == 503, headers
    for route in _http_routes(app):
        if route.path == "/health":
            continue
        for method in sorted(route.methods):
            response = client.request(method, route.path, headers={API_KEY_HEADER: ""})
            assert response.status_code == 503, (method, route.path)


def test_bearer_header_not_accepted(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    response = client.get("/api/models", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 401
    # A query parameter is not a carrier either.
    assert client.get(f"/api/models?api_key={key}").status_code == 401


def test_health_needs_no_key(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"status", "ready", "version"}
    # Health also answers when the configured key is empty.
    empty = TestClient(make_app(api_settings(clean_env, API_KEY="")))
    assert empty.get("/health").status_code == 200


def _http_routes(app) -> List[APIRoute]:
    return [route for route in app.routes if isinstance(route, APIRoute)]


def test_every_route_except_health_requires_key(clean_env, key):
    app = make_app(api_settings(clean_env, API_KEY=key))
    client = TestClient(app)
    routes = _http_routes(app)
    paths = {route.path for route in routes}
    assert "/health" in paths and "/api/chat" in paths and len(paths) >= 12
    # No documentation route is served (each would be an unauthenticated route).
    assert not paths & {"/docs", "/redoc", "/openapi.json"}
    for route in routes:
        calls = {dep.call for dep in route.dependant.dependencies}
        if route.path == "/health":
            assert require_api_key not in calls
            continue
        assert require_api_key in calls, route.path
        for method in sorted(route.methods):
            response = client.request(method, route.path)
            assert response.status_code == 401, (method, route.path)


def _messages(records) -> str:
    parts = []
    for record in records:
        parts.append(record.getMessage())
        parts.append(repr(record.args))
    return "\n".join(parts)


def _pieces(secret: str, size: int = 4) -> List[str]:
    return [secret[i : i + size] for i in range(len(secret) - size + 1)]


# A record's timestamp as the JSON and the text formats write it (timezone
# aware UTC, ISO 8601, microseconds).
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)


def _without_timestamps(text: str) -> str:
    """The log text with every record timestamp replaced by a placeholder.

    A timestamp changes between the baseline scenario and the one checked, so
    the baseline cannot exclude its digits, and a four-digit piece of a random
    hex key coincides with the seconds and microseconds about three times in a
    hundred runs (measured 2026-10-21: eight hits in 300 scenario runs, every
    one inside a ``timestamp`` field, no other hit; the full gate had failed
    on the piece 7791 inside ``22:20:01.779152``). A key cannot leak through a
    timestamp, so the check reads every record without it."""
    return _TIMESTAMP.sub("<timestamp>", text)


def _key_scenario(tmp_path: Path, configured: str, wrong: str) -> str:
    """Run requests with the right, a wrong and no key; return all log text."""
    collected: List[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    handler = Collector(level=logging.DEBUG)
    targets = [logging.getLogger(), logging.getLogger("mtc")]
    previous = [target.level for target in targets]
    for target in targets:
        target.addHandler(handler)
        target.setLevel(logging.DEBUG)
    try:
        cfg = api_settings(tmp_path, API_KEY=configured, LOG_LEVEL="DEBUG")
        with TestClient(make_app(cfg)) as client:
            client.get("/api/models", headers={API_KEY_HEADER: configured})
            client.get("/api/models", headers={API_KEY_HEADER: wrong})
            client.get("/api/models")
            client.post("/api/chat", json={"message": "hi"})
            client.post(
                "/api/chat",
                json={"message": "hi"},
                headers={API_KEY_HEADER: configured},
            )
    finally:
        for target, level in zip(targets, previous):
            target.removeHandler(handler)
            target.setLevel(level)
        reset_logging()
    log_files = sorted((tmp_path / "logs").glob("*.log"))
    file_text = "\n".join(path.read_text(encoding="utf-8") for path in log_files)
    return _without_timestamps(_messages(collected) + "\n" + file_text)


def test_key_never_logged(clean_env):
    baseline = _key_scenario(
        clean_env / "first", secrets.token_hex(24), secrets.token_hex(24)
    )
    assert baseline.strip(), "the scenario logs something to check"
    # Draw keys none of whose pieces occur in the baseline text, so any piece
    # found below can only have come from the key itself.
    for _ in range(100):
        configured, wrong = secrets.token_hex(24), secrets.token_hex(24)
        if not any(p in baseline for p in _pieces(configured) + _pieces(wrong)):
            break
    text = _key_scenario(clean_env / "second", configured, wrong)
    for secret in (configured, wrong):
        leaked = [piece for piece in _pieces(secret) if piece in text]
        assert not leaked, "a piece of an API key reached a log line"


def test_scenario_text_drops_record_timestamps(clean_env):
    """A record's timestamp is the one part of a log line a key cannot leak
    through, and the only part that changes between the baseline scenario and
    the one checked, so the scenario text carries none: the JSON field and
    the text format's prefix both read as a placeholder, and the message
    beside them is kept whole."""
    line = '{"timestamp": "2026-10-21T22:20:01.779152+00:00", "message": "key 7791"}'
    assert (
        _without_timestamps(line)
        == '{"timestamp": "<timestamp>", "message": "key 7791"}'
    )
    assert (
        _without_timestamps("2026-10-21T22:20:01.779152+00:00 INFO [mtc] ok")
        == "<timestamp> INFO [mtc] ok"
    )
    text = _key_scenario(
        clean_env / "run", secrets.token_hex(24), secrets.token_hex(24)
    )
    assert "<timestamp>" in text, "the scenario writes timestamped records"
    assert _TIMESTAMP.search(text) is None, "a timestamp survived in the text"


def test_auth_disabled_on_loopback_warns_once(clean_env, caplog):
    auth_module.reset_auth_warning_for_tests()
    cfg = api_settings(clean_env, API_AUTH_ENABLED=False, API_HOST="127.0.0.1")
    logger = logging.getLogger("mtc")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            assert_bind_allowed(cfg)
            client = TestClient(make_app(cfg))
            for _ in range(3):
                assert client.get("/api/models").status_code == 200
    finally:
        logger.removeHandler(caplog.handler)
    # The handler sits on the root and the mtc logger, so one record may be
    # captured twice; distinct records are what count.
    warnings = {
        id(record)
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "authentication is disabled" in record.getMessage()
    }
    assert len(warnings) == 1


def test_auth_disabled_non_loopback_refuses_to_start(clean_env):
    for host in ("host.invalid", "0.0.0.0", "::"):
        cfg = api_settings(clean_env, API_AUTH_ENABLED=False, API_HOST=host)
        with pytest.raises(UnsafeBindError) as info:
            assert_bind_allowed(cfg)
        message = str(info.value)
        assert "API_HOST" in message and "API_AUTH_ENABLED" in message
    # The application refuses to start under the same settings.
    cfg = api_settings(clean_env, API_AUTH_ENABLED=False, API_HOST="host.invalid")
    with pytest.raises(UnsafeBindError):
        with TestClient(make_app(cfg)):
            pass
    # Every loopback host, and any host with auth on, is allowed.
    for host in ("127.0.0.1", "::1", "localhost", "[::1]", " LOCALHOST "):
        assert_bind_allowed(
            api_settings(clean_env, API_AUTH_ENABLED=False, API_HOST=host)
        )
    assert_bind_allowed(
        api_settings(clean_env, API_AUTH_ENABLED=True, API_HOST="host.invalid")
    )


def test_compare_digest_used(clean_env, key, monkeypatch):
    calls = []
    real = auth_module.secrets.compare_digest

    def spy(left, right):
        calls.append((type(left), type(right)))
        return real(left, right)

    monkeypatch.setattr(auth_module.secrets, "compare_digest", spy)
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    assert client.get("/api/models", headers={API_KEY_HEADER: key}).status_code == 200
    assert client.get("/api/models", headers={API_KEY_HEADER: "x"}).status_code == 401
    assert calls == [(bytes, bytes), (bytes, bytes)]
