"""Websocket handshake auth: the key comes from the header only. A refusal
accepts the connection and then closes it with 1008 and one neutral reason,
so a client under a real server sees the documented close code (a close sent
before the accept surfaces as an HTTP 403 handshake rejection instead) and
learns nothing about why the key failed."""

import asyncio
import secrets

import pytest
from fastapi.testclient import TestClient

from mtc.api.auth import API_KEY_HEADER, WS_POLICY_VIOLATION, WS_REFUSAL_REASON
from mtc.api.server import create_app
from mtc.api.websocket import WS_TRY_AGAIN_LATER
from tests.api.test_auth import api_settings, isolated_env, make_app


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


@pytest.fixture
def key():
    return secrets.token_hex(24)


def _first_message(client, path="/ws", headers=None):
    """The first message the server sends after the handshake. Entering the
    session proves the connection was accepted: the test client raises on a
    close sent before the accept."""
    with client.websocket_connect(path, headers=headers or {}) as ws:
        return ws.receive()


def _assert_refused(message, code=WS_POLICY_VIOLATION, reason=WS_REFUSAL_REASON):
    assert message["type"] == "websocket.close", message
    assert message["code"] == code
    assert message["reason"] == reason


def _asgi_messages(app, headers):
    """Drive the application as an ASGI server does and return what it sent.

    The order is what a real server puts on the wire: ``websocket.accept``
    completes the handshake; a ``websocket.close`` before it becomes an HTTP
    403 response instead of a close frame.
    """
    sent = []
    inbox = [{"type": "websocket.connect"}]

    async def receive():
        if inbox:
            return inbox.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": "/ws",
        "raw_path": b"/ws",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), *headers],
        "client": None,
        "server": None,
        "subprotocols": [],
        "state": {},
    }
    asyncio.run(app(scope, receive, send))
    return sent


def test_ws_header_key_accepted(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    with client.websocket_connect("/ws", headers={API_KEY_HEADER: key}) as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    assert frame["data"]["client_authenticated"] is True


def test_ws_query_param_key_rejected(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    for query in (f"api_key={key}", f"key={key}", f"token={key}"):
        _assert_refused(_first_message(client, f"/ws?{query}"))


def test_ws_missing_key_closes_1008(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    _assert_refused(_first_message(client))
    # A wrong key and a Bearer header are refused the same way, with the same
    # neutral reason.
    for headers in (
        {API_KEY_HEADER: secrets.token_hex(24)},
        {"Authorization": f"Bearer {key}"},
    ):
        _assert_refused(_first_message(client, headers=headers))


def test_ws_empty_configured_key_closes_1008(clean_env):
    client = TestClient(make_app(api_settings(clean_env, API_KEY="")))
    for headers in ({}, {API_KEY_HEADER: ""}, {API_KEY_HEADER: secrets.token_hex(8)}):
        _assert_refused(_first_message(client, headers=headers))


def test_ws_refusal_reason_never_echoes_the_key(clean_env, key):
    client = TestClient(make_app(api_settings(clean_env, API_KEY=key)))
    wrong = secrets.token_hex(24)
    message = _first_message(client, headers={API_KEY_HEADER: wrong})
    _assert_refused(message)
    reason = str(message.get("reason") or "")
    assert key not in reason and wrong not in reason


def test_ws_refusal_is_a_close_frame_after_the_accept(clean_env, key):
    app = make_app(api_settings(clean_env, API_KEY=key))
    wrong = secrets.token_hex(24).encode("ascii")
    for headers in ([], [(b"x-api-key", wrong)]):
        sent = _asgi_messages(app, headers)
        assert [m["type"] for m in sent] == ["websocket.accept", "websocket.close"]
        _assert_refused(sent[1])
    # a valid key is served: the status frame follows the accept
    sent = _asgi_messages(app, [(b"x-api-key", key.encode("ascii"))])
    assert sent[0]["type"] == "websocket.accept"
    assert sent[1]["type"] == "websocket.send"


def test_ws_runtime_not_ready_closes_1013_after_accept(clean_env, key):
    app = create_app(settings=api_settings(clean_env, API_KEY=key), runtime=None)
    client = TestClient(app)
    message = _first_message(client, headers={API_KEY_HEADER: key})
    _assert_refused(message, code=WS_TRY_AGAIN_LATER, reason="runtime not ready")
    sent = _asgi_messages(app, [(b"x-api-key", key.encode("ascii"))])
    assert [m["type"] for m in sent] == ["websocket.accept", "websocket.close"]
    # a key that fails is refused before the runtime is looked at
    _assert_refused(_first_message(client))


LISTED_ORIGIN = "http://ui.invalid"


def _auth_off(tmp_path):
    return api_settings(
        tmp_path,
        API_AUTH_ENABLED=False,
        API_HOST="127.0.0.1",
        API_CORS_ORIGINS=LISTED_ORIGIN,
    )


def test_ws_cross_origin_refused_when_auth_is_off(clean_env):
    """With authentication off a browser page from an origin the operator did
    not list cannot open the socket: the handshake is accepted and closed with
    1008 and the neutral reason, and no status frame is sent. The literal
    origin ``null`` (a sandboxed page or a file) is not a listed origin."""
    app = make_app(_auth_off(clean_env))
    client = TestClient(app)
    for origin in ("http://other.invalid", "null", "http://ui.invalid.other.invalid"):
        _assert_refused(_first_message(client, headers={"Origin": origin}))
        sent = _asgi_messages(app, [(b"origin", origin.encode("ascii"))])
        assert [m["type"] for m in sent] == ["websocket.accept", "websocket.close"]
        _assert_refused(sent[1])


def test_ws_listed_origin_and_no_origin_accepted_when_auth_is_off(clean_env):
    """A listed origin (case and a trailing slash aside) and a client that
    sends no Origin header at all (a non-browser client) are served."""
    client = TestClient(make_app(_auth_off(clean_env)))
    for headers in (
        {},
        {"Origin": LISTED_ORIGIN},
        {"Origin": "HTTP://UI.INVALID/"},
    ):
        with client.websocket_connect("/ws", headers=headers) as ws:
            frame = ws.receive_json()
        assert frame["type"] == "status", headers
        assert frame["data"]["client_authenticated"] is False


def test_ws_origin_is_not_checked_when_auth_is_on(clean_env, key):
    """With authentication on the key decides: a valid key from an unlisted
    origin is served, and an unlisted origin without the key is refused as
    before."""
    cfg = api_settings(clean_env, API_KEY=key, API_CORS_ORIGINS=LISTED_ORIGIN)
    client = TestClient(make_app(cfg))
    headers = {API_KEY_HEADER: key, "Origin": "http://other.invalid"}
    with client.websocket_connect("/ws", headers=headers) as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    assert frame["data"]["client_authenticated"] is True
    _assert_refused(_first_message(client, headers={"Origin": "http://other.invalid"}))


def test_ws_auth_disabled_is_allowed_but_not_an_authenticated_client(
    clean_env,
):
    cfg = api_settings(clean_env, API_AUTH_ENABLED=False, API_HOST="127.0.0.1")
    client = TestClient(make_app(cfg))
    with client.websocket_connect("/ws") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    assert frame["data"]["client_authenticated"] is False
