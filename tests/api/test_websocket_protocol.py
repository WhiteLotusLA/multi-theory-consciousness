"""The websocket protocol: frames, the reasoning filter, retries, state
pushes, thought broadcasts, errors, the frame limit and turn serialization."""

import asyncio
import json
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mtc.api.auth import API_KEY_HEADER
from mtc.api.schemas import INTEGRATION_PROXY_LABEL, MAX_FRAME_BYTES
from mtc.api.server import create_app, push_state_once
from mtc.api.websocket import ConnectionManager, ThinkFilter
from mtc.brain.brain import STREAM_RETRY_SENTINEL
from mtc.core.embeddings import HashEmbedder
from mtc.pipeline.context_types import TurnResult
from mtc.pipeline.runtime import build_runtime
from tests.api.test_auth import (
    T0,
    FakeRuntime,
    api_settings,
    isolated_env,
    make_app,
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


pytestmark = pytest.mark.timeout(300)

ROOT = Path(__file__).resolve().parents[2]


def _result(text: Optional[str] = "[fake] reply", **overrides: Any) -> TurnResult:
    values: Dict[str, Any] = {
        "success": text is not None,
        "response": text,
        "thinking": None,
        "conversation_id": None,
        "message_count": 2,
        "timestamp": T0,
        "cycle_ok": True,
        "speaker_label": "",
        "provider_id": "fake",
    }
    values.update(overrides)
    return TurnResult(**values)


class ScriptedPipeline:
    """Streams a scripted list of chunks, then returns a result.

    ``hang`` makes a turn wait forever (until a bound cancels it).
    """

    def __init__(self, chunks=(), result=None, hang: bool = False) -> None:
        self.chunks = list(chunks)
        self.result = result or _result()
        self.hang = hang
        self.calls: List[Dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def send_message(
        self,
        message,
        speaker=None,
        speaker_token=None,
        *,
        stream_callback=None,
        request_id=None,
        timeout_s=None,
    ):
        self.calls.append({"message": message, "speaker": speaker})
        if self.hang:
            await asyncio.Event().wait()
        for chunk in self.chunks:
            if stream_callback is not None:
                await stream_callback(chunk)
        return self.result

    def status(self):
        return {"turns": len(self.calls)}


class FakeMachine:
    def __init__(self) -> None:
        self.broadcaster = None
        self.running = False

    def get_status(self):
        return {"running": self.running, "quiet_reason": "no_embedder"}


@pytest.fixture
def key():
    return secrets.token_hex(24)


def _app(tmp, key, pipeline=None, **settings):
    cfg = api_settings(tmp, API_KEY=key, METRICS_UPDATE_INTERVAL=0, **settings)
    runtime = FakeRuntime(cfg)
    if pipeline is not None:
        runtime.pipeline = pipeline
    return make_app(cfg, runtime), runtime


def _connect(client, key):
    return client.websocket_connect("/ws", headers={API_KEY_HEADER: key})


def _until(ws, frame_type: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Frames received up to and including the first of ``frame_type``."""
    frames = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] == frame_type:
            return frames
    raise AssertionError(f"no {frame_type} frame within {limit} frames")


def _turn_frames(ws, request_id: str) -> List[Dict[str, Any]]:
    frames = []
    for _ in range(200):
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] in ("response", "error") and (
            frame["data"].get("request_id") == request_id
        ):
            return frames
    raise AssertionError("the turn did not finish")


def _send(ws, message: str, request_id: str, **extra: Any) -> None:
    ws.send_json(
        {
            "type": "message",
            "data": {"message": message, "request_id": request_id, **extra},
        }
    )


def _spoken(frames) -> str:
    return "".join(f["data"]["content"] for f in frames if f["type"] == "stream_chunk")


# ---------------------------------------------------------------------------
# Reasoning filter (unit)
# ---------------------------------------------------------------------------


def _filtered(chunks) -> str:
    spoken = ThinkFilter()
    out = [spoken.feed(chunk) for chunk in chunks]
    out.append(spoken.flush())
    return "".join(out)


def test_filter_drops_reasoning_blocks_and_split_tags():
    assert _filtered(["<thi", "nk>secret plan</th", "ink>\n\nVisible", " text"]) == (
        "Visible text"
    )
    assert _filtered(["Before <think>x</think> after"]) == "Before  after"
    assert _filtered(["plain", " words"]) == "plain words"
    assert _filtered(["a < b and c", " > d"]) == "a < b and c > d"


def test_filter_drops_text_before_a_stray_close_tag():
    # The template opened the block; the close tag arrives split.
    assert _filtered(["hidden reason", "ing</th", "ink>", "Shown"]) == "Shown"
    assert _filtered(["hidden", "</think>", "Shown"]) == "Shown"


def test_filter_reset_forgets_the_discarded_attempt():
    spoken = ThinkFilter()
    assert spoken.feed("first attempt") == ""
    spoken.reset()
    assert spoken.feed("second") == ""
    assert spoken.flush() == "second"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_status_on_connect(clean_env, key):
    app, _ = _app(clean_env, key, SYSTEM_NAME="Testagent")
    with TestClient(app) as client, _connect(client, key) as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    data = frame["data"]
    assert data["system_name"] == "Testagent"
    assert data["client_authenticated"] is True
    assert data["max_frame_bytes"] == MAX_FRAME_BYTES
    assert data["version"]


def test_message_turn_sends_thinking_chunks_and_response(clean_env, key):
    pipeline = ScriptedPipeline(
        chunks=["Hello", " world"], result=_result("Hello world")
    )
    app, _ = _app(clean_env, key, pipeline)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        _send(ws, "hi there", "req-1", speaker="Alice")
        frames = _turn_frames(ws, "req-1")
    types = [f["type"] for f in frames]
    assert types[0] == "thinking" and frames[0]["data"]["elapsed_s"] == 0.0
    assert "stream_chunk" in types and types[-1] == "response"
    assert _spoken(frames) == "Hello world"
    response = frames[-1]["data"]
    assert response["request_id"] == "req-1"
    assert response["response"] == "Hello world" and response["success"] is True
    assert pipeline.calls == [{"message": "hi there", "speaker": "Alice"}]


def test_think_blocks_filtered_from_chunks(clean_env, key):
    pipeline = ScriptedPipeline(
        chunks=["<thi", "nk>secret plan</th", "ink>\n\nVisible", " text"],
        result=_result("Visible text"),
    )
    app, _ = _app(clean_env, key, pipeline)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        _send(ws, "hi", "req-1")
        frames = _turn_frames(ws, "req-1")
        pipeline.chunks = ["private reason", "ing</th", "ink>", "Shown"]
        _send(ws, "again", "req-2")
        stray = _turn_frames(ws, "req-2")
    assert _spoken(frames) == "Visible text"
    assert _spoken(stray) == "Shown"
    for frame in frames + stray:
        if frame["type"] == "stream_chunk":
            text = frame["data"]["content"]
            assert "secret" not in text and "private" not in text
            assert "<" not in text and ">" not in text


def test_stream_retry_forwarded(clean_env, key):
    pipeline = ScriptedPipeline(
        chunks=["first try", STREAM_RETRY_SENTINEL, "second try"],
        result=_result("second try"),
    )
    app, _ = _app(clean_env, key, pipeline)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        _send(ws, "hi", "req-1")
        frames = _turn_frames(ws, "req-1")
    types = [f["type"] for f in frames]
    assert "stream_retry" in types
    retry_at = types.index("stream_retry")
    assert frames[retry_at]["data"]["request_id"] == "req-1"
    after = _spoken(frames[retry_at:])
    assert after == "second try"
    assert all(STREAM_RETRY_SENTINEL not in json.dumps(f) for f in frames)


def test_ping_pong(clean_env, key):
    app, _ = _app(clean_env, key)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        ws.send_json({"type": "ping", "data": {}})
        frame = ws.receive_json()
    assert frame["type"] == "pong" and frame["data"]["timestamp"]


def test_subscribe_state_receives_consciousness_state(clean_env, key):
    app, _ = _app(clean_env, key)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        ws.send_json({"type": "subscribe_state", "data": {}})
        frame = ws.receive_json()
        assert frame["type"] == "consciousness_state"
        proxy = frame["data"]["integration_proxy"]
        assert proxy["label"] == INTEGRATION_PROXY_LABEL
        # After a turn the subscriber gets the state again.
        _send(ws, "hi", "req-1")
        frames = _until(ws, "consciousness_state")
        assert any(f["type"] == "response" for f in frames)
        ws.send_json({"type": "unsubscribe_state", "data": {}})
        _send(ws, "hi", "req-2")
        after = _turn_frames(ws, "req-2")
        ws.send_json({"type": "ping", "data": {}})
        tail = _until(ws, "pong")
    assert all(f["type"] != "consciousness_state" for f in after + tail)


def test_periodic_state_push_reaches_subscribers_only(clean_env, key):
    app, _ = _app(clean_env, key)
    with TestClient(app) as client:
        with _connect(client, key) as watcher, _connect(client, key) as other:
            watcher.receive_json()
            other.receive_json()
            assert client.portal.call(push_state_once, app) == 0
            watcher.send_json({"type": "subscribe_state", "data": {}})
            assert watcher.receive_json()["type"] == "consciousness_state"
            # One step of the METRICS_UPDATE_INTERVAL loop.
            assert client.portal.call(push_state_once, app) == 1
            pushed = watcher.receive_json()
            other.send_json({"type": "ping", "data": {}})
            assert other.receive_json()["type"] == "pong"
    assert pushed["type"] == "consciousness_state"
    assert pushed["data"]["integration_proxy"]["label"] == INTEGRATION_PROXY_LABEL


def test_spontaneous_thought_broadcast_to_authenticated_clients(clean_env, key):
    app, runtime = _app(clean_env, key)
    runtime.ctm = FakeMachine()
    message = {
        "type": "spontaneous_thought",
        "thought": {
            "id": "thought-1",
            "type": "reflection",
            "content": "A seeded reflection.",
            "timestamp": T0.isoformat(),
        },
        "stream_state": {},
    }
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        assert runtime.ctm.broadcaster is not None, "the server attached its hook"
        client.portal.call(runtime.ctm.broadcaster, message)
        frame = ws.receive_json()
    assert frame["type"] == "spontaneous_thought"
    assert frame["data"]["thought_id"] == "thought-1"
    assert frame["data"]["content"] == "A seeded reflection."
    assert runtime.ctm.broadcaster is None, "the hook is detached at shutdown"


@pytest.mark.asyncio
async def test_thought_broadcast_skips_clients_without_the_key():
    class Socket:
        def __init__(self):
            self.sent = []

        async def accept(self):
            return None

        async def send_text(self, text):
            self.sent.append(json.loads(text))

    manager = ConnectionManager()
    keyed, unkeyed = Socket(), Socket()
    await manager.connect(keyed, client_authenticated=True)
    await manager.connect(unkeyed, client_authenticated=False)
    reached = await manager.broadcast_thought(
        {"thought": {"id": "t", "type": "reflection", "content": "x"}}
    )
    assert reached == 1
    assert [f["type"] for f in keyed.sent] == ["spontaneous_thought"]
    assert unkeyed.sent == []


def test_invalid_message_error(clean_env, key):
    app, runtime = _app(clean_env, key)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        ws.send_text("not json")
        not_json = ws.receive_json()
        ws.send_json({"type": "dance", "data": {}})
        unknown = ws.receive_json()
        ws.send_json({"type": "message", "data": {"request_id": "req-9"}})
        missing = ws.receive_json()
        token = secrets.token_hex(16)
        ws.send_json(
            {
                "type": "message",
                "data": {"message": "hi", "speaker_token": token, "extra": token},
            }
        )
        extra = ws.receive_json()
    assert not_json["type"] == "error" and not_json["data"]["code"] == "invalid_message"
    assert unknown["data"]["code"] == "unknown_type"
    assert missing["data"]["code"] == "invalid_message"
    assert missing["data"]["request_id"] == "req-9"
    assert "message" in missing["data"]["message"]
    assert extra["data"]["code"] == "invalid_message"
    assert token not in json.dumps(extra)
    assert runtime.pipeline.calls == []


def test_oversize_frame_rejected(clean_env, key):
    app, runtime = _app(clean_env, key)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        ws.send_text("x" * (MAX_FRAME_BYTES + 1))
        with pytest.raises(WebSocketDisconnect) as info:
            ws.receive_json()
    assert info.value.code == 1009
    assert runtime.pipeline.calls == []


def test_generation_timeout_error_code(clean_env, key):
    hanging = ScriptedPipeline(hang=True)
    app, _ = _app(clean_env, key, hanging, GENERATION_TIMEOUT_S=0.05)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        _send(ws, "hi", "req-1")
        frames = _turn_frames(ws, "req-1")
    error = frames[-1]
    assert error["type"] == "error"
    assert error["data"]["code"] == "generation_timeout"
    assert "first token" in error["data"]["message"]
    # A timeout the pipeline itself reports carries the same code.
    timed_out = ScriptedPipeline(
        result=_result(
            None,
            error="The generation ran past its time bound.",
            error_code="generation_timeout",
        )
    )
    app, _ = _app(clean_env, key, timed_out)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        _send(ws, "hi", "req-2")
        frames = _turn_frames(ws, "req-2")
    assert frames[-1]["data"]["code"] == "generation_timeout"


def test_failed_turn_codes_reach_the_client(clean_env, key):
    for code in ("blocked", "no_answer", "brain_unavailable"):
        pipeline = ScriptedPipeline(
            result=_result(
                None, error="no reply", error_code=code, blocked=code == "blocked"
            )
        )
        app, _ = _app(clean_env, key, pipeline)
        with TestClient(app) as client, _connect(client, key) as ws:
            ws.receive_json()
            _send(ws, "hi", "req-1")
            frames = _turn_frames(ws, "req-1")
        assert frames[-1]["type"] == "error"
        assert frames[-1]["data"]["code"] == code
        assert not any(f["type"] == "response" for f in frames)


def _real_app(tmp, key, **settings):
    """The server over a runtime built with the mock provider, in memory."""
    cfg = api_settings(
        tmp,
        API_KEY=key,
        METRICS_UPDATE_INTERVAL=0,
        CTM_CONFIG_DIR=str(ROOT / "config" / "ctm"),
        **settings,
    )

    async def builder(active):
        return await build_runtime(
            active, embedder=HashEmbedder(), allow_non_semantic=True, substrates=False
        )

    return create_app(settings=cfg, runtime_builder=builder)


def test_mock_turn_over_the_real_pipeline(clean_env, key):
    app = _real_app(clean_env, key, THINKING_MODE=True)
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        _send(ws, "Describe the room.", "req-1", speaker="Alice")
        frames = _turn_frames(ws, "req-1")
    response = frames[-1]
    assert response["type"] == "response", frames[-1]
    text = response["data"]["response"]
    assert text.startswith("[mock]")
    streamed = _spoken(frames)
    assert streamed == text
    assert "mock reasoning" not in streamed
    assert response["data"]["speaker_label"] == "[Alice]"


def test_expressed_thoughts_reach_keyed_clients_through_the_runtime(clean_env, key):
    app = _real_app(clean_env, key)
    message = {
        "type": "spontaneous_thought",
        "thought": {"id": "t-9", "type": "question", "content": "A seeded question."},
        "stream_state": {},
    }
    with TestClient(app) as client, _connect(client, key) as ws:
        ws.receive_json()
        runtime = app.state.runtime
        manager = app.state.connections
        listeners = list(runtime.thought_listeners)
        ours = [f for f in listeners if getattr(f, "__self__", None) is manager]
        assert len(ours) == 1, "the server listens to the runtime once"

        async def deliver():
            for listener in listeners:
                await listener(message)

        client.portal.call(deliver)
        frame = ws.receive_json()
    assert frame["type"] == "spontaneous_thought"
    assert frame["data"]["thought_id"] == "t-9"
    assert not [
        f for f in runtime.thought_listeners if getattr(f, "__self__", None) is manager
    ], "the listener is removed at shutdown"


def test_turns_from_two_connections_serialize(clean_env, key):
    app = _real_app(clean_env, key)
    record = {"active": 0, "max": 0, "calls": 0}
    with TestClient(app) as client:
        orchestrator = app.state.runtime.pipeline.orchestrator
        process = orchestrator.process

        async def watched(raw):
            # Runs inside the turn: an overlap of two turns would show here.
            record["active"] += 1
            record["calls"] += 1
            record["max"] = max(record["max"], record["active"])
            try:
                for _ in range(10):
                    await asyncio.sleep(0)
                return await process(raw)
            finally:
                record["active"] -= 1

        orchestrator.process = watched
        try:
            with _connect(client, key) as first, _connect(client, key) as second:
                first.receive_json()
                second.receive_json()
                _send(first, "from one", "req-a")
                _send(second, "from two", "req-b")
                _send(first, "again from one", "req-c")
                a = _turn_frames(first, "req-a")
                b = _turn_frames(second, "req-b")
                c = _turn_frames(first, "req-c")
        finally:
            orchestrator.process = process
    assert a[-1]["type"] == b[-1]["type"] == c[-1]["type"] == "response"
    assert record["calls"] == 3
    assert record["max"] == 1, "two turns overlapped"
