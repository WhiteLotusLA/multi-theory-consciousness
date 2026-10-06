"""
Websocket Protocol
==================

``/ws`` speaks JSON text frames ``{"type", "data"}`` of at most
``MAX_FRAME_BYTES`` (1 MiB).

Client to server:

- ``message``: ``message``, optional ``speaker``, ``speaker_token`` and
  ``request_id``. One conversation turn through the runtime's pipeline.
- ``ping``: answered with ``pong``.
- ``subscribe_state`` and ``unsubscribe_state``: the connection receives (or
  stops receiving) ``consciousness_state`` frames.

Server to client:

- ``status`` on connect (system name, version, whether the client presented
  the configured API key, the brain's availability, the frame limit);
- ``thinking`` as soon as a turn starts and after every
  ``THINKING_HEARTBEAT_S`` of silence, with ``elapsed_s``;
- ``stream_chunk`` with spoken text only: reasoning blocks are filtered out
  with a one-chunk lookahead (``ThinkFilter``);
- ``stream_retry`` when the brain restarts a looping generation (the chunks
  already sent belong to the discarded attempt);
- ``response`` with ``TurnResult.to_payload()`` and the ``request_id``; the
  response text is authoritative, the streamed chunks are a preview;
- ``consciousness_state`` after each turn and every
  ``METRICS_UPDATE_INTERVAL`` seconds to subscribers;
- ``spontaneous_thought`` for thoughts the continuous thought machine
  expressed, sent only to clients that presented the configured API key;
- ``error`` with ``code``, ``message`` and the ``request_id`` when known;
- ``pong``.

Auth and trust: the handshake needs the ``X-API-Key`` header (a key in the
query string is ignored). A refused handshake is accepted and then closed at
once with 1008 and one neutral reason, so the client sees the close code
under a real server as under the test client (a close before the accept
reaches it as an HTTP 403); a server whose runtime is not built yet closes the
same way with 1013. The API key authenticates the client application only. Who is
speaking is decided by the speaker trust source from the ``speaker_token`` of
each ``message`` frame, which is handed on unchanged, never logged, stored or
echoed; a connection's key never makes its speaker trusted. A frame over the
limit closes the socket with 1009.

Turns from every connection go through the pipeline, whose turn lock
serializes them; a connection keeps answering pings while its turn runs. The
first-token and stall bounds of ``mtc.api.progress`` apply to every turn and
a bound that fires is reported as ``generation_timeout``. Duplicate request
ids are counted by the application's ``RequestObserver`` and still processed.

Config keys read: ``THINKING_HEARTBEAT_S``, ``GENERATION_TIMEOUT_S`` and
``GENERATION_STALL_TIMEOUT_S`` (through ``mtc.api.progress``), ``SYSTEM_NAME``
and the auth keys (through ``mtc.api.auth``). Nothing is persisted here.
Messages, speaker names, streamed text and thoughts are private conversation
data; no log line carries any of them.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from mtc import __version__
from mtc.api.auth import (
    API_KEY_HEADER,
    WS_MESSAGE_TOO_BIG,
    client_authenticated,
    settings_for,
    ws_api_key_valid,
)
from mtc.api.diagnostics import RequestObserver
from mtc.api.progress import GenerationTimeout, run_with_heartbeat
from mtc.api.schemas import (
    MAX_FRAME_BYTES,
    SERVER_FRAME_TYPES,
    ClientFrame,
    WSMessageData,
    jsonable,
    state_payload,
    validation_details,
)
from mtc.brain.brain import STREAM_RETRY_SENTINEL
from mtc.brain.thinking import THINK_CLOSE, THINK_OPEN

logger = logging.getLogger(__name__)

# Close code for a server that cannot serve yet (RFC 6455 registry: try again).
WS_TRY_AGAIN_LATER = 1013


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Reasoning filter
# ---------------------------------------------------------------------------


def _partial_tag_suffix(text: str, tags: tuple) -> int:
    """Length of the longest tail of ``text`` that could start one of ``tags``."""
    longest = 0
    for tag in tags:
        for size in range(min(len(tag) - 1, len(text)), longest, -1):
            if tag.startswith(text[-size:]):
                longest = size
                break
    return longest


class ThinkFilter:
    """Keeps only spoken text from a stream that may carry reasoning blocks.

    Text between an open and a close tag is dropped. A close tag without an
    open tag means the chat template opened the block, so the spoken text
    not yet released is dropped as reasoning: the filter releases a chunk's
    text only after the next chunk ended without completing such a tag (the
    one-chunk lookahead), and a tag split across chunks holds the release
    until it is complete. Reasoning that ran longer than that before a stray
    close tag cannot be recalled; the final ``response`` frame carries the
    authoritative split. Leading whitespace of the spoken text is dropped.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget everything (the brain restarted the generation)."""
        self._inside = False
        self._partial = ""
        self._held = ""
        self._current = ""
        self._started = False

    def _release(self, text: str) -> str:
        if not self._started:
            text = text.lstrip()
            if text:
                self._started = True
        return text

    def feed(self, chunk: str) -> str:
        """Spoken text that is safe to send after this chunk (may be empty)."""
        buffer = self._partial + (chunk or "")
        self._partial = ""
        self._current = ""
        while buffer:
            if self._inside:
                end = buffer.find(THINK_CLOSE)
                if end == -1:
                    keep = _partial_tag_suffix(buffer, (THINK_CLOSE,))
                    self._partial = buffer[len(buffer) - keep :] if keep else ""
                    buffer = ""
                    break
                buffer = buffer[end + len(THINK_CLOSE) :]
                self._inside = False
                continue
            opened = buffer.find(THINK_OPEN)
            closed = buffer.find(THINK_CLOSE)
            if opened != -1 and (closed == -1 or opened < closed):
                self._current += buffer[:opened]
                buffer = buffer[opened + len(THINK_OPEN) :]
                self._inside = True
                continue
            if closed != -1:
                # A stray close tag: everything not yet released was reasoning.
                self._held = ""
                self._current = ""
                buffer = buffer[closed + len(THINK_CLOSE) :]
                continue
            keep = _partial_tag_suffix(buffer, (THINK_OPEN, THINK_CLOSE))
            if keep:
                self._current += buffer[: len(buffer) - keep]
                self._partial = buffer[len(buffer) - keep :]
            else:
                self._current += buffer
            buffer = ""
        if self._partial and not self._inside:
            # A tag may be completing: hold everything until it resolves.
            self._held += self._current
            self._current = ""
            return ""
        out, self._held = self._held, self._current
        self._current = ""
        return self._release(out)

    def flush(self) -> str:
        """Whatever spoken text is still held at the end of the stream."""
        rest = self._held + self._current
        if not self._inside:
            rest += self._partial
        self._held = self._current = self._partial = ""
        return self._release(rest)


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Connection:
    """One open socket and what it may receive."""

    websocket: Any
    client_authenticated: bool
    subscribed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: Set["asyncio.Task[Any]"] = field(default_factory=set)


class ConnectionManager:
    """The open websocket connections of one application."""

    def __init__(self) -> None:
        self._connections: Dict[int, Connection] = {}
        self.sent = 0
        self.send_failures = 0
        self.thoughts_broadcast = 0

    def __len__(self) -> int:
        return len(self._connections)

    async def connect(
        self, websocket: Any, *, client_authenticated: bool = False
    ) -> Connection:
        """Accept the socket and track it."""
        await websocket.accept()
        connection = Connection(websocket, bool(client_authenticated))
        self._connections[id(websocket)] = connection
        return connection

    def get(self, websocket: Any) -> Optional[Connection]:
        return self._connections.get(id(websocket))

    def disconnect(self, websocket: Any) -> None:
        self._connections.pop(id(websocket), None)

    def connections(self) -> List[Connection]:
        return list(self._connections.values())

    def subscribers(self) -> List[Connection]:
        return [conn for conn in self._connections.values() if conn.subscribed]

    async def send(
        self, target: Any, frame_type: str, data: Optional[Mapping[str, Any]] = None
    ) -> bool:
        """Send one frame; False (and the connection dropped) when it fails."""
        if frame_type not in SERVER_FRAME_TYPES:
            raise ValueError(f"unknown server frame type {frame_type!r}")
        connection = target if isinstance(target, Connection) else self.get(target)
        websocket = connection.websocket if connection is not None else target
        text = json.dumps(
            {"type": frame_type, "data": jsonable(dict(data or {}))},
            allow_nan=False,
        )
        try:
            if connection is not None:
                async with connection.lock:
                    await websocket.send_text(text)
            else:
                await websocket.send_text(text)
        except Exception as exc:  # noqa: BLE001 - a gone client is not an error
            self.send_failures += 1
            logger.debug("websocket send failed: %s", type(exc).__name__)
            self.disconnect(websocket)
            return False
        self.sent += 1
        return True

    async def broadcast_thought(self, message: Mapping[str, Any]) -> int:
        """Send an expressed thought to the clients that presented the key.

        ``message`` is the continuous thought machine's broadcast record
        (``thought`` plus ``stream_state``). Returns the clients reached.
        """
        thought = dict(message.get("thought") or {})
        data = {
            "thought_id": thought.get("id"),
            "content": thought.get("content"),
            "thought_type": thought.get("type"),
            "timestamp": thought.get("timestamp"),
            "intensity": thought.get("intensity"),
            "salience": thought.get("salience"),
        }
        reached = 0
        for connection in self.connections():
            if not connection.client_authenticated:
                continue
            if await self.send(connection, "spontaneous_thought", data):
                reached += 1
        self.thoughts_broadcast += 1
        return reached

    async def push_state(self, payload: Mapping[str, Any]) -> int:
        """Send a ``consciousness_state`` frame to every subscriber."""
        reached = 0
        for connection in self.subscribers():
            if await self.send(connection, "consciousness_state", payload):
                reached += 1
        return reached

    def status(self) -> Dict[str, int]:
        conns = self.connections()
        return {
            "connections": len(conns),
            "client_authenticated": sum(1 for c in conns if c.client_authenticated),
            "subscribers": sum(1 for c in conns if c.subscribed),
            "sent": self.sent,
            "send_failures": self.send_failures,
            "thoughts_broadcast": self.thoughts_broadcast,
        }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _app_state(websocket: Any) -> Any:
    return getattr(getattr(websocket, "app", None), "state", None)


def _manager(websocket: Any) -> ConnectionManager:
    state = _app_state(websocket)
    manager = getattr(state, "connections", None)
    if not isinstance(manager, ConnectionManager):
        manager = ConnectionManager()
        if state is not None:
            state.connections = manager
    return manager


def _observer(websocket: Any) -> RequestObserver:
    state = _app_state(websocket)
    observer = getattr(state, "observer", None)
    if not isinstance(observer, RequestObserver):
        observer = RequestObserver()
        if state is not None:
            state.observer = observer
    return observer


async def _error(
    manager: ConnectionManager,
    connection: Connection,
    code: str,
    message: str,
    request_id: Optional[str] = None,
) -> None:
    await manager.send(
        connection,
        "error",
        {"code": code, "message": message, "request_id": request_id},
    )


async def _run_turn(
    runtime: Any,
    manager: ConnectionManager,
    observer: RequestObserver,
    connection: Connection,
    data: WSMessageData,
    settings: Any,
) -> None:
    request_id = data.request_id
    receipts = observer.observe_receipt(request_id)
    if receipts > 1:
        logger.warning(
            "request id received %d times within the window; processed again "
            "(%d open connections)",
            receipts,
            len(manager),
        )
    token = data.speaker_token.get_secret_value() if data.speaker_token else None
    await manager.send(
        connection, "thinking", {"elapsed_s": 0.0, "request_id": request_id}
    )

    spoken = ThinkFilter()

    async def sink(chunk: str) -> None:
        if chunk == STREAM_RETRY_SENTINEL:
            spoken.reset()
            await manager.send(connection, "stream_retry", {"request_id": request_id})
            return
        text = spoken.feed(chunk)
        if text:
            await manager.send(
                connection, "stream_chunk", {"content": text, "request_id": request_id}
            )

    async def beat(elapsed: float) -> None:
        await manager.send(
            connection,
            "thinking",
            {"elapsed_s": round(float(elapsed), 1), "request_id": request_id},
        )

    pipeline = getattr(runtime, "pipeline", None)
    if pipeline is None:
        await _error(
            manager,
            connection,
            "runtime_not_ready",
            "the pipeline is not built",
            request_id,
        )
        return
    try:
        result = await run_with_heartbeat(
            lambda mark: pipeline.send_message(
                data.message,
                data.speaker,
                token,
                stream_callback=mark.wrap(sink),
                request_id=request_id,
            ),
            on_heartbeat=beat,
            settings=settings,
        )
    except GenerationTimeout as exc:
        logger.warning("websocket turn cancelled: %s", exc.detail())
        observer.observe_response(request_id)
        await _error(
            manager, connection, "generation_timeout", exc.describe(), request_id
        )
        return
    except Exception as exc:  # noqa: BLE001 - reported to the client, counted
        logger.error("websocket turn failed with %s", type(exc).__name__)
        observer.observe_response(request_id)
        await _error(manager, connection, "turn_failed", "the turn failed", request_id)
        return
    rest = spoken.flush()
    if rest:
        await manager.send(
            connection, "stream_chunk", {"content": rest, "request_id": request_id}
        )
    responses = observer.observe_response(request_id)
    if responses > 1:
        logger.warning("more than one response sent for one request id (%d)", responses)
    payload = result.to_payload()
    if result.success:
        payload["request_id"] = request_id
        await manager.send(connection, "response", payload)
    else:
        await _error(
            manager,
            connection,
            str(result.error_code or "turn_failed"),
            str(result.error or "the turn ended without a reply"),
            request_id,
        )
    try:
        await manager.push_state(state_payload(runtime))
    except Exception as exc:  # noqa: BLE001 - a state push never fails a turn
        logger.warning("state push after a turn failed: %s", type(exc).__name__)


async def _handle_frame(
    runtime: Any,
    manager: ConnectionManager,
    observer: RequestObserver,
    connection: Connection,
    text: str,
    settings: Any,
) -> None:
    try:
        raw = json.loads(text)
    except (TypeError, ValueError):
        await _error(manager, connection, "invalid_message", "frames must be JSON")
        return
    if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
        await _error(
            manager, connection, "invalid_message", "a frame is an object with a type"
        )
        return
    try:
        frame = ClientFrame.model_validate(raw)
    except ValidationError as exc:
        kinds = {item["type"] for item in validation_details(exc)}
        code = "unknown_type" if "literal_error" in kinds else "invalid_message"
        await _error(manager, connection, code, "the frame is not a client frame")
        return
    if frame.type == "ping":
        await manager.send(connection, "pong", {"timestamp": _now_iso()})
        return
    if frame.type == "subscribe_state":
        connection.subscribed = True
        await manager.send(connection, "consciousness_state", state_payload(runtime))
        return
    if frame.type == "unsubscribe_state":
        connection.subscribed = False
        return
    request_id = frame.data.get("request_id")
    request_id = request_id if isinstance(request_id, str) else None
    try:
        data = WSMessageData.model_validate(frame.data)
    except ValidationError as exc:
        fields = sorted({".".join(item["loc"]) for item in validation_details(exc)})
        await _error(
            manager,
            connection,
            "invalid_message",
            "invalid message fields: " + ", ".join(fields),
            request_id,
        )
        return
    task = asyncio.get_running_loop().create_task(
        _run_turn(runtime, manager, observer, connection, data, settings),
        name="websocket-turn",
    )
    connection.tasks.add(task)
    task.add_done_callback(connection.tasks.discard)
    state = _app_state(connection.websocket)
    pending = getattr(state, "turn_tasks", None)
    if isinstance(pending, set):
        pending.add(task)
        task.add_done_callback(pending.discard)


async def _refuse(websocket: WebSocket, code: int, reason: str) -> None:
    """Accept the connection, then close it with ``code``. A close sent before
    the accept reaches a client under a real server as an HTTP 403 handshake
    rejection, never as the close code; nothing is read or sent in between."""
    await websocket.accept()
    await websocket.close(code=code, reason=reason)


async def handle_ws(websocket: WebSocket, runtime: Any) -> None:
    """Serve one websocket connection until the client leaves."""
    allowed, code, reason = ws_api_key_valid(websocket)
    if not allowed:
        await _refuse(websocket, code, reason)
        return
    if runtime is None:
        await _refuse(websocket, WS_TRY_AGAIN_LATER, "runtime not ready")
        return
    settings = settings_for(getattr(websocket, "app", None))
    manager = _manager(websocket)
    observer = _observer(websocket)
    authenticated = client_authenticated(
        settings, websocket.headers.get(API_KEY_HEADER)
    )
    connection = await manager.connect(websocket, client_authenticated=authenticated)
    brain = getattr(runtime, "brain", None)
    await manager.send(
        connection,
        "status",
        {
            "system_name": settings.SYSTEM_NAME,
            "version": __version__,
            "client_authenticated": authenticated,
            "brain_available": getattr(brain, "available", None),
            "max_frame_bytes": MAX_FRAME_BYTES,
            "heartbeat_s": float(settings.THINKING_HEARTBEAT_S),
        },
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text = message.get("text")
            if text is None:
                if message.get("bytes") is not None:
                    if len(message["bytes"]) > MAX_FRAME_BYTES:
                        await websocket.close(
                            code=WS_MESSAGE_TOO_BIG, reason="frame too large"
                        )
                        break
                    await _error(
                        manager, connection, "invalid_message", "text frames only"
                    )
                continue
            if len(text.encode("utf-8")) > MAX_FRAME_BYTES:
                await websocket.close(code=WS_MESSAGE_TOO_BIG, reason="frame too large")
                break
            await _handle_frame(runtime, manager, observer, connection, text, settings)
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
