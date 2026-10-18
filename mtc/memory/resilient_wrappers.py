"""
Resilient Store Wrappers
========================

Opt-in layers over real store clients. Each wrapper puts a ``CircuitBreaker``
in front of its store so a down server costs one failed call, not a hung
pipeline, and says plainly what it did instead:

- ``ResilientPool`` (asyncpg pool): ``execute`` and ``execute_returning``
  buffer the statement while the store is unavailable and return ``BUFFERED``;
  ``fetch``, ``fetchrow``, ``fetchval`` and ``acquire`` raise
  ``StoreUnavailable``, because an empty list is not an answer. A buffered
  episodic insert loses its returned id, so the memory manager counts it as
  unindexed and the re-embed backfill indexes it later.
- ``ResilientMongoStore`` (a ``ConversationTurnStore``): turns and session
  records buffer and return ``BUFFERED``; reads raise ``StoreUnavailable``.
- ``ResilientRedisManager`` (a ``WorkingMemory``): the one place a fallback
  answers a read, and it serves only what was written to it during the outage
  (nothing else is invented); on recovery that state replays into Redis.

Wrappers wrap real clients only; a missing store is the in-memory
implementation from ``mtc.memory.inmemory``, never a wrapper around None.

Only a connection-class failure (the server could not be reached) is an outage:
it buffers the write and counts toward the breaker. For PostgreSQL that is a
connection error or a closed or closing connection or pool
(``mtc.memory.postgres_stores.is_unreachable``); a client-side driver error (a
wrong argument count, invalid input for an argument) is not. Any other error
(a NUL byte in text, a constraint violation, a value that is not JSON) is the
caller's data problem: it propagates unbuffered, and the server's answer
counts as proof it is reachable. A buffered write that later fails for such a reason on
replay is dropped with an ERROR log and counted in ``status()["rejected"]``, so
one bad entry cannot hold the queue.

Buffers hold up to ``buffer_size`` writes (default 500); when full the oldest
write is dropped, logged at ERROR and counted in ``status()["dropped"]``, in
every wrapper. Replay is first in,
first out, runs when the breaker recovers (the breaker's recovery listener) and
also before the next write or read reaches the store, so a later write never
overtakes an earlier one; a replay that fails keeps the rest buffered and
nothing replays twice. The buffers live in process memory.

Config keys read at call time: ``STORE_FALLBACK_LOG_ENABLED`` and
``LOG_DIRECTORY``. When enabled, each buffered write is also appended to
``<LOG_DIRECTORY>/store_fallback/<store>.jsonl`` so a process that dies
mid-outage leaves a record for hand recovery. Those files hold buffered
statements with their arguments and turn text, which is private conversation
content, and belong in a gitignored or operator-controlled directory. A
breadcrumb that cannot be written is logged at WARNING once per file name per
process (later failures at DEBUG) and counted in the wrapper's
``status()["breadcrumb_failures"]``; the write itself stays buffered.

Every wrapper exposes ``status()`` with ``backend``, ``available``,
``last_error``, ``mode`` (``live`` or ``buffering``), the circuit state and the
buffered, dropped, replayed and rejected counts; the PostgreSQL and MongoDB
wrappers, which write breadcrumbs, add ``breadcrumb_failures``.
"""

import asyncio
import copy
import inspect
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set

from mtc.core.circuit_breaker import CircuitBreaker
from mtc.core.config import Settings, get_settings
from mtc.core.paths import ensure_dir, log_dir
from mtc.memory.postgres_stores import is_unreachable
from mtc.memory.protocols import (
    ConversationTurnStore,
    StoreUnavailable,
    WorkingMemory,
    clamp_working_capacity,
)

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SIZE = 500
FALLBACK_CACHE_LIMIT = 100


class _Buffered:
    """Marker returned by a write that was buffered instead of executed."""

    def __repr__(self) -> str:
        return "BUFFERED"

    def __bool__(self) -> bool:
        return False


BUFFERED = _Buffered()


class _Rejected:
    """A non-connection error carried through the breaker as a success.

    The server answered (with a refusal), so the breaker must not count an
    outage; the wrapper raises the original error once the breaker returns.
    """

    def __init__(self, error: Exception) -> None:
        self.error = error


def _settle(result: Any) -> Any:
    if isinstance(result, _Rejected):
        raise result.error
    return result


# Driver exception class names that mean the server was not reached. Matched by
# name so this module never imports a driver.
_MONGO_UNREACHABLE_NAMES = frozenset(
    {
        "ConnectionFailure",
        "AutoReconnect",
        "ServerSelectionTimeoutError",
        "NetworkTimeout",
        "NotPrimaryError",
        "WaitQueueTimeoutError",
    }
)
_REDIS_UNREACHABLE_NAMES = frozenset(
    {
        "ConnectionError",
        "TimeoutError",
        "BusyLoadingError",
        "ReadOnlyError",
        "MasterDownError",
        "ClusterDownError",
        "TryAgainError",
    }
)
# redis-py reports a rejected command (WRONGTYPE, a bad argument) as these.
_REDIS_REFUSAL_NAMES = frozenset({"ResponseError", "DataError"})


def _class_names(exc: BaseException) -> frozenset:
    return frozenset(cls.__name__ for cls in type(exc).__mro__)


def _postgres_connection_error(exc: BaseException) -> bool:
    return isinstance(exc, StoreUnavailable) or is_unreachable(exc)


def _mongo_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, StoreUnavailable):
        cause = exc.__cause__
        return cause is None or not (
            _class_names(cause) & {"DuplicateKeyError", "WriteError", "BulkWriteError"}
        )
    return isinstance(exc, (OSError, asyncio.TimeoutError)) or bool(
        _class_names(exc) & _MONGO_UNREACHABLE_NAMES
    )


def _redis_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, StoreUnavailable):
        cause = exc.__cause__
        if cause is None:
            return True
        names = _class_names(cause)
        # RedisManager chains every client failure; a refused command is data
        return bool(names & _REDIS_UNREACHABLE_NAMES) or not (
            names & _REDIS_REFUSAL_NAMES
        )
    return isinstance(exc, (OSError, asyncio.TimeoutError)) or bool(
        _class_names(exc) & _REDIS_UNREACHABLE_NAMES
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Breadcrumb file names whose write failure was already logged at WARNING.
_BREADCRUMB_WARNED: Set[str] = set()


def _breadcrumb(
    settings: Optional[Settings], filename: str, record: Dict[str, Any]
) -> Optional[bool]:
    """Append a JSON line under ``<LOG_DIRECTORY>/store_fallback`` when enabled.

    Returns True when the line was written, None when breadcrumbs are off and
    False when the write failed. A failure is logged at WARNING the first time
    per file name in a process and at DEBUG after that; the record (private
    content) never reaches the log.
    """
    try:
        active = settings if settings is not None else get_settings()
        if not active.STORE_FALLBACK_LOG_ENABLED:
            return None
        directory = ensure_dir(log_dir(active) / "store_fallback")
        with open(directory / filename, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - counted by the caller, never raised
        description = f"{type(exc).__name__}: {exc}"
        if filename not in _BREADCRUMB_WARNED:
            _BREADCRUMB_WARNED.add(filename)
            logger.warning(
                "fallback breadcrumb %s not written: %s (later failures for "
                "this file are logged at DEBUG)",
                filename,
                description,
            )
        else:
            logger.debug(
                "fallback breadcrumb %s not written: %s", filename, description
            )
        return False
    return True


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class _WriteBuffer:
    """A bounded FIFO of buffered writes with an exactly-once replay."""

    def __init__(self, label: str, size: int) -> None:
        if size < 1:
            raise ValueError("buffer_size must be at least 1")
        self.label = label
        self.size = int(size)
        self.items: Deque[Dict[str, Any]] = deque()
        self.dropped = 0
        self.replayed = 0
        self.rejected = 0
        self._lock = asyncio.Lock()
        self._announced = False

    def add(self, entry: Dict[str, Any]) -> None:
        if len(self.items) >= self.size:
            self.items.popleft()
            self.dropped += 1
            logger.error(
                "%s outage: write buffer full (%d); the oldest write was dropped",
                self.label,
                self.size,
            )
        self.items.append(entry)
        if not self._announced:
            self._announced = True
            logger.error(
                "%s outage: writes are buffering in memory and replay on recovery",
                self.label,
            )

    def _discard(self, entry: Dict[str, Any]) -> None:
        """Remove ``entry`` if still at the head (an overflow drop may have)."""
        if self.items and self.items[0] is entry:
            self.items.popleft()

    async def replay(
        self,
        apply: Callable[[Dict[str, Any]], Awaitable[Any]],
        is_connection_error: Callable[[BaseException], bool],
    ) -> None:
        """Apply buffered writes in order.

        A connection failure stops the replay and keeps the rest buffered; any
        other error drops that one entry (logged, counted as rejected) so a
        bad write cannot block the queue.
        """
        async with self._lock:
            total = len(self.items)
            done = 0
            while self.items:
                entry = self.items[0]
                try:
                    await apply(entry)
                except Exception as exc:
                    if not is_connection_error(exc):
                        self._discard(entry)
                        self.rejected += 1
                        logger.error(
                            "%s replay: the store refused a buffered write (%s: %s); "
                            "it was dropped (%d rejected so far)",
                            self.label,
                            type(exc).__name__,
                            exc,
                            self.rejected,
                        )
                        continue
                    logger.error(
                        "%s replay stopped after %d of %d writes: %s; %d still buffered",
                        self.label,
                        done,
                        total,
                        exc,
                        len(self.items),
                    )
                    return
                self._discard(entry)
                self.replayed += 1
                done += 1
            if total:
                logger.warning(
                    "%s recovered: replayed %d of %d buffered writes",
                    self.label,
                    done,
                    total,
                )
            self._announced = False


def _mode(breaker: CircuitBreaker) -> str:
    return "live" if breaker.state == "closed" else "buffering"


class ResilientPool:
    """An asyncpg pool behind a circuit breaker: buffered writes, strict reads."""

    def __init__(
        self,
        pool: Any,
        breaker: Optional[CircuitBreaker] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        settings: Optional[Settings] = None,
    ) -> None:
        if pool is None:
            raise ValueError(
                "ResilientPool wraps a real pool; a missing store is the "
                "in-memory implementation, not a wrapper around None"
            )
        self._pool = pool
        self._settings = settings
        self._breaker = breaker or CircuitBreaker(name="postgresql")
        self._buffer = _WriteBuffer("PostgreSQL", buffer_size)
        self._last_error: Optional[str] = None
        self._breadcrumb_failures = 0
        self._breaker.add_recovery_listener(self._replay)

    # -- plumbing -----------------------------------------------------------

    async def _execute_entry(self, entry: Dict[str, Any]) -> Any:
        method = "fetchval" if entry["method"] == "execute_returning" else "execute"
        return await getattr(self._pool, method)(
            entry["query"], *entry["args"], **entry["kwargs"]
        )

    async def _replay(self) -> None:
        await self._buffer.replay(self._execute_entry, _postgres_connection_error)

    async def _drain(self) -> None:
        """Replay older writes before a new call reaches the store."""
        if not self._buffer.items:
            return
        await self._replay()
        if self._buffer.items:
            raise StoreUnavailable("older buffered writes could not be replayed yet")

    def _track(self, exc: BaseException) -> None:
        self._last_error = str(exc)

    async def _write(
        self, method: str, query: str, args: Any, kwargs: Dict[str, Any]
    ) -> Any:
        entry = {"method": method, "query": query, "args": args, "kwargs": kwargs}

        async def operation() -> Any:
            try:
                await self._drain()
                result = await self._execute_entry(entry)
            except Exception as exc:
                self._track(exc)
                if _postgres_connection_error(exc):
                    raise
                return _Rejected(exc)
            self._last_error = None
            return result

        async def fallback() -> Any:
            self._buffer.add({**entry, "time": _utc_now()})
            written = _breadcrumb(
                self._settings,
                "postgresql.jsonl",
                {"time": _utc_now(), "method": method, "query": query, "args": args},
            )
            if written is False:
                self._breadcrumb_failures += 1
            return BUFFERED

        return _settle(await self._breaker.call(operation, fallback))

    async def _read(
        self, method: str, query: str, args: Any, kwargs: Dict[str, Any]
    ) -> Any:
        async def operation() -> Any:
            try:
                await self._drain()
                result = await getattr(self._pool, method)(query, *args, **kwargs)
            except Exception as exc:
                self._track(exc)
                if _postgres_connection_error(exc):
                    raise
                return _Rejected(exc)
            self._last_error = None
            return result

        async def fallback() -> Any:
            raise StoreUnavailable(
                f"PostgreSQL is unavailable (circuit {self._breaker.state}); "
                "reads never answer from a stand-in"
            )

        return _settle(await self._breaker.call(operation, fallback))

    # -- writes -------------------------------------------------------------

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Run a statement; buffered and ``BUFFERED`` while unavailable."""
        return await self._write("execute", query, args, kwargs)

    async def execute_returning(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Run a statement and return its first value (for ``RETURNING``).

        While unavailable the statement is buffered and ``BUFFERED`` comes back
        in place of the value; it is falsy, so a caller reading an id can tell.
        """
        return await self._write("execute_returning", query, args, kwargs)

    # -- reads (raise StoreUnavailable when the store is unavailable) -------

    async def fetch(self, query: str, *args: Any, **kwargs: Any) -> Any:
        return await self._read("fetch", query, args, kwargs)

    async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> Any:
        return await self._read("fetchrow", query, args, kwargs)

    async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
        return await self._read("fetchval", query, args, kwargs)

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        """Return the pool's connection context manager, or raise when unavailable.

        Work done on an acquired connection bypasses the breaker and the buffer.
        """
        if self._breaker.state != "closed":
            raise StoreUnavailable(
                f"PostgreSQL circuit is {self._breaker.state}; cannot acquire a "
                "connection"
            )
        return self._pool.acquire(*args, **kwargs)

    # -- state --------------------------------------------------------------

    def is_available(self) -> bool:
        """True when a call would reach PostgreSQL (the breaker is closed)."""
        return self._breaker.state == "closed"

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    @property
    def circuit_stats(self) -> Dict[str, Any]:
        return self._breaker.stats

    async def close(self) -> None:
        """Close the pool; safe whether the pool's close is async or not."""
        await _maybe_await(self._pool.close())

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "postgresql",
            "available": self.is_available(),
            "last_error": self._last_error,
            "mode": _mode(self._breaker),
            "circuit": self._breaker.state,
            "buffered": len(self._buffer.items),
            "dropped": self._buffer.dropped,
            "replayed": self._buffer.replayed,
            "rejected": self._buffer.rejected,
            "breadcrumb_failures": self._breadcrumb_failures,
        }


class ResilientMongoStore:
    """A ``ConversationTurnStore`` behind a circuit breaker.

    Turn, session-start and session-end writes buffer in order and replay in
    order. A buffered turn or session start returns ``BUFFERED`` (falsy) in
    place of the id it would have returned. Reads raise ``StoreUnavailable``.
    The buffered turns are private conversation text.
    """

    def __init__(
        self,
        store: ConversationTurnStore,
        breaker: Optional[CircuitBreaker] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        settings: Optional[Settings] = None,
    ) -> None:
        if store is None:
            raise ValueError(
                "ResilientMongoStore wraps a real store; a missing store is the "
                "in-memory implementation, not a wrapper around None"
            )
        self._store = store
        self._settings = settings
        self._breaker = breaker or CircuitBreaker(name="mongodb")
        self._buffer = _WriteBuffer("MongoDB", buffer_size)
        self._last_error: Optional[str] = None
        self._breadcrumb_failures = 0
        self._breaker.add_recovery_listener(self._replay)

    async def _apply(self, entry: Dict[str, Any]) -> Any:
        kind, kwargs = entry["kind"], entry["kwargs"]
        if kind == "turn":
            return await self._store.store_conversation_turn(**kwargs)
        if kind == "start":
            return await self._store.start_conversation_session(**kwargs)
        return await self._store.end_conversation_session(**kwargs)

    async def _replay(self) -> None:
        await self._buffer.replay(self._apply, _mongo_connection_error)

    async def _drain(self) -> None:
        if not self._buffer.items:
            return
        await self._replay()
        if self._buffer.items:
            raise StoreUnavailable("older buffered writes could not be replayed yet")

    async def _write(self, kind: str, kwargs: Dict[str, Any]) -> Any:
        entry = {"kind": kind, "kwargs": kwargs}

        async def operation() -> Any:
            try:
                await self._drain()
                result = await self._apply(entry)
            except Exception as exc:
                self._last_error = str(exc)
                if _mongo_connection_error(exc):
                    raise
                return _Rejected(exc)
            self._last_error = None
            return result

        async def fallback() -> Any:
            self._buffer.add({**copy.deepcopy(entry), "time": _utc_now()})
            written = _breadcrumb(
                self._settings,
                "mongodb_turns.jsonl",
                {"time": _utc_now(), "kind": kind, **kwargs},
            )
            if written is False:
                self._breadcrumb_failures += 1
            return None if kind == "end" else BUFFERED

        return _settle(await self._breaker.call(operation, fallback))

    async def store_conversation_turn(self, **kwargs: Any) -> Any:
        return await self._write("turn", kwargs)

    async def start_conversation_session(
        self, conversation_id: str, primary_participant: str, **kwargs: Any
    ) -> Any:
        return await self._write(
            "start",
            {
                "conversation_id": conversation_id,
                "primary_participant": primary_participant,
                **kwargs,
            },
        )

    async def end_conversation_session(
        self, conversation_id: str, **kwargs: Any
    ) -> None:
        await self._write("end", {"conversation_id": conversation_id, **kwargs})

    async def get_conversation_turns(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        async def operation() -> List[Dict[str, Any]]:
            try:
                await self._drain()
                result = await self._store.get_conversation_turns(
                    conversation_id, limit
                )
            except Exception as exc:
                self._last_error = str(exc)
                if _mongo_connection_error(exc):
                    raise
                return _Rejected(exc)
            self._last_error = None
            return result

        async def fallback() -> List[Dict[str, Any]]:
            raise StoreUnavailable(
                f"MongoDB is unavailable (circuit {self._breaker.state}); "
                "reads never answer from a stand-in"
            )

        return _settle(await self._breaker.call(operation, fallback))

    def is_available(self) -> bool:
        return self._breaker.state == "closed"

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    @property
    def circuit_stats(self) -> Dict[str, Any]:
        return self._breaker.stats

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "mongodb",
            "available": self.is_available(),
            "last_error": self._last_error,
            "mode": _mode(self._breaker),
            "circuit": self._breaker.state,
            "buffered": len(self._buffer.items),
            "dropped": self._buffer.dropped,
            "replayed": self._buffer.replayed,
            "rejected": self._buffer.rejected,
            "breadcrumb_failures": self._breadcrumb_failures,
        }


class ResilientRedisManager:
    """A ``WorkingMemory`` behind a circuit breaker with an outage fallback.

    While Redis is unavailable, writes land in process memory and reads serve
    only those writes (a session with nothing written during the outage reads
    as empty or None). The fallback buffer, emotional state and session context
    replay into Redis on recovery, in order, together with the clears and
    session deletes issued during the outage; the cache is not replayed. The
    ``record``, ``get_averages`` and ``sample_count`` calls of the emotion
    averages provider pass through the breaker and raise ``StoreUnavailable``
    when Redis is unavailable (the emotion extractor then applies its prior).
    """

    def __init__(
        self,
        manager: WorkingMemory,
        breaker: Optional[CircuitBreaker] = None,
        settings: Optional[Settings] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if manager is None:
            raise ValueError(
                "ResilientRedisManager wraps a real manager; a missing store is "
                "the in-memory implementation, not a wrapper around None"
            )
        self._manager = manager
        self._settings = settings
        self._breaker = breaker or CircuitBreaker(name="redis")
        self._last_error: Optional[str] = None
        self._served = 0
        self._buffer_size = int(buffer_size)
        self._log: Deque[Dict[str, Any]] = deque()
        self._dropped = 0
        self._rejected = 0
        self._messages: Dict[str, Deque[Dict[str, Any]]] = {}
        self._emotions: Dict[str, Dict[str, Any]] = {}
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, Any] = {}
        self._replay_lock = asyncio.Lock()
        self._replayed = 0
        self._breaker.add_recovery_listener(self._replay)

    # -- plumbing -----------------------------------------------------------

    def _capacity(self) -> int:
        active = self._settings if self._settings is not None else get_settings()
        return clamp_working_capacity(active.WORKING_MEMORY_CAPACITY)

    def _hold(self, entry: Dict[str, Any]) -> None:
        """Keep an outage write for replay; a full buffer drops the oldest."""
        if len(self._log) >= self._buffer_size:
            self._log.popleft()
            self._dropped += 1
            logger.error(
                "Redis outage: write buffer full (%d); the oldest write was dropped",
                self._buffer_size,
            )
        self._log.append(entry)

    async def _guard(
        self,
        operation: Callable[[], Awaitable[Any]],
        fallback: Callable[[], Awaitable[Any]],
    ) -> Any:
        async def tracked() -> Any:
            try:
                result = await operation()
            except Exception as exc:
                self._last_error = str(exc)
                if _redis_connection_error(exc):
                    raise
                return _Rejected(exc)
            self._last_error = None
            return result

        return _settle(await self._breaker.call(tracked, fallback))

    async def _apply(self, entry: Dict[str, Any]) -> None:
        kind = entry["kind"]
        if kind == "push":
            await self._manager.add_to_working_memory(
                entry["session"], copy.deepcopy(entry["message"])
            )
        elif kind == "emotion":
            await self._manager.update_emotional_state(
                entry["session"], copy.deepcopy(entry["state"])
            )
        elif kind == "clear":
            await self._manager.clear_working_memory(entry["session"])
        elif kind == "delete":
            await self._manager.delete_session(entry["session"])
        else:
            await self._manager.set_session_context(
                entry["session"], copy.deepcopy(entry["context"]), entry["ttl"]
            )

    def _discard(self, entry: Dict[str, Any]) -> None:
        if self._log and self._log[0] is entry:
            self._log.popleft()

    async def _replay(self) -> None:
        async with self._replay_lock:
            while self._log:
                entry = self._log[0]
                try:
                    await self._apply(entry)
                except Exception as exc:
                    if not _redis_connection_error(exc):
                        self._discard(entry)
                        self._rejected += 1
                        logger.error(
                            "Redis replay: the store refused a held write (%s: %s); "
                            "it was dropped (%d rejected so far)",
                            type(exc).__name__,
                            exc,
                            self._rejected,
                        )
                        continue
                    logger.error(
                        "Redis replay stopped: %s; %d writes still held",
                        exc,
                        len(self._log),
                    )
                    return
                self._discard(entry)
                self._replayed += 1
            self._messages.clear()
            self._emotions.clear()
            self._contexts.clear()

    async def _drain(self) -> None:
        if self._log:
            await self._replay()
            if self._log:
                raise StoreUnavailable("fallback writes could not be replayed yet")

    # -- WorkingMemory ------------------------------------------------------

    async def add_to_working_memory(
        self, session_id: str, message: Dict[str, Any]
    ) -> int:
        async def operation() -> int:
            await self._drain()
            return await self._manager.add_to_working_memory(session_id, message)

        async def fallback() -> int:
            stored = copy.deepcopy(message)
            stored.setdefault("timestamp", _utc_now())
            buffer = self._messages.setdefault(
                session_id, deque(maxlen=self._capacity())
            )
            buffer.append(stored)
            self._hold({"kind": "push", "session": session_id, "message": stored})
            return len(buffer)

        return await self._guard(operation, fallback)

    async def get_working_memory(self, session_id: str) -> List[Dict[str, Any]]:
        async def fallback() -> List[Dict[str, Any]]:
            self._served += 1
            return copy.deepcopy(list(self._messages.get(session_id, [])))

        async def operation() -> List[Dict[str, Any]]:
            await self._drain()
            return await self._manager.get_working_memory(session_id)

        return await self._guard(operation, fallback)

    async def clear_working_memory(self, session_id: str) -> None:
        async def operation() -> None:
            await self._drain()
            await self._manager.clear_working_memory(session_id)

        async def fallback() -> None:
            self._messages.pop(session_id, None)
            self._hold({"kind": "clear", "session": session_id})

        await self._guard(operation, fallback)

    async def update_emotional_state(
        self, session_id: str, state: Dict[str, Any]
    ) -> None:
        async def operation() -> None:
            await self._drain()
            await self._manager.update_emotional_state(session_id, state)

        async def fallback() -> None:
            stored = copy.deepcopy(state)
            self._emotions[session_id] = stored
            self._hold({"kind": "emotion", "session": session_id, "state": stored})

        await self._guard(operation, fallback)

    async def get_emotional_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        async def fallback() -> Optional[Dict[str, Any]]:
            self._served += 1
            return copy.deepcopy(self._emotions.get(session_id))

        async def operation() -> Optional[Dict[str, Any]]:
            await self._drain()
            return await self._manager.get_emotional_state(session_id)

        return await self._guard(operation, fallback)

    async def set_session_context(
        self, session_id: str, context: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        async def operation() -> None:
            await self._drain()
            await self._manager.set_session_context(session_id, context, ttl)

        async def fallback() -> None:
            stored = copy.deepcopy(context)
            self._contexts[session_id] = stored
            self._hold(
                {
                    "kind": "context",
                    "session": session_id,
                    "context": stored,
                    "ttl": ttl,
                }
            )

        await self._guard(operation, fallback)

    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        async def fallback() -> Optional[Dict[str, Any]]:
            self._served += 1
            return copy.deepcopy(self._contexts.get(session_id))

        async def operation() -> Optional[Dict[str, Any]]:
            await self._drain()
            return await self._manager.get_session_context(session_id)

        return await self._guard(operation, fallback)

    async def delete_session(self, session_id: str) -> None:
        async def operation() -> None:
            await self._drain()
            await self._manager.delete_session(session_id)

        async def fallback() -> None:
            self._messages.pop(session_id, None)
            self._emotions.pop(session_id, None)
            self._contexts.pop(session_id, None)
            self._hold({"kind": "delete", "session": session_id})

        await self._guard(operation, fallback)

    async def cache_set(self, key: str, value: Any, ttl: int = 30) -> None:
        async def fallback() -> None:
            if len(self._cache) >= FALLBACK_CACHE_LIMIT:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (copy.deepcopy(value), time.monotonic() + ttl)

        await self._guard(lambda: self._manager.cache_set(key, value, ttl), fallback)

    async def cache_get(self, key: str) -> Optional[Any]:
        async def fallback() -> Optional[Any]:
            self._served += 1
            entry = self._cache.get(key)
            if entry is None or time.monotonic() >= entry[1]:
                self._cache.pop(key, None)
                return None
            return copy.deepcopy(entry[0])

        return await self._guard(lambda: self._manager.cache_get(key), fallback)

    async def get_metrics(self) -> Dict[str, Any]:
        async def fallback() -> Dict[str, Any]:
            return {
                "backend": "fallback",
                "served_from_fallback": self._served,
                "held_writes": len(self._log),
            }

        return await self._guard(self._manager.get_metrics, fallback)

    async def disconnect(self) -> None:
        """Disconnect the wrapped manager; never raises."""
        try:
            await _maybe_await(self._manager.disconnect())
        except Exception as exc:
            logger.warning("redis disconnect failed: %s", exc)

    # -- emotion averages (no fallback: an honest failure) -------------------

    async def _strict(self, name: str, *args: Any, **kwargs: Any) -> Any:
        async def operation() -> Any:
            return await getattr(self._manager, name)(*args, **kwargs)

        async def fallback() -> Any:
            raise StoreUnavailable(
                f"Redis is unavailable (circuit {self._breaker.state}); "
                f"{name} has no fallback"
            )

        return await self._guard(operation, fallback)

    async def record(self, session_id: str, sample: Dict[str, float]) -> None:
        await self._strict("record", session_id, sample)

    async def get_averages(
        self, session_id: str, *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, float]]:
        return await self._strict("get_averages", session_id, *args, **kwargs)

    async def sample_count(self, session_id: str) -> int:
        return await self._strict("sample_count", session_id)

    # -- state --------------------------------------------------------------

    def is_available(self) -> bool:
        return self._breaker.state == "closed"

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    @property
    def circuit_stats(self) -> Dict[str, Any]:
        return self._breaker.stats

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "redis",
            "available": self.is_available(),
            "last_error": self._last_error,
            "mode": _mode(self._breaker),
            "circuit": self._breaker.state,
            "buffered": len(self._log),
            "dropped": self._dropped,
            "replayed": self._replayed,
            "rejected": self._rejected,
            "served_from_fallback": self._served,
        }
