"""
Redis Working Memory
====================

``RedisManager`` keeps a session's fast, short-lived state in Redis and
implements the ``WorkingMemory`` protocol and the emotion extractor's
``EmotionAveragesProvider`` (``mtc.consciousness.emotion_extractor``).
Everything it stores is private conversation state and lives only in the
operator's Redis.

Config keys read at call time: ``REDIS_KEY_PREFIX`` (every key starts with it),
``WORKING_MEMORY_CAPACITY`` (clamped to five to nine messages) and, through the
client factory, ``REDIS_HOST``, ``REDIS_PORT``, ``REDIS_DB``,
``REDIS_PASSWORD``, ``REDIS_MAX_CONNECTIONS`` and ``REDIS_SOCKET_TIMEOUT``.
The ``redis`` client package is imported only when a client is built, through
``mtc.memory.clients``; a missing package raises an ImportError naming the
``redis`` extra.

Key families (see ``KEY_FAMILIES``, the source for the operator documentation):

- ``wm:conversation:{session}``: list, the newest messages, 1800 s expiry
  refreshed on each push.
- ``emotion:current:{session}``: JSON string, 14400 s expiry.
- ``emotion:history:{session}``: stream of raw-dimension samples capped at
  ``emotion_history_maxlen`` entries, no expiry.
- ``session:{id}``: JSON string, optional expiry.
- ``cache:{key}``: JSON string, 30 s default expiry.

Some keys carry no expiry, so the server should evict volatile keys under
memory pressure (``maxmemory-policy volatile-lru``); ``EVICTION_RECOMMENDATION``
records it.

A server that cannot be reached raises ``StoreUnavailable`` from the operation
that needed it; no operation answers from a stand-in. ``ResilientRedisManager``
in ``mtc.memory.resilient_wrappers`` is the opt-in layer that serves what was
written during an outage. Cold-start averages are not invented: with no samples
``get_averages`` returns None and the emotion extractor applies its own prior.
"""

import copy
import inspect
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from mtc.core.config import Settings, get_settings
from mtc.memory.clients import create_redis_client
from mtc.memory.protocols import (
    CACHE_TTL_S,
    EMOTION_STATE_TTL_S,
    WORKING_MEMORY_TTL_S,
    StoreUnavailable,
    clamp_working_capacity,
)

logger = logging.getLogger(__name__)

EVICTION_RECOMMENDATION = "volatile-lru"
DEFAULT_EMOTION_HISTORY_MAXLEN = 1000


class KeyFamily(NamedTuple):
    """One Redis key family under ``REDIS_KEY_PREFIX``."""

    pattern: str
    redis_type: str
    ttl_seconds: Optional[int]
    purpose: str


KEY_FAMILIES: Tuple[KeyFamily, ...] = (
    KeyFamily(
        "wm:conversation:{session}",
        "list",
        WORKING_MEMORY_TTL_S,
        "newest conversation messages, expiry refreshed on each push",
    ),
    KeyFamily(
        "emotion:current:{session}",
        "string (JSON)",
        EMOTION_STATE_TTL_S,
        "current emotional state of the session",
    ),
    KeyFamily(
        "emotion:history:{session}",
        "stream",
        None,
        "raw emotion-dimension samples, capped by length, no expiry",
    ),
    KeyFamily(
        "session:{id}",
        "string (JSON)",
        None,
        "session context, expiry only when the caller passes a ttl",
    ),
    KeyFamily(
        "cache:{key}",
        "string (JSON)",
        CACHE_TTL_S,
        "short-lived cached values",
    ),
)

_WM = "wm:conversation:"
_EMOTION = "emotion:current:"
_HISTORY = "emotion:history:"
_SESSION = "session:"
_CACHE = "cache:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class RedisManager:
    """Working memory, emotional state, session context and cache in Redis."""

    def __init__(
        self,
        client: Any = None,
        settings: Optional[Settings] = None,
        working_capacity: Optional[int] = None,
        emotion_history_maxlen: int = DEFAULT_EMOTION_HISTORY_MAXLEN,
    ) -> None:
        if emotion_history_maxlen < 1:
            raise ValueError("emotion_history_maxlen must be at least 1")
        self._client = client
        self._owns_client = client is None
        self._settings = settings
        self._working_capacity = working_capacity
        self.emotion_history_maxlen = int(emotion_history_maxlen)
        self._last_error: Optional[str] = None
        self._warned_capacity: Optional[int] = None
        self._metrics = {
            "total_reads": 0,
            "total_writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "RedisManager":
        """Build a manager that creates its client from Settings on first use."""
        return cls(settings=settings)

    # ------------------------------------------------------------------ setup

    @property
    def settings(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    @property
    def capacity(self) -> int:
        """Working-memory capacity, clamped to five to nine, read at call time."""
        wanted = (
            self._working_capacity
            if self._working_capacity is not None
            else self.settings.WORKING_MEMORY_CAPACITY
        )
        clamped = clamp_working_capacity(wanted)
        if clamped != wanted and self._warned_capacity != wanted:
            self._warned_capacity = wanted
            logger.warning(
                "working memory capacity %s outside 5-9; using %s", wanted, clamped
            )
        return clamped

    def _key(self, family: str, ident: str) -> str:
        return f"{self.settings.REDIS_KEY_PREFIX}{family}{ident}"

    async def connect(self) -> bool:
        """Build the client when none was given and ping the server.

        Returns True when the server answered. A missing ``redis`` package
        raises an ImportError naming the extra; an unreachable server raises
        ``StoreUnavailable``.
        """
        if self._client is None:
            self._client = create_redis_client(self.settings)
            self._owns_client = True
        try:
            await self._client.ping()
        except Exception as exc:
            self._record_error(exc)
            raise StoreUnavailable(f"Redis did not answer: {exc}") from exc
        self._last_error = None
        return True

    async def _ensure_client(self) -> Any:
        if self._client is None:
            await self.connect()
        return self._client

    def _record_error(self, exc: BaseException) -> None:
        self._metrics["errors"] += 1
        self._last_error = str(exc)

    async def _call(self, operation: str, action: Any) -> Any:
        """Run ``action(client)``; a client failure becomes StoreUnavailable."""
        client = await self._ensure_client()
        try:
            return await action(client)
        except StoreUnavailable:
            raise
        except Exception as exc:
            self._record_error(exc)
            logger.error("redis %s failed: %s", operation, exc)
            raise StoreUnavailable(f"Redis {operation} failed: {exc}") from exc

    def _count_read(self, hit: bool) -> None:
        self._metrics["total_reads"] += 1
        self._metrics["cache_hits" if hit else "cache_misses"] += 1

    async def disconnect(self) -> None:
        """Close the client when this manager created it."""
        client, self._client = self._client, None
        if client is None or not self._owns_client:
            return
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is not None:
            try:
                await _maybe_await(closer())
            except Exception as exc:  # shutdown must not raise
                logger.warning("redis close failed: %s", exc)

    def status(self) -> Dict[str, Any]:
        """Return the backend state; ``available`` is False after a failure."""
        return {
            "backend": "redis",
            "available": self._client is not None and self._last_error is None,
            "connected": self._client is not None,
            "last_error": self._last_error,
            "key_prefix": self.settings.REDIS_KEY_PREFIX,
            "capacity": self.capacity,
        }

    # ----------------------------------------------------------- working memory

    async def add_to_working_memory(
        self, session_id: str, message: Dict[str, Any]
    ) -> int:
        stored = copy.deepcopy(message)
        stored.setdefault("timestamp", _now_iso())
        payload = json.dumps(stored)
        key = self._key(_WM, session_id)
        capacity = self.capacity

        async def action(client: Any) -> int:
            pipe = client.pipeline(transaction=True)
            pipe.rpush(key, payload)
            pipe.ltrim(key, -capacity, -1)
            pipe.expire(key, WORKING_MEMORY_TTL_S)
            results = await pipe.execute()
            return min(int(results[0]), capacity)

        size = await self._call("push", action)
        self._metrics["total_writes"] += 1
        return size

    async def get_working_memory(self, session_id: str) -> List[Dict[str, Any]]:
        key = self._key(_WM, session_id)
        raw = await self._call("read", lambda c: c.lrange(key, 0, -1))
        self._count_read(bool(raw))
        return [json.loads(item) for item in raw]

    async def clear_working_memory(self, session_id: str) -> None:
        key = self._key(_WM, session_id)
        await self._call("clear", lambda c: c.delete(key))

    # ---------------------------------------------------------- emotional state

    async def update_emotional_state(
        self, session_id: str, state: Dict[str, Any]
    ) -> None:
        stored = copy.deepcopy(state)
        stored["updated_at"] = _now_iso()
        payload = json.dumps(stored)
        key = self._key(_EMOTION, session_id)
        await self._call(
            "set emotional state",
            lambda c: c.set(key, payload, ex=EMOTION_STATE_TTL_S),
        )
        self._metrics["total_writes"] += 1

    async def get_emotional_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        key = self._key(_EMOTION, session_id)
        raw = await self._call("get emotional state", lambda c: c.get(key))
        self._count_read(raw is not None)
        return json.loads(raw) if raw is not None else None

    # --------------------------------------------------- emotion history stream

    async def record(self, session_id: str, sample: Dict[str, float]) -> None:
        """Append one raw-dimension sample to the session's history stream.

        Dimensions that are None or not finite are dropped; a sample with no
        usable dimension is not a measurement and is not stored.
        """
        usable = {
            str(dim): float(value)
            for dim, value in sample.items()
            if value is not None and math.isfinite(float(value))
        }
        if not usable:
            return
        key = self._key(_HISTORY, session_id)
        payload = json.dumps(usable)
        maxlen = self.emotion_history_maxlen
        await self._call(
            "append emotion history",
            lambda c: c.xadd(key, {"data": payload}, maxlen=maxlen, approximate=False),
        )
        self._metrics["total_writes"] += 1

    async def get_emotion_history(
        self, session_id: str, count: int = 100
    ) -> List[Dict[str, float]]:
        """Return the newest ``count`` samples, oldest first."""
        key = self._key(_HISTORY, session_id)
        entries = await self._call(
            "read emotion history", lambda c: c.xrevrange(key, count=count)
        )
        self._metrics["total_reads"] += 1
        samples: List[Dict[str, float]] = []
        for _entry_id, fields in reversed(list(entries)):
            data = fields.get("data") if "data" in fields else fields.get(b"data")
            if data is None:
                continue
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            samples.append(json.loads(data))
        return samples

    async def get_averages(
        self, session_id: str, window: int = 50
    ) -> Optional[Dict[str, float]]:
        """Flat ``{dim: mean, dim_std: spread}`` over the newest ``window`` samples.

        The keys and values match ``InMemoryEmotionAverages`` (population
        spread, zero for a single sample). None when no sample was recorded.
        """
        samples = await self.get_emotion_history(session_id, count=window)
        columns: Dict[str, List[float]] = {}
        for sample in samples:
            for dim, value in sample.items():
                columns.setdefault(dim, []).append(float(value))
        if not columns:
            return None
        out: Dict[str, float] = {}
        for dim, values in columns.items():
            mean = math.fsum(values) / len(values)
            out[dim] = mean
            out[f"{dim}_std"] = (
                math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
                if len(values) > 1
                else 0.0
            )
        return out

    async def sample_count(self, session_id: str) -> int:
        """Number of samples in the history stream (capped at its maximum)."""
        key = self._key(_HISTORY, session_id)
        return int(await self._call("count emotion history", lambda c: c.xlen(key)))

    # ---------------------------------------------------------- session context

    async def set_session_context(
        self, session_id: str, context: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        payload = json.dumps(context)
        key = self._key(_SESSION, session_id)
        if ttl:
            await self._call("set context", lambda c: c.set(key, payload, ex=ttl))
        else:
            await self._call("set context", lambda c: c.set(key, payload))
        self._metrics["total_writes"] += 1

    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        key = self._key(_SESSION, session_id)
        raw = await self._call("get context", lambda c: c.get(key))
        self._count_read(raw is not None)
        return json.loads(raw) if raw is not None else None

    async def delete_session(self, session_id: str) -> None:
        """Delete the session's buffer, emotional state, history and context."""
        keys = [
            self._key(_WM, session_id),
            self._key(_EMOTION, session_id),
            self._key(_HISTORY, session_id),
            self._key(_SESSION, session_id),
        ]
        await self._call("delete session", lambda c: c.delete(*keys))

    # -------------------------------------------------------------------- cache

    async def cache_set(self, key: str, value: Any, ttl: int = CACHE_TTL_S) -> None:
        payload = json.dumps(value)
        cache_key = self._key(_CACHE, key)
        await self._call("cache set", lambda c: c.set(cache_key, payload, ex=ttl))
        self._metrics["total_writes"] += 1

    async def cache_get(self, key: str) -> Optional[Any]:
        cache_key = self._key(_CACHE, key)
        raw = await self._call("cache get", lambda c: c.get(cache_key))
        self._count_read(raw is not None)
        return json.loads(raw) if raw is not None else None

    # ------------------------------------------------------------------ metrics

    async def get_metrics(self) -> Dict[str, Any]:
        """Counters plus the server's memory figures (None when unmeasured).

        ``cache_hit_rate_percent`` is None before the first cache read.
        """
        hits = self._metrics["cache_hits"]
        accesses = hits + self._metrics["cache_misses"]
        used: Optional[float] = None
        peak: Optional[float] = None
        if self._client is not None:
            try:
                info = await self._client.info("memory")
                used = float(info["used_memory"]) / 1024 / 1024
                peak = float(info["used_memory_peak"]) / 1024 / 1024
            except Exception as exc:
                logger.warning("redis memory info unavailable: %s", exc)
        return {
            **self._metrics,
            # No rate is measured before the first cache read.
            "cache_hit_rate_percent": (hits / accesses * 100.0) if accesses else None,
            "redis_memory_mb": used,
            "redis_peak_memory_mb": peak,
        }


def get_redis_manager(settings: Optional[Settings] = None) -> RedisManager:
    """Build a manager over Settings; it connects on first use.

    Each call returns a new manager; the caller keeps the instance it shares.
    """
    return RedisManager.from_settings(settings)
