"""
PostgreSQL Research Stores
==========================

``PostgresSnapshotStore`` implements ``SnapshotStore``
(``mtc.memory.protocols``) over the research tables ``consciousness_snapshots``
and ``consciousness_events``. PostgreSQL holds the aggregate assessment
snapshots and their events; per-module snapshots live in MongoDB. The store
passes the contract tests in ``tests/memory/test_store_contracts.py``.

The pool is an asyncpg pool from ``mtc.memory.clients.create_postgres_pool``
(JSON and JSONB codecs installed per connection, so the JSONB columns take and
return dicts) or a ``ResilientPool`` around one. Reads through an unavailable
``ResilientPool`` raise ``StoreUnavailable``; a write it buffers returns no id,
so ``add_snapshot`` and ``add_event`` raise ``StoreUnavailable`` rather than
invent one. Timestamps are ``TIMESTAMPTZ`` and come back timezone-aware; a naive
timestamp raises ValueError before anything is sent. A null ``overall_score``
or ``phi_value`` stays null.

Schema artifact: ``RESEARCH_SCHEMA_SQL``, applied idempotently by
``ensure_schema()`` as a safety net. ``migrations/postgres/002_research_schema.sql``
is the canonical schema and is generated from this constant by
``python3 -m mtc.db.migrations --write``. Host, port, database and
credentials come from Settings through the pool factory; this module holds none.
The store never closes the pool: the pool is shared, and its owner closes it.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, List, Mapping, Optional

from mtc.memory.clients import apply_schema
from mtc.memory.protocols import SnapshotEvent, SnapshotRecord, StoreUnavailable

logger = logging.getLogger(__name__)

RESEARCH_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS consciousness_snapshots (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        trigger VARCHAR(50) NOT NULL,
        overall_score DOUBLE PRECISION NULL,
        passing_count INTEGER,
        phi_value DOUBLE PRECISION NULL,
        theory_scores JSONB,
        indicator_results JSONB,
        processing_time_ms DOUBLE PRECISION,
        status VARCHAR(20) NOT NULL,
        coverage DOUBLE PRECISION,
        metadata JSONB
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS consciousness_snapshots_timestamp_idx
    ON consciousness_snapshots (timestamp)
    """,
    """
    CREATE TABLE IF NOT EXISTS consciousness_events (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        description TEXT,
        snapshot_id BIGINT REFERENCES consciousness_snapshots(id) ON DELETE SET NULL,
        old_value DOUBLE PRECISION,
        new_value DOUBLE PRECISION
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS consciousness_events_timestamp_idx
    ON consciousness_events (timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS consciousness_events_type_timestamp_idx
    ON consciousness_events (event_type, timestamp DESC)
    """,
)

_SNAPSHOT_COLUMNS = (
    "id, timestamp, trigger, overall_score, passing_count, phi_value, "
    "theory_scores, indicator_results, processing_time_ms, status, coverage, "
    "metadata"
)
_EVENT_COLUMNS = (
    "id, timestamp, event_type, description, snapshot_id, old_value, new_value"
)


# Driver exception class names (asyncpg) that mean the server was not reached.
# They are matched by name over the class hierarchy so this module never
# imports the driver. InterfaceError is not among them: asyncpg raises it and
# its subclasses (the client-side DataError, ClientConfigurationError,
# UnsupportedClientFeatureError) for the caller's own mistakes, such as a wrong
# argument count. Only an InterfaceError of exactly that class whose message
# says the connection or pool is closed or closing is an outage (_CLOSED_TEXT).
_UNREACHABLE_NAMES = frozenset(
    {
        "ConnectionDoesNotExistError",
        "PostgresConnectionError",
        "ConnectionFailureError",
        "ConnectionRejectionError",
        "CannotConnectNowError",
        "TooManyConnectionsError",
        "ClientCannotConnectError",
    }
)


_INTERFACE_ERROR = "InterfaceError"
_CLOSED_TEXT = re.compile(r"\b(?:closed|closing)\b", re.IGNORECASE)


def is_unreachable(exc: BaseException) -> bool:
    """True when an exception means PostgreSQL could not be reached.

    OSError, a timeout and the connection-class driver errors are outages. An
    ``InterfaceError`` is one only when its own class (not a subclass) is named
    ``InterfaceError`` and its message says closed or closing (a closed
    connection or pool); every other client-side error is the caller's.
    """
    if isinstance(exc, (OSError, asyncio.TimeoutError)):
        return True
    if type(exc).__name__ == _INTERFACE_ERROR:
        return bool(_CLOSED_TEXT.search(str(exc)))
    return any(cls.__name__ in _UNREACHABLE_NAMES for cls in type(exc).__mro__)


async def guard_unreachable(operation: Awaitable[Any]) -> Any:
    """Await ``operation``; a connection failure becomes ``StoreUnavailable``.

    Every other error (a constraint violation, a bad statement) propagates.
    """
    try:
        return await operation
    except StoreUnavailable:
        raise
    except Exception as exc:
        if is_unreachable(exc):
            raise StoreUnavailable(
                f"PostgreSQL could not be reached: {type(exc).__name__}: {exc}"
            ) from exc
        raise


class GuardedPool:
    """A pool adapter the PostgreSQL stores use: connection failures surface as
    ``StoreUnavailable`` (the protocol contract) whether the inner pool is a
    plain asyncpg pool or a ``ResilientPool``.

    ``execute_returning`` uses the inner pool's own (the buffering one of a
    ``ResilientPool``) when it has one, and ``fetchval`` otherwise. The adapter
    records the last unreachable error it raised and clears it on the next
    call that reaches the server, so ``status()`` reports an unreachable plain
    pool as unavailable instead of assuming it is up.
    """

    def __init__(self, pool: Any) -> None:
        self.inner = pool
        self.last_unreachable: Optional[str] = None

    async def guard(self, operation: Awaitable[Any]) -> Any:
        """Await ``operation`` through ``guard_unreachable`` and record the
        outcome for ``status()``."""
        try:
            result = await guard_unreachable(operation)
        except StoreUnavailable as exc:
            self.last_unreachable = str(exc)
            raise
        self.last_unreachable = None
        return result

    async def execute(self, query: str, *args: Any) -> Any:
        return await self.guard(self.inner.execute(query, *args))

    async def execute_returning(self, query: str, *args: Any) -> Any:
        runner = getattr(self.inner, "execute_returning", None)
        if runner is None:
            return await self.guard(self.inner.fetchval(query, *args))
        return await self.guard(runner(query, *args))

    async def fetch(self, query: str, *args: Any) -> Any:
        return await self.guard(self.inner.fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> Any:
        return await self.guard(self.inner.fetchrow(query, *args))

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self.guard(self.inner.fetchval(query, *args))

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.acquire(*args, **kwargs)

    def close(self) -> Any:
        return self.inner.close()

    def status(self) -> Dict[str, Any]:
        """The inner pool's status (when it has one) joined with the last
        unreachable error this adapter saw; never raises."""
        reported: Dict[str, Any] = {
            "backend": "postgres",
            "available": True,
            "last_error": None,
        }
        reader = getattr(self.inner, "status", None)
        if callable(reader):
            try:
                reported.update(dict(reader()))
            except Exception as exc:  # noqa: BLE001 - status never raises
                reported["available"] = False
                reported["last_error"] = f"{type(exc).__name__}: {exc}"
        if self.last_unreachable is not None:
            reported["available"] = False
            reported["last_error"] = self.last_unreachable
        return reported


def pool_status(pool: GuardedPool) -> Dict[str, Any]:
    """``backend``, ``available`` and ``last_error`` of a store over ``pool``."""
    reported = pool.status()
    return {
        "backend": "postgres",
        "available": bool(reported.get("available", False)),
        "last_error": reported.get("last_error"),
    }


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _utc(moment: datetime) -> datetime:
    """Read a stored instant as aware UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _record(row: Mapping[str, Any]) -> SnapshotRecord:
    return SnapshotRecord(
        timestamp=_utc(row["timestamp"]),
        trigger=row["trigger"],
        overall_score=_optional_float(row["overall_score"]),
        passing_count=int(row["passing_count"] or 0),
        phi_value=_optional_float(row["phi_value"]),
        theory_scores=dict(row["theory_scores"] or {}),
        indicator_results=dict(row["indicator_results"] or {}),
        processing_time_ms=_optional_float(row["processing_time_ms"]),
        status=row["status"],
        coverage=_optional_float(row["coverage"]),
        metadata=dict(row["metadata"] or {}),
        id=int(row["id"]),
    )


def _event(row: Mapping[str, Any]) -> SnapshotEvent:
    snapshot_id = row["snapshot_id"]
    return SnapshotEvent(
        timestamp=_utc(row["timestamp"]),
        event_type=row["event_type"],
        description=row["description"] or "",
        snapshot_id=None if snapshot_id is None else int(snapshot_id),
        old_value=_optional_float(row["old_value"]),
        new_value=_optional_float(row["new_value"]),
        id=int(row["id"]),
    )


async def _insert_returning_id(pool: Any, sql: str, *args: Any) -> int:
    """Run an ``INSERT ... RETURNING id`` and return the id.

    A ``ResilientPool`` that buffers the insert returns no id; the store raises
    ``StoreUnavailable`` instead of inventing one (the buffered row is still
    written on replay).
    """
    value = await pool.execute_returning(sql, *args)
    if not isinstance(value, int):
        raise StoreUnavailable(
            "PostgreSQL insert was buffered; no id is available until replay"
        )
    return value


class PostgresSnapshotStore:
    """Assessment snapshots and events in PostgreSQL, newest first."""

    def __init__(self, pool: Any) -> None:
        if pool is None:
            raise ValueError(
                "PostgresSnapshotStore needs a pool; use InMemorySnapshotStore "
                "when no database is configured"
            )
        self._pool = GuardedPool(pool)

    async def ensure_schema(self) -> None:
        """Apply ``RESEARCH_SCHEMA_SQL`` (idempotent safety net)."""
        await self._pool.guard(apply_schema(self._pool, RESEARCH_SCHEMA_SQL))

    async def add_snapshot(self, record: SnapshotRecord) -> int:
        stamp = _aware(record.timestamp)
        return await _insert_returning_id(
            self._pool,
            """
            INSERT INTO consciousness_snapshots (
                timestamp, trigger, overall_score, passing_count, phi_value,
                theory_scores, indicator_results, processing_time_ms, status,
                coverage, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
            """,
            stamp,
            record.trigger,
            record.overall_score,
            int(record.passing_count),
            record.phi_value,
            dict(record.theory_scores or {}),
            dict(record.indicator_results or {}),
            record.processing_time_ms,
            record.status,
            record.coverage,
            dict(record.metadata or {}),
        )

    async def add_event(self, event: SnapshotEvent) -> int:
        stamp = _aware(event.timestamp)
        return await _insert_returning_id(
            self._pool,
            """
            INSERT INTO consciousness_events (
                timestamp, event_type, description, snapshot_id, old_value,
                new_value
            ) VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            stamp,
            event.event_type,
            event.description,
            event.snapshot_id,
            event.old_value,
            event.new_value,
        )

    async def latest_snapshot(self) -> Optional[SnapshotRecord]:
        row = await self._pool.fetchrow(
            f"SELECT {_SNAPSHOT_COLUMNS} FROM consciousness_snapshots "
            "ORDER BY timestamp DESC, id DESC LIMIT 1"
        )
        return _record(row) if row is not None else None

    async def snapshots(
        self, since: Optional[datetime] = None, limit: int = 100
    ) -> List[SnapshotRecord]:
        if limit <= 0:
            return []
        if since is None:
            rows = await self._pool.fetch(
                f"SELECT {_SNAPSHOT_COLUMNS} FROM consciousness_snapshots "
                "ORDER BY timestamp DESC, id DESC LIMIT $1",
                int(limit),
            )
        else:
            rows = await self._pool.fetch(
                f"SELECT {_SNAPSHOT_COLUMNS} FROM consciousness_snapshots "
                "WHERE timestamp >= $1 ORDER BY timestamp DESC, id DESC LIMIT $2",
                _aware(since),
                int(limit),
            )
        return [_record(row) for row in rows]

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[SnapshotEvent]:
        if limit <= 0:
            return []
        clauses: List[str] = []
        args: List[Any] = []
        if since is not None:
            args.append(_aware(since))
            clauses.append(f"timestamp >= ${len(args)}")
        if event_type is not None:
            args.append(event_type)
            clauses.append(f"event_type = ${len(args)}")
        args.append(int(limit))
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rows = await self._pool.fetch(
            f"SELECT {_EVENT_COLUMNS} FROM consciousness_events {where}"
            f"ORDER BY timestamp DESC, id DESC LIMIT ${len(args)}",
            *args,
        )
        return [_event(row) for row in rows]

    async def snapshot_before(self, moment: datetime) -> Optional[SnapshotRecord]:
        row = await self._pool.fetchrow(
            f"SELECT {_SNAPSHOT_COLUMNS} FROM consciousness_snapshots "
            "WHERE timestamp < $1 ORDER BY timestamp DESC, id DESC LIMIT 1",
            _aware(moment),
        )
        return _record(row) if row is not None else None

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error``.

        An unreachable server is reported as unavailable with the error of the
        call that failed, until a later call reaches it.
        """
        return pool_status(self._pool)
