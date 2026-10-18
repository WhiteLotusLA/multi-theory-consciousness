"""
Safety Audit Store (PostgreSQL)
===============================

``PostgresSafetyAuditStore`` implements ``SafetyAuditStore``
(``mtc.memory.protocols``) over the ``safety_audit`` table. The pool is an
asyncpg pool from ``mtc.memory.clients.create_postgres_pool`` (JSON and JSONB
codecs installed per connection, so ``checks_performed`` takes and returns a
dict) or a ``ResilientPool`` around one. Connection failures surface as
``StoreUnavailable``; a write a ``ResilientPool`` buffered returns no id, so
``add_audit`` raises ``StoreUnavailable`` rather than invent one (the buffered
row is still written on replay). The store never closes the pool: the pool is
shared and its owner closes it.

Contract (pinned by ``tests/memory/test_audit_store_contracts.py``): a record
holds the fields of ``SAFETY_AUDIT_FIELDS`` (``action_proposed``,
``safety_level`` and ``decision`` required; a missing or extra field raises
ValueError); a missing ``timestamp`` is the database's current time and a naive
one raises ValueError; ids come back as strings; reads are newest first (later
write first among equal instants, through the ``seq`` column).

Schema artifact: ``SAFETY_SCHEMA_SQL``, applied idempotently by
``ensure_schema()`` as a safety net. The migration files under
``migrations/postgres`` are canonical and are written from this constant.
``gen_random_uuid()`` is part of core PostgreSQL from version 13. Host, port,
database and credentials come from Settings through the pool factory; this
module holds none.

Rows hold the proposed action, which can contain conversation text, and the
scores and actor of the check. They are private data and belong in the
operator's database. The store logs no row content.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

from mtc.memory.clients import apply_schema
from mtc.memory.postgres_stores import GuardedPool, pool_status
from mtc.memory.protocols import StoreUnavailable, normalize_safety_audit

logger = logging.getLogger(__name__)

SAFETY_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS safety_audit (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        seq BIGSERIAL,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
        action_proposed TEXT NOT NULL,
        safety_level VARCHAR(20) NOT NULL,
        checks_performed JSONB,
        decision VARCHAR(20) NOT NULL,
        reasoning TEXT,
        requires_review BOOLEAN DEFAULT FALSE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS safety_audit_timestamp_idx
    ON safety_audit (timestamp DESC, seq DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS safety_audit_review_idx
    ON safety_audit (timestamp DESC, seq DESC) WHERE requires_review
    """,
)

_COLUMNS = (
    "id, timestamp, action_proposed, safety_level, checks_performed, "
    "decision, reasoning, requires_review"
)


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("safety audit timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _row(row: Mapping[str, Any]) -> Dict[str, Any]:
    checks = row["checks_performed"]
    return {
        "id": str(row["id"]),
        "timestamp": _utc(row["timestamp"]),
        "action_proposed": row["action_proposed"],
        "safety_level": row["safety_level"],
        "checks_performed": dict(checks) if checks else {},
        "decision": row["decision"],
        "reasoning": row["reasoning"] or "",
        "requires_review": bool(row["requires_review"]),
    }


class PostgresSafetyAuditStore:
    """Safety audit rows in PostgreSQL, newest first."""

    def __init__(self, pool: Any) -> None:
        if pool is None:
            raise ValueError(
                "PostgresSafetyAuditStore needs a pool; use "
                "InMemorySafetyAuditStore when no database is configured"
            )
        self._pool = GuardedPool(pool)

    async def ensure_schema(self) -> None:
        """Apply ``SAFETY_SCHEMA_SQL`` (idempotent safety net)."""
        await self._pool.guard(apply_schema(self._pool, SAFETY_SCHEMA_SQL))

    async def add_audit(self, record: Dict[str, Any]) -> str:
        row = normalize_safety_audit(record)
        moment = row.get("timestamp")
        stamp = None if moment is None else _aware(moment)
        value = await self._pool.execute_returning(
            """
            INSERT INTO safety_audit (
                timestamp, action_proposed, safety_level, checks_performed,
                decision, reasoning, requires_review
            ) VALUES (COALESCE($1, now()), $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            stamp,
            row["action_proposed"],
            row["safety_level"],
            row["checks_performed"],
            row["decision"],
            row["reasoning"],
            row["requires_review"],
        )
        if not isinstance(value, (uuid.UUID, str)):
            raise StoreUnavailable(
                "PostgreSQL insert was buffered; no id is available until replay"
            )
        return str(value)

    async def pending_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_COLUMNS} FROM safety_audit WHERE requires_review "
            "ORDER BY timestamp DESC, seq DESC LIMIT $1",
            int(limit),
        )
        return [_row(row) for row in rows]

    async def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_COLUMNS} FROM safety_audit "
            "ORDER BY timestamp DESC, seq DESC LIMIT $1",
            int(limit),
        )
        return [_row(row) for row in rows]

    def status(self) -> Dict[str, Any]:
        """``backend``, ``available`` and ``last_error``; an unreachable server
        is reported as unavailable until a later call reaches it."""
        return pool_status(self._pool)
