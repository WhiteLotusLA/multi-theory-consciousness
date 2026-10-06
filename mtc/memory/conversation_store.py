"""
Conversation Repository (PostgreSQL)
====================================

``PostgresConversationRepository`` implements ``ConversationRepository``
(``mtc.memory.protocols``) over three tables: ``conversations`` (a UUID id,
the participants, a summary, start and end times), ``messages`` (speaker,
stored role, content, timestamp and the ``to_speaker`` addressee) and
``emotional_history`` (a label and the PAD coordinates of a turn with its
trigger context). The rows hold private conversation text and speaker names
at runtime; they live only in an operator-controlled database and are never
logged by this module. ``InMemoryConversationRepository`` in
``mtc.memory.inmemory`` is the default twin with the same contract (pinned by
``tests/memory/test_conversation_repository_contract.py``).

The pool is an asyncpg pool from ``mtc.memory.clients.create_postgres_pool``
or a ``ResilientPool`` around one, wrapped in ``GuardedPool`` so a connection
failure surfaces as ``StoreUnavailable``. Through an unavailable
``ResilientPool`` reads raise ``StoreUnavailable`` and writes are buffered:
``add_message`` and ``add_emotional_state`` then return the pool's buffered
marker instead of an id, ``add_conversation`` returns the caller's id (the row
is written on replay) and ``end_conversation`` reads False because nothing is
confirmed yet. A duplicate conversation id and a message for an unknown
conversation raise ValueError (the driver's integrity errors are matched by
class name, so this module never imports the driver).

Time windows are computed from the injected clock (default
``datetime.now(timezone.utc)``) and sent as instants, so both backends read
the same window and no interval arithmetic runs in SQL. Timestamps are
``TIMESTAMPTZ`` and come back as aware UTC; a naive one raises ValueError
before anything is sent.

Schema artifact: ``CONVERSATION_SCHEMA_SQL``, registered in
``mtc.core.schema_registry.SCHEMA_ARTIFACTS`` and applied idempotently by
``ensure_schema()`` as a safety net. Phase 7's migration written from this
constant is canonical. The addressee column is part of the table definition
(no runtime ALTER), the role column checks the four ``MESSAGE_ROLES`` (a
plugin's reply is a ``tool`` row), participants have no default, and the DDL
holds no database role, password, seed row or default participant. The
module reads no config key; host, port, database and credentials come from
Settings through the pool factory. The repository never closes the pool: its
owner does.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from mtc.memory.clients import apply_schema
from mtc.memory.postgres_stores import GuardedPool, pool_status
from mtc.memory.protocols import (
    check_message_fields,
    conversation_uuid,
    normalize_emotional_state,
)

logger = logging.getLogger(__name__)

CONVERSATION_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id UUID PRIMARY KEY,
        participants TEXT[] NOT NULL,
        summary TEXT,
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        conversation_id UUID NOT NULL
            REFERENCES conversations(id) ON DELETE CASCADE,
        speaker VARCHAR(200) NOT NULL,
        role VARCHAR(20) NOT NULL
            CHECK (role IN ('user', 'assistant', 'system', 'tool')),
        content TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        to_speaker VARCHAR(200)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS messages_conversation_timestamp_idx
    ON messages (conversation_id, timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS messages_timestamp_idx
    ON messages (timestamp DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS emotional_history (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
        primary_emotion VARCHAR(50),
        intensity DOUBLE PRECISION,
        valence DOUBLE PRECISION,
        arousal DOUBLE PRECISION,
        dominance DOUBLE PRECISION,
        trigger_context TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS emotional_history_timestamp_idx
    ON emotional_history (timestamp DESC)
    """,
)

_MESSAGE_COLUMNS = (
    "m.id, m.conversation_id, m.speaker, m.role, m.content, m.timestamp, "
    "m.to_speaker"
)
_EMOTION_COLUMNS = (
    "id, timestamp, primary_emotion, intensity, valence, arousal, dominance, "
    "trigger_context"
)
_ADDRESSED = "(m.to_speaker IS NULL OR m.to_speaker = $%d)"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(moment: datetime) -> datetime:
    if not isinstance(moment, datetime):
        raise ValueError("timestamps must be datetimes")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _affected(status: Any) -> int:
    """Read the row count of an asyncpg status string ("UPDATE 1")."""
    if isinstance(status, str):
        tail = status.rsplit(" ", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


async def _integrity(operation: Awaitable[Any], messages: Mapping[str, str]) -> Any:
    """Await ``operation``; a driver integrity error named in ``messages``
    becomes ValueError with that message. Everything else propagates."""
    try:
        return await operation
    except Exception as exc:
        for cls in type(exc).__mro__:
            if cls.__name__ in messages:
                raise ValueError(messages[cls.__name__]) from exc
        raise


def _message(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "conversation_id": str(row["conversation_id"]),
        "speaker": row["speaker"],
        "role": row["role"],
        "content": row["content"],
        "timestamp": _utc(row["timestamp"]),
        "to_speaker": row["to_speaker"],
    }


def _emotion(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "timestamp": _utc(row["timestamp"]),
        "primary_emotion": row["primary_emotion"],
        "intensity": _optional_float(row["intensity"]),
        "valence": _optional_float(row["valence"]),
        "arousal": _optional_float(row["arousal"]),
        "dominance": _optional_float(row["dominance"]),
        "trigger_context": row["trigger_context"],
    }


class PostgresConversationRepository:
    """Conversations, messages and emotional history in PostgreSQL."""

    def __init__(self, pool: Any, clock: Optional[Clock] = None) -> None:
        if pool is None:
            raise ValueError(
                "PostgresConversationRepository needs a pool; use "
                "InMemoryConversationRepository when no database is configured"
            )
        self._pool = GuardedPool(pool)
        self._clock: Clock = clock or _utc_now

    def _cutoff(self, hours: float) -> datetime:
        return _aware(self._clock()) - timedelta(hours=float(hours))

    async def ensure_schema(self) -> None:
        """Apply ``CONVERSATION_SCHEMA_SQL`` (idempotent safety net)."""
        await self._pool.guard(apply_schema(self._pool, CONVERSATION_SCHEMA_SQL))

    async def add_conversation(
        self,
        conversation_id: str,
        participants: Sequence[str],
        started_at: datetime,
        summary: Optional[str] = None,
    ) -> str:
        key = conversation_uuid(conversation_id)
        names = [str(name) for name in participants]
        stamp = _aware(started_at)
        await _integrity(
            self._pool.execute(
                "INSERT INTO conversations (id, participants, summary, started_at) "
                "VALUES ($1, $2, $3, $4)",
                uuid.UUID(key),
                names,
                summary,
                stamp,
            ),
            {"UniqueViolationError": "conversation id is already stored"},
        )
        return key

    async def end_conversation(
        self,
        conversation_id: str,
        ended_at: datetime,
        summary: Optional[str] = None,
    ) -> bool:
        key = conversation_uuid(conversation_id)
        stamp = _aware(ended_at)
        status = await self._pool.execute(
            "UPDATE conversations SET ended_at = $2, "
            "summary = COALESCE($3, summary) WHERE id = $1",
            uuid.UUID(key),
            stamp,
            summary,
        )
        return _affected(status) > 0

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        key = conversation_uuid(conversation_id)
        row = await self._pool.fetchrow(
            "SELECT id, participants, summary, started_at, ended_at "
            "FROM conversations WHERE id = $1",
            uuid.UUID(key),
        )
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "participants": list(row["participants"] or []),
            "summary": row["summary"],
            "started_at": _utc(row["started_at"]),
            "ended_at": _utc(row["ended_at"]),
        }

    async def add_message(
        self,
        conversation_id: str,
        speaker: str,
        role: str,
        content: str,
        timestamp: datetime,
        to_speaker: Optional[str] = None,
    ) -> int:
        key = conversation_uuid(conversation_id)
        check_message_fields(speaker, role, content, to_speaker)
        stamp = _aware(timestamp)
        result = await _integrity(
            self._pool.execute_returning(
                "INSERT INTO messages "
                "(conversation_id, speaker, role, content, timestamp, to_speaker) "
                "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                uuid.UUID(key),
                speaker,
                role,
                content,
                stamp,
                to_speaker,
            ),
            {"ForeignKeyViolationError": "message refers to an unknown conversation"},
        )
        return int(result) if isinstance(result, int) else result

    async def add_emotional_state(self, record: Dict[str, Any]) -> int:
        row = normalize_emotional_state(record)
        moment = row["timestamp"]
        stamp = _aware(self._clock() if moment is None else moment)
        result = await self._pool.execute_returning(
            "INSERT INTO emotional_history (timestamp, primary_emotion, intensity, "
            "valence, arousal, dominance, trigger_context) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
            stamp,
            row["primary_emotion"],
            row["intensity"],
            row["valence"],
            row["arousal"],
            row["dominance"],
            row["trigger_context"],
        )
        return int(result) if isinstance(result, int) else result

    async def recent_emotional_states(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_EMOTION_COLUMNS} FROM emotional_history "
            "ORDER BY timestamp DESC, id DESC LIMIT $1",
            int(limit),
        )
        return [_emotion(row) for row in rows]

    async def latest_message_time(
        self, participant: Optional[str] = None, *, addressee: Optional[str] = None
    ) -> Optional[datetime]:
        if participant is None:
            value = await self._pool.fetchval(
                "SELECT m.timestamp FROM messages m "
                f"WHERE {_ADDRESSED % 1} "
                "ORDER BY m.timestamp DESC, m.id DESC LIMIT 1",
                addressee,
            )
        else:
            value = await self._pool.fetchval(
                "SELECT m.timestamp FROM messages m "
                "JOIN conversations c ON m.conversation_id = c.id "
                f"WHERE $1 = ANY(c.participants) AND {_ADDRESSED % 2} "
                "ORDER BY m.timestamp DESC, m.id DESC LIMIT 1",
                participant,
                addressee,
            )
        return _utc(value)

    async def recent_messages(
        self,
        participant: str,
        hours: float,
        limit: int,
        *,
        addressee: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            f"WHERE $1 = ANY(c.participants) AND m.timestamp > $2 "
            f"AND {_ADDRESSED % 3} "
            "ORDER BY m.timestamp DESC, m.id DESC LIMIT $4",
            participant,
            self._cutoff(hours),
            addressee,
            int(limit),
        )
        return [_message(row) for row in reversed(list(rows))]

    async def timeline(
        self, hours: float, limit: int, *, addressee: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages m "
            f"WHERE m.timestamp > $1 AND {_ADDRESSED % 2} "
            "ORDER BY m.timestamp DESC, m.id DESC LIMIT $3",
            self._cutoff(hours),
            addressee,
            int(limit),
        )
        return [_message(row) for row in reversed(list(rows))]

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error``.

        An unreachable server is reported as unavailable with the error of the
        call that failed, until a later call reaches it.
        """
        return pool_status(self._pool)


__all__ = ("CONVERSATION_SCHEMA_SQL", "PostgresConversationRepository")
