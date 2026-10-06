"""The conversation repository's DDL: idempotent, safe and timezone-aware."""

import re

import pytest

from mtc.memory.conversation_store import (
    CONVERSATION_SCHEMA_SQL,
    PostgresConversationRepository,
)
from mtc.memory.protocols import MESSAGE_ROLES


def _flat(sql: str) -> str:
    return " ".join(sql.split()).upper()


def _table(name: str) -> str:
    for sql in CONVERSATION_SCHEMA_SQL:
        flat = _flat(sql)
        if flat.startswith(f"CREATE TABLE IF NOT EXISTS {name.upper()} ("):
            return flat
    raise AssertionError(f"no CREATE TABLE for {name}")


def test_statements_are_idempotent_creates():
    assert CONVERSATION_SCHEMA_SQL and isinstance(CONVERSATION_SCHEMA_SQL, tuple)
    for sql in CONVERSATION_SCHEMA_SQL:
        assert _flat(sql).startswith(
            ("CREATE TABLE IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS")
        ), _flat(sql)[:60]
    messages = _table("messages")
    assert "TO_SPEAKER VARCHAR(200)" in messages
    assert "ROLE VARCHAR(20) NOT NULL" in messages
    assert "REFERENCES CONVERSATIONS(ID) ON DELETE CASCADE" in messages
    indexes = " ".join(
        _flat(sql) for sql in CONVERSATION_SCHEMA_SQL if "CREATE INDEX" in _flat(sql)
    )
    assert "ON MESSAGES (CONVERSATION_ID, TIMESTAMP)" in indexes
    assert "ON MESSAGES (TIMESTAMP DESC)" in indexes
    emotions = _table("emotional_history")
    for column in ("INTENSITY", "VALENCE", "AROUSAL", "DOMINANCE"):
        assert f"{column} DOUBLE PRECISION" in emotions
    # the addressee column is in the table, never added at runtime
    assert not any("ALTER" in _flat(sql) for sql in CONVERSATION_SCHEMA_SQL)


def test_role_check_matches_the_message_roles():
    """The table refuses a role outside ``MESSAGE_ROLES``, the set both
    backends check before writing, so the tool role is stored in PostgreSQL
    exactly as in memory."""
    messages = _table("messages")
    match = re.search(
        r"ROLE VARCHAR\(20\) NOT NULL CHECK \(ROLE IN \(([^)]*)\)\)", messages
    )
    assert match, messages
    roles = tuple(part.strip().strip("'").lower() for part in match.group(1).split(","))
    assert roles == MESSAGE_ROLES
    assert "tool" in roles


def test_no_drop_password_seed_or_default_participants():
    for sql in CONVERSATION_SCHEMA_SQL:
        flat = _flat(sql)
        for forbidden in (
            "DROP",
            "PASSWORD",
            "INSERT",
            "GRANT",
            "ROLE ",
            "EXTENSION",
            "COPY ",
            "OWNER",
        ):
            # "role VARCHAR" is a column; a database role statement is not
            if forbidden == "ROLE " and "ROLE VARCHAR" in flat:
                continue
            assert forbidden not in flat, (forbidden, flat[:60])
    conversations = _table("conversations")
    assert "PARTICIPANTS TEXT[] NOT NULL" in conversations
    participants = conversations.split("PARTICIPANTS", 1)[1].split(",", 1)[0]
    assert "DEFAULT" not in participants
    assert "ID UUID PRIMARY KEY" in conversations


def test_timestamptz_everywhere():
    for sql in CONVERSATION_SCHEMA_SQL:
        flat = _flat(sql)
        if not flat.startswith("CREATE TABLE"):
            continue
        # every TIMESTAMP type is the zone-aware one
        for match in re.finditer(r"\bTIMESTAMP\w*", flat):
            word = match.group(0)
            if word == "TIMESTAMP":
                tail = flat[match.end() :].lstrip()
                assert tail.startswith("TIMESTAMPTZ"), flat
            else:
                assert word == "TIMESTAMPTZ", (word, flat)
    for table, column in (
        ("conversations", "STARTED_AT TIMESTAMPTZ NOT NULL"),
        ("conversations", "ENDED_AT TIMESTAMPTZ"),
        ("messages", "TIMESTAMP TIMESTAMPTZ NOT NULL"),
        ("emotional_history", "TIMESTAMP TIMESTAMPTZ NOT NULL DEFAULT NOW()"),
    ):
        assert column in _table(table), (table, column)


def test_repository_refuses_a_missing_pool():
    with pytest.raises(ValueError, match="InMemoryConversationRepository"):
        PostgresConversationRepository(None)


class _RecordingPool:
    """Records statements; never reaches a server."""

    def __init__(self) -> None:
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        return "UPDATE 0"

    async def fetchval(self, query, *args):
        self.calls.append((query, args))
        return 7


@pytest.mark.asyncio
async def test_naive_timestamps_refused_before_anything_is_sent():
    from datetime import datetime

    pool = _RecordingPool()
    repo = PostgresConversationRepository(pool)
    cid = "00000000-0000-4000-8000-000000000001"
    with pytest.raises(ValueError, match="timezone-aware"):
        await repo.add_conversation(cid, ["Testagent", "Alice"], datetime(2000, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        await repo.add_message(cid, "Alice", "user", "hi", datetime(2000, 1, 1))
    with pytest.raises(ValueError, match="UUID"):
        await repo.end_conversation("not-a-uuid", datetime(2000, 1, 1))
    assert pool.calls == []
