"""PostgreSQL schema artifacts: idempotent creates, aware timestamps, no extras.

The text checks run without a database. The ``db`` cases run only against a
throwaway database (``MTC_TEST_STORES=1``) and skip otherwise.
"""

import re
from datetime import datetime, timezone
from typing import List, Tuple

import pytest

from mtc.memory.episodic_store import EPISODIC_SCHEMA_SQL, PostgresEpisodicStore
from mtc.memory.postgres_stores import RESEARCH_SCHEMA_SQL, PostgresSnapshotStore
from mtc.memory.protocols import SnapshotRecord, StoreUnavailable

SCHEMAS = {"research": RESEARCH_SCHEMA_SQL, "episodic": EPISODIC_SCHEMA_SQL}
T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _statements() -> List[Tuple[str, str]]:
    return [(name, sql) for name, schema in SCHEMAS.items() for sql in schema]


def _table_columns(sql: str) -> List[Tuple[str, str]]:
    """Return ``(column, type)`` pairs of a CREATE TABLE statement."""
    body = sql[sql.index("(") + 1 : sql.rindex(")")]
    parts, depth, current = [], 0, ""
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    columns = []
    for part in parts:
        tokens = part.split()
        if len(tokens) >= 2 and tokens[0].upper() not in ("PRIMARY", "CHECK"):
            columns.append((tokens[0].lower(), tokens[1].lower()))
    return columns


def test_schemas_are_tuples_of_statements():
    for name, schema in SCHEMAS.items():
        assert isinstance(schema, tuple) and schema, name
        assert all(isinstance(sql, str) and sql.strip() for sql in schema), name


def test_schema_statements_are_idempotent_creates():
    for name, sql in _statements():
        head = " ".join(sql.split()).upper()
        assert head.startswith("CREATE TABLE IF NOT EXISTS") or head.startswith(
            "CREATE INDEX IF NOT EXISTS"
        ), (name, sql)


def test_schema_has_no_drop_password_seed_or_pgvector():
    for name, sql in _statements():
        upper = sql.upper()
        for forbidden in ("DROP", "PASSWORD", "INSERT", "EXTENSION", "PGVECTOR"):
            assert forbidden not in upper, (name, forbidden)
        assert not re.search(r"\bVECTOR\s*\(", upper), name
        assert "FLOAT[]" not in upper, name


def test_timestamps_are_timestamptz():
    seen = 0
    for name, sql in _statements():
        if not sql.lstrip().upper().startswith("CREATE TABLE"):
            continue
        for column, kind in _table_columns(sql):
            if kind.startswith("timestamp"):
                seen += 1
                assert kind == "timestamptz", (name, column, kind)
    assert seen >= 8


def test_snapshot_tables_match_the_protocol_columns():
    by_table = {}
    for sql in RESEARCH_SCHEMA_SQL:
        match = re.match(r"\s*CREATE TABLE IF NOT EXISTS (\w+)", sql)
        if match:
            by_table[match.group(1)] = dict(_table_columns(sql))
    snapshots = by_table["consciousness_snapshots"]
    for column in SnapshotRecord.__dataclass_fields__:
        assert column in snapshots, column
    assert "status" in snapshots and "coverage" in snapshots
    overall = next(
        sql for sql in RESEARCH_SCHEMA_SQL if "consciousness_snapshots (" in sql
    )
    overall_line = next(
        line for line in overall.splitlines() if "overall_score" in line
    )
    assert "NOT NULL" not in overall_line.upper()
    events = by_table["consciousness_events"]
    for column in ("timestamp", "event_type", "description", "snapshot_id"):
        assert column in events


def test_episodic_schema_has_the_indexes_the_queries_need():
    joined = "\n".join(EPISODIC_SCHEMA_SQL).lower()
    assert "using gin (tsv_content)" in joined
    assert "generated always as (to_tsvector('english', content)) stored" in joined
    assert "(importance desc)" in joined
    assert "where not consolidated" in joined
    for table in ("core_memory", "episodic_memory", "semantic_memory"):
        assert f"create table if not exists {table}" in joined
    assert "importance double precision" in joined
    assert "source_memories bigint[]" in joined


class InterfaceError(Exception):
    """Named like the driver's connection error; matched by name only."""


class RaisingPool:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def fetch(self, *args):
        raise self.error

    async def fetchrow(self, *args):
        raise self.error

    async def fetchval(self, *args):
        raise self.error

    async def execute(self, *args):
        raise self.error


class BufferingPool:
    """Stands in for a ResilientPool whose breaker is open."""

    MARKER = object()

    async def execute_returning(self, *args):
        return self.MARKER

    async def execute(self, *args):
        return self.MARKER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError("refused"), InterfaceError("closed"), TimeoutError()],
)
async def test_unreachable_server_raises_store_unavailable(error):
    episodic = PostgresEpisodicStore(RaisingPool(error))
    snapshots = PostgresSnapshotStore(RaisingPool(error))
    with pytest.raises(StoreUnavailable):
        await episodic.load_core_blocks()
    with pytest.raises(StoreUnavailable):
        await episodic.add_episodic("text")
    with pytest.raises(StoreUnavailable):
        await snapshots.latest_snapshot()


class SwitchPool:
    """A plain pool (no status of its own) that is unreachable while down."""

    def __init__(self) -> None:
        self.down = True

    async def fetch(self, *args):
        if self.down:
            raise OSError("connection refused")
        return []

    async def fetchrow(self, *args):
        if self.down:
            raise OSError("connection refused")
        return None


@pytest.mark.asyncio
async def test_status_reports_an_unreachable_plain_pool_until_it_answers():
    episodic_pool, snapshot_pool = SwitchPool(), SwitchPool()
    episodic = PostgresEpisodicStore(episodic_pool)
    snapshots = PostgresSnapshotStore(snapshot_pool)
    for store in (episodic, snapshots):
        assert store.status() == {
            "backend": "postgres",
            "available": True,
            "last_error": None,
        }
    with pytest.raises(StoreUnavailable):
        await episodic.load_core_blocks()
    with pytest.raises(StoreUnavailable):
        await snapshots.latest_snapshot()
    for store in (episodic, snapshots):
        status = store.status()
        assert status["available"] is False
        assert "connection refused" in status["last_error"]
    episodic_pool.down = snapshot_pool.down = False
    assert await episodic.load_core_blocks() == {}
    assert await snapshots.latest_snapshot() is None
    for store in (episodic, snapshots):
        assert store.status()["available"] is True
        assert store.status()["last_error"] is None


@pytest.mark.asyncio
async def test_other_database_errors_propagate():
    class UniqueViolationError(Exception):
        pass

    episodic = PostgresEpisodicStore(RaisingPool(UniqueViolationError("dup")))
    with pytest.raises(UniqueViolationError):
        await episodic.load_core_blocks()


@pytest.mark.asyncio
async def test_buffered_writes_return_the_marker_or_raise_for_snapshots():
    pool = BufferingPool()
    episodic = PostgresEpisodicStore(pool)
    assert await episodic.add_episodic("during an outage") is BufferingPool.MARKER
    assert await episodic.archive([1]) == 0
    snapshots = PostgresSnapshotStore(pool)
    record = SnapshotRecord(
        timestamp=T0,
        trigger="manual",
        overall_score=None,
        passing_count=0,
        phi_value=None,
        theory_scores={},
        indicator_results={},
        processing_time_ms=None,
        status="insufficient_data",
        coverage=None,
        metadata={},
    )
    with pytest.raises(StoreUnavailable):
        await snapshots.add_snapshot(record)


@pytest.mark.asyncio
async def test_naive_snapshot_timestamp_rejected_before_sending():
    snapshots = PostgresSnapshotStore(RaisingPool(AssertionError("sent")))
    record = SnapshotRecord(
        timestamp=datetime(2000, 1, 1),
        trigger="manual",
        overall_score=0.5,
        passing_count=1,
        phi_value=None,
        theory_scores={},
        indicator_results={},
        processing_time_ms=None,
        status="measured",
        coverage=1.0,
        metadata={},
    )
    with pytest.raises(ValueError):
        await snapshots.add_snapshot(record)


@pytest.mark.db
@pytest.mark.asyncio
async def test_ensure_schema_twice_is_idempotent(throwaway_postgres):
    snapshots = PostgresSnapshotStore(throwaway_postgres)
    episodic = PostgresEpisodicStore(throwaway_postgres)
    for _ in range(2):
        await snapshots.ensure_schema()
        await episodic.ensure_schema()
    tables = await throwaway_postgres.fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public'"
    )
    names = {row["table_name"] for row in tables}
    assert {
        "consciousness_snapshots",
        "consciousness_events",
        "core_memory",
        "episodic_memory",
        "semantic_memory",
    } <= names


@pytest.mark.db
@pytest.mark.asyncio
async def test_snapshot_round_trip_keeps_null_overall(throwaway_postgres):
    store = PostgresSnapshotStore(throwaway_postgres)
    await store.ensure_schema()
    record = SnapshotRecord(
        timestamp=T0,
        trigger="manual",
        overall_score=None,
        passing_count=0,
        phi_value=None,
        theory_scores={},
        indicator_results={"GWT-1": {"status": "insufficient_data"}},
        processing_time_ms=None,
        status="insufficient_data",
        coverage=None,
        metadata={"note": "synthetic"},
    )
    snapshot_id = await store.add_snapshot(record)
    loaded = await store.latest_snapshot()
    assert loaded.id == snapshot_id
    assert loaded.overall_score is None and loaded.phi_value is None
    assert loaded.status == "insufficient_data"
    assert loaded.indicator_results == {"GWT-1": {"status": "insufficient_data"}}
    assert loaded.timestamp == T0 and loaded.timestamp.tzinfo is not None
