"""
Episodic Store
==============

Two implementations of ``EpisodicStore`` (``mtc.memory.protocols``): the core
memory blocks, episodic memories and semantic memories the memory manager
reads and writes. Episodic rows hold conversation text and semantic rows are
derived from it, so both are private data at runtime and live only in
gitignored or operator-controlled stores.

- ``InMemoryEpisodicStore`` keeps everything in process memory (a restart
  starts empty). Text search scores by term frequency over the query terms in
  place of ``ts_rank``: every query term must appear (as with
  ``plainto_tsquery``), a few English stop words are dropped, and there is no
  stemming.
- ``PostgresEpisodicStore`` runs over an asyncpg pool from
  ``mtc.memory.clients.create_postgres_pool`` (JSON and JSONB codecs
  installed, so the JSONB columns take and return dicts) or a
  ``ResilientPool`` around one. Text search uses the generated ``tsv_content``
  column with ``plainto_tsquery('english', ...)`` and ``ts_rank``.

Shared contract (``tests/memory/test_episodic_contract.py``): importance is on
a 0 to 10 scale and 10 round-trips (a value outside the scale raises
ValueError); search ties and ``recent`` break newest first; ``unconsolidated``
and ``retention_candidates`` are oldest first (the latter pages with an
``after=(created_at, id)`` cursor, the last row of the previous page, and can
leave out memory types through ``exclude_types``); archiving sets the memory type
to ``archived``, which ``get_episodic``, search, ``recent``, ``unconsolidated``
and ``retention_candidates`` exclude; returned rows are copies with
timezone-aware UTC timestamps. Both stores read time from the injected clock
(default ``datetime.now(timezone.utc)``), so time windows and the created and
last-accessed times use one clock. Through a ``ResilientPool`` that is
unavailable, reads raise ``StoreUnavailable`` and writes are buffered:
``add_episodic`` and ``add_semantic`` then return the pool's buffered marker
instead of an id (the caller counts the row as unindexed), and the update,
seed and delete counts read 0 or False because nothing is confirmed yet. A
connection failure of a plain pool raises ``StoreUnavailable`` too
(``GuardedPool``); other database errors propagate. ``status()`` reports
the store unavailable, with the error, from a failed call until a later call
reaches the server. ``mark_consolidated`` with a None strength leaves the
stored strength unchanged (an unmeasured strength is never written as 0).

Row shape: ``id``, ``content``, ``memory_type``, ``importance``,
``emotional_context`` (a dict, empty when none was given), ``created_at``,
``last_accessed``, ``accessed_count``, ``consolidated``,
``consolidation_strength`` and ``replay_count``; search rows add
``relevance``. ``get_semantic`` (outside the protocol) reads semantic rows
back by id.

Schema artifact: ``EPISODIC_SCHEMA_SQL``, applied idempotently by
``ensure_schema()`` as a safety net; Phase 7's ``migrations/001`` is the
canonical schema and is written from this constant. No table holds a vector
column and the schema needs no extension: the vectors live in the vector
store, keyed by ``point_id_for_memory`` of the row id.
"""

import copy
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from mtc.memory.clients import apply_schema
from mtc.memory.postgres_stores import GuardedPool, pool_status

logger = logging.getLogger(__name__)

ARCHIVED_TYPE = "archived"
IMPORTANCE_MAX = 10.0

EPISODIC_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS core_memory (
        block_name VARCHAR(100) PRIMARY KEY,
        content TEXT NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS episodic_memory (
        id BIGSERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        memory_type VARCHAR(50) DEFAULT 'episodic',
        importance DOUBLE PRECISION DEFAULT 5.0,
        emotional_context JSONB,
        tsv_content TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        last_accessed TIMESTAMPTZ DEFAULT NOW(),
        accessed_count INTEGER DEFAULT 0,
        consolidated BOOLEAN DEFAULT FALSE,
        consolidation_strength DOUBLE PRECISION DEFAULT 0,
        replay_count INTEGER DEFAULT 0,
        CHECK (importance >= 0 AND importance <= 10)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS episodic_memory_tsv_idx
    ON episodic_memory USING GIN (tsv_content)
    """,
    """
    CREATE INDEX IF NOT EXISTS episodic_memory_created_at_idx
    ON episodic_memory (created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS episodic_memory_type_idx
    ON episodic_memory (memory_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS episodic_memory_importance_idx
    ON episodic_memory (importance DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS episodic_memory_unconsolidated_idx
    ON episodic_memory (created_at) WHERE NOT consolidated
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_memory (
        id BIGSERIAL PRIMARY KEY,
        concept TEXT NOT NULL,
        knowledge TEXT DEFAULT '',
        associations JSONB DEFAULT '{}',
        strength DOUBLE PRECISION DEFAULT 0.5,
        confidence DOUBLE PRECISION DEFAULT 0.5,
        emotional_context JSONB DEFAULT '{}',
        source_memories BIGINT[] DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        last_accessed TIMESTAMPTZ DEFAULT NOW()
    )
    """,
)

_EPISODIC_COLUMNS = (
    "id, content, memory_type, importance, emotional_context, created_at, "
    "last_accessed, accessed_count, consolidated, consolidation_strength, "
    "replay_count"
)
_SEMANTIC_COLUMNS = (
    "id, concept, knowledge, associations, strength, confidence, "
    "emotional_context, source_memories, created_at, updated_at, last_accessed"
)

# A short list of English stop words the in-memory search drops, so a query
# reads closer to what plainto_tsquery keeps.
_STOP_WORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to was "
    "were with".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(moment: datetime, name: str = "timestamp") -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _check_importance(importance: float) -> float:
    value = float(importance)
    if math.isnan(value) or not 0.0 <= value <= IMPORTANCE_MAX:
        raise ValueError(f"importance must be within [0, 10], got {importance!r}")
    return value


def _check_semantic(record: Mapping[str, Any]) -> Dict[str, Any]:
    concept = str(record.get("concept") or "").strip()
    if not concept:
        raise ValueError("a semantic record needs a non-empty 'concept'")
    return {
        "concept": concept,
        "knowledge": str(record.get("knowledge") or ""),
        "associations": copy.deepcopy(dict(record.get("associations") or {})),
        "strength": float(record.get("strength", 0.5)),
        "confidence": float(record.get("confidence", 0.5)),
        "emotional_context": copy.deepcopy(dict(record.get("emotional_context") or {})),
        "source_memories": [int(i) for i in record.get("source_memories") or []],
    }


def _ids(ids: Sequence[int]) -> List[int]:
    seen: Dict[int, None] = {}
    for value in ids:
        seen.setdefault(int(value), None)
    return list(seen)


def _cursor(after: Tuple[datetime, int]) -> Tuple[datetime, int]:
    """Validate a ``(created_at, id)`` paging cursor."""
    moment, last_id = after
    return _aware(moment, "cursor time"), int(last_id)


def _terms(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP_WORDS]


class InMemoryEpisodicStore:
    """Core blocks, episodic and semantic memories held in process memory."""

    def __init__(self, clock: Optional[Callable[[], datetime]] = None) -> None:
        self._clock = clock or _utcnow
        self._core: Dict[str, str] = {}
        self._rows: Dict[int, Dict[str, Any]] = {}
        self._semantic: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        self._next_semantic_id = 1

    def _now(self) -> datetime:
        return _aware(self._clock(), "clock time")

    def _visible(self) -> List[Dict[str, Any]]:
        return [r for r in self._rows.values() if r["memory_type"] != ARCHIVED_TYPE]

    @staticmethod
    def _newest_first(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(rows, key=lambda r: (r["created_at"], r["id"]), reverse=True)

    @staticmethod
    def _oldest_first(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(rows, key=lambda r: (r["created_at"], r["id"]))

    async def ensure_schema(self) -> None:
        return None

    async def load_core_blocks(self) -> Dict[str, str]:
        return dict(self._core)

    async def set_core_block(self, name: str, content: str) -> None:
        self._core[str(name)] = str(content)

    async def seed_core_blocks(
        self, blocks: Mapping[str, str], overwrite: bool = False
    ) -> Dict[str, bool]:
        written: Dict[str, bool] = {}
        for name, content in blocks.items():
            if overwrite or name not in self._core:
                self._core[str(name)] = str(content)
                written[name] = True
            else:
                written[name] = False
        return written

    async def add_episodic(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 5.0,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        value = _check_importance(importance)
        now = self._now()
        memory_id = self._next_id
        self._next_id += 1
        self._rows[memory_id] = {
            "id": memory_id,
            "content": str(content),
            "memory_type": memory_type,
            "importance": value,
            "emotional_context": copy.deepcopy(dict(emotional_context or {})),
            "created_at": now,
            "last_accessed": now,
            "accessed_count": 0,
            "consolidated": False,
            "consolidation_strength": 0.0,
            "replay_count": 0,
        }
        return memory_id

    async def get_episodic(self, ids: Sequence[int]) -> List[Dict[str, Any]]:
        rows = []
        for memory_id in _ids(ids):
            row = self._rows.get(memory_id)
            if row is not None and row["memory_type"] != ARCHIVED_TYPE:
                rows.append(copy.deepcopy(row))
        return rows

    async def search_text(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        wanted = set(_terms(query))
        if not wanted or limit <= 0:
            return []
        scored = []
        for row in self._visible():
            if row["importance"] < min_importance:
                continue
            if memory_type is not None and row["memory_type"] != memory_type:
                continue
            counts = Counter(_terms(row["content"]))
            if not all(term in counts for term in wanted):
                continue
            relevance = float(sum(counts[term] for term in wanted))
            scored.append((relevance, row))
        scored.sort(
            key=lambda item: (item[0], item[1]["created_at"], item[1]["id"]),
            reverse=True,
        )
        results = []
        for relevance, row in scored[:limit]:
            hit = copy.deepcopy(row)
            hit["relevance"] = relevance
            results.append(hit)
        return results

    async def recent(
        self, hours: float, limit: int, min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        floor = self._now() - timedelta(hours=float(hours))
        rows = [
            r
            for r in self._visible()
            if r["created_at"] > floor and r["importance"] >= min_importance
        ]
        return [copy.deepcopy(r) for r in self._newest_first(rows)[: max(limit, 0)]]

    async def unconsolidated(self, limit: int) -> List[Dict[str, Any]]:
        rows = [r for r in self._visible() if not r["consolidated"]]
        return [copy.deepcopy(r) for r in self._oldest_first(rows)[: max(limit, 0)]]

    async def add_semantic(self, record: Dict[str, Any]) -> int:
        clean = _check_semantic(record)
        now = self._now()
        semantic_id = self._next_semantic_id
        self._next_semantic_id += 1
        self._semantic[semantic_id] = {
            "id": semantic_id,
            **clean,
            "created_at": now,
            "updated_at": now,
            "last_accessed": now,
        }
        return semantic_id

    async def get_semantic(self, ids: Sequence[int]) -> List[Dict[str, Any]]:
        """Return semantic rows by id (outside the protocol; tests and tools)."""
        return [
            copy.deepcopy(self._semantic[i]) for i in _ids(ids) if i in self._semantic
        ]

    async def mark_consolidated(
        self, ids: Sequence[int], strength: Optional[float]
    ) -> int:
        changed = 0
        for memory_id in _ids(ids):
            row = self._rows.get(memory_id)
            if row is None:
                continue
            row["consolidated"] = True
            if strength is not None:
                row["consolidation_strength"] = float(strength)
            row["replay_count"] += 1
            changed += 1
        return changed

    async def touch(self, ids: Sequence[int]) -> None:
        now = self._now()
        for memory_id in _ids(ids):
            row = self._rows.get(memory_id)
            if row is not None:
                row["accessed_count"] += 1
                row["last_accessed"] = now

    async def retention_candidates(
        self,
        older_than: datetime,
        limit: int,
        *,
        after: Optional[Tuple[datetime, int]] = None,
        exclude_types: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        cutoff = _aware(older_than, "older_than")
        excluded = set(exclude_types)
        rows = [
            r
            for r in self._visible()
            if r["created_at"] < cutoff and r["memory_type"] not in excluded
        ]
        if after is not None:
            key = _cursor(after)
            rows = [r for r in rows if (r["created_at"], r["id"]) > key]
        return [copy.deepcopy(r) for r in self._oldest_first(rows)[: max(limit, 0)]]

    async def archive(self, ids: Sequence[int]) -> int:
        changed = 0
        for memory_id in _ids(ids):
            row = self._rows.get(memory_id)
            if row is not None and row["memory_type"] != ARCHIVED_TYPE:
                row["memory_type"] = ARCHIVED_TYPE
                changed += 1
        return changed

    async def delete(self, ids: Sequence[int]) -> int:
        return sum(1 for i in _ids(ids) if self._rows.pop(i, None) is not None)

    async def stats(self) -> Dict[str, Any]:
        visible = self._visible()
        return {
            "core_blocks": len(self._core),
            "episodic": len(visible),
            "archived": len(self._rows) - len(visible),
            "unconsolidated": sum(1 for r in visible if not r["consolidated"]),
            "semantic": len(self._semantic),
            "total_importance": float(sum(r["importance"] for r in visible)),
        }

    def status(self) -> Dict[str, Any]:
        return {"backend": "memory", "available": True, "last_error": None}

    async def close(self) -> None:
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    """Decode a JSONB value (a dict through the codec, text without one)."""
    if value is None:
        return {}
    if isinstance(value, str):
        decoded = json.loads(value)
        return dict(decoded) if isinstance(decoded, dict) else {}
    return dict(value)


def _utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _episodic_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = {
        "id": int(row["id"]),
        "content": row["content"],
        "memory_type": row["memory_type"],
        "importance": float(row["importance"]),
        "emotional_context": _as_dict(row["emotional_context"]),
        "created_at": _utc(row["created_at"]),
        "last_accessed": _utc(row["last_accessed"]),
        "accessed_count": int(row["accessed_count"] or 0),
        "consolidated": bool(row["consolidated"]),
        "consolidation_strength": float(row["consolidation_strength"] or 0.0),
        "replay_count": int(row["replay_count"] or 0),
    }
    if "relevance" in row.keys():
        result["relevance"] = float(row["relevance"])
    return result


def _semantic_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "concept": row["concept"],
        "knowledge": row["knowledge"] or "",
        "associations": _as_dict(row["associations"]),
        "strength": float(row["strength"]),
        "confidence": float(row["confidence"]),
        "emotional_context": _as_dict(row["emotional_context"]),
        "source_memories": [int(i) for i in row["source_memories"] or []],
        "created_at": _utc(row["created_at"]),
        "updated_at": _utc(row["updated_at"]),
        "last_accessed": _utc(row["last_accessed"]),
    }


def _affected(status: Any) -> int:
    """Read the row count of an asyncpg status string ("UPDATE 3")."""
    if isinstance(status, str):
        tail = status.rsplit(" ", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return 0


class PostgresEpisodicStore:
    """Core blocks, episodic and semantic memories in PostgreSQL."""

    def __init__(
        self, pool: Any, clock: Optional[Callable[[], datetime]] = None
    ) -> None:
        if pool is None:
            raise ValueError(
                "PostgresEpisodicStore needs a pool; use InMemoryEpisodicStore "
                "when no database is configured"
            )
        self._pool = GuardedPool(pool)
        self._clock = clock or _utcnow

    def _now(self) -> datetime:
        return _aware(self._clock(), "clock time")

    async def ensure_schema(self) -> None:
        """Apply ``EPISODIC_SCHEMA_SQL`` (idempotent safety net)."""
        await self._pool.guard(apply_schema(self._pool, EPISODIC_SCHEMA_SQL))

    async def load_core_blocks(self) -> Dict[str, str]:
        rows = await self._pool.fetch("SELECT block_name, content FROM core_memory")
        return {row["block_name"]: row["content"] for row in rows}

    async def set_core_block(self, name: str, content: str) -> None:
        await self._pool.execute(
            """
            INSERT INTO core_memory (block_name, content, updated_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (block_name)
            DO UPDATE SET content = EXCLUDED.content, updated_at = EXCLUDED.updated_at
            """,
            str(name),
            str(content),
            self._now(),
        )

    async def seed_core_blocks(
        self, blocks: Mapping[str, str], overwrite: bool = False
    ) -> Dict[str, bool]:
        if overwrite:
            for name, content in blocks.items():
                await self.set_core_block(name, content)
            return {name: True for name in blocks}
        written: Dict[str, bool] = {}
        for name, content in blocks.items():
            status = await self._pool.execute(
                """
                INSERT INTO core_memory (block_name, content, updated_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (block_name) DO NOTHING
                """,
                str(name),
                str(content),
                self._now(),
            )
            written[name] = _affected(status) == 1
        return written

    async def add_episodic(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 5.0,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        value = _check_importance(importance)
        now = self._now()
        sql = """
            INSERT INTO episodic_memory (
                content, memory_type, importance, emotional_context, created_at,
                last_accessed
            ) VALUES ($1, $2, $3, $4, $5, $5)
            RETURNING id
            """
        args = (
            str(content),
            memory_type,
            value,
            copy.deepcopy(dict(emotional_context or {})),
            now,
        )
        result = await self._pool.execute_returning(sql, *args)
        if isinstance(result, int):
            return result
        logger.warning(
            "episodic insert was buffered (PostgreSQL unavailable); the row has "
            "no id until replay"
        )
        return result

    async def get_episodic(self, ids: Sequence[int]) -> List[Dict[str, Any]]:
        wanted = _ids(ids)
        if not wanted:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_EPISODIC_COLUMNS} FROM episodic_memory "
            "WHERE id = ANY($1::bigint[]) AND memory_type <> $2",
            wanted,
            ARCHIVED_TYPE,
        )
        by_id = {int(row["id"]): _episodic_row(row) for row in rows}
        return [by_id[i] for i in wanted if i in by_id]

    async def search_text(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if limit <= 0 or not str(query).strip():
            return []
        args: List[Any] = [str(query), int(limit), float(min_importance), ARCHIVED_TYPE]
        type_clause = ""
        if memory_type is not None:
            args.append(memory_type)
            type_clause = f"AND memory_type = ${len(args)}"
        rows = await self._pool.fetch(
            f"""
            SELECT {_EPISODIC_COLUMNS},
                   ts_rank(tsv_content, plainto_tsquery('english', $1)) AS relevance
            FROM episodic_memory
            WHERE tsv_content @@ plainto_tsquery('english', $1)
              AND importance >= $3
              AND memory_type <> $4
              {type_clause}
            ORDER BY relevance DESC, created_at DESC, id DESC
            LIMIT $2
            """,
            *args,
        )
        return [_episodic_row(row) for row in rows]

    async def recent(
        self, hours: float, limit: int, min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        floor = self._now() - timedelta(hours=float(hours))
        rows = await self._pool.fetch(
            f"SELECT {_EPISODIC_COLUMNS} FROM episodic_memory "
            "WHERE created_at > $1 AND importance >= $2 AND memory_type <> $3 "
            "ORDER BY created_at DESC, id DESC LIMIT $4",
            floor,
            float(min_importance),
            ARCHIVED_TYPE,
            int(limit),
        )
        return [_episodic_row(row) for row in rows]

    async def unconsolidated(self, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_EPISODIC_COLUMNS} FROM episodic_memory "
            "WHERE NOT consolidated AND memory_type <> $1 "
            "ORDER BY created_at ASC, id ASC LIMIT $2",
            ARCHIVED_TYPE,
            int(limit),
        )
        return [_episodic_row(row) for row in rows]

    async def add_semantic(self, record: Dict[str, Any]) -> int:
        clean = _check_semantic(record)
        now = self._now()
        sql = """
            INSERT INTO semantic_memory (
                concept, knowledge, associations, strength, confidence,
                emotional_context, source_memories, created_at, updated_at,
                last_accessed
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::bigint[], $8, $8, $8)
            RETURNING id
            """
        args = (
            clean["concept"],
            clean["knowledge"],
            clean["associations"],
            clean["strength"],
            clean["confidence"],
            clean["emotional_context"],
            clean["source_memories"],
            now,
        )
        return await self._pool.execute_returning(sql, *args)

    async def get_semantic(self, ids: Sequence[int]) -> List[Dict[str, Any]]:
        """Return semantic rows by id (outside the protocol; tests and tools)."""
        wanted = _ids(ids)
        if not wanted:
            return []
        rows = await self._pool.fetch(
            f"SELECT {_SEMANTIC_COLUMNS} FROM semantic_memory "
            "WHERE id = ANY($1::bigint[])",
            wanted,
        )
        by_id = {int(row["id"]): _semantic_row(row) for row in rows}
        return [by_id[i] for i in wanted if i in by_id]

    async def mark_consolidated(
        self, ids: Sequence[int], strength: Optional[float]
    ) -> int:
        wanted = _ids(ids)
        if not wanted:
            return 0
        status = await self._pool.execute(
            """
            UPDATE episodic_memory
            SET consolidated = TRUE,
                consolidation_strength = COALESCE(
                    $1::double precision, consolidation_strength
                ),
                replay_count = replay_count + 1
            WHERE id = ANY($2::bigint[])
            """,
            None if strength is None else float(strength),
            wanted,
        )
        return _affected(status)

    async def touch(self, ids: Sequence[int]) -> None:
        wanted = _ids(ids)
        if not wanted:
            return
        await self._pool.execute(
            """
            UPDATE episodic_memory
            SET accessed_count = accessed_count + 1, last_accessed = $1
            WHERE id = ANY($2::bigint[])
            """,
            self._now(),
            wanted,
        )

    async def retention_candidates(
        self,
        older_than: datetime,
        limit: int,
        *,
        after: Optional[Tuple[datetime, int]] = None,
        exclude_types: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        excluded = [ARCHIVED_TYPE, *[str(t) for t in exclude_types]]
        args: List[Any] = [_aware(older_than, "older_than"), excluded, int(limit)]
        cursor_clause = ""
        if after is not None:
            moment, last_id = _cursor(after)
            args.extend([moment, last_id])
            cursor_clause = "AND (created_at, id) > ($4, $5) "
        rows = await self._pool.fetch(
            f"SELECT {_EPISODIC_COLUMNS} FROM episodic_memory "
            "WHERE created_at < $1 AND memory_type <> ALL($2::text[]) "
            f"{cursor_clause}"
            "ORDER BY created_at ASC, id ASC LIMIT $3",
            *args,
        )
        return [_episodic_row(row) for row in rows]

    async def archive(self, ids: Sequence[int]) -> int:
        wanted = _ids(ids)
        if not wanted:
            return 0
        status = await self._pool.execute(
            "UPDATE episodic_memory SET memory_type = $1 "
            "WHERE id = ANY($2::bigint[]) AND memory_type <> $1",
            ARCHIVED_TYPE,
            wanted,
        )
        return _affected(status)

    async def delete(self, ids: Sequence[int]) -> int:
        wanted = _ids(ids)
        if not wanted:
            return 0
        status = await self._pool.execute(
            "DELETE FROM episodic_memory WHERE id = ANY($1::bigint[])", wanted
        )
        return _affected(status)

    async def stats(self) -> Dict[str, Any]:
        row = await self._pool.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM core_memory) AS core_blocks,
                COUNT(*) FILTER (WHERE memory_type <> $1) AS episodic,
                COUNT(*) FILTER (WHERE memory_type = $1) AS archived,
                COUNT(*) FILTER (
                    WHERE memory_type <> $1 AND NOT consolidated
                ) AS unconsolidated,
                (SELECT COUNT(*) FROM semantic_memory) AS semantic,
                COALESCE(SUM(importance) FILTER (WHERE memory_type <> $1), 0)
                    AS total_importance
            FROM episodic_memory
            """,
            ARCHIVED_TYPE,
        )
        return {
            "core_blocks": int(row["core_blocks"]),
            "episodic": int(row["episodic"]),
            "archived": int(row["archived"]),
            "unconsolidated": int(row["unconsolidated"]),
            "semantic": int(row["semantic"]),
            "total_importance": float(row["total_importance"]),
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error``.

        An unreachable server is reported as unavailable with the error of the
        call that failed, until a later call reaches it.
        """
        return pool_status(self._pool)

    async def close(self) -> None:
        """Close the pool (awaitable-safe)."""
        result = self._pool.close()
        if hasattr(result, "__await__"):
            await result
