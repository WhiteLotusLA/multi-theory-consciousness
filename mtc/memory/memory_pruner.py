"""
Memory Pruner
=============

Selective forgetting across the store seams, run by the sleep scheduler (it
satisfies the scheduler's ``MemoryPruner`` protocol: ``prune(dry_run)``
returns integer counts).

- Episodic rows (``EpisodicStore``) older than ``min_age_days`` are scored by
  ``retention``; below ``delete_below`` a row is deleted, below
  ``archive_below`` it is archived (kept, hidden from recall and search).
  Rows of the memory type ``core`` (formed core memories) are never pruned,
  and the core memory blocks are never touched. A row without a readable
  ``created_at`` is not scored (its age is unknown) and is counted under
  ``episodic_unscored``.
- Candidates are read in pages of ``BATCH_SIZE`` with the store's
  ``after=(created_at, id)`` cursor, so rows that are kept never hide newer
  prunable rows. A pass reads at most ``MAX_BATCHES_PER_PASS`` pages; when it
  stops at that cap (``episodic_truncated``) the next pass resumes after the
  last row read, and a pass that reaches the end starts the next one from the
  oldest row again (a dry run reads from the same point and moves it
  nowhere). A store whose ``retention_candidates`` takes no cursor is
  read one page per pass, and ``status()`` names that (``paging``).
- A deleted row's vector point (``point_id_for_memory`` of the row id, so no
  search is needed) is deleted from the ``VectorStore`` and its memory node
  from the ``GraphStore``. A real pass reports the archive and delete counts
  the store confirmed (a call that fails reports 0), and the index deletes run
  only when the store confirmed at least one deleted row: a delete that
  confirms none (what a buffered delete reports) keeps the index entries and
  is counted as an error. A partial count proceeds, since the shortfall is
  rows already gone.
- Substrate telemetry older than ``neural_max_age_days``, thoughts older than
  ``thought_max_age_days`` with coherence below ``thought_max_coherence`` and
  no child thoughts, and unprotected gists whose stored retention is below
  ``gist_expire_below`` are removed through the ``NeuralRetentionStore``.
  Protected gists are never expired; the consolidation pipeline refreshes the
  stored gist retention before a prune.

Learning experiences are never touched (no learning store is passed). A
missing store is skipped and named in the result (``skipped_<store>``) and in
``status()``; a store failure is counted under ``errors`` and named in
``status()``, never raised. ``dry_run`` (the default) reports the planned
archive and delete counts under the same keys and changes nothing, so the
vector and graph counts stay 0.

Retention follows the consolidation pipeline's forgetting curve with true
half-lives: ``min(1, max(importance / 10 * 0.85, 0.5 ** (age / half_life) *
(1 + access_boost + recency_boost)))`` with a 14 day half-life (28 days once
consolidated), an access boost of 0.15 per access up to 0.5 and a recency
boost of up to 0.2 that halves every 7 days since the last access. The clock is
injectable. Thresholds are ``PruningConfig`` fields. No config keys; the stores
come from the caller.
"""

import asyncio
import inspect
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from mtc.memory.protocols import (
    TELEMETRY_COLLECTIONS,
    EpisodicStore,
    GraphStore,
    NeuralRetentionStore,
    VectorStore,
    point_id_for_memory,
)

logger = logging.getLogger(__name__)

PROTECTED_MEMORY_TYPES = ("core",)
BATCH_SIZE = 200
MAX_BATCHES_PER_PASS = 50
IMPORTANCE_SCALE_MAX = 10.0
EPISODIC_HALF_LIFE_DAYS = 14.0
CONSOLIDATED_HALF_LIFE_DAYS = 28.0
ACCESS_BOOST = 0.15
MAX_ACCESS_BOOST = 0.5
IMPORTANCE_FLOOR_FACTOR = 0.85
RECENCY_HALF_LIFE_DAYS = 7.0
RECENCY_BOOST = 0.2

STORE_NAMES = ("episodic", "vectors", "graph", "neural")


@dataclass
class PruningConfig:
    """Retention thresholds and age limits of a pruning pass."""

    archive_below: float = 0.2
    delete_below: float = 0.05
    min_age_days: float = 7.0
    neural_max_age_days: float = 30.0
    thought_max_age_days: float = 21.0
    thought_max_coherence: float = 0.5
    gist_expire_below: float = 0.1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _accepts_cursor(store: Any) -> bool:
    """True when the store's ``retention_candidates`` takes ``after``."""
    method = getattr(store, "retention_candidates", None)
    if method is None:
        return False
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "after" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class MemoryPruner:
    """Forgetting pass over episodic rows, their indexes and neural records."""

    def __init__(
        self,
        episodic: Optional[EpisodicStore] = None,
        vectors: Optional[VectorStore] = None,
        graph: Optional[GraphStore] = None,
        neural: Optional[NeuralRetentionStore] = None,
        config: Optional[PruningConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.episodic = episodic
        self.vectors = vectors
        self.graph = graph
        self.neural = neural
        self.config = config or PruningConfig()
        self._clock = clock or _utcnow
        self.last_result: Dict[str, int] = {}
        self.last_skipped: List[str] = [
            name for name in STORE_NAMES if getattr(self, name) is None
        ]
        self.last_errors: Dict[str, str] = {}
        self._warned: set = set()
        self._resume_after: Optional[Tuple[datetime, int]] = None
        self.paging = _accepts_cursor(episodic)

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the pruner clock must return aware datetimes")
        return moment.astimezone(timezone.utc)

    def retention(self, memory: Mapping[str, Any], now: datetime) -> float:
        """Retention strength in [0, 1] of an episodic row at ``now``.

        Raises ValueError when the row has no readable ``created_at``: its age
        is unknown, and it is never treated as fresh.
        """
        moment = _as_utc(now) or self._now()
        created = _as_utc(memory.get("created_at"))
        if created is None:
            raise ValueError("an episodic row without created_at cannot be scored")
        age_days = max(0.0, (moment - created).total_seconds() / 86400.0)
        importance = float(memory.get("importance", 0.0) or 0.0)
        floor = min(importance / IMPORTANCE_SCALE_MAX, 1.0) * IMPORTANCE_FLOOR_FACTOR
        half_life = (
            CONSOLIDATED_HALF_LIFE_DAYS
            if memory.get("consolidated")
            else EPISODIC_HALF_LIFE_DAYS
        )
        base = 0.5 ** (age_days / half_life)
        access = min(
            int(memory.get("accessed_count", 0) or 0) * ACCESS_BOOST, MAX_ACCESS_BOOST
        )
        last = _as_utc(memory.get("last_accessed"))
        recency = 0.0
        if last is not None:
            since = max(0.0, (moment - last).total_seconds() / 86400.0)
            recency = (0.5 ** (since / RECENCY_HALF_LIFE_DAYS)) * RECENCY_BOOST
        return min(1.0, max(floor, base * (1.0 + access + recency)))

    async def prune(self, dry_run: bool = True) -> Dict[str, int]:
        """Run one forgetting pass; return integer counts.

        ``episodic_archived`` and ``episodic_deleted`` are the planned counts
        on a dry run and the counts the store confirmed on a real pass.
        """
        now = self._now()
        result: Dict[str, int] = {
            "dry_run": int(bool(dry_run)),
            "episodic_scored": 0,
            "episodic_protected": 0,
            "episodic_unscored": 0,
            "episodic_batches": 0,
            "episodic_truncated": 0,
            "episodic_archived": 0,
            "episodic_deleted": 0,
            "vectors_deleted": 0,
            "graph_nodes_deleted": 0,
            "telemetry_deleted": 0,
            "thoughts_deleted": 0,
            "gists_expired": 0,
            "errors": 0,
        }
        for name in TELEMETRY_COLLECTIONS:
            result[f"telemetry_{name}_deleted"] = 0
        self.last_errors = {}
        self.last_skipped = [
            name for name in STORE_NAMES if getattr(self, name) is None
        ]
        for name in STORE_NAMES:
            result[f"skipped_{name}"] = int(name in self.last_skipped)
            if name in self.last_skipped and name not in self._warned:
                self._warned.add(name)
                logger.warning("memory pruner has no %s store; it is skipped", name)

        await self._prune_episodic(result, now, dry_run)
        await self._prune_neural(result, now, dry_run)
        result["errors"] = len(self.last_errors)
        self.last_result = dict(result)
        logger.info(
            "memory pruning (%s): archived %d, deleted %d, telemetry %d, "
            "thoughts %d, gists %d, errors %d",
            "dry run" if dry_run else "live",
            result["episodic_archived"],
            result["episodic_deleted"],
            result["telemetry_deleted"],
            result["thoughts_deleted"],
            result["gists_expired"],
            result["errors"],
        )
        return result

    def _fail(self, part: str, exc: BaseException) -> None:
        self.last_errors[part] = _describe(exc)
        logger.warning("memory pruning of %s failed: %s", part, _describe(exc))

    async def _prune_episodic(
        self, result: Dict[str, int], now: datetime, dry_run: bool
    ) -> None:
        if self.episodic is None:
            return
        cutoff = now - timedelta(days=self.config.min_age_days)
        to_archive: List[int] = []
        to_delete: List[int] = []
        after = self._resume_after if self.paging else None
        exhausted = False
        while result["episodic_batches"] < MAX_BATCHES_PER_PASS:
            try:
                if self.paging:
                    candidates = await self.episodic.retention_candidates(
                        cutoff, BATCH_SIZE, after=after
                    )
                else:
                    candidates = await self.episodic.retention_candidates(
                        cutoff, BATCH_SIZE
                    )
            except Exception as exc:  # noqa: BLE001 - counted, never raised
                self._fail("episodic", exc)
                return
            result["episodic_batches"] += 1
            for row in candidates:
                self._score(row, now, result, to_archive, to_delete)
            moved = False
            if candidates:
                last = candidates[-1]
                stamp = _as_utc(last.get("created_at"))
                if stamp is not None:
                    after = (stamp, int(last["id"]))
                    moved = True
            if len(candidates) < BATCH_SIZE or not moved:
                exhausted = True
                break
            if not self.paging:
                # A full page from a store without a cursor may hide more rows.
                break
        if not exhausted:
            result["episodic_truncated"] = 1
        if self.paging and not dry_run:
            # A dry run leaves the resume point where it was.
            self._resume_after = None if exhausted else after
        if dry_run:
            # A dry run reports the planned counts under the same keys.
            result["episodic_archived"] = len(to_archive)
            result["episodic_deleted"] = len(to_delete)
            return
        # A real pass reports what the store confirmed; a call that fails
        # leaves its count at 0.
        result["episodic_archived"] = 0
        result["episodic_deleted"] = 0
        try:
            if to_archive:
                result["episodic_archived"] = int(
                    await self.episodic.archive(to_archive)
                )
            if to_delete:
                result["episodic_deleted"] = int(await self.episodic.delete(to_delete))
        except Exception as exc:  # noqa: BLE001 - counted, never raised
            self._fail("episodic", exc)
            return
        if not to_delete:
            return
        if result["episodic_deleted"] == 0:
            # No row delete was confirmed (a buffered delete reports 0): the
            # rows may still exist, so their index entries stay.
            message = f"delete confirmed 0 of {len(to_delete)} rows; index entries kept"
            self.last_errors["episodic"] = message
            logger.warning("memory pruning of episodic: %s", message)
            return
        if self.vectors is not None:
            try:
                result["vectors_deleted"] = int(
                    await asyncio.to_thread(
                        self.vectors.delete_memories,
                        [point_id_for_memory(i) for i in to_delete],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - counted, never raised
                self._fail("vectors", exc)
        if self.graph is not None:
            try:
                result["graph_nodes_deleted"] = int(
                    await asyncio.to_thread(self.graph.delete_memory_nodes, to_delete)
                )
            except Exception as exc:  # noqa: BLE001 - counted, never raised
                self._fail("graph", exc)

    def _score(
        self,
        row: Mapping[str, Any],
        now: datetime,
        result: Dict[str, int],
        to_archive: List[int],
        to_delete: List[int],
    ) -> None:
        if row.get("memory_type") in PROTECTED_MEMORY_TYPES:
            result["episodic_scored"] += 1
            result["episodic_protected"] += 1
            return
        if _as_utc(row.get("created_at")) is None:
            result["episodic_unscored"] += 1
            return
        result["episodic_scored"] += 1
        retention = self.retention(row, now)
        if retention < self.config.delete_below:
            to_delete.append(int(row["id"]))
        elif retention < self.config.archive_below:
            to_archive.append(int(row["id"]))

    async def _prune_neural(
        self, result: Dict[str, int], now: datetime, dry_run: bool
    ) -> None:
        if self.neural is None:
            return
        cfg = self.config
        try:
            counts = await self.neural.prune_telemetry(
                now - timedelta(days=cfg.neural_max_age_days), dry_run
            )
            for name, count in counts.items():
                result[f"telemetry_{name}_deleted"] = int(count)
            result["telemetry_deleted"] = int(sum(counts.values()))
        except Exception as exc:  # noqa: BLE001 - counted, never raised
            self._fail("telemetry", exc)
        try:
            result["thoughts_deleted"] = int(
                await self.neural.prune_thoughts(
                    now - timedelta(days=cfg.thought_max_age_days),
                    cfg.thought_max_coherence,
                    dry_run,
                )
            )
        except Exception as exc:  # noqa: BLE001 - counted, never raised
            self._fail("thoughts", exc)
        try:
            result["gists_expired"] = int(
                await self.neural.expire_gists(cfg.gist_expire_below, dry_run)
            )
        except Exception as exc:  # noqa: BLE001 - counted, never raised
            self._fail("gists", exc)

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the last pass."""
        present = [name for name in STORE_NAMES if getattr(self, name) is not None]
        last_error = next(iter(self.last_errors.values()), None)
        return {
            "backend": "+".join(present) if present else "none",
            "available": self.episodic is not None or self.neural is not None,
            "last_error": last_error,
            "skipped": list(self.last_skipped),
            "paging": self.paging,
            "resume_after": (
                None
                if self._resume_after is None
                else [self._resume_after[0].isoformat(), self._resume_after[1]]
            ),
            "errors": dict(self.last_errors),
            "last_result": dict(self.last_result),
            "thresholds": asdict(self.config),
        }
