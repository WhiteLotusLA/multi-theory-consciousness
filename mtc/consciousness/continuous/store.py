"""Thought stores: the one place continuous thoughts persist.

``ThoughtStore`` is the async protocol the machine, the surfacing read and the
runtime use. ``InMemoryThoughtStore`` keeps thoughts in process (the default,
``CTM_STORE=memory``); ``MongoThoughtStore`` writes the ``thought_streams``
collection through the neural data manager (``NeuralDataManager`` of
``mtc.neural.mongodb_schemas``, ``CTM_STORE=mongo``). Both
keep the thought-stream document of ``ThoughtRecord.to_doc()``, so the pruner
and the consolidation pass read every writer's documents.

Contract (pinned by ``tests/consciousness/continuous/test_thought_store_contract.py``):

- ``insert(record)`` stores a copy and returns its ``thought_id``.
- ``recent(limit)`` returns the newest thoughts first, of every type; a limit
  of 0 or less returns an empty list.
- ``expressible(limit, exclude_type_prefixes)`` returns thoughts marked
  ``should_express`` whose type does not start with an excluded prefix, newest
  first, one per distinct text (the newest of a repeated text).
- ``bump_access(text)`` counts one more occurrence on the newest thought with
  exactly this text, stamps ``last_seen`` and returns its id (None when there is
  none), instead of storing a copy.
- ``link_child(parent_thought_id, child_thought_id)`` records the child on the
  parent found by its ``thought_id`` and returns False when no parent has that
  id. A burst id never resolves a parent.
- Records handed in and out are copies. ``max_items`` bounds the in-memory
  store (the oldest inserted thought is dropped first); the MongoDB store is
  bounded by the pruner, not by the store.
- ``count()`` and ``status()`` report the stored number and the backend state.

The MongoDB store runs every driver call through ``asyncio.to_thread``; an
unreachable server raises ``StoreUnavailable`` and a missing ``pymongo``
raises ``ImportError`` naming the ``mongo`` extra when the data manager is
built. A stored document it cannot read honestly (an unknown thought type, no
intensity) is skipped and counted in ``status()['skipped_unreadable']``.
Thought text, emotions and memory ids are private runtime data and are never
logged by this module. No config keys are read directly; the runtime picks the
backend from ``CTM_STORE``.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from mtc.consciousness.continuous.config import SurfacingConfig
from mtc.consciousness.continuous.schemas import ThoughtRecord
from mtc.core.config import Settings
from mtc.memory.mongo_stores import mongo_thread
from mtc.memory.protocols import StoreUnavailable

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITEMS = 10000
DEFAULT_EXCLUDED_PREFIXES: Tuple[str, ...] = ("dream",)
# How many times the MongoDB read doubles its window to fill a limit.
_MAX_WINDOW_ROUNDS = 6

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@runtime_checkable
class ThoughtStore(Protocol):
    """Where continuous thoughts persist (private content)."""

    async def insert(self, record: ThoughtRecord) -> str:
        """Store a copy of the record and return its thought id."""
        ...

    async def bump_access(self, text: str) -> Optional[str]:
        """Count one more occurrence of a thought text; return its id or None."""
        ...

    async def recent(self, limit: int) -> List[ThoughtRecord]:
        """Return the newest thoughts first."""
        ...

    async def expressible(
        self,
        limit: int,
        exclude_type_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
    ) -> List[ThoughtRecord]:
        """Return expressible thoughts, newest first, one per distinct text."""
        ...

    async def link_child(self, parent_thought_id: str, child_thought_id: str) -> bool:
        """Record a child on the parent with this thought id; False if absent."""
        ...

    async def count(self) -> int:
        """Return the number of stored thoughts."""
        ...

    def status(self) -> Dict[str, Any]:
        """Return the backend state (``backend``, ``available``, ``count``...)."""
        ...


def select_expressible(
    records: Iterable[ThoughtRecord],
    limit: int,
    exclude_type_prefixes: Sequence[str],
) -> List[ThoughtRecord]:
    """Filter newest-first records the way ``expressible`` is defined."""
    out: List[ThoughtRecord] = []
    seen = set()
    prefixes = tuple(exclude_type_prefixes)
    for record in records:
        if len(out) >= limit:
            break
        if not record.should_express:
            continue
        if prefixes and record.thought_type.value.startswith(prefixes):
            continue
        if record.content in seen:
            continue
        seen.add(record.content)
        out.append(record)
    return out


class InMemoryThoughtStore:
    """Thoughts held in process as thought-stream documents.

    ``max_items`` bounds the store (oldest inserted dropped first); ``clock``
    returns an aware UTC datetime and stamps ``last_seen`` on a bump. A
    duplicate thought id raises ``ValueError``.
    """

    def __init__(
        self, max_items: int = DEFAULT_MAX_ITEMS, clock: Optional[Clock] = None
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        self.max_items = max_items
        self._clock: Clock = clock or _utc_now
        self._docs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._order: Dict[str, int] = {}
        self._counter = 0
        self._dropped = 0

    async def insert(self, record: ThoughtRecord) -> str:
        if record.thought_id in self._docs:
            raise ValueError(f"thought id {record.thought_id!r} is already stored")
        self._docs[record.thought_id] = record.to_doc()
        self._counter += 1
        self._order[record.thought_id] = self._counter
        while len(self._docs) > self.max_items:
            oldest, _ = self._docs.popitem(last=False)
            self._order.pop(oldest, None)
            self._dropped += 1
        return record.thought_id

    def _newest_first(self) -> List[Dict[str, Any]]:
        return sorted(
            self._docs.values(),
            key=lambda d: (d["timestamp"], self._order[d["thought_id"]]),
            reverse=True,
        )

    async def bump_access(self, text: str) -> Optional[str]:
        for doc in self._newest_first():
            if doc["thought_text"] == text:
                doc["access_count"] = int(doc.get("access_count") or 0) + 1
                doc["metadata"]["last_seen"] = self._clock().astimezone(timezone.utc)
                return doc["thought_id"]
        return None

    async def recent(self, limit: int) -> List[ThoughtRecord]:
        if limit <= 0:
            return []
        return [ThoughtRecord.from_doc(d) for d in self._newest_first()[:limit]]

    async def expressible(
        self,
        limit: int,
        exclude_type_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
    ) -> List[ThoughtRecord]:
        if limit <= 0:
            return []
        records = (ThoughtRecord.from_doc(d) for d in self._newest_first())
        return select_expressible(records, limit, exclude_type_prefixes)

    async def link_child(self, parent_thought_id: str, child_thought_id: str) -> bool:
        doc = self._docs.get(parent_thought_id)
        if doc is None:
            return False
        if child_thought_id not in doc["child_thoughts"]:
            doc["child_thoughts"].append(child_thought_id)
        return True

    async def count(self) -> int:
        return len(self._docs)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": True,
            "count": len(self._docs),
            "max_items": self.max_items,
            "dropped": self._dropped,
        }


class MongoThoughtStore:
    """Thoughts in the ``thought_streams`` collection of the neural database.

    ``data_manager`` is the ``NeuralDataManager`` whose database handle
    and collection are used; None builds one from ``settings`` (which connects
    lazily and raises ``ImportError`` naming the ``mongo`` extra when
    ``pymongo`` is missing). No client is built at import. ``surfacing`` (the
    CTM config's surfacing section) sizes the window ``expressible`` reads:
    the newest ``max(min_window, limit * window_multiplier)`` thoughts, doubled
    until the limit is filled or the collection is exhausted.
    """

    def __init__(
        self,
        data_manager: Optional[Any] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Clock] = None,
        *,
        surfacing: Optional[SurfacingConfig] = None,
    ) -> None:
        if data_manager is None:
            from mtc.neural.mongodb_schemas import NeuralDataManager

            data_manager = NeuralDataManager(settings=settings)
        self._dm = data_manager
        self._clock: Clock = clock or _utc_now
        self._surfacing = surfacing if surfacing is not None else SurfacingConfig()
        self._skipped_unreadable = 0
        self._last_error: Optional[str] = None

    async def _run(self, func: Callable[..., Any], *args: Any) -> Any:
        try:
            result = await mongo_thread(func, *args)
        except StoreUnavailable as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None
        return result

    def _readable(self, docs: Iterable[Dict[str, Any]]) -> List[ThoughtRecord]:
        records: List[ThoughtRecord] = []
        for doc in docs:
            try:
                records.append(ThoughtRecord.from_doc(doc))
            except ValueError:
                self._skipped_unreadable += 1
        return records

    async def insert(self, record: ThoughtRecord) -> str:
        from mtc.neural.mongodb_schemas import ThoughtStream

        stored = await self._run(
            self._dm.store_thought, ThoughtStream(**record.to_doc())
        )
        if stored is None:
            self._last_error = "the thought was not stored: MongoDB is unavailable"
            raise StoreUnavailable(self._last_error)
        return record.thought_id

    async def bump_access(self, text: str) -> Optional[str]:
        found = await self._run(self._dm.touch_thought, text, self._clock())
        if found is None:
            return None
        return found.get("thought_id") or found.get("id")

    async def recent(self, limit: int) -> List[ThoughtRecord]:
        if limit <= 0:
            return []
        docs = await self._run(self._dm.find_thoughts, {}, limit)
        return self._readable(docs)

    async def expressible(
        self,
        limit: int,
        exclude_type_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
    ) -> List[ThoughtRecord]:
        if limit <= 0:
            return []
        window = max(
            self._surfacing.min_window, limit * self._surfacing.window_multiplier
        )
        query = {"metadata.should_express": True}
        chosen: List[ThoughtRecord] = []
        for _ in range(_MAX_WINDOW_ROUNDS):
            docs = await self._run(self._dm.find_thoughts, query, window)
            chosen = select_expressible(
                self._readable(docs), limit, exclude_type_prefixes
            )
            if len(chosen) >= limit or len(docs) < window:
                break
            window *= 2
        return chosen

    async def link_child(self, parent_thought_id: str, child_thought_id: str) -> bool:
        return bool(
            await self._run(
                self._dm.link_child_thought, parent_thought_id, child_thought_id
            )
        )

    async def count(self) -> int:
        return int(await self._run(self._dm.thought_streams.count_documents, {}))

    def status(self) -> Dict[str, Any]:
        manager = copy.copy(self._dm.status())
        manager["backend"] = "mongo"
        manager["skipped_unreadable"] = self._skipped_unreadable
        manager["store_error"] = self._last_error
        return manager


__all__ = (
    "DEFAULT_EXCLUDED_PREFIXES",
    "InMemoryThoughtStore",
    "MongoThoughtStore",
    "ThoughtStore",
    "select_expressible",
)
