"""
In-Memory Stores
================

Default implementations of the protocols in ``mtc.memory.protocols``. Each
store keeps at most ``max_items`` entries (default 10000) and drops the oldest
first. Documents and records are deep-copied on write and on read, so callers
never share state with the store. Nothing is persisted: a process restart
starts empty. Keyed stores (beliefs, speaker models, goals) move an upserted
key to the newest position, which makes "newest first" mean most recently
written. The documents hold real people's names at runtime and are private
data.
"""

import copy
from collections import OrderedDict, deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence

from mtc.memory.protocols import SnapshotEvent, SnapshotRecord

DEFAULT_MAX_ITEMS = 10000


def _aware(moment: datetime) -> datetime:
    """Return the instant in UTC; a naive datetime raises ValueError."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _check_cap(max_items: int) -> int:
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    return int(max_items)


class _KeyedDocs:
    """Keyed documents, most recently written last, oldest dropped at the cap."""

    def __init__(self, max_items: int) -> None:
        self._max_items = _check_cap(max_items)
        self._docs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def put(self, key: str, doc: Dict[str, Any]) -> None:
        self._docs.pop(key, None)
        self._docs[key] = copy.deepcopy(doc)
        while len(self._docs) > self._max_items:
            self._docs.popitem(last=False)

    def newest_first(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(doc) for doc in reversed(self._docs.values())]

    def delete(self, key: str) -> bool:
        return self._docs.pop(key, None) is not None


class _DocLog:
    """Append-only documents, newest first on read, oldest dropped at the cap."""

    def __init__(self, max_items: int) -> None:
        self._docs: Deque[Dict[str, Any]] = deque(maxlen=_check_cap(max_items))

    def append(self, doc: Dict[str, Any]) -> None:
        self._docs.append(copy.deepcopy(doc))

    def newest_first(self, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        items = list(self._docs)[::-1][:limit]
        return [copy.deepcopy(doc) for doc in items]


class InMemorySnapshotStore:
    """Snapshots and events held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        cap = _check_cap(max_items)
        self._snapshots: Deque[SnapshotRecord] = deque(maxlen=cap)
        self._events: Deque[SnapshotEvent] = deque(maxlen=cap)
        self._next_snapshot_id = 1
        self._next_event_id = 1

    async def add_snapshot(self, record: SnapshotRecord) -> int:
        stamp = _aware(record.timestamp)
        stored = replace(
            copy.deepcopy(record), timestamp=stamp, id=self._next_snapshot_id
        )
        self._next_snapshot_id += 1
        self._snapshots.append(stored)
        return stored.id

    async def add_event(self, event: SnapshotEvent) -> int:
        stamp = _aware(event.timestamp)
        stored = replace(copy.deepcopy(event), timestamp=stamp, id=self._next_event_id)
        self._next_event_id += 1
        self._events.append(stored)
        return stored.id

    @staticmethod
    def _newest_first(items: Sequence[Any]) -> List[Any]:
        # Stable sort keeps later insertions first among equal timestamps.
        return sorted(reversed(list(items)), key=lambda x: x.timestamp, reverse=True)

    async def latest_snapshot(self) -> Optional[SnapshotRecord]:
        ordered = self._newest_first(self._snapshots)
        return copy.deepcopy(ordered[0]) if ordered else None

    async def snapshots(
        self, since: Optional[datetime] = None, limit: int = 100
    ) -> List[SnapshotRecord]:
        floor = _aware(since) if since is not None else None
        ordered = self._newest_first(self._snapshots)
        if floor is not None:
            ordered = [r for r in ordered if r.timestamp >= floor]
        return [copy.deepcopy(r) for r in ordered[: max(limit, 0)]]

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[SnapshotEvent]:
        floor = _aware(since) if since is not None else None
        ordered = self._newest_first(self._events)
        if floor is not None:
            ordered = [e for e in ordered if e.timestamp >= floor]
        if event_type is not None:
            ordered = [e for e in ordered if e.event_type == event_type]
        return [copy.deepcopy(e) for e in ordered[: max(limit, 0)]]

    async def snapshot_before(self, moment: datetime) -> Optional[SnapshotRecord]:
        limit = _aware(moment)
        for record in self._newest_first(self._snapshots):
            if record.timestamp < limit:
                return copy.deepcopy(record)
        return None


class InMemoryLearningStore:
    """Experiences and beliefs held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._experiences = _DocLog(max_items)
        self._beliefs = _KeyedDocs(max_items)

    async def add_experience(self, doc: Dict[str, Any]) -> None:
        self._experiences.append(doc)

    async def recent_experiences(self, limit: int) -> List[Dict[str, Any]]:
        return self._experiences.newest_first(limit)

    async def upsert_belief(self, belief_id: str, doc: Dict[str, Any]) -> None:
        self._beliefs.put(belief_id, doc)

    async def load_beliefs(self, limit: int) -> List[Dict[str, Any]]:
        return self._beliefs.newest_first()[: max(limit, 0)]


class InMemorySpeakerModelStore:
    """Speaker models held in process memory, keyed by speaker name."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._models = _KeyedDocs(max_items)

    async def upsert_model(self, speaker_name: str, doc: Dict[str, Any]) -> None:
        self._models.put(speaker_name, doc)

    async def load_models(self) -> List[Dict[str, Any]]:
        return self._models.newest_first()

    async def delete_model(self, speaker_name: str) -> bool:
        return self._models.delete(speaker_name)


class InMemoryGoalStore:
    """Goals held in process memory, keyed by goal id."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._goals = _KeyedDocs(max_items)

    async def upsert_goal(self, goal_id: str, doc: Dict[str, Any]) -> None:
        self._goals.put(goal_id, doc)

    async def load_goals(self, statuses: Sequence[str]) -> List[Dict[str, Any]]:
        wanted = set(statuses)
        return [g for g in self._goals.newest_first() if g.get("status") in wanted]


class InMemoryAttentionGrabStore:
    """Attention-capture journal entries held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._grabs = _DocLog(max_items)

    async def add_grab(self, doc: Dict[str, Any]) -> None:
        self._grabs.append(doc)

    async def recent_grabs(self, limit: int) -> List[Dict[str, Any]]:
        return self._grabs.newest_first(limit)


class InMemoryDreamStore:
    """Dream records held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._dreams = _DocLog(max_items)

    async def add_dream(self, doc: Dict[str, Any]) -> None:
        self._dreams.append(doc)

    async def recent_dreams(self, limit: int) -> List[Dict[str, Any]]:
        return self._dreams.newest_first(limit)
