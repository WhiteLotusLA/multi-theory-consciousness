"""
Privacy Audit Store
===================

``MongoPrivacyAuditStore`` is the MongoDB implementation of
``mtc.memory.protocols.PrivacyAuditStore`` over the ``privacy_audit_log``
collection. It takes ``(database=None, settings=None)``: a pymongo
``Database``, or None to build one from Settings through
``mtc.memory.clients`` (``MONGODB_HOST``, ``MONGODB_PORT``, ``MONGODB_DB`` and
the credential keys, read when the store is built). Building the store opens no
connection; the first operation does. pymongo is the optional ``mongo`` extra
and is imported only inside the client factory.

Every async method runs its pymongo call through ``asyncio.to_thread``; a
connection failure is raised as ``StoreUnavailable`` chained to the driver's
error and every other driver error propagates (a duplicate ``event_id`` is the
driver's duplicate-key error). Events are read newest first by ``timestamp``,
later writes first among equal instants; ``_id`` is never returned.

Schema artifact: ``PRIVACY_INDEXES`` (``event_id`` unique, ``timestamp``
descending, ``user_id``, ``event_type``), applied idempotently by
``ensure_indexes()`` or ``ensure_privacy_indexes()`` as a safety net; the
``scripts/db/init_mongo.py`` script is canonical.

Private data: events hold user ids, roles and access records. They live in the
operator's MongoDB and nowhere in version control or logs; the store logs no
document content.
"""

import copy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from mtc.core.config import Settings
from mtc.memory.clients import IndexSpec, apply_indexes
from mtc.memory.mongo_stores import (
    ASCENDING,
    DESCENDING,
    MongoDocumentStore,
    mongo_call,
)

COLLECTION = "privacy_audit_log"

PRIVACY_INDEXES: Dict[str, List[IndexSpec]] = {
    COLLECTION: [
        IndexSpec((("event_id", ASCENDING),), unique=True),
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("user_id", ASCENDING),)),
        IndexSpec((("event_type", ASCENDING),)),
    ]
}

REQUIRED_FIELDS = ("event_id", "event_type", "user_id")

Clock = Callable[[], datetime]


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


def _find(collection: Any, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    cursor = (
        collection.find(query, {"_id": 0})
        .sort([("timestamp", DESCENDING), ("_id", DESCENDING)])
        .limit(limit)
    )
    return list(cursor)


class MongoPrivacyAuditStore(MongoDocumentStore):
    """Privacy audit events in ``privacy_audit_log`` (see the module docstring)."""

    def __init__(
        self,
        database: Any = None,
        settings: Optional[Settings] = None,
        *,
        clock: Optional[Clock] = None,
    ) -> None:
        super().__init__(database=database, settings=settings)
        self._clock = clock or _system_clock

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {COLLECTION: PRIVACY_INDEXES[COLLECTION]}

    async def add_event(self, event: Dict[str, Any]) -> None:
        """Store one event; a missing key field or a naive timestamp raises
        ValueError."""
        missing = [key for key in REQUIRED_FIELDS if event.get(key) in (None, "")]
        if missing:
            raise ValueError(f"privacy event lacks required fields {missing}")
        stored = copy.deepcopy(event)
        moment = stored.get("timestamp")
        stored["timestamp"] = _aware(self._clock() if moment is None else moment)
        await self._run(self._database[COLLECTION].insert_one, stored)

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return events at or after ``since``, newest first, optionally of one
        user and one type."""
        query: Dict[str, Any] = {}
        if since is not None:
            query["timestamp"] = {"$gte": _aware(since)}
        if user_id is not None:
            query["user_id"] = user_id
        if event_type is not None:
            query["event_type"] = event_type
        size = max(int(limit), 0)
        if size == 0:
            return []
        return await self._run(_find, self._database[COLLECTION], query, size)


def ensure_privacy_indexes(database: Any) -> int:
    """Create every index in ``PRIVACY_INDEXES`` (idempotent); return the count.

    A safety net: the ``scripts/db/init_mongo.py`` script is the canonical
    schema.
    """
    return mongo_call(apply_indexes, database, PRIVACY_INDEXES)
