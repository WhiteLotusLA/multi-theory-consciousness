"""
Alert Store (MongoDB)
=====================

``MongoAlertStore`` implements ``AlertStore`` (``mtc.memory.protocols``) over
the ``alerts`` collection. It takes ``(database=None, settings=None)`` like the
other MongoDB stores: a pymongo ``Database``, or None to build one lazily from
Settings through ``mtc.memory.clients`` (``MONGODB_HOST``, ``MONGODB_PORT``,
``MONGODB_DB`` and the credential keys, read at call time). Building a store
opens no connection; the first operation does. Every async method runs the
blocking driver call on a worker thread, and a connection failure surfaces as
``StoreUnavailable`` (the shape the resilient wrappers buffer on).

Contract (pinned by ``tests/memory/test_audit_store_contracts.py``): an alert
needs ``alert_id``, ``alert_type`` and a ``severity`` from
``ALERT_SEVERITY_ORDER`` (otherwise ValueError); a missing ``timestamp`` is the
store's current UTC time and a naive one raises ValueError; a second alert with
an ``alert_id`` already stored raises ValueError; reads are newest first (later
write first among equal instants), ``min_severity`` keeps that level and every
level above it, ``acknowledge`` sets ``acknowledged`` and ``acknowledged_at``
and reports whether the alert exists, and nothing returned shares state with
the store. ``_id`` is never returned.

Schema artifact: ``ALERT_INDEXES`` (the ``alerts`` collection: unique
``alert_id``; ``timestamp`` descending; ``alert_type``; ``severity``), applied
idempotently by ``ensure_indexes()`` as a safety net. Phase 7's ``init_mongo``
script is canonical and is written from this constant.

Alerts can carry private context (conversation excerpts, speaker names,
wellbeing readings); they live in the operator's MongoDB and nowhere in
version control. The store logs no alert content.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from mtc.core.config import Settings
from mtc.memory.clients import IndexSpec
from mtc.memory.mongo_stores import (
    ASCENDING,
    DESCENDING,
    MongoDocumentStore,
    _bounded,
    _prepared,
)
from mtc.memory.protocols import severities_at_or_above

logger = logging.getLogger(__name__)

ALERTS_COLLECTION = "alerts"

ALERT_INDEXES: Dict[str, List[IndexSpec]] = {
    ALERTS_COLLECTION: [
        IndexSpec((("alert_id", ASCENDING),), unique=True),
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("alert_type", ASCENDING),)),
        IndexSpec((("severity", ASCENDING),)),
    ]
}

_HIDDEN = {"_id": 0}
_REQUIRED = ("alert_id", "alert_type", "severity")


def _is_duplicate_key(exc: BaseException) -> bool:
    return any("DuplicateKey" in cls.__name__ for cls in type(exc).__mro__)


def _insert(collection: Any, document: Dict[str, Any]) -> None:
    try:
        collection.insert_one(document)
    except Exception as exc:
        if _is_duplicate_key(exc):
            raise ValueError(
                f"alert id {document.get('alert_id')!r} is already stored"
            ) from exc
        raise


def _find(collection: Any, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    cursor = collection.find(query, _HIDDEN)
    cursor = cursor.sort([("timestamp", DESCENDING), ("_id", DESCENDING)])
    return list(cursor.limit(limit))


class MongoAlertStore(MongoDocumentStore):
    """Alerts in the ``alerts`` collection, newest first."""

    def __init__(
        self,
        database: Any = None,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__(database=database, settings=settings)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return ALERT_INDEXES

    def _stamp(self, moment: Optional[datetime]) -> datetime:
        if moment is None:
            return self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("alert timestamps must be timezone-aware")
        return moment.astimezone(timezone.utc)

    async def add_alert(self, alert: Dict[str, Any]) -> None:
        """Store one alert (see the module docstring for the contract)."""
        missing = [key for key in _REQUIRED if alert.get(key) in (None, "")]
        if missing:
            raise ValueError(f"alert lacks required fields {missing}")
        severities_at_or_above(alert["severity"])
        stored = _prepared(alert)
        stored["timestamp"] = self._stamp(stored.get("timestamp"))
        await self._run(_insert, self._database[ALERTS_COLLECTION], stored)

    async def recent_alerts(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Alerts newest first, optionally of one type and at or above a
        severity (an unknown severity raises ValueError)."""
        query: Dict[str, Any] = {}
        if min_severity is not None:
            query["severity"] = {"$in": list(severities_at_or_above(min_severity))}
        if alert_type is not None:
            query["alert_type"] = alert_type
        size = _bounded(limit)
        if size == 0:
            return []
        return await self._run(_find, self._database[ALERTS_COLLECTION], query, size)

    async def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert acknowledged; False when no such alert exists."""
        result = await self._run(
            self._database[ALERTS_COLLECTION].update_one,
            {"alert_id": str(alert_id)},
            {"$set": {"acknowledged": True, "acknowledged_at": self._clock()}},
        )
        return result.matched_count > 0
