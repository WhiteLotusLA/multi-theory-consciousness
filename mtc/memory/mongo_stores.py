"""
MongoDB Document Stores
=======================

The MongoDB implementations of the document protocols in
``mtc.memory.protocols``: ``MongoLearningStore``, ``MongoSpeakerModelStore``,
``MongoGoalStore``, ``MongoAttentionGrabStore`` and ``MongoDreamStore``. Each
takes ``(database=None, settings=None)``: a pymongo ``Database``, or None to
build one from Settings through ``mtc.memory.clients`` (``MONGODB_HOST``,
``MONGODB_PORT``, ``MONGODB_DB`` and the credential keys, read at call time).
Building a store opens no connection; the first operation does. pymongo is the
optional ``mongo`` extra and is imported only through ``require_client`` inside
the client factory, so importing this module loads no driver.

Every async method runs the same pymongo call through ``asyncio.to_thread``.
A connection failure (the server was not reached) is raised as
``StoreUnavailable`` chained to the driver's error, which is what the resilient
wrappers buffer on; every other driver error propagates. Reads never answer
with an invented or empty stand-in for an unreachable server.

Document contract (pinned by ``tests/memory/test_store_contracts.py``):

- Lists are newest first. Append-only logs (experiences, grab entries, dreams)
  sort by the document ``_id`` descending, so the order is the write order.
  Keyed collections (beliefs, speaker models, goals) carry a hidden
  ``_written`` counter, set on every upsert, and sort by it descending, so
  newest means most recently written. A keyed collection is small and the sort
  is not indexed on purpose.
- Upserts replace the whole document under the unique key
  (``replace_one(..., upsert=True)``) and set the key field from the argument,
  so the argument wins over a different value inside the document.
- ``_id`` and ``_written`` are never returned. Documents handed in are copied
  first (the driver would otherwise add ``_id`` to the caller's dict).
- ``add_grab`` gives an entry without a ``grab_id`` a generated one: the unique
  index on ``grab_id`` would otherwise treat every missing id as the same null.

Schema artifact: ``DOCUMENT_INDEXES``, applied idempotently by each store's
``ensure_indexes()`` or ``ensure_document_indexes()`` as a safety net. Phase 7's
``init_mongo`` script is canonical. Speaker models, beliefs and grab entries
hold real names at runtime and are private data; they live in the operator's
MongoDB and nowhere in version control or logs. The stores log no document
content.
"""

import asyncio
import copy
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.core.config import Settings, get_settings
from mtc.memory.clients import (
    IndexSpec,
    apply_indexes,
    create_mongo_client,
    mongo_database,
)
from mtc.memory.protocols import StoreUnavailable

logger = logging.getLogger(__name__)

ASCENDING = 1
DESCENDING = -1

DOCUMENT_INDEXES: Dict[str, List[IndexSpec]] = {
    "learning_experiences": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("speaker", ASCENDING),)),
        IndexSpec((("outcome", ASCENDING),)),
        IndexSpec((("topic", ASCENDING),)),
    ],
    "learning_beliefs": [IndexSpec((("belief_id", ASCENDING),), unique=True)],
    "speaker_models": [IndexSpec((("speaker_name", ASCENDING),), unique=True)],
    "intrinsic_goals": [
        IndexSpec((("goal_id", ASCENDING),), unique=True),
        IndexSpec((("status", ASCENDING),)),
    ],
    "attention_grabs": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("grab_id", ASCENDING),), unique=True),
    ],
    "dreams": [IndexSpec((("timestamp", DESCENDING),))],
}

# Driver exception class names that mean the server was not reached. Matched by
# name so this module never imports the driver.
MONGO_UNREACHABLE_NAMES = frozenset(
    {
        "ConnectionFailure",
        "AutoReconnect",
        "ServerSelectionTimeoutError",
        "NetworkTimeout",
        "NotPrimaryError",
        "WaitQueueTimeoutError",
    }
)

_WRITTEN = "_written"
_HIDDEN = {"_id": 0, _WRITTEN: 0}

_written_lock = threading.Lock()
_last_written = 0


def is_mongo_unreachable(exc: BaseException) -> bool:
    """True when an exception means MongoDB could not be reached."""
    if isinstance(exc, (OSError, asyncio.TimeoutError)):
        return True
    return any(cls.__name__ in MONGO_UNREACHABLE_NAMES for cls in type(exc).__mro__)


def mongo_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a blocking driver function; a connection failure becomes
    ``StoreUnavailable`` chained to the driver's error. Every other error
    propagates unchanged."""
    try:
        return func(*args, **kwargs)
    except StoreUnavailable:
        raise
    except Exception as exc:
        if is_mongo_unreachable(exc):
            raise StoreUnavailable(
                f"MongoDB could not be reached: {type(exc).__name__}: {exc}"
            ) from exc
        raise


async def mongo_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``mongo_call`` on a worker thread."""
    return await asyncio.to_thread(mongo_call, func, *args, **kwargs)


def next_written() -> int:
    """A strictly increasing write stamp (nanoseconds, never repeated)."""
    global _last_written
    with _written_lock:
        _last_written = max(_last_written + 1, time.time_ns())
        return _last_written


def _lazy_database(settings: Optional[Settings] = None) -> Any:
    """The configured database of a client built with ``connect=False``.

    pymongo starts connecting in the background as soon as a client is built
    unless told not to, so the store builds the client itself and the first
    operation is what reaches the server.
    """
    client = create_mongo_client(settings, connect=False)
    return mongo_database(client, settings=settings)


def _bounded(limit: int) -> int:
    return max(int(limit), 0)


class MongoDocumentStore:
    """Shared plumbing: lazy database, thread offload and status."""

    def __init__(self, database: Any = None, settings: Optional[Settings] = None):
        self._settings = settings if settings is not None else get_settings()
        self._database = (
            database
            if database is not None
            else _lazy_database(settings=self._settings)
        )
        self._last_error: Optional[str] = None

    @property
    def database(self) -> Any:
        """The pymongo database the store reads and writes."""
        return self._database

    async def _run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            result = await mongo_thread(func, *args, **kwargs)
        except StoreUnavailable as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None
        return result

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        raise NotImplementedError

    def ensure_indexes(self) -> int:
        """Create this store's indexes if missing (idempotent); return the count."""
        return mongo_call(apply_indexes, self._database, self._indexes())

    def status(self) -> Dict[str, Any]:
        """``backend``, ``available`` (False after a failed call) and
        ``last_error``."""
        return {
            "backend": "mongodb",
            "available": self._last_error is None,
            "last_error": self._last_error,
        }


def _prepared(doc: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    stored = copy.deepcopy(doc)
    stored.update(fields)
    return stored


def _log_find(collection: Any, limit: int) -> List[Dict[str, Any]]:
    cursor = collection.find({}, _HIDDEN).sort("_id", DESCENDING).limit(limit)
    return list(cursor)


def _keyed_find(
    collection: Any, query: Dict[str, Any], limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    cursor = collection.find(query, _HIDDEN).sort(_WRITTEN, DESCENDING)
    if limit is not None:
        cursor = cursor.limit(limit)
    return list(cursor)


def _keyed_upsert(collection: Any, key_field: str, key: str, doc: Dict[str, Any]):
    stored = _prepared(doc, **{key_field: key, _WRITTEN: next_written()})
    collection.replace_one({key_field: key}, stored, upsert=True)


class MongoLearningStore(MongoDocumentStore):
    """Experiences (``learning_experiences``, append-only) and beliefs
    (``learning_beliefs``, unique ``belief_id``)."""

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {
            name: DOCUMENT_INDEXES[name]
            for name in ("learning_experiences", "learning_beliefs")
        }

    async def add_experience(self, doc: Dict[str, Any]) -> None:
        """Append one experience document."""
        stored = _prepared(doc)
        await self._run(self._database["learning_experiences"].insert_one, stored)

    async def recent_experiences(self, limit: int) -> List[Dict[str, Any]]:
        """Return the latest experiences, newest first."""
        size = _bounded(limit)
        if size == 0:
            return []
        return await self._run(_log_find, self._database["learning_experiences"], size)

    async def upsert_belief(self, belief_id: str, doc: Dict[str, Any]) -> None:
        """Store a belief under its id, replacing any previous document."""
        await self._run(
            _keyed_upsert,
            self._database["learning_beliefs"],
            "belief_id",
            belief_id,
            doc,
        )

    async def load_beliefs(self, limit: int) -> List[Dict[str, Any]]:
        """Return beliefs, most recently written first."""
        size = _bounded(limit)
        if size == 0:
            return []
        return await self._run(
            _keyed_find, self._database["learning_beliefs"], {}, size
        )


class MongoSpeakerModelStore(MongoDocumentStore):
    """Speaker models in ``speaker_models``, unique ``speaker_name``."""

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {"speaker_models": DOCUMENT_INDEXES["speaker_models"]}

    async def upsert_model(self, speaker_name: str, doc: Dict[str, Any]) -> None:
        """Store a model under the speaker name, replacing any previous one."""
        await self._run(
            _keyed_upsert,
            self._database["speaker_models"],
            "speaker_name",
            speaker_name,
            doc,
        )

    async def load_models(self) -> List[Dict[str, Any]]:
        """Return all models, most recently written first."""
        return await self._run(_keyed_find, self._database["speaker_models"], {})

    async def delete_model(self, speaker_name: str) -> bool:
        """Delete a model; True when one existed."""
        result = await self._run(
            self._database["speaker_models"].delete_one, {"speaker_name": speaker_name}
        )
        return result.deleted_count > 0


class MongoGoalStore(MongoDocumentStore):
    """Intrinsic goals in ``intrinsic_goals``, unique ``goal_id``."""

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {"intrinsic_goals": DOCUMENT_INDEXES["intrinsic_goals"]}

    async def upsert_goal(self, goal_id: str, doc: Dict[str, Any]) -> None:
        """Store a goal under its id, replacing any previous document."""
        await self._run(
            _keyed_upsert, self._database["intrinsic_goals"], "goal_id", goal_id, doc
        )

    async def load_goals(self, statuses: Sequence[str]) -> List[Dict[str, Any]]:
        """Return goals whose ``status`` is in ``statuses``, newest first."""
        wanted = list(statuses)
        if not wanted:
            return []
        return await self._run(
            _keyed_find, self._database["intrinsic_goals"], {"status": {"$in": wanted}}
        )


class MongoAttentionGrabStore(MongoDocumentStore):
    """Attention-capture journal entries in ``attention_grabs``."""

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {"attention_grabs": DOCUMENT_INDEXES["attention_grabs"]}

    async def add_grab(self, doc: Dict[str, Any]) -> None:
        """Append one entry; an entry with no ``grab_id`` gets a generated one."""
        stored = _prepared(doc)
        if not stored.get("grab_id"):
            stored["grab_id"] = uuid.uuid4().hex
        await self._run(self._database["attention_grabs"].insert_one, stored)

    async def recent_grabs(self, limit: int) -> List[Dict[str, Any]]:
        """Return the latest entries, newest first."""
        size = _bounded(limit)
        if size == 0:
            return []
        return await self._run(_log_find, self._database["attention_grabs"], size)


class MongoDreamStore(MongoDocumentStore):
    """Dream records in ``dreams``."""

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {"dreams": DOCUMENT_INDEXES["dreams"]}

    async def add_dream(self, doc: Dict[str, Any]) -> None:
        """Append one dream document."""
        await self._run(self._database["dreams"].insert_one, _prepared(doc))

    async def recent_dreams(self, limit: int) -> List[Dict[str, Any]]:
        """Return the latest dreams, newest first."""
        size = _bounded(limit)
        if size == 0:
            return []
        return await self._run(_log_find, self._database["dreams"], size)


def ensure_document_indexes(
    database: Any = None, settings: Optional[Settings] = None
) -> int:
    """Create every index in ``DOCUMENT_INDEXES`` (idempotent); return the count.

    A safety net: Phase 7's ``init_mongo`` script is the canonical schema.
    """
    target = database if database is not None else _lazy_database(settings=settings)
    return mongo_call(apply_indexes, target, DOCUMENT_INDEXES)
