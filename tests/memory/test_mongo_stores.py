"""MongoDB document stores over a fake driver; no server is ever contacted.

``FakeMongoClient`` is a small in-memory stand-in for the part of pymongo the
stores use (filters with comparison, ``$in``, ``$or`` and ``$exists``, sorting,
limits, ``$set``, ``$inc`` and ``$addToSet`` updates, upserts, unique indexes
and a three-stage ``aggregate``). It records every call in ``calls`` so tests
pin query shapes, mimics BSON datetime handling (aware values become UTC with
millisecond precision, a client with ``tz_aware`` returns aware values) and adds
an ``_id`` to inserted documents the way the driver does. Other test modules
import it from here. It is a model of the driver, not of a server: the ``db``
marked contract cases are what check a real MongoDB.
"""

import asyncio
import copy
import itertools
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from mtc.memory import mongo_stores
from mtc.memory.clients import IndexSpec
from mtc.memory.mongo_stores import (
    DOCUMENT_INDEXES,
    MongoAttentionGrabStore,
    MongoDreamStore,
    MongoGoalStore,
    MongoLearningStore,
    MongoSpeakerModelStore,
)
from mtc.memory.protocols import StoreUnavailable

_MISSING = object()
_ids = itertools.count(1)


class FakeDuplicateKeyError(Exception):
    """Stands in for pymongo's DuplicateKeyError."""


class ServerSelectionTimeoutError(Exception):
    """Same class name as the driver's; the stores match failures by name."""


class FakeObjectId:
    """A monotonically increasing id that prints like a hex string."""

    def __init__(self) -> None:
        self.value = next(_ids)

    def __str__(self) -> str:
        return f"{self.value:024x}"

    def __repr__(self) -> str:
        return f"FakeObjectId({self})"

    def __lt__(self, other: "FakeObjectId") -> bool:
        return self.value < other.value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, FakeObjectId) and other.value == self.value

    def __hash__(self) -> int:
        return hash(self.value)


def _bson_datetime(value: datetime) -> datetime:
    """Aware values become UTC, naive ones are read as UTC, milliseconds only."""
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(microsecond=(value.microsecond // 1000) * 1000)


def _normalize(value: Any) -> Any:
    if isinstance(value, datetime):
        return _bson_datetime(value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return copy.deepcopy(value)


def _lookup(doc: Any, path: str) -> Any:
    current = doc
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _comparable(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool)
    numbers = (int, float)
    if isinstance(left, numbers) and isinstance(right, numbers):
        return True
    return type(left) is type(right) and isinstance(left, (str, datetime))


def _equal(value: Any, wanted: Any) -> bool:
    if isinstance(wanted, datetime):
        wanted = _bson_datetime(wanted)
    if value == wanted:
        return True
    return isinstance(value, list) and wanted in value


def _match_condition(value: Any, condition: Any) -> bool:
    if not (
        isinstance(condition, dict)
        and condition
        and next(iter(condition)).startswith("$")
    ):
        return (
            value is not _MISSING
            and _equal(value, condition)
            or (value is _MISSING and condition is None)
        )
    for operator, operand in condition.items():
        if isinstance(operand, datetime):
            operand = _bson_datetime(operand)
        if operator == "$exists":
            if (value is not _MISSING) != bool(operand):
                return False
        elif operator == "$eq":
            if not _match_condition(value, operand):
                return False
        elif operator == "$ne":
            if value is not _MISSING and _equal(value, operand):
                return False
        elif operator == "$in":
            options = list(operand)
            if value is _MISSING:
                if None not in options:
                    return False
            elif not any(_equal(value, option) for option in options):
                return False
        elif operator == "$nin":
            if value is not _MISSING and any(_equal(value, o) for o in operand):
                return False
        elif operator in ("$lt", "$lte", "$gt", "$gte"):
            if value is _MISSING or not _comparable(value, operand):
                return False
            ok = {
                "$lt": value < operand,
                "$lte": value <= operand,
                "$gt": value > operand,
                "$gte": value >= operand,
            }[operator]
            if not ok:
                return False
        else:
            raise NotImplementedError(f"fake does not implement {operator}")
    return True


def _matches(doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
    for key, condition in query.items():
        if key == "$or":
            if not any(_matches(doc, sub) for sub in condition):
                return False
        elif key == "$and":
            if not all(_matches(doc, sub) for sub in condition):
                return False
        elif not _match_condition(_lookup(doc, key), condition):
            return False
    return True


def _sort_key(value: Any) -> Any:
    if value is _MISSING or value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (3, value)
    if isinstance(value, (int, float)):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    if isinstance(value, datetime):
        return (4, value)
    return (5, str(value))


def _project(doc: Dict[str, Any], projection: Optional[Dict[str, int]]):
    out = copy.deepcopy(doc)
    if projection:
        for field, flag in projection.items():
            if not flag:
                out.pop(field, None)
    return out


class FakeCursor:
    def __init__(self, docs: List[Dict[str, Any]], projection=None) -> None:
        self._docs = docs
        self._projection = projection
        self._limit = 0

    def sort(self, key_or_list: Any, direction: Optional[int] = None):
        keys = (
            [(key_or_list, direction)] if direction is not None else list(key_or_list)
        )
        for field, way in reversed(keys):
            self._docs.sort(
                key=lambda d, f=field: _sort_key(_lookup(d, f)), reverse=way < 0
            )
        return self

    def limit(self, count: int):
        self._limit = int(count)
        return self

    def __iter__(self):
        docs = self._docs[: self._limit] if self._limit else self._docs
        return iter(_project(d, self._projection) for d in docs)


class FakeResult:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _apply_update(doc: Dict[str, Any], update: Dict[str, Any]) -> None:
    for operator, fields in update.items():
        for path, value in fields.items():
            parts = path.split(".")
            target = doc
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            leaf = parts[-1]
            value = _normalize(value)
            if operator == "$set":
                target[leaf] = value
            elif operator == "$inc":
                target[leaf] = target.get(leaf, 0) + value
            elif operator == "$addToSet":
                current = target.setdefault(leaf, [])
                if value not in current:
                    current.append(value)
            else:
                raise NotImplementedError(f"fake does not implement {operator}")


class FakeCollection:
    def __init__(self, database: "FakeDatabase", name: str) -> None:
        self.database = database
        self.name = name
        self.docs: List[Dict[str, Any]] = []
        self.indexes: List[Dict[str, Any]] = []

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.database.client.check_reachable()
        self.database.client.calls.append((self.name, method, args, kwargs))

    def _check_unique(self, candidate: Dict[str, Any], skip: Any = None) -> None:
        for index in self.indexes:
            if not index["unique"]:
                continue
            fields = [key for key, _ in index["keys"]]
            probe = tuple(
                None if (v := _lookup(candidate, f)) is _MISSING else v for f in fields
            )
            for other in self.docs:
                if other is skip:
                    continue
                same = tuple(
                    None if (v := _lookup(other, f)) is _MISSING else v for f in fields
                )
                if same == probe:
                    raise FakeDuplicateKeyError(f"duplicate key {fields}")

    def create_index(self, keys: Any, **options: Any) -> str:
        self._record("create_index", keys, **options)
        spec = {
            "keys": [tuple(pair) for pair in keys],
            "unique": bool(options.get("unique", False)),
        }
        if spec not in self.indexes:
            self.indexes.append(spec)
        return "index"

    def insert_one(self, doc: Dict[str, Any]):
        self._record("insert_one", copy.deepcopy(doc))
        stored = _normalize(doc)
        stored.setdefault("_id", FakeObjectId())
        self._check_unique(stored)
        doc["_id"] = stored["_id"]
        self.docs.append(stored)
        return FakeResult(inserted_id=stored["_id"])

    def insert_many(self, docs: List[Dict[str, Any]]):
        self._record("insert_many", copy.deepcopy(docs))
        ids = []
        for doc in docs:
            stored = _normalize(doc)
            stored.setdefault("_id", FakeObjectId())
            self._check_unique(stored)
            doc["_id"] = stored["_id"]
            self.docs.append(stored)
            ids.append(stored["_id"])
        return FakeResult(inserted_ids=ids)

    def find(self, query: Optional[Dict[str, Any]] = None, projection=None):
        self._record("find", copy.deepcopy(query), projection)
        found = [d for d in self.docs if _matches(d, query or {})]
        return FakeCursor(found, projection)

    def find_one(self, query=None, projection=None, sort=None):
        self._record("find_one", copy.deepcopy(query), projection, sort)
        found = [d for d in self.docs if _matches(d, query or {})]
        cursor = FakeCursor(found, projection)
        if sort:
            cursor.sort(sort)
        for doc in cursor.limit(1):
            return doc
        return None

    def _first(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for doc in self.docs:
            if _matches(doc, query):
                return doc
        return None

    def find_one_and_update(self, query, update, sort=None, upsert=False):
        self._record("find_one_and_update", copy.deepcopy(query), update, sort)
        found = [d for d in self.docs if _matches(d, query)]
        cursor = FakeCursor(found)
        if sort:
            cursor.sort(sort)
        target = found[0] if found else None
        if target is None:
            return None
        before = copy.deepcopy(target)
        _apply_update(target, update)
        return before

    def update_one(self, query, update, upsert=False):
        self._record("update_one", copy.deepcopy(query), update, upsert=upsert)
        target = self._first(query)
        if target is None:
            return FakeResult(matched_count=0, modified_count=0, upserted_id=None)
        _apply_update(target, update)
        return FakeResult(matched_count=1, modified_count=1, upserted_id=None)

    def update_many(self, query, update):
        self._record("update_many", copy.deepcopy(query), update)
        targets = [d for d in self.docs if _matches(d, query)]
        for target in targets:
            _apply_update(target, update)
        return FakeResult(matched_count=len(targets), modified_count=len(targets))

    def replace_one(self, query, replacement, upsert=False):
        self._record("replace_one", copy.deepcopy(query), copy.deepcopy(replacement))
        target = self._first(query)
        stored = _normalize(replacement)
        if target is None:
            if not upsert:
                return FakeResult(matched_count=0, modified_count=0, upserted_id=None)
            stored["_id"] = FakeObjectId()
            self._check_unique(stored)
            self.docs.append(stored)
            return FakeResult(
                matched_count=0, modified_count=0, upserted_id=stored["_id"]
            )
        stored["_id"] = target["_id"]
        self._check_unique(stored, skip=target)
        target.clear()
        target.update(stored)
        return FakeResult(matched_count=1, modified_count=1, upserted_id=None)

    def delete_one(self, query):
        self._record("delete_one", copy.deepcopy(query))
        target = self._first(query)
        if target is None:
            return FakeResult(deleted_count=0)
        self.docs.remove(target)
        return FakeResult(deleted_count=1)

    def delete_many(self, query):
        self._record("delete_many", copy.deepcopy(query))
        doomed = [d for d in self.docs if _matches(d, query)]
        for doc in doomed:
            self.docs.remove(doc)
        return FakeResult(deleted_count=len(doomed))

    def count_documents(self, query):
        self._record("count_documents", copy.deepcopy(query))
        return sum(1 for d in self.docs if _matches(d, query))

    def aggregate(self, pipeline):
        self._record("aggregate", copy.deepcopy(pipeline))
        rows = [copy.deepcopy(d) for d in self.docs]
        for stage in pipeline:
            if "$collStats" in stage:
                rows = [{"storageStats": {"size": len(str(self.docs))}}]
            elif "$match" in stage:
                rows = [r for r in rows if _matches(r, stage["$match"])]
            elif "$group" in stage:
                spec = stage["$group"]
                field = spec["_id"].lstrip("$")
                groups: Dict[Any, List[Dict[str, Any]]] = {}
                for row in rows:
                    value = _lookup(row, field)
                    groups.setdefault(None if value is _MISSING else value, []).append(
                        row
                    )
                rows = []
                for key, members in groups.items():
                    out: Dict[str, Any] = {"_id": key}
                    for name, accumulator in spec.items():
                        if name == "_id":
                            continue
                        ((op, operand),) = accumulator.items()
                        if op == "$sum":
                            out[name] = sum(
                                operand if isinstance(operand, int) else 0
                                for _ in members
                            )
                        else:
                            source = operand.lstrip("$")
                            values = [
                                v
                                for m in members
                                if (v := _lookup(m, source)) is not _MISSING
                            ]
                            out[name] = (
                                sum(values) / len(values)
                                if op == "$avg"
                                else max(values)
                            )
                    rows.append(out)
            elif "$sort" in stage:
                for field, way in reversed(list(stage["$sort"].items())):
                    rows.sort(
                        key=lambda r, f=field: _sort_key(_lookup(r, f)),
                        reverse=way < 0,
                    )
            else:
                raise NotImplementedError(f"fake does not implement {stage}")
        return iter(rows)


class FakeDatabase:
    def __init__(self, client: "FakeMongoClient", name: str) -> None:
        self.client = client
        self.name = name
        self.collections: Dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection(self, name)
        return self.collections[name]

    def __getattr__(self, name: str) -> FakeCollection:
        if name.startswith("_") or name in ("client", "name", "collections"):
            raise AttributeError(name)
        return self[name]

    def command(self, name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.client.check_reachable()
        self.client.calls.append(("<db>", "command", (name, *args), kwargs))
        return {"ok": 1}


class FakeMongoClient:
    """A client with ``tz_aware`` semantics, a call log and an outage switch."""

    def __init__(self) -> None:
        self.databases: Dict[str, FakeDatabase] = {}
        self.calls: List[Any] = []
        self.unreachable = False
        self.closed = False

    def check_reachable(self) -> None:
        if self.unreachable:
            raise ServerSelectionTimeoutError("no server available (fake)")

    def __getitem__(self, name: str) -> FakeDatabase:
        if name not in self.databases:
            self.databases[name] = FakeDatabase(self, name)
        return self.databases[name]

    def close(self) -> None:
        self.closed = True

    def methods(self, name: Optional[str] = None) -> List[str]:
        return [c[1] for c in self.calls if name is None or c[0] == name]


@pytest.fixture
def fake_client() -> FakeMongoClient:
    return FakeMongoClient()


@pytest.fixture
def fake_database(fake_client) -> FakeDatabase:
    return fake_client["fake_db"]


STORE_CLASSES = (
    MongoLearningStore,
    MongoSpeakerModelStore,
    MongoGoalStore,
    MongoAttentionGrabStore,
    MongoDreamStore,
)


def _specs(indexes) -> List[Any]:
    return [(tuple(spec.keys), spec.unique) for spec in indexes]


def test_indexes_match_artifact(fake_database):
    for cls in STORE_CLASSES:
        cls(database=fake_database).ensure_indexes()
    for collection, specs in DOCUMENT_INDEXES.items():
        built = [
            (tuple(index["keys"]), index["unique"])
            for index in fake_database[collection].indexes
        ]
        assert built == _specs(specs), collection
    assert set(DOCUMENT_INDEXES) == {
        "learning_experiences",
        "learning_beliefs",
        "speaker_models",
        "intrinsic_goals",
        "attention_grabs",
        "dreams",
    }
    # the indexes the reference never created
    assert (("belief_id", 1),) in [
        s[0] for s in _specs(DOCUMENT_INDEXES["learning_beliefs"])
    ]
    assert (("grab_id", 1),) in [
        s[0] for s in _specs(DOCUMENT_INDEXES["attention_grabs"])
    ]
    assert (("timestamp", -1),) in [s[0] for s in _specs(DOCUMENT_INDEXES["dreams"])]
    # idempotent
    before = {name: list(c.indexes) for name, c in fake_database.collections.items()}
    for cls in STORE_CLASSES:
        cls(database=fake_database).ensure_indexes()
    after = {name: list(c.indexes) for name, c in fake_database.collections.items()}
    assert before == after


def test_module_level_ensure_applies_every_collection(fake_database):
    count = mongo_stores.ensure_document_indexes(fake_database)
    assert count == sum(len(specs) for specs in DOCUMENT_INDEXES.values())


@pytest.mark.asyncio
async def test_id_field_never_returned(fake_database):
    learning = MongoLearningStore(database=fake_database)
    await learning.add_experience({"speaker": "Alice", "outcome": "ok"})
    await learning.upsert_belief("b1", {"confidence": 0.5})
    speakers = MongoSpeakerModelStore(database=fake_database)
    await speakers.upsert_model("Alice", {"trust": 0.4})
    goals = MongoGoalStore(database=fake_database)
    await goals.upsert_goal("g1", {"status": "active"})
    grabs = MongoAttentionGrabStore(database=fake_database)
    await grabs.add_grab({"note": "n"})
    dreams = MongoDreamStore(database=fake_database)
    await dreams.add_dream({"text": "d"})
    documents = (
        await learning.recent_experiences(5)
        + await learning.load_beliefs(5)
        + await speakers.load_models()
        + await goals.load_goals(["active"])
        + await grabs.recent_grabs(5)
        + await dreams.recent_dreams(5)
    )
    assert len(documents) == 6
    for doc in documents:
        assert "_id" not in doc and "_written" not in doc, doc
    # the stored documents do carry them (so the projection is what hides them)
    assert "_id" in fake_database["learning_beliefs"].docs[0]
    assert "_written" in fake_database["learning_beliefs"].docs[0]


@pytest.mark.asyncio
async def test_documents_handed_in_are_not_mutated(fake_database):
    doc = {"note": "n", "nested": {"values": [1]}}
    await MongoDreamStore(database=fake_database).add_dream(doc)
    assert doc == {"note": "n", "nested": {"values": [1]}}


@pytest.mark.asyncio
async def test_upserts_set_on_the_unique_key(fake_client, fake_database):
    learning = MongoLearningStore(database=fake_database)
    speakers = MongoSpeakerModelStore(database=fake_database)
    goals = MongoGoalStore(database=fake_database)
    await learning.upsert_belief("b1", {"belief_id": "other", "confidence": 0.2})
    await learning.upsert_belief("b1", {"confidence": 0.9})
    await speakers.upsert_model("Alice", {"trust": 0.5})
    await goals.upsert_goal("g1", {"status": "active"})
    replaced = [c for c in fake_client.calls if c[1] == "replace_one"]
    assert [(c[0], c[2][0]) for c in replaced] == [
        ("learning_beliefs", {"belief_id": "b1"}),
        ("learning_beliefs", {"belief_id": "b1"}),
        ("speaker_models", {"speaker_name": "Alice"}),
        ("intrinsic_goals", {"goal_id": "g1"}),
    ]
    assert fake_client.methods("learning_beliefs").count("insert_one") == 0
    beliefs = await learning.load_beliefs(10)
    assert beliefs == [{"belief_id": "b1", "confidence": 0.9}]  # the argument won


@pytest.mark.asyncio
async def test_lists_sorted_newest_first(fake_database):
    learning = MongoLearningStore(database=fake_database)
    for index in range(3):
        await learning.add_experience({"n": index})
        await learning.upsert_belief(f"b{index}", {"n": index})
    await learning.upsert_belief("b0", {"n": 9})  # rewriting moves it to the front
    assert [d["n"] for d in await learning.recent_experiences(10)] == [2, 1, 0]
    assert [d["n"] for d in await learning.load_beliefs(10)] == [9, 2, 1]
    assert await learning.recent_experiences(0) == []
    assert await learning.load_beliefs(-1) == []
    sorts = [
        c
        for c in fake_database.client.calls
        if c[1] == "find" and c[0] in ("learning_experiences", "learning_beliefs")
    ]
    assert len(sorts) == 2


@pytest.mark.asyncio
async def test_grab_without_id_gets_one_and_ids_stay_unique(fake_database):
    grabs = MongoAttentionGrabStore(database=fake_database)
    grabs.ensure_indexes()
    await grabs.add_grab({"n": 1})
    await grabs.add_grab({"n": 2})
    await grabs.add_grab({"n": 3, "grab_id": "kept"})
    entries = await grabs.recent_grabs(10)
    assert [e["n"] for e in entries] == [3, 2, 1]
    assert entries[0]["grab_id"] == "kept"
    assert len({e["grab_id"] for e in entries}) == 3


@pytest.mark.asyncio
async def test_goals_filter_by_status_and_delete_reports(fake_database):
    goals = MongoGoalStore(database=fake_database)
    await goals.upsert_goal("a", {"status": "active"})
    await goals.upsert_goal("p", {"status": "paused"})
    assert {g["goal_id"] for g in await goals.load_goals(["active"])} == {"a"}
    assert await goals.load_goals([]) == []
    speakers = MongoSpeakerModelStore(database=fake_database)
    await speakers.upsert_model("Alice", {"trust": 0.5})
    assert await speakers.delete_model("Alice") is True
    assert await speakers.delete_model("Alice") is False


@pytest.mark.asyncio
async def test_unreachable_server_raises_store_unavailable_and_status_names_it(
    fake_client, fake_database
):
    store = MongoDreamStore(database=fake_database)
    assert store.status() == {
        "backend": "mongodb",
        "available": True,
        "last_error": None,
    }
    fake_client.unreachable = True
    with pytest.raises(StoreUnavailable) as caught:
        await store.recent_dreams(3)
    assert isinstance(caught.value.__cause__, ServerSelectionTimeoutError)
    status = store.status()
    assert status["available"] is False and "ServerSelectionTimeoutError" in str(
        status["last_error"]
    )
    fake_client.unreachable = False
    assert await store.recent_dreams(3) == []
    assert store.status()["available"] is True


@pytest.mark.asyncio
async def test_data_errors_propagate_unchanged(fake_database):
    grabs = MongoAttentionGrabStore(database=fake_database)
    grabs.ensure_indexes()
    await grabs.add_grab({"grab_id": "x"})
    with pytest.raises(FakeDuplicateKeyError):
        await grabs.add_grab({"grab_id": "x"})
    assert grabs.status()["available"] is True


def test_stores_satisfy_their_protocols(fake_database):
    from mtc.memory import protocols

    pairs = (
        (MongoLearningStore, protocols.LearningStore),
        (MongoSpeakerModelStore, protocols.SpeakerModelStore),
        (MongoGoalStore, protocols.GoalStore),
        (MongoAttentionGrabStore, protocols.AttentionGrabStore),
        (MongoDreamStore, protocols.DreamStore),
    )
    for cls, protocol in pairs:
        assert isinstance(cls(database=fake_database), protocol)


def test_construction_connects_nowhere(fake_client, fake_database):
    for cls in STORE_CLASSES:
        cls(database=fake_database)
    assert fake_client.calls == []


def test_default_stores_build_their_client_with_connect_false(monkeypatch):
    """Without a database the store builds its own client, told not to connect."""
    from mtc.memory import clients

    built = FakeMongoClient()
    seen: List[Dict[str, Any]] = []

    def factory(settings=None, **options):
        seen.append(options)
        return built

    monkeypatch.setattr(mongo_stores, "create_mongo_client", factory)
    for cls in STORE_CLASSES:
        cls()
    assert seen == [{"connect": False}] * len(STORE_CLASSES)
    assert built.calls == []
    assert clients.create_mongo_client is not factory  # patched in one place only


def test_real_driver_default_store_opens_no_socket(monkeypatch):
    """pymongo connects in the background unless told not to; none may open."""
    pytest.importorskip("pymongo")
    import socket
    import time

    from mtc.core.config import Settings

    attempts: List[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> None:
        attempts.append(args)
        raise OSError("no sockets in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)  # the lookup comes first
    store = MongoLearningStore(settings=Settings(MONGODB_HOST="mongo.invalid"))
    time.sleep(0.5)
    assert store.database is not None
    assert attempts == []


def test_missing_pymongo_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymongo", None)
    with pytest.raises(ImportError) as caught:
        MongoLearningStore()
    assert "mongo" in str(caught.value) and "pymongo" in str(caught.value)


def test_module_import_loads_no_pymongo():
    code = (
        "import sys, mtc.memory.mongo_stores;"
        "sys.exit(1 if 'pymongo' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-1000:]


def test_no_private_field_or_filter():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mongo_stores))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            assert "private" not in node.value.lower(), node.value


def test_index_spec_type_is_shared_with_clients():
    assert all(
        isinstance(spec, IndexSpec)
        for specs in DOCUMENT_INDEXES.values()
        for spec in specs
    )


def test_runs_on_a_fresh_loop():
    # the stores offload to a worker thread; a loop other than the fixture's works
    async def scenario() -> int:
        database = FakeMongoClient()["loop_check"]
        store = MongoDreamStore(database=database)
        await store.add_dream({"n": 1})
        return len(await store.recent_dreams(5))

    assert asyncio.run(scenario()) == 1
