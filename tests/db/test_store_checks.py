"""Store setup modules against recording fakes (no network, no driver): apply
order, honest statuses, never a recreate or a destructive call, a missing
client that names its extra, quick probe timeouts and expectations that come
from the schema registry.

The fakes at the top are shared with the bootstrap and verifier tests. Each
records what it was asked to do and refuses every destructive call (a drop, a
delete, a recreate, a flush, CONFIG SET), so a test fails if any setup path
tries one.
"""

import asyncio
import re
import sys
import time
import types
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from mtc.core import schema_registry
from mtc.core.checks import CheckStatus
from mtc.core.config import Settings
from mtc.db import migrations, mongo_store, neo4j_store, postgres_store
from mtc.db import qdrant_store, redis_store
from mtc.db.migrations import expected_names, statements_in_order

_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)")
_INDEX = re.compile(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+)")


class DestructiveCall(AssertionError):
    """A fake was asked to drop, delete, recreate or flush something."""


def _refuse(name: str):
    def refused(*args, **kwargs):
        raise DestructiveCall(f"{name} was called")

    return refused


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


class FakePgConnection:
    def __init__(self, pool: "FakePool") -> None:
        self.pool = pool

    @asynccontextmanager
    async def transaction(self):
        self.pool.transactions += 1
        yield self

    async def execute(self, statement: str) -> str:
        if re.search(r"\b(DROP|TRUNCATE|DELETE\s+FROM)\b", statement, re.I):
            raise DestructiveCall(statement[:60])
        self.pool.executed.append(statement)
        table = _TABLE.search(statement)
        if table:
            self.pool.tables.add(table.group(1))
        index = _INDEX.search(statement)
        if index:
            self.pool.indexes.add(index.group(1))
        return "CREATE"


class FakePool:
    def __init__(self, tables=(), indexes=()) -> None:
        self.tables: Set[str] = set(tables)
        self.indexes: Set[str] = set(indexes)
        self.executed: List[str] = []
        self.transactions = 0
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        yield FakePgConnection(self)

    async def fetch(self, sql: str):
        if "pg_tables" in sql:
            return [(name,) for name in sorted(self.tables)]
        if "pg_indexes" in sql:
            return [(name,) for name in sorted(self.indexes)]
        raise AssertionError(f"unexpected query {sql}")

    async def fetchval(self, sql: str):
        assert sql == "SELECT 1"
        return 1

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.indexes: Dict[str, Dict[str, Any]] = {}
        self.created: List[Tuple[Any, bool]] = []

    drop = _refuse("drop")
    drop_index = _refuse("drop_index")
    drop_indexes = _refuse("drop_indexes")
    delete_many = _refuse("delete_many")
    delete_one = _refuse("delete_one")

    def create_index(self, keys, unique=False, name=None):
        keys = [tuple(item) for item in keys]
        self.created.append((tuple(keys), unique))
        label = name or "_".join(f"{field}_{value}" for field, value in keys)
        if label not in self.indexes:
            self.indexes[label] = {"key": keys, "v": 2}
            if unique:
                self.indexes[label]["unique"] = True
        return label

    def index_information(self):
        info = {"_id_": {"key": [("_id", 1)], "v": 2}}
        info.update(self.indexes)
        return info


class FakeMongoDatabase:
    def __init__(self) -> None:
        self.collections: Dict[str, FakeCollection] = {}

    drop_collection = _refuse("drop_collection")

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]


class FakeMongoClient:
    def __init__(self, database: Optional[FakeMongoDatabase] = None, **kwargs) -> None:
        self.kwargs = kwargs
        self.database = database or FakeMongoDatabase()
        self.closed = False
        self.admin = types.SimpleNamespace(command=self._command)

    drop_database = _refuse("drop_database")

    def _command(self, name: str):
        assert name == "ping"
        return {"ok": 1}

    def __getitem__(self, name: str) -> FakeMongoDatabase:
        return self.database

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


class _Enum(str):
    @property
    def value(self) -> str:
        return str(self)


def fake_qdrant_models() -> types.ModuleType:
    """A stand-in for ``qdrant_client.models`` with the names the setup uses."""
    module = types.ModuleType("qdrant_client.models")
    module.Distance = types.SimpleNamespace(
        COSINE=_Enum("Cosine"), DOT=_Enum("Dot"), EUCLID=_Enum("Euclid")
    )
    module.PayloadSchemaType = types.SimpleNamespace(
        INTEGER=_Enum("integer"), KEYWORD=_Enum("keyword"), FLOAT=_Enum("float")
    )
    module.VectorParams = lambda **kw: types.SimpleNamespace(**kw)
    module.HnswConfigDiff = lambda **kw: types.SimpleNamespace(**kw)
    return module


class FakeQdrantClient:
    def __init__(
        self,
        collections: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.collections: Dict[str, Dict[str, Any]] = collections or {}
        self.created: List[str] = []
        self.payload_created: List[Tuple[str, str]] = []
        self.closed = False

    delete_collection = _refuse("delete_collection")
    recreate_collection = _refuse("recreate_collection")
    delete = _refuse("delete")

    def get_collections(self):
        return types.SimpleNamespace(
            collections=[types.SimpleNamespace(name=n) for n in self.collections]
        )

    def create_collection(self, collection_name, vectors_config, hnsw_config):
        assert collection_name not in self.collections
        self.created.append(collection_name)
        self.collections[collection_name] = {
            "size": vectors_config.size,
            "distance": str(vectors_config.distance),
            "hnsw": dict(vars(hnsw_config)),
            "payload": {},
        }
        return True

    def get_collection(self, name):
        entry = self.collections[name]
        return collection_info(
            entry["size"], entry["distance"], entry["hnsw"], entry["payload"]
        )

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.payload_created.append((field_name, str(field_schema)))
        self.collections[collection_name]["payload"][field_name] = str(field_schema)

    def close(self) -> None:
        self.closed = True


def collection_info(size, distance, hnsw, payload):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            params=types.SimpleNamespace(
                vectors=types.SimpleNamespace(size=size, distance=_Enum(distance))
            ),
            hnsw_config=types.SimpleNamespace(**hnsw),
        ),
        payload_schema={
            field: types.SimpleNamespace(data_type=_Enum(kind))
            for field, kind in payload.items()
        },
    )


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

_N_CONSTRAINT = re.compile(
    r"CREATE CONSTRAINT (\w+) IF NOT EXISTS FOR \(\w+:(\w+)\) REQUIRE \w+\.(\w+) "
    r"IS UNIQUE"
)
_N_INDEX = re.compile(
    r"CREATE INDEX (\w+) IF NOT EXISTS FOR \(\w+:(\w+)\) ON \(\w+\.(\w+)\)"
)


class FakeNeo4jDriver:
    def __init__(self, constraints=(), indexes=()) -> None:
        self.constraints: List[Dict[str, Any]] = list(constraints)
        self.indexes: List[Dict[str, Any]] = list(indexes)
        self.queries: List[Tuple[str, Optional[str]]] = []
        self.closed = False
        self.verified = 0

    def verify_connectivity(self) -> None:
        self.verified += 1

    @contextmanager
    def session(self, database=None):
        yield _FakeNeo4jSession(self, database)

    def close(self) -> None:
        self.closed = True


class _FakeNeo4jSession:
    def __init__(self, driver: FakeNeo4jDriver, database: Optional[str]) -> None:
        self.driver = driver
        self.database = database

    def run(self, query: str, params=None):
        driver = self.driver
        driver.queries.append((query, self.database))
        if re.search(r"\b(DROP|DETACH\s+DELETE|DELETE)\b", query, re.I):
            raise DestructiveCall(query[:60])
        if query.startswith("SHOW CONSTRAINTS"):
            return [dict(row) for row in driver.constraints]
        if query.startswith("SHOW INDEXES"):
            return [dict(row) for row in driver.indexes]
        constraint = _N_CONSTRAINT.match(query)
        if constraint:
            name, label, prop = constraint.groups()
            if not any(row["name"] == name for row in driver.constraints):
                driver.constraints.append(
                    constraint_row(name, label, [prop], "UNIQUENESS")
                )
            return []
        index = _N_INDEX.match(query)
        if index:
            name, label, prop = index.groups()
            if not any(row["name"] == name for row in driver.indexes):
                driver.indexes.append(index_row(name, label, [prop], "RANGE"))
            return []
        raise AssertionError(f"unexpected query {query}")


def constraint_row(name, label, properties, kind="UNIQUENESS"):
    return {
        "name": name,
        "type": kind,
        "entityType": "NODE",
        "labelsOrTypes": [label],
        "properties": list(properties),
    }


def index_row(name, label, properties, kind="RANGE"):
    return {
        "name": name,
        "type": kind,
        "entityType": "NODE",
        "labelsOrTypes": [label],
        "properties": list(properties),
    }


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


class FakeRedis:
    def __init__(
        self,
        policy: Optional[str] = "volatile-lru",
        keys=(),
        config_error: Optional[BaseException] = None,
        ping_delay: float = 0.0,
        ping_error: Optional[BaseException] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.policy = policy
        self.keys = list(keys)
        self.config_error = config_error
        self.ping_delay = ping_delay
        self.ping_error = ping_error
        self.config_gets: List[str] = []
        self.closed = False

    config_set = _refuse("config_set")
    flushdb = _refuse("flushdb")
    flushall = _refuse("flushall")
    delete = _refuse("delete")
    unlink = _refuse("unlink")

    async def ping(self):
        if self.ping_delay:
            await asyncio.sleep(self.ping_delay)
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def config_get(self, name):
        self.config_gets.append(name)
        if self.config_error is not None:
            raise self.config_error
        return {name: self.policy} if self.policy is not None else {}

    async def scan_iter(self, match=None, count=None):
        prefix = (match or "*").rstrip("*")
        for key in self.keys:
            if key.startswith(prefix):
                yield key

    async def aclose(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def make_settings(tmp_path, **values) -> Settings:
    return Settings(_env_file=None, PROJECT_ROOT=str(tmp_path), **values)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def fake_models(monkeypatch):
    module = fake_qdrant_models()
    monkeypatch.setitem(sys.modules, "qdrant_client.models", module)
    return module


def _full_pool() -> FakePool:
    tables, indexes = expected_names()
    return FakePool(tables=tables, indexes=indexes)


class InvalidCatalogNameError(Exception):
    """Stands in for asyncpg's missing-database error (matched by name)."""


class InvalidPasswordError(Exception):
    """Stands in for asyncpg's refused-password error (matched by name)."""


class AuthenticationError(Exception):
    """Stands in for redis-py's refused-authentication error."""


class ResponseError(Exception):
    """Stands in for redis-py's error reply (for example NOPERM)."""


def test_postgres_apply_runs_statements_in_file_order(tmp_path):
    settings = make_settings(tmp_path)
    pool = FakePool()
    count = run(postgres_store.apply_migrations(pool))
    assert pool.executed == statements_in_order()
    assert count == len(statements_in_order())
    assert pool.transactions == 1
    fresh = FakePool()
    result = run(postgres_store.init(fresh, settings))
    tables, indexes = expected_names()
    assert result.status is CheckStatus.OK
    assert result.data["created_tables"] == sorted(tables)
    assert result.data["created_indexes"] == sorted(indexes)
    assert fresh.executed == statements_in_order()
    again = run(postgres_store.init(fresh, settings))
    assert again.data["created_tables"] == [] and again.data["created_indexes"] == []
    assert "nothing created" in again.detail


def test_postgres_check_reports_missing_tables_and_indexes(tmp_path):
    settings = make_settings(tmp_path)
    results = run(postgres_store.check(_full_pool(), settings))
    assert [(r.name, r.status) for r in results] == [
        ("postgres:tables", CheckStatus.OK),
        ("postgres:indexes", CheckStatus.OK),
    ]
    pool = _full_pool()
    pool.tables.discard("messages")
    pool.indexes.discard("messages_timestamp_idx")
    results = run(postgres_store.check(pool, settings))
    by_name = {r.name: r for r in results}
    assert by_name["postgres:tables"].status is CheckStatus.FAILED
    assert by_name["postgres:tables"].data["missing"] == ["messages"]
    assert by_name["postgres:indexes"].status is CheckStatus.FAILED
    assert by_name["postgres:indexes"].data["missing"] == ["messages_timestamp_idx"]
    assert all(r.required for r in results)


def test_postgres_missing_database_is_failed_with_the_role_script_hint(
    tmp_path, monkeypatch
):
    secret = "pw-" + "q" * 24
    settings = make_settings(tmp_path, POSTGRES_PASSWORD=secret)

    def raising(error):
        async def builder(*args, **kwargs):
            try:
                raise error
            except Exception as exc:
                from mtc.memory.protocols import StoreUnavailable

                raise StoreUnavailable(f"PostgreSQL did not accept: {exc}") from exc

        return builder

    missing = InvalidCatalogNameError(f'database "mtc_consciousness" missing {secret}')
    monkeypatch.setattr(postgres_store, "create_postgres_pool", raising(missing))
    for result in (
        run(postgres_store.probe(settings, timeout_s=0.5)),
        run(postgres_store.init(settings=settings)),
        run(postgres_store.check(settings=settings))[0],
    ):
        assert result.status is CheckStatus.FAILED
        assert "init_postgres.sh" in result.detail
        assert secret not in result.detail
    refused = InvalidPasswordError(f"password authentication failed {secret}")
    monkeypatch.setattr(postgres_store, "create_postgres_pool", raising(refused))
    result = run(postgres_store.probe(settings, timeout_s=0.5))
    assert result.status is CheckStatus.FAILED
    assert "MTC_POSTGRES_PASSWORD" in result.detail and secret not in result.detail
    monkeypatch.setattr(
        postgres_store, "create_postgres_pool", raising(ConnectionRefusedError())
    )
    result = run(postgres_store.probe(settings, timeout_s=0.5))
    assert result.status is CheckStatus.UNREACHABLE


def test_mongo_init_applies_every_registry_index(tmp_path):
    settings = make_settings(tmp_path)
    database = FakeMongoDatabase()
    result = run(mongo_store.init(database, settings))
    artifacts = mongo_store.mongo_artifacts()
    total = sum(len(specs) for specs in artifacts.values())
    assert result.status is CheckStatus.OK
    assert result.data["applied"] == total
    assert len(result.data["created_indexes"]) == total
    for collection, specs in artifacts.items():
        made = database.collections[collection].created
        assert [keys for keys, _ in made] == [tuple(spec.keys) for spec in specs]
        assert [unique for _, unique in made] == [spec.unique for spec in specs]
    again = run(mongo_store.init(database, settings))
    assert again.data["created_indexes"] == []
    assert "nothing created" in again.detail


def test_mongo_check_compares_keys_and_uniqueness(tmp_path):
    settings = make_settings(tmp_path)
    database = FakeMongoDatabase()
    run(mongo_store.init(database, settings))
    [result] = run(mongo_store.check(database, settings))
    assert result.status is CheckStatus.OK
    # A unique index that lost its uniqueness is reported, not accepted.
    beliefs = database.collections["learning_beliefs"]
    for entry in beliefs.indexes.values():
        entry.pop("unique", None)
    # A missing index is reported.
    turns = database.collections["neural_spikes"]
    turns.indexes.pop(next(iter(turns.indexes)))
    [result] = run(mongo_store.check(database, settings))
    assert result.status is CheckStatus.FAILED
    assert result.data["wrong_unique"] == ["learning_beliefs(belief_id 1)"]
    assert result.data["missing"] == ["neural_spikes(timestamp -1)"]


def test_mongo_artifacts_have_no_collection_collision(monkeypatch):
    artifacts = mongo_store.mongo_artifacts()
    registered = [
        schema_registry.load_artifact(path)
        for path in schema_registry.SCHEMA_ARTIFACTS["mongodb"]
    ]
    assert len(artifacts) == sum(len(table) for table in registered)
    real = mongo_store.load_artifact

    def colliding(dotted):
        table = dict(real(dotted))
        if dotted.endswith("ALERT_INDEXES"):
            table["dreams"] = []
        return table

    monkeypatch.setattr(mongo_store, "load_artifact", colliding)
    with pytest.raises(ValueError, match="dreams"):
        mongo_store.mongo_artifacts()


def test_qdrant_init_creates_with_registry_params(tmp_path, fake_models):
    settings = make_settings(tmp_path)
    client = FakeQdrantClient()
    result = run(qdrant_store.init(client, settings))
    params = schema_registry.load_artifact(
        "mtc.neural.vectors.coordinator.COLLECTION_PARAMS"
    )
    entry = client.collections[settings.QDRANT_COLLECTION]
    assert result.status is CheckStatus.OK and result.data["created"] is True
    assert entry["size"] == settings.VECTOR_DIMENSIONS == params["size"]
    assert entry["distance"] == "Cosine"
    assert entry["hnsw"] == params["hnsw_config"]
    assert [field for field, _ in client.payload_created] == [
        field for field, _ in qdrant_store.payload_indexes()
    ]
    again = run(qdrant_store.init(client, settings))
    assert again.data["created"] is False
    assert again.data["payload_indexes_created"] == []
    assert client.created == [settings.QDRANT_COLLECTION]


def test_qdrant_init_never_recreates_on_mismatch(tmp_path, fake_models):
    settings = make_settings(tmp_path)
    client = FakeQdrantClient(
        collections={
            settings.QDRANT_COLLECTION: {
                "size": 8192,
                "distance": "Cosine",
                "hnsw": {},
                "payload": {},
            }
        }
    )
    result = run(qdrant_store.init(client, settings))
    assert result.status is CheckStatus.FAILED
    assert "8192" in result.detail
    assert "MTC_QDRANT_COLLECTION" in result.detail
    assert "mtc.db.reembed" in result.detail
    assert client.created == [] and client.payload_created == []
    assert client.collections[settings.QDRANT_COLLECTION]["size"] == 8192


def test_qdrant_check_reports_hnsw_and_payload_mismatch(tmp_path, fake_models):
    from mtc.core.checks import exit_code_for

    settings = make_settings(tmp_path)
    params = schema_registry.load_artifact(
        "mtc.neural.vectors.coordinator.COLLECTION_PARAMS"
    )
    hnsw = dict(params["hnsw_config"], m=16)
    client = FakeQdrantClient(
        collections={
            settings.QDRANT_COLLECTION: {
                "size": 384,
                "distance": "Cosine",
                "hnsw": hnsw,
                "payload": {"postgres_id": "integer", "memory_type": "text"},
            }
        }
    )
    results = run(qdrant_store.check(client, settings))
    by_name = {r.name: r for r in results}
    assert by_name["qdrant:collection"].status is CheckStatus.OK
    assert by_name["qdrant:hnsw"].status is CheckStatus.WARNING
    assert "m 16 (expected 32)" in by_name["qdrant:hnsw"].detail
    payload = by_name["qdrant:payload_indexes"]
    assert payload.status is CheckStatus.WARNING
    assert "memory_type (text, expected keyword)" in payload.detail
    assert "importance (missing, expected float)" in payload.detail
    assert exit_code_for(results) == 0
    assert exit_code_for(results, strict=True) == 1
    client.collections[settings.QDRANT_COLLECTION]["distance"] = "Dot"
    by_name = {r.name: r for r in run(qdrant_store.check(client, settings))}
    assert by_name["qdrant:collection"].status is CheckStatus.FAILED
    [missing] = run(qdrant_store.check(FakeQdrantClient(), settings))
    assert missing.status is CheckStatus.FAILED and "does not exist" in missing.detail


def test_neo4j_init_applies_constraints_through_graph_manager(tmp_path, monkeypatch):
    from mtc.knowledge import graph_manager

    settings = make_settings(tmp_path, NEO4J_DATABASE="graphdb")
    calls = []
    original = graph_manager.KnowledgeGraphManager.bootstrap_constraints

    def recording(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(
        graph_manager.KnowledgeGraphManager, "bootstrap_constraints", recording
    )
    driver = FakeNeo4jDriver()
    result = run(neo4j_store.init(driver, settings))
    statements = schema_registry.load_artifact(
        "mtc.knowledge.graph_manager.CONSTRAINTS"
    )
    assert len(calls) == 1
    assert result.status is CheckStatus.OK
    assert result.data["applied"] == len(statements)
    assert len(result.data["created"]) == len(neo4j_store.registry_items())
    applied = [query for query, _ in driver.queries if query.startswith("CREATE")]
    assert applied == list(statements)
    assert {database for _, database in driver.queries} == {"graphdb"}
    again = run(neo4j_store.init(driver, settings))
    assert again.data["created"] == []
    assert driver.closed is False  # an injected driver stays open


def test_neo4j_check_matches_by_label_property_and_type(tmp_path):
    settings = make_settings(tmp_path)
    items = neo4j_store.registry_items()
    constraints = [
        constraint_row(f"renamed_{i}", item.label, item.properties)
        for i, item in enumerate(items)
        if item.kind == "unique"
    ]
    indexes = [
        index_row(f"other_name_{i}", item.label, item.properties)
        for i, item in enumerate(items)
        if item.kind == "index"
    ]
    [result] = run(neo4j_store.check(FakeNeo4jDriver(constraints, indexes), settings))
    assert result.status is CheckStatus.OK  # names are not compared
    wrong_type = [dict(row) for row in constraints]
    wrong_type[0]["type"] = "NODE_KEY"
    text_index = [dict(row, type="TEXT") for row in indexes]
    wrong_label = [dict(row) for row in constraints]
    wrong_label[1]["labelsOrTypes"] = ["Elsewhere"]
    for driver, expected in (
        (FakeNeo4jDriver(wrong_type, indexes), items[0].describe()),
        (FakeNeo4jDriver(constraints, text_index), "index on :Person(name)"),
        (FakeNeo4jDriver(wrong_label, indexes), items[1].describe()),
    ):
        [result] = run(neo4j_store.check(driver, settings))
        assert result.status is CheckStatus.FAILED
        assert result.data["missing"] == [expected]


def test_redis_ping_required(tmp_path):
    from mtc.core.checks import exit_code_for

    settings = make_settings(tmp_path)
    results = run(redis_store.check(FakeRedis(), settings))
    assert results[0].name == "redis:ping" and results[0].required
    assert results[0].status is CheckStatus.OK
    refused = run(
        redis_store.check(FakeRedis(ping_error=ConnectionRefusedError()), settings)
    )
    assert [(r.name, r.status) for r in refused] == [
        ("redis:ping", CheckStatus.UNREACHABLE)
    ]
    assert exit_code_for(refused) == 3
    denied = run(
        redis_store.check(
            FakeRedis(ping_error=AuthenticationError("invalid password")), settings
        )
    )
    assert denied[0].status is CheckStatus.FAILED
    assert exit_code_for(denied) == 1


@pytest.mark.parametrize(
    "policy, status",
    [
        ("volatile-lru", CheckStatus.OK),
        ("allkeys-lru", CheckStatus.OK),
        ("noeviction", CheckStatus.WARNING),
        ("allkeys-random", CheckStatus.WARNING),
    ],
)
def test_maxmemory_policy_advisory(tmp_path, policy, status):
    settings = make_settings(tmp_path)
    client = FakeRedis(policy=policy)
    results = {r.name: r for r in run(redis_store.check(client, settings))}
    result = results["redis:maxmemory_policy"]
    assert result.status is status
    assert result.required is False
    assert client.config_gets == ["maxmemory-policy"]
    if status is CheckStatus.WARNING:
        assert "volatile-lru" in result.detail and "TTL" in result.detail


def test_config_refused_is_insufficient_data_not_ok(tmp_path):
    from mtc.core.checks import exit_code_for

    settings = make_settings(tmp_path, DB_PREFIX="lab")
    client = FakeRedis(
        config_error=ResponseError("NOPERM this user has no permissions"),
        keys=["lab:wm:conversation:s1", "lab:cache:x", "other:cache:y"],
    )
    results = run(redis_store.check(client, settings))
    by_name = {r.name: r for r in results}
    policy = by_name["redis:maxmemory_policy"]
    assert policy.status is CheckStatus.INSUFFICIENT_DATA
    assert policy.status is not CheckStatus.OK
    assert "CONFIG GET" in policy.detail
    families = by_name["redis:key_families"]
    assert families.status is CheckStatus.OK
    assert families.data["counts"]["wm:conversation:{session}"] == 1
    assert families.data["counts"]["cache:{key}"] == 1
    assert exit_code_for(results) == 0
    assert exit_code_for(results, strict=True) == 1


@pytest.mark.parametrize(
    "module, hidden, extra",
    [
        (postgres_store, ("asyncpg",), "postgres"),
        (mongo_store, ("pymongo",), "mongo"),
        (redis_store, ("redis", "redis.asyncio"), "redis"),
        (qdrant_store, ("qdrant_client",), "qdrant"),
        (neo4j_store, ("neo4j",), "neo4j"),
    ],
)
def test_missing_client_names_the_extra(tmp_path, monkeypatch, module, hidden, extra):
    for name in hidden:
        monkeypatch.setitem(sys.modules, name, None)
    settings = make_settings(tmp_path)
    result = run(module.probe(settings, timeout_s=0.5))
    assert result.status is CheckStatus.FAILED
    assert f"'{extra}'" in result.detail
    assert f"multi-theory-consciousness[{extra}]" in result.detail
    assert result.data["missing_extra"] == extra


def test_probe_times_out_quickly(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        redis_store, "create_redis_client", lambda *a, **k: FakeRedis(ping_delay=30)
    )

    async def slow_pool(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(postgres_store, "create_postgres_pool", slow_pool)
    for module in (redis_store, postgres_store):
        started = time.monotonic()
        result = run(module.probe(settings, timeout_s=0.2))
        elapsed = time.monotonic() - started
        assert result.status is CheckStatus.UNREACHABLE, module.__name__
        assert elapsed < 3.0, (module.__name__, elapsed)


def test_verify_expectations_derive_from_registry(tmp_path, monkeypatch, fake_models):
    settings = make_settings(tmp_path)
    extra_table = "CREATE TABLE IF NOT EXISTS registry_probe (id BIGSERIAL PRIMARY KEY)"
    real = schema_registry.load_artifact

    def with_extras(dotted):
        value = real(dotted)
        if dotted.endswith("EPISODIC_SCHEMA_SQL"):
            return tuple(value) + (extra_table,)
        if dotted.endswith("ALERT_INDEXES"):
            table = dict(value)
            table["registry_probe"] = [
                schema_registry_index_spec((("probe_field", 1),), True)
            ]
            return table
        if dotted.endswith("PAYLOAD_INDEXES"):
            return tuple(value) + (("probe_field", "keyword"),)
        if dotted.endswith("CONSTRAINTS"):
            return tuple(value) + (
                "CREATE CONSTRAINT probe_unique IF NOT EXISTS FOR (n:Probe) "
                "REQUIRE n.id IS UNIQUE",
            )
        if dotted.endswith("KEY_FAMILIES"):
            return (types.SimpleNamespace(pattern="probe:{id}"),)
        return value

    # Each check reads the registry: a new artifact entry becomes an expectation.
    pool = _full_pool()
    monkeypatch.setattr(migrations, "load_artifact", with_extras)
    tables = {r.name: r for r in run(postgres_store.check(pool, settings))}
    assert tables["postgres:tables"].data["missing"] == ["registry_probe"]
    assert "registry_probe" in expected_names()[0]

    database = FakeMongoDatabase()
    run(mongo_store.init(database, settings))
    monkeypatch.setattr(mongo_store, "load_artifact", with_extras)
    [mongo] = run(mongo_store.check(database, settings))
    assert mongo.data["missing"] == ["registry_probe(probe_field 1)"]

    client = FakeQdrantClient()
    run(qdrant_store.init(client, settings))
    monkeypatch.setattr(qdrant_store, "load_artifact", with_extras)
    payload = {r.name: r for r in run(qdrant_store.check(client, settings))}[
        "qdrant:payload_indexes"
    ]
    assert "probe_field (missing, expected keyword)" in payload.detail

    driver = FakeNeo4jDriver()
    run(neo4j_store.init(driver, settings))
    monkeypatch.setattr(neo4j_store, "load_artifact", with_extras)
    [graph] = run(neo4j_store.check(driver, settings))
    assert graph.data["missing"] == ["unique on :Probe(id)"]

    monkeypatch.setattr(redis_store, "load_artifact", with_extras)
    redis_results = run(
        redis_store.check(FakeRedis(keys=["mtc:probe:1", "mtc:probe:2"]), settings)
    )
    families = {r.name: r for r in redis_results}["redis:key_families"]
    assert families.data["counts"] == {"probe:{id}": 2}


def schema_registry_index_spec(keys, unique):
    from mtc.memory.clients import IndexSpec

    return IndexSpec(keys=keys, unique=unique)
