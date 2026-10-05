"""Importing mtc loads no store client; a missing client names its extra."""

import ast
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.memory import clients

ROOT = Path(__file__).resolve().parents[2]
CLIENT_PACKAGES = ("asyncpg", "motor", "pymongo", "redis", "neo4j", "qdrant_client")

# Modules that still import a client package at module level. The walk below
# does not import them; the vector work empties this tuple.
PENDING_LAZY_IMPORTS = (
    "mtc.neural.vectors.coordinator",
    "mtc.neural.gpu_integration",
)

_WALK = r"""
import importlib, json, pkgutil, sys
import mtc

pending = set(json.loads(sys.argv[1]))
clients = json.loads(sys.argv[2])
skipped, failed, imported = [], [], 0
for info in pkgutil.walk_packages(mtc.__path__, "mtc."):
    name = info.name
    if name in pending:
        continue
    try:
        importlib.import_module(name)
        imported += 1
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.split(".")[0] == "mtc":
            failed.append([name, repr(exc)])
        else:
            skipped.append([name, missing])
    except Exception as exc:
        failed.append([name, repr(exc)])
loaded = [c for c in clients if c in sys.modules]
print(json.dumps(
    {"loaded": loaded, "skipped": skipped, "failed": failed, "imported": imported}
))
"""


def test_importing_mtc_imports_no_store_client():
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WALK,
            json.dumps(list(PENDING_LAZY_IMPORTS)),
            json.dumps(list(CLIENT_PACKAGES)),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["imported"] > 20, report
    assert report["failed"] == [], report["failed"]
    assert report["loaded"] == [], (
        f"importing mtc loaded store clients {report['loaded']}; "
        f"modules skipped for a missing optional dependency: {report['skipped']}"
    )


def _top_level_client_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        found += [n for n in names if n.split(".")[0] in CLIENT_PACKAGES]
    return found


@pytest.mark.parametrize("module", PENDING_LAZY_IMPORTS)
def test_pending_entries_still_import_a_client(module):
    path = ROOT / (module.replace(".", "/") + ".py")
    if module == "mtc.neural.gpu_integration":
        # imports the coordinator at module level, which loads the client
        text = path.read_text(encoding="utf-8")
        assert "mtc.neural.vectors.coordinator" in text, (
            f"{module} no longer imports the coordinator; remove it from "
            "PENDING_LAZY_IMPORTS"
        )
        return
    assert _top_level_client_imports(path), (
        f"{module} no longer imports a client at module level; remove it from "
        "PENDING_LAZY_IMPORTS"
    )


def test_no_other_module_imports_a_client_at_module_level():
    offenders = {}
    for path in (ROOT / "mtc").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        if module in PENDING_LAZY_IMPORTS:
            continue
        found = _top_level_client_imports(path)
        if found:
            offenders[module] = found
    assert offenders == {}


def test_require_client_returns_the_module_or_names_the_extra(monkeypatch):
    assert clients.require_client("json", "none") is json
    with pytest.raises(ImportError) as info:
        clients.require_client("mtc_surely_absent_client_package", "postgres")
    message = str(info.value)
    assert "'postgres'" in message and "multi-theory-consciousness[postgres]" in message


@pytest.mark.parametrize(
    "extra,blocked,call",
    [
        (
            "postgres",
            ("asyncpg",),
            lambda: asyncio.run(clients.create_postgres_pool()),
        ),
        ("mongo", ("pymongo",), lambda: clients.create_mongo_client()),
        ("redis", ("redis", "redis.asyncio"), lambda: clients.create_redis_client()),
    ],
)
def test_missing_client_names_the_extra(monkeypatch, extra, blocked, call):
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError) as info:
        call()
    assert f"[{extra}]" in str(info.value) and f"'{extra}'" in str(info.value)


# ---------------------------------------------------------------------------
# The factories build their clients from Settings (fake client modules only)
# ---------------------------------------------------------------------------

import types  # noqa: E402

from mtc.core.config import Settings  # noqa: E402
from mtc.memory.protocols import StoreUnavailable  # noqa: E402


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return Settings()


def test_postgres_pool_arguments_come_from_settings(monkeypatch, settings):
    captured = {}

    async def create_pool(**kwargs):
        captured.update(kwargs)
        return "pool"

    monkeypatch.setitem(
        sys.modules, "asyncpg", types.SimpleNamespace(create_pool=create_pool)
    )
    monkeypatch.setenv("MTC_POSTGRES_HOST", "db.example")
    monkeypatch.setenv("MTC_POSTGRES_PORT", "6543")
    monkeypatch.setenv("MTC_DB_POOL_MAX_SIZE", "4")
    monkeypatch.setenv("MTC_DB_POOL_IDLE_TIMEOUT", "77")
    active = Settings()
    assert asyncio.run(clients.create_postgres_pool(active, min_size=1)) == "pool"
    assert captured["host"] == "db.example" and captured["port"] == 6543
    assert captured["database"] == active.POSTGRES_DB
    assert captured["user"] == active.POSTGRES_USER
    assert captured["password"] is None  # an empty password is not sent
    assert captured["min_size"] == 1  # the override wins
    assert captured["max_size"] == 4
    assert captured["timeout"] == active.DB_POOL_TIMEOUT
    assert captured["command_timeout"] == active.DB_POOL_COMMAND_TIMEOUT
    assert captured["max_inactive_connection_lifetime"] == 77


def test_postgres_pool_installs_json_codecs(monkeypatch, settings):
    captured = {}

    async def create_pool(**kwargs):
        captured.update(kwargs)
        return "pool"

    monkeypatch.setitem(
        sys.modules, "asyncpg", types.SimpleNamespace(create_pool=create_pool)
    )
    asyncio.run(clients.create_postgres_pool(settings))

    class Connection:
        def __init__(self):
            self.codecs = {}

        async def set_type_codec(self, name, encoder, decoder, schema):
            self.codecs[name] = (encoder, decoder, schema)

    connection = Connection()
    asyncio.run(captured["init"](connection))
    assert set(connection.codecs) == {"json", "jsonb"}
    encoder, decoder, schema = connection.codecs["jsonb"]
    assert schema == "pg_catalog"
    assert decoder(encoder({"a": [1, 2]})) == {"a": [1, 2]}


def test_postgres_connection_failure_is_store_unavailable(monkeypatch, settings):
    async def create_pool(**kwargs):
        raise OSError("refused")

    monkeypatch.setitem(
        sys.modules, "asyncpg", types.SimpleNamespace(create_pool=create_pool)
    )
    with pytest.raises(StoreUnavailable) as info:
        asyncio.run(clients.create_postgres_pool(settings))
    assert isinstance(info.value.__cause__, OSError)


def test_mongo_client_arguments_come_from_settings(monkeypatch, settings):
    captured = {}

    class MongoClient(dict):
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            super().__init__()

        def __getitem__(self, name):
            return f"database:{name}"

    monkeypatch.setitem(
        sys.modules, "pymongo", types.SimpleNamespace(MongoClient=MongoClient)
    )
    monkeypatch.setenv("MTC_MONGODB_SERVER_SELECTION_TIMEOUT_MS", "1234")
    active = Settings()
    client = clients.create_mongo_client(active)
    assert captured["tz_aware"] is True
    assert captured["serverSelectionTimeoutMS"] == 1234
    assert captured["host"] == active.MONGODB_HOST
    assert captured["port"] == active.MONGODB_PORT
    assert "username" not in captured and "password" not in captured
    assert clients.mongo_database(client, settings=active) == (
        f"database:{active.MONGODB_DB}"
    )
    assert clients.mongo_database(client, settings=active, name="x") == "database:x"
    monkeypatch.setenv("MTC_MONGODB_USER", "operator")
    monkeypatch.setenv("MTC_MONGODB_PASSWORD", "placeholder")
    clients.create_mongo_client(Settings(), serverSelectionTimeoutMS=50)
    assert captured["username"] == "operator"
    assert captured["authSource"] == Settings().MONGODB_DB
    assert captured["serverSelectionTimeoutMS"] == 50


def test_redis_client_arguments_come_from_settings(monkeypatch, settings):
    captured = {}

    class Redis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    package = types.ModuleType("redis")
    asyncio_module = types.ModuleType("redis.asyncio")
    asyncio_module.Redis = Redis
    package.asyncio = asyncio_module
    monkeypatch.setitem(sys.modules, "redis", package)
    monkeypatch.setitem(sys.modules, "redis.asyncio", asyncio_module)
    monkeypatch.setenv("MTC_REDIS_DB", "3")
    monkeypatch.setenv("MTC_REDIS_MAX_CONNECTIONS", "9")
    active = Settings()
    assert isinstance(clients.create_redis_client(active), Redis)
    assert captured["db"] == 3 and captured["max_connections"] == 9
    assert captured["decode_responses"] is True
    assert captured["password"] is None
    assert (
        captured["host"] == active.REDIS_HOST and captured["port"] == active.REDIS_PORT
    )
    assert captured["socket_timeout"] == active.REDIS_SOCKET_TIMEOUT


def test_apply_indexes_creates_each_spec():
    created = []

    class Collection:
        def __init__(self, name):
            self.name = name

        def create_index(self, keys, **options):
            created.append((self.name, keys, options))

    class Database(dict):
        def __getitem__(self, name):
            return Collection(name)

    spec = clients.IndexSpec(keys=(("belief_id", 1),), unique=True)
    named = clients.IndexSpec(keys=(("timestamp", -1),), name="by_time")
    applied = clients.apply_indexes(
        Database(), {"beliefs": [spec], "events": [named, spec]}
    )
    assert applied == 3
    assert created[0] == ("beliefs", [("belief_id", 1)], {"unique": True})
    assert created[1] == (
        "events",
        [("timestamp", -1)],
        {"unique": False, "name": "by_time"},
    )


def test_apply_schema_runs_statements_in_one_transaction():
    log = []

    class Transaction:
        async def __aenter__(self):
            log.append("begin")

        async def __aexit__(self, *exc):
            log.append("commit")

    class Connection:
        def transaction(self):
            return Transaction()

        async def execute(self, statement):
            log.append(statement)

    class Acquired:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, *exc):
            return False

    class Pool:
        def acquire(self):
            return Acquired()

    count = asyncio.run(clients.apply_schema(Pool(), ["CREATE A", "CREATE B"]))
    assert count == 2
    assert log == ["begin", "CREATE A", "CREATE B", "commit"]
