"""The bootstrap: store order with isolation, idempotence, honest skips, the
install date before seeding, the given Settings, shared exit codes and a
source scan that finds no destructive operation in any init path."""

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mtc.core.checks import CheckResult, CheckStatus
from mtc.core.config import Settings
from mtc.db import bootstrap as bootstrap_module
from mtc.db import mongo_store, neo4j_store, postgres_store, qdrant_store
from mtc.db import redis_store
from mtc.db.bootstrap import STORE_ORDER, BootstrapResult, bootstrap, enabled_stores
from mtc.onboarding import seed_db
from tests.db.test_store_checks import (
    FakeMongoClient,
    FakeMongoDatabase,
    FakeNeo4jDriver,
    FakePool,
    FakeQdrantClient,
    FakeRedis,
    fake_qdrant_models,
)

ROOT = Path(__file__).resolve().parents[2]
INSTANT = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
ALL_ON = dict(
    POSTGRES_ENABLED=True,
    MONGODB_ENABLED=True,
    QDRANT_ENABLED=True,
    NEO4J_ENABLED=True,
    REDIS_ENABLED=True,
)


def make_settings(tmp_path, **values) -> Settings:
    base = dict(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Tessera",
        CREATOR_NAME="Alice",
        PERSONA_PATH=str(tmp_path / "absent-persona.yaml"),
    )
    base.update(values)
    return Settings(_env_file=None, **base)


class _Manager:
    def __init__(self, blocks):
        self.blocks = blocks

    async def seed_core_memory(self, blocks=None, overwrite=False):
        written = {}
        for name, text in blocks.items():
            written[name] = overwrite or name not in self.blocks
            if written[name]:
                self.blocks[name] = text
        return written


class _Stack:
    def __init__(self, blocks):
        self.manager = _Manager(blocks)

    async def close(self):
        return None


class _Graph:
    nodes = set()

    def __init__(self, active):
        self.active = active

    def connect(self):
        return None

    def seed_core_entities(self, include_creator=True):
        names = ["system", "creator"] if include_creator else ["system"]
        new_persons = [n for n in names if ("person", n) not in self.nodes]
        concepts = ["consciousness", "emotions", "neural networks"]
        new_concepts = [c for c in concepts if ("concept", c) not in self.nodes]
        self.nodes.update(("person", n) for n in names)
        self.nodes.update(("concept", c) for c in concepts)
        return {
            "persons_created": len(new_persons),
            "concepts_created": len(new_concepts),
        }

    def close(self):
        return None


@pytest.fixture
def fake_stores(monkeypatch):
    """Every store backed by a shared recording fake; nothing touches a network."""
    monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_qdrant_models())
    pool = FakePool()
    database = FakeMongoDatabase()
    qdrant = FakeQdrantClient()
    driver = FakeNeo4jDriver()
    redis = FakeRedis()
    seen = {"postgres": [], "mongodb": []}

    async def pool_factory(active, **kwargs):
        seen["postgres"].append(active.POSTGRES_DB)
        return pool

    def mongo_factory(active, **kwargs):
        seen["mongodb"].append(active.MONGODB_DB)
        return FakeMongoClient(database)

    monkeypatch.setattr(postgres_store, "create_postgres_pool", pool_factory)
    monkeypatch.setattr(mongo_store, "create_mongo_client", mongo_factory)
    monkeypatch.setattr(qdrant_store, "create_client", lambda *a, **k: qdrant)
    monkeypatch.setattr(neo4j_store, "create_driver", lambda *a, **k: driver)
    monkeypatch.setattr(redis_store, "create_redis_client", lambda *a, **k: redis)
    blocks = {}

    async def build_stack(active):
        return _Stack(blocks)

    _Graph.nodes = set()
    monkeypatch.setattr(seed_db, "_build_stack", build_stack)
    monkeypatch.setattr(seed_db, "_build_graph", _Graph)
    return {
        "pool": pool,
        "database": database,
        "qdrant": qdrant,
        "driver": driver,
        "redis": redis,
        "blocks": blocks,
        "seen": seen,
    }


def run(coro):
    return asyncio.run(coro)


def test_bootstrap_order_and_isolation(tmp_path, monkeypatch):
    order = []

    def recorder(store, fail=False):
        async def init(settings=None, **kwargs):
            order.append(store)
            if fail:
                raise RuntimeError("simulated init failure")
            return CheckResult(f"store:{store}", CheckStatus.OK, "done")

        return init

    async def redis_check(settings=None, **kwargs):
        order.append("redis")
        return [CheckResult("redis:ping", CheckStatus.OK, "pong")]

    monkeypatch.setattr(postgres_store, "init", recorder("postgres"))
    monkeypatch.setattr(mongo_store, "init", recorder("mongodb", fail=True))
    monkeypatch.setattr(qdrant_store, "init", recorder("qdrant"))
    monkeypatch.setattr(neo4j_store, "init", recorder("neo4j"))
    monkeypatch.setattr(redis_store, "check", redis_check)
    settings = make_settings(tmp_path, **ALL_ON)
    outcome = run(bootstrap(settings, seed=False))
    assert order == ["postgres", "mongodb", "qdrant", "neo4j", "redis"]
    by_name = {r.name: r for r in outcome.results}
    assert by_name["store:mongodb"].status is CheckStatus.FAILED
    assert "simulated init failure" in by_name["store:mongodb"].detail
    for name in ("store:postgres", "store:qdrant", "store:neo4j", "redis:ping"):
        assert by_name[name].status is CheckStatus.OK, name
    assert bootstrap_module.command_exit_code(outcome.results) == 1


def test_bootstrap_twice_is_idempotent(tmp_path, fake_stores):
    settings = make_settings(tmp_path, **ALL_ON)
    env = tmp_path / "project.env"
    first = run(bootstrap(settings, env_path=env, clock=lambda: INSTANT))
    assert bootstrap_module.command_exit_code(first.results) == 0
    created = {r.name: r.data for r in first.results}
    assert created["store:postgres"]["created_tables"]
    assert created["store:mongodb"]["created_indexes"]
    assert created["store:qdrant"]["created"] is True
    assert created["store:neo4j"]["created"]
    assert first.birth_date == "2000-01-01" and first.birth_date_written
    assert all(first.seed.core_blocks.values())
    assert first.seed.graph == {"persons_created": 2, "concepts_created": 3}
    executed = len(fake_stores["pool"].executed)
    env_before = env.read_bytes()
    blocks_before = dict(fake_stores["blocks"])

    later = datetime(2000, 3, 1, 12, 0, tzinfo=timezone.utc)
    second = run(bootstrap(settings, env_path=env, clock=lambda: later))
    again = {r.name: r for r in second.results}
    assert bootstrap_module.command_exit_code(second.results) == 0
    assert again["store:postgres"].data["created_tables"] == []
    assert again["store:postgres"].data["created_indexes"] == []
    assert again["store:mongodb"].data["created_indexes"] == []
    assert again["store:qdrant"].data["created"] is False
    assert again["store:qdrant"].data["payload_indexes_created"] == []
    assert again["store:neo4j"].data["created"] == []
    assert second.birth_date == "2000-01-01" and not second.birth_date_written
    assert not any(second.seed.core_blocks.values())
    assert second.seed.graph == {"persons_created": 0, "concepts_created": 0}
    assert env.read_bytes() == env_before
    assert fake_stores["blocks"] == blocks_before
    # The statements ran again (idempotent), and only creates ever ran.
    assert len(fake_stores["pool"].executed) == 2 * executed
    assert fake_stores["qdrant"].created == [settings.QDRANT_COLLECTION]


def test_disabled_stores_skipped_not_ok(tmp_path, monkeypatch):
    for module in (postgres_store, mongo_store, qdrant_store, neo4j_store):
        monkeypatch.setattr(module, "init", _never)
    monkeypatch.setattr(redis_store, "check", _never)
    settings = make_settings(tmp_path)
    assert enabled_stores(settings) == []
    outcome = run(
        bootstrap(settings, env_path=tmp_path / "x.env", clock=lambda: INSTANT)
    )
    stores = [r for r in outcome.results if r.name.startswith("store:")]
    assert [r.name for r in stores] == [f"store:{s}" for s in STORE_ORDER]
    for result in stores:
        assert result.status is CheckStatus.SKIPPED
        assert result.status is not CheckStatus.OK
        assert "_ENABLED is false" in result.detail
    seed = {r.name: r for r in outcome.results}["seed"]
    assert seed.status is CheckStatus.SKIPPED
    assert "seeded at run time from the persona" in seed.detail
    # The install date is still recorded with no store.
    assert outcome.birth_date == "2000-01-01" and outcome.birth_date_written
    assert bootstrap_module.command_exit_code(outcome.results) == 0


async def _never(*args, **kwargs):
    raise AssertionError("a disabled store must not be touched")


def test_only_flag_limits_stores(tmp_path, fake_stores, monkeypatch, capsys):
    settings = make_settings(tmp_path, **ALL_ON)
    outcome = run(bootstrap(settings, only=["qdrant"], seed=True))
    names = [r.name for r in outcome.results]
    assert names == ["store:qdrant", "birth_date", "seed"]
    assert fake_stores["pool"].executed == []
    assert fake_stores["database"].collections == {}
    seed = {r.name: r for r in outcome.results}["seed"]
    assert seed.status is CheckStatus.SKIPPED
    assert "no seeded store was selected" in seed.detail
    with pytest.raises(ValueError):
        run(bootstrap(settings, only=["bogus"]))
    with pytest.raises(SystemExit) as info:
        bootstrap_module.main(["--only", "bogus"])
    assert info.value.code == 2
    capsys.readouterr()


def test_bootstrap_records_birth_date_before_seeding(tmp_path, monkeypatch):
    calls = []
    real = seed_db.record_birth_date

    def record(env_path, *, settings=None, clock=None):
        result = real(env_path, settings=settings, clock=clock)
        calls.append(("birth", result))
        return result

    async def seed(settings=None, *, persona=None, birth_date=None, graph=True):
        calls.append(("seed", birth_date, settings.POSTGRES_ENABLED, graph))
        return seed_db.SeedReport(core_blocks={"identity": True})

    async def init(settings=None, **kwargs):
        calls.append(("init", "postgres"))
        return CheckResult("store:postgres", CheckStatus.OK, "done")

    monkeypatch.setattr(seed_db, "record_birth_date", record)
    monkeypatch.setattr(seed_db, "seed_stores", seed)
    monkeypatch.setattr(postgres_store, "init", init)
    settings = make_settings(tmp_path, POSTGRES_ENABLED=True, TIMEZONE="Asia/Tokyo")
    late = datetime(2000, 1, 1, 23, 30, tzinfo=timezone.utc)
    outcome = run(bootstrap(settings, env_path=tmp_path / "a.env", clock=lambda: late))
    assert calls == [
        ("init", "postgres"),
        ("birth", ("2000-01-02", True)),
        ("seed", "2000-01-02", True, False),
    ]
    assert outcome.birth_date == "2000-01-02"


def test_bootstrap_uses_the_given_settings_not_the_singleton(
    tmp_path, fake_stores, monkeypatch
):
    from mtc.core import config as config_module

    singleton = make_settings(tmp_path, DB_PREFIX="singleton", **ALL_ON)
    monkeypatch.setattr(config_module, "_settings", singleton)
    given = make_settings(tmp_path, DB_PREFIX="given", TIMEZONE="Asia/Tokyo", **ALL_ON)
    late = datetime(2000, 1, 1, 23, 30, tzinfo=timezone.utc)
    outcome = run(bootstrap(given, env_path=tmp_path / "given.env", clock=lambda: late))
    assert fake_stores["seen"]["postgres"] == ["given_consciousness"]
    assert fake_stores["seen"]["mongodb"] == ["given_neural"]
    assert "given_memories" in fake_stores["qdrant"].collections
    assert "singleton_memories" not in fake_stores["qdrant"].collections
    assert outcome.birth_date == "2000-01-02"  # the given TIMEZONE's day


@pytest.mark.parametrize(
    "results, code",
    [
        ([CheckResult("store:postgres", CheckStatus.OK)], 0),
        ([CheckResult("store:postgres", CheckStatus.FAILED)], 1),
        ([CheckResult("store:mongodb", CheckStatus.UNREACHABLE)], 3),
        (
            [
                CheckResult("store:mongodb", CheckStatus.UNREACHABLE),
                CheckResult("store:postgres", CheckStatus.FAILED),
            ],
            1,
        ),
        (
            [
                CheckResult(
                    "store:qdrant",
                    CheckStatus.FAILED,
                    "qdrant_client is not installed",
                    data={"missing_extra": "qdrant"},
                )
            ],
            2,
        ),
        ([CheckResult("seed", CheckStatus.SKIPPED, required=False)], 0),
    ],
)
def test_exit_codes_from_results(tmp_path, monkeypatch, capsys, results, code):
    async def fake_bootstrap(settings=None, **kwargs):
        return BootstrapResult(results=list(results))

    monkeypatch.setattr(bootstrap_module, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        bootstrap_module, "load_settings", lambda env_file: make_settings(tmp_path)
    )
    assert bootstrap_module.main([]) == code
    text = capsys.readouterr().out
    assert text.startswith("mtc db init")
    assert bootstrap_module.main(["--json"]) == code
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == code
    assert len(payload["checks"]) == len(results)


def test_interrupt_exits_130(tmp_path, monkeypatch, capsys):
    async def interrupted(settings=None, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(bootstrap_module, "bootstrap", interrupted)
    monkeypatch.setattr(
        bootstrap_module, "load_settings", lambda env_file: make_settings(tmp_path)
    )
    assert bootstrap_module.main([]) == 130
    capsys.readouterr()


DESTRUCTIVE = [
    re.compile(r"\bDROP\b"),
    re.compile(r"\bTRUNCATE\b"),
    re.compile(r"\bDELETE\s+FROM\b"),
    re.compile(r"\bDETACH\s+DELETE\b"),
    re.compile(r"\bFLUSH(DB|ALL)\b", re.IGNORECASE),
    re.compile(r"\bCONFIG\s+SET\b", re.IGNORECASE),
    re.compile(r"config_set\("),
    re.compile(r"\.drop\("),
    re.compile(r"drop_(database|collection|index|indexes)\b"),
    re.compile(r"(delete|recreate)_collection\b"),
    re.compile(r"delete_(many|one)\b"),
    re.compile(r"\.delete\("),
    re.compile(r"\bdropdb\b"),
    re.compile(r"\brm\s+-[a-zA-Z]*r"),
    re.compile(r"\bALTER\s+ROLE\b", re.IGNORECASE),
]


def _scan(text):
    return [pattern.pattern for pattern in DESTRUCTIVE if pattern.search(text)]


def test_no_drop_in_any_init_path():
    paths = (
        sorted((ROOT / "mtc" / "db").glob("*.py"))
        + sorted((ROOT / "scripts" / "db").iterdir())
        + sorted((ROOT / "migrations").rglob("*.sql"))
        + [ROOT / "mtc" / "onboarding" / "seed_db.py"]
    )
    paths = [p for p in paths if p.is_file()]
    assert len(paths) >= 20
    hits = {}
    for path in paths:
        found = _scan(path.read_text(encoding="utf-8"))
        if found:
            hits[path.relative_to(ROOT).as_posix()] = found
    assert hits == {}
    # The scanner finds what it claims to find.
    assert _scan("DROP TABLE x;")
    assert _scan("client.delete_collection(name)")
    assert _scan("await client.flushdb()")
    assert _scan("db.drop_collection('x')")
    assert not _scan("REFERENCES conversations(id) ON DELETE CASCADE")
    assert not _scan("Nothing is dropped, deleted or recreated.")


SECRET_VALUES = dict(
    POSTGRES_PASSWORD="pgsecretvalue1",
    MONGODB_USER="operator",
    MONGODB_PASSWORD="mongosecretvalue2",
    REDIS_PASSWORD="redissecretvalue3",
    NEO4J_PASSWORD="neosecretvalue4",
)


def _echoing_error(store):
    # A driver message that quotes every configured password, as a connection
    # string in an error text would.
    joined = " ".join(SECRET_VALUES[key] for key in sorted(SECRET_VALUES))
    return RuntimeError(f"{store} refused the connection string with {joined}")


def _assert_no_secret(results):
    text = json.dumps([r.detail for r in results])
    for key, value in SECRET_VALUES.items():
        if key.endswith("PASSWORD"):
            assert value not in text, key


def test_bootstrap_error_details_mask_every_store_password(tmp_path, monkeypatch):
    async def failing_init(settings=None, **kwargs):
        raise _echoing_error("init")

    async def failing_check(settings=None, **kwargs):
        raise _echoing_error("check")

    for module in (postgres_store, mongo_store, qdrant_store, neo4j_store):
        monkeypatch.setattr(module, "init", failing_init)
    monkeypatch.setattr(redis_store, "check", failing_check)

    def failing_record(env_path, *, settings=None, clock=None):
        raise _echoing_error("birth date")

    monkeypatch.setattr(seed_db, "record_birth_date", failing_record)
    settings = make_settings(tmp_path, **ALL_ON, **SECRET_VALUES)
    outcome = run(bootstrap(settings, env_path=tmp_path / "a.env"))
    failed = [r for r in outcome.results if r.status is not CheckStatus.SKIPPED]
    assert {r.name for r in failed} >= {
        "store:postgres",
        "store:mongodb",
        "store:qdrant",
        "store:neo4j",
        "store:redis",
        "birth_date",
    }
    assert all("refused the connection string" in r.detail for r in failed)
    _assert_no_secret(outcome.results)


def test_bootstrap_seed_error_detail_masks_store_passwords(tmp_path, monkeypatch):
    async def init(settings=None, **kwargs):
        return CheckResult("store:postgres", CheckStatus.OK, "done")

    async def seed(settings=None, *, persona=None, birth_date=None, graph=True):
        raise _echoing_error("seed")

    monkeypatch.setattr(postgres_store, "init", init)
    monkeypatch.setattr(seed_db, "seed_stores", seed)
    settings = make_settings(tmp_path, POSTGRES_ENABLED=True, **SECRET_VALUES)
    outcome = run(bootstrap(settings))
    seeded = [r for r in outcome.results if r.name == "seed"]
    assert len(seeded) == 1 and "refused the connection string" in seeded[0].detail
    _assert_no_secret(outcome.results)
