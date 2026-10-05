"""Shared fixtures: throwaway stores for the ``db`` tests.

Tests of a real store implementation are marked ``db`` and use the five
fixtures below. Each fixture skips unless ``MTC_TEST_STORES=1`` is set, the
client package is installed and the server answers within two seconds. Each
creates a name from ``MTC_TEST_DB_PREFIX`` (default ``mtc_test``) plus eight
random hex digits (Neo4j names use dashes), asserts the name matches that
pattern before creating it and again before dropping it, and touches nothing
else on the server: no fixture ever flushes a Redis database or writes to a
default graph, database, collection or key space.

``MTC_TEST_STORES`` and ``MTC_TEST_DB_PREFIX`` are test environment variables,
not Settings keys. Connection details come from ``Settings`` (the ``MTC_*``
environment or a ``.env`` file).

What each fixture yields:

- ``throwaway_postgres``: an asyncpg pool on a new database.
- ``throwaway_mongo``: a pymongo database with a new name.
- ``throwaway_qdrant``: a ``QdrantTarget`` (client and a new collection).
- ``throwaway_redis``: a ``RedisTarget`` (client, key prefix and a Settings
  copy whose ``REDIS_KEY_PREFIX`` is that prefix).
- ``throwaway_neo4j``: a ``Neo4jTarget`` (driver and a new database name).

The session also defaults ``MTC_DEVICE`` to ``cpu`` (set before any test
module is imported); an exported ``MTC_DEVICE`` still wins.
"""

import os
import re
import secrets
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import pytest_asyncio

# The machine running the suite may share its accelerator with a live system,
# so no test claims it by default: accelerator tests opt in explicitly.
os.environ.setdefault("MTC_DEVICE", "cpu")

OPT_IN_VARIABLE = "MTC_TEST_STORES"
PREFIX_VARIABLE = "MTC_TEST_DB_PREFIX"
DEFAULT_PREFIX = "mtc_test"
PROBE_TIMEOUT_S = 2.0
OPT_IN_REASON = "set MTC_TEST_STORES=1 to run tests against throwaway stores"
NEO4J_COMMUNITY_REASON = (
    "Neo4j server cannot create a separate database (community edition); "
    "tests never write to the default graph"
)


def prefix_in_use() -> str:
    """The prefix every throwaway name starts with."""
    return os.environ.get(PREFIX_VARIABLE, DEFAULT_PREFIX)


def throwaway_name(*, dashes: bool = False) -> str:
    """Return a fresh name: the prefix, a separator and eight hex digits."""
    prefix = prefix_in_use()
    separator = "-" if dashes else "_"
    if dashes:
        prefix = prefix.replace("_", "-")
    return f"{prefix}{separator}{secrets.token_hex(4)}"


def assert_throwaway(name: str, *, dashes: bool = False) -> str:
    """Return ``name`` when it is a throwaway name; raise AssertionError if not."""
    prefix = prefix_in_use()
    if dashes:
        prefix = prefix.replace("_", "-")
    separator = "-" if dashes else "_"
    pattern = re.compile(rf"^{re.escape(prefix)}{separator}[0-9a-f]{{8}}$")
    if not pattern.match(name):
        raise AssertionError(
            f"refusing to touch {name!r}: not a throwaway name "
            f"({prefix}{separator}<8 hex digits>)"
        )
    return name


def assert_throwaway_prefix(key_prefix: str) -> str:
    """Check a Redis key prefix of the form ``<throwaway name>:``."""
    if not key_prefix.endswith(":"):
        raise AssertionError(f"refusing key prefix {key_prefix!r}: no trailing colon")
    assert_throwaway(key_prefix[:-1])
    return key_prefix


def require_opt_in() -> None:
    """Skip the calling test unless ``MTC_TEST_STORES=1``."""
    if os.environ.get(OPT_IN_VARIABLE) != "1":
        pytest.skip(OPT_IN_REASON)


def require_package(module: str, extra: str) -> Any:
    """Import a client package or skip naming the extra that installs it."""
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        pytest.skip(f"{module} is not installed (install the {extra!r} extra)")


def neo4j_can_create_databases(driver: Any) -> bool:
    """True when the server edition supports ``CREATE DATABASE``."""
    with driver.session(database="system") as session:
        rows = list(session.run("CALL dbms.components() YIELD edition RETURN edition"))
    return bool(rows) and str(rows[0]["edition"]).lower() == "enterprise"


def require_neo4j_database_creation(driver: Any) -> None:
    """Skip with the community-edition reason unless databases can be created."""
    try:
        creatable = neo4j_can_create_databases(driver)
    except Exception as exc:  # a server that cannot even answer the probe
        pytest.skip(f"{NEO4J_COMMUNITY_REASON} (edition probe failed: {exc})")
    if not creatable:
        pytest.skip(NEO4J_COMMUNITY_REASON)


@dataclass
class QdrantTarget:
    client: Any
    collection: str


@dataclass
class RedisTarget:
    client: Any
    prefix: str
    settings: Any


@dataclass
class Neo4jTarget:
    driver: Any
    database: str


def _settings() -> Any:
    from mtc.core.config import get_settings

    return get_settings()


@pytest_asyncio.fixture
async def throwaway_postgres():
    """An asyncpg pool on a new ``<prefix>_<hex>`` database, dropped after."""
    require_opt_in()
    asyncpg = require_package("asyncpg", "postgres")
    from mtc.memory.clients import create_postgres_pool

    settings = _settings()
    admin: Optional[Any] = None
    try:
        admin = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD or None,
            database="postgres",
            timeout=PROBE_TIMEOUT_S,
        )
    except Exception as exc:
        pytest.skip(f"PostgreSQL did not answer within 2 s: {exc}")
    name = assert_throwaway(throwaway_name())
    pool = None
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        pool = await create_postgres_pool(settings, database=name, min_size=1)
        yield pool
    finally:
        if pool is not None:
            await pool.close()
        assert_throwaway(name)
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()


@pytest.fixture
def throwaway_mongo():
    """A pymongo database with a new ``<prefix>_<hex>`` name, dropped after."""
    require_opt_in()
    require_package("pymongo", "mongo")
    from mtc.memory.clients import create_mongo_client

    settings = _settings()
    client = create_mongo_client(
        settings, serverSelectionTimeoutMS=int(PROBE_TIMEOUT_S * 1000)
    )
    try:
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        pytest.skip(f"MongoDB did not answer within 2 s: {exc}")
    name = assert_throwaway(throwaway_name())
    try:
        yield client[name]
    finally:
        assert_throwaway(name)
        client.drop_database(name)
        client.close()


@pytest.fixture
def throwaway_qdrant():
    """A Qdrant collection with a new ``<prefix>_<hex>`` name, deleted after."""
    require_opt_in()
    qdrant = require_package("qdrant_client", "qdrant")
    models = require_package("qdrant_client.models", "qdrant")
    settings = _settings()
    client = qdrant.QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=int(PROBE_TIMEOUT_S),
    )
    try:
        client.get_collections()
    except Exception as exc:
        client.close()
        pytest.skip(f"Qdrant did not answer within 2 s: {exc}")
    name = assert_throwaway(throwaway_name())
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=settings.VECTOR_DIMENSIONS, distance=models.Distance.COSINE
            ),
        )
        yield QdrantTarget(client=client, collection=name)
    finally:
        assert_throwaway(name)
        try:
            client.delete_collection(collection_name=name)
        finally:
            client.close()


@pytest_asyncio.fixture
async def throwaway_redis():
    """A Redis key prefix ``<prefix>_<hex>:``; only keys under it are deleted."""
    require_opt_in()
    require_package("redis", "redis")
    from mtc.memory.clients import create_redis_client

    settings = _settings()
    client = create_redis_client(
        settings,
        socket_timeout=PROBE_TIMEOUT_S,
        socket_connect_timeout=PROBE_TIMEOUT_S,
    )
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis did not answer within 2 s: {exc}")
    prefix = assert_throwaway_prefix(f"{assert_throwaway(throwaway_name())}:")
    scoped = settings.model_copy(update={"REDIS_KEY_PREFIX": prefix})
    try:
        yield RedisTarget(client=client, prefix=prefix, settings=scoped)
    finally:
        assert_throwaway_prefix(prefix)
        async for key in client.scan_iter(match=f"{prefix}*", count=200):
            await client.delete(key)
        await client.aclose()


@pytest.fixture
def throwaway_neo4j():
    """A Neo4j database with a new dashed name; skips on community edition."""
    require_opt_in()
    neo4j = require_package("neo4j", "neo4j")
    settings = _settings()
    driver = neo4j.GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        connection_timeout=PROBE_TIMEOUT_S,
    )
    try:
        driver.verify_connectivity()
    except Exception as exc:
        driver.close()
        pytest.skip(f"Neo4j did not answer within 2 s: {exc}")
    try:
        require_neo4j_database_creation(driver)
    except BaseException:
        driver.close()
        raise
    name = assert_throwaway(throwaway_name(dashes=True), dashes=True)
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{name}` WAIT").consume()
        yield Neo4jTarget(driver=driver, database=name)
    finally:
        assert_throwaway(name, dashes=True)
        try:
            with driver.session(database="system") as session:
                session.run(f"DROP DATABASE `{name}` IF EXISTS WAIT").consume()
        finally:
            driver.close()
