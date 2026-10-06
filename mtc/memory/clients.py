"""
Store Client Factories
======================

The only place the framework builds network clients for its stores. Every
client package (``asyncpg``, ``pymongo``, ``redis``, and in later modules
``neo4j`` and ``qdrant_client``) is optional and imported through
``require_client`` inside a factory call, never at import: importing any
``mtc`` module loads none of them, and a missing package raises an
``ImportError`` that names the extra that installs it (``postgres``,
``mongo``, ``redis``, ``neo4j`` or ``qdrant``).

Each factory reads Settings at call time; no host, port or URI literal appears
here:

- ``create_postgres_pool``: ``POSTGRES_HOST``, ``POSTGRES_PORT``,
  ``POSTGRES_DB``, ``POSTGRES_USER``, ``POSTGRES_PASSWORD`` and the ``DB_POOL_*``
  keys (min and max size, connect timeout, command timeout and
  ``DB_POOL_IDLE_TIMEOUT`` as the idle connection lifetime). JSON and JSONB
  codecs are installed per connection so stores pass and receive dicts.
- ``create_mongo_client``: ``MONGODB_HOST``, ``MONGODB_PORT``, the optional
  credentials with ``MONGODB_AUTH_SOURCE`` as the authentication database
  (empty keeps ``MONGODB_DB``, the database ``MONGODB_URL`` names; no
  ``authSource`` is passed without credentials), ``tz_aware=True``,
  ``MONGODB_SERVER_SELECTION_TIMEOUT_MS`` and
  ``connect=False``, so building a client opens no connection (the driver
  connects on the first operation).
- ``create_redis_client``: ``REDIS_HOST``, ``REDIS_PORT``, ``REDIS_DB``,
  ``REDIS_PASSWORD``, ``REDIS_MAX_CONNECTIONS``, ``REDIS_SOCKET_TIMEOUT`` and
  ``decode_responses=True``.

A server that cannot be reached raises ``StoreUnavailable`` chained to the
driver's error. ``IndexSpec``, ``apply_indexes`` and ``apply_schema`` are the
shared helpers each store module's ``ensure_*`` function uses to apply its
schema artifact as a safety net; the migrations and init scripts are canonical.
"""

import importlib
import json
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from mtc.core.config import Settings, get_settings
from mtc.memory.protocols import StoreUnavailable

logger = logging.getLogger(__name__)


def require_client(module: str, extra: str) -> ModuleType:
    """Import an optional client package or raise an ImportError naming the extra."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is not installed; install the {extra!r} extra: "
            f"pip install multi-theory-consciousness[{extra}]"
        ) from exc


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


async def _install_json_codecs(connection: Any) -> None:
    for type_name in ("json", "jsonb"):
        await connection.set_type_codec(
            type_name, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


async def create_postgres_pool(
    settings: Optional[Settings] = None, **overrides: Any
) -> Any:
    """Create an asyncpg pool from Settings; ``overrides`` replace pool arguments."""
    asyncpg = require_client("asyncpg", "postgres")
    active = _settings(settings)
    arguments = {
        "host": active.POSTGRES_HOST,
        "port": active.POSTGRES_PORT,
        "database": active.POSTGRES_DB,
        "user": active.POSTGRES_USER,
        "password": active.POSTGRES_PASSWORD or None,
        "min_size": active.DB_POOL_MIN_SIZE,
        "max_size": active.DB_POOL_MAX_SIZE,
        "timeout": active.DB_POOL_TIMEOUT,
        "command_timeout": active.DB_POOL_COMMAND_TIMEOUT,
        "max_inactive_connection_lifetime": active.DB_POOL_IDLE_TIMEOUT,
        "init": _install_json_codecs,
    }
    arguments.update(overrides)
    try:
        return await asyncpg.create_pool(**arguments)
    except Exception as exc:
        raise StoreUnavailable(
            f"PostgreSQL did not accept a connection: {exc}"
        ) from exc


def create_mongo_client(settings: Optional[Settings] = None, **overrides: Any) -> Any:
    """Create a pymongo client from Settings without connecting.

    The client is built with ``connect=False``: construction starts no
    background connection or server monitoring, and the driver connects on the
    first operation. ``overrides`` replace client arguments, so an explicit
    ``connect=True`` still connects at construction.
    """
    pymongo = require_client("pymongo", "mongo")
    active = _settings(settings)
    arguments = {
        "host": active.MONGODB_HOST,
        "port": active.MONGODB_PORT,
        "tz_aware": True,
        "serverSelectionTimeoutMS": active.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
        "connect": False,
    }
    if active.MONGODB_USER and active.MONGODB_PASSWORD:
        arguments["username"] = active.MONGODB_USER
        arguments["password"] = active.MONGODB_PASSWORD
        # An empty MONGODB_AUTH_SOURCE keeps the database MONGODB_URL names.
        arguments["authSource"] = active.MONGODB_AUTH_SOURCE or active.MONGODB_DB
    arguments.update(overrides)
    try:
        return pymongo.MongoClient(**arguments)
    except Exception as exc:
        raise StoreUnavailable(f"MongoDB client could not be built: {exc}") from exc


def mongo_database(
    client: Any = None,
    *,
    settings: Optional[Settings] = None,
    name: Optional[str] = None,
) -> Any:
    """Return the configured (or named) MongoDB database of a client.

    Without ``client`` one is built through ``create_mongo_client``, so it
    does not connect until the first operation.
    """
    active = _settings(settings)
    owner = client if client is not None else create_mongo_client(active)
    return owner[name or active.MONGODB_DB]


def create_redis_client(settings: Optional[Settings] = None, **overrides: Any) -> Any:
    """Create an asyncio Redis client from Settings (it connects lazily)."""
    aioredis = require_client("redis.asyncio", "redis")
    active = _settings(settings)
    arguments = {
        "host": active.REDIS_HOST,
        "port": active.REDIS_PORT,
        "db": active.REDIS_DB,
        "password": active.REDIS_PASSWORD or None,
        "max_connections": active.REDIS_MAX_CONNECTIONS,
        "socket_timeout": active.REDIS_SOCKET_TIMEOUT,
        "decode_responses": True,
    }
    arguments.update(overrides)
    try:
        return aioredis.Redis(**arguments)
    except Exception as exc:
        raise StoreUnavailable(f"Redis client could not be built: {exc}") from exc


@dataclass(frozen=True)
class IndexSpec:
    """One MongoDB index: its keys, uniqueness and optional name."""

    keys: Tuple[Tuple[str, Union[int, str]], ...]
    unique: bool = False
    name: Optional[str] = None


def apply_indexes(database: Any, indexes: Mapping[str, Sequence[IndexSpec]]) -> int:
    """Create each index (idempotent); return how many specs were applied."""
    applied = 0
    for collection, specs in indexes.items():
        for spec in specs:
            options = {"unique": spec.unique}
            if spec.name:
                options["name"] = spec.name
            database[collection].create_index(list(spec.keys), **options)
            applied += 1
    return applied


async def apply_schema(pool: Any, statements: Sequence[str]) -> int:
    """Run idempotent DDL statements in one transaction; return how many ran."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            for statement in statements:
                await connection.execute(statement)
    return len(statements)
