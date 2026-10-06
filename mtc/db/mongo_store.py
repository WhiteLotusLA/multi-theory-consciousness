"""
MongoDB Setup and Check
=======================

Probes, initializes and checks the MongoDB database against the registry's
index tables (``SCHEMA_ARTIFACTS["mongodb"]``: the neural, document, privacy
and alert indexes).

- ``mongo_artifacts`` merges the four registered index tables into one
  ``{collection: [IndexSpec, ...]}`` mapping; a collection named by two
  tables raises ValueError, since the tables would then disagree.
- ``probe`` builds a client through ``mtc.memory.clients.create_mongo_client``
  with a short server-selection timeout and runs ``ping``.
- ``init`` applies every index with ``apply_indexes`` (``create_index`` is
  idempotent) and reports which indexes were created; ``check`` compares each
  collection's indexes by keys and uniqueness.

pymongo is synchronous, so each call runs in ``asyncio.to_thread``. A refused
authentication is ``failed`` with a hint naming ``MONGODB_AUTH_SOURCE``; an
unreachable server is ``unreachable``; a missing ``pymongo`` is ``failed``
naming the ``mongo`` extra. Nothing is dropped or deleted.

Config keys read at call time: ``MONGODB_HOST``, ``MONGODB_PORT``,
``MONGODB_DB``, ``MONGODB_USER``, ``MONGODB_PASSWORD``, ``MONGODB_AUTH_SOURCE``
and ``MONGODB_SERVER_SELECTION_TIMEOUT_MS``. The database holds conversation
turns, documents, neural telemetry and alerts; no document is read here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple

from mtc.core.checks import CheckResult, CheckStatus, describe_error, status_for_error
from mtc.core.config import Settings, get_settings
from mtc.core.schema_registry import SCHEMA_ARTIFACTS, load_artifact
from mtc.memory.clients import IndexSpec, apply_indexes, create_mongo_client

STORE = "mongodb"
EXTRA = "mongo"
CHECK_NAME = "store:mongodb"

IndexKey = Tuple[str, Tuple[Tuple[str, Any], ...]]


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def mongo_artifacts() -> Dict[str, List[IndexSpec]]:
    """Every registered MongoDB index table merged by collection."""
    merged: Dict[str, List[IndexSpec]] = {}
    owner: Dict[str, str] = {}
    for dotted in SCHEMA_ARTIFACTS["mongodb"]:
        for collection, specs in load_artifact(dotted).items():
            if collection in merged:
                raise ValueError(
                    f"collection {collection!r} is named by both {owner[collection]} "
                    f"and {dotted}"
                )
            owner[collection] = dotted
            merged[collection] = list(specs)
    return merged


def _direction(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_keys(keys: Any) -> Tuple[Tuple[str, Any], ...]:
    items = keys.items() if hasattr(keys, "items") else keys
    return tuple((str(field), _direction(value)) for field, value in items)


def _present(database: Any, collections: List[str]) -> Dict[str, Dict[Any, bool]]:
    """``{collection: {keys: unique}}`` for the indexes that exist now."""
    found: Dict[str, Dict[Any, bool]] = {}
    for collection in collections:
        info = database[collection].index_information()
        found[collection] = {
            _normalize_keys(entry.get("key", ())): bool(entry.get("unique", False))
            for entry in info.values()
        }
    return found


def _client_error(exc: BaseException, active: Settings) -> CheckResult:
    if isinstance(exc, ImportError):
        return CheckResult(
            CHECK_NAME, CheckStatus.FAILED, f"{exc}", data={"missing_extra": EXTRA}
        )
    status = status_for_error(exc)
    detail = describe_error(exc, secrets=[active.MONGODB_PASSWORD])
    where = f"{active.MONGODB_HOST}:{active.MONGODB_PORT}"
    if status is CheckStatus.UNREACHABLE:
        hint = f"MongoDB at {where} did not answer"
    elif "auth" in detail.lower():
        hint = (
            "authentication was refused; check MTC_MONGODB_USER, "
            "MTC_MONGODB_PASSWORD and MTC_MONGODB_AUTH_SOURCE in .env"
        )
    else:
        hint = f"MongoDB at {where} refused the operation"
    return CheckResult(CHECK_NAME, status, f"{hint} ({detail})")


def _client(active: Settings, timeout_s: float) -> Any:
    timeout_ms = max(1, int(timeout_s * 1000))
    return create_mongo_client(
        active,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
    )


def _close(client: Any) -> None:
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # closing is best effort after the result is known
            pass


async def probe(
    settings: Optional[Settings] = None, *, timeout_s: float = 2.0
) -> CheckResult:
    """Quick reachability check: ``ping`` with a short selection timeout."""
    active = _settings(settings)
    client = None
    try:
        client = _client(active, timeout_s)
        await asyncio.wait_for(
            asyncio.to_thread(client.admin.command, "ping"), timeout=timeout_s + 1.0
        )
    except Exception as exc:
        return _client_error(exc, active)
    finally:
        if client is not None:
            _close(client)
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        f"answered at {active.MONGODB_HOST}:{active.MONGODB_PORT}, database "
        f"{active.MONGODB_DB}",
    )


def _database(
    database: Any, active: Settings, timeout_s: float
) -> Tuple[Any, Optional[Any]]:
    if database is not None:
        return database, None
    client = _client(active, timeout_s)
    return client[active.MONGODB_DB], client


def _init_sync(database: Any) -> Tuple[int, List[str]]:
    artifacts = mongo_artifacts()
    collections = list(artifacts)
    before = _present(database, collections)
    applied = apply_indexes(database, artifacts)
    after = _present(database, collections)
    created = []
    for collection in collections:
        for keys in after[collection]:
            if keys not in before[collection]:
                created.append(_label(collection, keys))
    return applied, sorted(created)


def _label(collection: str, keys: Tuple[Tuple[str, Any], ...]) -> str:
    rendered = ", ".join(f"{field} {value}" for field, value in keys)
    return f"{collection}({rendered})"


async def init(
    database: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 10.0,
) -> CheckResult:
    """Apply every registry index; report the indexes created."""
    active = _settings(settings)
    client = None
    try:
        target, client = _database(database, active, timeout_s)
        applied, created = await asyncio.to_thread(_init_sync, target)
    except Exception as exc:
        return _client_error(exc, active)
    finally:
        if client is not None:
            _close(client)
    if created:
        detail = f"applied {applied} index specs: created {len(created)} indexes"
    else:
        detail = f"applied {applied} index specs: all present, nothing created"
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        detail,
        data={"applied": applied, "created_indexes": created},
    )


def _check_sync(database: Any) -> List[CheckResult]:
    artifacts = mongo_artifacts()
    present = _present(database, list(artifacts))
    missing: List[str] = []
    wrong_unique: List[str] = []
    total = 0
    for collection, specs in artifacts.items():
        for spec in specs:
            total += 1
            keys = _normalize_keys(spec.keys)
            label = _label(collection, keys)
            if keys not in present[collection]:
                missing.append(label)
            elif present[collection][keys] != bool(spec.unique):
                wrong_unique.append(label)
    problems: Set[str] = set(missing) | set(wrong_unique)
    if problems:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing: {', '.join(missing)}")
        if wrong_unique:
            parts.append(
                f"{len(wrong_unique)} with the wrong uniqueness: "
                f"{', '.join(wrong_unique)}"
            )
        return [
            CheckResult(
                "mongodb:indexes",
                CheckStatus.FAILED,
                f"{'; '.join(parts)} (of {total}); run mtc db init",
                data={"missing": missing, "wrong_unique": wrong_unique},
            )
        ]
    return [
        CheckResult(
            "mongodb:indexes",
            CheckStatus.OK,
            f"all {total} indexes present with their uniqueness",
            data={"missing": [], "wrong_unique": []},
        )
    ]


async def check(
    database: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 10.0,
) -> List[CheckResult]:
    """Compare every registry index by keys and uniqueness."""
    active = _settings(settings)
    client = None
    try:
        target, client = _database(database, active, timeout_s)
        return await asyncio.to_thread(_check_sync, target)
    except Exception as exc:
        return [_client_error(exc, active)]
    finally:
        if client is not None:
            _close(client)
