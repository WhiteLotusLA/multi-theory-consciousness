"""
Qdrant Setup and Check
======================

Probes, initializes and checks the vector collection against the registry's
``COLLECTION_PARAMS`` (size and distance filled from Settings by
``collection_params``) and ``PAYLOAD_INDEXES``.

- ``probe`` lists the collections within ``timeout_s``.
- ``init`` creates the collection with the registry's vector size, distance
  and HNSW parameters when it is missing, then any missing payload index. An
  existing collection whose size or distance differs is ``failed`` and is
  never recreated: the remedy is a new ``QDRANT_COLLECTION`` name plus the
  re-embed backfill (``python3 -m mtc.db.reembed --all``). An older
  8192-dimension collection is handled the same way.
- ``check`` reports the collection (existence, size and distance; required),
  its HNSW parameters (advisory ``warning`` on a difference) and the payload
  indexes (advisory ``warning`` when one is missing or has another type).

The client comes from ``qdrant_client`` through ``require_client`` with the
host and port from Settings; its calls run in ``asyncio.to_thread``. Config
keys read at call time: ``QDRANT_HOST``, ``QDRANT_PORT``,
``QDRANT_COLLECTION``, ``VECTOR_DIMENSIONS`` and ``VECTOR_SIMILARITY_METRIC``.
The collection holds embeddings of conversation memories with their
PostgreSQL ids; no point is read here, and nothing is deleted or recreated.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple

from mtc.core.checks import CheckResult, CheckStatus, describe_error, status_for_error
from mtc.core.config import Settings, get_settings
from mtc.core.schema_registry import load_artifact
from mtc.memory.clients import require_client

STORE = "qdrant"
EXTRA = "qdrant"
CHECK_NAME = "store:qdrant"
REEMBED_HINT = (
    "set MTC_QDRANT_COLLECTION to a new name, run mtc db init, then backfill it "
    "with python3 -m mtc.db.reembed --all"
)
HNSW_KEYS = ("m", "ef_construct", "full_scan_threshold", "on_disk")
DISTANCES = ("cosine", "dot", "euclid")
COLLECTION_PARAMS_PATH = "mtc.neural.vectors.coordinator.COLLECTION_PARAMS"
PAYLOAD_INDEXES_PATH = "mtc.neural.vectors.coordinator.PAYLOAD_INDEXES"


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def registry_params(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """The registry's collection parameters with the size and distance set
    from ``VECTOR_DIMENSIONS`` and ``VECTOR_SIMILARITY_METRIC``."""
    active = _settings(settings)
    params = copy.deepcopy(load_artifact(COLLECTION_PARAMS_PATH))
    metric = str(active.VECTOR_SIMILARITY_METRIC).strip().lower()
    if metric not in DISTANCES:
        raise ValueError(
            f"VECTOR_SIMILARITY_METRIC {active.VECTOR_SIMILARITY_METRIC!r} is not "
            f"one of {DISTANCES}"
        )
    params["size"] = int(active.VECTOR_DIMENSIONS)
    params["distance"] = metric
    return params


def payload_indexes() -> Tuple[Tuple[str, str], ...]:
    """The registry's payload indexes as ``(field, type)`` pairs."""
    return tuple(load_artifact(PAYLOAD_INDEXES_PATH))


def create_client(settings: Optional[Settings] = None, *, timeout_s: float = 2.0):
    """A Qdrant client for the configured host and port (no request is sent)."""
    qdrant = require_client("qdrant_client", "qdrant")
    active = _settings(settings)
    arguments: Dict[str, Any] = {
        "host": active.QDRANT_HOST,
        "port": int(active.QDRANT_PORT),
        "timeout": max(1, int(round(timeout_s))),
    }
    try:
        accepted = inspect.signature(qdrant.QdrantClient).parameters
    except (TypeError, ValueError):
        accepted = {}
    if "check_compatibility" in accepted:
        arguments["check_compatibility"] = False
    return qdrant.QdrantClient(**arguments)


def _error(exc: BaseException) -> CheckResult:
    if isinstance(exc, ImportError):
        return CheckResult(
            CHECK_NAME, CheckStatus.FAILED, f"{exc}", data={"missing_extra": EXTRA}
        )
    status = status_for_error(exc)
    detail = describe_error(exc)
    if status is CheckStatus.UNREACHABLE:
        return CheckResult(CHECK_NAME, status, f"Qdrant did not answer ({detail})")
    return CheckResult(CHECK_NAME, status, f"Qdrant refused the operation ({detail})")


def _close(client: Any) -> None:
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # closing is best effort after the result is known
            pass


def _names(client: Any) -> List[str]:
    return [item.name for item in client.get_collections().collections]


async def probe(
    settings: Optional[Settings] = None, *, timeout_s: float = 2.0
) -> CheckResult:
    """Quick reachability check: list the collections."""
    active = _settings(settings)
    client = None
    try:
        client = create_client(active, timeout_s=timeout_s)
        names = await asyncio.wait_for(
            asyncio.to_thread(_names, client), timeout=timeout_s + 1.0
        )
    except Exception as exc:
        return _error(exc)
    finally:
        if client is not None:
            _close(client)
    present = "present" if active.QDRANT_COLLECTION in names else "not created yet"
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        f"answered at {active.QDRANT_HOST}:{active.QDRANT_PORT}; collection "
        f"{active.QDRANT_COLLECTION} {present}",
    )


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def describe_collection(info: Any) -> Dict[str, Any]:
    """Size, distance, HNSW parameters and payload index types of a collection."""
    vectors = info.config.params.vectors
    hnsw = info.config.hnsw_config
    schema = getattr(info, "payload_schema", None) or {}
    return {
        "size": getattr(vectors, "size", None),
        "distance": str(_value(getattr(vectors, "distance", ""))).lower(),
        "hnsw": {key: getattr(hnsw, key, None) for key in HNSW_KEYS},
        "payload": {
            str(field): str(_value(getattr(entry, "data_type", ""))).lower()
            for field, entry in schema.items()
        },
    }


def _mismatch(found: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    problems = []
    if found["size"] != params["size"]:
        problems.append(
            f"holds {found['size']}-dimension vectors but VECTOR_DIMENSIONS is "
            f"{params['size']}"
        )
    if found["distance"] != params["distance"]:
        problems.append(
            f"uses {found['distance']} distance but VECTOR_SIMILARITY_METRIC is "
            f"{params['distance']}"
        )
    return problems


def _init_sync(client: Any, active: Settings) -> CheckResult:
    models = require_client("qdrant_client.models", "qdrant")
    params = registry_params(active)
    name = active.QDRANT_COLLECTION
    created = name not in _names(client)
    if created:
        distances = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=params["size"], distance=distances[params["distance"]]
            ),
            hnsw_config=models.HnswConfigDiff(**params["hnsw_config"]),
        )
    found = describe_collection(client.get_collection(name))
    problems = _mismatch(found, params)
    if problems:
        return CheckResult(
            CHECK_NAME,
            CheckStatus.FAILED,
            f"collection {name} {'; '.join(problems)}; it is never recreated: "
            f"{REEMBED_HINT}",
            data={"collection": name, "found": found, "created": False},
        )
    types = {
        "integer": models.PayloadSchemaType.INTEGER,
        "keyword": models.PayloadSchemaType.KEYWORD,
        "float": models.PayloadSchemaType.FLOAT,
    }
    indexes_created = []
    for field, kind in payload_indexes():
        if field in found["payload"]:
            continue
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=types[kind]
        )
        indexes_created.append(field)
    if created:
        detail = (
            f"created collection {name} ({params['size']} dimensions, "
            f"{params['distance']}) with {len(indexes_created)} payload indexes"
        )
    elif indexes_created:
        detail = (
            f"collection {name} present; created payload indexes "
            f"{', '.join(indexes_created)}"
        )
    else:
        detail = f"collection {name} present with its payload indexes; nothing created"
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        detail,
        data={
            "collection": name,
            "created": created,
            "payload_indexes_created": indexes_created,
        },
    )


async def init(
    client: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 10.0,
) -> CheckResult:
    """Create the collection and payload indexes when missing; never recreate."""
    active = _settings(settings)
    owned = client is None
    try:
        if owned:
            client = create_client(active, timeout_s=timeout_s)
        return await asyncio.to_thread(_init_sync, client, active)
    except Exception as exc:
        return _error(exc)
    finally:
        if owned and client is not None:
            _close(client)


def _check_sync(client: Any, active: Settings) -> List[CheckResult]:
    params = registry_params(active)
    name = active.QDRANT_COLLECTION
    if name not in _names(client):
        return [
            CheckResult(
                "qdrant:collection",
                CheckStatus.FAILED,
                f"collection {name} does not exist; run mtc db init",
                data={"collection": name},
            )
        ]
    found = describe_collection(client.get_collection(name))
    results = []
    problems = _mismatch(found, params)
    if problems:
        results.append(
            CheckResult(
                "qdrant:collection",
                CheckStatus.FAILED,
                f"collection {name} {'; '.join(problems)}; {REEMBED_HINT}",
                data={"collection": name, "found": found},
            )
        )
    else:
        results.append(
            CheckResult(
                "qdrant:collection",
                CheckStatus.OK,
                f"collection {name}: {found['size']} dimensions, {found['distance']}",
                data={"collection": name},
            )
        )
    expected_hnsw = params["hnsw_config"]
    differing = [
        f"{key} {found['hnsw'].get(key)} (expected {expected_hnsw[key]})"
        for key in HNSW_KEYS
        if key in expected_hnsw and found["hnsw"].get(key) != expected_hnsw[key]
    ]
    if differing:
        results.append(
            CheckResult(
                "qdrant:hnsw",
                CheckStatus.WARNING,
                f"HNSW parameters differ from the registry: {', '.join(differing)}",
                required=False,
                data={"found": found["hnsw"], "expected": expected_hnsw},
            )
        )
    else:
        results.append(
            CheckResult(
                "qdrant:hnsw",
                CheckStatus.OK,
                "HNSW parameters match the registry",
                required=False,
            )
        )
    wrong = []
    for field, kind in payload_indexes():
        actual = found["payload"].get(field)
        if actual != kind:
            wrong.append(f"{field} ({actual or 'missing'}, expected {kind})")
    if wrong:
        results.append(
            CheckResult(
                "qdrant:payload_indexes",
                CheckStatus.WARNING,
                f"payload indexes differ: {', '.join(wrong)}; mtc db init creates "
                "missing ones",
                required=False,
                data={"found": found["payload"]},
            )
        )
    else:
        results.append(
            CheckResult(
                "qdrant:payload_indexes",
                CheckStatus.OK,
                f"all {len(payload_indexes())} payload indexes present",
                required=False,
            )
        )
    return results


async def check(
    client: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 10.0,
) -> List[CheckResult]:
    """Check the collection, its HNSW parameters and its payload indexes."""
    active = _settings(settings)
    owned = client is None
    try:
        if owned:
            client = create_client(active, timeout_s=timeout_s)
        return await asyncio.to_thread(_check_sync, client, active)
    except Exception as exc:
        return [_error(exc)]
    finally:
        if owned and client is not None:
            _close(client)
