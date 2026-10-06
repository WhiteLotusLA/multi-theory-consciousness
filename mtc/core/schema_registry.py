"""
Schema Registry
===============

One registry of every store module's schema artifact. ``SCHEMA_ARTIFACTS`` maps
a store to the dotted paths of the module constants that define its schema:
PostgreSQL DDL tuples, MongoDB index tables, the Qdrant collection parameters
and payload indexes, the Neo4j constraints and neutral seed concepts, and the
Redis key families. ``load_artifact`` imports one by its dotted path and
returns the constant.

Phase 7's migrations, ``init_mongo``, ``init_qdrant``, ``init_neo4j``,
``bootstrap`` and ``verify`` read this registry and nothing else; they do not
import store modules by name. Each store module keeps its own ``ensure_*``
safety net, and Phase 7's scripts are canonical. A phase that adds a store
registers its constant here (the conversation repository's DDL,
``CONVERSATION_SCHEMA_SQL``, is registered under ``postgres``); a test scans
``mtc`` for the artifact naming convention (``*_SCHEMA_SQL``, ``*_INDEXES``,
``COLLECTION_PARAMS``, ``PAYLOAD_INDEXES``, ``CONSTRAINTS``, ``KEY_FAMILIES``)
and fails when a constant is not registered.

Loading an artifact imports its module, and importing any ``mtc`` module loads
none of the store client packages (asyncpg, pymongo, redis, neo4j,
qdrant_client), so reading the registry needs no extra and opens no
connection. No artifact holds a password, a DROP statement, a seed row of
identity data or the vector-extension name. The module reads no config key and
persists nothing.
"""

import importlib
from typing import Any, Dict, List, Tuple

SCHEMA_ARTIFACTS: Dict[str, Tuple[str, ...]] = {
    "postgres": (
        "mtc.memory.episodic_store.EPISODIC_SCHEMA_SQL",
        "mtc.memory.postgres_stores.RESEARCH_SCHEMA_SQL",
        "mtc.safety.stores.SAFETY_SCHEMA_SQL",
        "mtc.memory.conversation_store.CONVERSATION_SCHEMA_SQL",
    ),
    "mongodb": (
        "mtc.neural.mongodb_schemas.NEURAL_INDEXES",
        "mtc.memory.mongo_stores.DOCUMENT_INDEXES",
        "mtc.privacy.stores.PRIVACY_INDEXES",
        "mtc.monitoring.stores.ALERT_INDEXES",
    ),
    "qdrant": (
        "mtc.neural.vectors.coordinator.COLLECTION_PARAMS",
        "mtc.neural.vectors.coordinator.PAYLOAD_INDEXES",
    ),
    "neo4j": (
        "mtc.knowledge.graph_manager.CONSTRAINTS",
        "mtc.knowledge.graph_manager.SEED_CONCEPTS",
    ),
    "redis": ("mtc.memory.redis_manager.KEY_FAMILIES",),
}


def registered_paths() -> List[str]:
    """Every dotted path in the registry, in store order."""
    return [path for paths in SCHEMA_ARTIFACTS.values() for path in paths]


def load_artifact(dotted: str) -> Any:
    """Import the module part of a registered dotted path and return the
    constant.

    A path that is not in ``SCHEMA_ARTIFACTS`` raises ValueError, so the
    registry cannot be used to import arbitrary modules. A registered path that
    cannot be imported or whose constant is missing raises ImportError or
    AttributeError naming the path.
    """
    if dotted not in registered_paths():
        raise ValueError(f"{dotted!r} is not a registered schema artifact")
    module_name, _, attribute = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(
            f"schema artifact {dotted!r}: module has no constant {attribute!r}"
        ) from exc


def artifacts_for(store: str) -> Dict[str, Any]:
    """Load every artifact registered for ``store`` (``postgres``, ``mongodb``,
    ``qdrant``, ``neo4j`` or ``redis``), keyed by dotted path. An unknown store
    raises ValueError naming the known ones."""
    if store not in SCHEMA_ARTIFACTS:
        raise ValueError(
            f"unknown store {store!r}; registered: {', '.join(SCHEMA_ARTIFACTS)}"
        )
    return {path: load_artifact(path) for path in SCHEMA_ARTIFACTS[store]}
