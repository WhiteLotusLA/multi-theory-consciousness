"""
Memory Factory
==============

``build_memory_manager`` is the one place that builds the memory layer's
stores from Settings and wires them into an ``EnhancedMemoryManager``:

- PostgreSQL (``postgres=True``): an asyncpg pool from ``create_postgres_pool``
  (the ``POSTGRES_*`` and ``DB_POOL_*`` keys) wrapped in ``ResilientPool``, and
  the ``PostgresEpisodicStore`` over it. It is the primary store, so a missing
  ``asyncpg`` package raises the ImportError that names the ``postgres`` extra
  and an unreachable server raises ``StoreUnavailable``; nothing else is built
  then. With ``postgres=False`` the episodic store is in memory.
- Vectors (``vectors=True``): a ``VectorCoordinator`` (``QDRANT_*`` and
  ``VECTOR_*`` keys) with the given embedder, or the default
  sentence-transformers one.
- Graph (``graph=True``; off by default): a ``KnowledgeGraphManager``
  (``NEO4J_*`` keys) that has answered ``connect()``, with a
  ``ConceptExtractor`` as the concept matcher unless one is passed.

Each side store is built in its own try: a vector or graph store that fails is
left out, named in ``components`` with its error and logged at WARNING, and
never takes PostgreSQL down with it (the reference implementation lost every
store when one was missing). A store the caller skipped is never constructed
and is reported as ``skipped``. Every component's ``status()`` (``backend``,
``available``, ``last_error``) is recorded in ``MemoryStack.components``.

The ``*_builder`` arguments replace the default constructors (tests, a GPU
coordinator from ``mtc.neural.gpu_integration``, or a store an operator built
by hand). The factory does not initialize the manager: the first call loads
and seeds core memory. The caller reads the per-store enable keys
(``POSTGRES_ENABLED``, ``QDRANT_ENABLED``, ``NEO4J_ENABLED``) and passes them
as the ``postgres``, ``vectors`` and ``graph`` flags; this module reads none
of them.
No persisted state of its own.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from mtc.core.config import Settings, get_settings
from mtc.core.embeddings import Embedder
from mtc.memory.clients import create_postgres_pool
from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager
from mtc.memory.episodic_store import InMemoryEpisodicStore, PostgresEpisodicStore
from mtc.memory.protocols import (
    ConceptMatcher,
    EpisodeConsolidator,
    EpisodicStore,
    GraphStore,
    VectorStore,
)
from mtc.memory.resilient_wrappers import ResilientPool

logger = logging.getLogger(__name__)

PoolBuilder = Callable[[Settings], Awaitable[Any]]
VectorBuilder = Callable[[Settings, Optional[Embedder]], VectorStore]
GraphBuilder = Callable[[Settings], GraphStore]


@dataclass
class MemoryStack:
    """The built manager, its stores and each component's status."""

    manager: EnhancedMemoryManager
    pool: Optional[Any]
    vector_store: Optional[VectorStore]
    graph: Optional[GraphStore]
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    async def close(self) -> None:
        """Shut the manager down (it closes the episodic, vector and graph stores)."""
        await self.manager.shutdown()


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _status_of(obj: Any) -> Dict[str, Any]:
    reader = getattr(obj, "status", None)
    if callable(reader):
        try:
            report = dict(reader())
        except Exception as exc:  # noqa: BLE001 - status never raises
            report = {"available": False, "last_error": _describe(exc)}
    else:
        report = {"available": True, "last_error": None}
    report.setdefault("backend", type(obj).__name__)
    report.setdefault("available", True)
    report.setdefault("last_error", None)
    return report


def _skipped() -> Dict[str, Any]:
    return {"backend": None, "available": False, "last_error": None, "skipped": True}


def _failed(backend: str, exc: BaseException) -> Dict[str, Any]:
    return {"backend": backend, "available": False, "last_error": _describe(exc)}


async def _default_pool(settings: Settings) -> Any:
    return await create_postgres_pool(settings)


def _default_vectors(settings: Settings, embedder: Optional[Embedder]) -> VectorStore:
    from mtc.neural.vectors.coordinator import VectorCoordinator

    return VectorCoordinator(settings=settings, embedder=embedder)


def _default_graph(settings: Settings) -> GraphStore:
    from mtc.knowledge.graph_manager import KnowledgeGraphManager

    graph = KnowledgeGraphManager(settings=settings)
    graph.connect()
    return graph


def _default_matcher() -> ConceptMatcher:
    from mtc.knowledge.concept_extraction import ConceptExtractor

    return ConceptExtractor()


async def build_memory_manager(
    settings: Optional[Settings] = None,
    *,
    postgres: bool = True,
    vectors: bool = True,
    graph: bool = False,
    consolidator: Optional[EpisodeConsolidator] = None,
    embedder: Optional[Embedder] = None,
    concept_matcher: Optional[ConceptMatcher] = None,
    pool_builder: Optional[PoolBuilder] = None,
    vector_builder: Optional[VectorBuilder] = None,
    graph_builder: Optional[GraphBuilder] = None,
    clock: Optional[Callable[[], datetime]] = None,
) -> MemoryStack:
    """Build the stores that were asked for, each in its own try, and the
    manager over them; see the module notes for what fails and what degrades."""
    active = settings if settings is not None else get_settings()
    components: Dict[str, Dict[str, Any]] = {}

    pool: Optional[Any] = None
    episodic: EpisodicStore
    if postgres:
        try:
            raw = await (pool_builder or _default_pool)(active)
            pool = ResilientPool(raw, settings=active)
            episodic = PostgresEpisodicStore(pool, clock=clock)
        except Exception as exc:
            components["postgres"] = _failed("postgres", exc)
            logger.error("PostgreSQL could not be built: %s", _describe(exc))
            raise
        components["postgres"] = _status_of(pool)
    else:
        episodic = InMemoryEpisodicStore(clock=clock)
        components["postgres"] = _skipped()
    components["episodic"] = _status_of(episodic)

    vector_store: Optional[VectorStore] = None
    if vectors:
        try:
            vector_store = (vector_builder or _default_vectors)(active, embedder)
        except Exception as exc:  # noqa: BLE001 - recorded, PostgreSQL stays
            components["vectors"] = _failed("qdrant", exc)
            logger.warning(
                "vector store left out: %s", components["vectors"]["last_error"]
            )
        else:
            components["vectors"] = _status_of(vector_store)
    else:
        components["vectors"] = _skipped()

    graph_store: Optional[GraphStore] = None
    if graph:
        try:
            graph_store = (graph_builder or _default_graph)(active)
        except Exception as exc:  # noqa: BLE001 - recorded, PostgreSQL stays
            components["graph"] = _failed("neo4j", exc)
            logger.warning(
                "graph store left out: %s", components["graph"]["last_error"]
            )
        else:
            components["graph"] = _status_of(graph_store)
    else:
        components["graph"] = _skipped()

    matcher = concept_matcher
    if matcher is None and graph_store is not None:
        matcher = _default_matcher()
    components["concept_matcher"] = (
        _status_of(matcher)
        if matcher is not None
        else {
            "backend": None,
            "available": False,
            "last_error": None,
            "reason": "absent",
        }
    )
    components["consolidator"] = (
        _status_of(consolidator)
        if consolidator is not None
        else {
            "backend": None,
            "available": False,
            "last_error": None,
            "reason": "absent",
        }
    )

    manager = EnhancedMemoryManager(
        episodic_store=episodic,
        vector_store=vector_store,
        graph=graph_store,
        concept_matcher=matcher,
        consolidator=consolidator,
        settings=active,
        clock=clock,
    )
    logger.info(
        "memory stack built: %s",
        {name: report.get("available") for name, report in components.items()},
    )
    return MemoryStack(
        manager=manager,
        pool=pool,
        vector_store=vector_store,
        graph=graph_store,
        components=components,
    )
