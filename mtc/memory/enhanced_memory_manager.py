"""
Enhanced Memory Manager
=======================

Core memory, episodic storage, recall and consolidation behind store seams
(``mtc.memory.protocols``):

- ``EpisodicStore`` holds the core blocks, episodic rows and semantic rows. A
  None store means ``InMemoryEpisodicStore``; ``PostgresEpisodicStore`` is the
  persistent one, and its schema artifact ``EPISODIC_SCHEMA_SQL`` is the safety
  net for Phase 7's ``migrations/001``.
- ``VectorStore`` (semantic recall), ``GraphStore`` with a ``ConceptMatcher``
  (concept links and enrichment) and ``EpisodeConsolidator`` (the substrate
  consolidator) are optional parts. An absent part is reported by
  ``status()`` and logged once at WARNING; the call that needs it says which
  part answered or was missing.

Core memory: ``initialize`` loads the ``identity``, ``persona`` and ``human``
blocks and seeds any that are missing (never overwriting) from the persona
(the ``persona`` argument or the file at ``PERSONA_PATH``) or the neutral
templates filled from Settings (``SYSTEM_NAME``, ``CREATOR_NAME``,
``CREATOR_RELATIONSHIP``, ``CORE_VALUES`` and the Big Five seeds); see
``mtc.memory.core_blocks``. ``seed_core_memory`` is the hook Phase 7's
bootstrap calls. The manager keeps the last loaded blocks; when the store is
unavailable it serves them marked stale (or, before any load, the rendered
blocks marked stale with source ``rendered``). While degraded, the next call
after a backoff (``RECOVERY_BACKOFF_SECONDS`` doubling up to
``RECOVERY_BACKOFF_MAX_SECONDS`` on the injected clock) retries the load and
the seeding, and a successful episodic write retries at once; a successful
retry clears ``degraded`` and the stale mark.

Storing: the episodic row is written first. Its id gives the vector point
(``point_id_for_memory``) written with ``postgres_id``, ``memory_type`` and
``created_at`` metadata, then concept links over the graph's concept names
through the matcher (up to five concepts with relevance at least 0.3, method
``keyword_matching``). Each side store runs in its own try: its failure is
recorded in ``StoreResult.errors`` and ``status()`` and never undoes the row.
An unavailable episodic store is reported, not raised. A row the pool buffered
has no id yet, so it is counted as unindexed; Phase 7's re-embed backfill
indexes it later. Synchronous vector, graph, matcher and consolidator calls run
in ``asyncio.to_thread``. Nothing in this module composes a vector: an
unavailable embedder means no point is written and the error says so.

Recall: vector hits are joined to their rows in one ``get_episodic`` call;
a hit whose row is gone or archived is dropped and counted as an orphan, and a
hit with no row id (a gist) is returned from its payload text labelled
``vector_payload``. A missing or failing vector store falls back to text
search labelled ``search_stats.method = "text"`` with the error named; when
that fails too the method is ``none``. Text matches carry no similarity score.
Recency-anchored recall carries ``recency_anchored`` and no similarity score.

Lifecycle boilerplate: rows matching ``boilerplate_patterns`` are kept out of
recency recall and the recent-memory feed. The default pattern matches the
end-of-conversation summary line ``Conversation with <name> ended. <n>
messages exchanged.``; Phase 6's pipeline writes that format or passes its own
pattern.

Consolidation: with a consolidator, the unconsolidated rows are replayed as
episodes and each pattern the pass discovered becomes a semantic row (method
``htm``); without one, rows are grouped by dominant emotion and each group of
two or more becomes a semantic row (method ``simple``). Rows with no numeric
emotional context form the ``unlabelled`` group, whose semantic row carries an
empty emotional context (no emotion is invented for them). Both mark the rows
consolidated; when the consolidator reports no anomaly the strength is
unmeasured, the report says None and the stored strength is left unchanged. ``consolidate()`` is the sleep scheduler's ``Consolidator`` seam.

Episodic memories are private conversation content and live only in
gitignored or operator-controlled stores. Config keys read: ``PERSONA_PATH``,
``SYSTEM_NAME``, ``CREATOR_NAME``, ``CREATOR_RELATIONSHIP``, ``CORE_VALUES``,
``OPENNESS`` to ``NEUROTICISM`` (through the core blocks) and
``VECTOR_SEARCH_LIMIT``.
"""

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from mtc.core.config import Settings, get_settings
from mtc.core.persona import load_persona
from mtc.memory.core_blocks import CORE_BLOCK_NAMES, render_core_blocks
from mtc.memory.episodic_store import InMemoryEpisodicStore
from mtc.memory.protocols import (
    ConceptMatcher,
    EpisodeConsolidator,
    EpisodicStore,
    GraphStore,
    StoreUnavailable,
    VectorStore,
    point_id_for_memory,
)

logger = logging.getLogger(__name__)

DEFAULT_BOILERPLATE_PATTERNS = (
    r"Conversation with .+ ended\.\s*\d+\s+messages? exchanged",
)

# Concept linking on store: the matcher's relevance floor and the link cap.
CONCEPT_MIN_RELEVANCE = 0.3
CONCEPT_MAX_LINKS = 5
CONCEPT_LINK_METHOD = "keyword_matching"
# Query concepts enriched with related concepts during recall.
RECALL_QUERY_CONCEPTS = 3
# The simple consolidation path's design values (no substrate consolidator).
SIMPLE_MIN_GROUP = 2
SIMPLE_CONFIDENCE = 0.7
SIMPLE_STRENGTH = 0.5
SIMPLE_THEME_CHARS = 50
SIMPLE_THEME_ROWS = 3
UNLABELLED_GROUP = "unlabelled"
# Retry of a degraded core-memory load (seconds on the injected clock).
RECOVERY_BACKOFF_SECONDS = 5.0
RECOVERY_BACKOFF_MAX_SECONDS = 300.0

PARTS = ("vector", "graph", "concept_matcher", "consolidator")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: Any) -> Any:
    return moment.isoformat() if hasattr(moment, "isoformat") else moment


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


@dataclass
class StoreResult:
    """What ``store_memory`` wrote: the row id, the vector point, the concept
    links, whether the row was buffered, and the error of each part that
    failed (keys ``episodic``, ``vector``, ``graph``)."""

    postgres_id: Optional[int]
    point_id: Optional[str]
    concepts_linked: int
    buffered: bool
    errors: Dict[str, str] = field(default_factory=dict)


class EnhancedMemoryManager:
    """Core memory, episodic memory, recall and consolidation over seams."""

    def __init__(
        self,
        episodic_store: Optional[EpisodicStore] = None,
        vector_store: Optional[VectorStore] = None,
        graph: Optional[GraphStore] = None,
        concept_matcher: Optional[ConceptMatcher] = None,
        consolidator: Optional[EpisodeConsolidator] = None,
        persona: Optional[Mapping[str, Any]] = None,
        boilerplate_patterns: Optional[Sequence[str]] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._clock = clock or _utcnow
        self.episodic = (
            episodic_store
            if episodic_store is not None
            else InMemoryEpisodicStore(clock=self._clock)
        )
        self.vector_store = vector_store
        self.graph = graph
        self.concept_matcher = concept_matcher
        self.consolidator = consolidator
        self.persona = persona
        self._settings = settings
        patterns = (
            DEFAULT_BOILERPLATE_PATTERNS
            if boilerplate_patterns is None
            else tuple(boilerplate_patterns)
        )
        self._boilerplate = [re.compile(p, re.IGNORECASE) for p in patterns]

        self.core_memory: Dict[str, str] = {}
        self._core_source: Optional[str] = None
        self._core_stale = False
        self.initialized = False
        self.degraded = False
        self._retry_at: Optional[datetime] = None
        self._backoff = RECOVERY_BACKOFF_SECONDS
        self._errors: Dict[str, str] = {}
        self._warned_absent: Set[str] = set()
        self._written_patterns: Set[str] = set()
        self._counts: Dict[str, int] = {
            "stored": 0,
            "buffered": 0,
            "unindexed": 0,
            "episodic_failures": 0,
            "vector_failures": 0,
            "graph_failures": 0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the memory manager clock must return aware datetimes")
        return moment.astimezone(timezone.utc)

    def _record_error(self, part: str, exc: BaseException) -> str:
        message = _describe(exc)
        self._errors[part] = message
        return message

    def _warn_absent_parts(self) -> None:
        present = {
            "vector": self.vector_store is not None,
            "graph": self.graph is not None,
            "concept_matcher": self.concept_matcher is not None,
            "consolidator": self.consolidator is not None,
        }
        for part, available in present.items():
            if not available and part not in self._warned_absent:
                self._warned_absent.add(part)
                logger.warning("memory manager part %s is absent", part)

    def _is_boilerplate(self, content: str) -> bool:
        return any(pattern.search(content or "") for pattern in self._boilerplate)

    def _rendered_blocks(self) -> Dict[str, str]:
        persona = self.persona
        if persona is None:
            persona = load_persona(settings=self.settings)
        return render_core_blocks(persona, settings=self.settings)

    async def _ensure_initialized(self) -> None:
        if not self.initialized:
            await self.initialize()
        elif self.degraded and (
            self._retry_at is None or self._now() >= self._retry_at
        ):
            await self._recover()

    async def _recover(self) -> None:
        """Retry the degraded core-memory load; never raises for an outage."""
        report = await self.initialize()
        if report["status"] == "ready":
            logger.info("memory manager recovered: episodic store reachable again")

    @staticmethod
    def _shape(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "content": row["content"],
            "memory_type": row.get("memory_type"),
            "importance": float(row["importance"]),
            "emotional_context": row.get("emotional_context") or {},
            "created_at": _iso(row.get("created_at")),
        }

    # ------------------------------------------------------------------
    # Lifecycle and core memory
    # ------------------------------------------------------------------

    async def initialize(self) -> Dict[str, Any]:
        """Load (and seed missing) core blocks; report ``ready`` or ``degraded``.

        Never raises for an unavailable store: the report carries the error and
        core memory keeps serving the last loaded blocks, marked stale.
        """
        self._warn_absent_parts()
        seeded: Dict[str, bool] = {}
        try:
            await self.episodic.ensure_schema()
            blocks = await self.episodic.load_core_blocks()
            missing = [name for name in CORE_BLOCK_NAMES if name not in blocks]
            if missing:
                rendered = self._rendered_blocks()
                seeded = await self.episodic.seed_core_blocks(
                    {name: rendered[name] for name in CORE_BLOCK_NAMES},
                    overwrite=False,
                )
                blocks = await self.episodic.load_core_blocks()
            else:
                seeded = {name: False for name in CORE_BLOCK_NAMES}
        except StoreUnavailable as exc:
            message = self._record_error("episodic", exc)
            logger.warning("memory manager degraded: episodic store %s", message)
            self.degraded = True
            self._core_stale = True
            self._retry_at = self._now() + timedelta(seconds=self._backoff)
            self._backoff = min(self._backoff * 2.0, RECOVERY_BACKOFF_MAX_SECONDS)
            if not self.core_memory:
                self.core_memory = self._rendered_blocks()
                self._core_source = "rendered"
            self.initialized = True
            return {
                "status": "degraded",
                "core_blocks": sorted(self.core_memory),
                "seeded": {},
                "errors": dict(self._errors),
                "parts": self._parts(),
            }
        self.core_memory = dict(blocks)
        self._core_source = "store"
        self._mark_recovered()
        self.initialized = True
        logger.info("memory manager ready with %d core blocks", len(blocks))
        return {
            "status": "ready",
            "core_blocks": sorted(self.core_memory),
            "seeded": seeded,
            "errors": dict(self._errors),
            "parts": self._parts(),
        }

    def _mark_recovered(self) -> None:
        self._core_stale = False
        self.degraded = False
        self._retry_at = None
        self._backoff = RECOVERY_BACKOFF_SECONDS
        self._errors.pop("episodic", None)

    async def get_core_memory(self, block_name: str) -> Optional[str]:
        """Return a core block from the last load, or None."""
        await self._ensure_initialized()
        return self.core_memory.get(block_name)

    async def update_core_memory(self, block_name: str, content: str) -> None:
        """Write a core block; the cached block changes only after the write.

        Raises ``StoreUnavailable`` when the store cannot take the write.
        """
        await self._ensure_initialized()
        try:
            await self.episodic.set_core_block(block_name, content)
        except StoreUnavailable as exc:
            self._record_error("episodic", exc)
            raise
        self.core_memory[block_name] = content
        logger.info("core memory block %s updated", block_name)

    async def seed_core_memory(
        self, blocks: Optional[Mapping[str, str]] = None, overwrite: bool = False
    ) -> Dict[str, bool]:
        """Write core blocks (default: rendered from the persona or Settings).

        Without ``overwrite`` an existing block is kept. Returns which blocks
        were written and reloads the cache.
        """
        chosen = dict(blocks) if blocks is not None else self._rendered_blocks()
        written = await self.episodic.seed_core_blocks(chosen, overwrite=overwrite)
        self.core_memory = await self.episodic.load_core_blocks()
        self._core_source = "store"
        self._mark_recovered()
        self.initialized = True
        return written

    # ------------------------------------------------------------------
    # Storing
    # ------------------------------------------------------------------

    async def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 5.0,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> StoreResult:
        """Write the episodic row, then its vector point, then concept links."""
        await self._ensure_initialized()
        errors: Dict[str, str] = {}
        try:
            row_id = await self.episodic.add_episodic(
                content,
                memory_type=memory_type,
                importance=importance,
                emotional_context=emotional_context,
            )
        except StoreUnavailable as exc:
            errors["episodic"] = self._record_error("episodic", exc)
            self._counts["episodic_failures"] += 1
            logger.warning("episodic write failed: %s", errors["episodic"])
            return StoreResult(None, None, 0, False, errors)

        self._counts["stored"] += 1
        if self.degraded and isinstance(row_id, int):
            # The store took a write, so it is back: reload core memory now.
            await self._recover()
        if not isinstance(row_id, int):
            self._counts["buffered"] += 1
            if self.vector_store is not None:
                self._counts["unindexed"] += 1
            logger.warning(
                "episodic row buffered for replay; vector and graph indexing "
                "skipped for it"
            )
            return StoreResult(None, None, 0, True, errors)

        point_id: Optional[str] = None
        if self.vector_store is not None:
            wanted = point_id_for_memory(row_id)
            try:
                point_id = await asyncio.to_thread(
                    self.vector_store.store_memory,
                    text=content,
                    metadata={
                        "postgres_id": row_id,
                        "memory_type": memory_type,
                        "created_at": self._now().isoformat(),
                    },
                    importance=float(importance),
                    emotional_weight=float(importance) / 10.0,
                    point_id=wanted,
                )
            except Exception as exc:  # noqa: BLE001 - any vector failure is recorded
                errors["vector"] = self._record_error("vector", exc)
                self._counts["vector_failures"] += 1
                self._counts["unindexed"] += 1
                logger.warning("vector write failed: %s", errors["vector"])

        linked = 0
        if self.graph is not None and self.concept_matcher is not None:
            try:
                linked = await self._link_concepts(row_id, content)
            except Exception as exc:  # noqa: BLE001 - any graph failure is recorded
                errors["graph"] = self._record_error("graph", exc)
                self._counts["graph_failures"] += 1
                logger.warning("concept linking failed: %s", errors["graph"])
        return StoreResult(row_id, point_id, linked, False, errors)

    async def _link_concepts(self, row_id: int, content: str) -> int:
        known = await asyncio.to_thread(self.graph.list_concept_names)
        if not known:
            return 0
        matches = await asyncio.to_thread(
            self.concept_matcher.match_concepts,
            text=content,
            known_concepts=known,
            fuzzy=True,
        )
        chosen = [
            (name, float(score))
            for name, score in matches
            if float(score) >= CONCEPT_MIN_RELEVANCE
        ][:CONCEPT_MAX_LINKS]
        if not chosen:
            return 0
        relevance = sum(score for _, score in chosen) / len(chosen)
        return int(
            await asyncio.to_thread(
                self.graph.link_memory_to_concepts,
                memory_id=row_id,
                memory_content=content,
                concept_names=[name for name, _ in chosen],
                relevance=relevance,
                extraction_method=CONCEPT_LINK_METHOD,
            )
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Full-text search; rows carry the text relevance as ``similarity``.

        Raises ``StoreUnavailable`` when the store cannot answer.
        """
        await self._ensure_initialized()
        rows = await self.episodic.search_text(
            query, limit=limit, memory_type=memory_type, min_importance=min_importance
        )
        if rows:
            await self.episodic.touch([row["id"] for row in rows])
        results = []
        for row in rows:
            shaped = self._shape(row)
            shaped["similarity"] = float(row.get("relevance", 0.0))
            results.append(shaped)
        return results

    async def semantic_recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.5,
        include_concepts: bool = True,
    ) -> Dict[str, Any]:
        """Recall by meaning, falling back to text search, labelled either way.

        Returns ``memories``, ``concepts`` and ``search_stats`` (``method`` is
        ``vector``, ``text`` or ``none``, plus the error of any part that
        failed). Never raises for an unavailable store.
        """
        await self._ensure_initialized()
        limit = int(top_k) if top_k is not None else self.settings.VECTOR_SEARCH_LIMIT
        stats: Dict[str, Any] = {
            "query": query,
            "top_k": limit,
            "min_score": min_score,
            "method": "none",
            "orphan_hits": 0,
        }
        memories: List[Dict[str, Any]] = []

        hits: Optional[List[Any]] = None
        if self.vector_store is None:
            stats["vector_error"] = "vector store absent"
        else:
            try:
                hits = await asyncio.to_thread(
                    self.vector_store.search_similar,
                    query=query,
                    top_k=limit,
                    min_score=min_score,
                )
            except Exception as exc:  # noqa: BLE001 - labelled, then text search
                stats["vector_error"] = self._record_error("vector", exc)
                logger.warning("vector recall failed: %s", stats["vector_error"])

        if hits is not None:
            stats["vector_hits"] = len(hits)
            try:
                memories, stats["orphan_hits"] = await self._join_hits(hits)
                stats["method"] = "vector"
            except StoreUnavailable as exc:
                stats["join_error"] = self._record_error("episodic", exc)
        else:
            try:
                rows = await self.episodic.search_text(query, limit=limit)
                for row in rows:
                    shaped = self._shape(row)
                    shaped["similarity_score"] = None
                    shaped["relevance"] = float(row.get("relevance", 0.0))
                    shaped["match"] = "text"
                    memories.append(shaped)
                stats["method"] = "text"
            except StoreUnavailable as exc:
                stats["text_error"] = self._record_error("episodic", exc)

        row_ids = [m["id"] for m in memories if m.get("source") != "vector_payload"]
        if row_ids:
            try:
                await self.episodic.touch(row_ids)
            except StoreUnavailable as exc:
                stats["touch_error"] = self._record_error("episodic", exc)

        concepts: List[Dict[str, Any]] = []
        if include_concepts:
            if self.graph is None or self.concept_matcher is None:
                stats["concepts"] = "graph or concept matcher absent"
            else:
                try:
                    concepts = await self._related_concepts(query)
                except Exception as exc:  # noqa: BLE001 - labelled
                    stats["concept_error"] = self._record_error("graph", exc)
        stats["results"] = len(memories)
        return {"memories": memories, "concepts": concepts, "search_stats": stats}

    async def _join_hits(self, hits: Sequence[Any]) -> Any:
        ids: List[int] = []
        for _point, _score, payload in hits:
            row_id = (payload or {}).get("postgres_id")
            if isinstance(row_id, int):
                ids.append(row_id)
        rows = {row["id"]: row for row in await self.episodic.get_episodic(ids)}
        memories: List[Dict[str, Any]] = []
        orphans = 0
        for point_id, score, payload in hits:
            payload = payload or {}
            row_id = payload.get("postgres_id")
            if isinstance(row_id, int):
                row = rows.get(row_id)
                if row is None:
                    orphans += 1
                    continue
                shaped = self._shape(row)
                shaped["similarity_score"] = float(score)
                shaped["match"] = "vector"
                memories.append(shaped)
                continue
            text = payload.get("text")
            if not text:
                orphans += 1
                continue
            memories.append(
                {
                    "id": point_id,
                    "content": text,
                    "memory_type": payload.get("type") or payload.get("memory_type"),
                    "importance": payload.get("importance"),
                    "emotional_context": payload.get("emotional_context") or {},
                    "created_at": payload.get("created_at") or payload.get("stored_at"),
                    "similarity_score": float(score),
                    "match": "vector",
                    "source": "vector_payload",
                }
            )
        return memories, orphans

    async def _related_concepts(self, query: str) -> List[Dict[str, Any]]:
        known = await asyncio.to_thread(self.graph.list_concept_names)
        if not known:
            return []
        matches = await asyncio.to_thread(
            self.concept_matcher.match_concepts,
            text=query,
            known_concepts=known,
            fuzzy=True,
        )
        concepts = []
        for name, relevance in list(matches)[:RECALL_QUERY_CONCEPTS]:
            related = await asyncio.to_thread(self.graph.get_related_concepts, name)
            if related:
                concepts.append(
                    {"concept": name, "relevance": relevance, "related": related}
                )
        return concepts

    async def get_recent_important_memories(
        self, hours: float = 24, min_importance: float = 5.0, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Recent rows at or above ``min_importance``, newest first.

        Each carries ``recency_anchored`` and no similarity score, so a prompt
        can label it as recent and important rather than as a semantic match.
        Lifecycle boilerplate is skipped. Raises ``StoreUnavailable`` when the
        store cannot answer.
        """
        await self._ensure_initialized()
        fetch_limit = max(limit * 6, 20)
        rows = await self.episodic.recent(hours, fetch_limit, min_importance)
        memories: List[Dict[str, Any]] = []
        for row in rows:
            if self._is_boilerplate(row["content"]):
                continue
            shaped = self._shape(row)
            shaped["similarity_score"] = None
            shaped["recency_anchored"] = True
            memories.append(shaped)
            if len(memories) >= limit:
                break
        return memories

    async def get_recent_memories(
        self, hours: float = 24, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Recent rows, newest first, without lifecycle boilerplate.

        This is the consolidation pipeline's input. Raises ``StoreUnavailable``
        when the store cannot answer.
        """
        await self._ensure_initialized()
        rows = await self.episodic.recent(hours, limit)
        memories = []
        for row in rows:
            if self._is_boilerplate(row["content"]):
                continue
            shaped = self._shape(row)
            shaped["accessed_count"] = row.get("accessed_count", 0)
            memories.append(shaped)
        return memories

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def consolidate_memories(self, max_memories: int = 100) -> Dict[str, Any]:
        """Consolidate unconsolidated rows into semantic rows.

        ``method`` is ``htm`` with a consolidator and ``simple`` without one.
        Raises ``StoreUnavailable`` when the store cannot answer.
        """
        await self._ensure_initialized()
        method = "htm" if self.consolidator is not None else "simple"
        report: Dict[str, Any] = {
            "method": method,
            "memories_analyzed": 0,
            "episodic_processed": 0,
            "semantic_created": 0,
            "patterns_found": 0,
        }
        rows = await self.episodic.unconsolidated(max_memories)
        report["memories_analyzed"] = len(rows)
        if not rows:
            return report
        if self.consolidator is not None:
            await self._consolidate_htm(rows, report)
        else:
            await self._consolidate_simple(rows, report)
        logger.info(
            "memory consolidation (%s): %d rows, %d semantic rows",
            method,
            report["episodic_processed"],
            report["semantic_created"],
        )
        return report

    async def _consolidate_htm(
        self, rows: List[Dict[str, Any]], report: Dict[str, Any]
    ) -> None:
        consolidator = self.consolidator

        def replay() -> Any:
            before = {
                str(getattr(p, "pattern_id", id(p)))
                for p in consolidator.get_semantic_patterns(min_frequency=1)
            }
            episodes = [
                consolidator.add_episodic_memory(
                    memory_type="episodic",
                    emotional_state=_numeric(row.get("emotional_context")),
                    context=row["content"],
                )
                for row in rows
            ]
            htm_report = consolidator.consolidate_memories(
                memories=episodes, sleep_duration="short"
            )
            patterns = consolidator.get_semantic_patterns(min_frequency=1)
            return before, htm_report, patterns

        before, htm_report, patterns = await asyncio.to_thread(replay)
        row_ids = [row["id"] for row in rows]
        fresh = [
            p
            for p in patterns
            if str(getattr(p, "pattern_id", id(p))) not in before
            and str(getattr(p, "pattern_id", id(p))) not in self._written_patterns
        ]
        for pattern in fresh:
            concept = str(getattr(pattern, "pattern_type", "") or "").strip()
            if not concept:
                logger.debug("consolidator pattern without a type skipped")
                continue
            frequency = int(getattr(pattern, "frequency", 0) or 0)
            record: Dict[str, Any] = {
                "concept": concept,
                "knowledge": str(getattr(pattern, "description", "") or ""),
                "associations": {
                    "frequency": frequency,
                    "source_count": len(getattr(pattern, "source_memories", []) or []),
                },
                "strength": min(1.0, frequency / 10.0),
                "emotional_context": dict(
                    getattr(pattern, "emotional_signature", {}) or {}
                ),
                "source_memories": row_ids,
            }
            confidence = getattr(pattern, "confidence", None)
            if isinstance(confidence, (int, float)):
                record["confidence"] = float(confidence)
            await self.episodic.add_semantic(record)
            self._written_patterns.add(str(getattr(pattern, "pattern_id", id(pattern))))
            report["semantic_created"] += 1
        final_anomaly = (htm_report or {}).get("avg_final_anomaly")
        # Without a reported anomaly the strength is unmeasured: None, and the
        # stored strength is left as it was.
        strength: Optional[float] = (
            max(0.0, min(1.0, 1.0 - float(final_anomaly)))
            if isinstance(final_anomaly, (int, float))
            and not isinstance(final_anomaly, bool)
            else None
        )
        report["consolidation_strength"] = strength
        report["patterns_found"] = len(fresh)
        report["htm_report"] = htm_report
        report["episodic_processed"] = await self.episodic.mark_consolidated(
            row_ids, strength
        )

    async def _consolidate_simple(
        self, rows: List[Dict[str, Any]], report: Dict[str, Any]
    ) -> None:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            emotions = _numeric(row.get("emotional_context"))
            dominant = (
                max(emotions.items(), key=lambda item: item[1])[0]
                if emotions
                else UNLABELLED_GROUP
            )
            groups[dominant].append(row)
        for emotion, members in groups.items():
            if len(members) < SIMPLE_MIN_GROUP:
                continue
            theme = " ".join(
                m["content"][:SIMPLE_THEME_CHARS] for m in members[:SIMPLE_THEME_ROWS]
            )
            await self.episodic.add_semantic(
                {
                    "concept": f"{emotion}_pattern",
                    "associations": {"theme": theme},
                    "strength": min(1.0, len(members) / 10.0),
                    "emotional_context": (
                        {} if emotion == UNLABELLED_GROUP else {emotion: 1.0}
                    ),
                    "confidence": SIMPLE_CONFIDENCE,
                    "source_memories": [m["id"] for m in members],
                }
            )
            report["semantic_created"] += 1
            report["patterns_found"] += 1
        report["episodic_processed"] = await self.episodic.mark_consolidated(
            [row["id"] for row in rows], SIMPLE_STRENGTH
        )

    async def consolidate(self) -> Dict[str, int]:
        """The sleep scheduler's ``Consolidator`` seam: integer counts only."""
        report = await self.consolidate_memories()
        return {
            key: int(report.get(key, 0))
            for key in (
                "memories_analyzed",
                "episodic_processed",
                "semantic_created",
                "patterns_found",
            )
        }

    # ------------------------------------------------------------------
    # Stats, status and shutdown
    # ------------------------------------------------------------------

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Row counts and importance totals; raises ``StoreUnavailable``."""
        await self._ensure_initialized()
        counts = await self.episodic.stats()
        episodic = int(counts.get("episodic", 0))
        total = float(counts.get("total_importance", 0.0))
        return {
            "core_memory_blocks": len(self.core_memory),
            "episodic_memories": episodic,
            "archived_memories": int(counts.get("archived", 0)),
            "unconsolidated_memories": int(counts.get("unconsolidated", 0)),
            "semantic_memories": int(counts.get("semantic", 0)),
            "total_importance": total,
            "avg_importance": total / episodic if episodic else None,
            "unindexed_memories": self._counts["unindexed"],
        }

    def _part_status(self, part: str, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {
                "backend": None,
                "available": False,
                "last_error": None,
                "reason": "absent",
            }
        reader = getattr(obj, "status", None)
        if not callable(reader):
            return {
                "backend": type(obj).__name__,
                "available": True,
                "last_error": None,
            }
        try:
            return dict(reader())
        except Exception as exc:  # noqa: BLE001 - status never raises
            return {
                "backend": type(obj).__name__,
                "available": False,
                "last_error": _describe(exc),
            }

    def _parts(self) -> Dict[str, Dict[str, Any]]:
        return {
            "episodic": self._part_status("episodic", self.episodic),
            "vector": self._part_status("vector", self.vector_store),
            "graph": self._part_status("graph", self.graph),
            "concept_matcher": self._part_status(
                "concept_matcher", self.concept_matcher
            ),
            "consolidator": self._part_status("consolidator", self.consolidator),
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the details."""
        parts = self._parts()
        episodic = parts["episodic"]
        last_error = self._errors.get("episodic") or episodic.get("last_error")
        return {
            "backend": episodic.get("backend"),
            "available": bool(episodic.get("available")) and not self.degraded,
            "last_error": last_error,
            "initialized": self.initialized,
            "degraded": self.degraded,
            "core_memory": {
                "blocks": sorted(self.core_memory),
                "stale": self._core_stale,
                "source": self._core_source,
            },
            "parts": parts,
            "counts": dict(self._counts),
            "errors": dict(self._errors),
        }

    async def shutdown(self) -> None:
        """Close the stores; plain and awaitable ``close`` methods both work."""
        for part, obj in (
            ("episodic", self.episodic),
            ("vector", self.vector_store),
            ("graph", self.graph),
        ):
            closer = getattr(obj, "close", None)
            if not callable(closer):
                continue
            try:
                await _maybe_await(closer())
            except Exception as exc:  # noqa: BLE001 - shutdown keeps going
                self._record_error(part, exc)
                logger.warning("closing %s failed: %s", part, _describe(exc))
        self.initialized = False
        logger.info("memory manager shut down")


def _numeric(context: Any) -> Dict[str, float]:
    """The numeric entries of an emotional context mapping."""
    if not isinstance(context, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in context.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
