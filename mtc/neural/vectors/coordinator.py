"""
Vector Coordinator
==================

The Qdrant implementation of the ``VectorStore`` protocol
(``mtc.memory.protocols``): semantic memory search over native embeddings of
``VECTOR_DIMENSIONS`` (384) dimensions. Vectors come from the
``mtc.core.embeddings`` seam (default ``SentenceTransformerEmbedder``, the
``EMBEDDING_MODEL`` key) and are stored as the embedder returns them: no
projection to another width, no quantization and no optimizer overrides.

Nothing is invented. There is no mock path: when the embedder cannot embed,
the call raises (``EmbeddingUnavailableError``) and nothing is written. A
non-semantic embedder (the hash embedder) is refused at construction unless
``allow_non_semantic=True`` is passed, which only tests and demonstrations do.
An embedder whose width differs from the collection raises
``VectorDimensionMismatch``. A server that cannot be reached raises
``StoreUnavailable``; reads raise rather than answer from a stand-in.

Config keys, read when the coordinator is built (or at call time for the
search limit): ``QDRANT_HOST`` and ``QDRANT_PORT`` (only when no client is
injected), ``QDRANT_COLLECTION``, ``VECTOR_DIMENSIONS``,
``VECTOR_SIMILARITY_METRIC`` (``cosine``, ``dot`` or ``euclid``) and
``VECTOR_SEARCH_LIMIT`` (the default ``top_k``). The ``qdrant_client`` package
(the ``qdrant`` extra) is imported inside the constructor and the calls that
need it, never at import.

Schema artifact: ``COLLECTION_PARAMS`` (the vector size and distance at the
default settings, HNSW ``m`` 32, ``ef_construct`` 400, ``full_scan_threshold``
20000, in memory) and ``PAYLOAD_INDEXES`` (``postgres_id`` integer,
``memory_type`` keyword, ``importance`` float). ``collection_params()`` fills
the size and distance from Settings; ``ensure_collection()`` applies both as a
safety net, and the store setup (``mtc db init``) is canonical. A collection that
exists with another width (one created at 8192 dimensions, for example) is
refused with ``VectorDimensionMismatch``; it is rebuilt with the re-embed
script, never reused.

Point ids are UUID strings (random unless the caller derives one with
``point_id_for_memory``). Payloads hold the caller's metadata plus ``text``,
``importance`` (1.0 when not given), ``emotional_weight`` (null when no
emotion was measured) and ``stored_at`` (aware UTC). Payload
text is private memory content at runtime and lives only in an
operator-controlled store. ``get_metrics()`` reports measured latencies (None
before anything was measured) and makes no claim about targets.
"""

import copy
import logging
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from mtc.core.config import Settings, get_settings
from mtc.core.embeddings import (
    Embedder,
    EmbeddingUnavailableError,
    SentenceTransformerEmbedder,
)
from mtc.memory.clients import require_client
from mtc.memory.protocols import StoreUnavailable

logger = logging.getLogger(__name__)

T = TypeVar("T")

COLLECTION_PARAMS: Dict[str, Any] = {
    "size": 384,
    "distance": "cosine",
    "hnsw_config": {
        "m": 32,
        "ef_construct": 400,
        "full_scan_threshold": 20000,
        "on_disk": False,
    },
}

PAYLOAD_INDEXES: Tuple[Tuple[str, str], ...] = (
    ("postgres_id", "integer"),
    ("memory_type", "keyword"),
    ("importance", "float"),
)

DISTANCES = ("cosine", "dot", "euclid")
SEARCH_HNSW_EF = 128
_LOCAL_INDEX_WARNING = "Payload indexes have no effect in the local Qdrant"

# Client exception class names that mean the server was not reached. Matched by
# name so this module never imports the client or its transport.
_UNREACHABLE_NAMES = frozenset(
    {
        "ResponseHandlingException",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "RemoteProtocolError",
        "RpcError",
        "AioRpcError",
        "_InactiveRpcError",
    }
)


class VectorDimensionMismatch(RuntimeError):
    """An embedder or an existing collection has another vector width."""


def _models() -> Any:
    return require_client("qdrant_client.models", "qdrant")


def _unreachable(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    return bool({cls.__name__ for cls in type(exc).__mro__} & _UNREACHABLE_NAMES)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _valid_point_id(point_id: Any) -> bool:
    if isinstance(point_id, bool):
        return False
    if isinstance(point_id, int):
        return point_id >= 0
    try:
        uuid.UUID(str(point_id))
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def collection_params(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Return ``COLLECTION_PARAMS`` with the size and distance from Settings."""
    active = settings if settings is not None else get_settings()
    metric = str(active.VECTOR_SIMILARITY_METRIC).strip().lower()
    if metric not in DISTANCES:
        raise ValueError(
            f"VECTOR_SIMILARITY_METRIC {active.VECTOR_SIMILARITY_METRIC!r} is not "
            f"one of {DISTANCES}"
        )
    params = copy.deepcopy(COLLECTION_PARAMS)
    params["size"] = int(active.VECTOR_DIMENSIONS)
    params["distance"] = metric
    return params


def _distance(models: Any, metric: str) -> Any:
    return {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }[metric]


def _schema_type(models: Any, name: str) -> Any:
    return {
        "integer": models.PayloadSchemaType.INTEGER,
        "keyword": models.PayloadSchemaType.KEYWORD,
        "float": models.PayloadSchemaType.FLOAT,
    }[name]


def _collection_size(info: Any) -> Optional[int]:
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    return int(size) if size is not None else None


class VectorCoordinator:
    """Qdrant-backed semantic memory at the embedder's native width."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        embedder: Optional[Embedder] = None,
        client: Any = None,
        allow_non_semantic: bool = False,
        create_collection: bool = True,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings
        active = self._active()
        self.collection_name = collection_name or active.QDRANT_COLLECTION
        self.vector_size = int(vector_size or active.VECTOR_DIMENSIONS)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(expected_dim=self.vector_size)
        if not getattr(embedder, "semantic", False) and not allow_non_semantic:
            raise EmbeddingUnavailableError(
                "the embedder is not marked semantic; vector search compares "
                "meanings, so pass a trained embedder or allow_non_semantic=True "
                "for tests and demonstrations"
            )
        width = getattr(embedder, "dim", None)
        if width is not None and int(width) != self.vector_size:
            raise VectorDimensionMismatch(
                f"the embedder produces {int(width)} dimensions but the collection "
                f"uses {self.vector_size} (VECTOR_DIMENSIONS); change one of them"
            )
        self.embedder = embedder
        self.allow_non_semantic = bool(allow_non_semantic)
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()
        self.metrics: Dict[str, Any] = {
            "total_embeddings": 0,
            "failed_embeddings": 0,
            "total_searches": 0,
            "embedding_time_s": 0.0,
            "search_time_s": 0.0,
            "last_embedding_ms": None,
            "last_search_ms": None,
        }
        if client is None:
            qdrant = require_client("qdrant_client", "qdrant")
            self.host = host if host is not None else active.QDRANT_HOST
            self.port = int(port) if port is not None else int(active.QDRANT_PORT)
            client = qdrant.QdrantClient(host=self.host, port=self.port)
            self._owns_client = True
        else:
            self.host, self.port = host, port
            self._owns_client = False
        self.client = client
        if create_collection:
            self.ensure_collection()
        logger.info(
            "vector coordinator on collection %s at %d dimensions",
            self.collection_name,
            self.vector_size,
        )

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _active(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _call(self, operation: str, fn: Callable[[], T]) -> T:
        try:
            result = fn()
        except Exception as exc:
            self._last_error = f"{operation}: {_describe(exc)}"
            if _unreachable(exc):
                logger.warning("Qdrant unavailable during %s: %s", operation, exc)
                raise StoreUnavailable(
                    f"Qdrant is unavailable ({operation}): {exc}"
                ) from exc
            raise
        self._last_error = None
        return result

    def _record(self, kind: str, seconds: float, count: int = 1) -> None:
        with self._lock:
            if kind == "embedding":
                self.metrics["total_embeddings"] += count
                self.metrics["embedding_time_s"] += seconds
                self.metrics["last_embedding_ms"] = seconds * 1000.0
            else:
                self.metrics["total_searches"] += count
                self.metrics["search_time_s"] += seconds
                self.metrics["last_search_ms"] = seconds * 1000.0

    def _fail_embedding(self, exc: BaseException, count: int) -> None:
        self._last_error = f"embedding: {_describe(exc)}"
        with self._lock:
            self.metrics["failed_embeddings"] += count

    def ensure_collection(self) -> bool:
        """Create the collection and payload indexes when missing.

        Returns True when the collection was created. An existing collection
        of another width raises ``VectorDimensionMismatch``.
        """
        models = _models()
        params = collection_params(self._active())
        name = self.collection_name

        def apply() -> bool:
            existing = {c.name for c in self.client.get_collections().collections}
            created = name not in existing
            if created:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=_distance(models, params["distance"]),
                    ),
                    hnsw_config=models.HnswConfigDiff(**params["hnsw_config"]),
                )
                logger.info("created vector collection %s", name)
            info = self.client.get_collection(name)
            size = _collection_size(info)
            if size != self.vector_size:
                raise VectorDimensionMismatch(
                    f"collection {name!r} holds {size}-dimension vectors but "
                    f"VECTOR_DIMENSIONS is {self.vector_size}; rebuild it with the "
                    "re-embed script (an older 8192-dimension collection cannot "
                    "be reused)"
                )
            indexed = set((getattr(info, "payload_schema", None) or {}).keys())
            for field_name, kind in PAYLOAD_INDEXES:
                if field_name in indexed:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=_LOCAL_INDEX_WARNING)
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field_name,
                        field_schema=_schema_type(models, kind),
                    )
            return created

        return self._call("ensure_collection", apply)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts into an ``(n, vector_size)`` float32 array of unit rows."""
        started = time.perf_counter()
        try:
            vectors = np.asarray(self.embedder.encode(list(texts)), dtype=np.float32)
            if vectors.shape != (len(texts), self.vector_size):
                raise VectorDimensionMismatch(
                    f"the embedder returned shape {vectors.shape}; expected "
                    f"({len(texts)}, {self.vector_size})"
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.all(np.isfinite(vectors)) or np.any(norms == 0.0):
                raise EmbeddingUnavailableError(
                    "the embedder returned a zero or non-finite vector"
                )
        except Exception as exc:
            self._fail_embedding(exc, len(texts))
            raise
        self._last_error = None
        self._record("embedding", time.perf_counter() - started, len(texts))
        return vectors / norms

    def embed_memory(self, text: str) -> np.ndarray:
        """Return the unit-norm ``vector_size`` embedding of one text."""
        return self._encode([text])[0]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    @staticmethod
    def _per_item(values: Optional[Sequence[Any]], count: int, name: str) -> List[Any]:
        if values is None:
            return [None] * count
        items = list(values)
        if len(items) != count:
            raise ValueError(f"{name} has {len(items)} items for {count} texts")
        return items

    def _upsert_vectors(
        self,
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[Optional[Dict[str, Any]]],
        importances: Sequence[Optional[float]],
        emotional_weights: Sequence[Optional[float]],
        point_ids: Sequence[Optional[str]],
    ) -> List[str]:
        models = _models()
        stored_at = datetime.now(timezone.utc).isoformat()
        ids: List[str] = []
        points = []
        for index, text in enumerate(texts):
            wanted = point_ids[index]
            if wanted is not None and not _valid_point_id(wanted):
                raise ValueError(f"point id {wanted!r} is not a UUID or unsigned int")
            point_id = str(wanted) if wanted is not None else str(uuid.uuid4())
            payload = copy.deepcopy(metadatas[index]) if metadatas[index] else {}
            importance = importances[index]
            weight = emotional_weights[index]
            payload.update(
                {
                    "text": text,
                    "importance": float(1.0 if importance is None else importance),
                    "emotional_weight": None if weight is None else float(weight),
                    "stored_at": stored_at,
                }
            )
            ids.append(point_id)
            points.append(
                models.PointStruct(
                    id=point_id, vector=vectors[index].tolist(), payload=payload
                )
            )
        self._call(
            "upsert",
            lambda: self.client.upsert(
                collection_name=self.collection_name, points=points
            ),
        )
        return ids

    def store_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        emotional_weight: Optional[float] = None,
        point_id: Optional[str] = None,
    ) -> str:
        """Embed and store one text; return its point id (upserting ``point_id``).

        ``emotional_weight`` is a measured emotion intensity, stored as null
        when the caller passed None (nothing was measured).
        """
        if point_id is not None and not _valid_point_id(point_id):
            raise ValueError(f"point id {point_id!r} is not a UUID or unsigned int")
        vectors = self._encode([text])
        return self._upsert_vectors(
            [text], vectors, [metadata], [importance], [emotional_weight], [point_id]
        )[0]

    def store_memories(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        importances: Optional[Sequence[float]] = None,
        emotional_weights: Optional[Sequence[Optional[float]]] = None,
        point_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> List[str]:
        """Embed texts in one call and store them in one upsert; return the ids.

        A missing importance reads 1.0; a missing emotional weight is stored
        as null.
        """
        items = list(texts)
        count = len(items)
        if count == 0:
            return []
        metas = self._per_item(metadatas, count, "metadatas")
        weights = self._per_item(importances, count, "importances")
        emotional = self._per_item(emotional_weights, count, "emotional_weights")
        ids = self._per_item(point_ids, count, "point_ids")
        for wanted in ids:
            if wanted is not None and not _valid_point_id(wanted):
                raise ValueError(f"point id {wanted!r} is not a UUID or unsigned int")
        vectors = self._encode(items)
        return self._upsert_vectors(items, vectors, metas, weights, emotional, ids)

    def update_memory_importance(self, point_id: str, new_importance: float) -> None:
        """Set the ``importance`` payload field of one point."""
        self._call(
            "set_payload",
            lambda: self.client.set_payload(
                collection_name=self.collection_name,
                payload={"importance": float(new_importance)},
                points=[point_id],
            ),
        )

    def delete_memories(self, point_ids: Sequence[str]) -> int:
        """Delete the points that exist; return how many were removed."""
        ids = list(dict.fromkeys(p for p in point_ids if _valid_point_id(p)))
        if not ids:
            return 0
        models = _models()
        found = self._call(
            "retrieve",
            lambda: self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=False,
                with_vectors=False,
            ),
        )
        existing = [record.id for record in found]
        if existing:
            self._call(
                "delete",
                lambda: self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=existing),
                ),
            )
        return len(existing)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def _filter(models: Any, filters: Optional[Dict[str, Any]]) -> Any:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, float):
                match = models.FieldCondition(
                    key=key, range=models.Range(gte=value, lte=value)
                )
            else:
                match = models.FieldCondition(
                    key=key, match=models.MatchValue(value=value)
                )
            conditions.append(match)
        return models.Filter(must=conditions)

    def _query(
        self,
        vector: np.ndarray,
        limit: int,
        min_score: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        models = _models()
        started = time.perf_counter()
        response = self._call(
            "query_points",
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=np.asarray(vector, dtype=np.float32).tolist(),
                limit=int(limit),
                score_threshold=float(min_score),
                with_payload=True,
                query_filter=self._filter(models, filters),
                search_params=models.SearchParams(hnsw_ef=SEARCH_HNSW_EF, exact=False),
            ),
        )
        self._record("search", time.perf_counter() - started)
        return [
            (str(point.id), float(point.score), dict(point.payload or {}))
            for point in response.points
        ]

    def search_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return ``(point_id, score, payload)`` best first, scores >= ``min_score``.

        ``top_k`` defaults to ``VECTOR_SEARCH_LIMIT`` read now; ``filters``
        keeps payloads whose fields equal the given values.
        """
        limit = top_k if top_k is not None else self._active().VECTOR_SEARCH_LIMIT
        if limit <= 0:
            return []
        return self._query(self.embed_memory(query), limit, min_score, filters)

    def get_memory_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Return a point's payload, or None when there is no such point."""
        if not _valid_point_id(point_id):
            return None
        records = self._call(
            "retrieve",
            lambda: self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            ),
        )
        return dict(records[0].payload or {}) if records else None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return the collection's point count, status and vector config."""
        info = self._call(
            "get_collection", lambda: self.client.get_collection(self.collection_name)
        )
        vectors = info.config.params.vectors
        return {
            "points_count": info.points_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            "segments_count": getattr(info, "segments_count", None),
            "status": str(getattr(info, "status", "")),
            "config": {
                "vector_size": getattr(vectors, "size", None),
                "distance": str(getattr(vectors, "distance", "")),
            },
        }

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return measured counts and latencies (None until measured)."""
        with self._lock:
            metrics = dict(self.metrics)
        embeddings, searches = metrics["total_embeddings"], metrics["total_searches"]
        return {
            "total_embeddings": embeddings,
            "failed_embeddings": metrics["failed_embeddings"],
            "total_searches": searches,
            "avg_embedding_time_ms": (
                metrics["embedding_time_s"] * 1000.0 / embeddings
                if embeddings
                else None
            ),
            "avg_search_time_ms": (
                metrics["search_time_s"] * 1000.0 / searches if searches else None
            ),
            "last_embedding_ms": metrics["last_embedding_ms"],
            "last_search_ms": metrics["last_search_ms"],
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the details."""
        report: Dict[str, Any] = {
            "backend": "qdrant",
            "available": self._last_error is None,
            "last_error": self._last_error,
            "collection": self.collection_name,
            "vector_size": self.vector_size,
            "semantic": bool(getattr(self.embedder, "semantic", False)),
        }
        reader = getattr(self.embedder, "status", None)
        if callable(reader):
            try:
                report["embedder"] = dict(reader())
            except Exception as exc:  # noqa: BLE001 - status never raises
                report["embedder"] = {"available": False, "last_error": str(exc)}
        return report

    def close(self) -> None:
        """Close a client this coordinator built; an injected one stays open."""
        if self._owns_client:
            closer = getattr(self.client, "close", None)
            if callable(closer):
                closer()

    def __repr__(self) -> str:
        return (
            f"VectorCoordinator(collection={self.collection_name!r}, "
            f"vectors={self.vector_size}D)"
        )
