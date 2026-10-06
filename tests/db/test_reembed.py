"""The re-embed backfill: dry runs write nothing, point ids come from the row
ids, archived rows are skipped, a non-semantic or unavailable embedder writes
nothing, a second run finds nothing missing and keyset pagination visits every
row once. Fakes stand in for PostgreSQL, Qdrant and the coordinator."""

import asyncio
import json
import types
from datetime import datetime, timezone

import numpy as np
import pytest

from mtc.core.config import Settings
from mtc.core.embeddings import EmbeddingUnavailableError, HashEmbedder
from mtc.db import reembed as reembed_module
from mtc.db.reembed import PAGE_SQL, ReembedRefused, reembed
from mtc.memory.protocols import point_id_for_memory

CREATED = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_settings(tmp_path, **values) -> Settings:
    return Settings(
        _env_file=None,
        PROJECT_ROOT=str(tmp_path),
        POSTGRES_ENABLED=True,
        QDRANT_ENABLED=True,
        **values,
    )


def row(row_id, memory_type="episodic", importance=5.0):
    return {
        "id": row_id,
        "content": f"memory number {row_id}",
        "memory_type": memory_type,
        "importance": importance,
        "created_at": CREATED,
    }


class FakeEpisodicPool:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: r["id"])
        self.pages = []
        self.closed = False

    async def fetch(self, sql, after, limit):
        assert sql == PAGE_SQL
        self.pages.append((after, limit))
        return [r for r in self.rows if r["id"] > after][:limit]

    async def close(self):
        self.closed = True


class FakeVectorClient:
    def __init__(self, size=384, exists=True):
        self.size = size
        self.exists = exists
        self.points = {}
        self.retrieved = []

    def get_collections(self):
        names = ["mtc_memories"] if self.exists else []
        return types.SimpleNamespace(
            collections=[types.SimpleNamespace(name=n) for n in names]
        )

    def get_collection(self, name):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                params=types.SimpleNamespace(
                    vectors=types.SimpleNamespace(size=self.size, distance="Cosine")
                ),
                hnsw_config=types.SimpleNamespace(),
            ),
            payload_schema={},
        )

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        assert with_payload is False and with_vectors is False
        self.retrieved.append(list(ids))
        return [types.SimpleNamespace(id=pid) for pid in ids if pid in self.points]

    def delete_collection(self, *args, **kwargs):
        raise AssertionError("the backfill never deletes a collection")


class FakeCoordinator:
    def __init__(self, client, embedder):
        self.client = client
        self.embedder = embedder
        self.collection_name = "mtc_memories"
        self.calls = 0
        self.closed = False

    def store_memories(self, texts, metadatas, importances, point_ids):
        vectors = np.asarray(self.embedder.encode(list(texts)))
        assert vectors.shape[0] == len(texts)
        self.calls += 1
        for text, meta, importance, pid in zip(
            texts, metadatas, importances, point_ids
        ):
            payload = dict(meta, text=text, importance=importance)
            self.client.points[pid] = payload
        return list(point_ids)

    def close(self):
        self.closed = True


class SemanticStandIn:
    """A semantic-marked embedder for tests (hash vectors underneath)."""

    semantic = True

    def __init__(self, fail=False):
        self.fail = fail
        self.dim = 384

    def encode(self, texts):
        if self.fail:
            raise EmbeddingUnavailableError("model could not be loaded")
        return HashEmbedder(384).encode(texts)


@pytest.fixture
def stores(monkeypatch):
    state = {"client": FakeVectorClient(), "pool": None, "built": []}

    def use_rows(rows):
        state["pool"] = FakeEpisodicPool(rows)
        return state["pool"]

    async def open_pool(active):
        return state["pool"]

    def build(active, embedder, allow_non_semantic):
        coordinator = FakeCoordinator(state["client"], embedder)
        state["built"].append(coordinator)
        return coordinator

    monkeypatch.setattr(reembed_module, "_open_pool", open_pool)
    monkeypatch.setattr(reembed_module, "_build_coordinator", build)
    state["rows"] = use_rows
    return state


def run(coro):
    return asyncio.run(coro)


def test_dry_run_counts_missing_without_writing(tmp_path, stores):
    stores["rows"]([row(i) for i in range(1, 6)])
    stores["client"].points[point_id_for_memory(2)] = {"postgres_id": 2}
    result = run(
        reembed(make_settings(tmp_path), dry_run=True, embedder=SemanticStandIn())
    )
    assert (result.scanned, result.missing, result.embedded) == (5, 4, 0)
    assert list(stores["client"].points) == [point_id_for_memory(2)]
    assert stores["built"][0].calls == 0
    assert result.errors == {}


def test_point_ids_from_postgres_ids(tmp_path, stores):
    stores["rows"]([row(7, importance=8.5), row(11)])
    result = run(reembed(make_settings(tmp_path), embedder=SemanticStandIn()))
    assert result.embedded == 2
    points = stores["client"].points
    assert set(points) == {point_id_for_memory(7), point_id_for_memory(11)}
    payload = points[point_id_for_memory(7)]
    assert payload["postgres_id"] == 7
    assert payload["memory_type"] == "episodic"
    assert payload["importance"] == 8.5
    assert payload["created_at"] == CREATED.isoformat()
    assert payload["text"] == "memory number 7"
    assert stores["built"][0].closed and stores["pool"].closed


def test_archived_rows_skipped(tmp_path, stores):
    stores["rows"]([row(1), row(2, memory_type="archived"), row(3)])
    result = run(reembed(make_settings(tmp_path), embedder=SemanticStandIn()))
    assert result.scanned == 3
    assert result.skipped_archived == 1
    assert result.embedded == 2
    assert point_id_for_memory(2) not in stores["client"].points
    # Archived rows are not even looked up in Qdrant.
    looked_up = {pid for batch in stores["client"].retrieved for pid in batch}
    assert point_id_for_memory(2) not in looked_up


def test_non_semantic_embedder_refused_without_opt_in(tmp_path, stores):
    stores["rows"]([row(1)])
    with pytest.raises(ReembedRefused, match="not semantic"):
        run(reembed(make_settings(tmp_path), embedder=HashEmbedder(384)))
    assert stores["built"] == []
    assert stores["pool"].pages == []
    allowed = run(
        reembed(
            make_settings(tmp_path),
            embedder=HashEmbedder(384),
            allow_non_semantic=True,
        )
    )
    assert allowed.embedded == 1


def test_unavailable_embedder_exits_nonzero_and_writes_nothing(
    tmp_path, stores, monkeypatch, capsys
):
    stores["rows"]([row(i) for i in range(1, 4)])
    failing = SemanticStandIn(fail=True)
    result = run(reembed(make_settings(tmp_path), embedder=failing))
    assert result.embedded == 0
    assert "embedding" in result.errors
    assert stores["client"].points == {}
    monkeypatch.setattr(reembed_module, "_default_embedder", lambda active: failing)
    monkeypatch.setattr(
        "mtc.db.bootstrap.load_settings", lambda env_file: make_settings(tmp_path)
    )
    assert reembed_module.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["embedded"] == 0 and payload["exit_code"] == 1
    assert stores["client"].points == {}


def test_second_run_finds_nothing_missing(tmp_path, stores):
    rows = [row(i) for i in (1, 2, 3, 4)]
    stores["rows"](rows)
    first = run(reembed(make_settings(tmp_path), embedder=SemanticStandIn()))
    assert first.missing == 4 and first.embedded == 4
    stores["rows"](rows)
    second = run(reembed(make_settings(tmp_path), embedder=SemanticStandIn()))
    assert second.scanned == 4
    assert second.missing == 0 and second.embedded == 0
    assert len(stores["client"].points) == 4
    # --all re-embeds every live row into the same points.
    stores["rows"](rows)
    again = run(
        reembed(make_settings(tmp_path), embedder=SemanticStandIn(), all_rows=True)
    )
    assert again.embedded == 4 and len(stores["client"].points) == 4


def test_keyset_pagination_visits_every_row_once(tmp_path, stores):
    ids = [1, 2, 5, 9, 10, 11, 20, 21]
    stores["rows"]([row(i) for i in ids])
    result = run(
        reembed(make_settings(tmp_path), embedder=SemanticStandIn(), batch_size=3)
    )
    assert result.scanned == len(ids)
    assert stores["pool"].pages == [(0, 3), (5, 3), (11, 3), (21, 3)]
    looked_up = [pid for batch in stores["client"].retrieved for pid in batch]
    assert looked_up == [point_id_for_memory(i) for i in ids]
    assert len(set(looked_up)) == len(ids)
    assert set(stores["client"].points) == {point_id_for_memory(i) for i in ids}
    stores["rows"]([row(i) for i in ids])
    limited = run(
        reembed(
            make_settings(tmp_path), embedder=SemanticStandIn(), all_rows=True, limit=2
        )
    )
    assert limited.missing == 2 and limited.embedded == 2


def test_collection_must_exist_at_the_configured_width(tmp_path, stores):
    stores["rows"]([row(1)])
    stores["client"].exists = False
    with pytest.raises(ReembedRefused, match="mtc db init"):
        run(reembed(make_settings(tmp_path), embedder=SemanticStandIn()))
    stores["client"].exists = True
    stores["client"].size = 8192
    with pytest.raises(ReembedRefused) as info:
        run(reembed(make_settings(tmp_path), embedder=SemanticStandIn()))
    assert "8192" in str(info.value) and "MTC_QDRANT_COLLECTION" in str(info.value)
    assert stores["pool"].pages == []
