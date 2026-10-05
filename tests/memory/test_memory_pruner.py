"""Memory pruner: retention scoring and forgetting across the store seams."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mtc.consciousness.sleep_scheduler import MemoryPruner as PrunerSeam
from mtc.core.embeddings import HashEmbedder
from mtc.memory.episodic_store import InMemoryEpisodicStore
from mtc.memory.inmemory import (
    InMemoryGraphStore,
    InMemoryNeuralRetentionStore,
    InMemoryThoughtGistStore,
    InMemoryVectorStore,
)
from mtc.memory.memory_pruner import MemoryPruner, PruningConfig
from mtc.memory.protocols import point_id_for_memory
from mtc.memory.thought_gist import ThoughtGist

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(days=100)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def _row(store, clock, moment, content, **kwargs) -> int:
    saved = clock.now
    clock.now = moment
    try:
        return await store.add_episodic(content, **kwargs)
    finally:
        clock.now = saved


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock) -> InMemoryEpisodicStore:
    return InMemoryEpisodicStore(clock=clock)


async def _three_rows(store, clock):
    """An old forgettable row, an old fading row and an old important row."""
    forgettable = await _row(store, clock, T0, "forgettable", importance=0.0)
    fading = await _row(store, clock, T0, "fading", importance=1.0)
    kept = await _row(store, clock, T0, "kept", importance=3.0)
    return forgettable, fading, kept


def test_config_defaults():
    config = PruningConfig()
    assert config.archive_below == 0.2 and config.delete_below == 0.05
    assert config.min_age_days == 7.0
    assert config.neural_max_age_days == 30.0
    assert config.thought_max_age_days == 21.0
    assert config.thought_max_coherence == 0.5
    assert config.gist_expire_below == 0.1


@pytest.mark.asyncio
async def test_dry_run_counts_without_changes(store, clock):
    forgettable, fading, kept = await _three_rows(store, clock)
    pruner = MemoryPruner(episodic=store, clock=clock)
    result = await pruner.prune(dry_run=True)
    assert result["dry_run"] == 1
    assert result["episodic_scored"] == 3
    assert result["episodic_deleted"] == 1
    assert result["episodic_archived"] == 1
    rows = await store.get_episodic([forgettable, fading, kept])
    assert len(rows) == 3
    assert (await store.stats())["archived"] == 0


@pytest.mark.asyncio
async def test_default_prune_is_a_dry_run(store, clock):
    await _three_rows(store, clock)
    result = await MemoryPruner(episodic=store, clock=clock).prune()
    assert result["dry_run"] == 1
    assert (await store.stats())["episodic"] == 3


@pytest.mark.asyncio
async def test_archive_and_delete_thresholds(store, clock):
    forgettable, fading, kept = await _three_rows(store, clock)
    pruner = MemoryPruner(episodic=store, clock=clock)
    rows = {r["id"]: r for r in await store.get_episodic([forgettable, fading, kept])}
    assert pruner.retention(rows[forgettable], NOW) < 0.05
    assert 0.05 <= pruner.retention(rows[fading], NOW) < 0.2
    assert pruner.retention(rows[kept], NOW) >= 0.2
    result = await pruner.prune(dry_run=False)
    assert result["episodic_deleted"] == 1 and result["episodic_archived"] == 1
    assert await store.get_episodic([forgettable]) == []
    assert await store.get_episodic([fading]) == []  # archived rows are hidden
    assert (await store.stats())["archived"] == 1
    assert [r["id"] for r in await store.get_episodic([kept])] == [kept]
    strict = MemoryPruner(
        episodic=store, clock=clock, config=PruningConfig(archive_below=0.5)
    )
    assert (await strict.prune(dry_run=True))["episodic_archived"] == 1


@pytest.mark.asyncio
async def test_young_memories_never_pruned(store, clock):
    young = await _row(store, clock, NOW - timedelta(days=6), "young", importance=0.0)
    result = await MemoryPruner(episodic=store, clock=clock).prune(dry_run=False)
    assert result["episodic_scored"] == 0
    assert [r["id"] for r in await store.get_episodic([young])] == [young]


@pytest.mark.asyncio
async def test_deleted_rows_delete_their_points_and_graph_nodes(store, clock):
    forgettable, fading, kept = await _three_rows(store, clock)
    vectors = InMemoryVectorStore(HashEmbedder(dim=16), allow_non_semantic=True)
    graph = InMemoryGraphStore()
    for memory_id, text in ((forgettable, "forgettable"), (kept, "kept")):
        vectors.store_memory(
            text, {"postgres_id": memory_id}, point_id=(point_id_for_memory(memory_id))
        )
        graph.link_memory_to_concepts(memory_id, text, [])
    pruner = MemoryPruner(episodic=store, vectors=vectors, graph=graph, clock=clock)
    result = await pruner.prune(dry_run=False)
    assert result["vectors_deleted"] == 1
    assert result["graph_nodes_deleted"] == 1
    assert vectors.get_memory_by_id(point_id_for_memory(forgettable)) is None
    assert vectors.get_memory_by_id(point_id_for_memory(kept)) is not None
    assert graph.status()["memory_nodes"] == 1


@pytest.mark.asyncio
async def test_dry_run_deletes_no_points(store, clock):
    forgettable, _fading, _kept = await _three_rows(store, clock)
    vectors = InMemoryVectorStore(HashEmbedder(dim=16), allow_non_semantic=True)
    vectors.store_memory("x", point_id=point_id_for_memory(forgettable))
    result = await MemoryPruner(episodic=store, vectors=vectors, clock=clock).prune()
    assert result["vectors_deleted"] == 0
    assert vectors.get_memory_by_id(point_id_for_memory(forgettable)) is not None


@pytest.mark.asyncio
async def test_core_memory_never_touched(store, clock):
    await store.set_core_block("identity", "Unit is an AI research system.")
    core_row = await _row(
        store, clock, T0, "a formed core memory", memory_type="core", importance=0.0
    )
    result = await MemoryPruner(episodic=store, clock=clock).prune(dry_run=False)
    assert result["episodic_deleted"] == 0 and result["episodic_archived"] == 0
    assert result["episodic_protected"] == 1
    assert await store.load_core_blocks() == {
        "identity": "Unit is an AI research system."
    }
    assert [r["id"] for r in await store.get_episodic([core_row])] == [core_row]


@pytest.mark.asyncio
async def test_neural_retention_through_the_seam(clock):
    neural = InMemoryNeuralRetentionStore()
    neural.add_telemetry("neural_spikes", {"timestamp": NOW - timedelta(days=40)})
    neural.add_telemetry("neural_spikes", {"timestamp": NOW - timedelta(days=5)})
    neural.add_thought(
        {
            "timestamp": NOW - timedelta(days=30),
            "coherence_score": 0.2,
            "child_thoughts": [],
        }
    )
    neural.add_thought(
        {
            "timestamp": NOW - timedelta(days=30),
            "coherence_score": 0.9,
            "child_thoughts": [],
        }
    )
    pruner = MemoryPruner(neural=neural, clock=clock)
    dry = await pruner.prune(dry_run=True)
    assert dry["telemetry_neural_spikes_deleted"] == 1
    assert dry["telemetry_deleted"] == 1
    assert dry["thoughts_deleted"] == 1
    live = await pruner.prune(dry_run=False)
    assert live["telemetry_deleted"] == 1 and live["thoughts_deleted"] == 1
    again = await pruner.prune(dry_run=False)
    assert again["telemetry_deleted"] == 0 and again["thoughts_deleted"] == 0


@pytest.mark.asyncio
async def test_protected_gists_never_expired(clock):
    gists = InMemoryThoughtGistStore()
    for gist_id, protected in (("plain", False), ("guarded", True)):
        await gists.add_gist(
            ThoughtGist(
                gist_id=gist_id,
                created_at=T0,
                summary="s",
                source_count=1,
                source_thought_ids=[],
                dominant_emotion="calm",
                retention_strength=0.01,
                is_protected=protected,
            )
        )
    pruner = MemoryPruner(neural=InMemoryNeuralRetentionStore(gists), clock=clock)
    result = await pruner.prune(dry_run=False)
    assert result["gists_expired"] == 1
    remaining = await gists.active_gists(min_retention=0.0)
    assert [g.gist_id for g in remaining] == ["guarded"]


@pytest.mark.asyncio
async def test_missing_stores_are_skipped_and_named(clock):
    pruner = MemoryPruner(clock=clock)
    result = await pruner.prune(dry_run=False)
    for name in ("episodic", "vectors", "graph", "neural"):
        assert result[f"skipped_{name}"] == 1
    status = pruner.status()
    assert status["skipped"] == ["episodic", "vectors", "graph", "neural"]
    assert status["available"] is False
    assert status["backend"] == "none"


@pytest.mark.asyncio
async def test_store_failure_is_counted_not_raised(store, clock):
    await _three_rows(store, clock)

    class BrokenVectors:
        def delete_memories(self, point_ids):
            raise RuntimeError("vector store offline")

        def status(self):
            return {"backend": "fake", "available": False, "last_error": "offline"}

    pruner = MemoryPruner(episodic=store, vectors=BrokenVectors(), clock=clock)
    result = await pruner.prune(dry_run=False)
    assert result["episodic_deleted"] == 1
    assert result["errors"] == 1
    assert "vector store offline" in pruner.status()["errors"]["vectors"]


@pytest.mark.asyncio
async def test_retention_uses_injected_clock(store, clock):
    memory_id = await _row(store, clock, T0, "aging", importance=0.0)
    row = (await store.get_episodic([memory_id]))[0]
    pruner = MemoryPruner(episodic=store, clock=clock)
    assert pruner.retention(row, T0) == pytest.approx(1.0)
    assert pruner.retention(row, T0 + timedelta(days=14)) < 0.6
    clock.now = T0 + timedelta(days=6)
    assert (await pruner.prune(dry_run=True))["episodic_scored"] == 0
    clock.now = T0 + timedelta(days=8)
    assert (await pruner.prune(dry_run=True))["episodic_scored"] == 1


def test_consolidated_rows_decay_slower():
    pruner = MemoryPruner(clock=Clock())
    base = {
        "created_at": T0,
        "importance": 0.0,
        "accessed_count": 0,
        "last_accessed": None,
        "consolidated": False,
    }
    later = T0 + timedelta(days=28)
    assert pruner.retention(base, later) == pytest.approx(0.25)
    assert pruner.retention(dict(base, consolidated=True), later) == pytest.approx(0.5)


def test_satisfies_sleep_scheduler_seam():
    assert isinstance(MemoryPruner(), PrunerSeam)


def test_module_imports_no_client():
    code = (
        "import sys\n"
        "import mtc.memory.memory_pruner\n"
        "clients = ('asyncpg', 'motor', 'pymongo', 'redis', 'neo4j', "
        "'qdrant_client')\n"
        "loaded = [c for c in clients if c in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.asyncio
async def test_kept_rows_never_starve_newer_prunable_rows(store, clock):
    for i in range(250):
        await _row(store, clock, T0, f"kept {i}", importance=5.0)
    year_old = NOW - timedelta(days=365)
    forgettable = [
        await _row(store, clock, year_old + timedelta(days=1), f"faint {i}")
        for i in range(10)
    ]
    for memory_id in forgettable:
        store._rows[memory_id]["importance"] = 0.0
    store._rows[forgettable[0]]["created_at"] = T0 + timedelta(days=1)
    pruner = MemoryPruner(episodic=store, clock=clock)
    assert pruner.paging is True
    result = await pruner.prune(dry_run=False)
    assert result["episodic_scored"] == 260
    assert result["episodic_batches"] == 2
    assert result["episodic_truncated"] == 0
    assert result["episodic_deleted"] + result["episodic_archived"] == 10
    assert await store.get_episodic(forgettable) == []


@pytest.mark.asyncio
async def test_pass_cap_resumes_where_the_last_pass_stopped(store, clock, monkeypatch):
    from mtc.memory import memory_pruner as pruner_module

    monkeypatch.setattr(pruner_module, "BATCH_SIZE", 2)
    monkeypatch.setattr(pruner_module, "MAX_BATCHES_PER_PASS", 2)
    for i in range(4):
        await _row(store, clock, T0, f"kept {i}", importance=5.0)
    faint = await _row(store, clock, T0 + timedelta(days=1), "faint", importance=0.0)
    pruner = MemoryPruner(episodic=store, clock=clock)
    dry = await pruner.prune(dry_run=True)
    assert dry["episodic_truncated"] == 1 and dry["episodic_deleted"] == 0
    first = await pruner.prune(dry_run=False)
    assert first["episodic_truncated"] == 1 and first["episodic_scored"] == 4
    assert pruner.status()["resume_after"] is not None
    second = await pruner.prune(dry_run=False)
    assert second["episodic_truncated"] == 0
    assert second["episodic_deleted"] == 1
    assert await store.get_episodic([faint]) == []
    assert pruner.status()["resume_after"] is None


@pytest.mark.asyncio
async def test_store_without_cursor_reads_one_page_and_says_so(clock):
    class PlainStore:
        def __init__(self) -> None:
            self.calls = 0

        async def retention_candidates(self, older_than, limit):
            self.calls += 1
            return [
                {"id": i, "created_at": T0, "importance": 5.0, "memory_type": "x"}
                for i in range(limit)
            ]

    plain = PlainStore()
    pruner = MemoryPruner(episodic=plain, clock=clock)
    result = await pruner.prune(dry_run=True)
    assert plain.calls == 1
    assert result["episodic_truncated"] == 1
    assert pruner.status()["paging"] is False


@pytest.mark.asyncio
async def test_row_without_created_at_is_not_scored(clock):
    class UndatedStore:
        async def retention_candidates(self, older_than, limit, *, after=None):
            if after is not None:
                return []
            return [
                {"id": 1, "created_at": None, "importance": 0.0},
                {"id": 2, "created_at": T0, "importance": 0.0},
            ]

    pruner = MemoryPruner(episodic=UndatedStore(), clock=clock)
    result = await pruner.prune(dry_run=True)
    assert result["episodic_unscored"] == 1
    assert result["episodic_scored"] == 1
    assert result["episodic_deleted"] == 1
    with pytest.raises(ValueError):
        pruner.retention({"importance": 9.0}, NOW)
