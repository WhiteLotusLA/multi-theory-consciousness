"""Consolidation pipeline: clustering, protection, gists through the brain seam."""

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from mtc.brain.protocol import BrainReply, FakeBrain, NoAnswerError
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager
from mtc.memory.episodic_store import InMemoryEpisodicStore
from mtc.memory.inmemory import InMemoryThoughtGistStore, InMemoryVectorStore
from mtc.memory.memory_consolidation_pipeline import (
    ConsolidationConfig,
    ConsolidationResult,
    ConsolidationStrategy,
    MemoryConsolidationPipeline,
    MemoryPattern,
    ThoughtConsolidationReport,
)
from mtc.memory.thought_gist import ThoughtGist

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(days=5)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(PROJECT_ROOT=str(tmp_path), SYSTEM_NAME="Unit")


@pytest.fixture
def clock() -> Clock:
    return Clock()


def _pipeline(settings, clock, **kwargs) -> MemoryConsolidationPipeline:
    return MemoryConsolidationPipeline(settings=settings, clock=clock, **kwargs)


def _thought(i: int, content: str, minutes: float = 0.0, **extra: Any) -> Dict:
    doc = {
        "thought_id": f"t{i}",
        "content": content,
        "timestamp": T0 + timedelta(minutes=minutes),
        "emotional_context": {"curiosity": 0.5},
    }
    doc.update(extra)
    return doc


def _cluster(representative: str, **extra: Any) -> Dict:
    cluster = {
        "representative": representative,
        "count": 1,
        "thought_ids": ["t0"],
        "burst_windows": [],
        "timestamps": [T0],
        "emotional_contexts": [{"curiosity": 0.3}],
    }
    cluster.update(extra)
    return cluster


# ----------------------------------------------------------------------
# Thought clustering and scoring
# ----------------------------------------------------------------------


def test_clusters_by_text_similarity(settings, clock):
    pipeline = _pipeline(settings, clock)
    thoughts = [
        _thought(0, "I wonder about the stars"),
        _thought(1, "i wonder   about the STARS"),
        _thought(2, "The kettle needs water"),
    ]
    clusters = pipeline.cluster_thoughts(thoughts)
    assert len(clusters) == 2
    first = next(c for c in clusters if c["count"] == 2)
    assert first["thought_ids"] == ["t0", "t1"]
    assert first["representative"] == "I wonder about the stars"
    loose = pipeline.cluster_thoughts(thoughts, similarity_threshold=0.0)
    assert len(loose) == 1


def test_burst_detection(settings, clock):
    pipeline = _pipeline(settings, clock)
    thoughts = [
        _thought(i, "same thought", minutes=m)
        for i, m in enumerate([0, 10, 20, 120, 125])
    ]
    cluster = pipeline.cluster_thoughts(thoughts)[0]
    bursts = cluster["burst_windows"]
    assert [b["count"] for b in bursts] == [3, 2]
    assert bursts[0]["start"] == T0.isoformat()
    assert bursts[0]["end"] == (T0 + timedelta(minutes=20)).isoformat()
    assert bursts[1]["start"] == (T0 + timedelta(minutes=120)).isoformat()


def test_relationship_keywords_raise_importance_whole_words_only(settings, clock):
    pipeline = _pipeline(settings, clock)
    plain = pipeline.score_cluster_importance(_cluster("a thought about tea"), 10)
    friend = pipeline.score_cluster_importance(_cluster("a thought about a friend"), 10)
    friendship = pipeline.score_cluster_importance(
        _cluster("a thought about friendship"), 10
    )
    assert friend == pytest.approx(plain + 0.20)
    assert friendship == pytest.approx(plain)
    # The keywords come from RELATIONSHIP_KEYWORDS at call time.
    settings.RELATIONSHIP_KEYWORDS = "neighbour"
    assert pipeline.score_cluster_importance(
        _cluster("a thought about a friend"), 10
    ) == pytest.approx(plain)
    assert pipeline.score_cluster_importance(
        _cluster("the Neighbour waved"), 10
    ) == pytest.approx(plain + 0.20)


def test_conversation_linked_ids_raise_importance(settings, clock):
    pipeline = _pipeline(settings, clock)
    plain = pipeline.score_cluster_importance(_cluster("tea"), 10)
    linked = pipeline.score_cluster_importance(
        _cluster("tea", thought_ids=["conversation-7"]), 10
    )
    assert linked == pytest.approx(plain + 0.20)


def test_emotion_and_size_drive_importance(settings, clock):
    pipeline = _pipeline(settings, clock)
    calm = pipeline.score_cluster_importance(_cluster("tea"), 10)
    intense = pipeline.score_cluster_importance(
        _cluster("tea", emotional_contexts=[{"joy": 0.95}]), 10
    )
    assert intense > calm
    small = pipeline.score_cluster_importance(_cluster("tea"), 100)
    large = _cluster("tea", count=10, thought_ids=[f"t{i}" for i in range(10)])
    assert pipeline.score_cluster_importance(large, 100) > small
    assert 0.0 <= pipeline.score_cluster_importance(large, 100) <= 1.0


def test_protected_markers_protect(settings, clock):
    pipeline = _pipeline(settings, clock)
    assert pipeline.check_protection(_cluster("I am Unit and I think"))[0] is True
    assert pipeline.check_protection(_cluster("my identity feels clear"))[0] is True
    assert pipeline.check_protection(_cluster("I am unity of purpose"))[0] is False
    assert pipeline.check_protection(_cluster("the kettle boils")) == (False, "")
    settings.SYSTEM_NAME = "Probe"
    assert pipeline.check_protection(_cluster("I am Probe"))[0] is True
    assert pipeline.check_protection(_cluster("I am Unit"))[0] is False
    hot = _cluster("the kettle boils", emotional_contexts=[{"fear": 0.95}])
    protected, reason = pipeline.check_protection(hot)
    assert protected is True and "0.95" in reason


def test_no_participant_protected_by_default(settings, clock):
    pipeline = _pipeline(settings, clock)
    cluster = _cluster("Alice and Operator talked about tea", participants=["Alice"])
    assert pipeline.check_protection(cluster) == (False, "")
    configured = _pipeline(
        settings, clock, config=ConsolidationConfig(protected_participants=("Alice",))
    )
    protected, reason = configured.check_protection(cluster)
    assert protected is True and reason == "Protected participant"
    assert configured.check_protection(_cluster("Alicea talked"))[0] is False


# ----------------------------------------------------------------------
# Gists through the brain seam
# ----------------------------------------------------------------------


def _old_thoughts(count: int = 6, content: str = "I keep wondering about tides"):
    return [
        _thought(i, content, minutes=i, emotional_context={"curiosity": 0.6})
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_gist_from_brain_answer_stored_and_embedded(settings, clock):
    brain = FakeBrain(replies=["Unit returned to tides with steady curiosity."])
    gists = InMemoryThoughtGistStore()
    vectors = InMemoryVectorStore(HashEmbedder(dim=16), allow_non_semantic=True)
    pipeline = _pipeline(
        settings, clock, brain=brain, gist_store=gists, vector_store=vectors
    )
    report = await pipeline.run_thought_consolidation(_old_thoughts())
    assert isinstance(report, ThoughtConsolidationReport)
    assert report.thoughts_processed == 6
    assert report.gists_created == 1 and report.gists_stored == 1
    assert report.gists_embedded == 1
    assert report.thoughts_to_delete == 6
    assert report.delete_ids == [f"t{i}" for i in range(6)]
    stored = await gists.active_gists(min_retention=0.0)
    assert len(stored) == 1
    gist = stored[0]
    assert gist.summary == "Unit returned to tides with steady curiosity."
    assert gist.source_count == 6 and gist.dominant_emotion == "curiosity"
    assert "Dominant emotion: curiosity." in brain.calls[0]["prompt"]
    assert gist.created_at == NOW
    hits = vectors.search_similar(gist.summary, top_k=1, min_score=-1.0)
    assert hits[0][2]["type"] == "gist"
    assert hits[0][2]["gist_id"] == gist.gist_id
    assert hits[0][2]["source_count"] == 6


@pytest.mark.asyncio
async def test_gist_without_emotional_data_has_no_mood(settings, clock):
    brain = FakeBrain(replies=["Unit returned to tides."])
    gists = InMemoryThoughtGistStore()
    vectors = InMemoryVectorStore(HashEmbedder(dim=16), allow_non_semantic=True)
    pipeline = _pipeline(
        settings, clock, brain=brain, gist_store=gists, vector_store=vectors
    )
    thoughts = [
        _thought(i, "I keep wondering about tides", minutes=i, emotional_context={})
        for i in range(6)
    ]
    report = await pipeline.run_thought_consolidation(thoughts)
    assert report.gists_created == 1 and report.gists_stored == 1
    (gist,) = await gists.active_gists(min_retention=0.0)
    assert gist.dominant_emotion is None
    assert gist.emotional_arc == {}
    prompt = brain.calls[0]["prompt"]
    for word in ("Dominant emotion", "neutral", "None"):
        assert word not in prompt, word
    assert "No emotion was measured" in prompt
    hits = vectors.search_similar(gist.summary, top_k=1, min_score=-1.0)
    assert hits[0][2]["dominant_emotion"] is None
    # Non-numeric scores are no measurement either.
    arc = MemoryConsolidationPipeline._emotional_arc([{}, {"calm": "high"}, None])
    assert arc == (None, {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply", [NoAnswerError("timed out"), "", BrainReply(text="   ")]
)
async def test_no_answer_stores_no_gist(settings, clock, reply):
    brain = FakeBrain(replies=[reply])
    gists = InMemoryThoughtGistStore()
    pipeline = _pipeline(settings, clock, brain=brain, gist_store=gists)
    report = await pipeline.run_thought_consolidation(_old_thoughts())
    assert report.gists_created == 0 and report.gists_stored == 0
    assert report.no_answer == 1
    assert report.thoughts_to_delete == 0 and report.delete_ids == []
    assert await gists.active_gists(min_retention=0.0) == []
    gist = await _pipeline(settings, clock, brain=FakeBrain([reply])).generate_gist(
        pipeline.cluster_thoughts(_old_thoughts())[0]
    )
    assert gist is None


@pytest.mark.asyncio
async def test_brain_failure_is_no_answer_and_reported(settings, clock):
    brain = FakeBrain(replies=[RuntimeError("provider offline")])
    pipeline = _pipeline(settings, clock, brain=brain)
    report = await pipeline.run_thought_consolidation(_old_thoughts())
    assert report.gists_created == 0 and report.no_answer == 1
    assert "provider offline" in report.errors["brain"]
    assert report.delete_ids == []


def test_status_names_absent_parts(settings, clock):
    status = _pipeline(settings, clock).status()
    assert status["backend"] == "none"
    assert status["available"] is False and status["last_error"] is None
    assert status["absent"] == [
        "memory_manager",
        "brain",
        "gist_store",
        "vector_store",
    ]
    gists = InMemoryThoughtGistStore()
    present = _pipeline(settings, clock, brain=FakeBrain(), gist_store=gists)
    status = present.status()
    assert status["backend"] == "brain+gist_store"
    assert status["parts"]["gist_store"]["present"] is True
    assert status["absent"] == ["memory_manager", "vector_store"]


@pytest.mark.asyncio
async def test_status_reports_the_last_errors(settings, clock):
    brain = FakeBrain(replies=[RuntimeError("provider offline")])
    pipeline = _pipeline(settings, clock, brain=brain)
    await pipeline.run_thought_consolidation(_old_thoughts())
    status = pipeline.status()
    assert "provider offline" in status["last_error"]
    assert "provider offline" in status["errors"]["brain"]


@pytest.mark.asyncio
async def test_reasoning_stripped_from_summaries(settings, clock):
    brain = FakeBrain(
        replies=["<think>long private reasoning</think> Unit watched the tides."]
    )
    pipeline = _pipeline(settings, clock, brain=brain)
    cluster = pipeline.cluster_thoughts(_old_thoughts())[0]
    gist = await pipeline.generate_gist(cluster)
    assert isinstance(gist, ThoughtGist)
    assert gist.summary == "Unit watched the tides."


@pytest.mark.asyncio
async def test_no_brain_skips_llm_steps_and_says_so(settings, clock):
    gists = InMemoryThoughtGistStore()
    pipeline = _pipeline(settings, clock, gist_store=gists)
    report = await pipeline.run_thought_consolidation(_old_thoughts())
    assert report.gists_created == 0 and report.thoughts_to_delete == 0
    assert report.skipped["gist_generation"] == "no brain"
    assert await gists.active_gists(min_retention=0.0) == []
    pattern = MemoryPattern("learning", "Learning about tides", 0.7)
    synthesis = await pipeline.synthesize_knowledge([pattern])
    assert synthesis == {"skipped": "no brain"}
    assert await pipeline.generate_gist(_cluster("tides")) is None


@pytest.mark.asyncio
async def test_synthesis_through_the_brain(settings, clock):
    brain = FakeBrain(replies=["Unit links tides to the moon.", NoAnswerError("x")])
    pipeline = _pipeline(settings, clock, brain=brain)
    pattern = MemoryPattern("learning", "Learning about tides", 0.7)
    assert await pipeline.synthesize_knowledge([pattern]) == {
        "synthesis": "Unit links tides to the moon."
    }
    assert await pipeline.synthesize_knowledge([pattern]) == {"no_answer": True}
    assert await pipeline.synthesize_knowledge([]) == {"skipped": "no patterns"}


@pytest.mark.asyncio
async def test_recent_thoughts_skipped(settings, clock):
    brain = FakeBrain(replies=["unused"])
    pipeline = _pipeline(settings, clock, brain=brain)
    recent = [
        _thought(i, "recent thought", minutes=0, timestamp=NOW - timedelta(hours=1))
        for i in range(4)
    ]
    report = await pipeline.run_thought_consolidation(recent)
    assert report.thoughts_processed == 0 and report.gists_created == 0
    assert brain.calls == []


@pytest.mark.asyncio
async def test_protected_clusters_keep_their_thoughts(settings, clock):
    brain = FakeBrain(replies=["Unit thought about tides."])
    pipeline = _pipeline(settings, clock, brain=brain)
    thoughts = _old_thoughts(3) + [
        _thought(10 + i, "I am Unit and this matters", minutes=i) for i in range(2)
    ]
    report = await pipeline.run_thought_consolidation(thoughts)
    assert report.protected_count == 2
    assert report.gists_created == 1
    assert set(report.delete_ids) == {"t0", "t1", "t2"}


@pytest.mark.asyncio
async def test_prompts_are_neutral_third_person(settings, clock):
    brain = FakeBrain(replies=["Unit wondered.", "Unit connects ideas."])
    pipeline = _pipeline(settings, clock, brain=brain)
    thoughts = [
        _thought(i, "wondering about tides", minutes=i, speaker="Alice")
        for i in range(3)
    ]
    await pipeline.run_thought_consolidation(thoughts)
    await pipeline.synthesize_knowledge([MemoryPattern("social", "tides", 0.5)])
    assert len(brain.calls) == 2
    for call in brain.calls:
        text = call["prompt"] + " " + call["system_prompt"]
        assert "Unit" in text
        assert "Alice" not in text
        assert not re.search(r"\b(she|her|he|his|him)\b", text, re.IGNORECASE)
        assert "third person" in call["system_prompt"]
        assert text.isascii()
    settings.SYSTEM_NAME = "Probe"
    brain2 = FakeBrain(replies=["Probe wondered."])
    await _pipeline(settings, clock, brain=brain2).run_thought_consolidation(thoughts)
    assert "Probe" in brain2.calls[0]["prompt"]
    assert "Unit" not in brain2.calls[0]["prompt"] + brain2.calls[0]["system_prompt"]


# ----------------------------------------------------------------------
# Retention
# ----------------------------------------------------------------------


def test_forgetting_curve_half_lives(settings, clock):
    pipeline = _pipeline(settings, clock)
    memories = [
        {"content": "a", "timestamp": NOW - timedelta(days=7), "importance": 0.0},
        {"content": "b", "timestamp": NOW - timedelta(days=14), "importance": 0.0},
        {"content": "c", "timestamp": NOW, "importance": 0.0},
    ]
    decayed = pipeline.apply_forgetting_curve(memories)
    assert [m["retention_strength"] for m in decayed] == pytest.approx([0.5, 0.25, 1.0])
    floor = pipeline.apply_forgetting_curve(
        [{"timestamp": NOW - timedelta(days=70), "importance": 0.8}]
    )
    assert floor[0]["retention_strength"] == pytest.approx(0.68)
    gist = {"created_at": NOW - timedelta(days=14), "importance_score": 0.0}
    assert pipeline.compute_gist_retention(gist) == pytest.approx(0.5)
    gist["created_at"] = NOW - timedelta(days=28)
    assert pipeline.compute_gist_retention(gist) == pytest.approx(0.25)
    custom = _pipeline(
        settings, clock, config=ConsolidationConfig(gist_half_life_days=28.0)
    )
    assert custom.compute_gist_retention(gist) == pytest.approx(0.5)


def test_missing_inputs_get_no_invented_retention(settings, clock):
    pipeline = _pipeline(settings, clock)
    old = NOW - timedelta(days=70)
    no_importance = pipeline.apply_forgetting_curve([{"timestamp": old}])
    assert no_importance[0]["retention_strength"] == pytest.approx(0.5**10)
    undated = pipeline.apply_forgetting_curve([{"content": "x", "importance": 0.8}])
    assert undated[0]["retention_strength"] is None
    assert undated[0]["retention_unmeasured"] == "no timestamp"
    assert pipeline.compute_gist_retention({"importance_score": 0.9}) is None


def test_gist_retention_rules(settings, clock):
    pipeline = _pipeline(settings, clock)
    old = NOW - timedelta(days=14)
    base = {"created_at": old, "importance_score": 0.3, "retrieval_count": 0}
    accessed = dict(base, retrieval_count=5)
    assert pipeline.compute_gist_retention(accessed) > pipeline.compute_gist_retention(
        base
    )
    ancient = {"created_at": NOW - timedelta(days=365), "importance_score": 0.1}
    assert pipeline.compute_gist_retention(ancient) < 0.2
    assert pipeline.compute_gist_retention(dict(ancient, is_protected=True)) >= 0.2
    gist = ThoughtGist(
        gist_id="g",
        created_at=NOW,
        summary="s",
        source_count=1,
        source_thought_ids=[],
        dominant_emotion="calm",
    )
    assert pipeline.compute_gist_retention(gist) == pytest.approx(1.0)
    assert pipeline.classify_retention(0.8) == "active"
    assert pipeline.classify_retention(0.35) == "fading"
    assert pipeline.classify_retention(0.1) == "expired"


@pytest.mark.asyncio
async def test_refresh_gist_retention_updates_the_store(settings, clock):
    gists = InMemoryThoughtGistStore()
    await gists.add_gist(
        ThoughtGist(
            gist_id="old",
            created_at=NOW - timedelta(days=28),
            summary="s",
            source_count=1,
            source_thought_ids=[],
            dominant_emotion="calm",
        )
    )
    pipeline = _pipeline(settings, clock, gist_store=gists)
    assert await pipeline.refresh_gist_retention() == 1
    stored = (await gists.active_gists(min_retention=0.0))[0]
    assert stored.retention_strength == pytest.approx(0.25)


def test_conflict_resolution_prefers_recent_high_confidence(settings, clock):
    pipeline = _pipeline(settings, clock)
    conflicts = [
        {"content": "the door was red", "timestamp": T0, "confidence": 0.9},
        {
            "content": "the door was blue",
            "timestamp": T0 + timedelta(days=1),
            "confidence": 0.4,
        },
        {
            "content": "the door was green",
            "timestamp": T0 + timedelta(days=1),
            "confidence": 0.8,
        },
    ]
    result = pipeline.resolve_conflicts(conflicts)
    assert result["resolution"] == "the door was green"
    assert result["confidence"] == pytest.approx(0.6)
    assert result["conflict_history"][0] == "the door was green"
    assert result["conflict_history"][-1] == "the door was red"
    single = pipeline.resolve_conflicts([conflicts[0]])
    assert single["confidence"] == pytest.approx(1.0)
    assert pipeline.resolve_conflicts([])["resolution"] is None


# ----------------------------------------------------------------------
# Memory patterns and the consolidation cycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pattern_extraction(settings, clock):
    pipeline = _pipeline(settings, clock)
    memories = [
        {"content": "Learned red is a color", "tags": ["learning", "color"]},
        {"content": "Learned blue is a color", "tags": ["learning", "color"]},
        {"content": "Alice showed a red flower", "tags": ["social", "nature"]},
    ]
    patterns = await pipeline.extract_patterns(memories)
    types = {p.pattern_type for p in patterns}
    assert {"learning", "social", "conceptual", "nature"} <= types
    learning = next(p for p in patterns if p.pattern_type == "learning")
    assert learning.content.startswith("Learning about color")
    assert learning.strength > 0.7


@pytest.mark.asyncio
async def test_social_patterns_use_generic_tags_only(settings, clock):
    pipeline = _pipeline(settings, clock)
    patterns = await pipeline.extract_patterns(
        [{"content": "talked", "tags": ["alice"]}]
    )
    assert "social" not in {p.pattern_type for p in patterns}
    with_participants = await pipeline.extract_patterns(
        [{"content": "talked", "participants": ["Alice", "Guest"]}]
    )
    assert "social" in {p.pattern_type for p in with_participants}


@pytest.mark.asyncio
async def test_conceptual_patterns_follow_the_developmental_stage(settings, clock):
    memories = [{"content": "why does it rain", "tags": []}]
    pipeline = _pipeline(settings, clock)
    assert "conceptual" in {
        p.pattern_type for p in await pipeline.extract_patterns(memories)
    }
    settings.DEVELOPMENTAL_STAGE_ENABLED = True
    early = await pipeline.extract_patterns(memories, developmental_stage=1)
    assert "conceptual" not in {p.pattern_type for p in early}
    later = await pipeline.extract_patterns(memories, developmental_stage=3)
    assert "conceptual" in {p.pattern_type for p in later}


def test_importance_ranking(settings, clock):
    pipeline = _pipeline(settings, clock)
    ranked = pipeline.rank_by_importance(
        [
            {
                "content": "warm talk",
                "emotional_intensity": 0.9,
                "participants": ["Alice", "Guest"],
                "repetitions": 3,
            },
            {
                "content": "the wall is white",
                "emotional_intensity": 0.1,
                "participants": ["Guest"],
                "repetitions": 1,
            },
            {
                "content": "counted to ten",
                "emotional_intensity": 0.7,
                "tags": ["achievement", "learning"],
                "repetitions": 2,
            },
        ]
    )
    assert ranked[0]["content"] == "warm talk"
    assert ranked[-1]["content"] == "the wall is white"


def test_strategy_selection(settings, clock):
    pipeline = _pipeline(settings, clock)
    assert pipeline.select_strategy([]) is ConsolidationStrategy.SLEEP_CONSOLIDATION
    assert (
        pipeline.select_strategy([{"emotional_intensity": 0.9}])
        is ConsolidationStrategy.SLEEP_CONSOLIDATION
    )
    assert (
        pipeline.select_strategy([{"repetitions": 4}, {"repetitions": 3}])
        is ConsolidationStrategy.ACTIVE_REHEARSAL
    )
    spread = [{"timestamp": T0}, {"timestamp": T0 + timedelta(days=4)}]
    assert pipeline.select_strategy(spread) is ConsolidationStrategy.SPACED_REPETITION


def test_emotional_prioritization(settings, clock):
    pipeline = _pipeline(settings, clock)
    ordered = pipeline.prioritize_for_consolidation(
        [{"emotional_intensity": 0.2}, {"emotional_intensity": 0.9}, {}]
    )
    assert [m.get("emotional_intensity", 0.0) for m in ordered] == [0.9, 0.2, 0.0]


@pytest.mark.asyncio
async def test_connection_strengthening(settings, clock):
    class Manager:
        def __init__(self) -> None:
            self.connections: List[Dict] = []

        async def update_memory_connections(self, connections):
            self.connections.extend(connections)

    manager = Manager()
    pipeline = _pipeline(settings, clock, memory_manager=manager)
    connections = await pipeline.strengthen_connections(
        [
            {"memory_id": "a", "content": "tides and moon", "tags": ["sea"]},
            {"memory_id": "b", "content": "moon rising", "tags": ["sea"]},
            {"memory_id": "c", "content": "kettle", "tags": []},
        ]
    )
    assert len(connections) == 1
    assert connections[0]["from"] == "a" and connections[0]["to"] == "b"
    assert "moon" in connections[0]["reason"]
    assert manager.connections == connections
    unpersisted = await _pipeline(settings, clock).strengthen_connections(
        [{"content": "moon one"}, {"content": "moon two"}]
    )
    assert len(unpersisted) == 1


@pytest.mark.asyncio
async def test_consolidation_cycle_over_the_memory_manager(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = EnhancedMemoryManager(
        episodic_store=store, settings=settings, clock=clock
    )
    await manager.initialize()
    for text in ("why does the tide turn", "how does the moon pull", "a calm walk"):
        await manager.store_memory(text, emotional_context={"curiosity": 0.95})
    pipeline = _pipeline(settings, clock, memory_manager=manager)
    result = await pipeline.run_consolidation_cycle()
    assert isinstance(result, ConsolidationResult)
    assert result.memories_processed == 3
    assert result.patterns_found >= 1
    assert result.core_memories_formed >= 1
    assert result.next_consolidation == NOW + timedelta(hours=4)
    rows = await store.recent(hours=1, limit=50)
    core_rows = [r for r in rows if r["memory_type"] == "core"]
    assert len(core_rows) == result.core_memories_formed
    assert all(r["importance"] >= 9.1 for r in core_rows)
    # Formed core rows do not feed the next cycle, and none is formed twice.
    again = await pipeline.run_consolidation_cycle()
    assert again.memories_processed == 3
    assert again.core_memories_formed == 0


@pytest.mark.asyncio
async def test_cycle_without_memory_manager_says_so(settings, clock):
    result = await _pipeline(settings, clock).run_consolidation_cycle()
    assert result.memories_processed == 0
    assert result.skipped == "no memory manager"
