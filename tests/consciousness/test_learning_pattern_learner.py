"""Pattern learner: belief confidence, decay, contradiction, pruning, persistence."""

from datetime import datetime, timedelta, timezone

import pytest

from mtc.consciousness.learning.experience_tracker import Experience
from mtc.consciousness.learning.pattern_learner import LearnedBelief, PatternLearner
from mtc.memory.inmemory import InMemoryLearningStore

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + timedelta(**kwargs)


def exp(speaker="Alice", outcome="positive", valence=0.2, **kwargs) -> Experience:
    base = dict(
        timestamp=T0.isoformat(),
        speaker=speaker,
        topic=None,
        message_length=10,
        response_length=20,
        valence_shift=valence,
        outcome=outcome,
        time_of_day="morning",
    )
    base.update(kwargs)
    return Experience(**base)


def make_learner(clock=None, **kwargs):
    store = InMemoryLearningStore()
    return PatternLearner(store, clock=clock or Clock(), **kwargs), store


@pytest.mark.asyncio
async def test_belief_confidence_grows_with_evidence():
    learner, _ = make_learner()
    batch = [exp() for _ in range(4)]
    seen = []
    for _ in range(4):
        await learner.learn_from_experiences(batch)
        seen.append(learner.beliefs["speaker_Alice_positive"].confidence)
    assert seen[0] > 0.1
    assert all(b > a for a, b in zip(seen, seen[1:]))
    assert seen[-1] < 1.0
    belief = learner.beliefs["speaker_Alice_positive"]
    assert belief.evidence_count == 4
    assert belief.category == "speaker"


def test_belief_decays_with_seven_day_half_life():
    clock = Clock()
    learner, _ = make_learner(clock=clock)
    belief = learner.make_belief("b", "topic", "desc", {"topic": "x"})
    belief.confidence = 0.8
    assert learner.strength(belief) == pytest.approx(0.8)
    clock.advance(days=7)
    assert learner.strength(belief) == pytest.approx(0.4)
    clock.advance(days=7)
    assert learner.strength(belief) == pytest.approx(0.2)
    custom, _ = make_learner(clock=clock, half_life_days=1.0)
    other = custom.make_belief("b", "topic", "desc", {})
    other.confidence = 0.8
    clock.advance(days=1)
    assert custom.strength(other) == pytest.approx(0.4)


def test_strength_never_exceeds_confidence_under_clock_skew():
    clock = Clock()
    learner, _ = make_learner(clock=clock)
    belief = learner.make_belief("b", "topic", "desc", {})
    belief.confidence = 0.5
    clock.advance(days=-3)
    assert learner.strength(belief) == pytest.approx(0.5)


def test_contradiction_lowers_confidence():
    clock = Clock()
    learner, _ = make_learner(clock=clock)
    belief = learner.make_belief("b", "topic", "desc", {})
    belief.confidence = 0.6
    clock.advance(hours=1)
    learner.contradict("b")
    assert belief.confidence == pytest.approx(0.6 - 0.1 * 0.6)
    assert belief.counter_evidence == 1
    assert belief.last_contradicted == clock.now.isoformat()
    learner.contradict("missing")


@pytest.mark.asyncio
async def test_min_evidence_three():
    learner, _ = make_learner()
    assert await learner.learn_from_experiences([exp(), exp()]) == []
    assert learner.beliefs == {}
    changed = await learner.learn_from_experiences([exp(), exp(), exp()])
    assert changed and "speaker_Alice_positive" in learner.beliefs
    # three experiences spread over two speakers give neither speaker evidence
    other, _ = make_learner()
    mixed = [exp("Alice"), exp("Guest"), exp("Alice")]
    await other.learn_from_experiences(mixed)
    assert not any(b.category == "speaker" for b in other.beliefs.values())


@pytest.mark.asyncio
async def test_none_metrics_are_skipped():
    learner, _ = make_learner()
    blank = [
        exp(valence=None, outcome=None, time_of_day="") for _ in range(6)
    ]  # no valence, no outcome, no bucket
    assert await learner.learn_from_experiences(blank) == []
    assert learner.beliefs == {}

    # none-valued entries do not count towards a group's evidence
    mixed = [exp(valence=None, outcome=None) for _ in range(2)] + [
        exp(outcome="positive", valence=0.3, time_of_day="") for _ in range(2)
    ]
    await learner.learn_from_experiences(mixed)
    assert not any(b.category == "speaker" for b in learner.beliefs.values())

    enough = mixed + [exp(outcome="positive", valence=0.3, time_of_day="")]
    await learner.learn_from_experiences(enough)
    belief = learner.beliefs["speaker_Alice_positive"]
    assert belief.pattern["ratio"] == pytest.approx(1.0)

    # consciousness metrics that are None never count as rich evidence
    rich_none = [exp(gwt_ignition_count=None, beautiful_loop_quality=None)] * 5
    other, _ = make_learner()
    await other.learn_from_experiences(rich_none)
    assert not any(b.category == "consciousness" for b in other.beliefs.values())


@pytest.mark.asyncio
async def test_six_families_each_produce_beliefs():
    learner, _ = make_learner()
    deep = [
        exp(
            speaker="Alice",
            outcome="deepening",
            valence=0.3,
            topic="music",
            time_of_day="evening",
            arousal_shift=0.5,
            conversation_depth=7,
            consciousness_score=0.6,
            gwt_ignition_count=3,
            beautiful_loop_quality=0.6,
            epistemic_depth=3,
            binding_quality=0.5 + 0.01 * i,
            protoself_stability=0.8,
        )
        for i in range(6)
    ]
    await learner.learn_from_experiences(deep)
    categories = {b.category for b in learner.beliefs.values()}
    assert categories == {
        "speaker",
        "topic",
        "temporal",
        "emotional",
        "social",
        "consciousness",
    }
    assert "topic_music_deepens" in learner.beliefs
    assert "depth_long_conversations" in learner.beliefs


@pytest.mark.asyncio
async def test_prune_respects_cap():
    learner, _ = make_learner(max_beliefs=3)
    for index in range(6):
        belief = learner.make_belief(f"b{index}", "topic", "d", {"topic": str(index)})
        belief.confidence = 0.1 * (index + 1)
        belief.evidence_count = 5
    learner.prune()
    assert len(learner.beliefs) == 3
    assert set(learner.beliefs) == {"b3", "b4", "b5"}


def test_prune_drops_decayed_weak_beliefs():
    clock = Clock()
    learner, _ = make_learner(clock=clock)
    weak = learner.make_belief("weak", "topic", "d", {})
    weak.confidence = 0.05
    solid = learner.make_belief("solid", "topic", "d", {})
    solid.confidence = 0.05
    solid.evidence_count = 5
    clock.advance(days=60)
    learner.prune()
    assert set(learner.beliefs) == {"solid"}


@pytest.mark.asyncio
async def test_beliefs_persist_through_store_round_trip():
    clock = Clock()
    learner, store = make_learner(clock=clock)
    batch = [exp(topic="music", outcome="deepening", valence=0.3) for _ in range(4)]
    changed = await learner.learn_from_experiences(batch)
    assert changed
    docs = await store.load_beliefs(100)
    assert {d["belief_id"] for d in docs} == set(learner.beliefs)

    fresh = PatternLearner(store, clock=clock)
    await fresh.initialize()
    assert set(fresh.beliefs) == set(learner.beliefs)
    original = learner.beliefs["speaker_Alice_deepening"]
    restored = fresh.beliefs["speaker_Alice_deepening"]
    assert isinstance(restored, LearnedBelief)
    assert restored.confidence == pytest.approx(original.confidence)
    assert restored.pattern == original.pattern


@pytest.mark.asyncio
async def test_store_failure_is_logged_not_raised(caplog):
    class Broken(InMemoryLearningStore):
        async def upsert_belief(self, belief_id, doc):
            raise RuntimeError("down")

    learner = PatternLearner(Broken(), clock=Clock())
    changed = await learner.learn_from_experiences([exp() for _ in range(4)])
    assert changed
    assert "Failed to persist belief" in caplog.text


@pytest.mark.asyncio
async def test_rate_argument_scales_reinforcement():
    slow, _ = make_learner()
    fast, _ = make_learner()
    batch = [exp() for _ in range(4)]
    await slow.learn_from_experiences(batch, rate=0.01)
    await fast.learn_from_experiences(batch, rate=0.2)
    assert (
        fast.beliefs["speaker_Alice_positive"].confidence
        > slow.beliefs["speaker_Alice_positive"].confidence
    )


@pytest.mark.asyncio
async def test_context_lookup_and_stats():
    learner, _ = make_learner()
    batch = [exp(topic="music") for _ in range(4)]
    for belief_args in range(2):
        await learner.learn_from_experiences(batch, rate=0.5)
    alice = learner.get_beliefs_for_context(speaker="Alice")
    other = learner.get_beliefs_for_context(speaker="Guest")
    assert alice and all(b.pattern.get("speaker") == "Alice" for b in alice)
    assert not any(b.pattern.get("speaker") == "Alice" for b in other)
    stats = learner.get_stats()
    assert stats["total_beliefs"] == len(learner.beliefs)
    assert "speaker" in stats["categories"]
    assert learner.get_active_beliefs(min_strength=0.99) == []


def test_belief_descriptions_are_ascii_and_neutral():
    learner, _ = make_learner()
    belief = learner.make_belief("b", "speaker", "Conversations with Alice", {})
    assert belief.description.isascii()


@pytest.mark.asyncio
async def test_consciousness_beliefs_never_name_a_speaker():
    learner, store = make_learner()
    # identical system-side metrics for two speakers: nothing tells them apart
    batch = [
        exp(
            speaker=speaker,
            outcome="neutral",
            valence=0.0,
            gwt_ignition_count=1.0,
            beautiful_loop_quality=0.35,
            epistemic_depth=None,
        )
        for speaker in ("Alice", "Guest")
        for _ in range(3)
    ]
    await learner.learn_from_experiences(batch, rate=0.1)
    rich = [
        exp(
            speaker=speaker,
            outcome="deepening",
            gwt_ignition_count=3,
            beautiful_loop_quality=0.6,
            epistemic_depth=3,
            protoself_stability=0.8,
        )
        for speaker in ("Alice", "Guest")
        for _ in range(3)
    ]
    await learner.learn_from_experiences(rich, rate=0.1)
    consciousness = [
        b for b in learner.beliefs.values() if b.category == "consciousness"
    ]
    assert consciousness  # the global metric beliefs still form
    for belief in consciousness:
        assert "speaker" not in belief.pattern, belief.belief_id
        assert "metric" in belief.pattern, belief.belief_id
        assert "Alice" not in belief.description
        assert "Guest" not in belief.description
        assert not belief.belief_id.startswith("consciousness_speaker_")
    stored = {doc["belief_id"] for doc in await store.load_beliefs(100)}
    assert not any(bid.startswith("consciousness_speaker_") for bid in stored)


@pytest.mark.asyncio
async def test_initialize_drops_stored_speaker_consciousness_beliefs():
    clock = Clock()
    store = InMemoryLearningStore()
    writer = PatternLearner(store, clock=clock)
    credited = writer.make_belief(
        "consciousness_speaker_Alice_quality",
        "consciousness",
        "Conversations with Alice produce high consciousness quality",
        {"speaker": "Alice", "avg_beautiful_loop": 0.35},
        {"curiosity": 0.1, "openness": 0.05},
    )
    kept = writer.make_belief(
        "consciousness_ignition_quality",
        "consciousness",
        "When workspace ignition is high",
        {"metric": "gwt_ignition_count", "threshold": 2},
    )
    for belief in (credited, kept):
        await store.upsert_belief(belief.belief_id, belief.to_dict())
    fresh = PatternLearner(store, clock=clock)
    await fresh.initialize()
    assert set(fresh.beliefs) == {"consciousness_ignition_quality"}
    assert fresh.get_beliefs_for_context(speaker="Alice") == []
