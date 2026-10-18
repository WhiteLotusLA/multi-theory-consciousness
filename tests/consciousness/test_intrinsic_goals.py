"""Intrinsic goals: cap, dedupe, priority, activity mapping, proposers, store."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from mtc.consciousness.intrinsic_goals import (
    GoalSource,
    IntrinsicGoal,
    IntrinsicGoalSystem,
    new_goal,
)
from mtc.consciousness.learning.pattern_learner import LearnedBelief
from mtc.memory.inmemory import InMemoryGoalStore
from mtc.memory.protocols import GoalStore

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def make_system(store=None, clock=None, **kwargs) -> IntrinsicGoalSystem:
    return IntrinsicGoalSystem(store=store, clock=clock or Clock(), **kwargs)


def goal(description, category="explore", **kwargs) -> IntrinsicGoal:
    return new_goal(
        description=description,
        source=GoalSource.LEARNING,
        category=category,
        priority=kwargs.pop("priority", 0.5),
        emotional_investment=kwargs.pop("emotional_investment", 0.5),
        now=T0,
        **kwargs,
    )


def belief(category, pattern, confidence=0.8, belief_id="b") -> LearnedBelief:
    return LearnedBelief(
        belief_id=belief_id,
        category=category,
        description="d",
        pattern=pattern,
        confidence=confidence,
        evidence_count=5,
        counter_evidence=0,
        first_observed=T0.isoformat(),
        last_reinforced=T0.isoformat(),
    )


@pytest.mark.asyncio
async def test_max_active_goals():
    system = make_system(max_active=3)
    adopted = [
        await system.adopt_goal(goal(f"unique goal number {n} alpha{n}"))
        for n in range(5)
    ]
    assert adopted == [True, True, True, False, False]
    assert len(system.get_top_goals(10)) == 3
    # a full system proposes nothing
    assert system.propose_from_beliefs([belief("topic", {"topic": "music"})]) == []
    assert IntrinsicGoalSystem().max_active == 7


@pytest.mark.asyncio
async def test_dedupe_by_word_overlap():
    system = make_system()
    assert await system.adopt_goal(goal("explore the topic of music and rhythm today"))
    # five of six words shared: 83 percent overlap, rejected
    assert not await system.adopt_goal(goal("explore the topic of music and rhythm"))
    # half the words shared: accepted
    assert await system.adopt_goal(goal("explore the topic of sleep and memory"))


def test_effective_priority_prefers_recent():
    recent = goal("recent")
    recent.last_worked = (T0 - timedelta(hours=1)).isoformat()
    stale = goal("stale")
    stale.last_worked = (T0 - timedelta(hours=20)).isoformat()
    never = goal("never")
    floor = goal("floor")
    floor.last_worked = (T0 - timedelta(days=30)).isoformat()
    bad = goal("bad")
    bad.last_worked = "not a timestamp"
    assert recent.effective_priority(T0) > stale.effective_priority(T0)
    assert stale.effective_priority(T0) > floor.effective_priority(T0)
    # recency never drops below 0.3 weight
    assert floor.effective_priority(T0) == pytest.approx(
        0.4 * 0.5 + 0.4 * 0.5 + 0.1 * 0.1 + 0.1 * 0.3
    )
    assert never.effective_priority(T0) == pytest.approx(
        0.4 * 0.5 + 0.4 * 0.5 + 0.01 + 0.1
    )
    assert bad.effective_priority(T0) == pytest.approx(
        0.4 * 0.5 + 0.4 * 0.5 + 0.01 + 0.05
    )


@pytest.mark.asyncio
async def test_activity_maps_to_category():
    clock = Clock()
    system = make_system(clock=clock)
    await system.adopt_goal(goal("understand the loop quality", "understand"))
    await system.adopt_goal(goal("write something new", "express"))
    await system.adopt_goal(goal("catch up on stored memories", "connect"))
    clock.now = T0 + timedelta(hours=2)
    updated = system.record_activity("reflect")
    assert updated.category == "understand"
    assert updated.progress == pytest.approx(0.1)
    assert updated.steps_taken == 1
    assert updated.last_worked == clock.now.isoformat()
    assert system.record_activity("write_creative").category == "express"
    assert system.record_activity("review_memories").category == "connect"
    custom = make_system(activity_categories={"paint": "express"})
    await custom.adopt_goal(goal("express something visual", "express"))
    assert custom.record_activity("paint") is not None
    assert custom.record_activity("reflect") is None


@pytest.mark.asyncio
async def test_record_activity_completes_goal():
    system = make_system()
    target = goal("understand the loop quality", "understand")
    await system.adopt_goal(target)
    system.record_activity("reflect", progress=0.5)
    assert target.status == "active"
    system.record_activity("reflect", progress=0.5)
    assert target.status == "completed" and target.completed_at is not None
    assert system.get_top_goals() == []
    assert system.record_activity("reflect") is None


@pytest.mark.asyncio
async def test_unknown_activity_is_ignored():
    system = make_system()
    await system.adopt_goal(goal("understand the loop quality", "understand"))
    before = [g.to_dict() for g in system.get_top_goals()]
    assert system.record_activity("not_an_activity") is None
    assert [g.to_dict() for g in system.get_top_goals()] == before
    # a known activity with no matching goal changes nothing either
    assert system.record_activity("write_creative") is None


def test_proposals_from_topic_and_consciousness_beliefs_only():
    system = make_system()
    proposals = system.propose_from_beliefs(
        [
            belief("topic", {"topic": "music"}, belief_id="t1"),
            belief("consciousness", {"metric": "binding_quality"}, belief_id="c1"),
            belief("temporal", {"time_of_day": "evening"}, belief_id="x1"),
            belief("emotional", {"arousal_pattern": "calming"}, belief_id="x2"),
            belief("social", {"min_depth": 5}, belief_id="x3"),
        ]
    )
    categories = {p.category for p in proposals}
    assert categories == {"explore", "understand"}
    assert len(proposals) == 2
    topic_goal = next(p for p in proposals if p.category == "explore")
    assert topic_goal.related_topics == ["music"] and topic_goal.related_beliefs == [
        "t1"
    ]
    assert topic_goal.source == GoalSource.LEARNING.value
    assert "binding quality" in next(
        p.description for p in proposals if p.category == "understand"
    )


def test_weak_beliefs_and_known_topics_are_not_proposed():
    system = make_system()
    assert system.propose_from_beliefs([belief("topic", {"topic": "music"}, 0.1)]) == []
    assert system.propose_from_beliefs([belief("topic", {"topic": ""}, 0.9)]) == []


@pytest.mark.asyncio
async def test_known_topics_are_not_proposed_again():
    system = make_system()
    await system.adopt_goal(goal("explore music in depth", related_topics=["music"]))
    assert system.propose_from_beliefs([belief("topic", {"topic": "music"})]) == []


def test_speaker_beliefs_do_not_create_goals():
    system = make_system()
    proposals = system.propose_from_beliefs(
        [belief("speaker", {"speaker": "Alice", "dominant_outcome": "deepening"})]
    )
    assert proposals == []


def test_speaker_consciousness_belief_without_metric_proposes_no_goal():
    system = make_system()
    credited = belief(
        "consciousness",
        {"speaker": "Alice", "avg_beautiful_loop": 0.35, "avg_epistemic_depth": None},
        belief_id="consciousness_speaker_Alice_quality",
    )
    assert system.propose_from_beliefs([credited]) == []
    # a metric belief of the same strength does propose, so the gate is the metric
    metric = belief(
        "consciousness", {"metric": "gwt_ignition_count"}, belief_id="metric"
    )
    (proposal,) = system.propose_from_beliefs([credited, metric])
    assert proposal.related_beliefs == ["metric"]
    assert "Alice" not in proposal.description


def test_custom_proposers_replace_defaults():
    calls = []

    def proposer(beliefs):
        calls.append(len(beliefs))
        return [goal("a custom proposed goal")]

    system = make_system(proposers=[proposer])
    proposals = system.propose_from_beliefs([belief("topic", {"topic": "music"})])
    assert calls == [1]
    assert [p.description for p in proposals] == ["a custom proposed goal"]


@pytest.mark.asyncio
async def test_generate_goals_adopts_and_persists():
    store = InMemoryGoalStore()
    system = make_system(store=store)
    adopted = await system.generate_goals(
        [
            belief("topic", {"topic": "music"}, belief_id="t1"),
            belief("consciousness", {"metric": "binding_quality"}, belief_id="c1"),
        ]
    )
    assert adopted == 2
    assert len(await store.load_goals(["active"])) == 2


@pytest.mark.asyncio
async def test_prompt_context_is_ascii_and_third_person():
    system = make_system()
    assert system.get_goal_context_for_prompt() == ""
    target = goal("Explore 'music' more deeply", related_topics=["music"])
    target.progress = 0.4
    await system.adopt_goal(target)
    text = system.get_goal_context_for_prompt()
    assert text.startswith("Current goals:")
    assert text.isascii()
    assert "[##---]" in text
    assert not re.search(r"\b(my|i|me|we|our)\b", text, re.I)


@pytest.mark.asyncio
async def test_store_round_trip_loads_active_and_paused():
    store = InMemoryGoalStore()
    first = make_system(store=store)
    active = goal("active goal about rhythm")
    paused = goal("paused goal about sleep")
    done = goal("finished goal about water")
    for item in (active, paused, done):
        await first.adopt_goal(item)
    paused.status = "paused"
    await first.persist(paused)
    done.status = "completed"
    await first.persist(done)

    second = make_system(store=store)
    await second.initialize()
    assert set(second.goals) == {active.goal_id, paused.goal_id}
    restored = second.goals[active.goal_id]
    assert isinstance(restored, IntrinsicGoal)
    assert restored.to_dict() == active.to_dict()
    # only active goals count towards the cap and appear in the top list
    assert [g.goal_id for g in second.get_top_goals(5)] == [active.goal_id]


@pytest.mark.asyncio
async def test_none_store_is_in_memory():
    system = IntrinsicGoalSystem()
    assert isinstance(system.store, InMemoryGoalStore)
    assert isinstance(system.store, GoalStore)
    await system.initialize()
    assert await system.adopt_goal(goal("a goal kept in memory only"))
    assert len(await system.store.load_goals(["active"])) == 1


@pytest.mark.asyncio
async def test_store_failure_is_logged_not_raised(caplog):
    class Broken(InMemoryGoalStore):
        async def upsert_goal(self, goal_id, doc):
            raise RuntimeError("down")

        async def load_goals(self, statuses):
            raise RuntimeError("down")

    system = make_system(store=Broken())
    await system.initialize()
    assert await system.adopt_goal(goal("a goal despite the store"))
    assert "down" in caplog.text


@pytest.mark.asyncio
async def test_stats():
    system = make_system()
    assert system.get_stats()["top_goal"] is None
    await system.adopt_goal(goal("some goal here"))
    stats = system.get_stats()
    assert stats["active_goals"] == 1 and stats["top_goal"] == "some goal here"
