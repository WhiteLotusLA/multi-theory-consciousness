"""Experience tracker: time buckets, honest None metrics, window, store."""

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
from mtc.consciousness.learning import experience_tracker as et
from mtc.consciousness.learning.experience_tracker import (
    STATE_FIELD_MAP,
    Experience,
    ExperienceTracker,
    InteractionOutcome,
    classify_outcome,
    experience_metrics_from_state,
)
from mtc.memory.inmemory import InMemoryLearningStore

INSTANT = datetime(2000, 1, 1, 23, 0, 0, tzinfo=timezone.utc)


def make_tracker(zone="UTC", store=None, window=200, instant=INSTANT):
    store = store if store is not None else InMemoryLearningStore()
    tracker = ExperienceTracker(
        store,
        window=window,
        settings=SimpleNamespace(TIMEZONE=zone),
        clock=lambda: instant,
    )
    return tracker, store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zone,expected",
    [("UTC", "evening"), ("Europe/Berlin", "night"), ("Asia/Tokyo", "morning")],
)
async def test_time_of_day_uses_configured_timezone(zone, expected):
    tracker, _ = make_tracker(zone)
    experience = await tracker.record("Alice", "hi", "hello")
    assert experience.time_of_day == expected
    assert experience.timestamp == INSTANT.isoformat()


@pytest.mark.asyncio
async def test_missing_metrics_stored_as_none():
    tracker, store = make_tracker()
    experience = await tracker.record("Alice", "hi", "hello")
    assert experience.valence_shift is None and experience.outcome is None
    assert experience.dominant_theory is None
    for key in STATE_FIELD_MAP:
        assert getattr(experience, key) is None
    stored = (await store.recent_experiences(1))[0]
    assert stored["pad_pleasure"] is None and stored["outcome"] is None

    partial = await tracker.record(
        "Alice",
        "hi",
        "hello",
        emotion_before={"valence": 0.0, "label": "calm"},
        emotion_after={"valence": 0.4, "label": "calm"},
        metrics={"pad_pleasure": 0.3},
    )
    assert partial.pad_pleasure == 0.3 and partial.pad_arousal is None
    assert partial.arousal_shift is None
    assert partial.valence_shift == pytest.approx(0.4)
    assert partial.outcome == InteractionOutcome.REPAIR.value


def test_metric_keys_map_onto_consciousness_state_fields():
    names = {f.name for f in dataclasses.fields(ConsciousnessState)}
    for key, attribute in STATE_FIELD_MAP.items():
        assert attribute in names, (key, attribute)
    experience_fields = {f.name for f in dataclasses.fields(Experience)}
    assert set(STATE_FIELD_MAP) <= experience_fields

    state = SimpleNamespace(
        ignition_events=2,
        beautiful_loop_quality=0.6,
        architectural_phi=None,
        higher_order_thoughts=[object(), object(), object()],
        prediction_error=0.2,
        pad_pleasure=-0.1,
        pad_arousal=0.5,
        pad_dominance=None,
        broadcast_coverage=0.8,
        integration_level=0.7,
        epistemic_depth=3,
        binding_quality=0.4,
        protoself_stability=0.9,
        feeling_of_knowing=0.3,
        homeostatic_urgency=("curiosity", 0.45),
    )
    metrics = experience_metrics_from_state(state)
    assert set(metrics) == set(STATE_FIELD_MAP)
    assert metrics["architectural_phi"] is None and metrics["pad_dominance"] is None
    assert metrics["metacognitive_events"] == 3.0
    assert metrics["homeostatic_urgency"] == pytest.approx(0.45)
    assert metrics["gwt_ignition_count"] == 2.0
    # a state whose slots are all None maps to all None
    empty = SimpleNamespace(**{a: None for a in STATE_FIELD_MAP.values()})
    assert all(v is None for v in experience_metrics_from_state(empty).values())


def test_outcome_classes_documented_as_heuristics():
    assert "heuristic" in (et.__doc__ or "").lower()
    assert "heuristic" in (classify_outcome.__doc__ or "").lower()
    assert {o.value for o in InteractionOutcome} == {
        "positive",
        "neutral",
        "negative",
        "deepening",
        "playful",
        "repair",
    }
    assert classify_outcome(0.2, 600, 5, "calm") is InteractionOutcome.DEEPENING
    assert classify_outcome(0.1, 100, 1, "joy") is InteractionOutcome.PLAYFUL
    assert classify_outcome(-0.3, 100, 1, "calm") is InteractionOutcome.NEGATIVE
    assert classify_outcome(0.35, 400, 1, "calm") is InteractionOutcome.REPAIR
    assert classify_outcome(0.1, 400, 1, "calm") is InteractionOutcome.POSITIVE
    assert classify_outcome(0.0, 400, 1, "calm") is InteractionOutcome.NEUTRAL
    assert classify_outcome(None, 400, 1, "calm") is None


@pytest.mark.asyncio
async def test_rolling_window_bounded():
    tracker, store = make_tracker(window=3)
    for index in range(5):
        await tracker.record("Alice", "m" * index, "r", conversation_depth=index)
    recent = tracker.get_recent(10)
    assert len(recent) == 3
    assert [e.message_length for e in recent] == [2, 3, 4]
    assert tracker.get_stats()["total_ever"] == 5
    assert len(await store.recent_experiences(10)) == 5


@pytest.mark.asyncio
async def test_initialize_loads_recent_from_store():
    first, store = make_tracker(window=10)
    for index in range(4):
        await first.record("Alice", "m" * index, "r")
    second, _ = make_tracker(store=store, window=3)
    await second.initialize()
    recent = second.get_recent(10)
    assert [e.message_length for e in recent] == [1, 2, 3]
    assert isinstance(recent[0], Experience)


@pytest.mark.asyncio
async def test_record_writes_through_store():
    tracker, store = make_tracker()
    await tracker.record(
        "Alice",
        "hello",
        "a reply",
        topics_discussed=list("abcdefg"),
        metrics={"architectural_phi": 0.2},
    )
    docs = await store.recent_experiences(5)
    assert len(docs) == 1
    assert docs[0]["speaker"] == "Alice" and docs[0]["architectural_phi"] == 0.2
    assert docs[0]["topics_discussed"] == list("abcde")
    assert docs[0]["dominant_theory"] == "iit"


@pytest.mark.asyncio
async def test_store_failure_keeps_window_and_does_not_raise():
    class Broken(InMemoryLearningStore):
        async def add_experience(self, doc):
            raise RuntimeError("down")

    tracker, _ = make_tracker(store=Broken())
    await tracker.record("Alice", "hi", "hello")
    assert len(tracker.get_recent()) == 1


@pytest.mark.asyncio
async def test_naive_clock_is_rejected():
    tracker = ExperienceTracker(
        InMemoryLearningStore(),
        settings=SimpleNamespace(TIMEZONE="UTC"),
        clock=lambda: datetime(2000, 1, 1, 12, 0, 0),
    )
    with pytest.raises(ValueError):
        await tracker.record("Alice", "hi", "hello")
