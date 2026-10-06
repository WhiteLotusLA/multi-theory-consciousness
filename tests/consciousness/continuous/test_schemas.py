"""Thought records: one document shape for memory and MongoDB."""

import ast
import inspect
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import pytest

from mtc.consciousness.continuous import schemas
from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtStreamState,
    ThoughtType,
    new_thought_id,
)
from mtc.neural.mongodb_schemas import NeuralDataManager, ThoughtStream
from tests.memory.test_mongo_stores import FakeMongoClient

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _full(**extra) -> ThoughtRecord:
    values = dict(
        thought_id="t-1",
        timestamp=T0,
        thought_type=ThoughtType.QUESTION,
        content="what is the quiet part of a day",
        intensity=ThoughtIntensity.STRONG,
        salience_score=0.7,
        triggered_by="a recent topic",
        associated_memories=["m1", "m2"],
        emotional_context={"calm": 0.4, "curious": 0.6},
        previous_thought_id="t-0",
        leads_to_thought_id="t-2",
        consciousness_state="drowsy",
        processing_context={"strategy": "question", "nested": {"a": [1, 2]}},
        should_express=True,
        expressed_at=T0 + timedelta(minutes=1),
        stream_id="burst-9",
        access_count=3,
        first_seen=T0,
        last_seen=T0 + timedelta(hours=1),
        importance_score=0.55,
        retention_strength=0.9,
        is_protected=True,
        protection_reason="kept on purpose",
        consolidated_into="gist-1",
    )
    values.update(extra)
    return ThoughtRecord(**values)


def test_record_doc_round_trip_every_field():
    record = _full()
    back = ThoughtRecord.from_doc(record.to_doc())
    assert back == record
    # Every dataclass field took part (no field is silently dropped).
    names = {f.name for f in fields(ThoughtRecord)}
    assert names == {
        "thought_id",
        "timestamp",
        "thought_type",
        "content",
        "intensity",
        "salience_score",
        "triggered_by",
        "associated_memories",
        "emotional_context",
        "previous_thought_id",
        "leads_to_thought_id",
        "consciousness_state",
        "processing_context",
        "should_express",
        "expressed_at",
        "stream_id",
        "access_count",
        "first_seen",
        "last_seen",
        "importance_score",
        "retention_strength",
        "is_protected",
        "protection_reason",
        "consolidated_into",
    }
    for f in fields(ThoughtRecord):
        assert getattr(back, f.name) == getattr(record, f.name), f.name


def test_defaults_round_trip_with_unmeasured_scores_none():
    record = ThoughtRecord(
        thought_id="t-2",
        timestamp=T0,
        thought_type="reflection",
        content="a plain thought",
        intensity="quiet",
        salience_score=0.2,
    )
    assert record.stream_id == "t-2"
    assert record.importance_score is None and record.retention_strength is None
    assert ThoughtRecord.from_doc(record.to_doc()) == record


def test_absent_consciousness_state_is_none_never_awake():
    record = ThoughtRecord(
        thought_id="t-3",
        timestamp=T0,
        thought_type="reflection",
        content="a thought with no recorded mode",
        intensity="quiet",
        salience_score=0.2,
    )
    assert record.consciousness_state is None
    assert record.to_doc()["metadata"]["consciousness_state"] is None
    assert ThoughtRecord.from_doc(record.to_doc()).consciousness_state is None
    # A stored document that lacks the field (a foreign or older writer) reads
    # None, not a waking state nobody recorded.
    doc = _full().to_doc()
    del doc["metadata"]["consciousness_state"]
    assert ThoughtRecord.from_doc(doc).consciousness_state is None
    doc["metadata"] = {"intensity": "strong"}
    assert ThoughtRecord.from_doc(doc).consciousness_state is None


def test_doc_shape_matches_mongo_thought_stream():
    doc = _full().to_doc()
    assert set(doc) == {f.name for f in fields(ThoughtStream)}
    assert doc["thought_text"] == "what is the quiet part of a day"
    assert doc["parent_thought_id"] == "t-0"
    assert doc["child_thoughts"] == ["t-2"]
    assert doc["triggers"] == ["a recent topic"]
    assert doc["emotional_state"] == {"calm": 0.4, "curious": 0.6}
    assert doc["coherence_score"] == 0.7
    assert doc["metadata"]["should_express"] is True
    assert doc["metadata"]["intensity"] == "strong"
    # The document constructs the Phase 5 dataclass unchanged.
    assert ThoughtStream(**doc).to_dict() == doc


def test_doc_written_by_data_manager_reads_back():
    client = FakeMongoClient()
    manager = NeuralDataManager(database_name="fake_neural", client=client)
    record = _full()
    manager.store_thought(ThoughtStream(**record.to_doc()))
    stored = manager.get_recent_thoughts()
    assert len(stored) == 1
    assert "id" in stored[0]  # the manager hands back "id" for "_id"
    assert ThoughtRecord.from_doc(stored[0]) == record


def test_from_doc_reads_naive_datetimes_as_utc():
    doc = _full().to_doc()
    doc["timestamp"] = doc["timestamp"].replace(tzinfo=None)
    doc["metadata"]["expressed_at"] = doc["metadata"]["expressed_at"].replace(
        tzinfo=None
    )
    back = ThoughtRecord.from_doc(doc)
    assert back.timestamp == T0
    assert back.expressed_at == T0 + timedelta(minutes=1)


def test_from_doc_refuses_what_it_cannot_read_honestly():
    good = _full().to_doc()
    for key, bad in (
        ("thought_type", "memory"),
        ("thought_text", None),
        ("coherence_score", None),
        ("timestamp", None),
    ):
        doc = dict(good)
        doc[key] = bad
        with pytest.raises(ValueError):
            ThoughtRecord.from_doc(doc)
    doc = dict(good)
    doc["metadata"] = {"importance": 1.0}
    with pytest.raises(ValueError, match="intensity"):
        ThoughtRecord.from_doc(doc)
    doc = dict(good)
    del doc["thought_id"]
    with pytest.raises(ValueError, match="thought_id"):
        ThoughtRecord.from_doc(doc)
    # A manager-style document carries "id" and stands in for a missing thought_id.
    doc["id"] = "abc123"
    assert ThoughtRecord.from_doc(doc).thought_id == "abc123"


def test_no_private_field():
    names = {f.name for f in fields(ThoughtRecord)} | {
        f.name for f in fields(ThoughtStreamState)
    }
    assert not any("private" in name or "mongo" in name for name in names)
    assert not any("private" in key for key in _full().to_doc())
    tree = ast.parse(inspect.getsource(schemas))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "private" not in node.value.lower().replace(
                "private runtime data", ""
            )


@pytest.mark.parametrize("zone", ["UTC", "Europe/Berlin", "Asia/Tokyo"])
def test_timestamps_aware_utc(zone):
    from zoneinfo import ZoneInfo

    local = T0.astimezone(ZoneInfo(zone))
    record = _full(timestamp=local, expressed_at=local, first_seen=local)
    assert record.timestamp == T0 and record.timestamp.utcoffset() == timedelta(0)
    assert record.expressed_at.utcoffset() == timedelta(0)
    assert record.first_seen.utcoffset() == timedelta(0)
    naive = datetime(2000, 1, 1, 12, 0, 0)
    for name in ("timestamp", "expressed_at", "first_seen", "last_seen"):
        with pytest.raises(ValueError, match=name):
            _full(**{name: naive})


def test_new_thought_id_unique():
    ids = {new_thought_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(isinstance(i, str) and i for i in ids)


def test_validation_and_enums():
    for bad in (-0.1, 1.2, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _full(salience_score=bad)
    with pytest.raises(ValueError):
        _full(thought_type="memory")
    with pytest.raises(ValueError):
        _full(thought_id="")
    assert [t.value for t in ThoughtType] == [
        "reflection",
        "question",
        "association",
        "emotion",
        "memory_replay",
        "creative",
        "planning",
        "self_aware",
        "dream",
        "spontaneous",
    ]
    assert ThoughtIntensity.WHISPER.level == pytest.approx(0.1)
    assert ThoughtIntensity.URGENT.level == pytest.approx(0.9)
    levels = [i.level for i in ThoughtIntensity]
    assert levels == sorted(levels)
    assert ThoughtIntensity.from_level(0.52) is ThoughtIntensity.MODERATE
    assert ThoughtIntensity.from_level(0.0) is ThoughtIntensity.WHISPER
    with pytest.raises(ValueError):
        ThoughtIntensity.from_level(float("nan"))


def test_to_doc_returns_copies_and_copy_is_independent():
    record = _full()
    doc = record.to_doc()
    doc["emotional_state"]["calm"] = 0.0
    doc["metadata"]["processing_context"]["nested"]["a"].append(3)
    doc["associated_memories"].append("m3")
    assert record.emotional_context["calm"] == 0.4
    assert record.processing_context["nested"]["a"] == [1, 2]
    assert record.associated_memories == ["m1", "m2"]
    clone = record.copy()
    clone.associated_memories.append("x")
    clone.processing_context["nested"]["a"].append(9)
    assert record.associated_memories == ["m1", "m2"]
    assert record.processing_context["nested"]["a"] == [1, 2]


def test_stream_state_to_dict():
    state = ThoughtStreamState(
        current_thought_id="t-1",
        last_thought_time=T0,
        dominant_themes=["a"],
        thoughts_per_type={"question": 2},
    )
    out = state.to_dict()
    assert out["last_thought_time"] == T0.isoformat()
    assert out["thoughts_per_type"] == {"question": 2}
    assert ThoughtStreamState().to_dict()["last_thought_time"] is None
    measured = ThoughtStreamState(thought_frequency=0.25, current_intensity=0.4)
    assert measured.to_dict()["thought_frequency"] == 0.25
    assert measured.to_dict()["current_intensity"] == 0.4


def test_stream_state_unmeasured_values_are_none():
    state = ThoughtStreamState()
    assert state.thought_frequency is None
    assert state.current_intensity is None
    out = state.to_dict()
    assert out["thought_frequency"] is None
    assert out["current_intensity"] is None
