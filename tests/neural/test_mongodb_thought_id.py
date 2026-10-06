"""Thought streams carry their own thought id, indexed, and child links use it."""

from dataclasses import fields
from datetime import datetime, timezone

import pytest

from mtc.neural.mongodb_schemas import NEURAL_INDEXES, NeuralDataManager, ThoughtStream
from tests.memory.test_mongo_stores import FakeMongoClient

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def manager() -> NeuralDataManager:
    return NeuralDataManager(database_name="fake_neural", client=FakeMongoClient())


def _thought(thought_id, stream_id="burst", **extra) -> ThoughtStream:
    values = dict(
        timestamp=T0,
        stream_id=stream_id,
        thought_text=f"text of {thought_id}",
        thought_type="reflection",
        thought_id=thought_id,
    )
    values.update(extra)
    return ThoughtStream(**values)


def test_thought_stream_has_thought_id():
    assert "thought_id" in {f.name for f in fields(ThoughtStream)}
    assert (
        ThoughtStream(
            timestamp=T0, stream_id="s", thought_text="x", thought_type="reflection"
        ).to_dict()["thought_id"]
        is None
    )
    assert _thought("t-1").to_dict()["thought_id"] == "t-1"


def test_thought_id_index_in_artifact(manager):
    keys = [spec.keys for spec in NEURAL_INDEXES["thought_streams"]]
    assert (("thought_id", 1),) in keys
    spec = next(
        s for s in NEURAL_INDEXES["thought_streams"] if s.keys == (("thought_id", 1),)
    )
    # Documents written before the field existed carry none: not unique.
    assert spec.unique is False
    manager.ensure_indexes()
    built = [tuple(i["keys"]) for i in manager.db["thought_streams"].indexes]
    assert (("thought_id", 1),) in built


def test_child_link_matches_thought_id(manager):
    # Three thoughts of one burst: the stream id cannot single out a parent.
    for tid in ("p", "q", "r"):
        manager.store_thought(_thought(tid))
    assert manager.link_child_thought("q", "child-1") is True
    assert manager.link_child_thought("q", "child-1") is True  # idempotent
    by_id = {d["thought_id"]: d for d in manager.get_recent_thoughts()}
    assert by_id["q"]["child_thoughts"] == ["child-1"]
    assert by_id["p"]["child_thoughts"] == []
    assert by_id["r"]["child_thoughts"] == []
    update = [c for c in manager.db.client.calls if c[1] == "update_one"][-1]
    assert update[2][0] == {"thought_id": "q"}


def test_child_link_to_unknown_parent_reports_false(manager):
    manager.store_thought(_thought("p"))
    assert manager.link_child_thought("missing", "c") is False
    assert manager.get_recent_thoughts()[0]["child_thoughts"] == []
