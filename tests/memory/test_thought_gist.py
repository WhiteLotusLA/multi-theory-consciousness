"""ThoughtGist document round trips and timestamp handling."""

import copy
from datetime import datetime, timedelta, timezone

import pytest

from mtc.memory.thought_gist import ThoughtGist

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _gist(**overrides) -> ThoughtGist:
    values = dict(
        gist_id="g1",
        created_at=T0,
        summary="a short summary",
        source_count=3,
        source_thought_ids=["t1", "t2", "t3"],
        dominant_emotion="calm",
        emotional_arc={"start": "calm", "end": "curious"},
        time_span={"minutes": 12},
        burst_windows=[{"start": 1, "end": 4}],
        avg_salience=0.6,
        total_access_count=2,
        content_variations=["v1"],
        importance_score=0.7,
        retention_strength=0.9,
        is_protected=True,
        protection_reason="synthetic",
        retrieval_count=4,
        last_retrieved=T0 + timedelta(hours=1),
    )
    values.update(overrides)
    return ThoughtGist(**values)


def test_document_round_trip():
    gist = _gist()
    restored = ThoughtGist.from_document(gist.to_document())
    assert restored == gist


def test_document_round_trip_keeps_source_count():
    gist = _gist(source_count=7, source_thought_ids=["a"])
    doc = gist.to_document()
    assert doc["source_count"] == 7
    assert "source_thought_count" not in doc
    assert ThoughtGist.from_document(doc).source_count == 7


def test_created_at_is_aware_utc():
    local = timezone(timedelta(hours=5))
    gist = _gist(created_at=datetime(2000, 1, 1, 17, 0, 0, tzinfo=local))
    assert gist.created_at == T0
    assert gist.created_at.utcoffset() == timedelta(0)
    with pytest.raises(ValueError):
        _gist(created_at=datetime(2000, 1, 1, 12, 0, 0))
    with pytest.raises(ValueError):
        _gist(last_retrieved=datetime(2000, 1, 1, 12, 0, 0))
    # a driver that returns naive UTC datetimes is read as UTC
    doc = _gist().to_document()
    doc["created_at"] = datetime(2000, 1, 1, 12, 0, 0)
    assert ThoughtGist.from_document(doc).created_at == T0


def test_from_document_tolerates_missing_optional_fields():
    minimal = {
        "gist_id": "g2",
        "created_at": T0,
        "summary": "s",
        "source_count": 1,
    }
    gist = ThoughtGist.from_document(minimal)
    assert gist.source_thought_ids == []
    assert gist.dominant_emotion == "neutral"
    assert gist.retention_strength == 1.0
    assert gist.is_protected is False
    assert gist.protection_reason is None
    assert gist.last_retrieved is None
    for required in ("gist_id", "created_at", "summary", "source_count"):
        broken = copy.deepcopy(minimal)
        del broken[required]
        with pytest.raises(KeyError):
            ThoughtGist.from_document(broken)


def test_to_document_is_a_copy():
    gist = _gist()
    doc = gist.to_document()
    doc["source_thought_ids"].append("x")
    doc["emotional_arc"]["start"] = "changed"
    assert gist.source_thought_ids == ["t1", "t2", "t3"]
    assert gist.emotional_arc["start"] == "calm"
