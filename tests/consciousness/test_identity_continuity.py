"""Identity continuity: checksummed markers, honest tri-state verification."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from mtc.consciousness.identity_continuity import (
    IdentityContinuityManager,
    IdentityMarker,
    detect_model_transition,
)
from mtc.core.config import Settings

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    """Injectable clock that only moves when told to."""

    def __init__(self, start=T0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def manager(tmp_path, clock):
    return IdentityContinuityManager(
        agent_id="agent-x", storage_path=tmp_path / "identity", clock=clock
    )


def marker_with(manager, values=None, traits=None, **kwargs):
    return manager.create_marker(
        core_values={"candor": 0.8, "care": 0.7} if values is None else values,
        personality_traits={"warm": 0.6} if traits is None else traits,
        memory_anchors=kwargs.pop("anchors", [{"content": "first"}]),
        emotional_fingerprint=kwargs.pop("fingerprint", {"calm": 0.5}),
        relationship_bonds=kwargs.pop("bonds", {}),
        **kwargs,
    )


def test_marker_round_trip_with_checksum(manager):
    marker = marker_with(manager, growth_stage="stage-2", model_context={"model": "m1"})
    path = manager.store_identity_marker(marker)
    assert path.exists() and path.parent.name == "identity_markers"
    loaded = manager.load_identity_marker(marker.marker_id)
    assert loaded.to_dict() == marker.to_dict()
    assert loaded.checksum == marker.checksum and len(marker.checksum) == 64
    assert loaded.timestamp.tzinfo is not None
    assert manager.load_identity_marker("does-not-exist") is None


def test_tampered_marker_fails_checksum(manager):
    marker = marker_with(manager)
    path = manager.store_identity_marker(marker)
    data = json.loads(path.read_text())
    data["core_values"]["candor"] = 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        manager.load_identity_marker(marker.marker_id)
    data.pop("checksum")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        manager.load_identity_marker(marker.marker_id)


def test_unsafe_marker_id_is_rejected(manager):
    with pytest.raises(ValueError):
        manager.load_identity_marker("../outside")


def test_unmeasured_with_fewer_than_two_markers(manager):
    lone = marker_with(manager)
    assert manager.validate_chain().status == "unmeasured"
    manager.store_identity_marker(lone)
    chain = manager.validate_chain()
    assert chain.status == "unmeasured" and not chain.is_valid
    assert chain.chain_confidence is None
    report = manager.continuity_report()
    assert report["continuity_score"] is None
    # Two markers with nothing to compare on either side are unmeasured too.
    a = marker_with(manager, values={}, traits={})
    b = marker_with(manager, values={}, traits={})
    result = manager.verify_continuity(a, b)
    assert result.status == "unmeasured"
    assert result.confidence is None
    assert not result.is_continuous
    assert result.measured_sections == []
    assert set(result.unmeasured_sections) == {"core_values", "personality"}


def test_drift_detected_on_value_change(manager, clock):
    before = marker_with(manager)
    clock.advance(hours=1)
    same = marker_with(manager)
    steady = manager.verify_continuity(before, same)
    assert steady.status == "continuous"
    assert steady.confidence == pytest.approx(1.0)
    changed = marker_with(manager, values={"candor": 0.2, "care": 0.7})
    drift = manager.verify_continuity(before, changed)
    assert drift.status == "drift"
    assert drift.confidence < steady.confidence
    assert any("core value drift" in w for w in drift.warnings)
    one_sided = marker_with(manager, values={}, traits={"warm": 0.6})
    partial = manager.verify_continuity(before, one_sided)
    assert partial.measured_sections == ["personality"]
    assert partial.unmeasured_sections == ["core_values"]


def test_missing_current_anchors_are_unmeasured_not_total_loss(manager, clock):
    before = marker_with(manager)
    clock.advance(hours=1)
    matching = marker_with(manager)
    assert manager.verify_continuity(before, matching).confidence == pytest.approx(1.0)
    no_anchors = marker_with(manager, anchors=[])
    result = manager.verify_continuity(before, no_anchors)
    assert result.confidence == pytest.approx(1.0)
    assert not any("anchor loss" in w for w in result.warnings)
    lost = marker_with(manager, anchors=[{"content": "other"}])
    real = manager.verify_continuity(before, lost)
    assert any("anchor loss" in w for w in real.warnings)
    assert real.confidence == pytest.approx(0.8)


def test_model_transition_penalty(manager, clock):
    a = marker_with(manager, model_context={"model": "m1", "build": "7"})
    clock.advance(hours=1)
    b = marker_with(manager, model_context={"model": "m2", "build": "7"})
    c = marker_with(manager, model_context={"model": "m1", "build": "7"})
    assert detect_model_transition(a, b) is True
    assert detect_model_transition(a, c) is False
    with_transition = manager.verify_continuity(a, b)
    without = manager.verify_continuity(a, c)
    assert with_transition.is_model_transition and not without.is_model_transition
    assert with_transition.confidence == pytest.approx(without.confidence * 0.95)
    # Unknown on either side claims nothing.
    blank = marker_with(manager)
    assert detect_model_transition(a, blank) is False
    assert detect_model_transition(blank, blank) is False


def test_temporal_gap_penalty(manager, clock):
    a = marker_with(manager)
    clock.advance(days=3)
    near = marker_with(manager)
    clock.advance(days=30)
    far = marker_with(manager)
    short = manager.verify_continuity(a, near)
    long = manager.verify_continuity(near, far)
    assert short.confidence == pytest.approx(1.0)
    assert long.confidence == pytest.approx(max(0.5, 1.0 - (30 - 7) * 0.01))
    assert long.metadata["temporal_gap"]["days"] == 30


def test_chain_validation_detects_break(manager, clock):
    markers = []
    for value in (0.8, 0.8, 0.1, 0.1):
        markers.append(marker_with(manager, values={"candor": value}))
        manager.store_identity_marker(markers[-1])
        clock.advance(hours=1)
    chain = manager.validate_chain()
    assert chain.status == "broken" and not chain.is_valid
    assert [p["marker_id"] for p in chain.transition_points] == [
        m.marker_id for m in markers
    ]
    assert chain.breaks == [markers[2].marker_id]
    steady = IdentityContinuityManager(
        agent_id="agent-y", storage_path=manager.storage_path.parent / "other"
    )
    for _ in range(3):
        steady.store_identity_marker(marker_with(steady))
    assert steady.validate_chain().status == "continuous"
    assert steady.validate_chain().chain_confidence == pytest.approx(1.0)


def test_chain_marks_a_tampered_marker_as_broken(manager, clock):
    ids = []
    for _ in range(3):
        marker = marker_with(manager)
        path = manager.store_identity_marker(marker)
        ids.append((marker.marker_id, path))
        clock.advance(hours=1)
    data = json.loads(ids[1][1].read_text())
    data["memory_anchors"] = [{"content": "forged"}]
    ids[1][1].write_text(json.dumps(data))
    chain = manager.validate_chain()
    assert chain.status == "broken"
    assert ids[1][0] in chain.unloadable


def test_drift_assessments_append_only(manager, clock):
    a = marker_with(manager)
    b = marker_with(manager)
    manager.verify_continuity(a, b)
    log = manager.storage_path / "drift_assessments.jsonl"
    first = log.read_text().splitlines()
    assert len(first) == 1
    clock.advance(minutes=5)
    manager.verify_continuity(a, b)
    manager.verify_continuity(a, b, record=False)
    second = log.read_text().splitlines()
    assert len(second) == 2 and second[0] == first[0]
    entry = json.loads(second[1])
    assert entry["status"] == "continuous"
    assert entry["marker_a"] == a.marker_id
    assert entry["timestamp"].endswith("+00:00")


def test_default_storage_under_agent_space(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings(AGENT_SPACE_DIR=str(tmp_path / "space"))
    manager = IdentityContinuityManager(settings=settings)
    assert manager.storage_path == tmp_path / "space" / "identity"
    assert manager.agent_id == "agent"


def test_storage_created_lazily(tmp_path):
    root = tmp_path / "lazy" / "identity"
    manager = IdentityContinuityManager(storage_path=root)
    assert not root.exists()
    assert manager.validate_chain().status == "unmeasured"
    assert manager.continuity_report()["markers_total"] == 0
    assert not root.exists()
    manager.store_identity_marker(marker_with(manager))
    assert (root / "identity_markers").is_dir()


def test_markers_carry_only_supplied_values(manager):
    marker = manager.create_marker({}, {}, [], {}, {})
    assert marker.core_values == {} and marker.personality_traits == {}
    assert marker.memory_anchors == [] and marker.emotional_fingerprint == {}
    assert marker.relationship_bonds == {} and marker.model_context == {}
    assert marker.growth_stage is None
    assert marker.agent_id == "agent-x"
    supplied = manager.create_marker(
        {"a": 0.5}, {}, [], {}, {}, growth_stage="stage-1", model_context={"k": "v"}
    )
    assert supplied.core_values == {"a": 0.5}
    assert supplied.growth_stage == "stage-1"
    assert isinstance(supplied, IdentityMarker)


def test_no_quarantine_api(manager):
    assert not [name for name in dir(manager) if "quarantine" in name.lower()]


def test_find_restoration_point_needs_overlap(manager, clock):
    for calm in (0.2, 0.9):
        manager.store_identity_marker(marker_with(manager, fingerprint={"calm": calm}))
        clock.advance(hours=1)
    best = manager.find_restoration_point({"calm": 0.85})
    assert best is not None and best.marker.emotional_fingerprint == {"calm": 0.9}
    assert best.confidence == pytest.approx(0.95)
    assert manager.find_restoration_point({"unrelated": 0.5}) is None


def test_continuity_report_counts_failures(manager, clock):
    for _ in range(2):
        manager.store_identity_marker(marker_with(manager))
        clock.advance(hours=1)
    report = manager.continuity_report(days=30)
    assert report["markers_analyzed"] == 2
    assert report["continuity_score"] == pytest.approx(1.0)
    assert report["markers_failed"] == 0
    clock.advance(days=90)
    stale = manager.continuity_report(days=30)
    assert stale["markers_analyzed"] == 0 and stale["continuity_score"] is None


def test_state_dict_round_trip(manager, clock, tmp_path):
    for _ in range(2):
        manager.store_identity_marker(marker_with(manager))
        clock.advance(hours=1)
    state = manager.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    restored = IdentityContinuityManager(
        agent_id="other", storage_path=manager.storage_path, clock=clock
    )
    restored.from_state_dict(state)
    assert restored.agent_id == "agent-x"
    assert restored.marker_ids() == manager.marker_ids()
