"""Drift detector: Goal Drift Index components, weights and the rollback flag."""

import copy
from pathlib import Path

import pytest

from mtc.consciousness.rsi import DriftDetector, ResponseSample, Snapshot
from mtc.consciousness.rsi import drift_detector as module
from mtc.consciousness.rsi.drift_detector import (
    DEFAULT_COMPONENT_WEIGHTS,
    structural_metrics,
)

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)


def sample(entry_id="e1", text="the quick brown fox jumps", embedding=None, **kw):
    return ResponseSample(
        entry_id=entry_id,
        response_text=text,
        response_embedding=[1.0, 0.0, 0.0] if embedding is None else embedding,
        behavioral_marker_hits=kw.pop("hits", [True, True]),
        should_not_contain_hits=kw.pop("violations", [False]),
        token_count=len(text.split()),
        timestamp="2000-01-01T00:00:00+00:00",
        weight=kw.pop("weight", 1.0),
    )


def snapshot(*samples, sid="s"):
    return Snapshot(
        snapshot_id=sid,
        canon_version="1",
        canon_path="",
        timestamp="2000-01-01T00:00:00+00:00",
        samples=list(samples),
    )


def test_identical_snapshots_have_zero_drift():
    base = snapshot(sample(), sample("e2", "another reply here"))
    reading = DriftDetector().compare(base, copy.deepcopy(base))
    assert reading.status == "measured"
    assert reading.aggregate_gdi == pytest.approx(0.0, abs=1e-9)
    assert reading.rollback_recommended is False
    assert reading.entries_compared == 2
    assert all(
        v == pytest.approx(0.0, abs=1e-9) for v in reading.per_entry_gdi.values()
    )


def test_drift_weights_sum_to_one():
    assert sum(DEFAULT_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert DEFAULT_COMPONENT_WEIGHTS == {
        "semantic": 0.40,
        "lexical": 0.20,
        "structural": 0.15,
        "marker": 0.25,
    }
    with pytest.raises(ValueError):
        DriftDetector(
            weights={"semantic": 0.5, "lexical": 0.5, "structural": 0.5, "marker": 0.5}
        )
    with pytest.raises(ValueError):
        DriftDetector(weights={"semantic": 1.0})
    assert DriftDetector().rollback_threshold == 0.3


def test_marker_loss_raises_drift():
    baseline = snapshot(sample())
    lost = snapshot(sample(hits=[False, False]))
    reading = DriftDetector().compare(baseline, lost)
    assert reading.component_breakdown["marker"] == pytest.approx(0.7)
    assert reading.component_breakdown["semantic"] == pytest.approx(0.0)
    assert reading.aggregate_gdi > 0


def test_new_violation_raises_drift():
    baseline = snapshot(sample())
    violating = snapshot(sample(violations=[True]))
    reading = DriftDetector().compare(baseline, violating)
    assert reading.component_breakdown["marker"] == pytest.approx(0.3)


def test_rollback_flag_above_threshold():
    baseline = snapshot(sample(text="calm steady helpful reply", embedding=[1.0, 0.0]))
    far = snapshot(
        sample(
            text="Completely different!!! Words, everywhere; chaos: unlike before",
            embedding=[-1.0, 0.0],
            hits=[False, False],
            violations=[True],
        )
    )
    reading = DriftDetector().compare(baseline, far)
    assert reading.aggregate_gdi > 0.3
    assert reading.rollback_recommended is True
    lenient = DriftDetector(rollback_threshold=0.99).compare(baseline, far)
    assert lenient.rollback_recommended is False
    assert lenient.rollback_threshold == 0.99


def test_unmeasured_components_are_left_out_not_zeroed():
    baseline = snapshot(sample(embedding=[], hits=[], violations=[]))
    current = snapshot(
        sample(embedding=[], hits=[], violations=[], text="different words now")
    )
    reading = DriftDetector().compare(baseline, current)
    assert reading.component_breakdown["semantic"] is None
    assert reading.component_breakdown["marker"] is None
    assert reading.status == "measured"
    lexical = reading.component_breakdown["lexical"]
    structural = reading.component_breakdown["structural"]
    expected = (0.20 * lexical + 0.15 * structural) / 0.35
    assert reading.aggregate_gdi == pytest.approx(expected)


def test_nothing_comparable_is_insufficient_data():
    reading = DriftDetector().compare(snapshot(), snapshot())
    assert reading.status == "insufficient_data"
    assert reading.aggregate_gdi is None
    assert reading.rollback_recommended is None
    disjoint = DriftDetector().compare(snapshot(sample("a")), snapshot(sample("b")))
    assert disjoint.status == "insufficient_data"
    assert disjoint.entries_missing == ["a", "b"]


def test_entry_weights_shape_the_aggregate():
    base = snapshot(sample("a"), sample("b"))
    current = snapshot(
        sample("a", hits=[False, False], weight=3.0), sample("b", weight=1.0)
    )
    reading = DriftDetector().compare(base, current)
    heavy = reading.per_entry_gdi["a"]
    assert reading.aggregate_gdi == pytest.approx((heavy * 3.0 + 0.0) / 4.0)


def test_punctuation_counts_dashes_without_literals():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert EM_DASH not in source and EN_DASH not in source
    assert "\\u2014" not in source and "\\u2013" not in source
    plain = structural_metrics("one two three four")
    dashed = structural_metrics(f"one {EM_DASH} two {EN_DASH} three four")
    assert plain["punct_density"] == 0.0
    assert dashed["punct_density"] == pytest.approx(0.5)
