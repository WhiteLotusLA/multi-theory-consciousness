"""Perceptual reality monitor: external versus internal signal attribution."""

import json

import pytest

from mtc.consciousness.perceptual_reality_monitor import (
    DEFAULT_SOURCES,
    PerceptualRealityMonitor,
)


def test_registry_is_configurable():
    mon = PerceptualRealityMonitor(sources={"probe": "external", "echo": "internal"})
    mon.record_signal("probe", attribution_confidence=0.8)
    mon.record_signal("echo", attribution_confidence=0.8)
    stats = mon.get_statistics()
    assert stats["external_count"] == 1
    assert stats["internal_count"] == 1
    # The default names are not special once a registry is supplied.
    mon.record_signal("user_message", attribution_confidence=0.9)
    assert mon.get_statistics()["unknown_count"] == 1
    assert DEFAULT_SOURCES["user_message"] == "external"
    assert DEFAULT_SOURCES["sensory"] == "external"
    for name in ("memory_retrieval", "thought_generator", "dream", "imagination"):
        assert DEFAULT_SOURCES[name] == "internal"
    with pytest.raises(ValueError):
        PerceptualRealityMonitor(sources={"x": "sideways"})


def test_discrimination_none_until_both_kinds_seen():
    mon = PerceptualRealityMonitor()
    assert mon.discrimination() is None
    mon.record_signal("user_message", attribution_confidence=0.9)
    mon.record_signal("sensory", attribution_confidence=0.9)
    assert mon.discrimination() is None
    mon.record_signal("dream", attribution_confidence=0.9)
    assert mon.discrimination() is not None


def test_discrimination_rises_with_diversity_and_confidence():
    skewed = PerceptualRealityMonitor()
    for _ in range(9):
        skewed.record_signal("user_message", attribution_confidence=0.9)
    skewed.record_signal("dream", attribution_confidence=0.9)

    balanced = PerceptualRealityMonitor()
    for _ in range(5):
        balanced.record_signal("user_message", attribution_confidence=0.9)
        balanced.record_signal("dream", attribution_confidence=0.9)

    unsure = PerceptualRealityMonitor()
    for _ in range(5):
        unsure.record_signal("user_message", attribution_confidence=0.3)
        unsure.record_signal("dream", attribution_confidence=0.3)

    assert balanced.discrimination() > skewed.discrimination()
    assert balanced.discrimination() > unsure.discrimination()
    assert balanced.discrimination() == pytest.approx(0.9)


def test_missing_attribution_is_not_measured():
    mon = PerceptualRealityMonitor()
    mon.record_signal("user_message")
    mon.record_signal("dream")
    stats = mon.get_statistics()
    assert stats["external_count"] == 1
    assert stats["internal_count"] == 1
    assert stats["default_attribution_fraction"] == 1.0
    assert stats["mean_attribution"] is None
    assert stats["discrimination"] is None
    assert mon.discrimination() is None
    assert mon.get_context_string() == ""
    assert "not measurable" in mon.generate_report()
    assert not hasattr(PerceptualRealityMonitor, "DEFAULT_ATTRIBUTION")
    assert all(s.attribution_confidence is None for s in mon._signals)
    assert all(s.attribution_is_default for s in mon._signals)


def test_supplied_attribution_alone_enters_the_product():
    mon = PerceptualRealityMonitor()
    mon.record_signal("user_message", attribution_confidence=0.6)
    mon.record_signal("dream")
    stats = mon.get_statistics()
    assert stats["source_diversity"] == 1.0
    assert stats["mean_attribution"] == pytest.approx(0.6)
    assert stats["default_attribution_fraction"] == 0.5
    assert mon.discrimination() == pytest.approx(0.6)


def test_restored_default_attribution_is_not_a_measurement():
    """A signal persisted with the old 0.9 prior restores as unattributed."""
    state = {
        "scalars": {"signal_count": 2},
        "arrays": {},
        "histories": {
            "signals": [
                {
                    "summary": "",
                    "source": "user_message",
                    "kind": "external",
                    "attribution_confidence": 0.9,
                    "attribution_is_default": True,
                    "timestamp": 1.0,
                },
                {
                    "summary": "",
                    "source": "dream",
                    "kind": "internal",
                    "attribution_confidence": 0.7,
                    "attribution_is_default": False,
                    "timestamp": 2.0,
                },
            ]
        },
    }
    mon = PerceptualRealityMonitor()
    mon.from_state_dict(state)
    stats = mon.get_statistics()
    assert stats["default_attribution_fraction"] == 0.5
    assert stats["mean_attribution"] == pytest.approx(0.7)
    assert mon.discrimination() == pytest.approx(0.7)
    assert mon._signals[0].attribution_confidence is None


def test_unknown_source_is_never_external():
    mon = PerceptualRealityMonitor()
    mon.record_signal("mystery", attribution_confidence=1.0)
    mon.record_signal("dream", attribution_confidence=1.0)
    stats = mon.get_statistics()
    assert stats["external_count"] == 0
    assert stats["unknown_count"] == 1
    assert mon.discrimination() is None


def test_window_bounded():
    mon = PerceptualRealityMonitor(window=5)
    for i in range(20):
        mon.record_signal("user_message", summary=str(i), attribution_confidence=0.9)
    stats = mon.get_statistics()
    assert stats["window_size"] == 5
    assert stats["signal_count"] == 20


def test_context_string_empty_until_measurable():
    mon = PerceptualRealityMonitor()
    assert mon.get_context_string() == ""
    mon.record_signal("user_message", attribution_confidence=0.9)
    assert mon.get_context_string() == ""
    mon.record_signal("dream", attribution_confidence=0.9)
    line = mon.get_context_string()
    assert line.startswith("[") and line.endswith("]")
    assert "discrimination=" in line


def test_report_mentions_not_measurable():
    mon = PerceptualRealityMonitor()
    assert "not measurable" in mon.generate_report()
    mon.record_signal("user_message", attribution_confidence=0.9)
    mon.record_signal("dream", attribution_confidence=0.9)
    assert "0.900" in mon.generate_report()


def test_state_dict_round_trip_rehydrates_signals():
    mon = PerceptualRealityMonitor(window=10)
    mon.record_signal("user_message", "hello", 0.9)
    mon.record_signal("dream", "replay", None)
    state = json.loads(json.dumps(mon.to_state_dict()))
    fresh = PerceptualRealityMonitor(window=10)
    fresh.from_state_dict(state)
    assert fresh.get_statistics() == mon.get_statistics()
    assert fresh.discrimination() == mon.discrimination()
    assert fresh._signals.maxlen == 10
    fresh.record_signal("sensory", attribution_confidence=0.5)
    assert fresh.get_statistics()["signal_count"] == 3
