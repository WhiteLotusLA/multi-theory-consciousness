"""Orchestrator persistence restores typed experiences and counters, skips
malformed entries, and the extra system metrics are None without components."""

import logging
from datetime import datetime

import pytest

from mtc.neural.orchestrator import Experience, NeuralOrchestrator


def _orchestrator_with_history() -> NeuralOrchestrator:
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    orchestrator.experience_buffer = [
        Experience(
            timestamp=datetime(2026, 1, 2, 3, 4, 5),
            experience_type="observation",
            emotions={"joy": 0.7},
            context="a synthetic observation",
            valence=0.4,
            arousal=0.6,
        ),
        Experience(
            timestamp=datetime(2026, 1, 2, 3, 5, 0),
            experience_type="thought",
            emotions={},
            context="",
            valence=-0.2,
            arousal=0.1,
        ),
    ]
    orchestrator.total_experiences = 9
    orchestrator.total_sleep_cycles = 2
    orchestrator.last_sleep = datetime(2026, 1, 1, 22, 0, 0)
    orchestrator.state.experiences_processed = 9
    orchestrator.state.episodic_memories_count = 4
    orchestrator.state.semantic_patterns_count = 3
    return orchestrator


def test_state_dict_round_trip_rebuilds_experiences():
    source = _orchestrator_with_history()
    state = source.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}

    restored = NeuralOrchestrator(enable_mongodb=False)
    restored.from_state_dict(state)

    assert restored.total_experiences == 9
    assert restored.total_sleep_cycles == 2
    assert restored.last_sleep == datetime(2026, 1, 1, 22, 0, 0)
    assert restored.state.experiences_processed == 9
    assert restored.state.episodic_memories_count == 4
    assert restored.state.semantic_patterns_count == 3
    assert len(restored.experience_buffer) == 2
    for original, rebuilt in zip(source.experience_buffer, restored.experience_buffer):
        assert isinstance(rebuilt, Experience)
        assert rebuilt.timestamp == original.timestamp
        assert rebuilt.experience_type == original.experience_type
        assert rebuilt.emotions == original.emotions
        assert rebuilt.context == original.context
        assert rebuilt.valence == pytest.approx(original.valence)
        assert rebuilt.arousal == pytest.approx(original.arousal)


def test_restore_skips_malformed_entries(caplog):
    state = _orchestrator_with_history().to_state_dict()
    good = state["histories"]["experience_buffer"][0]
    state["histories"]["experience_buffer"] = [
        good,
        "not an entry",
        {**good, "timestamp": "not a time"},
        {**good, "valence": float("nan")},
        {**good, "emotions": {"joy": "high"}},
        {**good, "arousal": None},
    ]
    state["scalars"]["last_sleep"] = "yesterday"

    restored = NeuralOrchestrator(enable_mongodb=False)
    with caplog.at_level(logging.WARNING, logger="mtc.neural.orchestrator"):
        restored.from_state_dict(state)

    assert len(restored.experience_buffer) == 1
    assert restored.experience_buffer[0].context == good["context"]
    assert restored.last_sleep is None
    assert any("skipped" in record.getMessage() for record in caplog.records)


def test_system_metrics_extras_are_none_without_components():
    metrics = NeuralOrchestrator(enable_mongodb=False).get_system_metrics()
    assert metrics["memory"]["consolidation_strength"] is None
    assert metrics["intuition"] is None
    assert metrics["htm_metrics"] is None
