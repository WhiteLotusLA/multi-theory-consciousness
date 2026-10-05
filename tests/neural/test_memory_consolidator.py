"""The memory consolidator: episodes replayed through an injected HTM.

Most cases use a deterministic stand-in HTM; a small ``ProductionHTM`` covers
the real interface. A recording stand-in replaces the data manager.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from mtc.core.config import Settings
from mtc.memory.protocols import EpisodeConsolidator
from mtc.neural.htm import (
    ConsolidatorConfig,
    EpisodicMemory,
    MemoryConsolidator,
    SemanticPattern,
    get_best_htm,
)
from mtc.neural.htm.htm_core import SDR
from mtc.neural.htm.memory_consolidator import _flatten_cells
from mtc.neural.htm.production_htm import HTMConfig, ProductionHTM

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class FakeHTM:
    """Maps a pattern's strongest input to a fixed block of columns."""

    def __init__(self, columns: int = 32, cells: int = 4, anomaly: float = 0.2):
        self.input_size = 16
        self.config = SimpleNamespace(num_columns=columns, cells_per_column=cells)
        self.anomaly = anomaly
        self.resets = 0
        self.processed: List[np.ndarray] = []
        self.learn_flags: List[bool] = []

    def reset(self) -> None:
        self.resets += 1

    def process(self, pattern, learn=True) -> Dict[str, Any]:
        self.processed.append(np.array(pattern))
        self.learn_flags.append(learn)
        start = int(np.argmax(pattern)) * 2
        active = np.zeros(self.config.num_columns)
        active[start : start + 3] = 1
        return {
            "active_columns": active,
            "active_cells": {(start, 1), (start + 1, 3)},
            "predicted_cells": {(0, 2)},
            "anomaly_score": self.anomaly,
            "num_active_columns": 3,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {"anomaly_score": self.anomaly}


class RecordingManager:
    def __init__(self) -> None:
        self.columns: List[Any] = []

    def store_htm_column(self, column):
        self.columns.append(column)
        return "id"

    def status(self):
        return {"backend": "recording", "available": True, "last_error": None}


class Ticker:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def _consolidator(htm=None, **kwargs) -> MemoryConsolidator:
    return MemoryConsolidator(htm=htm or FakeHTM(), clock=Ticker(), **kwargs)


def _small_htm() -> ProductionHTM:
    return ProductionHTM(
        HTMConfig(
            num_columns=64,
            cells_per_column=4,
            input_bits=128,
            input_sparsity=0.05,
            potential_radius=64,
            local_area_density=0.05,
            activation_threshold=2,
            max_segments_per_cell=16,
            max_synapses_per_segment=8,
        )
    )


def test_injected_htm_is_used():
    htm = FakeHTM()
    consolidator = _consolidator(htm)
    assert consolidator.htm is htm
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    consolidator.consolidate_memories(sleep_duration="short")
    assert htm.resets == 1 and len(htm.processed) == 10
    assert all(htm.learn_flags)
    assert consolidator.htm is htm


def test_injected_htm_interface_checked():
    with pytest.raises(TypeError):
        MemoryConsolidator(htm=object())
    broken = FakeHTM()
    broken.process = None
    with pytest.raises(TypeError):
        MemoryConsolidator(htm=broken)
    no_size = FakeHTM()
    del no_size.input_size
    with pytest.raises(TypeError):
        MemoryConsolidator(htm=no_size)


def test_default_htm_sizes_from_settings():
    settings = Settings(HTM_COLUMNS=64, HTM_CELLS_PER_COLUMN=4)
    consolidator = MemoryConsolidator(settings=settings)
    assert isinstance(consolidator.htm, ProductionHTM)
    assert consolidator.htm.config.num_columns == 64
    assert consolidator.htm.config.cells_per_column == 4
    default = Settings()
    assert default.HTM_COLUMNS == 4096 and default.HTM_CELLS_PER_COLUMN == 32
    assert (
        get_best_htm(HTMConfig(num_columns=32, input_bits=64)).config.num_columns == 32
    )


def test_importance_weights_configurable():
    default = _consolidator()
    memory = default.add_episodic_memory(
        "emotion", {"joy": 0.5, "fear": 0.9}, consciousness_level=0.5
    )
    assert memory.importance_score == pytest.approx(0.5 * 0.4 + 0.9 * 0.6)
    assert memory.importance_basis == "consciousness_and_emotion"
    assert memory.consciousness_level_at_encoding == 0.5
    tuned = _consolidator(
        config=ConsolidatorConfig(consciousness_weight=0.9, emotion_weight=0.1)
    )
    again = tuned.add_episodic_memory(
        "emotion", {"joy": 0.5, "fear": 0.9}, consciousness_level=0.5
    )
    assert again.importance_score == pytest.approx(0.5 * 0.9 + 0.9 * 0.1)
    clipped = default.add_episodic_memory("e", {"joy": 0.5}, consciousness_level=3.0)
    assert clipped.consciousness_level_at_encoding == 1.0
    with pytest.raises(ValueError):
        default.add_episodic_memory("e", {"joy": 0.5}, consciousness_level=float("nan"))


def test_importance_without_consciousness_level_uses_emotion_only_and_records_it():
    consolidator = _consolidator()
    memory = consolidator.add_episodic_memory("emotion", {"joy": 0.5, "fear": 0.9})
    assert memory.consciousness_level_at_encoding is None
    assert memory.importance_score == pytest.approx(0.9)
    assert memory.importance_basis == "emotion_only"
    quiet = consolidator.add_episodic_memory("emotion", {})
    assert quiet.importance_score == 0.0 and quiet.importance_basis == "emotion_only"
    assert memory.anomaly_score is None and memory.consolidation_strength == 0.0


def test_flattened_cell_ids_do_not_collide():
    assert sorted(_flatten_cells({(0, 3), (1, 0), (2, 3)}, 4)) == [3, 4, 11]
    pairs = {(c, k) for c in range(8) for k in range(4)}
    flat = _flatten_cells(pairs, 4)
    assert len(set(flat)) == len(pairs) == 32  # column * cells + cell is unique
    assert sorted(_flatten_cells({5, 9}, None)) == [5, 9]
    assert _flatten_cells({(1, 2)}, None) == []  # no width, no guess
    manager = RecordingManager()
    consolidator = _consolidator(neural_data_manager=manager)
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    consolidator.consolidate_memories()
    column = manager.columns[0]
    assert sorted(column.active_cells) == [1, 7]  # (0, 1) and (1, 3) with 4 cells
    assert column.predictive_cells == [2]


def test_consolidation_report_keys():
    consolidator = _consolidator()
    for emotion in ("joy", "fear", "love"):
        consolidator.add_episodic_memory("emotion", {emotion: 0.8})
    report = consolidator.consolidate_memories(sleep_duration="medium")
    assert report["status"] == "completed"
    for key in (
        "sleep_duration",
        "cycles",
        "memories_processed",
        "memories_skipped",
        "avg_initial_anomaly",
        "avg_final_anomaly",
        "anomaly_reduction",
        "patterns_discovered",
        "new_patterns",
        "elapsed_time_sec",
        "consolidation_success",
    ):
        assert key in report, key
    assert report["cycles"] == 30 and report["memories_processed"] == 3
    assert report["patterns_discovered"] == len(consolidator.semantic_patterns)
    assert report["anomaly_reduction"] == pytest.approx(0.0)
    assert report["consolidation_success"] is True  # 0.2 is below the 0.5 line
    assert consolidator.total_consolidation_cycles == 30
    assert consolidator.consolidate_memories([])["status"] == "no_memories"
    deep = consolidator.consolidate_memories(sleep_duration="deep")
    assert deep["cycles"] == 100
    other = _consolidator(consolidation_cycles=7)
    other.add_episodic_memory("emotion", {"joy": 0.8})
    assert other.consolidate_memories(sleep_duration="odd")["cycles"] == 7


def test_one_cycle_has_no_anomaly_reduction():
    consolidator = _consolidator(config=ConsolidatorConfig(short_cycles=1))
    consolidator.add_episodic_memory("emotion", {"joy": 0.8})
    report = consolidator.consolidate_memories()
    assert report["cycles"] == 1 and report["anomaly_reduction"] is None


def test_episodic_cap():
    consolidator = _consolidator(config=ConsolidatorConfig(episodic_cap=3))
    for index in range(5):
        consolidator.add_episodic_memory("emotion", {"joy": index / 10})
    assert [m.emotional_state["joy"] for m in consolidator.episodic_memories] == [
        0.2,
        0.3,
        0.4,
    ]
    assert ConsolidatorConfig().episodic_cap == 500


def test_replay_is_most_important_first_and_resizes_patterns():
    htm = FakeHTM()
    consolidator = _consolidator(htm)
    consolidator.add_episodic_memory("a", {"joy": 0.2})
    consolidator.add_episodic_memory(
        "b", {"joy": 0.9}, subconscious_pattern=[0.0, 1.0, 0.0]
    )
    consolidator.consolidate_memories(
        sleep_duration="short", prioritize_by_consciousness=True
    )
    assert all(len(p) == htm.input_size for p in htm.processed)
    first = htm.processed[0]
    assert first[1] == 1.0  # the more important episode (b) is replayed first


def test_non_finite_pattern_is_skipped_and_counted():
    consolidator = _consolidator()
    consolidator.add_episodic_memory("a", {"joy": 0.9})
    consolidator.add_episodic_memory(
        "b", {"joy": 0.9}, subconscious_pattern=[0.0, float("nan")]
    )
    report = consolidator.consolidate_memories()
    assert report["memories_processed"] == 1 and report["memories_skipped"] == 1


def test_frequency_counts_distinct_episodes_not_replays():
    consolidator = _consolidator()
    for _ in range(3):
        consolidator.add_episodic_memory("emotion", {"joy": 0.9, "love": 0.3})
    consolidator.consolidate_memories(sleep_duration="deep")
    patterns = consolidator.get_semantic_patterns()
    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.frequency == 3 and len(pattern.source_memories) == 3
    assert pattern.pattern_type == "emotional_sequence"
    again = consolidator.consolidate_memories(sleep_duration="short")
    assert again["new_patterns"] == 0
    assert consolidator.get_semantic_patterns()[0].frequency == 3
    assert consolidator.patterns_discovered == 1


def test_pattern_types_filters_and_ordering():
    consolidator = _consolidator()
    consolidator.add_episodic_memory("a", {"joy": 0.9})
    for _ in range(2):
        consolidator.add_episodic_memory("b", {"fear": 0.9, "joy": 0.4})
    consolidator.add_episodic_memory("c", {"curiosity": 0.9})
    consolidator.consolidate_memories()
    everything = consolidator.get_semantic_patterns()
    assert [p.frequency for p in everything] == sorted(
        (p.frequency for p in everything), reverse=True
    )
    assert {p.pattern_type for p in everything} <= {
        "emotional_sequence",
        "recurring_theme",
    }
    assert all(
        p.frequency >= 2 for p in consolidator.get_semantic_patterns(min_frequency=2)
    )
    themes = consolidator.get_semantic_patterns(pattern_type="recurring_theme")
    assert all(p.pattern_type == "recurring_theme" for p in themes)
    assert isinstance(everything[0], SemanticPattern)


def test_weak_episodes_form_no_pattern():
    consolidator = _consolidator(FakeHTM(anomaly=0.7))
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    report = consolidator.consolidate_memories()
    assert report["new_patterns"] == 0 and consolidator.semantic_patterns == {}
    assert report["consolidation_success"] is False


def test_context_text_not_stored_by_default():
    manager = RecordingManager()
    consolidator = _consolidator(neural_data_manager=manager)
    consolidator.add_episodic_memory(
        "emotion", {"joy": 0.9}, context="a synthetic sentence"
    )
    consolidator.consolidate_memories()
    assert "memory_context" not in manager.columns[0].metadata
    assert "synthetic" not in str(manager.columns[0].to_dict())
    state = consolidator.to_state_dict()
    assert state["histories"]["episodic_memories"][0]["context"] == ""
    assert all(p["description"] == "" for p in state["histories"]["semantic_patterns"])
    # in process the text stays available (the memory manager reads it)
    assert consolidator.get_semantic_patterns()[0].description == "a synthetic sentence"
    opted = _consolidator(
        neural_data_manager=RecordingManager(),
        config=ConsolidatorConfig(store_context_text=True),
    )
    opted.add_episodic_memory("emotion", {"joy": 0.9}, context="a synthetic sentence")
    opted.consolidate_memories()
    assert (
        opted.neural_data_manager.columns[0].metadata["memory_context"]
        == "a synthetic sentence"
    )
    assert (
        opted.to_state_dict()["histories"]["episodic_memories"][0]["context"]
        == "a synthetic sentence"
    )


def test_htm_columns_written_only_with_data_manager():
    quiet = _consolidator()
    quiet.add_episodic_memory("emotion", {"joy": 0.9})
    quiet.consolidate_memories()  # no manager: nothing to write and nothing fails
    manager = RecordingManager()
    consolidator = _consolidator(neural_data_manager=manager)
    for emotion in ("joy", "fear"):
        consolidator.add_episodic_memory("emotion", {emotion: 0.9})
    consolidator.consolidate_memories(sleep_duration="short")
    assert len(manager.columns) == 2  # final cycle only, one per episode
    assert {c.column_id for c in manager.columns} == {0, 1}
    column = manager.columns[0]
    assert column.timestamp.tzinfo is not None
    assert column.semantic_label == "emotion"
    assert column.prediction_confidence == pytest.approx(0.8)
    assert (
        column.metadata["total_cycles"] == 10
        and column.metadata["consolidation_cycle"] == 9
    )


def test_storage_failure_is_reported_not_raised():
    class Failing(RecordingManager):
        def store_htm_column(self, column):
            raise RuntimeError("write rejected")

    consolidator = _consolidator(neural_data_manager=Failing())
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    assert consolidator.consolidate_memories()["status"] == "completed"
    assert "write rejected" in consolidator.status()["last_error"]


def test_state_dict_round_trip():
    consolidator = _consolidator(config=ConsolidatorConfig(store_context_text=True))
    for emotion in ("joy", "fear"):
        consolidator.add_episodic_memory(
            "emotion",
            {emotion: 0.9},
            subconscious_pattern=[0.1, 0.9, 0.2],
            context=f"about {emotion}",
            consciousness_level=0.6,
        )
    consolidator.add_episodic_memory("raw", {"love": 0.4})
    consolidator.consolidate_memories()
    saved = consolidator.to_state_dict()
    assert set(saved) == {"scalars", "arrays", "histories"}
    restored = _consolidator(config=ConsolidatorConfig(store_context_text=True))
    restored.from_state_dict(saved)
    assert len(restored.episodic_memories) == 3
    assert len(restored.semantic_patterns) == len(consolidator.semantic_patterns)
    assert restored.patterns_discovered == consolidator.patterns_discovered
    assert restored.total_consolidation_cycles == 10
    first, original = restored.episodic_memories[0], consolidator.episodic_memories[0]
    assert first.memory_id == original.memory_id
    assert first.context == original.context
    assert first.htm_sdr.active_indices == original.htm_sdr.active_indices
    assert first.consciousness_level_at_encoding == 0.6
    assert np.allclose(first.subconscious_pattern, original.subconscious_pattern)
    assert restored.to_state_dict()["histories"] == saved["histories"]
    for pattern in restored.semantic_patterns.values():
        original_pattern = consolidator.semantic_patterns[pattern.pattern_id]
        assert pattern.frequency == original_pattern.frequency
        assert pattern.source_memories == original_pattern.source_memories


def test_restore_skips_malformed_entries_and_never_raises(caplog):
    consolidator = _consolidator()
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    consolidator.consolidate_memories()
    saved = consolidator.to_state_dict()
    good_episode = saved["histories"]["episodic_memories"][0]
    good_pattern = saved["histories"]["semantic_patterns"][0]
    broken = {
        "scalars": {"patterns_discovered": -4, "total_consolidation_cycles": True},
        "histories": {
            "episodic_memories": [
                "junk",
                {"timestamp": "bad"},
                {**good_episode, "importance_score": float("nan")},
                {**good_episode, "htm_sdr": {"size": 4, "active": [9]}},
                good_episode,
            ],
            "semantic_patterns": [{"pattern_id": "x"}, good_pattern],
        },
    }
    fresh = _consolidator()
    with caplog.at_level("WARNING"):
        fresh.from_state_dict(broken)
    assert len(fresh.episodic_memories) == 1 and len(fresh.semantic_patterns) == 1
    assert fresh.patterns_discovered == 0 and fresh.total_consolidation_cycles == 0
    assert any("skipped 4 malformed episodes" in r.getMessage() for r in caplog.records)
    fresh.from_state_dict({"histories": {"episodic_memories": "garbage"}})
    fresh.from_state_dict({})
    assert len(fresh.episodic_memories) == 1


def test_satisfies_episode_consolidator():
    consolidator = _consolidator()
    assert isinstance(consolidator, EpisodeConsolidator)
    memory = consolidator.add_episodic_memory(
        memory_type="emotion",
        emotional_state={"joy": 0.9},
        subconscious_pattern=None,
        context="a synthetic sentence",
        consciousness_level=None,
    )
    assert isinstance(memory, EpisodicMemory)
    assert isinstance(consolidator.consolidate_memories(), dict)
    assert isinstance(consolidator.get_semantic_patterns(), list)
    assert isinstance(consolidator.get_semantic_patterns("recurring_theme", 1), list)


def test_summary_and_status_report_measured_values_only():
    consolidator = _consolidator()
    summary = consolidator.get_consolidation_summary()
    assert summary["avg_consolidation_strength"] is None  # nothing replayed yet
    assert summary["episodic_memories"] == 0 and summary["is_sleeping"] is False
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    consolidator.consolidate_memories()
    summary = consolidator.get_consolidation_summary()
    assert summary["avg_consolidation_strength"] == pytest.approx(0.8)
    assert summary["htm_metrics"] == {"anomaly_score": 0.2}
    status = consolidator.status()
    assert status["backend"] == "numpy" and status["available"] is True
    assert status["htm"] == "FakeHTM" and status["last_error"] is None


def test_real_htm_replay_reports_measured_anomalies():
    consolidator = MemoryConsolidator(htm=_small_htm(), clock=Ticker())
    for emotion in ("joy", "fear", "joy"):
        consolidator.add_episodic_memory(
            "emotion",
            {emotion: 0.9},
            subconscious_pattern=np.linspace(0.0, 1.0, 20),
        )
    report = consolidator.consolidate_memories(sleep_duration="short")
    assert report["status"] == "completed" and report["cycles"] == 10
    assert 0.0 <= report["avg_final_anomaly"] <= 1.0
    for memory in consolidator.episodic_memories:
        assert memory.htm_sdr is not None and memory.htm_sdr.size == 64
        assert memory.replay_count == 10 and memory.anomaly_score is not None


def test_timestamps_are_aware_utc_by_default():
    consolidator = MemoryConsolidator(htm=FakeHTM())
    memory = consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    assert memory.timestamp.tzinfo is not None
    assert memory.timestamp.utcoffset() == timedelta(0)


def test_sdr_from_index_collection_uses_the_htm_column_count():
    class IndexHTM(FakeHTM):
        def process(self, pattern, learn=True):
            result = super().process(pattern, learn)
            result["active_columns"] = [1, 2, 3]
            return result

    consolidator = _consolidator(IndexHTM(columns=40))
    consolidator.add_episodic_memory("emotion", {"joy": 0.9})
    consolidator.consolidate_memories()
    sdr = consolidator.episodic_memories[0].htm_sdr
    assert isinstance(sdr, SDR) and sdr.size == 40 and sdr.active_indices == {1, 2, 3}
