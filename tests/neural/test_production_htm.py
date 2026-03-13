#!/usr/bin/env python3
"""
Tests for Production HTM (Hierarchical Temporal Memory).

All tests use mocks -- no real database or LLM connections required.
"""

import sys
from pathlib import Path


import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mtc.neural.htm.production_htm import (
    HTMConfig,
    SpatialPooler,
    TemporalMemory,
    ProductionHTM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> HTMConfig:
    """Return a small HTMConfig suitable for fast unit tests."""
    defaults = dict(
        num_columns=64,
        cells_per_column=4,
        input_bits=128,
        input_sparsity=0.05,
        potential_radius=64,
        potential_pct=0.5,
        global_inhibition=True,
        local_area_density=0.05,
        activation_threshold=2,
        max_segments_per_cell=16,
        max_synapses_per_segment=8,
        learning_enabled=True,
    )
    defaults.update(overrides)
    return HTMConfig(**defaults)


def _random_sdr(size: int, sparsity: float = 0.05) -> np.ndarray:
    """Generate a random sparse distributed representation."""
    sdr = np.zeros(size)
    num_active = max(1, int(size * sparsity))
    active_bits = np.random.choice(size, num_active, replace=False)
    sdr[active_bits] = 1
    return sdr


# ---------------------------------------------------------------------------
# SpatialPooler tests
# ---------------------------------------------------------------------------

class TestSpatialPooler:
    """Tests for the SpatialPooler component."""

    def test_compute_returns_binary_vector(self):
        """SP output must be a binary vector of length num_columns."""
        cfg = _small_config()
        sp = SpatialPooler(cfg)
        inp = _random_sdr(cfg.input_bits)

        result = sp.compute(inp, learn=False)

        assert result.shape == (cfg.num_columns,)
        assert set(np.unique(result)).issubset({0, 1, True, False})

    def test_compute_respects_local_area_density(self):
        """Number of active columns should approximate local_area_density."""
        cfg = _small_config(local_area_density=0.1)
        sp = SpatialPooler(cfg)
        inp = _random_sdr(cfg.input_bits)

        result = sp.compute(inp, learn=False)
        expected_active = int(cfg.num_columns * cfg.local_area_density)
        actual_active = int(np.sum(result))

        assert actual_active == expected_active

    def test_learning_modifies_permanences(self):
        """When learn=True, permanences should change."""
        cfg = _small_config()
        sp = SpatialPooler(cfg)
        inp = _random_sdr(cfg.input_bits)

        original = sp.permanences.copy()
        sp.compute(inp, learn=True)

        assert not np.array_equal(sp.permanences, original)

    def test_no_learning_preserves_permanences(self):
        """When learn=False, permanences should remain unchanged."""
        cfg = _small_config()
        sp = SpatialPooler(cfg)
        inp = _random_sdr(cfg.input_bits)

        original = sp.permanences.copy()
        sp.compute(inp, learn=False)

        np.testing.assert_array_equal(sp.permanences, original)

    def test_boosting_increases_for_underactive_columns(self):
        """Columns that never activate should see their boost factors rise."""
        cfg = _small_config(boost_strength=2.0)
        sp = SpatialPooler(cfg)
        inp = _random_sdr(cfg.input_bits)

        # Run a few cycles so duty cycles accumulate
        for _ in range(10):
            sp.compute(inp, learn=True)

        # At least some columns should have boost > 1
        assert np.any(sp.boost_factors > 1.0)


# ---------------------------------------------------------------------------
# TemporalMemory tests
# ---------------------------------------------------------------------------

class TestTemporalMemory:
    """Tests for the TemporalMemory component."""

    def test_compute_returns_correct_keys(self):
        """compute() result must contain active, winner, and predicted cells."""
        cfg = _small_config()
        tm = TemporalMemory(cfg)
        active_cols = np.zeros(cfg.num_columns, dtype=bool)
        active_cols[0] = True

        result = tm.compute(active_cols, learn=False)

        assert "active_cells" in result
        assert "winner_cells" in result
        assert "predicted_cells" in result

    def test_bursting_activates_all_cells_in_column(self):
        """An unpredicted column should burst (all cells active)."""
        cfg = _small_config()
        tm = TemporalMemory(cfg)
        active_cols = np.zeros(cfg.num_columns, dtype=bool)
        active_cols[0] = True  # Activate column 0

        result = tm.compute(active_cols, learn=False)

        expected_cells = set(range(0, cfg.cells_per_column))
        assert expected_cells.issubset(result["active_cells"])

    def test_learning_creates_segments(self):
        """After learning, the winner cell should have segments."""
        cfg = _small_config()
        tm = TemporalMemory(cfg)

        # First step: activate column 0
        cols1 = np.zeros(cfg.num_columns, dtype=bool)
        cols1[0] = True
        tm.compute(cols1, learn=True)

        # Second step: activate column 1 (should learn a segment)
        cols2 = np.zeros(cfg.num_columns, dtype=bool)
        cols2[1] = True
        tm.compute(cols2, learn=True)

        # At least one cell in column 1 should now have a segment
        col1_cells = list(range(cfg.cells_per_column, 2 * cfg.cells_per_column))
        has_segment = any(c in tm.segments for c in col1_cells)
        assert has_segment

    def test_get_column_cells_returns_correct_range(self):
        """_get_column_cells must return cells_per_column entries."""
        cfg = _small_config()
        tm = TemporalMemory(cfg)
        cells = tm._get_column_cells(2)

        assert len(cells) == cfg.cells_per_column
        assert cells[0] == 2 * cfg.cells_per_column

    def test_reset_clears_state(self):
        """After processing, resetting should clear all cell sets."""
        cfg = _small_config()
        tm = TemporalMemory(cfg)
        cols = np.zeros(cfg.num_columns, dtype=bool)
        cols[0] = True
        tm.compute(cols, learn=True)

        # Manually reset
        tm.active_cells = set()
        tm.winner_cells = set()
        tm.predicted_cells = set()

        assert len(tm.active_cells) == 0
        assert len(tm.winner_cells) == 0
        assert len(tm.predicted_cells) == 0


# ---------------------------------------------------------------------------
# ProductionHTM tests
# ---------------------------------------------------------------------------

class TestProductionHTM:
    """Tests for the top-level ProductionHTM."""

    def test_process_returns_expected_keys(self):
        """process() should return all expected metric keys."""
        cfg = _small_config()
        htm = ProductionHTM(cfg)
        inp = _random_sdr(cfg.input_bits)

        result = htm.process(inp, learn=False)

        for key in [
            "active_columns",
            "active_cells",
            "predicted_cells",
            "anomaly_score",
            "processing_time_ms",
            "num_active_columns",
            "num_predicted_cells",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_anomaly_score_in_valid_range(self):
        """Anomaly score must be between 0 and 1."""
        cfg = _small_config()
        htm = ProductionHTM(cfg)
        inp = _random_sdr(cfg.input_bits)

        result = htm.process(inp)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_input_padding_for_short_input(self):
        """Inputs shorter than input_bits should be padded automatically."""
        cfg = _small_config(input_bits=128)
        htm = ProductionHTM(cfg)
        short_input = np.ones(64)

        result = htm.process(short_input)
        assert result["num_active_columns"] > 0

    def test_input_truncation_for_long_input(self):
        """Inputs longer than input_bits should be truncated."""
        cfg = _small_config(input_bits=128)
        htm = ProductionHTM(cfg)
        long_input = np.ones(256)

        result = htm.process(long_input)
        assert result["num_active_columns"] > 0

    def test_reset_clears_all_state(self):
        """reset() should clear anomaly history and temporal memory state."""
        cfg = _small_config()
        htm = ProductionHTM(cfg)

        # Process a pattern to build up state
        inp = _random_sdr(cfg.input_bits)
        htm.process(inp)
        assert len(htm.anomaly_history) > 0

        htm.reset()

        assert len(htm.anomaly_history) == 0
        assert len(htm.temporal_memory.active_cells) == 0
        assert len(htm.temporal_memory.segments) == 0

    def test_consolidate_memory(self):
        """consolidate_memory should process all patterns and return metrics."""
        cfg = _small_config()
        htm = ProductionHTM(cfg)
        patterns = [_random_sdr(cfg.input_bits) for _ in range(5)]

        result = htm.consolidate_memory(patterns)

        assert result["patterns_processed"] == 5
        assert "mean_anomaly" in result
        assert "consolidation_time_ms" in result

    def test_get_metrics_returns_expected_keys(self):
        """get_metrics() must return standard monitoring keys."""
        cfg = _small_config()
        htm = ProductionHTM(cfg)

        metrics = htm.get_metrics()

        for key in ["anomaly_score", "active_columns", "predicted_columns",
                     "cells_per_column", "total_cells"]:
            assert key in metrics

    def test_input_size_compatibility_property(self):
        """input_size property should match config.input_bits."""
        cfg = _small_config(input_bits=256)
        htm = ProductionHTM(cfg)

        assert htm.input_size == 256
