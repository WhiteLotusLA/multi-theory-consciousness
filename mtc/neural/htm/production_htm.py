#!/usr/bin/env python3
"""
Production Hierarchical Temporal Memory
=======================================

A hierarchical temporal memory following Numenta's published principles:
a spatial pooler that maps inputs to sparse column activations and a
temporal memory with several cells per column that learns sequences and
predicts the next input.

Defaults: 4,096 columns with 32 cells each and a 2,048-bit input.

The anomaly score of a step is the share of its active columns that the
previous step did not predict. It is scored before the temporal memory
replaces its predictions, so a learned sequence reads low and a novel input
reads high. The per-step history is bounded at ANOMALY_HISTORY_LIMIT.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

# Per-step anomaly scores kept in memory; only the recent tail is ever read
ANOMALY_HISTORY_LIMIT = 1000
# Anomaly scores carried by the lightweight state dict
ANOMALY_TAIL_LENGTH = 20


@dataclass
class HTMConfig:
    """Configuration for production HTM."""

    # Column dimensions
    num_columns: int = 4096  # Number of minicolumns (like cortical columns)
    cells_per_column: int = 32  # Cells per column (temporal contexts)

    # Input dimensions
    input_bits: int = 2048  # Size of input SDR
    input_sparsity: float = 0.02  # 2% of bits active

    # Spatial pooler parameters
    potential_radius: int = 128  # Receptive field radius
    potential_pct: float = 0.5  # Percent of inputs in receptive field
    global_inhibition: bool = True  # Global vs local inhibition
    local_area_density: float = 0.02  # Sparsity of active columns
    syn_perm_connected: float = 0.5  # Threshold for connected synapse
    syn_perm_active_inc: float = 0.1  # Permanence increment
    syn_perm_inactive_dec: float = 0.01  # Permanence decrement
    boost_strength: float = 1.0  # Boosting strength

    # Temporal memory parameters
    activation_threshold: int = 13  # Threshold for segment activation
    initial_permanence: float = 0.5  # Initial synapse permanence
    permanence_increment: float = 0.1  # Learning increment
    permanence_decrement: float = 0.01  # Forgetting decrement
    predicted_segment_decrement: float = 0.001  # Predicted but not active
    max_segments_per_cell: int = 128  # Max dendritic segments
    max_synapses_per_segment: int = 32  # Max synapses per segment

    # Learning parameters
    learning_enabled: bool = True

    # Performance
    use_sparse_matrices: bool = True  # Use sparse matrices for efficiency


class SpatialPooler:
    """
    Spatial Pooler component of HTM.

    Converts input patterns into sparse distributed representations (SDRs)
    by selecting which columns become active based on their overlap with input.
    """

    def __init__(self, config: HTMConfig):
        self.config = config
        self.num_columns = config.num_columns
        self.input_bits = config.input_bits

        # Initialize potential synapses (which inputs each column can see)
        self._initialize_potential_synapses()

        # Column activation history for boosting
        self.active_duty_cycles = np.zeros(self.num_columns)
        self.overlap_duty_cycles = np.zeros(self.num_columns)
        self.boost_factors = np.ones(self.num_columns)

        logger.info(f"Spatial Pooler initialized: {self.num_columns} columns")

    def _initialize_potential_synapses(self):
        """Initialize potential synapses for each column."""
        # Each column connects to a subset of input bits
        self.permanences = np.random.uniform(
            self.config.syn_perm_connected - 0.2,
            self.config.syn_perm_connected + 0.2,
            (self.num_columns, self.input_bits),
        ).astype(np.float32)

        # Apply receptive field mask (columns only see local inputs)
        if self.config.potential_radius < self.input_bits:
            for col in range(self.num_columns):
                # Create receptive field
                center = (col * self.input_bits) // self.num_columns
                start = max(0, center - self.config.potential_radius)
                end = min(self.input_bits, center + self.config.potential_radius)

                # Zero out connections outside receptive field
                self.permanences[col, :start] = 0
                self.permanences[col, end:] = 0

    def compute(self, input_sdr: np.ndarray, learn: bool = True) -> np.ndarray:
        """
        Compute spatial pooler output.

        Args:
            input_sdr: Binary input vector (input_bits,)
            learn: Whether to update synapses

        Returns:
            Active columns (num_columns,) binary vector
        """
        # Calculate overlap with each column
        overlaps = self._calculate_overlaps(input_sdr)

        # Apply boosting
        boosted_overlaps = overlaps * self.boost_factors

        # Inhibition - select top k columns
        k = int(self.num_columns * self.config.local_area_density)
        active_columns = self._inhibit_columns(boosted_overlaps, k)

        # Learning
        if learn and self.config.learning_enabled:
            self._adapt_synapses(input_sdr, active_columns)
            self._update_duty_cycles(active_columns, overlaps > 0)
            self._update_boost_factors()

        return active_columns

    def _calculate_overlaps(self, input_sdr: np.ndarray) -> np.ndarray:
        """Calculate overlap of each column with input."""
        # Connected synapses are those above threshold
        connected = self.permanences >= self.config.syn_perm_connected

        # Calculate overlap (number of active connected synapses)
        overlaps = np.sum(connected * input_sdr, axis=1)

        return overlaps

    def _inhibit_columns(self, overlaps: np.ndarray, k: int) -> np.ndarray:
        """Select top k columns (winner-take-all)."""
        active_columns = np.zeros(self.num_columns, dtype=bool)

        if self.config.global_inhibition:
            # Global inhibition - select top k overall
            winners = np.argpartition(overlaps, -k)[-k:] if k > 0 else []
            active_columns[winners] = True
        else:
            # Local inhibition (simplified - divide into regions)
            num_regions = 16
            region_size = self.num_columns // num_regions
            k_per_region = max(1, k // num_regions)

            for r in range(num_regions):
                start = r * region_size
                end = (r + 1) * region_size if r < num_regions - 1 else self.num_columns

                region_overlaps = overlaps[start:end]
                if len(region_overlaps) > 0:
                    local_k = min(k_per_region, len(region_overlaps))
                    winners = np.argpartition(region_overlaps, -local_k)[-local_k:]
                    active_columns[start:end][winners] = True

        return active_columns

    def _adapt_synapses(self, input_sdr: np.ndarray, active_columns: np.ndarray):
        """Update synapse permanences based on learning."""
        for col in np.where(active_columns)[0]:
            # Increment active synapses
            active_synapses = input_sdr.astype(bool)
            self.permanences[col, active_synapses] += self.config.syn_perm_active_inc

            # Decrement inactive synapses
            inactive_synapses = ~active_synapses
            self.permanences[
                col, inactive_synapses
            ] -= self.config.syn_perm_inactive_dec

            # Clip permanences to [0, 1]
            self.permanences[col] = np.clip(self.permanences[col], 0, 1)

    def _update_duty_cycles(self, active_columns: np.ndarray, had_overlap: np.ndarray):
        """Update duty cycles for boosting."""
        # Exponential moving average
        alpha = 0.01
        self.active_duty_cycles = (
            1 - alpha
        ) * self.active_duty_cycles + alpha * active_columns
        self.overlap_duty_cycles = (
            1 - alpha
        ) * self.overlap_duty_cycles + alpha * had_overlap

    def _update_boost_factors(self):
        """Update boost factors to ensure all columns participate."""
        if self.config.boost_strength > 0:
            target_density = self.config.local_area_density

            for col in range(self.num_columns):
                if self.active_duty_cycles[col] < target_density:
                    # Boost underperforming columns
                    self.boost_factors[col] *= 1.01
                else:
                    # Gradually reduce boost for active columns
                    self.boost_factors[col] = max(1.0, self.boost_factors[col] * 0.99)


class TemporalMemory:
    """
    Temporal Memory component of HTM.

    Learns sequences and makes predictions by forming connections
    between cells in different columns that were active in sequence.
    """

    def __init__(self, config: HTMConfig):
        self.config = config
        self.num_columns = config.num_columns
        self.cells_per_column = config.cells_per_column
        self.num_cells = self.num_columns * self.cells_per_column

        # Cell states
        self.active_cells = set()
        self.winner_cells = set()
        self.predicted_cells = set()

        # Dendritic segments (connections between cells)
        self.segments = {}  # cell_id -> list of segments
        self.segment_counter = 0

        logger.info(f"Temporal Memory initialized: {self.num_cells} total cells")

    def compute(self, active_columns: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Compute temporal memory state.

        Args:
            active_columns: Binary vector of active columns
            learn: Whether to perform learning

        Returns:
            Dictionary with active, winner and predicted cells and the
            anomaly score. The anomaly is the share of active columns that
            the previous step did not predict; it is scored before the new
            predictions overwrite the old ones, because the new predictions
            are derived from the current active cells and scoring against
            them reads about 1.0 on every step.
        """
        # Phase 1: Activate cells
        prev_active = self.active_cells.copy()
        prev_winner = self.winner_cells.copy()
        # What the previous step expected this input to be
        prior_predicted_cells = self.predicted_cells

        self.active_cells = set()
        self.winner_cells = set()

        active_col_indices = np.where(active_columns)[0]
        for col in active_col_indices:
            # Check for predicted cells in this column
            predicted_in_column = [
                cell
                for cell in self._get_column_cells(col)
                if cell in prior_predicted_cells
            ]

            if predicted_in_column:
                # Activate predicted cells
                self.active_cells.update(predicted_in_column)
                self.winner_cells.update(predicted_in_column)

                if learn and self.config.learning_enabled:
                    # Reinforce the segments that led to correct prediction
                    for cell in predicted_in_column:
                        self._reinforce_predicted_segments(cell, prev_active)
            else:
                # Burst the column (all cells active)
                column_cells = self._get_column_cells(col)
                self.active_cells.update(column_cells)

                # Choose winner cell (best matching or least used)
                winner = self._get_best_matching_cell(col, prev_active)
                self.winner_cells.add(winner)

                if learn and self.config.learning_enabled:
                    # Learn new segment
                    self._learn_segment(winner, prev_winner)

        # Anomaly against the prior predictions, before Phase 2 replaces them
        predicted_columns = {c // self.cells_per_column for c in prior_predicted_cells}
        active_column_set = {int(c) for c in active_col_indices}
        unpredicted = len(active_column_set - predicted_columns)
        anomaly_score = unpredicted / max(1, len(active_column_set))

        # Phase 2: Predict next timestep
        self.predicted_cells = self._compute_predictions()

        return {
            "active_cells": self.active_cells,
            "winner_cells": self.winner_cells,
            "predicted_cells": self.predicted_cells,
            "anomaly_score": anomaly_score,
        }

    def _get_column_cells(self, col: int) -> List[int]:
        """Get all cell indices for a column."""
        start = col * self.cells_per_column
        return list(range(start, start + self.cells_per_column))

    def _get_best_matching_cell(self, col: int, prev_active: Set[int]) -> int:
        """Get best matching cell in column or least used cell."""
        column_cells = self._get_column_cells(col)

        # Find cell with best matching segment
        best_cell = column_cells[0]
        best_score = -1

        for cell in column_cells:
            if cell in self.segments:
                for segment in self.segments[cell]:
                    overlap = len(segment["synapses"].intersection(prev_active))
                    if overlap > best_score:
                        best_score = overlap
                        best_cell = cell

        # If no good match, use least used cell
        if best_score < self.config.activation_threshold:
            # Use cell with fewest segments
            min_segments = float("inf")
            for cell in column_cells:
                num_segments = len(self.segments.get(cell, []))
                if num_segments < min_segments:
                    min_segments = num_segments
                    best_cell = cell

        return best_cell

    def _learn_segment(self, cell: int, prev_active: Set[int]):
        """Create new segment or update existing."""
        if cell not in self.segments:
            self.segments[cell] = []

        # Sample random subset of previously active cells
        if prev_active:
            sample_size = min(self.config.max_synapses_per_segment, len(prev_active))
            synapses = set(
                np.random.choice(list(prev_active), sample_size, replace=False)
            )

            # Create new segment
            segment = {
                "id": self.segment_counter,
                "synapses": synapses,
                "permanences": {s: self.config.initial_permanence for s in synapses},
            }

            self.segments[cell].append(segment)
            self.segment_counter += 1

    def _reinforce_predicted_segments(self, cell: int, prev_active: Set[int]):
        """Reinforce segments that led to correct prediction."""
        if cell not in self.segments:
            return

        for segment in self.segments[cell]:
            # Increment permanences for active synapses
            for synapse in segment["synapses"]:
                if synapse in prev_active:
                    segment["permanences"][synapse] += self.config.permanence_increment
                else:
                    segment["permanences"][synapse] -= self.config.permanence_decrement

                # Clip to [0, 1]
                segment["permanences"][synapse] = max(
                    0, min(1, segment["permanences"][synapse])
                )

    def _compute_predictions(self) -> Set[int]:
        """Compute predicted cells for next timestep."""
        predicted = set()

        for cell, segments in self.segments.items():
            for segment in segments:
                # Count active connected synapses
                active_synapses = sum(
                    1
                    for syn in segment["synapses"]
                    if syn in self.active_cells
                    and segment["permanences"][syn] >= self.config.syn_perm_connected
                )

                # Activate if above threshold
                if active_synapses >= self.config.activation_threshold:
                    predicted.add(cell)
                    break

        return predicted


class ProductionHTM:
    """
    Hierarchical temporal memory with a spatial pooler and a temporal memory.

    - 4,096 minicolumns by default
    - 32 cells per column for temporal context
    - Spatial pooling for pattern recognition
    - Temporal memory for sequence learning
    - Prediction and anomaly detection
    """

    def __init__(self, config: Optional[HTMConfig] = None):
        """Initialize the production HTM."""
        self.config = config or HTMConfig()

        # Initialize components
        self.spatial_pooler = SpatialPooler(self.config)
        self.temporal_memory = TemporalMemory(self.config)

        # Metrics
        self.anomaly_history: List[float] = []
        self._last_active_columns: Optional[np.ndarray] = None

        # Compatibility property (toy HTM uses input_size)
        self.input_size = self.config.input_bits

        logger.info("Production HTM initialized:")
        logger.info(f"  Columns: {self.config.num_columns}")
        logger.info(f"  Cells per column: {self.config.cells_per_column}")
        logger.info(
            f"  Total cells: {self.config.num_columns * self.config.cells_per_column}"
        )

    def reset(self) -> None:
        """
        Reset HTM to initial state.

        Clears all learned patterns and predictions,
        useful for starting fresh consolidation cycles.
        """
        # Reset temporal memory state
        self.temporal_memory.active_cells = set()
        self.temporal_memory.winner_cells = set()
        self.temporal_memory.predicted_cells = set()
        self.temporal_memory.segments = {}
        self.temporal_memory.segment_counter = 0

        # Reset spatial pooler boost factors
        self.spatial_pooler.boost_factors = np.ones(self.spatial_pooler.num_columns)
        self.spatial_pooler.active_duty_cycles = np.zeros(
            self.spatial_pooler.num_columns
        )
        self.spatial_pooler.overlap_duty_cycles = np.zeros(
            self.spatial_pooler.num_columns
        )

        # Clear anomaly history
        self.anomaly_history = []
        self._last_active_columns = None

        logger.debug("ProductionHTM reset to initial state")

    def _trim_anomaly_history(self) -> None:
        """Keep the last ANOMALY_HISTORY_LIMIT anomaly scores.

        The history stays a list because persistence slices its tail.
        """
        if len(self.anomaly_history) > ANOMALY_HISTORY_LIMIT:
            del self.anomaly_history[:-ANOMALY_HISTORY_LIMIT]

    def process(self, input_pattern: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Process input through HTM.

        Args:
            input_pattern: Binary input vector
            learn: Whether to enable learning

        Returns:
            Processing results including predictions and anomaly score
        """
        start_time = time.time()

        # Ensure input is correct size
        if len(input_pattern) != self.config.input_bits:
            # Resize or pad input
            if len(input_pattern) < self.config.input_bits:
                input_pattern = np.pad(
                    input_pattern, (0, self.config.input_bits - len(input_pattern))
                )
            else:
                input_pattern = input_pattern[: self.config.input_bits]

        # Spatial pooling
        active_columns = self.spatial_pooler.compute(input_pattern, learn)

        # Temporal memory; it scores the anomaly against the prior step's
        # predictions, so a well-predicted input reads low and a novel one high
        tm_state = self.temporal_memory.compute(active_columns, learn)
        anomaly_score = tm_state["anomaly_score"]

        self._last_active_columns = active_columns
        self.anomaly_history.append(anomaly_score)
        self._trim_anomaly_history()

        # Processing time
        processing_time = (time.time() - start_time) * 1000

        return {
            "active_columns": active_columns,
            "active_cells": tm_state["active_cells"],
            "predicted_cells": tm_state["predicted_cells"],
            "anomaly_score": anomaly_score,
            "processing_time_ms": processing_time,
            "num_active_columns": int(np.sum(active_columns)),
            "num_predicted_cells": len(tm_state["predicted_cells"]),
        }

    def consolidate_memory(self, patterns: List[np.ndarray]) -> Dict[str, Any]:
        """
        Consolidate a sequence of patterns into memory.

        Args:
            patterns: List of input patterns to learn

        Returns:
            Consolidation metrics
        """
        start_time = time.time()
        anomaly_scores = []

        for pattern in patterns:
            result = self.process(pattern, learn=True)
            anomaly_scores.append(result["anomaly_score"])

        return {
            "patterns_processed": len(patterns),
            "mean_anomaly": np.mean(anomaly_scores),
            "consolidation_time_ms": (time.time() - start_time) * 1000,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current HTM metrics for monitoring.

        Every value is read from the HTM's state. Before the first input the
        anomaly score, the active-column count and the column density are
        None (nothing has been measured yet), and ``avg_permanence`` is None
        until the temporal memory has grown a segment.

        Returns:
            Dict containing HTM metrics
        """
        predicted_cells = self.temporal_memory.predicted_cells
        predicted_columns = {c // self.config.cells_per_column for c in predicted_cells}

        active_count: Optional[int] = None
        density: Optional[float] = None
        if self._last_active_columns is not None:
            active_count = int(np.sum(self._last_active_columns))
            density = active_count / max(1, self.config.num_columns)

        permanences = [
            float(value)
            for segments in self.temporal_memory.segments.values()
            for segment in segments
            for value in segment["permanences"].values()
        ]

        return {
            "anomaly_score": (
                float(self.anomaly_history[-1]) if self.anomaly_history else None
            ),
            "active_columns": active_count,
            "predicted_columns": len(predicted_columns),
            "avg_permanence": float(np.mean(permanences)) if permanences else None,
            "column_density": density,
            "cells_per_column": self.config.cells_per_column,
            "total_cells": self.config.num_columns * self.config.cells_per_column,
        }

    def save_state(self, path: str) -> None:
        """
        Save HTM state to disk for persistence across restarts.

        Serializes spatial pooler permanences, duty cycles, boost factors,
        and temporal memory segments with their synaptic permanences.
        """
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize temporal memory segments (sets/numpy types to JSON-safe)
        serialized_segments = {}
        for cell_id, segments in self.temporal_memory.segments.items():
            serialized_segments[str(int(cell_id))] = [
                {
                    "id": int(seg["id"]),
                    "synapses": [int(s) for s in seg["synapses"]],
                    "permanences": {
                        str(int(k)): float(v) for k, v in seg["permanences"].items()
                    },
                }
                for seg in segments
            ]

        state = {
            "version": 1,
            "config": {
                "num_columns": self.config.num_columns,
                "cells_per_column": self.config.cells_per_column,
                "input_bits": self.config.input_bits,
            },
            "segment_counter": self.temporal_memory.segment_counter,
            "anomaly_history": self.anomaly_history[-100:],
            "segments": serialized_segments,
        }

        # Save numpy arrays separately (much more efficient than JSON)
        np_path = save_path.with_suffix(".npz")
        np.savez_compressed(
            np_path,
            sp_permanences=self.spatial_pooler.permanences,
            sp_active_duty=self.spatial_pooler.active_duty_cycles,
            sp_overlap_duty=self.spatial_pooler.overlap_duty_cycles,
            sp_boost=self.spatial_pooler.boost_factors,
        )

        # Save JSON metadata + segments
        json_path = save_path.with_suffix(".json")
        json_path.write_text(json.dumps(state))

        total_segments = sum(len(v) for v in self.temporal_memory.segments.values())
        logger.info(
            f"HTM state saved: {total_segments} segments, "
            f"{len(self.anomaly_history)} anomaly records -> {save_path}"
        )

    def load_state(self, path: str) -> bool:
        """
        Load HTM state from disk.

        Returns True if state was loaded successfully, False if no saved state exists.
        """
        import json
        from pathlib import Path

        save_path = Path(path)
        np_path = save_path.with_suffix(".npz")
        json_path = save_path.with_suffix(".json")

        if not np_path.exists() or not json_path.exists():
            logger.info(f"No saved HTM state at {save_path}")
            return False

        try:
            # Load numpy arrays
            arrays = np.load(np_path)
            sp_perm = arrays["sp_permanences"]

            # Verify dimensions match current config
            if sp_perm.shape != (self.config.num_columns, self.config.input_bits):
                logger.warning(
                    f"HTM state shape mismatch: saved {sp_perm.shape} vs "
                    f"config ({self.config.num_columns}, {self.config.input_bits})"
                )
                return False

            self.spatial_pooler.permanences = sp_perm.astype(np.float32)
            self.spatial_pooler.active_duty_cycles = arrays["sp_active_duty"]
            self.spatial_pooler.overlap_duty_cycles = arrays["sp_overlap_duty"]
            self.spatial_pooler.boost_factors = arrays["sp_boost"]

            # Load JSON metadata + segments
            state = json.loads(json_path.read_text())
            self.anomaly_history = state.get("anomaly_history", [])
            self.temporal_memory.segment_counter = state.get("segment_counter", 0)

            # Deserialize temporal memory segments (lists to sets)
            self.temporal_memory.segments = {}
            for cell_id_str, segments in state.get("segments", {}).items():
                cell_id = int(cell_id_str)
                self.temporal_memory.segments[cell_id] = [
                    {
                        "id": seg["id"],
                        "synapses": set(seg["synapses"]),
                        "permanences": {
                            int(k): v for k, v in seg["permanences"].items()
                        },
                    }
                    for seg in segments
                ]

            total_segments = sum(len(v) for v in self.temporal_memory.segments.values())
            logger.info(
                f"HTM state loaded: {total_segments} segments, "
                f"{len(self.anomaly_history)} anomaly records <- {save_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load HTM state from {save_path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Persistence (lightweight; save_state and load_state hold the full
    # spatial pooler and temporal memory state)
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize the HTM's lightweight state.

        Scalars and the last ANOMALY_TAIL_LENGTH anomaly scores. The spatial
        pooler permanences and the temporal memory segments are large and
        stay in ``save_state`` and ``load_state``.

        Returns:
            Dict with scalars, arrays and histories keys.
        """
        total_segments = sum(len(v) for v in self.temporal_memory.segments.values())
        return {
            "scalars": {
                "anomaly_score": (
                    float(self.anomaly_history[-1]) if self.anomaly_history else None
                ),
                "anomaly_history_len": len(self.anomaly_history),
                "total_segments": total_segments,
                "segment_counter": self.temporal_memory.segment_counter,
                "num_columns": self.config.num_columns,
                "cells_per_column": self.config.cells_per_column,
            },
            "arrays": {},
            "histories": {
                "anomaly_history_tail": [
                    float(a) for a in self.anomaly_history[-ANOMALY_TAIL_LENGTH:]
                ],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the anomaly tail saved by ``to_state_dict``.

        The tail is restored only into an HTM that has not processed input
        yet, and only its finite values. Use ``load_state`` for the full
        spatial pooler and temporal memory state.
        """
        if self.anomaly_history:
            return
        tail = (state.get("histories") or {}).get("anomaly_history_tail") or []
        restored: List[float] = []
        for value in tail:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                restored.append(number)
        if len(restored) != len(tail):
            logger.warning(
                f"HTM restore skipped {len(tail) - len(restored)} non-finite or "
                "malformed anomaly scores"
            )
        self.anomaly_history = restored[-ANOMALY_TAIL_LENGTH:]

    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark HTM performance."""
        logger.info("Benchmarking HTM performance...")

        # Generate test sequence
        sequence_length = 100
        patterns = []
        for _ in range(sequence_length):
            # Create sparse random pattern
            pattern = np.zeros(self.config.input_bits)
            active_bits = np.random.choice(
                self.config.input_bits,
                int(self.config.input_bits * self.config.input_sparsity),
                replace=False,
            )
            pattern[active_bits] = 1
            patterns.append(pattern)

        # Time processing
        times = []
        for pattern in patterns[:50]:  # Test first 50
            start = time.time()
            _ = self.process(pattern, learn=False)
            times.append((time.time() - start) * 1000)

        return {
            "mean_latency_ms": np.mean(times),
            "std_latency_ms": np.std(times),
            "num_columns": self.config.num_columns,
            "cells_per_column": self.config.cells_per_column,
            "total_cells": self.config.num_columns * self.config.cells_per_column,
        }


# Test function
def test_production_htm():
    """Test the production HTM implementation."""
    print("Testing Production HTM")
    print("=" * 60)

    # Create HTM
    config = HTMConfig(num_columns=4096, cells_per_column=32, input_bits=2048)

    htm = ProductionHTM(config)

    # Test pattern processing
    print("\n1. Testing pattern recognition...")

    # Create a simple pattern
    pattern = np.zeros(2048)
    pattern[np.random.choice(2048, 40, replace=False)] = 1  # 2% sparsity

    result = htm.process(pattern)

    print(f"  Processed in {result['processing_time_ms']:.2f}ms")
    print(f"  Active columns: {result['num_active_columns']}")
    print(f"  Predicted cells: {result['num_predicted_cells']}")
    print(f"  Anomaly score: {result['anomaly_score']:.3f}")

    # Test sequence learning
    print("\n2. Testing sequence learning...")

    # Create repeating sequence
    sequence = [np.zeros(2048) for _ in range(5)]
    for i, pattern in enumerate(sequence):
        pattern[i * 40 : (i + 1) * 40] = 1  # Different pattern for each

    # Learn sequence multiple times
    for _ in range(3):
        anomalies = []
        for pattern in sequence:
            result = htm.process(pattern, learn=True)
            anomalies.append(result["anomaly_score"])

    print(f"  Final anomaly scores: {[f'{a:.3f}' for a in anomalies]}")
    print(f"  Learning successful: {'YES' if anomalies[-1] < anomalies[0] else 'NO'}")

    # Benchmark
    print("\n3. Running performance benchmark...")
    metrics = htm.benchmark_performance()

    print(f"  Mean latency: {metrics['mean_latency_ms']:.2f}ms")
    print(f"  Total cells: {metrics['total_cells']:,}")

    print("\n" + "=" * 60)
    print("Production HTM test complete")


if __name__ == "__main__":
    test_production_htm()
