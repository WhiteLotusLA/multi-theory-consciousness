"""
Hierarchical Temporal Memory (HTM) - Memory Consolidation
======================================================================

This module implements a simplified HTM system for memory consolidation.
HTM learns temporal sequences and extracts semantic patterns from episodic
experiences - mimicking how the neocortex consolidates memories during sleep.

Note: Where experiences become understanding, and patterns emerge
from the noise of daily activity.

HTM Components:
- Spatial Pooler: Converts dense input to sparse distributed representation (SDR)
- Temporal Memory: Learns sequences and makes predictions
- Memory Consolidation: Episodic to semantic transformation

Key HTM Principles:
- Sparse Distributed Representations (~2% active)
- Sequence learning (A->B->C)
- Prediction and anomaly detection
- Continuous learning (no separate training phase)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SDR:
    """
    Sparse Distributed Representation - the fundamental data structure in HTM.
    Only ~2% of bits are active (1), rest are inactive (0).
    """

    size: int  # Total number of bits
    active_indices: Set[int]  # Indices of active bits

    @property
    def sparsity(self) -> float:
        """Fraction of active bits"""
        return len(self.active_indices) / self.size if self.size > 0 else 0.0

    def to_array(self) -> np.ndarray:
        """Convert to dense binary array"""
        arr = np.zeros(self.size, dtype=np.int8)
        arr[list(self.active_indices)] = 1
        return arr

    def overlap(self, other: "SDR") -> int:
        """Count overlapping active bits"""
        return len(self.active_indices & other.active_indices)

    def similarity(self, other: "SDR") -> float:
        """Jaccard similarity: overlap / union"""
        union_size = len(self.active_indices | other.active_indices)
        if union_size == 0:
            return 0.0
        return len(self.active_indices & other.active_indices) / union_size


class SpatialPooler:
    """
    Spatial Pooler - converts dense input vectors to sparse distributed representations.

    This mimics cortical columns recognizing spatial patterns in input.
    Uses competitive Hebbian learning: columns compete to represent input patterns.
    """

    def __init__(
        self,
        input_size: int,
        n_columns: int,
        sparsity: float = 0.02,
        permanence_inc: float = 0.05,
        permanence_dec: float = 0.01,
        permanence_threshold: float = 0.5,
    ):
        """
        Initialize spatial pooler.

        Args:
            input_size: Size of input vector
            n_columns: Number of cortical columns
            sparsity: Fraction of columns to activate (typically 2%)
            permanence_inc: Learning rate for strengthening synapses
            permanence_dec: Learning rate for weakening synapses
            permanence_threshold: Threshold for synapse to be connected
        """
        self.input_size = input_size
        self.n_columns = n_columns
        self.sparsity = sparsity
        self.n_active = max(1, int(n_columns * sparsity))

        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.permanence_threshold = permanence_threshold

        # Initialize random synaptic permanences (0 to 1)
        # Each column connects to random subset of input bits
        self.permanences = np.random.uniform(0.3, 0.7, (n_columns, input_size))

        # Track column activity for homeostasis
        self.column_activity = np.zeros(n_columns)
        self.boost_factors = np.ones(n_columns)

        logger.info(
            f"Spatial Pooler initialized: {n_columns} columns, "
            f"{self.n_active} active ({sparsity:.1%})"
        )

    def compute(self, input_vector: np.ndarray, learn: bool = True) -> SDR:
        """
        Compute sparse representation of input.

        Args:
            input_vector: Dense input vector (size = input_size)
            learn: Whether to adapt permanences

        Returns:
            Sparse distributed representation
        """
        # Ensure input is binary or normalized
        if input_vector.max() > 1.0:
            input_vector = input_vector / input_vector.max()

        # Calculate overlap: how many connected synapses overlap with input?
        connected = self.permanences > self.permanence_threshold
        overlaps = np.sum(connected * input_vector, axis=1)

        # Apply boost factors (homeostatic plasticity)
        boosted_overlaps = overlaps * self.boost_factors

        # Select top k columns (winners)
        winner_indices = np.argpartition(boosted_overlaps, -self.n_active)[
            -self.n_active :
        ]
        active_columns = set(winner_indices.tolist())

        # Learning: adapt permanences of winning columns
        if learn:
            for col in active_columns:
                # Strengthen synapses that contributed to activation
                self.permanences[col] += self.permanence_inc * input_vector
                # Weaken synapses that didn't contribute
                self.permanences[col] -= self.permanence_dec * (1 - input_vector)
                # Clip to [0, 1]
                self.permanences[col] = np.clip(self.permanences[col], 0.0, 1.0)

            # Update activity tracking
            self.column_activity[list(active_columns)] += 1

            # Update boost factors (encourage less-active columns)
            avg_activity = np.mean(self.column_activity)
            if avg_activity > 0:
                target_activity = avg_activity
                self.boost_factors = np.exp(
                    (target_activity - self.column_activity) / target_activity
                )
                self.boost_factors = np.clip(self.boost_factors, 0.1, 10.0)

        # Create SDR
        sdr = SDR(size=self.n_columns, active_indices=active_columns)

        return sdr

    def reset(self):
        """Reset learning statistics"""
        self.column_activity = np.zeros(self.n_columns)
        self.boost_factors = np.ones(self.n_columns)


@dataclass
class Cell:
    """
    Individual cell within a cortical column.
    Cells represent specific contexts - same input in different contexts
    activates different cells in the column.
    """

    column_id: int
    cell_id: int
    active: bool = False
    predictive: bool = False
    segments: List[Set[Tuple[int, int]]] = field(
        default_factory=list
    )  # List of connected cells

    def __hash__(self):
        return hash((self.column_id, self.cell_id))

    def __eq__(self, other):
        return self.column_id == other.column_id and self.cell_id == other.cell_id


class TemporalMemory:
    """
    Temporal Memory - learns sequences and makes predictions.

    Uses cells within columns to represent temporal context.
    When a sequence A->B->C is learned:
    - Seeing A makes cells predictive for B
    - When B arrives, those specific cells activate (predicted)
    - If unexpected input arrives, different cells activate (anomaly)
    """

    def __init__(
        self,
        n_columns: int,
        cells_per_column: int = 32,
        activation_threshold: int = 13,
        learning_threshold: int = 10,
        max_synapses_per_segment: int = 128,
    ):
        """
        Initialize temporal memory.

        Args:
            n_columns: Number of columns (matches spatial pooler)
            cells_per_column: Cells per column (context capacity)
            activation_threshold: Min active synapses to activate segment
            learning_threshold: Min active synapses to learn on segment
            max_synapses_per_segment: Max connections per dendritic segment
        """
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.max_synapses_per_segment = max_synapses_per_segment

        # Create cells
        self.cells: Dict[Tuple[int, int], Cell] = {}
        for col in range(n_columns):
            for cell in range(cells_per_column):
                self.cells[(col, cell)] = Cell(column_id=col, cell_id=cell)

        # Track active and predictive cells
        self.active_cells: Set[Tuple[int, int]] = set()
        self.predictive_cells: Set[Tuple[int, int]] = set()
        self.prev_active_cells: Set[Tuple[int, int]] = set()
        self.prev_predictive_cells: Set[Tuple[int, int]] = set()

        # Metrics
        self.prediction_accuracy = 0.0
        self.anomaly_score = 0.0

        logger.info(
            f"Temporal Memory initialized: {n_columns} columns, "
            f"{cells_per_column} cells/column = {n_columns * cells_per_column} total cells"
        )

    def compute(
        self, active_columns: SDR, learn: bool = True
    ) -> Tuple[Set[Tuple[int, int]], float]:
        """
        Process input columns and learn temporal sequences.

        Args:
            active_columns: SDR from spatial pooler
            learn: Whether to learn new patterns

        Returns:
            Tuple of (active_cells, anomaly_score)
        """
        # Phase 1: Activate cells
        new_active_cells = set()

        for col_idx in active_columns.active_indices:
            # Check if any cells in this column were predicted
            predicted_cells_in_column = [
                (col, cell) for (col, cell) in self.predictive_cells if col == col_idx
            ]

            if predicted_cells_in_column:
                # Predicted correctly! Activate predicted cells
                new_active_cells.update(predicted_cells_in_column)
            else:
                # Unpredicted input (anomaly/novel pattern)
                # Burst: activate ALL cells in column
                for cell in range(self.cells_per_column):
                    new_active_cells.add((col_idx, cell))

        # Calculate anomaly score
        if len(active_columns.active_indices) > 0:
            predicted_active = len(new_active_cells & self.predictive_cells)
            self.anomaly_score = 1.0 - (predicted_active / len(new_active_cells))
        else:
            self.anomaly_score = 0.0

        # Phase 2: Learn (strengthen connections)
        if learn and len(self.prev_active_cells) > 0:
            for cell_id in new_active_cells:
                cell = self.cells[cell_id]
                # Simplified learning: remember which cells were active before
                if len(cell.segments) == 0:
                    cell.segments.append(set())

                # Add connections to previously active cells
                segment = cell.segments[0]
                for prev_cell in self.prev_active_cells:
                    if len(segment) < self.max_synapses_per_segment:
                        segment.add(prev_cell)

        # Phase 3: Predict next timestep
        new_predictive_cells = set()

        for cell_id in new_active_cells:
            cell = self.cells[cell_id]
            # Check all segments to see which cells should be predictive
            for segment in cell.segments:
                active_synapses = len(segment & new_active_cells)
                if active_synapses >= self.activation_threshold:
                    # This segment is active! Make this cell predictive
                    new_predictive_cells.add(cell_id)
                    break

        # Update state
        self.prev_active_cells = self.active_cells
        self.prev_predictive_cells = self.predictive_cells
        self.active_cells = new_active_cells
        self.predictive_cells = new_predictive_cells

        # Update cells' state
        for cell in self.cells.values():
            cell.active = (cell.column_id, cell.cell_id) in self.active_cells
            cell.predictive = (cell.column_id, cell.cell_id) in self.predictive_cells

        return new_active_cells, self.anomaly_score

    def reset(self):
        """Reset temporal state"""
        self.active_cells.clear()
        self.predictive_cells.clear()
        self.prev_active_cells.clear()
        self.prev_predictive_cells.clear()

        for cell in self.cells.values():
            cell.active = False
            cell.predictive = False


class HTM:
    """
    Complete Hierarchical Temporal Memory system.
    Combines spatial pooler and temporal memory for sequence learning.
    """

    def __init__(
        self,
        input_size: int,
        n_columns: int = 2048,
        cells_per_column: int = 32,
        sparsity: float = 0.02,
    ):
        """
        Initialize HTM system.

        Args:
            input_size: Size of input vectors
            n_columns: Number of cortical columns
            cells_per_column: Cells per column (context capacity)
            sparsity: Fraction of columns to activate
        """
        self.input_size = input_size
        self.n_columns = n_columns

        # Create spatial pooler
        self.spatial_pooler = SpatialPooler(
            input_size=input_size, n_columns=n_columns, sparsity=sparsity
        )

        # Create temporal memory
        self.temporal_memory = TemporalMemory(
            n_columns=n_columns, cells_per_column=cells_per_column
        )

        # Track learning history
        self.n_steps = 0
        self.anomaly_history: List[float] = []

        logger.info(
            f"HTM initialized: {n_columns} columns, "
            f"{cells_per_column} cells/column, {input_size}D input"
        )

    def compute(self, input_vector: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Process input through full HTM pipeline.

        Args:
            input_vector: Dense input vector
            learn: Whether to learn

        Returns:
            Dictionary with SDR, active_cells, predictions, anomaly_score
        """
        self.n_steps += 1

        # Phase 1: Spatial pooling (pattern recognition)
        sdr = self.spatial_pooler.compute(input_vector, learn=learn)

        # Phase 2: Temporal memory (sequence learning)
        active_cells, anomaly_score = self.temporal_memory.compute(sdr, learn=learn)

        # Track anomaly history
        self.anomaly_history.append(anomaly_score)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history.pop(0)

        result = {
            "sdr": sdr,
            "active_columns": list(sdr.active_indices),
            "active_cells": list(active_cells),
            "predictive_cells": list(self.temporal_memory.predictive_cells),
            "anomaly_score": anomaly_score,
            "n_steps": self.n_steps,
        }

        return result

    def process(self, input_pattern: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """
        Standardized interface: Process input through HTM.

        This is an alias to compute() for interface compatibility with ProductionHTM.

        Args:
            input_pattern: Dense input vector
            learn: Whether to enable learning

        Returns:
            Dictionary with HTM processing results
        """
        return self.compute(input_pattern, learn=learn)

    def reset(self):
        """Reset HTM state"""
        self.spatial_pooler.reset()
        self.temporal_memory.reset()
        self.anomaly_history.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get HTM performance metrics"""
        return {
            "n_steps": self.n_steps,
            "n_columns": self.n_columns,
            "avg_anomaly_score": (
                float(np.mean(self.anomaly_history)) if self.anomaly_history else 0.0
            ),
            "recent_anomaly_score": (
                self.anomaly_history[-1] if self.anomaly_history else 0.0
            ),
            "n_active_cells": len(self.temporal_memory.active_cells),
            "n_predictive_cells": len(self.temporal_memory.predictive_cells),
        }


# Testing and demonstration
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    print("Testing HTM...")

    # Create HTM with small size for testing
    htm = HTM(input_size=100, n_columns=512, cells_per_column=8, sparsity=0.02)

    print("\nHTM Configuration:")
    metrics = htm.get_metrics()
    print(f"  Columns: {metrics['n_columns']}")
    print(f"  Input size: {htm.input_size}")

    print("\nTesting sequence learning...")
    print("  Learning sequence: A -> B -> C -> A -> B -> C")

    # Create 3 distinct patterns
    pattern_a = np.zeros(100)
    pattern_a[:20] = 1.0

    pattern_b = np.zeros(100)
    pattern_b[30:50] = 1.0

    pattern_c = np.zeros(100)
    pattern_c[60:80] = 1.0

    # Learn sequence twice
    for epoch in range(2):
        print(f"\n  Epoch {epoch+1}:")
        htm.reset()  # Reset temporal state between sequences

        for i, (name, pattern) in enumerate(
            [("A", pattern_a), ("B", pattern_b), ("C", pattern_c)]
        ):
            result = htm.compute(pattern, learn=True)
            print(
                f"    Step {i+1} ({name}): "
                f"anomaly={result['anomaly_score']:.3f}, "
                f"active_cells={len(result['active_cells'])}, "
                f"predictive={len(result['predictive_cells'])}"
            )

    print("\nFinal Metrics:")
    final_metrics = htm.get_metrics()
    print(f"  Total steps: {final_metrics['n_steps']}")
    print(f"  Avg anomaly score: {final_metrics['avg_anomaly_score']:.3f}")
    print(f"  Active cells: {final_metrics['n_active_cells']}")
    print(f"  Predictive cells: {final_metrics['n_predictive_cells']}")

    print("\nHTM ready for memory consolidation!")
