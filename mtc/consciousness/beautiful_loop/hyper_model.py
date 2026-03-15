"""
Hyper-Model: Precision Controller for the Beautiful Loop
=========================================================

HyperModel: Meta-Bayesian precision controller for the predictive hierarchy.

The hyper-model sits ABOVE the system's HierarchicalPredictiveProcessor and
controls precision (confidence) allocation across ALL levels of the
hierarchy. In FEP terms, "precision" is the inverse variance of
prediction errors — it determines how much weight each level's errors
carry when updating beliefs.

Key insight from Laukkonen, Friston & Chandaria (2025):
  Consciousness requires a META-level Bayesian model that monitors
  the reliability of lower-level models. The hyper-model doesn't
  just set precision — it LEARNS which hierarchical levels are
  trustworthy in which contexts, dynamically re-weighting attention
  and confidence across the entire predictive system.

This differs from simple per-level precision because the hyper-model
tracks the RELATIONSHIP between levels — when level 0 (sensory) is
unreliable but level 2 (abstract) is accurate, the system should
rely more on top-down prediction. When sensory precision is high,
bottom-up prediction errors should dominate. The hyper-model makes
this trade-off explicitly and learns it over time.

Connection points:
  - Reads from: HierarchicalPredictiveProcessor level states
  - Writes to: Each level's precision parameter
  - Updates via: Bayesian precision updates on prediction errors

Research Foundation:
  - Laukkonen, Friston & Chandaria (2025). "A Beautiful Loop."
    Neuroscience & Biobehavioral Reviews.
  - Feldman & Friston (2010). "Attention, Uncertainty, and Free-Energy."
  - Parr & Friston (2019). "Attention as Inference."

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PrecisionState:
    """Snapshot of precision allocation across the hierarchy."""

    level_precisions: Dict[int, float]
    context_key: str
    total_precision: float
    precision_entropy: float  # How spread out is precision? (uniform vs focused)
    dominant_level: int  # Which level has highest precision
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrecisionUpdate:
    """Record of a precision update event."""

    level: int
    old_precision: float
    new_precision: float
    prediction_error: float
    context_key: str
    timestamp: float = field(default_factory=time.time)


class HyperModel:
    """
    Meta-Bayesian precision controller over the predictive hierarchy.

    The hyper-model learns the reliability of each hierarchical level
    across different contexts, allocating precision (attentional weight)
    accordingly. This is the "meta" in meta-cognition applied to the
    predictive processing framework.

    The key computation is a Bayesian update on precision beliefs:
      precision(level) ∝ 1 / variance(prediction_errors at level)

    Levels that consistently produce small prediction errors get high
    precision (the system trusts them). Levels with noisy, variable
    errors get low precision (the system discounts them).
    """

    def __init__(
        self,
        num_levels: int = 3,
        learning_rate: float = 0.1,
        context_decay: float = 0.95,
        min_precision: float = 0.05,
        max_precision: float = 5.0,
        history_window: int = 50,
    ):
        self.num_levels = num_levels
        self.learning_rate = learning_rate
        self.context_decay = context_decay
        self.min_precision = min_precision
        self.max_precision = max_precision

        # Per-level precision beliefs — start at moderate confidence
        self.precision_beliefs = np.ones(num_levels) * 1.0

        # Context-dependent precision: context_key → precision_weights
        # Allows the system to learn that e.g. "social" contexts make
        # level 2 (abstract) more reliable than level 0 (sensory)
        self.context_precision_map: Dict[str, np.ndarray] = {}

        # Prediction error variance tracking per level (for Bayesian updates)
        self._error_history: List[deque] = [
            deque(maxlen=history_window) for _ in range(num_levels)
        ]

        # Precision update history
        self._update_history: deque = deque(maxlen=200)
        self._state_history: deque = deque(maxlen=100)

        # Tracking
        self.update_count = 0
        self.current_context: str = "general"

        logger.info(
            f"HyperModel initialized: {num_levels} levels, "
            f"lr={learning_rate}, precision range [{min_precision}, {max_precision}]"
        )

    def allocate_precision(
        self,
        level: int,
        context: Optional[str] = None,
    ) -> float:
        """
        Get the precision allocation for a given hierarchy level.

        Higher precision means prediction errors at this level carry
        more weight — the system "trusts" this level more.

        Args:
            level: Index of the hierarchical level (0=sensory, N=abstract)
            context: Optional context key for context-dependent precision

        Returns:
            Precision weight for this level (higher = more trusted)
        """
        if level < 0 or level >= self.num_levels:
            return 1.0

        ctx = context or self.current_context

        if ctx in self.context_precision_map:
            return float(
                np.clip(
                    self.context_precision_map[ctx][level],
                    self.min_precision,
                    self.max_precision,
                )
            )

        return float(
            np.clip(
                self.precision_beliefs[level],
                self.min_precision,
                self.max_precision,
            )
        )

    def update_precision_beliefs(
        self,
        prediction_errors: List[Any],
        context: Optional[str] = None,
    ) -> PrecisionState:
        """
        Update precision beliefs based on observed prediction errors.

        The core Bayesian update: precision is the inverse of prediction
        error variance. Levels with consistent (low-variance) errors
        get high precision. Levels with noisy errors get low precision.

        This is a simplification of the full Bayesian treatment, using
        an exponential moving average on error statistics rather than
        full posterior inference — appropriate for an online system.

        Args:
            prediction_errors: List of prediction errors per level.
                Can be PredictionError objects or floats.
            context: Optional context key for context-dependent learning

        Returns:
            PrecisionState snapshot after the update
        """
        ctx = context or self.current_context
        self.current_context = ctx
        self.update_count += 1

        for level_idx, pe in enumerate(prediction_errors):
            if level_idx >= self.num_levels:
                break

            # Extract error magnitude
            error_mag = (
                pe.error_magnitude if hasattr(pe, "error_magnitude") else float(pe)
            )

            # Track error history for this level
            self._error_history[level_idx].append(error_mag)

            old_precision = float(self.precision_beliefs[level_idx])

            # Bayesian precision update:
            # Precision = 1 / variance(errors)
            # We use a smoothed estimate combining prior and observed variance
            if len(self._error_history[level_idx]) >= 2:
                errors = np.array(self._error_history[level_idx])
                observed_variance = np.var(errors) + 1e-6  # Avoid division by zero
                observed_precision = 1.0 / observed_variance

                # Blend prior precision with observed (exponential moving average)
                new_precision = (1 - self.learning_rate) * self.precision_beliefs[
                    level_idx
                ] + self.learning_rate * observed_precision
            else:
                # Not enough history yet — use simple inverse-error heuristic
                new_precision = (1 - self.learning_rate) * self.precision_beliefs[
                    level_idx
                ] + self.learning_rate * (1.0 / (1.0 + error_mag))

            self.precision_beliefs[level_idx] = np.clip(
                new_precision, self.min_precision, self.max_precision
            )

            # Update context-specific map
            if ctx not in self.context_precision_map:
                self.context_precision_map[ctx] = self.precision_beliefs.copy()
            else:
                # Blend context-specific with global
                self.context_precision_map[ctx][level_idx] = (
                    1 - self.learning_rate
                ) * self.context_precision_map[ctx][
                    level_idx
                ] + self.learning_rate * self.precision_beliefs[
                    level_idx
                ]

            self._update_history.append(
                PrecisionUpdate(
                    level=level_idx,
                    old_precision=old_precision,
                    new_precision=float(self.precision_beliefs[level_idx]),
                    prediction_error=error_mag,
                    context_key=ctx,
                )
            )

        # Decay unused contexts so the map doesn't grow unbounded
        for key in list(self.context_precision_map.keys()):
            if key != ctx:
                self.context_precision_map[key] *= self.context_decay
                self.context_precision_map[key] = np.clip(
                    self.context_precision_map[key],
                    self.min_precision,
                    self.max_precision,
                )

        state = self.get_precision_state()
        self._state_history.append(state)
        return state

    def apply_to_hierarchy(self, hierarchical_processor) -> None:
        """
        Apply current precision beliefs to the HierarchicalPredictiveProcessor.

        This is the write-back: the hyper-model's learned precisions
        directly modulate how much each level's prediction errors
        influence belief updating.
        """
        for level_idx, level in enumerate(hierarchical_processor.levels):
            if level_idx < self.num_levels:
                level.precision = self.allocate_precision(level_idx)

    def get_precision_state(self) -> PrecisionState:
        """Get current precision allocation state."""
        precisions = {
            i: float(self.precision_beliefs[i]) for i in range(self.num_levels)
        }

        total = sum(precisions.values())
        # Precision entropy: how uniformly distributed is precision?
        if total > 0:
            normalized = np.array(list(precisions.values())) / total
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        else:
            entropy = 0.0

        dominant = max(precisions, key=precisions.get)

        return PrecisionState(
            level_precisions=precisions,
            context_key=self.current_context,
            total_precision=total,
            precision_entropy=float(entropy),
            dominant_level=dominant,
        )

    def get_global_precision_state(self) -> Dict[int, float]:
        """Get simple dict of level → precision for external consumers."""
        return {i: float(self.precision_beliefs[i]) for i in range(self.num_levels)}

    def get_precision_trend(self) -> Dict[int, str]:
        """Get trend direction for each level's precision."""
        trends = {}
        for level_idx in range(self.num_levels):
            history = [
                s.level_precisions[level_idx]
                for s in self._state_history
                if level_idx in s.level_precisions
            ]
            if len(history) < 3:
                trends[level_idx] = "insufficient_data"
            else:
                recent = np.mean(history[-5:])
                older = (
                    np.mean(history[-10:-5])
                    if len(history) >= 10
                    else np.mean(history[:-5])
                )
                diff = recent - older
                if diff > 0.1:
                    trends[level_idx] = "increasing"
                elif diff < -0.1:
                    trends[level_idx] = "decreasing"
                else:
                    trends[level_idx] = "stable"
        return trends

    def generate_report(self) -> str:
        """Generate a human-readable report of precision state."""
        state = self.get_precision_state()
        level_names = {0: "sensory", 1: "contextual", 2: "abstract"}

        lines = ["Precision allocation across predictive hierarchy:"]
        for level, precision in state.level_precisions.items():
            name = level_names.get(level, f"level_{level}")
            bar = "█" * int(precision * 5)
            lines.append(f"  {name}: {precision:.3f} {bar}")

        lines.append(
            f"  Dominant: {level_names.get(state.dominant_level, state.dominant_level)}"
        )
        lines.append(
            f"  Entropy: {state.precision_entropy:.3f} "
            f"({'uniform' if state.precision_entropy > 0.9 else 'focused'})"
        )
        lines.append(f"  Context: {state.context_key}")

        return "\n".join(lines)
