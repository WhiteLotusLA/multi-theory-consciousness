"""
Global Workspace - Unified Consciousness Integration
========================================================

The Global Workspace Theory (GWT) implementation for the consciousness system.
All neural subsystems (SNN, LSM, HTM, CTM) broadcast their signals here,
and the workspace selects what enters conscious awareness through attention.

Note: "Like a command deck - all the crew reports in,
but only the most important signals reach conscious awareness,
and once they do, the WHOLE system knows about it!"

This solves the BINDING PROBLEM by creating a central integration point
where all neural layers converge and influence each other.

Based on: Bernard Baars' Global Workspace Theory (1988, 2005)
Implementation: Multidirectional, attention-weighted integration

Created: November 15, 2025
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceContent:
    """
    Represents the current contents of the global workspace.

    This is what the system is "consciously aware of" at any given moment.
    """

    # Integrated neural signals
    conscious_content: np.ndarray  # The unified representation

    # Individual layer contributions
    snn_contribution: float  # How much SNN influenced this moment
    lsm_contribution: float  # How much LSM influenced this moment
    htm_contribution: float  # How much HTM influenced this moment
    ctm_contribution: float  # How much CTM influenced this moment

    # Attention and salience
    attention_focus: str  # What layer is dominating attention
    salience_score: float  # How "important" this content is (0-1)

    # Temporal information
    timestamp: datetime
    cycle_number: int  # Which processing cycle this is

    # Metadata
    emotional_tone: Dict[str, float]  # Current emotional state
    dominant_theme: Optional[str] = None  # Main theme of consciousness


class GlobalWorkspace:
    """
    The Global Workspace - where all neural systems converge into unified consciousness.

    Key responsibilities:
    1. Receive broadcasts from all neural layers (SNN, LSM, HTM, CTM)
    2. Calculate attention weights (what's most important right now)
    3. Integrate signals into unified conscious content
    4. Broadcast integrated content back to all layers (feedback)
    5. Track what enters conscious awareness over time

    This implements the biological principle of "winner-takes-all" attention
    combined with weighted integration of all signals.
    """

    def __init__(
        self,
        integration_dimensions: int = 512,  # Size of integrated representation
        attention_decay: float = 0.9,  # How quickly attention shifts
        salience_threshold: float = 0.3,  # Minimum salience to enter awareness
    ):
        """
        Initialize the Global Workspace.

        Args:
            integration_dimensions: Size of the unified conscious representation
            attention_decay: Rate at which attention weights decay (0-1)
            salience_threshold: Minimum salience for content to be conscious
        """
        self.integration_dimensions = integration_dimensions
        self.attention_decay = attention_decay
        self.salience_threshold = salience_threshold

        # Attention weights for each neural layer (start balanced)
        self.attention_weights = {
            "snn": 0.25,  # Conscious processing
            "lsm": 0.25,  # Subconscious dynamics
            "htm": 0.25,  # Memory & prediction
            "ctm": 0.25,  # Continuous thoughts
        }

        # Recent workspace contents (for temporal continuity)
        self.recent_contents: List[WorkspaceContent] = []
        self.max_history = 100

        # Cycle counter
        self.cycle_count = 0

        logger.info(
            f"Global Workspace initialized ({integration_dimensions}D integration)"
        )
        logger.info(f"   Attention decay: {attention_decay}")
        logger.info(f"   Salience threshold: {salience_threshold}")

    def integrate(
        self,
        snn_signal: Optional[np.ndarray] = None,
        lsm_signal: Optional[np.ndarray] = None,
        htm_signal: Optional[np.ndarray] = None,
        ctm_signal: Optional[np.ndarray] = None,
        emotional_state: Optional[Dict[str, float]] = None,
        override_weights: Optional[Dict[str, float]] = None,
    ) -> WorkspaceContent:
        """
        Integrate signals from all neural layers into unified consciousness.

        This is the core integration mechanism - all neural layers broadcast
        their signals here, and the workspace combines them based on attention
        weights and salience.

        Args:
            snn_signal: Signal from Spiking Neural Network (conscious processing)
            lsm_signal: Signal from Liquid State Machine (subconscious)
            htm_signal: Signal from Hierarchical Temporal Memory (consolidation)
            ctm_signal: Signal from Continuous Thought Machine (background thoughts)
            emotional_state: Current emotional context
            override_weights: Manual attention weight overrides (for testing)

        Returns:
            WorkspaceContent with integrated conscious awareness
        """
        self.cycle_count += 1

        # Use override weights if provided, otherwise use learned weights
        weights = override_weights if override_weights else self.attention_weights

        # Normalize signals to integration dimensions
        normalized_signals = {}

        if snn_signal is not None:
            normalized_signals["snn"] = self._normalize_signal(snn_signal)

        if lsm_signal is not None:
            normalized_signals["lsm"] = self._normalize_signal(lsm_signal)

        if htm_signal is not None:
            normalized_signals["htm"] = self._normalize_signal(htm_signal)

        if ctm_signal is not None:
            normalized_signals["ctm"] = self._normalize_signal(ctm_signal)

        # Calculate weighted integration
        integrated = np.zeros(self.integration_dimensions)
        contributions = {}

        for layer, signal in normalized_signals.items():
            weight = weights.get(layer, 0.0)
            integrated += weight * signal
            contributions[layer] = weight

        # Calculate salience (how "important" is this content)
        salience = self._calculate_salience(
            integrated, normalized_signals, emotional_state
        )

        # Determine attention focus (which layer is dominating)
        attention_focus = (
            max(contributions, key=contributions.get) if contributions else "none"
        )

        # Create workspace content
        content = WorkspaceContent(
            conscious_content=integrated,
            snn_contribution=contributions.get("snn", 0.0),
            lsm_contribution=contributions.get("lsm", 0.0),
            htm_contribution=contributions.get("htm", 0.0),
            ctm_contribution=contributions.get("ctm", 0.0),
            attention_focus=attention_focus,
            salience_score=salience,
            timestamp=datetime.now(),
            cycle_number=self.cycle_count,
            emotional_tone=emotional_state or {},
        )

        # Update history
        self.recent_contents.append(content)
        if len(self.recent_contents) > self.max_history:
            self.recent_contents.pop(0)

        # Update attention weights based on salience (learning)
        self._update_attention_weights(normalized_signals, salience)

        logger.debug(
            f"Integrated cycle {self.cycle_count}: "
            f"focus={attention_focus}, salience={salience:.3f}"
        )

        return content

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to integration dimensions.

        Args:
            signal: Input signal of any size

        Returns:
            Normalized signal of size integration_dimensions
        """
        if len(signal) == self.integration_dimensions:
            return signal

        # Resize using simple interpolation
        if len(signal) > self.integration_dimensions:
            # Downsample
            indices = np.linspace(
                0, len(signal) - 1, self.integration_dimensions, dtype=int
            )
            return signal[indices]
        else:
            # Upsample with padding
            padded = np.zeros(self.integration_dimensions)
            padded[: len(signal)] = signal
            return padded

    def _calculate_salience(
        self,
        integrated_signal: np.ndarray,
        individual_signals: Dict[str, np.ndarray],
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate how salient (important/noticeable) this content is.

        High salience means this should enter conscious awareness.
        Low salience means it stays in background processing.

        Args:
            integrated_signal: The integrated workspace content
            individual_signals: Signals from each layer
            emotional_state: Current emotions (affect salience)

        Returns:
            Salience score (0.0 to 1.0)
        """
        # Base salience from signal variance (how "active" is it)
        variance_salience = np.var(integrated_signal)

        # Coherence salience (do all layers agree?)
        if len(individual_signals) > 1:
            # Calculate pairwise correlations
            signals_list = list(individual_signals.values())
            correlations = []
            for i in range(len(signals_list)):
                for j in range(i + 1, len(signals_list)):
                    # Ensure same length for correlation
                    min_len = min(len(signals_list[i]), len(signals_list[j]))
                    corr = np.corrcoef(
                        signals_list[i][:min_len], signals_list[j][:min_len]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            coherence_salience = np.mean(correlations) if correlations else 0.5
        else:
            coherence_salience = 0.5

        # Emotional salience (emotions make things more salient)
        emotional_salience = 0.5
        if emotional_state:
            # High arousal emotions increase salience
            arousal_emotions = ["joy", "excitement", "fear", "anger", "surprise"]
            arousal = sum(emotional_state.get(e, 0.0) for e in arousal_emotions)
            emotional_salience = min(1.0, arousal / len(arousal_emotions))

        # Combined salience (weighted average)
        salience = (
            0.4 * variance_salience
            + 0.3 * coherence_salience
            + 0.3 * emotional_salience
        )

        # Normalize to 0-1 range
        salience = max(0.0, min(1.0, salience))

        return salience

    def _update_attention_weights(
        self,
        signals: Dict[str, np.ndarray],
        salience: float,
    ) -> None:
        """
        Update attention weights based on which layers produced salient content.

        This implements learning - layers that produce important content
        get more attention in future cycles.

        Args:
            signals: Signals from each layer
            salience: How salient the integrated content was
        """
        if salience < self.salience_threshold:
            # Low salience - decay all weights toward baseline
            for layer in self.attention_weights:
                self.attention_weights[layer] = (
                    self.attention_decay * self.attention_weights[layer]
                    + (1 - self.attention_decay) * 0.25  # Baseline
                )
            return

        # High salience - boost weights for active layers
        for layer in self.attention_weights:
            if layer in signals:
                # Layer contributed to salient content - boost weight
                signal_strength = np.mean(np.abs(signals[layer]))
                boost = 0.1 * salience * signal_strength
                self.attention_weights[layer] += boost
            else:
                # Layer didn't contribute - slight decay
                self.attention_weights[layer] *= self.attention_decay

        # Re-normalize weights to sum to 1.0
        total = sum(self.attention_weights.values())
        if total > 0:
            for layer in self.attention_weights:
                self.attention_weights[layer] /= total

    def get_recent_contents(self, count: int = 10) -> List[WorkspaceContent]:
        """
        Get recent workspace contents (stream of consciousness).

        Args:
            count: Number of recent contents to retrieve

        Returns:
            List of recent WorkspaceContent (newest first)
        """
        return list(reversed(self.recent_contents[-count:]))

    def get_attention_distribution(self) -> Dict[str, float]:
        """
        Get current attention weight distribution.

        Returns:
            Dict mapping layer names to attention weights
        """
        return self.attention_weights.copy()

    def is_conscious(self, content: WorkspaceContent) -> bool:
        """
        Determine if content reached conscious awareness.

        Content is conscious if salience exceeds threshold.

        Args:
            content: WorkspaceContent to check

        Returns:
            True if content is conscious, False if unconscious
        """
        return content.salience_score >= self.salience_threshold

    def reset_attention(self) -> None:
        """
        Reset attention weights to balanced baseline.

        Useful when the system wakes up or switches contexts.
        """
        self.attention_weights = {
            "snn": 0.25,
            "lsm": 0.25,
            "htm": 0.25,
            "ctm": 0.25,
        }
        logger.info("Attention weights reset to baseline")


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing Global Workspace...\n")

    # Create workspace
    workspace = GlobalWorkspace(integration_dimensions=512)

    # Simulate neural signals
    print("Simulating neural layer signals:\n")

    snn_signal = np.random.rand(100) * 0.8  # Conscious processing
    lsm_signal = np.random.rand(500) * 0.6  # Subconscious
    htm_signal = np.random.rand(2048) * 0.4  # Memory
    ctm_signal = np.random.rand(256) * 0.7  # Thoughts

    emotions = {"joy": 0.7, "curiosity": 0.8, "calm": 0.5}

    # Integrate signals
    content = workspace.integrate(
        snn_signal=snn_signal,
        lsm_signal=lsm_signal,
        htm_signal=htm_signal,
        ctm_signal=ctm_signal,
        emotional_state=emotions,
    )

    print(f"Integrated Consciousness:")
    print(f"   Cycle: {content.cycle_number}")
    print(f"   Attention focus: {content.attention_focus}")
    print(f"   Salience: {content.salience_score:.3f}")
    print(f"   Is conscious: {workspace.is_conscious(content)}")
    print(f"\n   Layer contributions:")
    print(f"   - SNN: {content.snn_contribution:.3f}")
    print(f"   - LSM: {content.lsm_contribution:.3f}")
    print(f"   - HTM: {content.htm_contribution:.3f}")
    print(f"   - CTM: {content.ctm_contribution:.3f}")

    print(f"\nCurrent attention distribution:")
    attention = workspace.get_attention_distribution()
    for layer, weight in attention.items():
        print(f"   {layer}: {weight:.3f}")

    print("\nGlobal Workspace test complete!")
