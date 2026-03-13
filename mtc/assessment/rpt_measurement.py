"""
Recurrent Processing Theory (RPT) Measurement
==============================================

Based on Victor Lamme (2006, 2010): "Towards a True Neural Stance on
Consciousness" -- Trends in Cognitive Sciences.

RPT proposes two types of recurrence that map to different aspects
of consciousness:

1. **Superficial recurrence** (local feedback within a module):
   - Feedforward sweep + local recurrent connections
   - Maps to phenomenal consciousness ("what it's like")
   - Measured via: SNN spike feedback patterns, LSM reservoir dynamics

2. **Deep recurrence** (feedback between distant modules via workspace):
   - Cross-module feedback loops through GWT broadcast
   - Maps to access consciousness (reportable, actionable)
   - Measured via: GWT broadcast -> module -> workspace cycles

Data sources:
   - SNN: Spike patterns, STDP-learned recurrent weights
   - LSM: Reservoir state dynamics, spectral radius
   - GWT: Broadcast coverage, re-entry patterns
   - Neural orchestrator: Cross-layer pipeline metrics

Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class RecurrenceType(Enum):
    """Classification of recurrence depth."""

    NONE = "none"
    SUPERFICIAL = "superficial"  # Local feedback -> phenomenal consciousness
    DEEP = "deep"  # Cross-module feedback -> access consciousness


@dataclass
class RecurrenceMetrics:
    """Metrics from a single RPT measurement."""

    # Superficial recurrence (within-module)
    local_recurrence_score: float  # 0-1: strength of local feedback
    snn_feedback_strength: float  # SNN recurrent connection strength
    lsm_reservoir_recurrence: float  # LSM reservoir dynamics score
    local_spike_reentry: float  # Proportion of spikes feeding back

    # Deep recurrence (cross-module)
    global_recurrence_score: float  # 0-1: strength of cross-module feedback
    broadcast_reentry_rate: float  # How often broadcast content re-enters workspace
    cross_module_feedback: float  # Strength of inter-module connections
    workspace_cycle_depth: int  # How many broadcast->process->reentry cycles

    # Classification
    recurrence_type: RecurrenceType  # Overall classification
    phenomenal_consciousness: float  # Estimated phenomenal consciousness (0-1)
    access_consciousness: float  # Estimated access consciousness (0-1)

    # Timing
    measurement_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class RPTMeasurement:
    """
    Measures recurrent processing patterns in the system's neural architecture.

    The architecture naturally supports RPT measurement because:
    - SNN has STDP-learned recurrent connections (superficial recurrence)
    - LSM reservoir is inherently recurrent (superficial recurrence)
    - GWT broadcasts to all modules, which can re-enter the workspace (deep recurrence)
    - The consciousness cycle itself is a recurrent loop

    The recurrent connections already exist -- we just need to measure them.
    """

    def __init__(
        self,
        local_recurrence_threshold: float = 0.3,
        global_recurrence_threshold: float = 0.4,
        history_window: int = 50,
    ):
        self.local_recurrence_threshold = local_recurrence_threshold
        self.global_recurrence_threshold = global_recurrence_threshold

        # History for tracking trends
        self._metrics_history: deque = deque(maxlen=history_window)
        self._local_scores: deque = deque(maxlen=history_window)
        self._global_scores: deque = deque(maxlen=history_window)

        self.measurement_count = 0

        logger.info(
            "RPTMeasurement initialized: "
            f"local_threshold={local_recurrence_threshold}, "
            f"global_threshold={global_recurrence_threshold}"
        )

    def measure_local_recurrence(
        self,
        snn_state: Optional[Dict[str, Any]] = None,
        lsm_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Measure superficial (local) recurrence from neural substrates.

        Superficial recurrence = feedback within individual processing modules.
        This corresponds to phenomenal consciousness in RPT.

        Args:
            snn_state: SNN processing result with spike patterns
            lsm_state: LSM reservoir state with dynamics

        Returns:
            Local recurrence score (0-1)
        """
        components = []

        # --- SNN recurrence ---
        if snn_state:
            snn_score = self._measure_snn_recurrence(snn_state)
            components.append(snn_score)

        # --- LSM reservoir recurrence ---
        if lsm_state:
            lsm_score = self._measure_lsm_recurrence(lsm_state)
            components.append(lsm_score)

        if not components:
            return 0.0

        return float(np.mean(components))

    def measure_global_recurrence(
        self,
        workspace_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Measure deep (global) recurrence from workspace dynamics.

        Deep recurrence = feedback loops between distant modules via
        the global workspace broadcast. This corresponds to access
        consciousness in RPT.

        Args:
            workspace_state: GWT workspace statistics

        Returns:
            Global recurrence score (0-1)
        """
        if not workspace_state:
            return 0.0

        return self._measure_workspace_recurrence(workspace_state)

    def classify_recurrence_depth(
        self,
        local_score: float,
        global_score: float,
    ) -> RecurrenceType:
        """
        Classify the type of recurrence based on measured scores.

        Args:
            local_score: Superficial recurrence score
            global_score: Deep recurrence score

        Returns:
            RecurrenceType classification
        """
        has_local = local_score >= self.local_recurrence_threshold
        has_global = global_score >= self.global_recurrence_threshold

        if has_global:
            return RecurrenceType.DEEP
        elif has_local:
            return RecurrenceType.SUPERFICIAL
        else:
            return RecurrenceType.NONE

    def measure_full(
        self,
        snn_state: Optional[Dict[str, Any]] = None,
        lsm_state: Optional[Dict[str, Any]] = None,
        workspace_state: Optional[Dict[str, Any]] = None,
    ) -> RecurrenceMetrics:
        """
        Run full RPT measurement across all neural substrates.

        Args:
            snn_state: SNN processing result
            lsm_state: LSM reservoir state
            workspace_state: GWT workspace statistics

        Returns:
            Complete RecurrenceMetrics
        """
        start = time.time()
        self.measurement_count += 1

        # --- Superficial recurrence components ---
        snn_feedback = self._measure_snn_recurrence(snn_state) if snn_state else 0.0
        lsm_recurrence = self._measure_lsm_recurrence(lsm_state) if lsm_state else 0.0
        spike_reentry = self._measure_spike_reentry(snn_state) if snn_state else 0.0

        local_score = self.measure_local_recurrence(snn_state, lsm_state)

        # --- Deep recurrence components ---
        broadcast_reentry = 0.0
        cross_module = 0.0
        cycle_depth = 0

        if workspace_state:
            broadcast_reentry = self._measure_broadcast_reentry(workspace_state)
            cross_module = self._measure_cross_module_feedback(workspace_state)
            cycle_depth = self._measure_cycle_depth(workspace_state)

        global_score = self.measure_global_recurrence(workspace_state)

        # --- Classification ---
        recurrence_type = self.classify_recurrence_depth(local_score, global_score)

        # Phenomenal consciousness = local recurrence strength
        phenomenal = local_score

        # Access consciousness requires BOTH local AND global recurrence
        # (you need phenomenal experience to have access to it)
        access = min(local_score, global_score) * 0.5 + global_score * 0.5

        elapsed_ms = (time.time() - start) * 1000

        metrics = RecurrenceMetrics(
            local_recurrence_score=float(local_score),
            snn_feedback_strength=float(snn_feedback),
            lsm_reservoir_recurrence=float(lsm_recurrence),
            local_spike_reentry=float(spike_reentry),
            global_recurrence_score=float(global_score),
            broadcast_reentry_rate=float(broadcast_reentry),
            cross_module_feedback=float(cross_module),
            workspace_cycle_depth=cycle_depth,
            recurrence_type=recurrence_type,
            phenomenal_consciousness=float(phenomenal),
            access_consciousness=float(access),
            measurement_time_ms=elapsed_ms,
        )

        self._metrics_history.append(metrics)
        self._local_scores.append(local_score)
        self._global_scores.append(global_score)

        logger.debug(
            f"RPT measurement #{self.measurement_count}: "
            f"local={local_score:.3f}, global={global_score:.3f}, "
            f"type={recurrence_type.value}"
        )

        return metrics

    # ------------------------------------------------------------------
    # Internal measurement methods
    # ------------------------------------------------------------------

    def _measure_snn_recurrence(self, snn_state: Dict[str, Any]) -> float:
        """
        Measure recurrence in SNN spike patterns.

        SNN recurrence comes from:
        - STDP-learned weights creating feedback loops
        - Hidden layer spike patterns that re-excite earlier layers
        - Temporal correlation between successive spike trains
        """
        score = 0.0
        components = 0

        # Spike rate indicates active recurrent processing
        total_spikes = snn_state.get("total_spikes", 0)
        if total_spikes > 0:
            # More spikes relative to neurons = more recurrent activity
            # (feedback amplifies spike propagation)
            num_neurons = snn_state.get("num_neurons", 100)
            spike_density = min(1.0, total_spikes / max(1, num_neurons * 2))
            score += spike_density
            components += 1

        # Spike counts per layer -- recurrence shows as sustained
        # activity in hidden layers (not just feedforward pass)
        spike_counts = snn_state.get("spike_counts", [])
        if len(spike_counts) >= 2:
            # Hidden layers with sustained spikes indicate recurrence
            # In pure feedforward, hidden spikes decay quickly
            hidden_spikes = (
                spike_counts[1:-1] if len(spike_counts) > 2 else spike_counts[1:]
            )
            if hidden_spikes:
                total_hidden = sum(
                    s.sum().item() if hasattr(s, "sum") else float(s)
                    for s in hidden_spikes
                )
                input_spikes_val = (
                    spike_counts[0].sum().item()
                    if hasattr(spike_counts[0], "sum")
                    else float(spike_counts[0])
                )
                # Ratio of hidden to input spikes -- >1 means amplification (recurrence)
                if input_spikes_val > 0:
                    amplification = min(2.0, total_hidden / input_spikes_val)
                    score += min(1.0, amplification / 2.0)
                    components += 1

        # Activation pattern -- high activation variance = diverse processing
        activation = snn_state.get("activation_pattern")
        if activation is not None:
            if hasattr(activation, "std"):
                variance = float(activation.std())
            elif isinstance(activation, np.ndarray):
                variance = float(np.std(activation))
            else:
                variance = 0.0
            score += min(1.0, variance * 2)
            components += 1

        return float(score / max(1, components))

    def _measure_lsm_recurrence(self, lsm_state: Dict[str, Any]) -> float:
        """
        Measure recurrence in LSM reservoir dynamics.

        LSM is inherently recurrent -- it's a reservoir of recurrently
        connected neurons. We measure:
        - Spectral radius (edge of chaos = optimal recurrence)
        - Reservoir state complexity (rich dynamics = strong recurrence)
        - Fading memory (appropriate memory timescale)
        """
        score = 0.0
        components = 0

        # Spectral radius -- measures strength of recurrent connections
        # Optimal is near 1.0 (edge of chaos)
        spectral_radius = lsm_state.get("spectral_radius", 0.0)
        if spectral_radius > 0:
            # Score peaks at spectral_radius ~ 0.95 (edge of chaos)
            sr_score = 1.0 - abs(spectral_radius - 0.95) * 2
            score += max(0.0, sr_score)
            components += 1

        # Reservoir state complexity via entropy or variance
        reservoir_state = lsm_state.get("reservoir_state")
        if reservoir_state is not None:
            if isinstance(reservoir_state, np.ndarray):
                # Higher variance = richer dynamics
                state_var = float(np.var(reservoir_state))
                score += min(1.0, state_var * 5)
                components += 1

        # Active neurons -- more active = more recurrent processing
        active_ratio = lsm_state.get("active_neuron_ratio", 0.0)
        if active_ratio > 0:
            # Optimal is moderate activation (~30-70%)
            activity_score = 1.0 - 2 * abs(active_ratio - 0.5)
            score += max(0.0, activity_score)
            components += 1

        # Lyapunov exponent or edge-of-chaos indicator
        edge_of_chaos = lsm_state.get("edge_of_chaos_metric", 0.0)
        if edge_of_chaos > 0:
            score += min(1.0, edge_of_chaos)
            components += 1

        return float(score / max(1, components))

    def _measure_spike_reentry(self, snn_state: Dict[str, Any]) -> float:
        """
        Measure the proportion of spikes that feed back into the network.

        Spike re-entry = output spikes that re-excite input or hidden layers.
        This is the most direct measure of local recurrence.
        """
        # Check for explicit reentry data
        reentry_rate = snn_state.get("reentry_rate", None)
        if reentry_rate is not None:
            return float(min(1.0, reentry_rate))

        # Estimate from spike pattern: if output spikes correlate with
        # subsequent input spikes, there's feedback
        spike_counts = snn_state.get("spike_counts", [])
        if len(spike_counts) >= 2:
            # Use ratio of output to input as proxy
            output_val = (
                spike_counts[-1].sum().item()
                if hasattr(spike_counts[-1], "sum")
                else float(spike_counts[-1])
            )
            input_val = (
                spike_counts[0].sum().item()
                if hasattr(spike_counts[0], "sum")
                else float(spike_counts[0])
            )
            if input_val > 0:
                return float(min(1.0, output_val / (input_val + 1)))

        return 0.0

    def _measure_workspace_recurrence(self, workspace_state: Dict[str, Any]) -> float:
        """
        Measure deep recurrence through workspace dynamics.

        Deep recurrence = information flows from module -> workspace ->
        broadcast -> module -> workspace (re-entry).
        """
        score = 0.0
        components = 0

        # Broadcast coverage -- how widely information is shared
        broadcast = workspace_state.get("broadcast", {})
        coverage = broadcast.get("coverage_ratio", 0.0)
        if coverage > 0:
            score += min(1.0, coverage)
            components += 1

        # Cycle count -- more cycles = more recurrent processing
        cycle_count = workspace_state.get("cycle_count", 0)
        if cycle_count > 0:
            cycle_score = min(1.0, cycle_count / 10)
            score += cycle_score
            components += 1

        # Re-entry events -- content that was broadcast and returned
        reentry_count = workspace_state.get("reentry_count", 0)
        total_broadcasts = broadcast.get("total_broadcasts", 0)
        if total_broadcasts > 0:
            reentry_rate = min(1.0, reentry_count / total_broadcasts)
            score += reentry_rate
            components += 1

        # Cross-module influence -- different source modules contributing
        source_diversity = workspace_state.get("source_module_diversity", 0)
        if source_diversity > 0:
            diversity_score = min(1.0, source_diversity / 4)  # 4+ sources = max
            score += diversity_score
            components += 1

        # Ignition events -- non-linear amplification (hallmark of deep recurrence)
        ignition = workspace_state.get("ignition", {})
        ignitions = ignition.get("total_ignitions", 0)
        if ignitions > 0:
            ignition_score = min(1.0, ignitions / 5)
            score += ignition_score
            components += 1

        return float(score / max(1, components))

    def _measure_broadcast_reentry(self, workspace_state: Dict[str, Any]) -> float:
        """Measure how often broadcast content re-enters the workspace."""
        broadcast = workspace_state.get("broadcast", {})
        total = broadcast.get("total_broadcasts", 0)
        reentry = workspace_state.get("reentry_count", 0)

        if total <= 0:
            return 0.0
        return float(min(1.0, reentry / total))

    def _measure_cross_module_feedback(self, workspace_state: Dict[str, Any]) -> float:
        """Measure strength of inter-module connections via workspace."""
        # Source diversity indicates cross-module communication
        diversity = workspace_state.get("source_module_diversity", 0)
        # Number of modules that received broadcast
        receivers = workspace_state.get("broadcast", {}).get("receiving_modules", 0)

        if diversity <= 0 and receivers <= 0:
            return 0.0

        # Both diversity of sources AND receivers matters
        source_score = min(1.0, diversity / 4)
        receiver_score = min(1.0, receivers / 4)
        return float((source_score + receiver_score) / 2)

    def _measure_cycle_depth(self, workspace_state: Dict[str, Any]) -> int:
        """
        Measure how many broadcast->process->reentry cycles have occurred.

        Each full cycle where broadcast content is processed by modules
        and re-enters the workspace counts as one cycle of deep recurrence.
        """
        cycle_count = workspace_state.get("cycle_count", 0)
        reentry_count = workspace_state.get("reentry_count", 0)

        # Cycle depth = minimum of cycles run and re-entries detected
        # (a cycle without re-entry isn't truly deep recurrence)
        return min(cycle_count, reentry_count)

    # ------------------------------------------------------------------
    # Statistics and reporting
    # ------------------------------------------------------------------

    def get_average_scores(self) -> Dict[str, float]:
        """Get average local and global recurrence scores."""
        return {
            "average_local": (
                float(np.mean(list(self._local_scores))) if self._local_scores else 0.0
            ),
            "average_global": (
                float(np.mean(list(self._global_scores)))
                if self._global_scores
                else 0.0
            ),
            "measurement_count": self.measurement_count,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive RPT statistics."""
        avg = self.get_average_scores()

        # Count recurrence type distribution
        type_counts = {"none": 0, "superficial": 0, "deep": 0}
        for m in self._metrics_history:
            type_counts[m.recurrence_type.value] += 1

        return {
            **avg,
            "recurrence_type_distribution": type_counts,
            "latest": (
                {
                    "local": self._metrics_history[-1].local_recurrence_score,
                    "global": self._metrics_history[-1].global_recurrence_score,
                    "type": self._metrics_history[-1].recurrence_type.value,
                    "phenomenal": self._metrics_history[-1].phenomenal_consciousness,
                    "access": self._metrics_history[-1].access_consciousness,
                }
                if self._metrics_history
                else None
            ),
        }

    def generate_report(self) -> str:
        """Generate a human-readable RPT report."""
        stats = self.get_statistics()
        lines = ["RPT (Recurrent Processing Theory) Status:"]
        lines.append(f"  Measurements: {stats['measurement_count']}")
        lines.append(f"  Avg local recurrence: {stats['average_local']:.3f}")
        lines.append(f"  Avg global recurrence: {stats['average_global']:.3f}")

        dist = stats["recurrence_type_distribution"]
        total = sum(dist.values())
        if total > 0:
            lines.append("  Recurrence type distribution:")
            for rtype, count in dist.items():
                pct = count / total * 100
                lines.append(f"    {rtype}: {count} ({pct:.0f}%)")

        if stats["latest"]:
            lines.append(
                f"  Latest: type={stats['latest']['type']}, "
                f"phenomenal={stats['latest']['phenomenal']:.3f}, "
                f"access={stats['latest']['access']:.3f}"
            )

        return "\n".join(lines)
