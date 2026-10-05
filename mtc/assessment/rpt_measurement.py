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
   - SNN: Spike patterns, STDP statistics (``stdp_statistics`` in the SNN
     state, from the SNN's ``get_stdp_statistics()``)
   - LSM: Reservoir state dynamics, spectral radius
   - GWT: Broadcast coverage, re-entry patterns
   - Oscillatory binding: pushed snapshots or an attached binding module

Author: Multi-Theory Consciousness Contributors
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

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

    # Plasticity: min(1, 2 * mean STDP weight std) while STDP is active
    stdp_plasticity_score: float = 0.0

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
        stdp_plasticity_enabled: bool = True,
    ):
        """
        Args:
            local_recurrence_threshold: Local score at which recurrence counts
                as superficial.
            global_recurrence_threshold: Global score at which recurrence
                counts as deep.
            history_window: Number of measurements kept.
            stdp_plasticity_enabled: Count STDP plasticity as a component of
                SNN recurrence when the SNN state carries active STDP
                statistics. False reproduces v0.4.0.
        """
        self.local_recurrence_threshold = local_recurrence_threshold
        self.global_recurrence_threshold = global_recurrence_threshold
        self.stdp_plasticity_enabled = bool(stdp_plasticity_enabled)

        # Oscillatory binding: an attached module (pull) and the latest pushed
        # snapshot (push). The attribute name ``_oscillatory`` is kept because
        # the experiment runner wires the binding through it.
        self._oscillatory: Optional[Any] = None
        self._oscillatory_binding_stats: Optional[Dict[str, Any]] = None

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

        # --- Plasticity (recurrence with learning, Lamme 2006) ---
        stdp_plasticity = 0.0
        stdp = self._active_stdp_statistics(snn_state)
        if stdp is not None:
            stdp_plasticity = min(1.0, _finite(stdp.get("mean_weight_std")) * 2.0)

        # --- Classification ---
        recurrence_type = self.classify_recurrence_depth(local_score, global_score)

        # Phenomenal consciousness = local recurrence strength. Plasticity is
        # already a component of the local score (through the SNN term), so
        # it is not added here a second time.
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
            stdp_plasticity_score=float(stdp_plasticity),
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

        # STDP plasticity: recurrence with plasticity is the differentiator
        # RPT emphasises (Lamme 2006). Inert until the caller passes the SNN's
        # get_stdp_statistics() as snn_state["stdp_statistics"].
        stdp = self._active_stdp_statistics(snn_state)
        if stdp is not None:
            weight_std = _finite(stdp.get("mean_weight_std"))
            weight_diversity = _finite(stdp.get("weight_diversity"))
            score += min(1.0, weight_std * 2 + weight_diversity)
            components += 1

        return float(score / max(1, components))

    def _active_stdp_statistics(
        self, snn_state: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """The SNN's STDP statistics when plasticity counts, else None."""
        if not self.stdp_plasticity_enabled or not snn_state:
            return None
        stdp = snn_state.get("stdp_statistics")
        if not isinstance(stdp, dict) or not stdp.get("plasticity_active"):
            return None
        return stdp

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

    def set_oscillatory_binding_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        """Push the latest oscillatory binding snapshot.

        The snapshot carries ``global_order_parameter``, ``cross_coherence``
        (a dict of inter-group coherences) and ``binding_count``. None clears
        it, so the getter falls back to the attached binding.
        """
        self._oscillatory_binding_stats = dict(stats) if stats else None

    def attach_oscillatory_binding(self, oscillatory_binding: Any) -> None:
        """Attach an OscillatoryBinding module for the getter to read."""
        self._oscillatory = oscillatory_binding

    @property
    def oscillatory_binding(self) -> Optional[Any]:
        """The attached OscillatoryBinding module, if any (read-only)."""
        return self._oscillatory

    def get_oscillatory_binding_stats(self) -> Dict[str, Any]:
        """Oscillatory binding statistics for the binding-coherence indicator.

        A pushed snapshot wins and reports the latest values as pushed. Without
        one, the attached binding is read: ``global_order_parameter`` is then
        the mean Kuramoto order parameter over its history, not the latest
        value. Returns {} when neither source has data.

        Keys: global_order_parameter (0 to 1), cross_coherence (dict),
        binding_count (int).
        """
        if self._oscillatory_binding_stats:
            return dict(self._oscillatory_binding_stats)
        osc = self._oscillatory
        if osc is None:
            return {}
        history = getattr(osc, "_order_history", None)
        if not history:
            return {}
        return {
            "global_order_parameter": float(np.mean(list(history))),
            "cross_coherence": dict(getattr(osc, "_last_cross_coherence", {}) or {}),
            "binding_count": int(getattr(osc, "_binding_count", 0) or 0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize as {"scalars", "arrays", "histories"}."""
        return {
            "scalars": {"measurement_count": int(self.measurement_count)},
            "arrays": {},
            "histories": {
                "metrics_history": [
                    _metrics_to_record(m) for m in self._metrics_history
                ],
                "local_scores": [float(v) for v in self._local_scores],
                "global_scores": [float(v) for v in self._global_scores],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore typed metrics into deques that keep their configured maxlen.

        Malformed metric records and non-finite scores are skipped with a
        warning.
        """
        scalars = state.get("scalars", {}) or {}
        histories = state.get("histories", {}) or {}

        count = scalars.get("measurement_count")
        if isinstance(count, (int, float)) and math.isfinite(count) and count >= 0:
            self.measurement_count = int(count)

        if "metrics_history" in histories:
            restored = deque(maxlen=self._metrics_history.maxlen)
            skipped = 0
            for record in histories["metrics_history"] or []:
                metrics = _metrics_from_record(record)
                if metrics is None:
                    skipped += 1
                    continue
                restored.append(metrics)
            if skipped:
                logger.warning(
                    "RPTMeasurement restore skipped %d malformed metric records",
                    skipped,
                )
            self._metrics_history = restored

        for key, attr in (
            ("local_scores", "_local_scores"),
            ("global_scores", "_global_scores"),
        ):
            if key not in histories:
                continue
            current = getattr(self, attr)
            values = []
            skipped = 0
            for value in histories[key] or []:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(float(value))
                else:
                    skipped += 1
            if skipped:
                logger.warning(
                    "RPTMeasurement restore skipped %d non-finite %s", skipped, key
                )
            setattr(self, attr, deque(values, maxlen=current.maxlen))

    def generate_report(self) -> str:
        """Generate a human-readable RPT report."""
        stats = self.get_statistics()
        lines = ["RPT (Recurrent Processing Theory) Status:"]
        lines.append(f" Measurements: {stats['measurement_count']}")
        lines.append(f" Avg local recurrence: {stats['average_local']:.3f}")
        lines.append(f" Avg global recurrence: {stats['average_global']:.3f}")

        dist = stats["recurrence_type_distribution"]
        total = sum(dist.values())
        if total > 0:
            lines.append(" Recurrence type distribution:")
            for rtype, count in dist.items():
                pct = count / total * 100
                lines.append(f"   {rtype}: {count} ({pct:.0f}%)")

        if stats["latest"]:
            lines.append(
                f" Latest: type={stats['latest']['type']}, "
                f"phenomenal={stats['latest']['phenomenal']:.3f}, "
                f"access={stats['latest']['access']:.3f}"
            )

        return "\n".join(lines)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_METRIC_FLOAT_FIELDS = (
    "local_recurrence_score",
    "snn_feedback_strength",
    "lsm_reservoir_recurrence",
    "local_spike_reentry",
    "global_recurrence_score",
    "broadcast_reentry_rate",
    "cross_module_feedback",
    "phenomenal_consciousness",
    "access_consciousness",
    "stdp_plasticity_score",
    "measurement_time_ms",
    "timestamp",
)


def _finite(value: Any) -> float:
    """A finite float, or 0.0 for missing or non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _metrics_to_record(metrics: RecurrenceMetrics) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        name: float(getattr(metrics, name)) for name in _METRIC_FLOAT_FIELDS
    }
    record["workspace_cycle_depth"] = int(metrics.workspace_cycle_depth)
    record["recurrence_type"] = metrics.recurrence_type.value
    return record


def _metrics_from_record(record: Any) -> Optional[RecurrenceMetrics]:
    """Rehydrate a RecurrenceMetrics; None for a malformed record."""
    if not isinstance(record, dict):
        return None
    try:
        recurrence_type = RecurrenceType(record["recurrence_type"])
        values = {}
        for name in _METRIC_FLOAT_FIELDS:
            if name not in record:
                if name in ("stdp_plasticity_score", "measurement_time_ms"):
                    values[name] = 0.0
                    continue
                if name == "timestamp":
                    values[name] = time.time()
                    continue
                return None
            value = float(record[name])
            if not math.isfinite(value):
                return None
            values[name] = value
        depth = int(record.get("workspace_cycle_depth", 0))
    except (KeyError, TypeError, ValueError):
        return None
    return RecurrenceMetrics(
        workspace_cycle_depth=depth, recurrence_type=recurrence_type, **values
    )
