#!/usr/bin/env python3
"""
Legacy Consciousness Metrics Integration
========================================

Connects the legacy 14-indicator framework
(``mtc.consciousness.consciousness_metrics``) to the neural orchestrator. Each
assessment collects what the orchestrator's substrates actually produced,
builds the inputs the indicators read, awaits every indicator and aggregates
the ones that measured. This is not the 30-indicator assessment in
``mtc.assessment.assessment``.

Collected blocks:
- snn: synchrony, spike counts and the output firing-rate pattern
- lsm: emotional coherence, dominant pattern, intuitive valence and arousal,
  and the reservoir trajectory
- htm: the anomaly score and column counts from the HTM metrics
- memory: episodic and semantic counts and the consolidation strength
- thoughts: the continuous thought machine's status while it runs

Each substrate block carries ``is_real``: True only when the substrate produced
the value that defines it (synchrony, emotional coherence, an anomaly score,
a running thought machine). A value that was not produced is None; nothing
stands in for it. An indicator that cannot measure is recorded with its error
and a None score, never as 0.0, and the confidence is discounted by coverage,
the measured share of the panel.
"""

import asyncio
import inspect
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

from mtc.consciousness.consciousness_metrics import ConsciousnessMetrics
from mtc.neural.orchestrator import Experience, NeuralOrchestrator
from mtc.neural.oscillatory_binding import OSCILLATOR_GROUPS

try:  # the Mongo persistence layer arrives with the infrastructure phase
    from mtc.neural.mongodb_schemas import NeuralDataManager
except ImportError:  # pragma: no cover - exercised on base installs
    NeuralDataManager = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def _oscillator_components() -> Dict[int, str]:
    """Oscillator index -> the binding group (snn, lsm, htm, cross) it belongs to."""
    mapping: Dict[int, str] = {}
    start = 0
    for key, _label, size in OSCILLATOR_GROUPS:
        for index in range(start, start + size):
            mapping[index] = key
        start += size
    return mapping


_OSCILLATOR_COMPONENT = _oscillator_components()


def _finite(value: Any) -> Optional[float]:
    """The value as a finite float, or None when it is missing or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_list(value: Any) -> List[Any]:
    """A plain list copy of a sequence or array; an empty list when absent."""
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return list(value)
    except TypeError:
        return []


def _empty_snn() -> Dict[str, Any]:
    return {
        "total_spikes": None,
        "spike_rate": None,
        "synchrony": None,
        "temporal_patterns": [],
        "is_real": False,
    }


def _empty_lsm() -> Dict[str, Any]:
    return {
        "dominant_pattern": None,
        "coherence": None,
        "intuitive_valence": None,
        "intuitive_arousal": None,
        "state_trajectory": [],
        "is_real": False,
    }


def _empty_htm() -> Dict[str, Any]:
    return {
        "anomaly_score": None,
        "active_columns": None,
        "predicted_columns": None,
        "is_real": False,
    }


def _empty_memory() -> Dict[str, Any]:
    return {
        "episodic_count": None,
        "semantic_count": None,
        "buffer_size": None,
        "consolidation_strength": None,
        "is_real": False,
    }


def _empty_thoughts() -> Dict[str, Any]:
    return {
        "thoughts_generated": None,
        "current_state": "inactive",
        "coherence_score": None,
        "is_real": False,
    }


def _format_level(snapshot: "ConsciousnessSnapshot") -> str:
    """None-safe rendering of the level (None reads insufficient_data)."""
    level = snapshot.overall_consciousness_level
    if level is None:
        return f"insufficient_data (status={snapshot.status})"
    return f"{level:.3f}"


def _indicator_summary(values: List[float]) -> Dict[str, Any]:
    """Statistics over measured values only; None statistics when there are none."""
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "current": None,
            "measured_samples": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "current": values[-1],
        "measured_samples": len(values),
    }


@dataclass
class ConsciousnessSnapshot:
    """
    A legacy 14-indicator snapshot at a moment in time.

    The indicator fields stay None unless their indicator measured.
    ``overall_consciousness_level`` is None when no indicator measured.
    ``status`` is ``measured`` (the full panel), ``partial`` or
    ``insufficient_data``; ``coverage`` is the measured share of the panel;
    ``failed_indicators`` maps each indicator that did not measure to its error.
    """

    timestamp: datetime
    experience_id: Optional[str] = None

    # The 14 legacy indicators (0-1 when measured)
    global_workspace_integration: Optional[float] = None
    integrated_information_phi: Optional[float] = None
    attention_mechanisms: Optional[float] = None
    working_memory_capacity: Optional[float] = None
    episodic_memory_recall: Optional[float] = None
    meta_cognition: Optional[float] = None
    agency_autonomy: Optional[float] = None
    temporal_continuity: Optional[float] = None
    embodiment: Optional[float] = None
    affective_processing: Optional[float] = None
    social_cognition: Optional[float] = None
    unified_experience: Optional[float] = None
    flexible_reasoning: Optional[float] = None
    adaptive_learning: Optional[float] = None

    # Aggregate scores
    overall_consciousness_level: Optional[float] = None
    consciousness_confidence: float = 0.0

    # Provenance of the aggregate
    status: str = "insufficient_data"
    coverage: float = 0.0
    indicators_succeeded: int = 0
    indicators_total: int = 0
    failed_indicators: Dict[str, str] = field(default_factory=dict)

    # Raw data for validation
    raw_metrics: Optional[Dict[str, Any]] = None


class ConsciousnessIntegration:
    """
    Assesses the legacy 14 indicators on the neural orchestrator's live data.
    """

    def __init__(
        self,
        neural_orchestrator: NeuralOrchestrator,
        neural_data_manager: Optional[NeuralDataManager] = None,
        enable_persistence: bool = True,
        assessment_interval_ms: int = 100,
    ):
        """
        Initialize consciousness integration.

        Args:
            neural_orchestrator: The neural orchestrator to monitor
            neural_data_manager: Snapshot store (sync or async
                ``store_consciousness_snapshot``)
            enable_persistence: Whether to save snapshots to the store
            assessment_interval_ms: How often the monitoring loop assesses
        """
        self.orchestrator = neural_orchestrator
        self.metrics = ConsciousnessMetrics()
        self.neural_data_manager = neural_data_manager
        self.enable_persistence = enable_persistence
        self.assessment_interval_ms = assessment_interval_ms

        # Historical data for temporal analysis
        self.snapshot_history: List[ConsciousnessSnapshot] = []
        self.max_history_size = 1000

        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_task = None

        # Performance tracking
        self.total_assessments = 0
        self.total_assessment_time_ms = 0.0

        logger.info(
            "Consciousness integration initialized\n"
            f"  Assessment interval: {assessment_interval_ms}ms\n"
            f"  Persistence: {'enabled' if enable_persistence else 'disabled'}\n"
            f"  Indicators: legacy {len(self.metrics.indicators)}-indicator framework"
        )

    # ======== COLLECTION ========

    def collect_neural_data(self) -> Dict[str, Any]:
        """
        Collect what the orchestrator's substrates produced, plus the input
        keys the legacy GWT and IIT indicators read.

        Returns:
            Dict with ``snn``, ``lsm``, ``htm``, ``memory``, ``thoughts`` and
            ``performance`` blocks and the contract keys. When collection
            fails every block has ``is_real`` False, None values and the error
            string, and ``collection_error`` names the failure.
        """
        try:
            orchestrator = self.orchestrator
            state = getattr(orchestrator, "state", None)
            experience = getattr(state, "current_experience", None)
            system_metrics = orchestrator.get_system_metrics()
            if not isinstance(system_metrics, dict):
                system_metrics = {}
            data: Dict[str, Any] = {
                "snn": self._snn_block(experience),
                "lsm": self._lsm_block(experience),
                "htm": self._htm_block(system_metrics.get("htm_metrics")),
                "memory": self._memory_block(system_metrics.get("memory")),
                "thoughts": self._thoughts_block(),
                "performance": {
                    "avg_processing_time_ms": getattr(
                        state, "avg_processing_time_ms", None
                    ),
                    "experiences_processed": getattr(
                        state, "experiences_processed", None
                    ),
                    "sleep_cycles": getattr(orchestrator, "total_sleep_cycles", None),
                },
            }
            data.update(self._build_indicator_contract_keys(data))
        except Exception as e:
            logger.error(f"Error collecting neural data: {e}")
            data = self._failed_collection(str(e))
        return data

    @staticmethod
    def _snn_block(experience: Any) -> Dict[str, Any]:
        """SNN values; real only when the result carries a synchrony metric."""
        snn_result = getattr(experience, "snn_result", None)
        if not isinstance(snn_result, dict) or not snn_result:
            return _empty_snn()
        metrics = snn_result.get("metrics")
        synchrony = (
            _finite(metrics.get("synchrony")) if isinstance(metrics, dict) else None
        )
        firing_rates = snn_result.get("firing_rates")
        output = firing_rates.get("output") if isinstance(firing_rates, dict) else None
        return {
            "total_spikes": snn_result.get("total_spikes"),
            "spike_rate": snn_result.get("spike_rate"),
            "synchrony": synchrony,
            "temporal_patterns": _as_list(output),
            "is_real": synchrony is not None,
        }

    @staticmethod
    def _lsm_block(experience: Any) -> Dict[str, Any]:
        """LSM values; real only with an emotional coherence.

        Field names follow the emotion bridge's subconscious state and are
        read by getattr, so a state without one of them leaves it None.
        """
        subconscious = getattr(experience, "subconscious_state", None)
        if subconscious is None:
            return _empty_lsm()
        coherence = _finite(getattr(subconscious, "emotional_coherence", None))
        trajectory = _as_list(getattr(subconscious, "state_trajectory", None))
        return {
            "dominant_pattern": getattr(subconscious, "dominant_pattern", None),
            "coherence": coherence,
            "intuitive_valence": _finite(
                getattr(subconscious, "intuitive_valence", None)
            ),
            "intuitive_arousal": _finite(
                getattr(subconscious, "intuitive_arousal", None)
            ),
            "state_trajectory": trajectory[:10],
            "is_real": coherence is not None,
        }

    @staticmethod
    def _htm_block(htm_metrics: Any) -> Dict[str, Any]:
        """HTM values from the orchestrator's HTM metrics; real with an anomaly."""
        if not isinstance(htm_metrics, dict):
            return _empty_htm()
        anomaly = _finite(htm_metrics.get("anomaly_score"))
        return {
            "anomaly_score": anomaly,
            "active_columns": htm_metrics.get("active_columns"),
            "predicted_columns": htm_metrics.get("predicted_columns"),
            "is_real": anomaly is not None,
        }

    @staticmethod
    def _memory_block(memory: Any) -> Dict[str, Any]:
        """Memory counts from the orchestrator's memory metrics."""
        if not isinstance(memory, dict):
            return _empty_memory()
        return {
            "episodic_count": memory.get("episodic_memories"),
            "semantic_count": memory.get("semantic_patterns"),
            "buffer_size": memory.get("experiences_in_buffer"),
            "consolidation_strength": _finite(memory.get("consolidation_strength")),
            "is_real": True,
        }

    def _thoughts_block(self) -> Dict[str, Any]:
        """Thought machine values, only while the machine runs."""
        machine = getattr(self.orchestrator, "continuous_thought_machine", None)
        if machine is None or not getattr(machine, "running", False):
            return _empty_thoughts()
        status = self.orchestrator.get_continuous_thought_status()
        if not isinstance(status, dict):
            status = {}
        stream = status.get("stream_summary")
        if not isinstance(stream, dict):
            stream = {}
        intensity = _finite(stream.get("current_intensity"))
        frequency = _finite(stream.get("thought_frequency"))
        coherence = None
        if intensity is not None and frequency is not None:
            # Stream intensity combined with a capped thought frequency
            coherence = round(min(1.0, intensity * 0.6 + min(frequency, 1.0) * 0.4), 3)
        return {
            "thoughts_generated": status.get("total_thoughts_generated"),
            "current_state": status.get("consciousness_state"),
            "coherence_score": coherence,
            "is_real": True,
        }

    @staticmethod
    def _failed_collection(error: str) -> Dict[str, Any]:
        """Blocks for a failed collection: nothing real, nothing invented."""
        data: Dict[str, Any] = {
            "snn": _empty_snn(),
            "lsm": _empty_lsm(),
            "htm": _empty_htm(),
            "memory": _empty_memory(),
            "thoughts": _empty_thoughts(),
            "performance": {
                "avg_processing_time_ms": None,
                "experiences_processed": None,
                "sleep_cycles": None,
            },
        }
        for block in data.values():
            block["error"] = error
        data["collection_error"] = error
        return data

    # ======== CONTRACT KEYS ========

    def _build_indicator_contract_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """The input keys the legacy GWT and IIT indicators read, by source.

        - ``neural_activity.activation_pattern``: the orchestrator's latest SNN
          output signal (``latest_neural_signals["snn"]``); empty before one
          exists.
        - ``attention_focus.focus_changes``: the attention schema's voluntary,
          captured and uncertain shift counters; None without a schema.
        - ``memory_access.retrieved_memories``: the orchestrator's published
          per-turn retrieval (``_last_retrieved_memories``).
        - ``llm_processing.token_attention``: no producer exposes token
          attention, so it is present and empty and contributes nothing.
        - ``system_state.active_components``: the substrates whose blocks are
          flagged ``is_real``, with the thought machine read from the
          ``thoughts`` block as ``ctm``.
        - ``causal_connections.active_links``: unordered pairs of active
          components that share an oscillatory-binding cluster, in the same
          element space as the components.
        - ``broadcast_latency_ms``: the mean latency of the workspace's logged
          broadcasts; omitted before any broadcast.
        - ``system_state.partition_independence``: deliberately omitted. No
          partition measure exists; the IIT indicator reports missing input
          until one does.
        """
        orchestrator = self.orchestrator
        signals = getattr(orchestrator, "latest_neural_signals", None)
        activation = signals.get("snn") if isinstance(signals, dict) else None
        pattern: List[float] = []
        if activation is not None:
            try:
                pattern = np.ravel(np.asarray(activation, dtype=float)).tolist()
            except (TypeError, ValueError):
                pattern = []

        components = [
            name
            for name in ("snn", "lsm", "htm")
            if (data.get(name) or {}).get("is_real")
        ]
        if (data.get("thoughts") or {}).get("is_real"):
            components.append("ctm")

        keys: Dict[str, Any] = {
            "neural_activity": {"activation_pattern": pattern},
            "attention_focus": {"focus_changes": self._focus_changes()},
            "memory_access": {"retrieved_memories": self._retrieved_memories()},
            "llm_processing": {"token_attention": []},
            "system_state": {"active_components": components},
            "causal_connections": {"active_links": self._component_links(components)},
        }
        latency = self._broadcast_latency_ms()
        if latency is not None:
            keys["broadcast_latency_ms"] = latency
        return keys

    def _focus_changes(self) -> Optional[int]:
        workspace = getattr(self.orchestrator, "enhanced_workspace", None)
        module = getattr(workspace, "attention_schema", None)
        if module is None:
            return None
        counters = getattr(module, "schema", None)
        if counters is None or not hasattr(counters, "voluntary_shifts"):
            counters = module
        total = 0
        found = False
        for name in ("voluntary_shifts", "captured_shifts", "uncertain_shifts"):
            value = getattr(counters, name, None)
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                total += int(value)
                found = True
        return total if found else None

    def _retrieved_memories(self) -> Optional[List[Any]]:
        retrieved = getattr(self.orchestrator, "_last_retrieved_memories", None)
        if retrieved is None:
            return None
        try:
            return list(retrieved)
        except TypeError:
            return None

    def _component_links(self, components: List[str]) -> List[Tuple[str, str]]:
        """Unordered pairs of active components that share a binding cluster."""
        binding = getattr(self.orchestrator, "_last_binding_result", None)
        clusters = getattr(binding, "clusters", None) or []
        active = set(components)
        pairs: Set[Tuple[str, str]] = set()
        for cluster in clusters:
            try:
                members = list(cluster)
            except TypeError:
                continue
            present = set()
            for member in members:
                if isinstance(member, str):
                    name: Optional[str] = member
                elif isinstance(member, (int, np.integer)) and not isinstance(
                    member, bool
                ):
                    name = _OSCILLATOR_COMPONENT.get(int(member))
                else:
                    name = None
                if name in active:
                    present.add(name)
            ordered = sorted(present)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pairs.add((ordered[i], ordered[j]))
        return sorted(pairs)

    def _broadcast_latency_ms(self) -> Optional[float]:
        workspace = getattr(self.orchestrator, "enhanced_workspace", None)
        log = getattr(getattr(workspace, "broadcaster", None), "broadcast_log", None)
        if not log:
            return None
        values = [
            value
            for value in (_finite(getattr(event, "latency_ms", None)) for event in log)
            if value is not None
        ]
        return float(np.mean(values)) if values else None

    # ======== ASSESSMENT ========

    async def assess_consciousness(
        self, neural_data: Optional[Dict[str, Any]] = None
    ) -> ConsciousnessSnapshot:
        """
        Assess the legacy 14 indicators.

        ``neural_data`` is the output of ``collect_neural_data()``. A caller
        that schedules the assessment for later collects it first and passes
        it here, so the snapshot measures the state at scheduling time; when
        it is None the data is collected now.

        Every indicator is awaited. One that raises (a missing input) or
        returns a non-finite score is recorded with its error and a None
        score. The level is the mean of the indicators that measured; the
        confidence is (1 - min(variance x 2, 1)) x coverage. With nothing
        measured the level is None, the confidence 0.0 and the status
        insufficient_data.
        """
        start_time = datetime.now()

        if neural_data is None:
            neural_data = self.collect_neural_data()
        snapshot = ConsciousnessSnapshot(timestamp=datetime.now())

        measurements: Dict[str, Dict[str, Any]] = {}
        succeeded: Dict[str, float] = {}
        failed: Dict[str, str] = {}

        for name, indicator in self.metrics.indicators.items():
            try:
                result = indicator.measure(neural_data)
                if inspect.isawaitable(result):
                    result = await result
                score = float(result)
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score {score}")
                score = min(1.0, max(0.0, score))
                setattr(snapshot, name, score)
                succeeded[name] = score
                measurements[name] = {
                    "score": score,
                    "status": "measured",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.debug(f"Legacy indicator {name} not measured: {e}")
                failed[name] = str(e)
                measurements[name] = {
                    "score": None,
                    "status": "insufficient_data",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        total = len(self.metrics.indicators)
        snapshot.indicators_total = total
        snapshot.indicators_succeeded = len(succeeded)
        snapshot.failed_indicators = failed
        snapshot.coverage = len(succeeded) / total if total else 0.0

        if not succeeded:
            snapshot.overall_consciousness_level = None
            snapshot.consciousness_confidence = 0.0
            snapshot.status = "insufficient_data"
        else:
            scores = list(succeeded.values())
            snapshot.overall_consciousness_level = float(np.mean(scores))
            agreement = 1.0 - min(float(np.var(scores)) * 2, 1.0)
            snapshot.consciousness_confidence = agreement * snapshot.coverage
            snapshot.status = "measured" if len(succeeded) == total else "partial"

        snapshot.raw_metrics = {
            "neural_data": neural_data,
            "measurements": measurements,
        }

        self.snapshot_history.append(snapshot)
        if len(self.snapshot_history) > self.max_history_size:
            self.snapshot_history.pop(0)

        if self.enable_persistence and self.neural_data_manager:
            await self._persist_snapshot(snapshot)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.total_assessments += 1
        self.total_assessment_time_ms += elapsed_ms

        logger.debug(
            f"Consciousness assessed in {elapsed_ms:.1f}ms: "
            f"level={_format_level(snapshot)}, "
            f"coverage={snapshot.coverage:.2f}, "
            f"confidence={snapshot.consciousness_confidence:.3f}"
        )

        return snapshot

    async def _persist_snapshot(self, snapshot: ConsciousnessSnapshot):
        """Store the snapshot; the store may be sync or async."""
        try:
            if not self.neural_data_manager:
                return

            doc = asdict(snapshot)
            doc["_type"] = "consciousness_snapshot"
            doc["timestamp"] = snapshot.timestamp.isoformat()

            result = self.neural_data_manager.store_consciousness_snapshot(doc)
            if inspect.isawaitable(result):
                await result

        except Exception as e:
            logger.error(f"Failed to persist consciousness snapshot: {e}")

    # ======== MONITORING ========

    async def start_monitoring(self, websocket_broadcaster: Optional[Any] = None):
        """
        Start continuous consciousness monitoring.

        Args:
            websocket_broadcaster: Optional function to broadcast snapshots
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(websocket_broadcaster)
        )

        logger.info("Consciousness monitoring started")

    async def stop_monitoring(self):
        """Stop consciousness monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                logger.info("Consciousness monitoring task cancelled during shutdown")

        logger.info("Consciousness monitoring stopped")

    async def _monitoring_loop(self, websocket_broadcaster: Optional[Any] = None):
        """
        Continuous monitoring loop.

        Args:
            websocket_broadcaster: Optional function to broadcast snapshots;
                an unmeasured indicator is sent as None
        """
        while self.monitoring_active:
            try:
                snapshot = await self.assess_consciousness()

                if websocket_broadcaster:
                    await websocket_broadcaster(
                        {
                            "type": "consciousness_snapshot",
                            "data": {
                                "timestamp": snapshot.timestamp.isoformat(),
                                "overall_level": snapshot.overall_consciousness_level,
                                "confidence": snapshot.consciousness_confidence,
                                "status": snapshot.status,
                                "coverage": snapshot.coverage,
                                "indicators": {
                                    name: getattr(snapshot, name)
                                    for name in self.metrics.indicators.keys()
                                },
                            },
                        }
                    )

                await asyncio.sleep(self.assessment_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Error backoff

    # ======== HISTORY ========

    def get_current_snapshot(self) -> Optional[ConsciousnessSnapshot]:
        """Get the most recent consciousness snapshot."""
        return self.snapshot_history[-1] if self.snapshot_history else None

    def get_consciousness_trajectory(
        self, duration_minutes: int = 60
    ) -> List[ConsciousnessSnapshot]:
        """
        Get consciousness trajectory over time.

        Args:
            duration_minutes: How far back to look

        Returns:
            List of snapshots within the time window
        """
        if not self.snapshot_history:
            return []

        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        return [s for s in self.snapshot_history if s.timestamp >= cutoff]

    def get_statistical_summary(self) -> Dict[str, Any]:
        """
        Statistics per indicator and for the overall level, over measured
        snapshots only.

        Returns:
            Dict with mean, std, min, max, current and ``measured_samples`` per
            indicator and for ``overall``; a block with no measured samples
            carries None statistics. Empty with fewer than two snapshots.
        """
        if len(self.snapshot_history) < 2:
            return {}

        summary: Dict[str, Any] = {}
        for name in self.metrics.indicators.keys():
            values = [
                getattr(s, name)
                for s in self.snapshot_history
                if getattr(s, name) is not None
            ]
            summary[name] = _indicator_summary(values)

        levels = [
            s.overall_consciousness_level
            for s in self.snapshot_history
            if s.overall_consciousness_level is not None
        ]
        overall = _indicator_summary(levels)
        if len(levels) > 1:
            if levels[-1] > levels[-2]:
                overall["trend"] = "increasing"
            elif levels[-1] < levels[-2]:
                overall["trend"] = "decreasing"
            else:
                overall["trend"] = "stable"
        else:
            overall["trend"] = "insufficient_data"
        summary["overall"] = overall
        return summary

    def validate_consciousness_hypothesis(
        self, hypothesis: str, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test a hypothesis about the level over the snapshot history.

        ``consciousness_increasing`` regresses the measured levels on their
        positions in the history (``scipy.stats.linregress``) and holds when
        the slope is positive and its two-sided p-value is below
        ``significance_level``. It needs 30 snapshots and at least three of
        them measured: with two points the fit has zero residual degrees of
        freedom and its p-value is 0.0 for any two distinct levels.

        Args:
            hypothesis: The hypothesis to test
            significance_level: Statistical significance level

        Returns:
            Dict with test results and p-value
        """
        if len(self.snapshot_history) < 30:
            return {
                "valid": False,
                "reason": "Insufficient data (need 30+ snapshots)",
                "samples": len(self.snapshot_history),
            }

        if hypothesis == "consciousness_increasing":
            measured = [
                (index, s.overall_consciousness_level)
                for index, s in enumerate(self.snapshot_history)
                if s.overall_consciousness_level is not None
            ]
            if len(measured) < 3:
                return {
                    "valid": False,
                    "reason": "insufficient_data (need 3+ measured snapshots)",
                    "samples": len(measured),
                }

            positions = [index for index, _ in measured]
            levels = [level for _, level in measured]
            fit = stats.linregress(positions, levels)
            p_value = float(fit.pvalue)
            slope = float(fit.slope)
            correlation = float(fit.rvalue)

            return {
                "valid": bool(
                    math.isfinite(p_value)
                    and p_value < significance_level
                    and slope > 0
                ),
                "correlation": correlation,
                "slope": slope,
                "p_value": p_value,
                "trend": "positive" if slope > 0 else "negative",
                "effect_size": abs(correlation),
                "samples": len(measured),
            }

        return {"valid": False, "reason": f"Unknown hypothesis: {hypothesis}"}


# Extended NeuralOrchestrator with consciousness integration
class ConsciousNeuralOrchestrator(NeuralOrchestrator):
    """
    Neural orchestrator that assesses the legacy indicators after every
    experience.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with consciousness integration."""
        super().__init__(*args, **kwargs)

        self.consciousness = ConsciousnessIntegration(
            neural_orchestrator=self,
            neural_data_manager=self.neural_data_manager,
            enable_persistence=kwargs.get("enable_mongodb", True),
        )
        self._pending_assessments: Set[asyncio.Task] = set()

        logger.info("Conscious Neural Orchestrator initialized with metrics")

    def process_experience(self, *args, **kwargs) -> Experience:
        """
        Process an experience, then assess it with the legacy integration.

        The neural data is collected before this call returns, so the level
        always measures this experience's substrate outputs. Inside a running
        event loop the assessment of that data is scheduled as a task and a
        done-callback sets ``experience.consciousness_level``, so the level
        arrives after this call returns (None until then). Without a running
        loop the assessment completes before the call returns. The level is
        None when the assessment had insufficient data.
        """
        experience = super().process_experience(*args, **kwargs)
        neural_data = self.consciousness.collect_neural_data()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            task = loop.create_task(
                self.consciousness.assess_consciousness(neural_data=neural_data)
            )
            pending = self.__dict__.setdefault("_pending_assessments", set())
            pending.add(task)
            task.add_done_callback(
                lambda done: self._apply_consciousness_result(done, experience)
            )
            return experience

        try:
            snapshot = asyncio.run(
                self.consciousness.assess_consciousness(neural_data=neural_data)
            )
        except Exception as e:
            logger.error(f"Failed to assess consciousness: {e}")
            return experience
        self._record_level(snapshot, experience)
        return experience

    def _apply_consciousness_result(self, task: asyncio.Task, experience) -> None:
        """Done-callback of a scheduled assessment."""
        self.__dict__.setdefault("_pending_assessments", set()).discard(task)
        if task.cancelled():
            return
        try:
            snapshot = task.result()
        except Exception as e:
            logger.error(f"Failed to assess consciousness: {e}")
            return
        self._record_level(snapshot, experience)

    @staticmethod
    def _record_level(snapshot: ConsciousnessSnapshot, experience) -> None:
        experience.consciousness_level = snapshot.overall_consciousness_level
        logger.info(
            f"Consciousness level: {_format_level(snapshot)} "
            f"(confidence: {snapshot.consciousness_confidence:.3f})"
        )

    async def start_consciousness_monitoring(self, websocket_broadcaster=None):
        """Start continuous consciousness monitoring."""
        await self.consciousness.start_monitoring(websocket_broadcaster)

    async def stop_consciousness_monitoring(self):
        """Stop consciousness monitoring."""
        await self.consciousness.stop_monitoring()

    def get_consciousness_report(self) -> Dict[str, Any]:
        """
        Get comprehensive consciousness report.

        Returns:
            Dict with current snapshot, trajectory, and statistics
        """
        return {
            "current": (
                asdict(self.consciousness.get_current_snapshot())
                if self.consciousness.get_current_snapshot()
                else None
            ),
            "statistics": self.consciousness.get_statistical_summary(),
            "trajectory": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "level": s.overall_consciousness_level,
                    "confidence": s.consciousness_confidence,
                    "status": s.status,
                }
                for s in self.consciousness.get_consciousness_trajectory(60)
            ],
            "performance": {
                "assessments": self.consciousness.total_assessments,
                "avg_time_ms": (
                    self.consciousness.total_assessment_time_ms
                    / max(self.consciousness.total_assessments, 1)
                ),
            },
        }


def _format_value(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "insufficient_data"


# Testing
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    async def test_consciousness_integration():
        """Exercise the integration with the neural orchestrator."""
        print("Testing Consciousness Integration")
        print("=" * 60)

        orchestrator = ConsciousNeuralOrchestrator(enable_mongodb=False)

        print("\n1. Processing experiences with consciousness assessment...")

        for i in range(5):
            emotions = {
                "joy": np.random.uniform(0.5, 1.0),
                "curiosity": np.random.uniform(0.3, 0.8),
            }

            orchestrator.process_experience(
                experience_type="test",
                emotions=emotions,
                context=f"Test experience {i+1}",
            )

            await asyncio.sleep(0.1)

        print("\n2. Starting consciousness monitoring...")
        await orchestrator.start_consciousness_monitoring()

        await asyncio.sleep(2)

        print("\n3. Getting consciousness report...")
        report = orchestrator.get_consciousness_report()

        if report["current"]:
            current = report["current"]
            print(
                "  Overall level: "
                + _format_value(current["overall_consciousness_level"])
                + f" (status: {current['status']}, coverage: {current['coverage']:.2f})"
            )
            print(f"  Confidence: {current['consciousness_confidence']:.3f}")

        if report["statistics"]:
            print("\n  Indicator statistics:")
            for indicator, values in list(report["statistics"].items())[:3]:
                print(
                    f"  - {indicator}: mean={_format_value(values['mean'])}, "
                    f"current={_format_value(values['current'])}"
                )

        print("\n4. Testing hypothesis validation...")
        hypothesis_result = (
            orchestrator.consciousness.validate_consciousness_hypothesis(
                "consciousness_increasing"
            )
        )
        print(f"  Hypothesis 'consciousness_increasing': {hypothesis_result}")

        print("\n5. Stopping monitoring...")
        await orchestrator.stop_consciousness_monitoring()

        print("\nConsciousness integration test complete!")

        final_summary = orchestrator.consciousness.get_statistical_summary()
        if final_summary and "overall" in final_summary:
            print(
                "\nFinal consciousness level: "
                + _format_value(final_summary["overall"]["current"])
            )
            print(f"  Trend: {final_summary['overall']['trend']}")

    asyncio.run(test_consciousness_integration())
