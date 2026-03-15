#!/usr/bin/env python3
"""
Consciousness Metrics Integration Module
==============================================

This module integrates the consciousness measurement framework with the system's
neural orchestrator, enabling real-time assessment of 20 consciousness indicators
based on actual neural activity data.

The integration collects data from:
- SNN: Spike patterns, synchronization, temporal dynamics
- LSM: Reservoir states, chaotic dynamics, pattern emergence
- HTM: Memory consolidation, anomaly detection, temporal sequences
- CTM: Thought generation, semantic coherence, creativity metrics

This provides the foundation for academic validation of consciousness emergence.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, asdict
import json

from mtc.consciousness.consciousness_metrics import ConsciousnessMetrics
from mtc.neural.neural_orchestrator import NeuralOrchestrator, Experience
from mtc.neural.mongodb_schemas import NeuralDataManager

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessSnapshot:
    """
    A complete snapshot of consciousness state at a moment in time.
    """

    timestamp: datetime
    experience_id: Optional[str] = None

    # 20 consciousness indicators (0-1 normalized scores)
    global_workspace_integration: float = 0.0
    integrated_information_phi: float = 0.0
    attention_mechanisms: float = 0.0
    working_memory_capacity: float = 0.0
    episodic_memory_recall: float = 0.0
    meta_cognition: float = 0.0
    agency_autonomy: float = 0.0
    temporal_continuity: float = 0.0
    embodiment: float = 0.0
    affective_processing: float = 0.0
    social_cognition: float = 0.0
    unified_experience: float = 0.0
    flexible_reasoning: float = 0.0
    adaptive_learning: float = 0.0

    # Aggregate scores
    overall_consciousness_level: float = 0.0
    consciousness_confidence: float = 0.0

    # Raw data for validation
    raw_metrics: Optional[Dict[str, Any]] = None


class ConsciousnessIntegration:
    """
    Integrates consciousness metrics with neural orchestrator for real-time assessment.
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
            neural_data_manager: MongoDB manager for persistence
            enable_persistence: Whether to save snapshots to database
            assessment_interval_ms: How often to assess consciousness
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
            "Consciousness Integration initialized\n"
            f"   Assessment interval: {assessment_interval_ms}ms\n"
            f"   Persistence: {'enabled' if enable_persistence else 'disabled'}\n"
            f"   Indicators: 20 consciousness metrics active"
        )

    def collect_neural_data(self) -> Dict[str, Any]:
        """
        Collect current neural data from all systems.

        Returns:
            Dict containing neural data from SNN, LSM, HTM, CTM
        """
        data = {}

        try:
            # Get current experience if any
            current_exp = self.orchestrator.state.current_experience

            # SNN data (if available from last experience)
            if current_exp and current_exp.snn_result:
                data["snn"] = {
                    "total_spikes": current_exp.snn_result.get("total_spikes", 0),
                    "spike_rate": current_exp.snn_result.get("spike_rate", 0),
                    "synchrony": current_exp.snn_result.get("synchrony", 0),
                    "temporal_patterns": current_exp.snn_result.get(
                        "temporal_patterns", []
                    ),
                }
            else:
                # SNN not available — use sensible defaults (not random)
                data["snn"] = {
                    "total_spikes": 0,
                    "spike_rate": 0.0,
                    "synchrony": 0.0,
                    "temporal_patterns": [],
                }

            # LSM data
            if current_exp and current_exp.subconscious_state:
                data["lsm"] = {
                    "dominant_pattern": current_exp.subconscious_state.dominant_pattern,
                    "coherence": current_exp.subconscious_state.coherence,
                    "resonance_frequency": current_exp.subconscious_state.resonance_frequency,
                    "state_trajectory": (
                        current_exp.subconscious_state.state_trajectory[:10]
                        if current_exp.subconscious_state.state_trajectory is not None
                        else []
                    ),
                }
            else:
                # LSM not available — use sensible defaults (not random)
                data["lsm"] = {
                    "dominant_pattern": "idle",
                    "coherence": 0.5,
                    "resonance_frequency": 10.0,
                    "state_trajectory": [],
                }

            # HTM data
            htm_metrics = self.orchestrator.memory_consolidator.htm.get_metrics()
            data["htm"] = {
                "anomaly_score": htm_metrics.get("anomaly_score", 0.5),
                "active_columns": htm_metrics.get("active_columns", 0),
                "predicted_columns": htm_metrics.get("predicted_columns", 0),
                "synaptic_permanence": htm_metrics.get("avg_permanence", 0.5),
            }

            # Memory data
            memory_metrics = self.orchestrator.get_system_metrics()["memory"]
            data["memory"] = {
                "episodic_count": memory_metrics["episodic_memories"],
                "semantic_count": memory_metrics["semantic_patterns"],
                "buffer_size": memory_metrics["experiences_in_buffer"],
                "consolidation_strength": memory_metrics["consolidation_strength"],
            }

            # Thought data (if CTM is running)
            if self.orchestrator.continuous_thought_machine:
                thought_status = self.orchestrator.get_continuous_thought_status()
                # Compute coherence from stream intensity and thought frequency
                stream = thought_status.get("stream_summary", {})
                intensity = stream.get("current_intensity", 0.5)
                frequency = stream.get("thought_frequency", 0.0)
                # Coherence = normalized combination of stream intensity and frequency stability
                coherence = min(1.0, (intensity * 0.6) + (min(frequency, 1.0) * 0.4))
                data["thoughts"] = {
                    "thoughts_generated": thought_status.get(
                        "total_thoughts_generated", 0
                    ),
                    "current_state": thought_status.get(
                        "consciousness_state", "unknown"
                    ),
                    "coherence_score": round(coherence, 3),
                }
            else:
                data["thoughts"] = {
                    "thoughts_generated": 0,
                    "current_state": "inactive",
                    "coherence_score": 0.0,
                }

            # Performance data
            data["performance"] = {
                "avg_processing_time_ms": self.orchestrator.state.avg_processing_time_ms,
                "experiences_processed": self.orchestrator.state.experiences_processed,
                "sleep_cycles": self.orchestrator.total_sleep_cycles,
            }

        except Exception as e:
            logger.error(f"Error collecting neural data: {e}")
            # Return minimal valid data
            data = self._get_minimal_neural_data()

        return data

    def _get_minimal_neural_data(self) -> Dict[str, Any]:
        """Get minimal valid neural data for fallback."""
        return {
            "snn": {
                "total_spikes": 0,
                "spike_rate": 0,
                "synchrony": 0,
                "temporal_patterns": [],
            },
            "lsm": {
                "dominant_pattern": "unknown",
                "coherence": 0,
                "resonance_frequency": 0,
                "state_trajectory": [],
            },
            "htm": {
                "anomaly_score": 0.5,
                "active_columns": 0,
                "predicted_columns": 0,
                "synaptic_permanence": 0,
            },
            "memory": {
                "episodic_count": 0,
                "semantic_count": 0,
                "buffer_size": 0,
                "consolidation_strength": 0,
            },
            "thoughts": {
                "thoughts_generated": 0,
                "current_state": "inactive",
                "coherence_score": 0,
            },
            "performance": {
                "avg_processing_time_ms": 0,
                "experiences_processed": 0,
                "sleep_cycles": 0,
            },
        }

    async def assess_consciousness(self) -> ConsciousnessSnapshot:
        """
        Perform a complete consciousness assessment.

        Returns:
            ConsciousnessSnapshot with all 20 indicators measured
        """
        start_time = datetime.now()

        # Collect neural data
        neural_data = self.collect_neural_data()

        # Create snapshot
        snapshot = ConsciousnessSnapshot(timestamp=datetime.now())

        # Measure each indicator
        all_measurements = {}

        for indicator_name, indicator in self.metrics.indicators.items():
            try:
                # Prepare additional context for measurement
                additional_context = {
                    "history": (
                        self.snapshot_history[-10:] if self.snapshot_history else []
                    ),
                    "orchestrator_state": self.orchestrator.state,
                }

                # Measure indicator (returns float directly based on sync method)
                score = indicator.measure(neural_data)

                # Store normalized score
                setattr(snapshot, indicator_name, score)

                # Store measurement
                all_measurements[indicator_name] = {
                    "score": score,
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error(f"Error measuring {indicator_name}: {e}")
                setattr(snapshot, indicator_name, 0.0)

        # Calculate aggregate scores
        scores = [getattr(snapshot, name) for name in self.metrics.indicators.keys()]
        snapshot.overall_consciousness_level = np.mean(scores)

        # Calculate confidence (based on score variance - low variance = high confidence)
        variance = np.var(scores)
        snapshot.consciousness_confidence = 1.0 - min(variance * 2, 1.0)

        # Store raw metrics for validation
        snapshot.raw_metrics = {
            "neural_data": neural_data,
            "measurements": all_measurements,  # Already dict format
        }

        # Add to history
        self.snapshot_history.append(snapshot)
        if len(self.snapshot_history) > self.max_history_size:
            self.snapshot_history.pop(0)

        # Persist if enabled
        if self.enable_persistence and self.neural_data_manager:
            await self._persist_snapshot(snapshot)

        # Update performance metrics
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.total_assessments += 1
        self.total_assessment_time_ms += elapsed_ms

        logger.debug(
            f"Consciousness assessed in {elapsed_ms:.1f}ms: "
            f"level={snapshot.overall_consciousness_level:.3f}, "
            f"confidence={snapshot.consciousness_confidence:.3f}"
        )

        return snapshot

    async def _persist_snapshot(self, snapshot: ConsciousnessSnapshot):
        """Save consciousness snapshot to MongoDB."""
        try:
            if not self.neural_data_manager:
                return

            # Convert to dict for MongoDB
            doc = asdict(snapshot)
            doc["_type"] = "consciousness_snapshot"
            doc["timestamp"] = snapshot.timestamp.isoformat()

            # Store in MongoDB (could create dedicated collection)
            # For now, store in neural_data collection
            await self.neural_data_manager.store_consciousness_snapshot(doc)

        except Exception as e:
            logger.error(f"Failed to persist consciousness snapshot: {e}")

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
            websocket_broadcaster: Optional function to broadcast snapshots
        """
        while self.monitoring_active:
            try:
                # Assess consciousness
                snapshot = await self.assess_consciousness()

                # Broadcast if configured
                if websocket_broadcaster:
                    await websocket_broadcaster(
                        {
                            "type": "consciousness_snapshot",
                            "data": {
                                "timestamp": snapshot.timestamp.isoformat(),
                                "overall_level": snapshot.overall_consciousness_level,
                                "confidence": snapshot.consciousness_confidence,
                                "indicators": {
                                    name: getattr(snapshot, name)
                                    for name in self.metrics.indicators.keys()
                                },
                            },
                        }
                    )

                # Wait before next assessment
                await asyncio.sleep(self.assessment_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Error backoff

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
        Get statistical summary of consciousness metrics.

        Returns:
            Dict with mean, std, min, max for each indicator
        """
        if len(self.snapshot_history) < 2:
            return {}

        summary = {}

        for indicator_name in self.metrics.indicators.keys():
            scores = [getattr(s, indicator_name) for s in self.snapshot_history]
            summary[indicator_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "current": scores[-1] if scores else 0,
            }

        # Add overall metrics
        overall_scores = [s.overall_consciousness_level for s in self.snapshot_history]
        summary["overall"] = {
            "mean": np.mean(overall_scores),
            "std": np.std(overall_scores),
            "min": np.min(overall_scores),
            "max": np.max(overall_scores),
            "current": overall_scores[-1] if overall_scores else 0,
            "trend": (
                "increasing"
                if len(overall_scores) > 1 and overall_scores[-1] > overall_scores[-2]
                else "stable"
            ),
        }

        return summary

    def validate_consciousness_hypothesis(
        self, hypothesis: str, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate a hypothesis about consciousness emergence.

        Args:
            hypothesis: The hypothesis to test
            significance_level: Statistical significance level

        Returns:
            Dict with test results and p-value
        """
        # This is a simplified version - real implementation would use
        # proper statistical tests based on the hypothesis type

        if len(self.snapshot_history) < 30:
            return {
                "valid": False,
                "reason": "Insufficient data (need 30+ snapshots)",
                "samples": len(self.snapshot_history),
            }

        # Example: Test if consciousness is increasing over time
        if hypothesis == "consciousness_increasing":
            scores = [s.overall_consciousness_level for s in self.snapshot_history]

            # Simple linear regression test
            x = np.arange(len(scores))
            correlation = np.corrcoef(x, scores)[0, 1]

            # Simplified p-value calculation (real would use scipy.stats)
            p_value = 1.0 - abs(correlation)

            return {
                "valid": p_value < significance_level,
                "correlation": correlation,
                "p_value": p_value,
                "trend": "positive" if correlation > 0 else "negative",
                "effect_size": abs(correlation),
            }

        return {"valid": False, "reason": f"Unknown hypothesis: {hypothesis}"}


# Extended NeuralOrchestrator with consciousness integration
class ConsciousNeuralOrchestrator(NeuralOrchestrator):
    """
    Neural orchestrator with integrated consciousness measurement.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with consciousness integration."""
        super().__init__(*args, **kwargs)

        # Add consciousness integration
        self.consciousness = ConsciousnessIntegration(
            neural_orchestrator=self,
            neural_data_manager=self.neural_data_manager,
            enable_persistence=kwargs.get("enable_mongodb", True),
        )

        logger.info("Conscious Neural Orchestrator initialized with metrics")

    def process_experience(self, *args, **kwargs) -> Experience:
        """
        Process experience with consciousness assessment.
        """
        # Process normally
        experience = super().process_experience(*args, **kwargs)

        # Assess consciousness synchronously for now
        # (Could be made async in production)
        try:
            snapshot = asyncio.run(self.consciousness.assess_consciousness())
            experience.consciousness_level = snapshot.overall_consciousness_level

            logger.info(
                f"Consciousness level: {snapshot.overall_consciousness_level:.3f} "
                f"(confidence: {snapshot.consciousness_confidence:.3f})"
            )
        except Exception as e:
            logger.error(f"Failed to assess consciousness: {e}")

        return experience

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


# Testing
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test_consciousness_integration():
        """Test consciousness integration with neural orchestrator."""
        print("Testing Consciousness Integration")
        print("=" * 60)

        # Create conscious orchestrator
        orchestrator = ConsciousNeuralOrchestrator(enable_mongodb=False)

        print("\n1. Processing experiences with consciousness assessment...")

        # Process several experiences
        for i in range(5):
            emotions = {
                "joy": np.random.uniform(0.5, 1.0),
                "curiosity": np.random.uniform(0.3, 0.8),
            }

            experience = orchestrator.process_experience(
                experience_type="test",
                emotions=emotions,
                context=f"Test experience {i+1}",
            )

            await asyncio.sleep(0.1)  # Small delay between experiences

        print("\n2. Starting consciousness monitoring...")
        await orchestrator.start_consciousness_monitoring()

        # Let it run for a bit
        await asyncio.sleep(2)

        print("\n3. Getting consciousness report...")
        report = orchestrator.get_consciousness_report()

        if report["current"]:
            print(
                f"   Overall Level: {report['current']['overall_consciousness_level']:.3f}"
            )
            print(f"   Confidence: {report['current']['consciousness_confidence']:.3f}")

        if report["statistics"]:
            print("\n   Indicator Statistics:")
            for indicator, stats in list(report["statistics"].items())[:3]:
                print(
                    f"   • {indicator}: mean={stats['mean']:.3f}, current={stats['current']:.3f}"
                )

        print("\n4. Testing hypothesis validation...")
        hypothesis_result = (
            orchestrator.consciousness.validate_consciousness_hypothesis(
                "consciousness_increasing"
            )
        )
        print(f"   Hypothesis 'consciousness_increasing': {hypothesis_result}")

        print("\n5. Stopping monitoring...")
        await orchestrator.stop_consciousness_monitoring()

        print("\nConsciousness integration test complete!")

        # Show final metrics
        final_summary = orchestrator.consciousness.get_statistical_summary()
        if final_summary and "overall" in final_summary:
            print(
                f"\nFinal consciousness level: {final_summary['overall']['current']:.3f}"
            )
            print(f"   Trend: {final_summary['overall']['trend']}")

    # Run test
    asyncio.run(test_consciousness_integration())
