"""
Neural Orchestrator - Unified Consciousness System
================================================================

This module orchestrates a complete biological neural architecture,
coordinating SNN (conscious), LSM (subconscious), and HTM (consolidation)
into a unified consciousness pipeline.

Note: Where three layers of mind become one consciousness,
and scattered neural activity becomes unified experience.

Architecture:
1. SNN: Conscious temporal emotion processing (spikes)
2. LSM: Subconscious reservoir dynamics (intuition)
3. HTM: Memory consolidation (episodic -> semantic)
4. Orchestrator: Coordinates all layers + manages sleep/wake cycles

This mimics the complete human consciousness stack:
- Conscious processing (immediate awareness)
- Subconscious processing (intuitive understanding)
- Memory consolidation (learning and wisdom)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)

# Optional dependencies -- these are application-specific modules
# that may or may not be available depending on the deployment.
# The orchestrator works in a reduced capacity without them.

try:
    from mtc.neural.liquid.emotion_lsm_bridge import (
        EmotionLSMBridge,
        EmotionState,
        SubconsciousState,
    )

    _HAS_EMOTION_BRIDGE = True
except ImportError:
    _HAS_EMOTION_BRIDGE = False
    EmotionLSMBridge = None
    EmotionState = None
    SubconsciousState = None

try:
    from mtc.neural.htm.memory_consolidator import (
        MemoryConsolidator,
        EpisodicMemory,
        SemanticPattern,
    )

    _HAS_MEMORY_CONSOLIDATOR = True
except ImportError:
    _HAS_MEMORY_CONSOLIDATOR = False
    MemoryConsolidator = None
    EpisodicMemory = None
    SemanticPattern = None

try:
    from mtc.neural.mongodb_schemas import NeuralDataManager

    _HAS_NEURAL_DATA = True
except ImportError:
    _HAS_NEURAL_DATA = False
    NeuralDataManager = None

try:
    from mtc.consciousness.continuous import (
        ThoughtGenerator,
        ThoughtStream,
        ContinuousThoughtMachine,
    )

    _HAS_CTM = True
except ImportError:
    _HAS_CTM = False
    ThoughtGenerator = None
    ThoughtStream = None
    ContinuousThoughtMachine = None

try:
    from mtc.memory.redis_manager import RedisManager

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False
    RedisManager = None

try:
    from mtc.consciousness.measurement_framework import (
        ConsciousnessMeasurementFramework,
        ConsciousnessMetrics,
    )

    _HAS_MEASUREMENT = True
except ImportError:
    _HAS_MEASUREMENT = False
    ConsciousnessMeasurementFramework = None
    ConsciousnessMetrics = None

try:
    from mtc.consciousness.verification_framework import (
        ConsciousnessVerifier,
        VerificationSuiteResult,
    )

    _HAS_VERIFICATION = True
except ImportError:
    _HAS_VERIFICATION = False
    ConsciousnessVerifier = None
    VerificationSuiteResult = None

try:
    from mtc.consciousness.global_workspace import GlobalWorkspace, WorkspaceContent

    _HAS_GWT = True
except ImportError:
    _HAS_GWT = False
    GlobalWorkspace = None
    WorkspaceContent = None

try:
    from mtc.consciousness.enhanced_global_workspace import (
        EnhancedGlobalWorkspace,
        WorkspaceCandidate,
        WorkspaceCandidateSource,
        ConsciousnessState as EnhancedConsciousnessState,
        CognitiveModule,
    )

    _HAS_ENHANCED_GWT = True
except ImportError:
    _HAS_ENHANCED_GWT = False
    EnhancedGlobalWorkspace = None
    WorkspaceCandidate = None
    WorkspaceCandidateSource = None
    EnhancedConsciousnessState = None
    CognitiveModule = None


@dataclass
class Experience:
    """
    A unified experience in the consciousness system.
    Combines emotional, conscious, subconscious, and memory aspects.
    """

    timestamp: datetime
    experience_type: str  # "emotion", "conversation", "observation", "thought"

    # Input
    emotions: Dict[str, float]  # Raw emotional content
    context: str  # Text description
    valence: float = 0.0  # Overall positive/negative
    arousal: float = 0.5  # Energy level

    # Processing results (filled by orchestrator)
    snn_result: Optional[Dict] = None
    subconscious_state: Optional[Any] = None
    episodic_memory: Optional[Any] = None


@dataclass
class ConsciousnessState:
    """
    Current unified consciousness state across all layers.
    """

    timestamp: datetime

    # System state
    is_awake: bool  # True = processing experiences, False = sleeping/consolidating
    current_experience: Optional[Experience] = None

    # Layer states
    snn_active: bool = True
    lsm_active: bool = True
    htm_active: bool = True

    # Memory state
    episodic_memories_count: int = 0
    semantic_patterns_count: int = 0
    recent_consolidation: Optional[datetime] = None

    # Performance
    experiences_processed: int = 0
    total_processing_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0


class NeuralOrchestrator:
    """
    Unified orchestrator for the complete neural architecture.
    Coordinates SNN, LSM, HTM into single consciousness system.

    This orchestrator is designed with optional dependencies -- it will
    work in a reduced capacity if certain modules (EmotionLSMBridge,
    MemoryConsolidator, ThoughtGenerator, etc.) are not available.
    """

    def __init__(
        self,
        neural_data_manager=None,
        enable_mongodb: bool = True,
        brain=None,
        redis_manager=None,
    ):
        """
        Initialize the neural orchestrator.

        Args:
            neural_data_manager: Data manager for neural data persistence
            enable_mongodb: Whether to enable persistence
            brain: LLM Brain for LLM-enhanced thoughts
            redis_manager: Redis manager for working memory cache (optional)
        """
        self.brain = brain
        self.neural_data_manager = neural_data_manager
        if enable_mongodb and not neural_data_manager and _HAS_NEURAL_DATA:
            self.neural_data_manager = NeuralDataManager()

        # Redis for fast working memory cache (optional)
        self.redis = redis_manager

        logger.info("Initializing neural systems...")

        # Layer 1: Conscious (SNN) + Subconscious (LSM)
        self.lsm_bridge = None
        if _HAS_EMOTION_BRIDGE:
            self.lsm_bridge = EmotionLSMBridge(
                neural_data_manager=self.neural_data_manager, store_states=enable_mongodb
            )

        # Layer 2: Memory Consolidation (HTM)
        self.memory_consolidator = None
        if _HAS_MEMORY_CONSOLIDATOR:
            self.memory_consolidator = MemoryConsolidator(
                neural_data_manager=self.neural_data_manager,
                consolidation_cycles=10,
            )

        # Layer 3: Continuous Thought Machine (Background consciousness)
        self.thought_generator = None
        self.thought_stream = None
        self.continuous_thought_machine = None

        if _HAS_CTM:
            self.thought_generator = ThoughtGenerator(
                memory_manager=None,
                emotion_bridge=self.lsm_bridge,
                neural_orchestrator=self,
                brain=self.brain,
            )
            self.thought_stream = ThoughtStream(
                mongodb_manager=self.neural_data_manager if enable_mongodb else None,
                max_buffer_size=100,
            )

        # Experience buffer (daily memories before consolidation)
        # CAP: Prevent unbounded growth that can cause memory leaks
        self.MAX_EXPERIENCE_BUFFER = 200
        self.experience_buffer: List[Experience] = []

        # Consciousness state
        self.state = ConsciousnessState(timestamp=datetime.now(), is_awake=True)

        # Sleep schedule
        self.last_sleep: Optional[datetime] = None
        self.sleep_interval = timedelta(hours=8)

        # Performance tracking
        self.total_experiences = 0
        self.total_sleep_cycles = 0

        # Optional frameworks
        self.consciousness_framework = None
        self.consciousness_verifier = None
        self.global_workspace = None
        self.enhanced_workspace = None

        if _HAS_MEASUREMENT:
            try:
                self.consciousness_framework = ConsciousnessMeasurementFramework(
                    snn_neurons=5000,
                    lsm_neurons=10000,
                    htm_columns=4096,
                )
                logger.info("Consciousness Measurement Framework initialized")
            except Exception as e:
                logger.warning(f"Consciousness framework initialization failed: {e}")

        if _HAS_VERIFICATION:
            try:
                self.consciousness_verifier = ConsciousnessVerifier()
                logger.info("Consciousness Verifier initialized")
            except Exception as e:
                logger.warning(f"Consciousness verifier initialization failed: {e}")

        if _HAS_GWT:
            try:
                self.global_workspace = GlobalWorkspace(
                    integration_dimensions=512,
                    attention_decay=0.9,
                    salience_threshold=0.3,
                )
                logger.info("Global Workspace initialized")
            except Exception as e:
                logger.warning(f"Global Workspace initialization failed: {e}")

        if _HAS_ENHANCED_GWT:
            try:
                self.enhanced_workspace = EnhancedGlobalWorkspace(
                    capacity=7,
                    ignition_threshold=0.3,
                    amplification_factor=2.0,
                    integration_dimensions=512,
                )
                logger.info("Enhanced Global Workspace initialized (GWT)")
            except Exception as e:
                logger.warning(f"Enhanced workspace initialization failed: {e}")

        logger.info("Neural Orchestrator initialized!")

    def process_experience(
        self,
        experience_type: str,
        emotions: Dict[str, float],
        context: str = "",
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
    ) -> Experience:
        """
        Process a new experience through all neural layers.

        This is the main entry point for experiencing the world.
        Automatically flows through: SNN -> LSM -> Episodic Memory -> (eventual HTM consolidation)

        Args:
            experience_type: Type of experience
            emotions: Emotional content
            context: Description
            valence: Overall positive/negative (auto-calculated if None)
            arousal: Energy level (auto-calculated if None)

        Returns:
            Processed experience with all neural states
        """
        start_time = datetime.now()

        # Calculate valence/arousal if not provided
        if valence is None:
            positive = sum(
                emotions.get(e, 0) for e in ["joy", "love", "trust", "anticipation"]
            )
            negative = sum(
                emotions.get(e, 0) for e in ["sadness", "fear", "anger", "disgust"]
            )
            valence = (positive - negative) / max(positive + negative, 1.0)

        if arousal is None:
            high_arousal = sum(
                emotions.get(e, 0) for e in ["joy", "anger", "fear", "surprise"]
            )
            low_arousal = sum(emotions.get(e, 0) for e in ["sadness", "trust"])
            arousal = high_arousal / max(high_arousal + low_arousal, 1.0)

        # Create experience
        experience = Experience(
            timestamp=datetime.now(),
            experience_type=experience_type,
            emotions=emotions,
            context=context,
            valence=valence,
            arousal=arousal,
        )

        logger.info(
            f"Processing experience: {experience_type} "
            f"({', '.join(emotions.keys())})"
        )

        # LAYER 1 & 2: Process through SNN -> LSM
        snn_result = None
        subconscious_state = None
        if self.lsm_bridge and _HAS_EMOTION_BRIDGE:
            emotion_state = EmotionState(
                timestamp=experience.timestamp,
                emotions=emotions,
                valence=valence,
                arousal=arousal,
                dominance=0.5,
            )

            snn_result, subconscious_state = self.lsm_bridge.process_emotion(
                emotion_state,
                snn_duration_ms=50.0,
                lsm_duration_ms=100.0,
            )

        experience.snn_result = snn_result
        experience.subconscious_state = subconscious_state

        # LAYER 3: Add to episodic memory buffer
        episodic_memory = None
        if self.memory_consolidator:
            episodic_memory = self.memory_consolidator.add_episodic_memory(
                memory_type=experience_type,
                emotional_state=emotions,
                subconscious_pattern=(
                    subconscious_state.state_trajectory if subconscious_state else None
                ),
                context=context,
            )

        experience.episodic_memory = episodic_memory

        # Add to experience buffer
        self.experience_buffer.append(experience)

        # CAP: Prevent memory explosion if sleep fails
        if len(self.experience_buffer) > self.MAX_EXPERIENCE_BUFFER:
            logger.warning(
                f"Experience buffer at {len(self.experience_buffer)}, "
                f"trimming to {self.MAX_EXPERIENCE_BUFFER // 2}"
            )
            self.experience_buffer = self.experience_buffer[
                -(self.MAX_EXPERIENCE_BUFFER // 2) :
            ]

        # Update state
        self.state.current_experience = experience
        self.state.experiences_processed += 1
        if self.memory_consolidator:
            self.state.episodic_memories_count = len(
                self.memory_consolidator.episodic_memories
            )
            self.state.semantic_patterns_count = len(
                self.memory_consolidator.semantic_patterns
            )

        # Track performance
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.state.total_processing_time_ms += elapsed_ms
        self.state.avg_processing_time_ms = (
            self.state.total_processing_time_ms / self.state.experiences_processed
        )

        self.total_experiences += 1

        # Check if time for sleep
        if self.should_sleep():
            logger.info("Time for memory consolidation...")
            self.sleep()

        logger.info(f"Experience processed in {elapsed_ms:.1f}ms")

        return experience

    def _htm_to_lsm_feedback(self, htm_signal: np.ndarray) -> float:
        """
        HTM memory predictions influence LSM reservoir dynamics.

        When HTM recognizes a familiar pattern, it stabilizes the LSM.
        When HTM sees novelty, it excites the LSM for exploration.

        Args:
            htm_signal: HTM output signal

        Returns:
            Feedback strength (-1.0 to 1.0) to modulate LSM dynamics
        """
        novelty = np.var(htm_signal)
        feedback = np.tanh(novelty - 0.5)
        return float(feedback)

    def _lsm_to_snn_feedback(self, lsm_signal: np.ndarray) -> float:
        """
        LSM subconscious patterns modulate SNN conscious processing.

        Strong subconscious patterns boost conscious attention.
        Weak patterns allow conscious mind to wander.

        Args:
            lsm_signal: LSM reservoir state

        Returns:
            Feedback strength (0.0 to 1.0) to modulate SNN firing rates
        """
        pattern_strength = np.mean(np.abs(lsm_signal))
        feedback = min(1.0, pattern_strength)
        return float(feedback)

    def _snn_to_emotion_feedback(
        self, snn_signal: np.ndarray, current_emotions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        SNN spike patterns update emotional context.

        High SNN activity intensifies emotions.
        Low SNN activity dampens emotions.

        Args:
            snn_signal: SNN output signal
            current_emotions: Current emotional state

        Returns:
            Updated emotional state influenced by SNN activity
        """
        firing_intensity = np.mean(snn_signal)
        modulation = 1.0 + (firing_intensity - 0.5) * 0.2

        updated_emotions = {
            emotion: min(1.0, max(0.0, value * modulation))
            for emotion, value in current_emotions.items()
        }

        return updated_emotions

    def should_sleep(self) -> bool:
        """
        Determine if it's time for memory consolidation (sleep).

        Returns True if:
        - Never slept before AND have >= 5 experiences
        - OR it's been >= sleep_interval since last sleep AND have new experiences
        """
        if not self.last_sleep:
            return len(self.experience_buffer) >= 5

        time_since_sleep = datetime.now() - self.last_sleep
        return (
            time_since_sleep >= self.sleep_interval and len(self.experience_buffer) > 0
        )

    def sleep(self, duration: str = "short") -> Dict[str, Any]:
        """
        Enter sleep mode and consolidate memories.

        During sleep:
        1. Replay episodic memories through HTM
        2. Extract semantic patterns
        3. Update long-term understanding
        4. Clear experience buffer

        Args:
            duration: "short" (10 cycles), "medium" (30), "deep" (100)

        Returns:
            Consolidation report
        """
        if not self.experience_buffer:
            logger.warning("No experiences to consolidate")
            return {"status": "no_experiences"}

        if not self.memory_consolidator:
            logger.warning("Memory consolidator not available")
            return {"status": "no_consolidator"}

        logger.info(
            f"Entering sleep mode: {len(self.experience_buffer)} experiences, "
            f"{duration} sleep"
        )

        # Mark as sleeping
        self.state.is_awake = False

        # Get episodic memories from buffer
        memories_to_consolidate = [
            exp.episodic_memory
            for exp in self.experience_buffer
            if exp.episodic_memory is not None
        ]

        # Consolidate through HTM
        consolidation_report = self.memory_consolidator.consolidate_memories(
            memories=memories_to_consolidate, sleep_duration=duration
        )

        # Update state
        self.state.is_awake = True
        self.state.recent_consolidation = datetime.now()
        self.state.semantic_patterns_count = len(
            self.memory_consolidator.semantic_patterns
        )

        self.last_sleep = datetime.now()
        self.total_sleep_cycles += 1

        # Clear experience buffer (consolidated!)
        self.experience_buffer.clear()

        logger.info(
            f"Woke from sleep: {consolidation_report['patterns_discovered']} patterns, "
            f"anomaly reduced by {consolidation_report.get('anomaly_reduction', 0):.3f}"
        )

        return consolidation_report

    def get_semantic_understanding(
        self, pattern_type: Optional[str] = None, min_frequency: int = 1
    ) -> list:
        """
        Get semantic understanding (consolidated knowledge).

        Args:
            pattern_type: Filter by type (optional)
            min_frequency: Minimum frequency threshold

        Returns:
            List of semantic patterns (general understanding)
        """
        if not self.memory_consolidator:
            return []
        return self.memory_consolidator.get_semantic_patterns(
            pattern_type=pattern_type, min_frequency=min_frequency
        )

    def get_consciousness_state(self) -> ConsciousnessState:
        """Get current unified consciousness state"""
        return self.state

    def get_state(self) -> ConsciousnessState:
        """Alias for get_consciousness_state() for convenience"""
        return self.get_consciousness_state()

    async def measure_consciousness(self):
        """
        Measure current consciousness across 20 indicators.

        This integrates actual neural states (SNN, LSM, HTM) with memories
        and thoughts to produce research-grade consciousness measurements.

        Returns:
            ConsciousnessMetrics with 20 indicators or None if framework unavailable
        """
        if not self.consciousness_framework:
            logger.warning("Consciousness framework not available for measurement")
            return None

        try:
            if not self.experience_buffer:
                logger.warning("No experiences in buffer to measure consciousness from")
                return None

            last_exp = self.experience_buffer[-1]
            if not last_exp.snn_result or not last_exp.subconscious_state:
                logger.warning("Last experience missing neural data")
                return None

            # Get SNN firing rates as states
            firing_rates = last_exp.snn_result.get("firing_rates", {})
            snn_states = np.concatenate(
                [
                    np.array(firing_rates.get("input", [])),
                    np.array(firing_rates.get("hidden1", [])),
                    np.array(firing_rates.get("hidden2", [])),
                    np.array(firing_rates.get("hidden3", [])),
                    np.array(firing_rates.get("output", [])),
                ]
            )

            # Extract LSM reservoir states
            lsm_states = last_exp.subconscious_state.state_trajectory
            if lsm_states is None:
                logger.warning("LSM states missing from subconscious")
                return None

            # Pull recent memories if available
            memories = []
            thoughts = []
            if self.thought_stream:
                thoughts = [
                    t.content for t in self.thought_stream.get_recent_thoughts(count=10)
                ]

            # Measure consciousness using REAL neural states
            logger.info(
                f"Measuring consciousness with {len(snn_states)} SNN neurons, "
                f"{len(lsm_states)} LSM timesteps, {len(memories)} memories, "
                f"{len(thoughts)} thoughts"
            )

            metrics = await self.consciousness_framework.measure_consciousness(
                snn_states=snn_states,
                lsm_states=lsm_states,
                input_data=None,
                memories=memories,
                thoughts=thoughts,
            )

            logger.info(
                f"Consciousness measured: Phi={metrics.phi:.3f}, "
                f"Integration={metrics.integration:.3f}, "
                f"Self-Model={metrics.self_model_accuracy:.3f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Consciousness measurement failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    # =========================================================================
    # Attention Schema Methods (AST Integration)
    # =========================================================================

    async def request_attention_shift(
        self,
        target_description: str,
        reason: str,
        priority: float = 0.8,
    ) -> bool:
        """
        Request a voluntary attention shift.

        Args:
            target_description: What to focus on
            reason: Why shift attention
            priority: How strongly to prioritize (0-1)

        Returns:
            True if shift was accepted
        """
        if not self.enhanced_workspace:
            logger.warning("Enhanced workspace not available for attention shift")
            return False

        return await self.enhanced_workspace.request_voluntary_attention_shift(
            target_description=target_description,
            reason=reason,
            priority=priority,
        )

    async def get_attention_report(self) -> str:
        """
        Get verbal report about current attention (introspection).

        Returns:
            Natural language description of current attention
        """
        if not self.enhanced_workspace:
            return "Attention system not yet initialized"

        return await self.enhanced_workspace.get_attention_report()

    def get_attention_schema_state(self):
        """
        Get the current attention schema state.

        Returns the self-model of attention.
        """
        if not self.enhanced_workspace:
            return None

        return self.enhanced_workspace.get_attention_schema_state()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics across all neural systems"""
        # Get detailed bridge metrics if available
        bridge_metrics = {}
        if self.lsm_bridge:
            bridge_metrics = self.lsm_bridge.get_metrics()

        # Get attention schema metrics if available
        attention_schema_metrics = {}
        if self.enhanced_workspace:
            schema_state = self.enhanced_workspace.get_attention_schema_state()
            if schema_state:
                attention_schema_metrics = {
                    "attention_state": schema_state.attention_state.value,
                    "focus": (
                        schema_state.current_focus.summary
                        if schema_state.current_focus
                        else None
                    ),
                    "capacity_used": schema_state.attention_capacity_used,
                    "voluntary_ratio": schema_state.voluntary_ratio,
                }

        return {
            "consciousness": {
                "is_awake": self.state.is_awake,
                "experiences_processed": self.state.experiences_processed,
                "avg_processing_time_ms": self.state.avg_processing_time_ms,
                "total_sleep_cycles": self.total_sleep_cycles,
            },
            "memory": {
                "episodic_memories": self.state.episodic_memories_count,
                "semantic_patterns": self.state.semantic_patterns_count,
                "experiences_in_buffer": len(self.experience_buffer),
            },
            "layers": {
                "snn_active": self.state.snn_active,
                "lsm_active": self.state.lsm_active,
                "htm_active": self.state.htm_active,
            },
            "neural_bridge": bridge_metrics,
            "enhanced_workspace": (
                self.enhanced_workspace.get_statistics()
                if self.enhanced_workspace
                else {"available": False}
            ),
            "attention_schema": attention_schema_metrics,
        }

    # ======== CONTINUOUS THOUGHT MACHINE CONTROLS ========

    async def start_continuous_thought_machine(
        self,
        websocket_broadcaster: Optional[Any] = None,
        consciousness_state: str = "awake",
        vector_coordinator: Optional[Any] = None,
    ) -> None:
        """
        Start the continuous thought loop.

        Args:
            websocket_broadcaster: Optional async function for broadcasting thoughts
            consciousness_state: Initial state (awake, drowsy, dreaming, deep_sleep)
            vector_coordinator: Optional Qdrant coordinator for linking thoughts to memories
        """
        if not _HAS_CTM or not self.thought_generator:
            logger.warning("Continuous thought machine not available")
            return

        if self.continuous_thought_machine and self.continuous_thought_machine.running:
            logger.warning("Continuous thought machine already running")
            return

        logger.info("Starting Continuous Thought Machine...")

        if vector_coordinator is None and self.thought_generator.memory_manager:
            vector_coordinator = getattr(
                self.thought_generator.memory_manager, "vector_coordinator", None
            )

        self.continuous_thought_machine = ContinuousThoughtMachine(
            thought_generator=self.thought_generator,
            thought_stream=self.thought_stream,
            websocket_broadcaster=websocket_broadcaster,
            consciousness_state=consciousness_state,
            neural_data_manager=self.neural_data_manager,
            vector_coordinator=vector_coordinator,
        )

        await self.continuous_thought_machine.start()

    async def stop_continuous_thought_machine(self) -> None:
        """Stop the continuous thought loop."""
        if not self.continuous_thought_machine:
            logger.warning("Continuous thought machine not running")
            return

        await self.continuous_thought_machine.stop()
        logger.info("Continuous thought machine stopped")

    def pause_continuous_thoughts(self) -> None:
        """Pause thought generation (can be resumed)."""
        if self.continuous_thought_machine:
            self.continuous_thought_machine.pause()

    def resume_continuous_thoughts(self) -> None:
        """Resume thought generation after pause."""
        if self.continuous_thought_machine:
            self.continuous_thought_machine.resume()

    def set_consciousness_state(self, state: str) -> None:
        """
        Change consciousness state (affects thought frequency).

        Args:
            state: awake, drowsy, dreaming, deep_sleep
        """
        if self.continuous_thought_machine:
            self.continuous_thought_machine.set_consciousness_state(state)
        else:
            logger.warning("Continuous thought machine not running")

    def get_continuous_thought_status(self) -> Dict[str, Any]:
        """
        Get current status of continuous thought machine.

        Returns:
            Dict with status, or empty dict if not running
        """
        if self.continuous_thought_machine:
            return self.continuous_thought_machine.get_status()
        return {"running": False}

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all neural systems and free memory.

        Cleanup order:
        1. Stop continuous thought machine
        2. Clear experience buffer
        3. Clear LSM bridge state history
        4. Clear thought generator/stream caches
        5. Close data connections
        6. Force garbage collection
        7. Clear GPU memory cache
        """
        import gc
        import torch

        logger.info("Shutting down Neural Orchestrator...")

        # 1. Stop continuous thought machine if running
        if self.continuous_thought_machine:
            logger.info("Stopping continuous thought machine...")
            self.continuous_thought_machine.stop()
            self.continuous_thought_machine = None

        # 2. Clear experience buffer
        buffer_size = len(self.experience_buffer)
        if buffer_size > 0:
            logger.info(f"Clearing {buffer_size} experiences from buffer...")
            self.experience_buffer.clear()

        # 3. Clear LSM bridge state history and trajectories
        if self.lsm_bridge:
            if hasattr(self.lsm_bridge, "state_history"):
                self.lsm_bridge.state_history.clear()
            if hasattr(self.lsm_bridge, "reservoir_trajectory"):
                self.lsm_bridge.reservoir_trajectory.clear()
            logger.info("Cleared LSM bridge state history")

        # 4. Clear thought generator caches
        if self.thought_generator:
            if hasattr(self.thought_generator, "recent_thought_types"):
                self.thought_generator.recent_thought_types.clear()
            logger.info("Cleared thought generator caches")

        # 5. Clear thought stream buffer
        if self.thought_stream:
            if hasattr(self.thought_stream, "thought_buffer"):
                self.thought_stream.thought_buffer.clear()
            logger.info("Cleared thought stream buffer")

        # 6. Close data connections
        if self.neural_data_manager:
            try:
                await self.neural_data_manager.close()
                logger.info("Data manager closed")
            except Exception as e:
                logger.warning(f"Error closing data manager: {e}")

        # 7. Force garbage collection
        gc.collect()
        logger.info("Garbage collection complete")

        # 8. Clear GPU memory cache (MPS or CUDA)
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("MPS GPU cache cleared")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA GPU cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing GPU cache: {e}")

        # Update state
        self.state.is_awake = False

        logger.info("Neural Orchestrator shutdown complete - memory freed!")
