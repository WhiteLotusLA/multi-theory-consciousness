"""
Neural Orchestrator
===================

Coordinates the neural substrates behind one ``process_experience`` entry
point, binds their outputs through oscillatory (Kuramoto) dynamics, and
schedules sleep and wake cycles for memory consolidation.

Architecture:
1. SNN: fast temporal processing of emotional input (spikes)
2. LSM: reservoir dynamics over the SNN output
3. HTM: sequence memory and consolidation (episodic to semantic)
4. Orchestrator: coordinates the layers, binds their outputs, stashes the
   latest substrate signals for the measurement layer and schedules
   consolidation

Every substrate except the oscillatory binding is optional. A component whose
module is not installed is reported by ``available_components()`` and named
once in the log at construction; nothing stands in for its output.

Components and stores:

- The emotion bridge (SNN and LSM) and the memory consolidator (HTM) are built
  from their default configs on the first ``process_experience`` or ``sleep``
  unless an instance is injected (``emotion_bridge``, ``memory_consolidator``)
  or ``False`` disables the component. A component class that does not declare
  ``build_on_first_use`` is built at construction instead. Constructing an
  orchestrator therefore stays cheap, and the measurement framework is sized
  from the substrates as soon as they exist.
- A data manager (MongoDB) exists only when one is injected or
  ``enable_mongodb=True`` is passed; the default is False. A Redis manager
  exists only when one is passed. Constructing an orchestrator with defaults
  creates no store client.
- Accelerator cache clears go through the device manager in
  ``mtc.neural.mps_utils``; this module names no accelerator backend.
"""

import importlib
import inspect
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mtc.neural.oscillatory_binding import OscillatoryBinding

logger = logging.getLogger(__name__)

# Optional dependencies. Each import names the module's port target; the
# orchestrator runs in a reduced capacity without the ones not yet ported.

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
    from mtc.assessment.framework import (
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


# Optional component -> (import path, the port phase that brings the module).
# A None phase means the module ships with MTC already.
OPTIONAL_COMPONENTS: Dict[str, Tuple[str, Optional[str]]] = {
    "emotion_lsm_bridge": ("mtc.neural.liquid.emotion_lsm_bridge", "phase 5"),
    "memory_consolidator": ("mtc.neural.htm.memory_consolidator", "phase 5"),
    "neural_data_manager": ("mtc.neural.mongodb_schemas", "phase 5"),
    "redis_manager": ("mtc.memory.redis_manager", "phase 5"),
    "measurement_framework": ("mtc.assessment.framework", None),
    "verification_framework": ("mtc.consciousness.verification_framework", None),
    "continuous_thought": ("mtc.consciousness.continuous", "phase 6"),
    "global_workspace": ("mtc.consciousness.global_workspace", None),
    "enhanced_workspace": ("mtc.consciousness.enhanced_global_workspace", None),
}


def _component_flags() -> Dict[str, bool]:
    """Import availability of each optional component, read at call time."""
    return {
        "emotion_lsm_bridge": bool(_HAS_EMOTION_BRIDGE),
        "memory_consolidator": bool(_HAS_MEMORY_CONSOLIDATOR),
        "neural_data_manager": bool(_HAS_NEURAL_DATA),
        "redis_manager": bool(_HAS_REDIS),
        "measurement_framework": bool(_HAS_MEASUREMENT),
        "verification_framework": bool(_HAS_VERIFICATION),
        "continuous_thought": bool(_HAS_CTM),
        "global_workspace": bool(_HAS_GWT),
        "enhanced_workspace": bool(_HAS_ENHANCED_GWT),
    }


# The eight SNN-group oscillators of the binding layer are emotion channels.
_EMOTION_OSCILLATORS = (
    "joy",
    "curiosity",
    "fear",
    "sadness",
    "surprise",
    "anger",
    "calm",
    "interest",
)


def _finite(value: Any) -> Optional[float]:
    """The value as a finite float, or None when it is missing or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_count(value: Any) -> Optional[int]:
    """A non-negative integer count, or None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value) if value >= 0 else None


def _as_1d(value: Any) -> Optional[np.ndarray]:
    """A 1-D float array (a 2-D trajectory becomes its last step), or None."""
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        if array.ndim == 2:
            array = array[-1]
        array = np.ravel(array)
        return array if array.size else None
    except Exception:
        return None


def _resize_pattern(pattern: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad or truncate a 1-D pattern to the HTM input size."""
    if len(pattern) == size:
        return pattern
    if len(pattern) < size:
        padded = np.zeros(size)
        padded[: len(pattern)] = pattern
        return padded
    return pattern[:size]


def _reservoir_binding_metrics(trajectory: Any, coherence: Any) -> Dict[str, float]:
    """LSM-group oscillator values from the reservoir output that exists.

    Band energies, spectral entropy, standard deviation and mean come from the
    trajectory and the fading-memory oscillator from the emotional coherence.
    Input sensitivity has no reservoir measurement and stays unset.
    """
    out: Dict[str, float] = {}
    value = _finite(coherence)
    if value is not None:
        out["fading_memory"] = value
    if trajectory is None:
        return out
    try:
        array = np.asarray(trajectory, dtype=float)
    except (TypeError, ValueError):
        return out
    if array.size == 0 or not np.all(np.isfinite(array)):
        return out
    # A 2-D trajectory (time by unit) is analyzed as its mean activity over time.
    if array.ndim <= 1:
        series = array.ravel()
    else:
        series = array.reshape(len(array), -1).mean(axis=1)
    if series.size >= 2:
        spectrum = np.abs(np.fft.rfft(series[:64]))
        third = max(1, len(spectrum) // 3)
        bands = {
            "low_freq_energy": spectrum[:third],
            "mid_freq_energy": spectrum[third : 2 * third],
            "high_freq_energy": spectrum[2 * third :],
        }
        for name, band in bands.items():
            if band.size:
                out[name] = float(np.mean(band))
        total = float(np.sum(spectrum))
        if total > 0.0:
            share = spectrum / total
            out["spectral_entropy"] = float(-np.sum(share * np.log(share + 1e-8)))
    out["nonlinear_response"] = float(np.std(array))
    out["echo_state"] = float(np.mean(np.abs(array)))
    return out


def _release_accelerator_cache(include_cuda: bool) -> Optional[str]:
    """Empty the accelerator allocator cache through the device manager.

    The manager (``mtc.neural.mps_utils``) is consulted only when torch is
    already loaded: when it is not, nothing has allocated accelerator memory,
    so there is nothing to release. A CUDA cache is released only when
    ``include_cuda`` is true. Returns the device that was cleared, or None.
    """
    if "torch" not in sys.modules:
        return None
    try:
        manager = importlib.import_module("mtc.neural.mps_utils").mps_manager
        device = getattr(getattr(manager, "device", None), "type", None)
        if not getattr(manager, "enabled", False) or device in (None, "cpu"):
            return None
        if device == "cuda" and not include_cuda:
            return None
        manager.empty_cache()
        return str(device)
    except Exception as e:
        logger.debug(f"Accelerator cache release skipped: {e}")
    return None


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

    # Set by an attached legacy integration once its assessment completes;
    # None until then and when the assessment had insufficient data.
    consciousness_level: Optional[float] = None


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
    Coordinates SNN, LSM and HTM into a single processing pipeline.

    This orchestrator is designed with optional dependencies: it works in a
    reduced capacity when modules such as the emotion bridge, the memory
    consolidator or the continuous thought package are not installed, and
    ``available_components()`` says which ones are.
    """

    def __init__(
        self,
        neural_data_manager=None,
        enable_mongodb: bool = False,
        brain=None,
        redis_manager=None,
        emotion_bridge=None,
        memory_consolidator=None,
    ):
        """
        Initialize the neural orchestrator.

        Args:
            neural_data_manager: Data manager for neural data persistence
            enable_mongodb: Build a data manager from Settings when none was
                passed (it connects lazily); False by default, so nothing
                touches MongoDB unless asked
            brain: LLM Brain for LLM-enhanced thoughts
            redis_manager: Redis manager for working memory cache (optional)
            emotion_bridge: An emotion bridge to use; None builds the default
                one on first use, False leaves the component out
            memory_consolidator: A memory consolidator to use; None builds the
                default one on first use, False leaves the component out
        """
        self.brain = brain
        self.enable_mongodb = bool(enable_mongodb)
        self.neural_data_manager = neural_data_manager
        if enable_mongodb and not neural_data_manager and _HAS_NEURAL_DATA:
            self.neural_data_manager = NeuralDataManager()

        # Redis for fast working memory cache (optional)
        self.redis = redis_manager

        logger.info("Initializing neural systems...")
        missing = []
        for name, installed in _component_flags().items():
            if not installed:
                phase = OPTIONAL_COMPONENTS[name][1]
                missing.append(name if phase is None else f"{name} ({phase})")
        if missing:
            logger.info(f"Optional components not installed: {', '.join(missing)}")

        # Layer 1: Conscious (SNN) + Subconscious (LSM)
        # Layer 2: Memory Consolidation (HTM)
        # An injected instance is used as is; False leaves the component out; a
        # real class (it declares build_on_first_use) is built by
        # _ensure_components on first use, and any other class now.
        self.lsm_bridge = None
        self.memory_consolidator = None
        self._deferred_components: Dict[str, bool] = {}
        self._place_component(
            "lsm_bridge", emotion_bridge, EmotionLSMBridge, _HAS_EMOTION_BRIDGE
        )
        self._place_component(
            "memory_consolidator",
            memory_consolidator,
            MemoryConsolidator,
            _HAS_MEMORY_CONSOLIDATOR,
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

        # Experience buffer (daily memories before consolidation), capped so a
        # failed consolidation cannot grow it without bound
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

        # Oscillatory binding over the substrate outputs of each experience
        self.oscillatory_binding = OscillatoryBinding()
        self._last_binding_result = None
        self._binds_run = 0

        # The substrate's latest real outputs, readable by the measurement
        # layer. None until the first experience produces them.
        self.latest_neural_signals: Optional[Dict[str, np.ndarray]] = None
        self.latest_substrate_states: Optional[Dict[str, Dict[str, Any]]] = None

        # Per-turn memory retrieval, published by the conversation pipeline;
        # honestly empty until a turn retrieves something.
        self._last_retrieved_memories: List[Any] = []

        # Optional frameworks
        self.consciousness_framework = None
        self.consciousness_verifier = None
        self.global_workspace = None
        self.enhanced_workspace = None

        self._build_framework()

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

    def _place_component(
        self, attribute: str, supplied: Any, cls: Any, available: bool
    ) -> None:
        """Wire one optional component: use ``supplied``, leave it out, defer it
        or build it now (see the module docstring)."""
        if supplied is False:
            return
        if supplied is not None:
            setattr(self, attribute, supplied)
            return
        if not available or cls is None:
            return
        if getattr(cls, "build_on_first_use", False):
            self._deferred_components[attribute] = True
        else:
            setattr(self, attribute, self._build_component(attribute))

    def _build_component(self, attribute: str) -> Any:
        manager = self.neural_data_manager
        if attribute == "lsm_bridge":
            return EmotionLSMBridge(
                neural_data_manager=manager,
                store_states=bool(self.enable_mongodb and manager is not None),
            )
        return MemoryConsolidator(neural_data_manager=manager, consolidation_cycles=10)

    def _ensure_components(self) -> None:
        """Build the deferred components (the first experience or sleep does).

        A failure (for example an explicit device that is not available)
        propagates and leaves the component deferred, so the next call tries
        again. Once substrates exist the measurement framework is rebuilt from
        their sizes.
        """
        pending = getattr(self, "_deferred_components", None)
        if not pending:
            return
        for attribute in list(pending):
            setattr(self, attribute, self._build_component(attribute))
            del pending[attribute]
        self._build_framework()

    def _build_framework(self) -> None:
        """(Re)create the measurement framework sized from the live substrates."""
        if not _HAS_MEASUREMENT:
            return
        sizes = self.substrate_sizes()
        try:
            self.consciousness_framework = ConsciousnessMeasurementFramework(
                snn_neurons=sizes["snn"] or 0,
                lsm_neurons=sizes["lsm"] or 0,
                htm_columns=sizes["htm_columns"] or 0,
            )
            logger.info(
                "Consciousness Measurement Framework initialized "
                "(14 framework metrics)"
            )
        except Exception as e:
            logger.warning(f"Consciousness framework initialization failed: {e}")

    # ======== COMPONENTS ========

    def available_components(self) -> Dict[str, bool]:
        """Whether each optional component's module is installed.

        The keys are the names in ``OPTIONAL_COMPONENTS``. A component whose
        module imports but whose construction failed is logged separately at
        construction time.
        """
        return _component_flags()

    def substrate_sizes(self) -> Dict[str, Optional[int]]:
        """Sizes read from the live substrates; None for an absent substrate.

        Keys: ``snn`` (neuron count), ``lsm`` (reservoir size), ``htm`` (columns
        times cells per column) and ``htm_columns`` (the column count the
        measurement framework takes).
        """
        bridge = getattr(self, "lsm_bridge", None)
        snn_neurons = None
        snn = getattr(bridge, "snn", None)
        count = getattr(snn, "_count_neurons", None)
        if callable(count):
            try:
                snn_neurons = _as_count(count())
            except Exception as e:
                logger.debug(f"SNN neuron count unavailable: {e}")
        elif callable(getattr(snn, "get_metrics", None)):
            try:
                snn_neurons = _as_count(snn.get_metrics().get("total_neurons"))
            except Exception as e:
                logger.debug(f"SNN neuron count unavailable: {e}")
        lsm = getattr(bridge, "lsm", None)
        lsm_neurons = _as_count(
            getattr(getattr(lsm, "config", None), "reservoir_size", None)
        )
        if lsm_neurons is None:
            lsm_neurons = _as_count(getattr(lsm, "n_reservoir", None))
        htm = getattr(getattr(self, "memory_consolidator", None), "htm", None)
        htm_config = getattr(htm, "config", None)
        columns = _as_count(getattr(htm_config, "num_columns", None))
        cells = _as_count(getattr(htm_config, "cells_per_column", None))
        return {
            "snn": snn_neurons,
            "lsm": lsm_neurons,
            "htm": (
                columns * cells if columns is not None and cells is not None else None
            ),
            "htm_columns": columns,
        }

    # ======== EXPERIENCE PROCESSING ========

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
        Automatically flows through: SNN -> LSM -> Episodic Memory -> oscillatory
        binding -> (eventual HTM consolidation).

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
        self._ensure_components()

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
                dominance=None,
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

        # Oscillatory binding over what the substrates produced this cycle
        self._run_binding(
            subconscious_state, episodic_memory, emotions, valence, arousal
        )

        # Add to experience buffer
        self.experience_buffer.append(experience)

        # Keep the most recent half when consolidation has not run
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

        # Release cached accelerator allocations each cycle; without it the
        # MPS cache grows until the next consolidation.
        _release_accelerator_cache(include_cuda=False)

        # Check if time for sleep
        if self.should_sleep():
            logger.info("Time for memory consolidation...")
            self.sleep()

        spikes = (
            snn_result.get("total_spikes") if isinstance(snn_result, dict) else None
        )
        pattern = getattr(subconscious_state, "dominant_pattern", None)
        logger.info(
            f"Experience processed in {elapsed_ms:.1f}ms "
            f"(SNN spikes: {spikes if spikes is not None else 'none'}, "
            f"LSM pattern: {pattern if pattern is not None else 'none'})"
        )

        # Leave this cycle's substrate outputs readable; the next measurement
        # cycle reads them.
        self._stash_latest_signals(snn_result, subconscious_state, episodic_memory)

        return experience

    # ======== OSCILLATORY BINDING ========

    def _run_binding(
        self,
        subconscious_state: Any,
        episodic_memory: Any,
        emotions: Dict[str, float],
        valence: Optional[float],
        arousal: Optional[float],
    ) -> None:
        """Bind this cycle's measured oscillator inputs and keep the result."""
        binding = getattr(self, "oscillatory_binding", None)
        if binding is None:
            return
        metrics = self._extract_binding_metrics(
            subconscious_state, episodic_memory, emotions, valence, arousal
        )
        try:
            result = binding.bind(metrics)
        except Exception as e:
            logger.warning(f"Oscillatory binding failed: {e}")
            return
        self._last_binding_result = result
        self._binds_run = getattr(self, "_binds_run", 0) + 1
        logger.debug(
            f"Oscillatory binding: r={result.global_order_parameter:.3f}, "
            f"{len(result.clusters)} clusters, {result.elapsed_ms:.2f}ms"
        )

    def _extract_binding_metrics(
        self,
        subconscious_state: Any,
        episodic_memory: Any,
        emotions: Optional[Dict[str, float]],
        valence: Optional[float],
        arousal: Optional[float],
    ) -> Dict[str, float]:
        """Oscillator inputs for the binding layer, from what exists only.

        Emotion oscillators come from the emotions supplied; LSM oscillators
        from the reservoir output when there is one; HTM oscillators from a
        live inference-only HTM pass when a consolidator with an HTM exists;
        cross-modal oscillators only where a value was measured. Attention
        focus, goal relevance and temporal urgency have no producer here and
        stay unset (the binding layer reads an unset oscillator as zero).
        """
        emotions = emotions or {}
        metrics: Dict[str, float] = {}
        for name in _EMOTION_OSCILLATORS:
            value = _finite(emotions.get(name))
            if value is not None:
                metrics[name] = value

        trajectory = getattr(subconscious_state, "state_trajectory", None)
        if subconscious_state is not None:
            metrics.update(
                _reservoir_binding_metrics(
                    trajectory,
                    getattr(subconscious_state, "emotional_coherence", None),
                )
            )

        if episodic_memory is not None:
            strength = _finite(getattr(episodic_memory, "consolidation_strength", None))
            if strength is not None:
                metrics["sequence_strength"] = strength

        htm = getattr(getattr(self, "memory_consolidator", None), "htm", None)
        if htm is not None and trajectory is not None:
            metrics.update(self._live_htm_binding_metrics(htm, trajectory))

        level = _finite(valence)
        if level is not None:
            metrics["valence"] = _unit((level + 1.0) / 2.0)
        surprise = _finite(emotions.get("surprise"))
        if surprise is not None:
            metrics["novelty"] = surprise
        level = _finite(arousal)
        if level is not None:
            metrics["arousal"] = _unit(level)
        return metrics

    def _live_htm_binding_metrics(self, htm: Any, pattern: Any) -> Dict[str, float]:
        """HTM oscillator values from one live, inference-only HTM pass.

        Accepts any HTM exposing ``process(pattern, learn=False)``. With
        ``learn=False`` the spatial pooler and temporal memory change no
        permanence or segment, so live binding cannot alter what consolidation
        learned. Sets the anomaly, the burst fraction (unpredicted active
        columns burst, so it equals the anomaly), the prediction accuracy and
        the active-column density relative to the pooler's target sparsity;
        leaves every other HTM oscillator unset; never raises.
        """
        out: Dict[str, float] = {}
        process = getattr(htm, "process", None)
        if not callable(process) or pattern is None:
            return out
        try:
            array = np.ravel(np.asarray(pattern, dtype=float))
            if array.size == 0 or not np.all(np.isfinite(array)):
                return out
            size = _as_count(getattr(htm, "input_size", None))
            if size:
                array = _resize_pattern(array, size)
            result = process(array, learn=False)
            if not isinstance(result, dict):
                return out
            anomaly = _finite(result.get("anomaly_score"))
            num_active = _finite(result.get("num_active_columns"))
        except Exception as e:
            logger.debug(f"Live HTM binding inference unavailable: {e}")
            return {}

        if anomaly is not None:
            anomaly = _unit(anomaly)
            out["anomaly_score"] = anomaly
            out["burst_fraction"] = anomaly
            out["prediction_accuracy"] = 1.0 - anomaly

        pooler = getattr(htm, "spatial_pooler", None)
        config = getattr(htm, "config", None)
        columns = _as_count(getattr(pooler, "num_columns", None)) or _as_count(
            getattr(config, "num_columns", None)
        )
        density = _finite(getattr(config, "local_area_density", None))
        if density is None:
            density = _finite(
                getattr(getattr(pooler, "config", None), "local_area_density", None)
            )
        if num_active is not None and columns and density and density > 0:
            target = max(1.0, columns * density)
            out["active_columns"] = _unit(num_active / target)
        return out

    def get_binding_context(self) -> str:
        """The latest binding context string, or an empty string before any."""
        result = getattr(self, "_last_binding_result", None)
        return result.context_string if result is not None else ""

    def get_binding_stats(self) -> Dict[str, Any]:
        """Latest binding values for RPT; empty before the first bind.

        Keys: ``global_order_parameter`` and ``cross_coherence`` from the last
        binding result, and ``binding_count``, the binds this orchestrator ran.
        """
        result = getattr(self, "_last_binding_result", None)
        if result is None:
            return {}
        return {
            "global_order_parameter": result.global_order_parameter,
            "cross_coherence": result.cross_coherence,
            "binding_count": getattr(self, "_binds_run", 0),
        }

    def _stash_latest_signals(
        self, snn_result: Any, subconscious_state: Any, episodic_memory: Any
    ) -> None:
        """Stash the substrate's latest real outputs for the measurement layer.

        ``latest_neural_signals`` holds 1-D float arrays (``snn``: output firing
        rates; ``lsm``: the last trajectory step; ``htm``: the episodic
        pattern) and ``latest_substrate_states`` the raw substrate dicts
        (``snn_state``, ``lsm_state``). Only what exists is stashed: a missing
        output stays missing, and either attribute is None when nothing exists.
        """
        signals: Dict[str, np.ndarray] = {}
        try:
            output = _as_1d(snn_result["firing_rates"]["output"])
        except (KeyError, TypeError, IndexError):
            output = None
        if output is not None:
            signals["snn"] = output
        trajectory = getattr(subconscious_state, "state_trajectory", None)
        lsm_signal = _as_1d(trajectory) if trajectory is not None else None
        if lsm_signal is not None:
            signals["lsm"] = lsm_signal
        pattern = getattr(episodic_memory, "subconscious_pattern", None)
        htm_signal = _as_1d(pattern) if pattern is not None else None
        if htm_signal is not None:
            signals["htm"] = htm_signal
        self.latest_neural_signals = signals or None

        states: Dict[str, Dict[str, Any]] = {}
        if isinstance(snn_result, dict) and snn_result:
            states["snn_state"] = snn_result
        lsm_state: Dict[str, Any] = {}
        if trajectory is not None:
            lsm_state["reservoir_state"] = trajectory
        lsm_config = getattr(
            getattr(getattr(self, "lsm_bridge", None), "lsm", None), "config", None
        )
        spectral_radius = _finite(getattr(lsm_config, "spectral_radius", None))
        if lsm_state and spectral_radius is not None:
            lsm_state["spectral_radius"] = spectral_radius
        if lsm_state:
            states["lsm_state"] = lsm_state
        self.latest_substrate_states = states or None

    # ======== SLEEP AND CONSOLIDATION ========

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

        self._ensure_components()
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

        reduction = consolidation_report.get("anomaly_reduction")
        logger.info(
            f"Woke from sleep: {consolidation_report.get('patterns_discovered', 0)} "
            f"patterns, anomaly reduced by "
            f"{'n/a' if reduction is None else f'{reduction:.3f}'}"
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
        Measure the 14 framework metrics on the latest substrate states.

        Feeds the last experience's real SNN firing rates and LSM trajectory
        (and recent thoughts when the thought stream exists) to the
        measurement framework in ``mtc.assessment.framework``.

        Returns:
            ConsciousnessMetrics with the 14 framework metrics, or None when the
            framework or the neural data is unavailable
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

            # Not ported: the reference implementation's read of recent
            # memories from the document store. It guarded on an
            # ``episodic_memories`` attribute that its data manager never
            # defined (the branch could not run), and episodic rows live in the
            # episodic store of the memory layer, which this class does not
            # hold. Recent memories arrive with the memory layer (Phase 6), so
            # the list stays empty here rather than being filled from a
            # collection that does not exist. Thoughts come from the thought
            # stream when it exists.
            memories = []
            thoughts = []
            if self.thought_stream:
                thoughts = [
                    t.content for t in self.thought_stream.get_recent_thoughts(count=10)
                ]

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

    def _verification_neural_data(self) -> Dict[str, Any]:
        """Substrate blocks for verification, from the latest experience only.

        Reads ``latest_substrate_states``, ``latest_neural_signals`` and the
        current experience. A block is real only when its substrate produced
        output in the latest experience: the SNN with a finite synchrony in the
        stashed SNN state, the LSM with a stashed trajectory and a finite
        emotional coherence, the HTM with a stashed episodic pattern and a
        finite anomaly score from the HTM's own metrics.
        """
        signals = getattr(self, "latest_neural_signals", None)
        signals = signals if isinstance(signals, dict) else {}
        states = getattr(self, "latest_substrate_states", None)
        states = states if isinstance(states, dict) else {}
        experience = getattr(getattr(self, "state", None), "current_experience", None)

        def signal(key: str) -> Optional[np.ndarray]:
            value = signals.get(key)
            return _as_1d(value) if value is not None else None

        def as_list(key: str) -> List[float]:
            array = signal(key)
            return array.tolist() if array is not None else []

        snn_state = states.get("snn_state")
        metrics = snn_state.get("metrics") if isinstance(snn_state, dict) else None
        synchrony = (
            _finite(metrics.get("synchrony")) if isinstance(metrics, dict) else None
        )
        snn = {
            "synchrony": synchrony,
            "temporal_patterns": as_list("snn"),
            "is_real": synchrony is not None,
        }

        lsm_produced = signal("lsm") is not None
        subconscious = (
            getattr(experience, "subconscious_state", None) if lsm_produced else None
        )
        coherence = _finite(getattr(subconscious, "emotional_coherence", None))
        lsm = {
            "coherence": coherence,
            "intuitive_valence": _finite(
                getattr(subconscious, "intuitive_valence", None)
            ),
            "intuitive_arousal": _finite(
                getattr(subconscious, "intuitive_arousal", None)
            ),
            "state_trajectory": as_list("lsm"),
            "is_real": coherence is not None,
        }

        anomaly = None
        if signal("htm") is not None:
            htm_metrics = self._component_metric(
                getattr(getattr(self, "memory_consolidator", None), "htm", None),
                "get_metrics",
            )
            if isinstance(htm_metrics, dict):
                anomaly = _finite(htm_metrics.get("anomaly_score"))
        htm = {"anomaly_score": anomaly, "is_real": anomaly is not None}
        return {"snn": snn, "lsm": lsm, "htm": htm}

    async def verify_consciousness(
        self,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        *,
        pad_state: Any = None,
        temporal_data: Optional[Dict[str, List[Any]]] = None,
    ) -> Optional["VerificationSuiteResult"]:
        """Run the verification suite on the latest experience's real outputs.

        The coherence, temporal consistency and emotional authenticity tests
        score only blocks the latest experience produced (see
        ``_verification_neural_data``); with no substrate output every test
        is ``insufficient_data`` and the overall score is None, whether or not
        ``conversation_history`` is passed: conversational continuity joins
        the coherence score only beside a real substrate or affect component.
        PAD affect and temporal samples come from the caller (the orchestrator
        owns neither).

        Args:
            conversation_history: Turns for the contextual continuity part,
                which never makes coherence measured on its own
            pad_state: Optional ``PADState`` or mapping of the PAD axes
            temporal_data: Optional ``ConsciousnessIntegration.get_temporal_data()``

        Returns:
            VerificationSuiteResult, or None when the verifier is unavailable
            or the run failed; never raises.
        """
        verifier = getattr(self, "consciousness_verifier", None)
        if verifier is None:
            logger.warning("Consciousness verifier not available")
            return None
        try:
            from mtc.consciousness.verification_state_assembler import (
                ConsciousnessStateAssembler,
            )

            state = ConsciousnessStateAssembler.from_neural_data(
                self._verification_neural_data(),
                pad_state=pad_state,
                conversation_history=conversation_history,
                temporal_data=temporal_data,
            )
            return verifier.run_verification_suite(state)
        except Exception as e:
            logger.error(f"Consciousness verification failed: {e}")
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

    def _consolidation_strength(self) -> Optional[float]:
        """Average consolidation strength, or None without a consolidator."""
        summary_fn = getattr(
            getattr(self, "memory_consolidator", None),
            "get_consolidation_summary",
            None,
        )
        if not callable(summary_fn):
            return None
        try:
            summary = summary_fn()
        except Exception as e:
            logger.debug(f"Consolidation summary unavailable: {e}")
            return None
        if not isinstance(summary, dict):
            return None
        return _finite(summary.get("avg_consolidation_strength"))

    @staticmethod
    def _component_metric(component: Any, method: str) -> Optional[Any]:
        """``component.method()``, or None without the component or on error."""
        fn = getattr(component, method, None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception as e:
            logger.debug(f"{method} unavailable: {e}")
            return None

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics across all neural systems.

        ``memory.consolidation_strength``, ``intuition`` and ``htm_metrics``
        are None when their component is absent; ``components`` is the
        ``available_components()`` map.
        """
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

        htm = getattr(self.memory_consolidator, "htm", None)
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
                "consolidation_strength": self._consolidation_strength(),
            },
            "layers": {
                "snn_active": self.state.snn_active,
                "lsm_active": self.state.lsm_active,
                "htm_active": self.state.htm_active,
            },
            "neural_bridge": bridge_metrics,
            "intuition": self._component_metric(
                self.lsm_bridge, "get_intuitive_summary"
            ),
            "htm_metrics": self._component_metric(htm, "get_metrics"),
            "enhanced_workspace": (
                self.enhanced_workspace.get_statistics()
                if self.enhanced_workspace
                else {"available": False}
            ),
            "attention_schema": attention_schema_metrics,
            "components": self.available_components(),
        }

    def simulate_day(
        self,
        n_experiences: int = 20,
        sleep_at_end: bool = True,
        sleep_duration: str = "medium",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a day of synthetic experiences through ``process_experience``.

        The emotions are stimuli drawn by a local generator (``seed`` makes the
        day reproducible; the global random state is untouched): positive in the
        first five experiences, curious in the next five, a mix in the next
        five and calm after that. They are inputs, not measurements, and every
        experience has the type ``simulated_experience``. With ``sleep_at_end``
        the day closes with ``sleep(sleep_duration)``.

        Returns:
            ``n_experiences``, ``elapsed_time_sec``, ``avg_experience_time_sec``,
            ``consolidation`` (the sleep report, or None) and ``final_metrics``.
        """
        if n_experiences < 1:
            raise ValueError("n_experiences must be at least 1")
        rng = np.random.default_rng(seed)
        afternoon = (
            {"trust": 0.7, "love": 0.6},
            {"curiosity": 0.8, "anticipation": 0.6},
            {"joy": 0.7, "surprise": 0.5},
        )
        logger.info(f"Simulating a day of {n_experiences} experiences")
        start_time = datetime.now()
        for i in range(n_experiences):
            if i < 5:
                emotions = {
                    "joy": float(rng.uniform(0.6, 0.9)),
                    "anticipation": float(rng.uniform(0.5, 0.8)),
                }
            elif i < 10:
                emotions = {
                    "curiosity": float(rng.uniform(0.7, 0.9)),
                    "joy": float(rng.uniform(0.4, 0.7)),
                }
            elif i < 15:
                emotions = dict(afternoon[int(rng.integers(len(afternoon)))])
            else:
                emotions = {
                    "trust": float(rng.uniform(0.6, 0.8)),
                    "love": float(rng.uniform(0.5, 0.7)),
                }
            self.process_experience(
                experience_type="simulated_experience",
                emotions=emotions,
                context=f"Experience {i + 1} of the day",
            )

        consolidation_report = None
        if sleep_at_end:
            consolidation_report = self.sleep(duration=sleep_duration)

        elapsed = (datetime.now() - start_time).total_seconds()
        summary = {
            "n_experiences": n_experiences,
            "elapsed_time_sec": elapsed,
            "avg_experience_time_sec": elapsed / n_experiences,
            "consolidation": consolidation_report,
            "final_metrics": self.get_system_metrics(),
        }
        logger.info(
            f"Day complete: {n_experiences} experiences in {elapsed:.1f}s, "
            f"{summary['final_metrics']['memory']['semantic_patterns']} patterns"
        )
        return summary

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

    # ======== PERSISTENCE ========

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize counters, the last sleep time and the experience buffer.

        The experience buffer is stored as lightweight entries (timestamp,
        type, emotions, context, valence, arousal), so the state contains the
        raw experience context text; treat persisted orchestrator state as
        private data. Substrate results are not persisted.
        """
        experiences: List[Dict[str, Any]] = []
        for exp in getattr(self, "experience_buffer", None) or []:
            try:
                experiences.append(
                    {
                        "timestamp": exp.timestamp.isoformat(),
                        "experience_type": str(exp.experience_type),
                        "emotions": {
                            str(k): float(v) for k, v in (exp.emotions or {}).items()
                        },
                        "context": str(exp.context or ""),
                        "valence": float(exp.valence),
                        "arousal": float(exp.arousal),
                    }
                )
            except (AttributeError, TypeError, ValueError):
                continue

        state = getattr(self, "state", None)
        last_sleep = getattr(self, "last_sleep", None)
        return {
            "scalars": {
                "total_experiences": int(getattr(self, "total_experiences", 0)),
                "total_sleep_cycles": int(getattr(self, "total_sleep_cycles", 0)),
                "binds_run": int(getattr(self, "_binds_run", 0)),
                "last_sleep": last_sleep.isoformat() if last_sleep else None,
                "experiences_processed": int(
                    getattr(state, "experiences_processed", 0)
                ),
                "episodic_memories_count": int(
                    getattr(state, "episodic_memories_count", 0)
                ),
                "semantic_patterns_count": int(
                    getattr(state, "semantic_patterns_count", 0)
                ),
            },
            "arrays": {},
            "histories": {"experience_buffer": experiences},
        }

    @staticmethod
    def _experience_from_entry(entry: Any) -> Optional[Experience]:
        """Rebuild one persisted experience, or None when it is malformed."""
        if not isinstance(entry, dict):
            return None
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
        except (KeyError, TypeError, ValueError):
            return None
        experience_type = entry.get("experience_type")
        context = entry.get("context", "")
        emotions = entry.get("emotions", {})
        if not isinstance(experience_type, str) or not isinstance(context, str):
            return None
        if not isinstance(emotions, dict):
            return None
        clean_emotions: Dict[str, float] = {}
        for name, value in emotions.items():
            number = _finite(value)
            if not isinstance(name, str) or number is None:
                return None
            clean_emotions[name] = number
        valence = _finite(entry.get("valence"))
        arousal = _finite(entry.get("arousal"))
        if valence is None or arousal is None:
            return None
        return Experience(
            timestamp=timestamp,
            experience_type=experience_type,
            emotions=clean_emotions,
            context=context,
            valence=valence,
            arousal=arousal,
        )

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore counters, the last sleep time and the experience buffer.

        Malformed entries are skipped with a logged warning; a restore never
        raises on bad data.
        """
        scalars = state.get("scalars") or {}
        histories = state.get("histories") or {}

        for key in ("total_experiences", "total_sleep_cycles"):
            value = _as_count(scalars.get(key))
            if value is not None:
                setattr(self, key, value)
        binds = _as_count(scalars.get("binds_run"))
        if binds is not None:
            self._binds_run = binds
        consciousness_state = getattr(self, "state", None)
        if consciousness_state is not None:
            for key in (
                "experiences_processed",
                "episodic_memories_count",
                "semantic_patterns_count",
            ):
                value = _as_count(scalars.get(key))
                if value is not None:
                    setattr(consciousness_state, key, value)

        last_sleep = scalars.get("last_sleep")
        if last_sleep is not None:
            try:
                self.last_sleep = datetime.fromisoformat(last_sleep)
            except (TypeError, ValueError):
                logger.warning("Neural orchestrator restore skipped a bad last_sleep")

        entries = histories.get("experience_buffer")
        if entries is None:
            return
        if not isinstance(entries, list):
            logger.warning(
                "Neural orchestrator restore skipped a bad experience buffer"
            )
            return
        rebuilt: List[Experience] = []
        skipped = 0
        for entry in entries:
            experience = self._experience_from_entry(entry)
            if experience is None:
                skipped += 1
            else:
                rebuilt.append(experience)
        limit = getattr(self, "MAX_EXPERIENCE_BUFFER", None)
        if isinstance(limit, int) and limit > 0:
            rebuilt = rebuilt[-limit:]
        self.experience_buffer = rebuilt
        if skipped:
            logger.warning(
                f"Neural orchestrator restore skipped {skipped} malformed "
                f"experience entries"
            )

    # ======== SHUTDOWN ========

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all neural systems and free memory.

        Tolerates a partly built instance: every step checks that its
        component exists.

        Cleanup order:
        1. Stop continuous thought machine (awaited)
        2. Clear experience buffer
        3. Clear LSM bridge state history
        4. Clear thought generator/stream caches
        5. Close data connections
        6. Force garbage collection
        7. Clear accelerator memory cache
        """
        import gc

        logger.info("Shutting down Neural Orchestrator...")

        # 1. Stop the continuous thought machine. Its stop() is a coroutine
        # and must be awaited, or the loop keeps running through shutdown.
        machine = getattr(self, "continuous_thought_machine", None)
        if machine is not None:
            logger.info("Stopping continuous thought machine...")
            stop = getattr(machine, "stop", None)
            if callable(stop):
                try:
                    result = stop()
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    logger.warning(f"Error stopping continuous thought machine: {e}")
            self.continuous_thought_machine = None

        # 2. Clear experience buffer
        buffer = getattr(self, "experience_buffer", None)
        if buffer:
            logger.info(f"Clearing {len(buffer)} experiences from buffer...")
            buffer.clear()

        # 3. Clear LSM bridge state history and trajectories
        bridge = getattr(self, "lsm_bridge", None)
        if bridge:
            if hasattr(bridge, "state_history"):
                bridge.state_history.clear()
            if hasattr(bridge, "reservoir_trajectory"):
                bridge.reservoir_trajectory.clear()
            logger.info("Cleared LSM bridge state history")

        # 4. Clear thought generator caches
        generator = getattr(self, "thought_generator", None)
        if generator:
            if hasattr(generator, "recent_thought_types"):
                generator.recent_thought_types.clear()
            logger.info("Cleared thought generator caches")

        # 5. Clear thought stream buffer
        stream = getattr(self, "thought_stream", None)
        if stream:
            if hasattr(stream, "thought_buffer"):
                stream.thought_buffer.clear()
            logger.info("Cleared thought stream buffer")

        # 6. Close data connections (sync or async close)
        manager = getattr(self, "neural_data_manager", None)
        close = getattr(manager, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
                logger.info("Data manager closed")
            except Exception as e:
                logger.warning(f"Error closing data manager: {e}")

        # 7. Force garbage collection
        gc.collect()
        logger.info("Garbage collection complete")

        # 8. Clear accelerator memory cache (MPS or CUDA) when torch is loaded
        released = _release_accelerator_cache(include_cuda=True)
        if released:
            logger.info(f"{released.upper()} cache cleared")

        # Update state
        state = getattr(self, "state", None)
        if state is not None:
            state.is_awake = False

        logger.info("Neural Orchestrator shutdown complete")
