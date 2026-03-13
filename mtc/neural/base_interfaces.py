"""
Neural Base Interfaces -- Abstract Foundations for All Substrates
===================================================================

Defines the contract that every neural system (SNN, LSM, HTM, CTM)
must satisfy. This enables the NeuralOrchestrator and NeuralRouter
to manage systems uniformly without knowing their internals.

Also provides:
- NeuralSystemType enum for substrate identification
- SystemHealth enum for lifecycle tracking
- ResourceAllocation for memory/compute budgeting
- NeuralSystemRegistry for system discovery

Created: March 7, 2026
Authors: Multi-Theory Consciousness Contributors
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generic, List, Optional, TypeVar, Any
import logging

from mtc.neural.protocols.message_format import NeuralMessage

logger = logging.getLogger(__name__)


class NeuralSystemType(Enum):
    """Identifiers for the four neural substrates."""

    SNN = "snn"  # Spiking Neural Network -- conscious processing
    LSM = "lsm"  # Liquid State Machine -- subconscious reservoir
    HTM = "htm"  # Hierarchical Temporal Memory -- consolidation
    CTM = "ctm"  # Continuous Thought Machine -- background thinking


class SystemHealth(Enum):
    """Lifecycle health states."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class ResourceAllocation:
    """Memory and compute budget for a neural system."""

    memory_gb: float = 1.0
    cpu_cores: int = 2
    priority: int = 5  # 0=highest, 9=lowest
    gpu_memory_gb: float = 0.0
    max_memory_gb: float = 4.0
    min_memory_gb: float = 0.5
    burst_memory_gb: float = 8.0
    burst_duration_seconds: float = 30.0


@dataclass
class NeuralSystemConfig:
    """Configuration for a neural system instance."""

    system_type: NeuralSystemType
    system_id: str
    resources: ResourceAllocation = field(default_factory=ResourceAllocation)
    neural_params: Dict[str, Any] = field(default_factory=dict)
    state_save_path: Optional[str] = None
    max_throughput_msgs_sec: float = 1000.0


# --- Abstract State ---


@dataclass
class NeuralState(ABC):
    """
    Base class for substrate-specific state snapshots.

    Each substrate (SNN, LSM, etc.) extends this with its own
    state fields while inheriting the common envelope.
    """

    system_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence_number: int = 0

    @abstractmethod
    def get_memory_usage(self) -> float:
        """Estimated memory usage in GB."""
        ...


@dataclass
class SNNState(NeuralState):
    """State snapshot for the Spiking Neural Network."""

    neuron_count: int = 0
    active_neurons: int = 0
    mean_firing_rate: float = 0.0
    spike_count: int = 0
    stdp_updates: int = 0

    def get_memory_usage(self) -> float:
        return self.neuron_count * 4 * 8 / (1024**3)  # ~4 params * 8B each


@dataclass
class LSMState(NeuralState):
    """State snapshot for the Liquid State Machine."""

    reservoir_size: int = 0
    spectral_radius: float = 0.0
    mean_activation: float = 0.0
    edge_of_chaos_metric: float = 0.0

    def get_memory_usage(self) -> float:
        # Reservoir weight matrix is N x N
        return (self.reservoir_size**2) * 8 / (1024**3)


@dataclass
class HTMState(NeuralState):
    """State snapshot for the Hierarchical Temporal Memory."""

    column_count: int = 0
    cell_count: int = 0
    active_columns: int = 0
    predicted_columns: int = 0
    anomaly_score: float = 0.0

    def get_memory_usage(self) -> float:
        return self.cell_count * 32 / (1024**3)  # ~32B per cell (SDR + permanence)


@dataclass
class CTMState(NeuralState):
    """State snapshot for the Continuous Thought Machine."""

    thoughts_generated: int = 0
    active_threads: int = 0
    mean_thought_quality: float = 0.0
    queue_depth: int = 0

    def get_memory_usage(self) -> float:
        return 0.1  # CTM is lightweight -- mostly orchestration


# --- Abstract System ---

T = TypeVar("T", bound=NeuralState)


class NeuralSystem(ABC, Generic[T]):
    """
    Abstract base class for all neural substrates.

    Every neural system can be initialized, shut down, process
    messages, update its state, and save/load its weights.
    """

    def __init__(self, config: NeuralSystemConfig):
        self.config = config
        self.health = SystemHealth.INITIALIZING
        self._message_count = 0

    @property
    def system_id(self) -> str:
        return self.config.system_id

    @property
    def system_type(self) -> NeuralSystemType:
        return self.config.system_type

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the system and transition to HEALTHY."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shut down the system."""
        ...

    @abstractmethod
    async def process_message(self, message: NeuralMessage) -> Optional[NeuralMessage]:
        """
        Process an incoming neural message.

        Returns an optional response message (e.g., SNN processes
        a spike train and returns an emotional state update).
        """
        ...

    @abstractmethod
    async def update(self, dt: float) -> None:
        """Advance the system by dt seconds."""
        ...

    @abstractmethod
    def get_state(self) -> T:
        """Return a snapshot of the current system state."""
        ...

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return health diagnostics."""
        ...

    @abstractmethod
    async def save_state(self, path: Optional[str] = None) -> None:
        """Persist system state to disk."""
        ...

    @abstractmethod
    async def load_state(self, path: Optional[str] = None) -> None:
        """Restore system state from disk."""
        ...


# --- Registry ---


class NeuralSystemRegistry:
    """
    Registry for discovering and managing neural system instances.

    The NeuralOrchestrator uses this to look up systems by ID or
    type without hard-coding references to specific implementations.
    """

    def __init__(self):
        self._systems: Dict[str, NeuralSystem] = {}

    def register(self, system: NeuralSystem) -> None:
        """Register a neural system."""
        self._systems[system.system_id] = system
        logger.info(
            f"Registered {system.system_type.value} system: {system.system_id}"
        )

    def unregister(self, system_id: str) -> None:
        """Unregister a neural system."""
        self._systems.pop(system_id, None)

    def get_system(self, system_id: str) -> Optional[NeuralSystem]:
        """Look up a system by its ID."""
        return self._systems.get(system_id)

    def get_systems_by_type(self, system_type: NeuralSystemType) -> List[NeuralSystem]:
        """Return all systems of a given type."""
        return [s for s in self._systems.values() if s.system_type == system_type]

    def get_all_systems(self) -> List[NeuralSystem]:
        """Return all registered systems."""
        return list(self._systems.values())

    def get_health_summary(self) -> Dict[str, str]:
        """Return health status for all systems."""
        return {sid: sys.health.value for sid, sys in self._systems.items()}
