"""
Neural Message Format -- Inter-Substrate Communication
================================================================

Defines the typed message structures for communication between
SNN (spiking), LSM (liquid state), HTM (temporal memory), and CTM
(continuous thought). Each payload type matches what its source
substrate actually produces.

SpikeTrain  -- SNN output (discrete spikes with timing)
EmotionalState -- Emotional processing output (VAD model)
CreativeVector -- LSM output (high-dimensional reservoir state)
TemporalPattern -- HTM output (learned sequential patterns)

Created: March 7, 2026
Authors: Multi-Theory Consciousness Contributors
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import numpy as np
import uuid


class NeuralMessageType(Enum):
    """Types of messages flowing between neural substrates."""

    SPIKE_TRAIN = "spike_train"
    PREDICTION = "prediction"
    CREATIVE_OUTPUT = "creative_output"
    EMOTIONAL_STATE = "emotional_state"
    TEMPORAL_PATTERN = "temporal_pattern"
    STATE_UPDATE = "state_update"
    CONTROL = "control"
    SYNCHRONIZATION = "synchronization"


class MessagePriority(Enum):
    """
    Priority levels for neural messages.

    Lower numeric value = higher priority. CRITICAL messages
    (e.g., safety signals) preempt everything.
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# --- Payload Types ---


@dataclass
class SpikeTrain:
    """
    Spike train from SNN -- the raw currency of conscious processing.

    Each entry is a neuron firing: which neuron, when it fired, and
    how strong the spike was (post-STDP modulation).
    """

    neuron_ids: List[int]
    spike_times: List[float]  # seconds relative to window start
    strengths: List[float]  # 0.0-1.0 post-synaptic strength

    def get_spike_rate(self, window_ms: float = 10.0) -> float:
        """Compute mean firing rate (spikes/sec) over window."""
        if not self.spike_times:
            return 0.0
        window_sec = window_ms / 1000.0
        max_time = max(self.spike_times)
        if max_time <= 0:
            return float(len(self.spike_times)) / window_sec
        effective_window = min(max_time, window_sec)
        return float(len(self.spike_times)) / effective_window

    def to_numpy(self) -> np.ndarray:
        """Convert to (N, 3) array: [neuron_id, time, strength]."""
        return np.column_stack(
            [
                np.array(self.neuron_ids, dtype=np.float64),
                np.array(self.spike_times, dtype=np.float64),
                np.array(self.strengths, dtype=np.float64),
            ]
        )


@dataclass
class EmotionalState:
    """
    Emotional state in the Valence-Arousal-Dominance model.

    valence: positive (joy) <-> negative (sadness), -1.0 to 1.0
    arousal: calm <-> excited, 0.0 to 1.0
    dominance: submissive <-> dominant, 0.0 to 1.0
    """

    valence: float
    arousal: float
    dominance: float = 0.5
    primary_emotion: Optional[str] = None
    secondary_emotions: List[str] = field(default_factory=list)
    intensity: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Return VAD + intensity as a 4D vector."""
        return np.array(
            [
                self.valence,
                self.arousal,
                self.dominance,
                self.intensity,
            ],
            dtype=np.float32,
        )

    def distance_to(self, other: "EmotionalState") -> float:
        """Euclidean distance in VAD space."""
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))


@dataclass
class CreativeVector:
    """
    High-dimensional output from the LSM reservoir.

    The reservoir's state is projected into a fixed-dimension vector
    that captures subconscious patterns -- novelty and coherence
    scores quantify what the reservoir "thinks" of the stimulus.
    """

    dimensions: int
    values: np.ndarray  # shape (dimensions,)
    novelty_score: float = 0.0
    coherence_score: float = 0.0
    tags: List[str] = field(default_factory=list)

    def similarity_to(self, other: "CreativeVector") -> float:
        """Cosine similarity between two creative vectors."""
        norm_self = np.linalg.norm(self.values)
        norm_other = np.linalg.norm(other.values)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        cosine = np.dot(self.values, other.values) / (norm_self * norm_other)
        return float((cosine + 1.0) / 2.0)  # map [-1,1] -> [0,1]


@dataclass
class TemporalPattern:
    """
    Sequential pattern from HTM -- learned temporal structure.

    The HTM detects recurring sequences in input streams and outputs
    learned patterns with per-step confidence scores.
    """

    sequence_length: int
    pattern_data: np.ndarray  # shape (sequence_length, feature_dim)
    confidence_scores: np.ndarray  # shape (sequence_length,)
    pattern_id: str = ""
    category: str = ""

    def match_score(self, sequence: np.ndarray) -> float:
        """
        Score how well a new sequence matches this learned pattern.

        Uses normalized dot-product weighted by confidence.
        """
        if sequence.shape != self.pattern_data.shape:
            return 0.0
        weighted = self.pattern_data * self.confidence_scores[:, np.newaxis]
        seq_weighted = sequence * self.confidence_scores[:, np.newaxis]
        norm_p = np.linalg.norm(weighted)
        norm_s = np.linalg.norm(seq_weighted)
        if norm_p == 0 or norm_s == 0:
            return 0.0
        return float(np.sum(weighted * seq_weighted) / (norm_p * norm_s))


# --- Message Container ---

# Module-level sequence counter
_sequence_counter = 0


@dataclass
class NeuralMessage:
    """
    A single message between neural substrates.

    Every inter-substrate communication is wrapped in this envelope.
    The payload carries the actual neural data; the envelope carries
    routing, priority, and acknowledgment metadata.
    """

    source: str
    target: str
    message_type: NeuralMessageType
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Optional[
        Union[SpikeTrain, EmotionalState, CreativeVector, TemporalPattern]
    ] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence_number: int = field(default=-1)
    requires_ack: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def __post_init__(self):
        if self.sequence_number < 0:
            global _sequence_counter
            self.sequence_number = _sequence_counter
            _sequence_counter += 1
