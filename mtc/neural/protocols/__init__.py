"""
Neural Protocol Layer
=================================

Structured communication between neural substrates (SNN, LSM, HTM, CTM).
Typed messages, priority routing, phase-locked synchronization, and
efficient binary serialization.
"""

from mtc.neural.protocols.message_format import (
    NeuralMessage,
    NeuralMessageType,
    MessagePriority,
    SpikeTrain,
    EmotionalState,
    CreativeVector,
    TemporalPattern,
)
from mtc.neural.protocols.routing import (
    NeuralRouter,
    RoutingRule,
    RouteType,
)
from mtc.neural.protocols.synchronization import (
    NeuralSynchronizer,
    SyncState,
    PhaseLocker,
    ClockReference,
)
from mtc.neural.protocols.serialization import (
    NeuralSerializer,
    SerializationFormat,
)

__all__ = [
    # Message format
    "NeuralMessage",
    "NeuralMessageType",
    "MessagePriority",
    "SpikeTrain",
    "EmotionalState",
    "CreativeVector",
    "TemporalPattern",
    # Routing
    "NeuralRouter",
    "RoutingRule",
    "RouteType",
    # Synchronization
    "NeuralSynchronizer",
    "SyncState",
    "PhaseLocker",
    "ClockReference",
    # Serialization
    "NeuralSerializer",
    "SerializationFormat",
]
