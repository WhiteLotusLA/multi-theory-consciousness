"""Brain package: the protocol, the runtime brain, the prompt assembler.

Importing this package loads no HTTP client or vendor SDK; the provider
modules under ``mtc.brain.providers`` are imported by path when needed.
"""

from mtc.brain.brain import LLMBrain, get_brain
from mtc.brain.identity_statement import load_identity_statement
from mtc.brain.prompt import PromptAssembler, SpeakerView
from mtc.brain.protocol import (
    BrainProtocol,
    BrainReply,
    BrainUnavailableError,
    CapabilityUnavailableError,
    ConfigurationError,
    ConversationalBrain,
    FakeBrain,
    NoAnswerError,
    strip_reasoning_blocks,
)

__all__ = [
    "BrainProtocol",
    "BrainReply",
    "BrainUnavailableError",
    "CapabilityUnavailableError",
    "ConfigurationError",
    "ConversationalBrain",
    "FakeBrain",
    "LLMBrain",
    "NoAnswerError",
    "PromptAssembler",
    "SpeakerView",
    "get_brain",
    "load_identity_statement",
    "strip_reasoning_blocks",
]
