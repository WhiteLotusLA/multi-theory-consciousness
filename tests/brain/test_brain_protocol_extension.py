"""The conversational additions to the brain protocol; the base contract stands."""

import inspect

import pytest

from mtc.brain import protocol
from mtc.brain.protocol import (
    BrainProtocol,
    BrainReply,
    BrainUnavailableError,
    CapabilityUnavailableError,
    ConfigurationError,
    ConversationalBrain,
    FakeBrain,
    NoAnswerError,
)


def test_fake_brain_still_satisfies_brain_protocol():
    brain = FakeBrain()
    assert isinstance(brain, BrainProtocol)
    # FakeBrain is the base protocol's test double: it only thinks.
    assert not isinstance(brain, ConversationalBrain)


def test_conversational_brain_members():
    for name in ("think", "think_stream", "think_with_context", "generate"):
        assert hasattr(ConversationalBrain, name), name
    signature = inspect.signature(ConversationalBrain.think_with_context)
    params = signature.parameters
    assert "current_prompt" in params
    for keyword in (
        "conversation_history",
        "core_memory",
        "ctm_thoughts",
        "thought_gists",
        "coherence_context",
        "retrieved_memories",
        "retrieved_concepts",
        "consciousness_context",
        "generation_params",
        "behavioral_directives",
        "consciousness_self_awareness",
        "conversation_summary",
        "window_notes",
        "image",
        "speaker",
        "stream_callback",
        "neural_state",
        "attention_guidance",
        "metacognitive_insights",
        "active_inference_context",
        "global_workspace_context",
        "max_tokens",
        "last_turn_at",
    ):
        assert params[keyword].kind is inspect.Parameter.KEYWORD_ONLY, keyword
        assert params[keyword].default is None, keyword

    class Complete:
        async def think(
            self,
            prompt,
            system_prompt="",
            max_tokens=512,
            temperature=0.7,
            thinking_mode=False,
        ):
            return BrainReply("x")

        async def think_stream(
            self,
            prompt,
            system_prompt="",
            max_tokens=512,
            temperature=0.7,
            thinking_mode=False,
        ):
            yield "x"

        async def think_with_context(self, current_prompt, **kwargs):
            return BrainReply("x")

        async def generate(self, prompt, max_tokens=256, temperature=0.7):
            return "x"

    class Partial:
        async def think(self, prompt, **kwargs):
            return BrainReply("x")

    assert isinstance(Complete(), ConversationalBrain)
    assert isinstance(Complete(), BrainProtocol)
    assert not isinstance(Partial(), ConversationalBrain)


def test_no_answer_error_reason_default():
    error = NoAnswerError("nothing came back")
    assert error.reason == "empty"
    assert str(error) == "nothing came back"
    assert NoAnswerError("only reasoning", reason="reasoning_only").reason == (
        "reasoning_only"
    )
    assert NoAnswerError(reason="loop").reason == "loop"
    assert set(protocol.NO_ANSWER_REASONS) == {"empty", "reasoning_only", "loop"}
    with pytest.raises(ValueError):
        NoAnswerError("x", reason="something else")
    assert issubclass(NoAnswerError, RuntimeError)


def test_brain_reply_new_fields_default_none():
    reply = BrainReply("hello")
    assert reply.input_tokens is None
    assert reply.provider_id is None
    assert reply.finish_reason is None and reply.output_tokens is None
    full = BrainReply("hi", "stop", 3, input_tokens=10, provider_id="mock")
    assert (full.output_tokens, full.input_tokens, full.provider_id) == (3, 10, "mock")


def test_error_types_exported():
    import mtc.brain.protocol as module

    assert issubclass(BrainUnavailableError, RuntimeError)
    assert issubclass(CapabilityUnavailableError, RuntimeError)
    assert issubclass(ConfigurationError, ValueError)
    for name in (
        "BrainUnavailableError",
        "CapabilityUnavailableError",
        "ConfigurationError",
        "ConversationalBrain",
    ):
        assert name in module.__all__, name
