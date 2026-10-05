"""The deterministic mock provider: labelled replies, scripted events, no network."""

import hashlib
import socket

import pytest

from mtc.brain.providers.base import CompletionParams, ModelProvider
from mtc.brain.providers.mock_provider import MockEvent, MockProvider
from mtc.brain.thinking import split_thinking
from mtc.brain.thinking_loop_detector import ThinkingLoopDetected, ThinkingLoopDetector


def _params(text="What is the boiling point of water?", **overrides):
    values = dict(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "An earlier question."},
            {"role": "assistant", "content": "An earlier answer."},
            {"role": "user", "content": text},
        ]
    )
    values.update(overrides)
    return CompletionParams(**values)


async def _collect(provider, params):
    return [chunk async for chunk in provider.stream(params)]


def test_satisfies_model_provider_protocol():
    provider = MockProvider()
    assert isinstance(provider, ModelProvider)
    info = provider.model_info()
    assert (info.id, info.provider_type, info.cost) == ("mock", "mock", "none")
    assert info.context_window == 32768
    assert provider.supports_thinking() is True
    assert MockProvider(reasoning=False).supports_thinking() is False


@pytest.mark.asyncio
async def test_mock_is_deterministic_across_instances():
    text = "What is the boiling point of water?"
    first = await MockProvider().complete(_params(text))
    second = await MockProvider().complete(_params(text))
    other = await MockProvider().complete(_params("A different question."))
    assert first.text == second.text
    assert first.text != other.text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    assert first.text.endswith(f"({digest})")


@pytest.mark.asyncio
async def test_mock_reply_is_labelled():
    long_text = "x" * 100
    result = await MockProvider().complete(_params(long_text))
    assert result.text.startswith("[mock]")
    assert result.text.startswith("[mock] reply to: " + "x" * 60 + " (")
    assert result.finish_reason == "stop"
    assert result.input_tokens is not None and result.output_tokens is not None


@pytest.mark.asyncio
async def test_mock_thinking_block_when_requested():
    provider = MockProvider()
    result = await provider.complete(_params(thinking_mode=True))
    assert result.reasoning is not None
    assert result.reasoning.startswith("mock reasoning ")
    assert len(result.reasoning.split()[-1]) == 8
    quiet = await provider.complete(_params(thinking_mode=False))
    assert quiet.reasoning is None


@pytest.mark.asyncio
async def test_mock_no_reasoning_mode():
    provider = MockProvider(reasoning=False)
    result = await provider.complete(_params(thinking_mode=True))
    assert result.reasoning is None
    chunks = await _collect(provider, _params(thinking_mode=True))
    assert "<think>" not in "".join(chunks)


@pytest.mark.asyncio
async def test_mock_scripted_events():
    provider = MockProvider(
        replies=[
            MockEvent.EMPTY,
            MockEvent.REASONING_ONLY,
            MockEvent.LOOP,
            RuntimeError("scripted failure"),
            "<think>planned</think>Scripted words.",
            "Plain scripted words.",
        ]
    )
    empty = await provider.complete(_params(thinking_mode=True))
    assert empty.text == "" and empty.reasoning is None
    only = await provider.complete(_params(thinking_mode=True))
    assert only.text == "" and only.reasoning
    loop = await provider.complete(_params(thinking_mode=True))
    with pytest.raises(ThinkingLoopDetected):
        ThinkingLoopDetector().check(loop.reasoning)
    with pytest.raises(RuntimeError, match="scripted failure"):
        await provider.complete(_params())
    scripted = await provider.complete(_params())
    assert (scripted.reasoning, scripted.text) == ("planned", "Scripted words.")
    plain = await provider.complete(_params())
    assert plain.text == "Plain scripted words." and plain.reasoning is None
    # The script is exhausted: generated replies follow.
    assert (await provider.complete(_params())).text.startswith("[mock]")
    assert len(provider.requests) == 7
    assert provider.requests[0].thinking_mode is True


@pytest.mark.asyncio
async def test_mock_loop_stream_trips_the_detector():
    provider = MockProvider(replies=[MockEvent.LOOP])
    detector = ThinkingLoopDetector()
    with pytest.raises(ThinkingLoopDetected):
        async for chunk in provider.stream(_params(thinking_mode=True)):
            detector.feed(chunk)


@pytest.mark.asyncio
async def test_mock_stream_chunks_reassemble():
    provider = MockProvider(chunk_chars=5)
    complete = await provider.complete(_params(thinking_mode=True))
    chunks = await _collect(MockProvider(chunk_chars=5), _params(thinking_mode=True))
    assert chunks[0] == "<think>"
    assert "</think>" in chunks
    reasoning, spoken = split_thinking("".join(chunks))
    assert (reasoning, spoken) == (complete.reasoning, complete.text)
    assert all(len(c) <= 5 for c in chunks if c not in ("<think>", "</think>"))
    empty = await _collect(MockProvider(replies=[MockEvent.EMPTY]), _params())
    assert empty == []
    only = await _collect(
        MockProvider(replies=[MockEvent.REASONING_ONLY]), _params(thinking_mode=True)
    )
    reasoning, spoken = split_thinking("".join(only))
    assert reasoning and spoken == ""


@pytest.mark.asyncio
async def test_mock_health_and_lifecycle():
    provider = MockProvider()
    await provider.initialize()
    status = await provider.health_check()
    assert status.ok is True and status.verified is True
    await provider.shutdown()
    down = await MockProvider(healthy=False).health_check()
    assert down.ok is False


@pytest.mark.asyncio
async def test_mock_never_touches_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the mock provider opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    provider = MockProvider(latency_s=0.0)
    await provider.initialize()
    await provider.complete(_params(thinking_mode=True))
    await _collect(provider, _params(thinking_mode=True))
    await provider.health_check()
    await provider.shutdown()
