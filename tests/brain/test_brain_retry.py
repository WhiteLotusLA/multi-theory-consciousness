"""Thinking-loop retries: penalty escalation, the stream sentinel, honest exhaustion."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from mtc.brain import brain as brain_module
from mtc.brain.brain import STREAM_RETRY_SENTINEL, LLMBrain
from mtc.brain.families import get_family
from mtc.brain.prompt import PromptAssembler
from mtc.brain.protocol import NoAnswerError
from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ModelInfo,
)
from mtc.brain.providers.mock_provider import MockEvent, MockProvider
from mtc.brain.providers.registry import ProviderRegistry
from mtc.brain.thinking import normalize_reasoning, split_thinking
from mtc.core import persona as persona_module
from mtc.core.config import Settings


@pytest.fixture(autouse=True)
def _fresh_warning_memory(monkeypatch):
    monkeypatch.setattr(persona_module, "_WARNED_MISSING", set())


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        LLM_PROVIDER="mock",
        DATA_DIRECTORY=str(tmp_path / "data"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
    )
    values.update(overrides)
    return Settings(**values)


def _brain(
    tmp_path: Path, provider: Optional[Any] = None, **overrides
) -> Tuple[LLMBrain, Any, Settings]:
    settings = _settings(tmp_path, **overrides)
    provider = provider if provider is not None else MockProvider()
    registry = ProviderRegistry(state_path=tmp_path / "data" / "provider_state.json")
    registry.register(provider)
    registry.set_active(provider.model_info().id)
    assembler = PromptAssembler(settings=settings, identity_statement="")
    brain = LLMBrain(registry=registry, assembler=assembler, settings=settings)
    return brain, provider, settings


class ReasoningProvider:
    """Returns one scripted result with reasoning, whatever the request."""

    def __init__(self, result: CompletionResult):
        self.result = result
        self.requests: List[CompletionParams] = []

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="fake",
            name="scripted",
            provider_type="local",
            context_window=4096,
            cost="local",
            supports_thinking=True,
        )

    @property
    def info(self) -> ModelInfo:
        return self.model_info()

    def supports_thinking(self) -> bool:
        return True

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def health_check(self) -> HealthStatus:
        return HealthStatus(True, "scripted", datetime.now(timezone.utc), True)

    async def complete(self, params: CompletionParams) -> CompletionResult:
        self.requests.append(params)
        return self.result

    async def stream(self, params: CompletionParams):
        self.requests.append(params)
        yield "<think>"
        yield self.result.reasoning or ""
        yield "</think>"
        yield self.result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["generic", "qwen"])
async def test_retry_escalates_penalty_on_loop(tmp_path, family):
    brain, mock, settings = _brain(
        tmp_path, MockProvider(replies=[MockEvent.LOOP]), LLM_TEMPLATE_FAMILY=family
    )
    reply = await brain.think("hello", thinking_mode=True)
    assert "[mock] reply to: hello" in reply.text
    assert len(mock.requests) == 2
    field = get_family(family).penalty_field
    first = getattr(mock.requests[0], field)
    second = getattr(mock.requests[1], field)
    if family == "generic":
        assert field == "frequency_penalty"
        assert first is None
        assert mock.requests[0].repetition_penalty is None
    else:
        assert field == "repetition_penalty"
        assert first == pytest.approx(settings.REPEAT_PENALTY)
        assert mock.requests[0].frequency_penalty is None
    assert second == pytest.approx((first or 0.0) + settings.THINKING_LOOP_PENALTY_STEP)
    assert mock.requests[1].thinking_mode is True
    status = brain.status()
    assert status["retries"] == 1
    assert status["calls"] == 2
    assert status["no_answers"]["loop"] == 0


@pytest.mark.asyncio
async def test_stream_retry_sentinel_sent(tmp_path):
    brain, mock, _settings_ = _brain(tmp_path, MockProvider(replies=[MockEvent.LOOP]))
    chunks: List[str] = []
    reply = await brain.think_with_context("hello", stream_callback=chunks.append)
    assert STREAM_RETRY_SENTINEL in chunks
    assert chunks.count(STREAM_RETRY_SENTINEL) == 1
    after = "".join(chunks[chunks.index(STREAM_RETRY_SENTINEL) + 1 :])
    assert normalize_reasoning(after, "generic") == split_thinking(reply.text)
    assert "[mock] reply to: hello" in reply.text
    assert len(mock.requests) == 2
    # The chunks before the sentinel belong to the looping attempt.
    before = "".join(chunks[: chunks.index(STREAM_RETRY_SENTINEL)])
    assert "same unfinished thought" in before

    # think_stream yields the same sentinel and keeps the finished reply.
    streaming, _mock2, _s = _brain(
        tmp_path / "s", MockProvider(replies=[MockEvent.LOOP])
    )
    seen = [
        chunk async for chunk in streaming.think_stream("hello", thinking_mode=True)
    ]
    assert STREAM_RETRY_SENTINEL in seen
    assert streaming.last_stream_reply is not None
    assert "[mock] reply to: hello" in streaming.last_stream_reply.text
    assert streaming.last_stream_reply.provider_id == "mock"

    # An async callback is awaited.
    received: List[str] = []

    async def collect(chunk: str) -> None:
        received.append(chunk)

    plain, _m3, _s3 = _brain(tmp_path / "a")
    reply = await plain.think_with_context("hello", stream_callback=collect)
    assert "".join(received).replace("</think>", "</think>\n\n") == reply.text


@pytest.mark.asyncio
async def test_exhausted_retries_raise_no_answer_reason_loop(tmp_path):
    brain, mock, _s = _brain(
        tmp_path,
        MockProvider(replies=[MockEvent.LOOP, MockEvent.LOOP]),
        THINKING_LOOP_MAX_RETRIES=1,
    )
    with pytest.raises(NoAnswerError) as caught:
        await brain.think("hello", thinking_mode=True)
    assert caught.value.reason == "loop"
    assert len(mock.requests) == 2
    status = brain.status()
    assert status["no_answers"]["loop"] == 1
    assert status["retries"] == 1

    # Streaming exhausts the same way and no reply is kept.
    streaming, smock, _s2 = _brain(
        tmp_path / "s",
        MockProvider(replies=[MockEvent.LOOP, MockEvent.LOOP]),
        THINKING_LOOP_MAX_RETRIES=1,
    )
    chunks: List[str] = []
    with pytest.raises(NoAnswerError) as caught:
        await streaming.think_with_context("hello", stream_callback=chunks.append)
    assert caught.value.reason == "loop"
    assert chunks.count(STREAM_RETRY_SENTINEL) == 1
    assert len(smock.requests) == 2

    # With retries set to zero the first loop is already the end.
    none, nmock, _s3 = _brain(
        tmp_path / "n",
        MockProvider(replies=[MockEvent.LOOP]),
        THINKING_LOOP_MAX_RETRIES=0,
    )
    with pytest.raises(NoAnswerError):
        await none.think("hello", thinking_mode=True)
    assert len(nmock.requests) == 1


@pytest.mark.asyncio
async def test_non_thinking_reply_never_runs_detector(tmp_path, monkeypatch):
    class Forbidden:
        @classmethod
        def for_family(cls, family, **kwargs):
            raise AssertionError("the loop detector ran on a non-thinking reply")

        def __init__(self, *args, **kwargs):
            raise AssertionError("the loop detector ran on a non-thinking reply")

    monkeypatch.setattr(brain_module, "ThinkingLoopDetector", Forbidden)
    looping = "The same unfinished thought again. " * 40
    provider = ReasoningProvider(CompletionResult(text="spoken", reasoning=looping))
    brain, _p, _s = _brain(tmp_path, provider)
    reply = await brain.think("hello", thinking_mode=False)
    assert reply.text.startswith("<think>")
    assert reply.text.endswith("\n\nspoken")
    chunks: List[str] = []
    streamed = await brain.think_with_context(
        "hello",
        generation_params={"thinking_mode": False},
        stream_callback=chunks.append,
    )
    assert streamed.text.endswith("spoken")
    assert STREAM_RETRY_SENTINEL not in chunks
    # A looping reasoning-only reply without thinking is an honest no-answer.
    mock_brain, mock, _s2 = _brain(
        tmp_path / "m", MockProvider(replies=[MockEvent.LOOP])
    )
    with pytest.raises(NoAnswerError) as caught:
        await mock_brain.think("hello", thinking_mode=False)
    assert caught.value.reason == "reasoning_only"
    assert len(mock.requests) == 1
