"""
Mock Provider
=============

A deterministic provider for tests, examples and the offline demo. It opens
no socket and imports no client. Every generated reply is labelled:

    [mock] reply to: <first 60 characters of the last user message> (<8 hex>)

where the hex digits are the start of the SHA-256 digest of that message
(never Python's salted ``hash``), so replies are identical across processes.
With ``reasoning`` on and a caller asking for thinking, the reply carries the
reasoning ``mock reasoning <8 hex>``.

``replies`` scripts the next calls, in order: a string (inline think tags are
split into reasoning and text), an exception to raise, or a ``MockEvent``:
``EMPTY`` (no text and no reasoning), ``REASONING_ONLY`` (reasoning and no
text) or ``LOOP`` (a reasoning block repeating one sentence until the
default loop detector fires, and no text). When the script runs out,
generated replies follow. Each request is recorded in ``requests``.

Usage counts are synthetic (one token per four characters, rounded up),
because no model ran. No config keys are read and no state is persisted.
"""

import asyncio
import dataclasses
import enum
import hashlib
import math
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional, Sequence, Tuple, Union

from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ModelInfo,
)
from mtc.brain.thinking import THINK_CLOSE, THINK_OPEN, split_thinking
from mtc.brain.thinking_loop_detector import ThinkingLoopDetected, ThinkingLoopDetector

__all__ = ["MockEvent", "MockProvider"]

_LOOP_SENTENCE = "The mock keeps returning to the same unfinished thought. "
_LOOP_CAP = 1000


class MockEvent(enum.Enum):
    """Scripted reply shapes that exercise the no-answer paths."""

    EMPTY = "empty"
    REASONING_ONLY = "reasoning_only"
    LOOP = "loop"


ScriptedReply = Union[str, MockEvent, BaseException]


def _loop_reasoning() -> str:
    """One sentence repeated until the default loop detector fires."""
    detector = ThinkingLoopDetector()
    text = ""
    for _ in range(_LOOP_CAP):
        text += _LOOP_SENTENCE
        try:
            detector.check(text)
        except ThinkingLoopDetected:
            return text
    return text


def _last_user_text(messages: Sequence[dict]) -> str:
    for message in reversed(list(messages)):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content or "")
    return ""


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


class MockProvider:
    """A labelled, deterministic stand-in for a model server."""

    def __init__(
        self,
        replies: Optional[Sequence[ScriptedReply]] = None,
        latency_s: float = 0.0,
        reasoning: bool = True,
        chunk_chars: int = 16,
        provider_id: str = "mock",
        context_window: int = 32768,
        *,
        healthy: bool = True,
    ) -> None:
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        self._script: List[ScriptedReply] = list(replies or [])
        self.latency_s = float(latency_s)
        self.reasoning = bool(reasoning)
        self.chunk_chars = int(chunk_chars)
        self.provider_id = provider_id
        self.context_window = int(context_window)
        self.healthy = bool(healthy)
        self.requests: List[CompletionParams] = []

    # -- protocol surface ---------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return self.model_info()

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.provider_id,
            name="mock provider",
            provider_type="mock",
            context_window=self.context_window,
            cost="none",
            supports_thinking=self.reasoning,
        )

    def supports_thinking(self) -> bool:
        return self.reasoning

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def health_check(self) -> HealthStatus:
        detail = "mock provider" if self.healthy else "mock provider set unhealthy"
        return HealthStatus(self.healthy, detail, datetime.now(timezone.utc), True)

    async def complete(self, params: CompletionParams) -> CompletionResult:
        reasoning, text = await self._reply(params)
        prompt = "".join(str(m.get("content", "")) for m in params.messages)
        output = _tokens((reasoning or "") + text)
        return CompletionResult(
            text=text,
            reasoning=reasoning,
            input_tokens=_tokens(prompt),
            output_tokens=output,
            total_tokens=_tokens(prompt) + output,
            finish_reason="stop",
        )

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        reasoning, text = await self._reply(params)
        if reasoning is not None:
            yield THINK_OPEN
            for chunk in self._chunks(reasoning):
                yield chunk
            yield THINK_CLOSE
        for chunk in self._chunks(text):
            yield chunk

    # -- helpers ------------------------------------------------------------

    def _chunks(self, text: str):
        for start in range(0, len(text), self.chunk_chars):
            yield text[start : start + self.chunk_chars]

    async def _reply(self, params: CompletionParams) -> Tuple[Optional[str], str]:
        self.requests.append(dataclasses.replace(params))
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        user_text = _last_user_text(params.messages)
        digest = hashlib.sha256(user_text.encode("utf-8")).hexdigest()[:8]
        mock_reasoning = f"mock reasoning {digest}"
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if item is MockEvent.EMPTY:
                return None, ""
            if item is MockEvent.REASONING_ONLY:
                return mock_reasoning, ""
            if item is MockEvent.LOOP:
                return _loop_reasoning(), ""
            reasoning, text = split_thinking(str(item))
            return reasoning, text
        reply = f"[mock] reply to: {user_text[:60]} ({digest})"
        if self.reasoning and params.thinking_mode:
            return mock_reasoning, reply
        return None, reply

    def __repr__(self) -> str:
        return f"MockProvider(provider_id={self.provider_id!r})"
