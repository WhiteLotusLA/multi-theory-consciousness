"""
Anthropic Provider
==================

An optional provider that reaches an Anthropic model through the vendor's
Python SDK (the ``anthropic`` extra, imported only when a client is built).
Using it sends the system prompt and the conversation to an external API.

Request shape (the Messages API):

- ``system`` messages are joined into the top-level ``system`` field; the
  rest go to ``messages`` as user and assistant turns; stop sequences go to
  ``stop_sequences``. Image parts are refused with
  ``CapabilityUnavailableError``: vision is not part of this release.
- Thinking, when a caller asks for it: with ``ANTHROPIC_THINKING_BUDGET`` 0
  the request carries adaptive thinking with a summarized display (current
  models decide how much to think, and without ``display`` they return empty
  thinking text); a positive budget sends a fixed ``budget_tokens`` for
  models that still accept one, and ``max_tokens`` is raised so the budget
  always stays below it. A caller that does not ask sends no ``thinking``
  field, so the model's own default applies.
- Sampling parameters: current models reject ``temperature``, ``top_p`` and
  ``top_k``, so none is sent unless ``ANTHROPIC_SAMPLING_ENABLED`` is true;
  then ``temperature`` (or ``top_p`` when no temperature is given, never both)
  and ``top_k`` are sent, and never while thinking.
- Completions stream internally (the SDK refuses long non-streamed requests)
  and read the final message; ``stream`` maps thinking blocks to standalone
  ``<think>`` and ``</think>`` chunks. Stop reasons map to the shared finish
  vocabulary in ``STOP_REASON_MAP``; an unknown one passes through.

``health_check`` never calls the API: it reports unverified until a real call
succeeds, ok and verified for ``health_ttl_s`` after a success, and the
failure for ``health_ttl_s`` after a failed call.

Config keys read by ``from_settings`` (at call time): ``ANTHROPIC_MODEL`` and
``ANTHROPIC_API_KEY`` (both required), ``ANTHROPIC_MAX_TOKENS``,
``ANTHROPIC_THINKING_BUDGET`` and ``ANTHROPIC_SAMPLING_ENABLED``. The API key
is a secret: it is never logged or shown in ``repr``. No state is persisted.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from mtc.brain.protocol import CapabilityUnavailableError, ConfigurationError
from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ModelInfo,
    ProviderError,
    ProviderTimeoutError,
    has_image_parts,
)
from mtc.brain.thinking import THINK_CLOSE, THINK_OPEN
from mtc.core.config import Settings, get_settings
from mtc.memory.clients import require_client

logger = logging.getLogger(__name__)

MIN_THINKING_BUDGET = 1024

STOP_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "pause_turn": "pause_turn",
    "refusal": "refusal",
}

_IMAGE_REFUSAL = "image input is not part of this release; send text only"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _finish(stop_reason: Optional[str]) -> Optional[str]:
    if stop_reason is None:
        return None
    return STOP_REASON_MAP.get(stop_reason, stop_reason)


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _usage(usage: Any) -> Dict[str, Optional[int]]:
    """Usage counts; cached prompt tokens (written or read) count as input."""
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    base = _int_or_none(getattr(usage, "input_tokens", None))
    output = _int_or_none(getattr(usage, "output_tokens", None))
    if base is not None:
        for name in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            base += _int_or_none(getattr(usage, name, None)) or 0
    total = base + output if base is not None and output is not None else None
    return {"input_tokens": base, "output_tokens": output, "total_tokens": total}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


def translate_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Split chat messages into the ``system`` text and the turn list."""
    system_parts: List[str] = []
    turns: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                system_parts.append(text)
        elif role in ("user", "assistant"):
            if isinstance(content, list):
                content = [
                    {"type": "text", "text": str(part.get("text", ""))}
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
            turns.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported message role {role!r} for Anthropic")
    return "\n\n".join(system_parts), turns


class AnthropicProvider:
    """A provider for Anthropic models through the vendor SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
        thinking_budget: int = 0,
        provider_id: str = "anthropic",
        client: Any = None,
        health_ttl_s: float = 600.0,
        *,
        sampling: bool = False,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not model:
            raise ConfigurationError(
                "ANTHROPIC_MODEL is empty; set it to an Anthropic model id to use "
                "the Anthropic provider"
            )
        if not api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is empty; the Anthropic provider needs a key"
            )
        if max_tokens <= 0:
            raise ConfigurationError("ANTHROPIC_MAX_TOKENS must be positive")
        if thinking_budget < 0 or 0 < thinking_budget < MIN_THINKING_BUDGET:
            raise ConfigurationError(
                f"ANTHROPIC_THINKING_BUDGET must be 0 or at least "
                f"{MIN_THINKING_BUDGET} tokens"
            )
        if thinking_budget and thinking_budget >= max_tokens:
            raise ConfigurationError(
                "ANTHROPIC_THINKING_BUDGET must stay below ANTHROPIC_MAX_TOKENS"
            )
        self.model = model
        self._api_key = api_key
        self.max_tokens = int(max_tokens)
        self.thinking_budget = int(thinking_budget)
        self.provider_id = provider_id
        self.health_ttl_s = float(health_ttl_s)
        self.sampling = bool(sampling)
        self.last_stream_usage: Optional[Dict[str, Any]] = None
        self._client = client
        self._owns_client = client is None
        self._clock = clock or _now
        self._last_call: Optional[Tuple[datetime, bool, str]] = None

    def __repr__(self) -> str:
        return (
            f"AnthropicProvider(model={self.model!r}, provider_id="
            f"{self.provider_id!r}, max_tokens={self.max_tokens})"
        )

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None, **overrides: Any
    ) -> "AnthropicProvider":
        """Build from Settings; raises ``ConfigurationError`` naming a missing key."""
        active = settings if settings is not None else get_settings()
        arguments: Dict[str, Any] = {
            "model": active.ANTHROPIC_MODEL,
            "api_key": active.ANTHROPIC_API_KEY,
            "max_tokens": active.ANTHROPIC_MAX_TOKENS,
            "thinking_budget": active.ANTHROPIC_THINKING_BUDGET,
            "sampling": active.ANTHROPIC_SAMPLING_ENABLED,
        }
        arguments.update(overrides)
        return cls(**arguments)

    # -- protocol surface ---------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return self.model_info()

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.provider_id,
            name=self.model,
            provider_type="anthropic",
            context_window=None,
            cost="paid",
            supports_thinking=True,
        )

    def supports_thinking(self) -> bool:
        return True

    async def initialize(self) -> None:
        """Build the SDK client (no request is made)."""
        self._ensure_client()

    async def shutdown(self) -> None:
        """Close the client this provider built; an injected one stays open."""
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    def build_request(self, params: CompletionParams) -> Dict[str, Any]:
        """The Messages API arguments for ``params``."""
        if has_image_parts(params.messages):
            raise CapabilityUnavailableError(_IMAGE_REFUSAL)
        system, turns = translate_messages(params.messages)
        max_tokens = int(params.max_tokens or self.max_tokens)
        request: Dict[str, Any] = {"model": params.model or self.model}
        if system:
            request["system"] = system
        request["messages"] = turns
        if params.stop:
            request["stop_sequences"] = list(params.stop)
        if params.thinking_mode:
            if self.thinking_budget:
                if max_tokens <= self.thinking_budget:
                    max_tokens += self.thinking_budget
                request["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            else:
                request["thinking"] = {"type": "adaptive", "display": "summarized"}
        elif self.sampling:
            if params.temperature is not None:
                request["temperature"] = params.temperature
            elif params.top_p is not None:
                request["top_p"] = params.top_p
            if params.top_k is not None:
                request["top_k"] = params.top_k
        request["max_tokens"] = max_tokens
        return request

    async def complete(self, params: CompletionParams) -> CompletionResult:
        request = self.build_request(params)
        client = self._ensure_client()
        try:
            async with client.messages.stream(**request) as stream:
                message = await stream.get_final_message()
        except Exception as exc:
            raise self._failure(exc) from exc
        self._record(True, "last call succeeded")
        reasoning_parts: List[str] = []
        text_parts: List[str] = []
        for block in getattr(message, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "thinking":
                reasoning_parts.append(str(getattr(block, "thinking", "") or ""))
            elif kind == "text":
                text_parts.append(str(getattr(block, "text", "") or ""))
        reasoning = "".join(reasoning_parts).strip() or None
        return CompletionResult(
            text="".join(text_parts).strip(),
            reasoning=reasoning,
            finish_reason=_finish(getattr(message, "stop_reason", None)),
            **_usage(getattr(message, "usage", None)),
        )

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        request = self.build_request(params)
        client = self._ensure_client()
        self.last_stream_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
        }
        in_thinking = False
        thinking_blocks = set()
        try:
            async with client.messages.stream(**request) as stream:
                async for event in stream:
                    kind = getattr(event, "type", None)
                    if kind == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if getattr(block, "type", None) == "thinking":
                            thinking_blocks.add(getattr(event, "index", None))
                            if not in_thinking:
                                in_thinking = True
                                yield THINK_OPEN
                        elif in_thinking:
                            in_thinking = False
                            yield THINK_CLOSE
                    elif kind == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_kind = getattr(delta, "type", None)
                        if delta_kind == "thinking_delta":
                            piece = getattr(delta, "thinking", "") or ""
                            if piece:
                                yield piece
                        elif delta_kind == "text_delta":
                            piece = getattr(delta, "text", "") or ""
                            if piece:
                                yield piece
                    elif kind == "content_block_stop":
                        if in_thinking and getattr(event, "index", None) in (
                            thinking_blocks
                        ):
                            in_thinking = False
                            yield THINK_CLOSE
                    elif kind == "message_delta":
                        stop = getattr(
                            getattr(event, "delta", None), "stop_reason", None
                        )
                        if stop:
                            self.last_stream_usage["finish_reason"] = _finish(stop)
                if in_thinking:
                    in_thinking = False
                    yield THINK_CLOSE
                final = await stream.get_final_message()
                self.last_stream_usage.update(_usage(getattr(final, "usage", None)))
                stop = getattr(final, "stop_reason", None)
                if stop:
                    self.last_stream_usage["finish_reason"] = _finish(stop)
        except (CapabilityUnavailableError, ProviderError):
            raise
        except Exception as exc:
            raise self._failure(exc) from exc
        self._record(True, "last call succeeded")

    async def health_check(self) -> HealthStatus:
        """Report health from configuration and the last real call only."""
        now = self._clock()
        if self._client is None:
            try:
                require_client("anthropic", "anthropic")
            except ImportError as exc:
                return HealthStatus(False, str(exc), now, False)
        if self._last_call is not None:
            at, ok, detail = self._last_call
            if (now - at).total_seconds() <= self.health_ttl_s:
                return HealthStatus(ok, detail, now, True)
        return HealthStatus(
            True,
            "configured; unverified until a real call succeeds",
            now,
            False,
        )

    # -- helpers ------------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            sdk = require_client("anthropic", "anthropic")
            self._client = sdk.AsyncAnthropic(api_key=self._api_key)
            self._owns_client = True
        return self._client

    def _record(self, ok: bool, detail: str) -> None:
        self._last_call = (self._clock(), ok, detail)

    def _failure(self, exc: BaseException) -> ProviderError:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        status = status if isinstance(status, int) else None
        detail = f"last call failed ({name}" + (f", {status})" if status else ")")
        self._record(False, detail)
        logger.warning("Anthropic provider call failed: %s", detail)
        if "Timeout" in name or isinstance(exc, asyncio.TimeoutError):
            return ProviderTimeoutError(f"Anthropic call timed out ({name})")
        return ProviderError(f"Anthropic call failed ({name})", status_code=status)
