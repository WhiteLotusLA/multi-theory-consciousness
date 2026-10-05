"""
Provider Base Types
===================

The contract every model provider implements, and the values that cross it.
A provider turns ``CompletionParams`` into a ``CompletionResult`` (or a stream
of text chunks), reports its health without inventing it, and describes
itself with a ``ModelInfo``.

Result semantics shared by every provider:

- ``CompletionResult.text`` is the visible content only; reasoning is never
  left inside it. ``reasoning`` holds the reasoning text when the model
  produced any and is None otherwise.
- Usage counts are None when the server did not report them; nothing is
  estimated in their place.
- A streamed reply carries reasoning between ``<think>`` and ``</think>``
  chunks (the canonical tags of ``mtc.brain.thinking``), whatever form the
  server used.
- A transport or server failure raises ``ProviderError`` with the HTTP status
  when there was one; a timeout raises ``ProviderTimeoutError``.

No config keys are read here and no state is held. Message content passed to
a provider is private conversation data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

PROVIDER_TYPES = ("local", "anthropic", "mock")
COSTS = ("local", "paid", "none")


@dataclass(frozen=True)
class ModelInfo:
    """What a provider is: its id, display name, kind, window and cost class."""

    id: str
    name: str
    provider_type: Literal["local", "anthropic", "mock"]
    context_window: Optional[int]
    cost: Literal["local", "paid", "none"]
    supports_thinking: bool
    supports_vision: bool = False

    def __post_init__(self) -> None:
        if self.provider_type not in PROVIDER_TYPES:
            raise ValueError(
                f"ModelInfo provider_type {self.provider_type!r} is not one of "
                f"{PROVIDER_TYPES}"
            )
        if self.cost not in COSTS:
            raise ValueError(f"ModelInfo cost {self.cost!r} is not one of {COSTS}")
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("ModelInfo context_window must be positive or None")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider_type": self.provider_type,
            "context_window": self.context_window,
            "cost": self.cost,
            "supports_thinking": self.supports_thinking,
            "supports_vision": self.supports_vision,
        }


@dataclass
class CompletionParams:
    """One completion request in the chat-messages form.

    ``messages`` are ``{"role", "content"}`` dicts (``system``, ``user`` or
    ``assistant``). A None sampling field is not sent. ``repetition_penalty``
    and ``frequency_penalty`` are both available; a provider sends only the one
    its template family uses. ``extra_body`` is merged into the request body by
    providers that speak a raw HTTP format. ``messages`` and ``extra_body`` are
    copied on construction, so a caller's later edits do not leak in.
    """

    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    thinking_mode: bool = False
    model: Optional[str] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.messages = [dict(message) for message in self.messages]
        self.extra_body = dict(self.extra_body or {})
        if self.stop is not None:
            self.stop = list(self.stop)


@dataclass
class CompletionResult:
    """A finished completion: visible text, reasoning and reported usage."""

    text: str
    reasoning: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class HealthStatus:
    """A provider's health as last observed.

    ``verified`` is True when the status comes from a real exchange with the
    provider (a health probe that answered or a completion that succeeded or
    failed) and False when it only reflects configuration.
    ``checked_at`` is timezone-aware.
    """

    ok: bool
    detail: str
    checked_at: datetime
    verified: bool

    def __post_init__(self) -> None:
        if self.checked_at.tzinfo is None:
            raise ValueError("HealthStatus checked_at must be timezone-aware")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "detail": self.detail,
            "checked_at": self.checked_at.isoformat(),
            "verified": self.verified,
        }


class ProviderError(RuntimeError):
    """A provider call failed; ``status_code`` is the HTTP status when known."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """A provider call ran past its configured timeout."""


@runtime_checkable
class ModelProvider(Protocol):
    """A language-model backend the brain reaches through one interface."""

    @property
    def info(self) -> ModelInfo:
        """The provider's ``ModelInfo`` (the same value as ``model_info()``)."""
        ...

    async def initialize(self) -> None:
        """Prepare clients; builds no network connection on its own."""
        ...

    async def shutdown(self) -> None:
        """Release what ``initialize`` built."""
        ...

    async def complete(self, params: CompletionParams) -> CompletionResult:
        """Return the whole completion."""
        ...

    def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        """Yield text chunks; reasoning arrives between think-tag chunks."""
        ...

    async def health_check(self) -> HealthStatus:
        """Report health without making a completion."""
        ...

    def model_info(self) -> ModelInfo:
        """Describe the provider."""
        ...

    def supports_thinking(self) -> bool:
        """Whether the provider can return reasoning."""
        ...


def has_image_parts(messages: Sequence[Dict[str, Any]]) -> bool:
    """True when any message carries an image content part."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and str(part.get("type", "")) in (
                    "image",
                    "image_url",
                    "input_image",
                ):
                    return True
    return False
