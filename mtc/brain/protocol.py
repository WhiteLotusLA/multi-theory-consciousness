"""
Brain Protocol
==============

The only way modules reach a language model. A module takes a ``BrainProtocol``
and awaits ``think``; it never imports a model client. The contract:

- ``think`` returns a ``BrainReply`` or raises ``NoAnswerError`` when the model
  produced no answer (a timeout, an empty completion or a refusal to answer).
  A caller treats ``NoAnswerError`` and an empty ``text`` the same way: nothing
  is stored, narrated or credited, and no replacement text is composed by code.
- Reasoning blocks in a reply are removed with ``strip_reasoning_blocks``
  before the text is used or stored.

``FakeBrain`` is the deterministic stand-in for tests. It is never a runtime
default: a component without a brain stays inactive rather than inventing
replies.

``ConversationalBrain`` is the richer surface the conversation pipeline uses
(streaming, the context-assembling turn and ``generate``); a brain that
implements it also satisfies ``BrainProtocol``. ``NoAnswerError.reason`` names
why there was no answer (``empty``, ``reasoning_only`` or ``loop``).
``BrainUnavailableError`` means the configured provider cannot be reached (it
is reported, never replaced by another provider),
``CapabilityUnavailableError`` means a requested capability does not run in
this build (for example image input), and ``ConfigurationError`` names the
Settings key that is missing or invalid.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

__all__ = [
    "NO_ANSWER_REASONS",
    "BrainProtocol",
    "BrainReply",
    "BrainUnavailableError",
    "CapabilityUnavailableError",
    "ConfigurationError",
    "ConversationalBrain",
    "FakeBrain",
    "NoAnswerError",
    "ScriptItem",
    "strip_reasoning_blocks",
]

# Why a brain call produced no answer.
NO_ANSWER_REASONS: Tuple[str, ...] = ("empty", "reasoning_only", "loop")


@dataclass
class BrainReply:
    """One model reply: the text and optional generation metadata.

    ``input_tokens`` and ``output_tokens`` are the provider's usage counts when
    it reported them (None otherwise); ``provider_id`` names the provider that
    produced the reply, for the turn's provenance record.
    """

    text: str
    finish_reason: Optional[str] = None
    output_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    provider_id: Optional[str] = None


class NoAnswerError(RuntimeError):
    """The brain produced no answer; nothing may be stored or credited.

    ``reason`` is ``empty`` (no text came back), ``reasoning_only`` (only a
    reasoning block came back) or ``loop`` (a thinking loop outlived its
    retries).
    """

    def __init__(self, message: str = "", reason: str = "empty") -> None:
        if reason not in NO_ANSWER_REASONS:
            raise ValueError(
                f"NoAnswerError reason {reason!r} is not one of {NO_ANSWER_REASONS}"
            )
        super().__init__(message)
        self.reason = reason


class BrainUnavailableError(RuntimeError):
    """The configured provider cannot answer; no other provider is used."""


class CapabilityUnavailableError(RuntimeError):
    """A requested capability (for example image input) does not run here."""


class ConfigurationError(ValueError):
    """A required Settings key is missing or invalid; the message names it."""


@runtime_checkable
class BrainProtocol(Protocol):
    """A language model reached through one async call."""

    async def think(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking_mode: bool = False,
    ) -> BrainReply:
        """Return a reply, or raise NoAnswerError when there is none."""
        ...


@runtime_checkable
class ConversationalBrain(Protocol):
    """The conversation pipeline's brain: one-shot, streamed and in-context turns.

    ``think_with_context`` assembles the system prompt from the keyword slots
    (each None slot renders nothing) and returns the reply; ``generate`` returns
    the spoken text only, with reasoning removed. Every method raises
    ``NoAnswerError`` when there is no answer and ``BrainUnavailableError``
    when the configured provider cannot answer.
    """

    async def think(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking_mode: bool = False,
    ) -> BrainReply:
        """Return a reply, or raise NoAnswerError when there is none."""
        ...

    def think_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking_mode: bool = False,
    ) -> AsyncIterator[str]:
        """Yield reply chunks; reasoning arrives inside think tags."""
        ...

    async def think_with_context(
        self,
        current_prompt: str,
        *,
        conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
        core_memory: Optional[Any] = None,
        ctm_thoughts: Optional[Sequence[Mapping[str, Any]]] = None,
        thought_gists: Optional[Sequence[Mapping[str, Any]]] = None,
        coherence_context: Optional[Any] = None,
        retrieved_memories: Optional[Sequence[Mapping[str, Any]]] = None,
        retrieved_concepts: Optional[Sequence[Any]] = None,
        consciousness_context: Optional[Any] = None,
        generation_params: Optional[Mapping[str, Any]] = None,
        behavioral_directives: Optional[Any] = None,
        consciousness_self_awareness: Optional[Any] = None,
        conversation_summary: Optional[str] = None,
        window_notes: Optional[Any] = None,
        image: Optional[Any] = None,
        speaker: Optional[Any] = None,
        stream_callback: Optional[Callable[[str], Any]] = None,
        neural_state: Optional[Any] = None,
        attention_guidance: Optional[Any] = None,
        metacognitive_insights: Optional[Any] = None,
        active_inference_context: Optional[Any] = None,
        global_workspace_context: Optional[Any] = None,
        max_tokens: Optional[int] = None,
    ) -> BrainReply:
        """Return the in-context reply, or raise NoAnswerError."""
        ...

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        """Return the spoken text only, or raise NoAnswerError."""
        ...


ScriptItem = Union[str, BrainReply, BaseException]


class FakeBrain:
    """Deterministic brain for tests.

    Replies are chosen in this order: the next scripted item (a string, a
    ``BrainReply`` or an exception to raise), then the first rule whose key is a
    substring of the prompt, then a reply derived from a SHA-256 digest of the
    system prompt and the prompt. Every call is appended to ``calls`` as a dict
    of the arguments.
    """

    def __init__(
        self,
        replies: Optional[Sequence[ScriptItem]] = None,
        rules: Optional[Dict[str, str]] = None,
    ) -> None:
        self._script: List[ScriptItem] = list(replies or [])
        self._rules: Dict[str, str] = dict(rules or {})
        self.calls: List[Dict[str, Any]] = []

    async def think(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking_mode: bool = False,
    ) -> BrainReply:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "thinking_mode": thinking_mode,
            }
        )
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, BrainReply):
                return item
            return BrainReply(text=str(item))
        for key, text in self._rules.items():
            if key in prompt:
                return BrainReply(text=text)
        digest = hashlib.sha256(f"{system_prompt}\n{prompt}".encode("utf-8"))
        return BrainReply(text=f"reply-{digest.hexdigest()[:12]}")


def strip_reasoning_blocks(text: str, tags: Sequence[str] = ("think",)) -> str:
    """Remove reasoning blocks and return the stripped visible text.

    For each tag, closed ``<tag>...</tag>`` blocks are removed (case-insensitive,
    across lines); an opening tag that is never closed removes everything from
    it to the end; a closing tag with no opening removes everything before it
    (a reply that starts inside a reasoning block).
    """
    result = text or ""
    for tag in tags:
        name = re.escape(tag)
        result = re.sub(
            rf"<{name}\b[^>]*>.*?</{name}\s*>", "", result, flags=re.S | re.I
        )
        result = re.sub(rf"<{name}\b[^>]*>.*\Z", "", result, flags=re.S | re.I)
        orphan = list(re.finditer(rf"</{name}\s*>", result, flags=re.I))
        if orphan:
            result = result[orphan[-1].end() :]
    return result.strip()
