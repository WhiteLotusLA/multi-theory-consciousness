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
"""

import hashlib
import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)


@dataclass
class BrainReply:
    """One model reply: the text and optional generation metadata."""

    text: str
    finish_reason: Optional[str] = None
    output_tokens: Optional[int] = None


class NoAnswerError(RuntimeError):
    """The brain produced no answer; nothing may be stored or credited."""


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
