"""
Chat-Template Families
======================

How a local chat-completions request is shaped for the model behind the
server. ``LLM_TEMPLATE_FAMILY`` selects one of three families:

- ``generic`` (the default): most installs run a non-reasoning instruct model
  behind a server that may reject unknown fields, so it sends only ``model``
  (when set), ``messages``, ``temperature``, ``top_p``, ``max_tokens``,
  ``stop``, ``stream`` and ``frequency_penalty``. Reasoning is still read when
  the server emits it (a ``reasoning_content`` or ``reasoning`` field, or
  inline think tags).
- ``qwen``: a reasoning model with think tags. Thinking is switched off
  through ``chat_template_kwargs`` (``enable_thinking`` false), its
  end-of-turn token is added as a stop sequence while thinking, and the
  extended sampling fields (``top_k``, ``min_p``, ``repetition_penalty``) are
  sent.
- ``gemma``: a reasoning model whose template marks thought with
  channel-style tags. Its end-of-turn tokens are added as stop sequences while
  thinking and the extended sampling fields are sent; it has no request switch
  for thinking.

The family's tag pairs are normalized to the canonical ``<think>`` and
``</think>`` by ``mtc.brain.thinking.normalize_reasoning``. No config keys are
read here (the provider passes the configured name) and no state is held.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

CANONICAL_TAGS: Tuple[str, str] = ("<think>", "</think>")


@dataclass(frozen=True)
class TemplateFamily:
    """One chat-template family's request and reasoning conventions."""

    name: str
    stop_when_thinking: Tuple[str, ...]
    thinking_off: Literal["chat_template_kwargs", "none"]
    reasoning_tags: Tuple[Tuple[str, str], ...]
    extended_sampling: bool
    penalty_field: Literal["repetition_penalty", "frequency_penalty"]
    stream_usage: bool

    @property
    def is_reasoning_family(self) -> bool:
        """True for families whose models think in tagged blocks."""
        return self.name != "generic"


FAMILIES: Dict[str, TemplateFamily] = {
    "generic": TemplateFamily(
        name="generic",
        stop_when_thinking=(),
        thinking_off="none",
        reasoning_tags=(CANONICAL_TAGS,),
        extended_sampling=False,
        penalty_field="frequency_penalty",
        stream_usage=False,
    ),
    "qwen": TemplateFamily(
        name="qwen",
        stop_when_thinking=("<|im_end|>",),
        thinking_off="chat_template_kwargs",
        reasoning_tags=(CANONICAL_TAGS,),
        extended_sampling=True,
        penalty_field="repetition_penalty",
        stream_usage=True,
    ),
    "gemma": TemplateFamily(
        name="gemma",
        stop_when_thinking=("<end_of_turn>", "<turn|>"),
        thinking_off="none",
        reasoning_tags=(CANONICAL_TAGS, ("<|channel>thought", "<channel|>")),
        extended_sampling=True,
        penalty_field="repetition_penalty",
        stream_usage=True,
    ),
}


def get_family(name: str) -> TemplateFamily:
    """Return the named family; an unknown name raises ``ValueError``."""
    family = FAMILIES.get(name)
    if family is None:
        raise ValueError(
            f"Unknown template family {name!r}; choose one of {sorted(FAMILIES)}"
        )
    return family
