"""
Reasoning Split
===============

One rule for where a reply's reasoning ends and its spoken words begin, used
by every reader that decides whether a reply has words to store.

- ``split_thinking`` splits greedily from the first ``<think>`` to the last
  ``</think>``, so a stray closing tag inside the reasoning stays reasoning; an
  unclosed block runs to the end of the text; a reply that begins inside a
  block the chat template opened (a closing tag with no opening tag before
  it) is reasoning up to that closing tag; a reply with no tag at all is
  returned unchanged as ``(None, text)``.
- ``normalize_reasoning`` first maps a template family's own tag pairs (for
  example channel-style thought markers) to the canonical tags, then splits.
- ``canonical`` builds the brain's reply form, ``<think>R</think>`` followed by
  a blank line and the content, or the content alone when there is no
  reasoning.
- ``StreamTagNormalizer`` performs the same tag mapping on a stream, holding
  back a chunk tail that could be the start of a tag.

No config keys are read and no state is persisted. The text handled here is
private conversation content.
"""

from typing import List, Optional, Tuple

from mtc.brain.families import CANONICAL_TAGS, get_family

THINK_OPEN, THINK_CLOSE = CANONICAL_TAGS


def _join(first: Optional[str], second: Optional[str]) -> Optional[str]:
    parts = [part for part in (first, second) if part]
    if not parts:
        return first if first is not None else second
    return "\n".join(parts)


def split_thinking(text: str) -> Tuple[Optional[str], str]:
    """Return ``(reasoning, spoken)`` for a reply; both halves are stripped.

    Reasoning is None only when the text holds no think tag at all.
    """
    text = text or ""
    first_open = text.find(THINK_OPEN)
    first_close = text.find(THINK_CLOSE)
    if first_open == -1 and first_close == -1:
        return None, text
    last_close = text.rfind(THINK_CLOSE)
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        # The reply began inside a block the template opened.
        reasoning = text[:last_close]
        spoken = text[last_close + len(THINK_CLOSE) :]
    elif last_close > first_open:
        reasoning = text[first_open + len(THINK_OPEN) : last_close]
        spoken = text[:first_open] + text[last_close + len(THINK_CLOSE) :]
    else:
        reasoning = text[first_open + len(THINK_OPEN) :]
        spoken = text[:first_open]
    reasoning = reasoning.strip()
    if THINK_OPEN in spoken or THINK_CLOSE in spoken:
        more, spoken = split_thinking(spoken)
        reasoning = _join(reasoning, more) or ""
    return reasoning, spoken.strip()


def to_canonical_tags(text: str, family: str) -> str:
    """Replace the family's own reasoning tags with the canonical pair."""
    for open_tag, close_tag in get_family(family).reasoning_tags:
        if (open_tag, close_tag) == CANONICAL_TAGS:
            continue
        text = text.replace(open_tag, THINK_OPEN).replace(close_tag, THINK_CLOSE)
    return text


def normalize_reasoning(text: str, family: str) -> Tuple[Optional[str], str]:
    """Split a reply after mapping the family's tag pairs to canonical tags."""
    return split_thinking(to_canonical_tags(text or "", family))


def canonical(reasoning: Optional[str], content: str) -> str:
    """The brain's reply form; blank reasoning leaves the content alone."""
    if reasoning is None or not reasoning.strip():
        return content
    return f"{THINK_OPEN}{reasoning}{THINK_CLOSE}\n\n{content}"


class StreamTagNormalizer:
    """Map a family's reasoning tags to canonical tags across stream chunks."""

    def __init__(self, family: str) -> None:
        pairs = get_family(family).reasoning_tags
        self._replacements: List[Tuple[str, str]] = []
        for open_tag, close_tag in pairs:
            if (open_tag, close_tag) == CANONICAL_TAGS:
                continue
            self._replacements.append((open_tag, THINK_OPEN))
            self._replacements.append((close_tag, THINK_CLOSE))
        self._pending = ""

    def _held_suffix(self, data: str) -> int:
        """Length of the longest tail of ``data`` that could begin a tag."""
        longest = 0
        for tag, _ in self._replacements:
            for size in range(min(len(tag) - 1, len(data)), longest, -1):
                if tag.startswith(data[-size:]):
                    longest = size
                    break
        return longest

    def feed(self, chunk: str) -> str:
        """Return the text that is safe to emit after this chunk."""
        if not self._replacements:
            return chunk
        data = self._pending + chunk
        for tag, replacement in self._replacements:
            data = data.replace(tag, replacement)
        held = self._held_suffix(data)
        self._pending = data[len(data) - held :] if held else ""
        return data[: len(data) - held] if held else data

    def flush(self) -> str:
        """Return whatever was held back at the end of the stream."""
        rest, self._pending = self._pending, ""
        return rest
