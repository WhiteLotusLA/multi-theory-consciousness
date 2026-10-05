"""
Thinking Loop Detector
======================

Catches a model that repeats itself inside its reasoning, so the brain can
retry with a raised penalty instead of waiting for the token limit. Two
strategies run over the reasoning text:

- n-gram fingerprinting: fixed windows of ``window_chars`` characters taken
  every ``step_chars``; a window seen at least ``min_repeats`` times that also
  dominates the text (more than 0.3 of the distinct windows) is a loop;
- sentence repetition: the text is split at sentence ends, paragraph breaks
  and the family's loop markers (lines that begin with an asterisk and a
  marker word such as ``Wait`` or ``Draft``), normalized, and any sentence of
  at least 20 characters seen ``min_repeats`` times is a loop.

``check`` examines a complete reasoning text. ``feed`` takes a reply stream
chunk by chunk (canonical think tags included, even when split across chunks)
and examines only the text inside reasoning: every 200 new reasoning
characters and once when a block closes. A stream that begins inside a block
the chat template opened is handled by ``starts_inside`` or by a closing tag
with no opening tag. Spoken text is never examined.

No config keys are read. ``ThinkingLoopDetected`` carries the repeated pattern
as an attribute, never in its message, because reasoning text is private.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from mtc.brain.thinking import THINK_CLOSE, THINK_OPEN

_REFERENCE_MARKERS: Tuple[str, ...] = ("Wait", "Okay", "Let's try", "Option", "Draft")

DEFAULT_LOOP_MARKERS: Dict[str, Tuple[str, ...]] = {
    "generic": _REFERENCE_MARKERS,
    "qwen": _REFERENCE_MARKERS,
    "gemma": _REFERENCE_MARKERS,
}

# Sentence ends: '. ', '.\n', a paragraph break, '?' or '!'; a bare '.' is not
# one, so abbreviations and decimals stay whole.
_SENTENCE_SPLIT = re.compile(r"(?:\. |\.\n|\n\n|[?!])")
_MIN_SENTENCE_CHARS = 20
_DOMINANCE_RATIO = 0.3


class ThinkingLoopDetected(Exception):
    """A repetition loop was found in reasoning text."""

    def __init__(self, pattern: str, strategy: str, occurrences: int) -> None:
        self.pattern = pattern
        self.strategy = strategy
        self.occurrences = occurrences
        super().__init__(
            f"Thinking loop detected ({strategy}): a pattern of {len(pattern)} "
            f"characters repeated {occurrences} times"
        )


class ThinkingLoopDetector:
    """Detects repetitive reasoning in a whole block or a stream.

    One instance serves one generation attempt; ``reset`` clears it for the
    next attempt.
    """

    CHECK_INTERVAL_CHARS = 200

    def __init__(
        self,
        markers: Optional[Sequence[str]] = None,
        window_chars: int = 60,
        step_chars: int = 20,
        min_repeats: int = 3,
        *,
        starts_inside: bool = False,
    ) -> None:
        if isinstance(markers, str):
            raise TypeError("markers is a sequence of marker words, not one string")
        if window_chars <= 0 or step_chars <= 0 or min_repeats < 2:
            raise ValueError(
                "window_chars and step_chars must be positive and min_repeats >= 2"
            )
        self.markers: Tuple[str, ...] = tuple(
            DEFAULT_LOOP_MARKERS["generic"] if markers is None else markers
        )
        self.window_chars = window_chars
        self.step_chars = step_chars
        self.min_repeats = min_repeats
        self._marker_split = (
            re.compile(
                r"^\s*\*(?:" + "|".join(re.escape(m) for m in self.markers) + r")",
                re.MULTILINE,
            )
            if self.markers
            else None
        )
        self._starts_inside = starts_inside
        self.reset()

    @classmethod
    def for_family(cls, family: str, **kwargs) -> "ThinkingLoopDetector":
        """A detector with the family's default markers."""
        if family not in DEFAULT_LOOP_MARKERS:
            raise ValueError(
                f"Unknown template family {family!r}; choose one of "
                f"{sorted(DEFAULT_LOOP_MARKERS)}"
            )
        return cls(markers=DEFAULT_LOOP_MARKERS[family], **kwargs)

    @property
    def in_reasoning(self) -> bool:
        """True while the fed stream is inside a reasoning block."""
        return self._inside

    def reset(self) -> None:
        """Forget everything fed so far."""
        self._inside = self._starts_inside
        self._pending = ""
        self._outside: List[str] = []
        self._reasoning: List[str] = []
        self._reasoning_len = 0
        self._checked_len = 0

    def check(self, text: str) -> None:
        """Examine a complete reasoning text; raises ``ThinkingLoopDetected``."""
        if not text or not text.strip():
            return
        text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        self._run_detection(text)

    def feed(self, chunk: str) -> None:
        """Take the next stream chunk; raises inside reasoning on a loop."""
        data = self._pending + (chunk or "")
        self._pending = ""
        while data:
            if self._inside:
                close_at = data.find(THINK_CLOSE)
                open_at = data.find(THINK_OPEN)
                if open_at != -1 and (close_at == -1 or open_at < close_at):
                    # A repeated opening tag is not reasoning text.
                    self._add_reasoning(data[:open_at])
                    data = data[open_at + len(THINK_OPEN) :]
                    continue
                if close_at != -1:
                    self._add_reasoning(data[:close_at])
                    data = data[close_at + len(THINK_CLOSE) :]
                    self._inside = False
                    self._block_closed()
                    continue
                held = _partial_tag_tail(data, (THINK_CLOSE, THINK_OPEN))
                self._add_reasoning(data[: len(data) - held])
                self._pending = data[len(data) - held :]
                self._maybe_check()
                return
            open_at = data.find(THINK_OPEN)
            close_at = data.find(THINK_CLOSE)
            if close_at != -1 and (open_at == -1 or close_at < open_at):
                # A closing tag with no opening tag: the reply began inside a
                # block the chat template opened, so what came before was
                # reasoning.
                before = "".join(self._outside) + data[:close_at]
                self._outside = []
                self._add_reasoning(before)
                data = data[close_at + len(THINK_CLOSE) :]
                self._block_closed()
                continue
            if open_at != -1:
                self._outside.append(data[:open_at])
                data = data[open_at + len(THINK_OPEN) :]
                self._inside = True
                continue
            held = _partial_tag_tail(data, (THINK_OPEN, THINK_CLOSE))
            self._outside.append(data[: len(data) - held])
            self._pending = data[len(data) - held :]
            return

    def _add_reasoning(self, text: str) -> None:
        if text:
            self._reasoning.append(text)
            self._reasoning_len += len(text)

    def _maybe_check(self) -> None:
        if self._reasoning_len - self._checked_len >= self.CHECK_INTERVAL_CHARS:
            self._checked_len = self._reasoning_len
            self._run_detection("".join(self._reasoning))

    def _block_closed(self) -> None:
        if self._reasoning_len > self._checked_len:
            self._checked_len = self._reasoning_len
            self._run_detection("".join(self._reasoning))

    def _run_detection(self, text: str) -> None:
        self._check_ngram(text)
        self._check_sentences(text)

    def _check_ngram(self, text: str) -> None:
        if len(text) < self.window_chars * 2:
            return
        last = len(text) - self.window_chars + 1
        windows = Counter(
            text[i : i + self.window_chars] for i in range(0, last, self.step_chars)
        )
        if not windows:
            return
        window, count = windows.most_common(1)[0]
        if count >= self.min_repeats and count / len(windows) > _DOMINANCE_RATIO:
            raise ThinkingLoopDetected(
                pattern=window, strategy="ngram", occurrences=count
            )

    def _check_sentences(self, text: str) -> None:
        parts: List[str] = []
        for part in _SENTENCE_SPLIT.split(text):
            if self._marker_split is not None:
                parts.extend(self._marker_split.split(part))
            else:
                parts.append(part)
        counts: Counter = Counter()
        first_seen: Dict[str, str] = {}
        for part in parts:
            normalized = _normalize_sentence(part)
            if len(normalized) < _MIN_SENTENCE_CHARS:
                continue
            counts[normalized] += 1
            first_seen.setdefault(normalized, part.strip())
        for normalized, count in counts.items():
            if count >= self.min_repeats:
                raise ThinkingLoopDetected(
                    pattern=first_seen[normalized],
                    strategy="sentence",
                    occurrences=count,
                )


def _normalize_sentence(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    return re.sub(r"\s+", " ", text).strip()


def _partial_tag_tail(data: str, tags: Sequence[str]) -> int:
    """Length of the longest tail of ``data`` that could begin one of ``tags``."""
    longest = 0
    for tag in tags:
        for size in range(min(len(tag) - 1, len(data)), longest, -1):
            if tag.startswith(data[-size:]):
                longest = size
                break
    return longest
