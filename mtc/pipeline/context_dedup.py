"""Repeated-sentence removal for the history a prompt is built from.

A model that sees its own sentence many times in its history tends to say it
again, so before the history reaches the brain ``deduplicate_context`` keeps
only the first ``max_occurrences`` copies of each repeated sentence in the
assistant turns and drops the rest. User and system turns pass through
unchanged, and the stored history is never touched: the function returns new
dicts and leaves its input as it was. Sentences shorter than ``min_length``
characters (after normalization) are never counted, so greetings and short
acknowledgements survive. An optional ``blocklist`` of phrases (empty by
default) removes every assistant sentence that contains one of them, for an
operator who has seen a phrase loop.

Sentences are compared after normalization: markdown emphasis, surrounding
quotes and leading filler words (``DEFAULT_FILLER_PREFIXES``, generic English
discourse markers) are removed, whitespace is collapsed and case folded, so
"Well, the plan holds." and "the plan holds." count as one sentence. The
splitter breaks on paragraph boundaries, sentence-ending punctuation, colons,
semicolons and the em and en dash characters (built from their code points,
so the module source holds none).

``check_response_repetition`` is the post-reply diagnostic: it returns the
normalized sentence of a new reply that already appears ``max_occurrences``
times in the assistant history (the copy dedup would remove next time), or
None. It modifies nothing. Conversation text is private runtime data: the log
lines carry counts only, never a sentence. No config keys are read.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

DEDUP_MIN_LENGTH = 40
DEDUP_MAX_OCCURRENCES = 2

# Generic English discourse markers stripped from the start of a sentence
# before comparison; they carry no content of their own.
DEFAULT_FILLER_PREFIXES = (
    "well",
    "so",
    "oh",
    "ok",
    "okay",
    "um",
    "uh",
    "hmm",
    "yes",
    "yeah",
    "actually",
    "honestly",
)

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)
_TYPOGRAPHIC_QUOTES = "".join(chr(code) for code in (0x201C, 0x201D, 0x2018, 0x2019))
_QUOTES = "\"'" + _TYPOGRAPHIC_QUOTES

_EMPHASIS = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_WHITESPACE = re.compile(r"\s+")
_PARAGRAPHS = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"*" + re.escape(_TYPOGRAPHIC_QUOTES) + r"])"
    r"|(?<=[:;])\s+"
    r"|(?<=" + EM_DASH + r")\s*"
    r"|(?<=" + EN_DASH + r")\s+"
)


def _filler_pattern(prefixes: Sequence[str]) -> Optional["re.Pattern[str]"]:
    words = [re.escape(word.lower()) for word in prefixes if word and word.strip()]
    if not words:
        return None
    return re.compile(r"^(?:(?:" + "|".join(words) + r")\b[,.!]*\s*)+")


_DEFAULT_FILLER = _filler_pattern(DEFAULT_FILLER_PREFIXES)


def normalize_sentence(
    sentence: str, filler_prefixes: Sequence[str] = DEFAULT_FILLER_PREFIXES
) -> str:
    """Return the comparison form of a sentence (lower case, no emphasis,
    quotes or leading filler words, single spaces)."""
    text = _EMPHASIS.sub(r"\1", sentence.strip())
    text = text.strip(_QUOTES)
    text = _WHITESPACE.sub(" ", text).lower().strip()
    pattern = (
        _DEFAULT_FILLER
        if tuple(filler_prefixes) == DEFAULT_FILLER_PREFIXES
        else _filler_pattern(filler_prefixes)
    )
    if pattern is not None:
        text = pattern.sub("", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks for comparison."""
    chunks: List[str] = []
    for paragraph in _PARAGRAPHS.split(text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        chunks.extend(
            part.strip() for part in _SENTENCE_BREAK.split(paragraph) if part.strip()
        )
    return chunks


def _blocked(norm: str, blocklist: Iterable[str]) -> bool:
    return any(phrase in norm for phrase in blocklist)


def _assistant_sentences(
    history: Sequence[Mapping[str, Any]],
) -> Iterable[str]:
    for message in history:
        if message.get("role") != "assistant":
            continue
        yield from split_into_sentences(str(message.get("content") or ""))


def deduplicate_context(
    history: Sequence[Mapping[str, Any]],
    max_occurrences: int = DEDUP_MAX_OCCURRENCES,
    min_length: int = DEDUP_MIN_LENGTH,
    blocklist: Sequence[str] = (),
    *,
    filler_prefixes: Sequence[str] = DEFAULT_FILLER_PREFIXES,
) -> List[Dict[str, Any]]:
    """Return a copy of ``history`` with repeated assistant sentences removed.

    Pass one counts each normalized sentence (at least ``min_length``
    characters) across the assistant turns; pass two keeps the first
    ``max_occurrences`` copies of every sentence counted more often, drops
    every sentence containing a ``blocklist`` phrase (case-insensitive) and
    rejoins the survivors with blank lines. A message that lost no sentence
    keeps its content exactly; an assistant message that lost every sentence
    is dropped. The input is never modified.
    """
    limit = max(1, int(max_occurrences))
    blocked = [phrase.lower() for phrase in blocklist if phrase and phrase.strip()]
    counts: Dict[str, int] = {}
    any_blocked = False
    for sentence in _assistant_sentences(history):
        norm = normalize_sentence(sentence, filler_prefixes)
        if len(norm) < min_length:
            continue
        if blocked and _blocked(norm, blocked):
            any_blocked = True
            continue
        counts[norm] = counts.get(norm, 0) + 1
    repeated = {norm for norm, count in counts.items() if count > limit}
    if not repeated and not any_blocked:
        return [dict(message) for message in history]

    emitted: Dict[str, int] = {}
    removed = 0
    result: List[Dict[str, Any]] = []
    for message in history:
        if message.get("role") != "assistant":
            result.append(dict(message))
            continue
        kept: List[str] = []
        dropped_here = 0
        for sentence in split_into_sentences(str(message.get("content") or "")):
            norm = normalize_sentence(sentence, filler_prefixes)
            if len(norm) < min_length:
                kept.append(sentence)
                continue
            if blocked and _blocked(norm, blocked):
                dropped_here += 1
                continue
            if norm in repeated:
                emitted[norm] = emitted.get(norm, 0) + 1
                if emitted[norm] > limit:
                    dropped_here += 1
                    continue
            kept.append(sentence)
        removed += dropped_here
        if not dropped_here:
            result.append(dict(message))
        elif kept:
            result.append({**message, "content": "\n\n".join(kept)})
    if removed:
        logger.info(
            "context dedup removed %d repeated sentence(s) of %d phrase(s)",
            removed,
            len(repeated),
        )
    return result


def check_response_repetition(
    response: str,
    history: Sequence[Mapping[str, Any]],
    *,
    min_length: int = DEDUP_MIN_LENGTH,
    max_occurrences: int = DEDUP_MAX_OCCURRENCES,
    filler_prefixes: Sequence[str] = DEFAULT_FILLER_PREFIXES,
) -> Optional[str]:
    """Return the first normalized sentence of ``response`` that the
    assistant history already holds ``max_occurrences`` or more times, or
    None. Nothing is modified; a WARNING with the count (never the text) is
    logged when a repetition is found."""
    if not response:
        return None
    wanted: List[str] = []
    for sentence in split_into_sentences(response):
        norm = normalize_sentence(sentence, filler_prefixes)
        if len(norm) >= min_length and norm not in wanted:
            wanted.append(norm)
    if not wanted:
        return None
    counts = {norm: 0 for norm in wanted}
    for sentence in _assistant_sentences(history):
        norm = normalize_sentence(sentence, filler_prefixes)
        if norm in counts:
            counts[norm] += 1
    for norm in wanted:
        if counts[norm] >= max(1, int(max_occurrences)):
            logger.warning(
                "reply repeats a sentence already in the history %d time(s)",
                counts[norm],
            )
            return norm
    return None


__all__ = (
    "DEDUP_MAX_OCCURRENCES",
    "DEDUP_MIN_LENGTH",
    "DEFAULT_FILLER_PREFIXES",
    "check_response_repetition",
    "deduplicate_context",
    "normalize_sentence",
    "split_into_sentences",
)
