"""Verbatim quotes kept when old history is summarized.

A summary paraphrases; some sentences must survive word for word because a
paraphrase would change what was said. ``extract_pinned_quotes`` picks those
sentences from the messages about to be summarized, by lexical category
(``PIN_CATEGORIES``):

- ``promise``: an explicit promise ("I promise", "you have my word").
- ``commitment``: a first-person plan or undertaking ("I will", "I'll",
  "we will", "I am going to", "let's").
- ``vulnerability``: a first-person statement of fear, worry, hurt or need
  ("I'm afraid", "I am worried", "it hurts", "I need help").
- ``protected_name``: a sentence naming one of the configured protected
  names, matched as a whole word, so ``Alice`` pins and ``Aliceville`` does
  not.

The rules read words, never sentiment: no valence, mood or label is guessed
or attached, and a sentence that only sounds emotional is not pinned. A quote
is the speaker (the message's ``speaker``, else its role) followed by the
sentence exactly as written, in message order, each quote once, until
``max_chars`` is spent (a quote that does not fit is skipped). Clock notes
are never quoted.

Config keys: ``PROTECTED_NAMES`` (a comma list, empty by default), read at
call time when ``protected_names`` is None. Conversation text and the
protected names are private runtime data; nothing here logs either.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mtc.core.config import csv_list, get_settings
from mtc.pipeline.context_dedup import split_into_sentences

DEFAULT_MAX_QUOTE_CHARS = 2000

_APOSTROPHE = "['" + chr(0x2019) + "]"

PIN_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "promise": (
        r"\bi promise\b",
        r"\bpromise me\b",
        r"\bwe promised\b",
        r"\bi promised\b",
        r"\byou have my word\b",
        r"\bi give you my word\b",
    ),
    "commitment": (
        r"\bi will\b",
        r"\bi" + _APOSTROPHE + r"ll\b",
        r"\bwe will\b",
        r"\bwe" + _APOSTROPHE + r"ll\b",
        r"\bi am going to\b",
        r"\bi" + _APOSTROPHE + r"m going to\b",
        r"\bwe are going to\b",
        r"\blet" + _APOSTROPHE + r"s\b",
        r"\bi commit to\b",
    ),
    "vulnerability": (
        r"\bi" + _APOSTROPHE + r"m (?:afraid|scared|worried|frightened|struggling)\b",
        r"\bi am (?:afraid|scared|worried|frightened|struggling)\b",
        r"\bi feel (?:afraid|scared|alone|lost|hurt|ashamed|overwhelmed)\b",
        r"\bit hurts\b",
        r"\bi need help\b",
        r"\bi can" + _APOSTROPHE + r"t cope\b",
        r"\bi cannot cope\b",
    ),
}

_COMPILED = {
    category: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for category, patterns in PIN_CATEGORIES.items()
}
PROTECTED_NAME_CATEGORY = "protected_name"


def _configured_names(settings: Any = None) -> List[str]:
    cfg = settings if settings is not None else get_settings()
    value = cfg.PROTECTED_NAMES
    if isinstance(value, str):
        return csv_list(value)
    return [str(name).strip() for name in value or () if str(name).strip()]


def _name_patterns(names: Iterable[str]) -> List["re.Pattern[str]"]:
    patterns = []
    for name in names:
        name = str(name).strip()
        if name:
            patterns.append(
                re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
            )
    return patterns


def pin_category(
    sentence: str, name_patterns: Sequence["re.Pattern[str]"] = ()
) -> Optional[str]:
    """Return the first category a sentence falls in, or None."""
    for category, patterns in _COMPILED.items():
        if any(pattern.search(sentence) for pattern in patterns):
            return category
    if any(pattern.search(sentence) for pattern in name_patterns):
        return PROTECTED_NAME_CATEGORY
    return None


def extract_pinned_quotes(
    messages: Sequence[Mapping[str, Any]],
    protected_names: Optional[Sequence[str]] = None,
    max_chars: int = DEFAULT_MAX_QUOTE_CHARS,
    settings: Any = None,
) -> List[str]:
    """Return the verbatim quotes to keep from ``messages``, within
    ``max_chars`` characters in total."""
    names = (
        list(protected_names)
        if protected_names is not None
        else _configured_names(settings)
    )
    name_patterns = _name_patterns(names)
    budget = max(0, int(max_chars))
    quotes: List[str] = []
    used = 0
    for message in messages:
        if message.get("is_clock_note"):
            continue
        content = str(message.get("content") or "")
        speaker = str(message.get("speaker") or message.get("role") or "unknown")
        for sentence in split_into_sentences(content):
            if pin_category(sentence, name_patterns) is None:
                continue
            quote = f'{speaker}: "{sentence}"'
            if quote in quotes:
                continue
            cost = len(quote) + (1 if quotes else 0)
            if used + cost > budget:
                continue
            quotes.append(quote)
            used += cost
    return quotes


__all__ = (
    "DEFAULT_MAX_QUOTE_CHARS",
    "PIN_CATEGORIES",
    "PROTECTED_NAME_CATEGORY",
    "extract_pinned_quotes",
    "pin_category",
)
