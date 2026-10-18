"""
Keyword Concept Extraction
==========================

``ConceptExtractor`` matches memory text against the concept names already in
the knowledge graph; it satisfies the ``ConceptMatcher`` protocol the memory
manager links memories through. It never invents a concept: only names it is
given can match.

Scoring (the reference implementation's rule): a concept found verbatim in the
lower-cased text, as whole words (no letter or digit directly before or after
it, so "art" is not found in "start"), scores ``0.5 + 0.1`` per occurrence,
capped at 1.0. With ``fuzzy`` matching, a concept that is not found verbatim
scores by the share of its content words that appear as words of the text,
times 0.7, with a floor of 0.3; both word sets are taken after punctuation is
cleaned away, as ``extract_keywords`` does, so "learning," is the word
"learning". Stop words count as neither concept words nor text words, so a
shared "of" or "the" alone never links a memory to a concept. Results are
sorted most relevant first. No config keys and no persisted state.
"""

import logging
import re
from typing import FrozenSet, List, Sequence, Tuple

logger = logging.getLogger(__name__)

STOP_WORDS: FrozenSet[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
    }
)

EXACT_BASE = 0.5
EXACT_STEP = 0.1
FUZZY_WEIGHT = 0.7
FUZZY_FLOOR = 0.3
MIN_KEYWORD_LENGTH = 4

_NON_WORD = re.compile(r"[^a-z0-9\s]")


def _content_words(text: str) -> set:
    """The words of lower-cased ``text`` after punctuation, minus stop words."""
    return {word for word in _NON_WORD.sub(" ", text).split() if word not in STOP_WORDS}


def _whole_words(concept_lower: str) -> "re.Pattern[str]":
    """A pattern that finds ``concept_lower`` only as whole words."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(concept_lower) + r"(?![a-z0-9])")


class ConceptExtractor:
    """Matches text against known concept names by keyword."""

    def __init__(self) -> None:
        self.stop_words = STOP_WORDS

    def extract_keywords(self, text: str) -> List[str]:
        """Return the words of ``text`` that are not stop words, in order.

        Words shorter than four characters are dropped as well.
        """
        cleaned = _NON_WORD.sub(" ", text.lower())
        return [
            word
            for word in cleaned.split()
            if word not in self.stop_words and len(word) >= MIN_KEYWORD_LENGTH
        ]

    def match_concepts(
        self, text: str, known_concepts: Sequence[str], fuzzy: bool = True
    ) -> List[Tuple[str, float]]:
        """Return ``(concept, relevance)`` pairs, most relevant first.

        A concept is found verbatim only as whole words of the text; the fuzzy
        path compares punctuation-free word sets.
        """
        text_lower = text.lower()
        text_words = _content_words(text_lower)
        matches: List[Tuple[str, float]] = []
        for concept in known_concepts:
            concept_lower = concept.lower()
            count = (
                len(_whole_words(concept_lower).findall(text_lower))
                if concept_lower
                else 0
            )
            if count:
                matches.append((concept, min(EXACT_BASE + count * EXACT_STEP, 1.0)))
                continue
            if not fuzzy:
                continue
            concept_words = _content_words(concept_lower)
            overlap = concept_words & text_words
            if overlap:
                share = len(overlap) / len(concept_words)
                matches.append((concept, max(share * FUZZY_WEIGHT, FUZZY_FLOOR)))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches

    def extract_concepts_from_memory(
        self,
        memory_content: str,
        known_concepts: Sequence[str],
        min_relevance: float = 0.3,
        max_concepts: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return up to ``max_concepts`` matches scoring at least ``min_relevance``."""
        matches = self.match_concepts(memory_content, known_concepts)
        result = [(c, s) for c, s in matches if s >= min_relevance][:max_concepts]
        logger.debug("matched %d concepts", len(result))
        return result


def create_concept_extractor() -> ConceptExtractor:
    """Return a new ``ConceptExtractor``."""
    return ConceptExtractor()
