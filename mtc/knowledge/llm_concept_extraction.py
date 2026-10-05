"""
LLM Concept Extraction
======================

``LLMConceptExtractor`` asks a language model, through the ``BrainProtocol``
seam, to name the concepts in a text, without needing any seed concepts in the
graph. The prompt requests a JSON array of objects with ``name``, ``category``
(one of ``CATEGORIES``) and ``relevance`` (0 to 1); the call runs with thinking
off, temperature 0.3 and 800 tokens by default.

Every result is a ``ConceptExtraction`` whose ``status`` says what happened:

- ``measured``: the reply parsed; ``concepts`` holds what the model named (it
  may be empty). A blank input text is ``measured`` with no concepts and the
  brain is not asked.
- ``no_answer``: the brain raised ``NoAnswerError`` or replied with nothing
  once reasoning blocks were stripped. No concept is composed in its place.
- ``unparseable``: the reply held no JSON array of concepts.

Parsing strips reasoning blocks first, then reads a fenced code block or the
outermost ``[...]`` span. A relevance is clamped to [0, 1]; a missing or
non-numeric one stays None rather than being invented. An unknown category
becomes ``concept``; names are deduplicated case-insensitively and capped at
``max_concepts``. ``GraphStore`` writers such as
``KnowledgeGraphManager.process_llm_extraction`` write only ``measured``
results.

Extracted ``person`` concepts are real names at runtime and are private data;
they belong only in operator-controlled stores. No config keys and no
persisted state.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks

logger = logging.getLogger(__name__)

CATEGORIES = ("person", "place", "emotion", "topic", "activity", "concept")
FALLBACK_CATEGORY = "concept"
STATUSES = ("measured", "no_answer", "unparseable")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.S | re.I)

_PROMPT_HEAD = """Extract the meaningful concepts from the text below.

TEXT:
"""

_PROMPT_BODY = """

INSTRUCTIONS:
1. Identify up to MAX_CONCEPTS important concepts in the text.
2. Concepts can be people, places, emotions, topics, activities or ideas.
3. Prefer concepts that would help connect this text to related memories.
4. Categorize each concept as one of:
   - "person" (people, characters, identities)
   - "place" (locations, spaces, environments)
   - "emotion" (feelings, moods, emotional states)
   - "topic" (subjects, themes, domains of knowledge)
   - "activity" (actions, events, behaviors)
   - "concept" (abstract ideas, principles, qualities)

RESPONSE FORMAT:
Return ONLY a JSON array of objects, each with:
- "name": the concept name (two to four words at most)
- "category": one of the categories above
- "relevance": 0.0 to 1.0 (how central the concept is to the text)

EXAMPLE:
[
  {"name": "river walk", "category": "activity", "relevance": 0.9},
  {"name": "the neighbour", "category": "person", "relevance": 0.7},
  {"name": "curiosity", "category": "emotion", "relevance": 0.6}
]

Extract the concepts now (JSON only, no explanation):"""


@dataclass(frozen=True)
class ExtractedConcept:
    """One concept the model named; ``relevance`` is None when it gave none."""

    name: str
    category: str
    relevance: Optional[float]


@dataclass
class ConceptExtraction:
    """The concepts of one text and how they were obtained."""

    concepts: List[ExtractedConcept] = field(default_factory=list)
    status: str = "measured"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")


def _relevance(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


def _json_candidates(text: str) -> List[str]:
    candidates = [match.group(1).strip() for match in _FENCE.finditer(text)]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


class LLMConceptExtractor:
    """Concept extraction by a language model behind ``BrainProtocol``."""

    def __init__(
        self,
        brain: BrainProtocol,
        max_concepts: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> None:
        if max_concepts < 1:
            raise ValueError("max_concepts must be at least 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if temperature < 0.0:
            raise ValueError("temperature must not be negative")
        self.brain = brain
        self.max_concepts = int(max_concepts)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def build_prompt(self, text: str) -> str:
        """Return the extraction prompt for ``text``."""
        body = _PROMPT_BODY.replace("MAX_CONCEPTS", str(self.max_concepts))
        return _PROMPT_HEAD + text + body

    async def extract(self, text: str) -> ConceptExtraction:
        """Return the concepts of ``text`` with a status (see the module notes)."""
        if not (text or "").strip():
            return ConceptExtraction(concepts=[], status="measured")
        try:
            reply = await self.brain.think(
                prompt=self.build_prompt(text),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                thinking_mode=False,
            )
        except NoAnswerError as exc:
            logger.warning("concept extraction got no answer: %s", exc)
            return ConceptExtraction(concepts=[], status="no_answer")
        visible = strip_reasoning_blocks(getattr(reply, "text", "") or "")
        if not visible:
            logger.warning("concept extraction got an empty reply")
            return ConceptExtraction(concepts=[], status="no_answer")
        items = self._parse(visible)
        if items is None:
            logger.warning("concept extraction reply held no JSON concept array")
            return ConceptExtraction(concepts=[], status="unparseable")
        return ConceptExtraction(concepts=self._clean(items), status="measured")

    async def extract_batch(self, texts: Sequence[str]) -> List[ConceptExtraction]:
        """Extract each text in turn (one model call per text), in order."""
        return [await self.extract(text) for text in texts]

    @staticmethod
    def _parse(text: str) -> Optional[List[Any]]:
        for candidate in _json_candidates(text):
            try:
                value = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get("concepts"), list):
                value = value["concepts"]
            if isinstance(value, list):
                return value
        return None

    def _clean(self, items: Sequence[Any]) -> List[ExtractedConcept]:
        concepts: List[ExtractedConcept] = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or name.lower() in seen:
                continue
            category = str(item.get("category") or "").strip().lower()
            if category not in CATEGORIES:
                category = FALLBACK_CATEGORY
            seen.add(name.lower())
            concepts.append(
                ExtractedConcept(name, category, _relevance(item.get("relevance")))
            )
            if len(concepts) >= self.max_concepts:
                break
        return concepts
