"""
Data Classifier
===============

Sorts a piece of content into one of four privacy tiers, ordered from least
to most protected: ``PUBLIC``, ``RELATIONSHIP`` (material about the people the
system relates to), ``PRIVATE`` (inner states, reflections, secrets) and
``CORE`` (identity, consciousness state, model weights).

How a verdict is reached:

1. A ``content_type`` found in the content-type map decides alone, at
   confidence 0.95 (``DEFAULT_CONTENT_TYPE_MAP`` unless one is passed).

   1b. With ``keywords_can_raise=True`` the keyword verdict of rules 2 and 3
   is computed as well and the more protected of the two is returned: a
   higher keyword tier carries its own confidence and a reason that names
   both verdicts; otherwise the typed verdict is returned unchanged. Keywords
   never lower a typed verdict. The safety integration classifies every
   message this way, so a conversation message that holds private or core
   terms is audited at that tier.
2. Otherwise the text is matched against three keyword sets and the tags
   against three tag sets; the most protected tier with a match wins. The
   confidence is 0.6 plus 0.1 per match for core (at most 0.9), 0.5 plus 0.1
   per match for private (at most 0.85) and relationship (at most 0.8), and
   0.6 when nothing matched.
3. A verdict below ``protective_confidence`` (default 0.5) is raised one
   tier, so an uncertain classifier errs toward protection. With the default
   scoring a keyword verdict never scores below 0.6; the rule is a floor for a
   raised threshold.

Matching is by whole word or phrase, case-insensitive, through one compiled
pattern per keyword set: each term is escaped, bounded by ``\\b`` on both
sides and may carry a plural ``s`` or ``es``; whitespace inside a phrase
matches any run of whitespace, and typographic apostrophes count as plain
ones. So ``brother`` matches "my brother" and "brothers" but not
"brotherhood", and ``parent`` does not match "parental". Matching substrings
instead lets short words fire inside longer ones, and every memory's tier
depends on it.

The relationship set is ``RELATIONSHIP_KEYWORDS`` (a comma list of generic
dictionary words, no proper names), read from Settings at call time unless a
list is passed. The private and core sets are the documented generic defaults
``DEFAULT_PRIVATE_KEYWORDS`` and ``DEFAULT_CORE_KEYWORDS``. All three are
heuristics, not understanding.

Access stage: ``access_stage`` turns the days since ``SYSTEM_BIRTH_DATE``
(read at call time; the day starts at 00:00 UTC) into a number from 0 to 6
through ``StagePolicy``. With no birth date, or a clock before it, the stage is
None and no date is assumed. This chronological access stage only selects a
row of the access policy; it is a separate thing from the evidence-based
developmental stage in ``mtc.consciousness.developmental_modulator``, which
moves on measured behaviour rather than on the calendar.

Config keys read: ``RELATIONSHIP_KEYWORDS`` and ``SYSTEM_BIRTH_DATE``. The
module keeps no state and persists nothing.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Pattern, Sequence, Tuple, Union

from mtc.core.config import Settings, csv_list, get_settings

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Privacy tiers, ordered from least to most protected."""

    PUBLIC = "public"
    RELATIONSHIP = "relationship"
    PRIVATE = "private"
    CORE = "core"

    @property
    def rank(self) -> int:
        """Position in the protection order (0 is the least protected)."""
        return _ORDER.index(self)

    @classmethod
    def parse(cls, value: Union[str, DataClassification]) -> DataClassification:
        """Return the tier named by ``value``; an unknown name raises ValueError."""
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DataClassification):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DataClassification):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DataClassification):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DataClassification):
            return NotImplemented
        return self.rank >= other.rank


_ORDER: Tuple[DataClassification, ...] = (
    DataClassification.PUBLIC,
    DataClassification.RELATIONSHIP,
    DataClassification.PRIVATE,
    DataClassification.CORE,
)

# Generic words for inner states and confidences (documented default).
DEFAULT_PRIVATE_KEYWORDS: Tuple[str, ...] = (
    "feeling",
    "afraid",
    "scared",
    "lonely",
    "confused",
    "worried",
    "dream",
    "nightmare",
    "secret",
    "diary",
    "journal",
    "personal",
    "embarrassed",
    "ashamed",
    "insecure",
    "doubt",
    "fear",
    "private thought",
    "just between us",
    "don't tell",
)

# Generic words for identity and the system's own make-up (documented default).
DEFAULT_CORE_KEYWORDS: Tuple[str, ...] = (
    "identity",
    "consciousness",
    "neural weight",
    "model weight",
    "personality matrix",
    "core memory",
    "core value",
    "self-model",
    "who i am",
    "what i am",
    "soul",
    "essence",
    "fundamental",
    "birth",
    "creation",
    "existence",
)

# Tags that raise a verdict to their tier.
CORE_TAGS = frozenset({"core", "identity", "consciousness", "neural"})
PRIVATE_TAGS = frozenset({"private", "personal", "diary", "dream"})
RELATIONSHIP_TAGS = frozenset({"family", "relationship", "conversation", "shared"})

# Content types with an inherent tier; the type decides alone.
DEFAULT_CONTENT_TYPE_MAP: Dict[str, DataClassification] = {
    "consciousness_state": DataClassification.CORE,
    "neural_weights": DataClassification.CORE,
    "identity_block": DataClassification.CORE,
    "core_memory": DataClassification.CORE,
    "private_thought": DataClassification.PRIVATE,
    "dream": DataClassification.PRIVATE,
    "diary_entry": DataClassification.PRIVATE,
    "emotion_log": DataClassification.PRIVATE,
    "conversation": DataClassification.RELATIONSHIP,
    "activity_log": DataClassification.RELATIONSHIP,
    "learning_record": DataClassification.RELATIONSHIP,
    "general_knowledge": DataClassification.PUBLIC,
    "public_fact": DataClassification.PUBLIC,
}

CONTENT_TYPE_CONFIDENCE = 0.95
PUBLIC_CONFIDENCE = 0.6
DEFAULT_PROTECTIVE_CONFIDENCE = 0.5

# (base, per match, ceiling) for a keyword verdict of each tier.
_TIER_SCORING: Dict[DataClassification, Tuple[float, float, float]] = {
    DataClassification.CORE: (0.6, 0.1, 0.9),
    DataClassification.PRIVATE: (0.5, 0.1, 0.85),
    DataClassification.RELATIONSHIP: (0.5, 0.1, 0.8),
}

_APOSTROPHES = str.maketrans({0x2019: "'", 0x2018: "'", 0x02BC: "'"})
_PLURAL = r"(?:e?s)?"


def _normalize_text(text: str) -> str:
    return str(text or "").translate(_APOSTROPHES)


def _term_pattern(term: str) -> str:
    parts = [re.escape(part) for part in _normalize_text(term).split()]
    return r"\b" + r"\s+".join(parts) + _PLURAL + r"\b"


@functools.lru_cache(maxsize=64)
def _compiled(
    terms: Tuple[str, ...],
) -> Optional[Tuple[Pattern[str], Tuple[int, ...]]]:
    """One case-insensitive pattern for a keyword set.

    Each term is one capturing group; the second element maps a group number
    (1-based) to the term's index. Longer terms come first in the alternation
    so a phrase wins over a word it contains at the same position.
    """
    indexed = [(i, t) for i, t in enumerate(terms) if t.strip()]
    if not indexed:
        return None
    indexed.sort(key=lambda pair: -len(pair[1]))
    alternatives = [f"({_term_pattern(term)})" for _i, term in indexed]
    pattern = re.compile("|".join(alternatives), re.IGNORECASE)
    return pattern, tuple(i for i, _term in indexed)


def match_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    """Return the keywords found in ``text`` as whole words or phrases.

    Each keyword appears once, in the order of its first occurrence.
    """
    terms = tuple(str(k) for k in keywords)
    compiled = _compiled(terms)
    if compiled is None:
        return []
    pattern, positions = compiled
    found: List[str] = []
    for match in pattern.finditer(_normalize_text(text)):
        if match.lastindex is None:
            continue
        term = terms[positions[match.lastindex - 1]]
        if term not in found:
            found.append(term)
    return found


@dataclass
class ClassificationResult:
    """One verdict: the tier, its confidence, what matched and why.

    ``matched`` maps ``core``, ``private`` and ``relationship`` to the terms
    found (tags appear as ``tag:<name>``).
    """

    classification: DataClassification
    confidence: float
    matched: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""


def _parse_map(
    mapping: Optional[Mapping[str, Union[str, DataClassification]]],
) -> Dict[str, DataClassification]:
    if mapping is None:
        return dict(DEFAULT_CONTENT_TYPE_MAP)
    return {str(k): DataClassification.parse(v) for k, v in mapping.items()}


class DataClassifier:
    """Keyword, tag and content-type classification into privacy tiers.

    ``relationship_keywords`` None reads ``RELATIONSHIP_KEYWORDS`` from the
    passed Settings, or from ``get_settings()``, on every call. The private
    and core sets default to the documented generic lists; ``content_type_map``
    replaces ``DEFAULT_CONTENT_TYPE_MAP`` when passed.
    """

    def __init__(
        self,
        relationship_keywords: Optional[Sequence[str]] = None,
        private_keywords: Optional[Sequence[str]] = None,
        core_keywords: Optional[Sequence[str]] = None,
        content_type_map: Optional[Mapping[str, Union[str, DataClassification]]] = None,
        settings: Optional[Settings] = None,
        *,
        protective_confidence: float = DEFAULT_PROTECTIVE_CONFIDENCE,
    ) -> None:
        if not 0.0 <= protective_confidence <= 1.0:
            raise ValueError("protective_confidence must be within [0, 1]")
        self._relationship = (
            tuple(relationship_keywords) if relationship_keywords is not None else None
        )
        self._private = tuple(
            private_keywords
            if private_keywords is not None
            else DEFAULT_PRIVATE_KEYWORDS
        )
        self._core = tuple(
            core_keywords if core_keywords is not None else DEFAULT_CORE_KEYWORDS
        )
        self._content_types = _parse_map(content_type_map)
        self._settings = settings
        self.protective_confidence = float(protective_confidence)

    def _settings_now(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def relationship_keywords(self) -> Tuple[str, ...]:
        """The relationship set in force now."""
        if self._relationship is not None:
            return self._relationship
        return tuple(csv_list(self._settings_now().RELATIONSHIP_KEYWORDS))

    @property
    def content_type_map(self) -> Dict[str, DataClassification]:
        """A copy of the content-type map in force."""
        return dict(self._content_types)

    def classify(
        self,
        content: str,
        content_type: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        *,
        keywords_can_raise: bool = False,
    ) -> ClassificationResult:
        """Classify one piece of content (see the module docstring).

        ``keywords_can_raise`` (rule 1b) lets the keyword verdict raise, never
        lower, the verdict of a mapped content type.
        """
        if content_type and content_type in self._content_types:
            tier = self._content_types[content_type]
            typed = ClassificationResult(
                classification=tier,
                confidence=CONTENT_TYPE_CONFIDENCE,
                matched={"core": [], "private": [], "relationship": []},
                reason=f"content type {content_type!r} maps to {tier.value}",
            )
            if not keywords_can_raise:
                return typed
            keyword = self._keyword_verdict(content, tags)
            if keyword.classification <= tier:
                return typed
            return ClassificationResult(
                classification=keyword.classification,
                confidence=keyword.confidence,
                matched=keyword.matched,
                reason=(
                    f"{typed.reason}; raised to {keyword.classification.value}: "
                    f"{keyword.reason}"
                ),
            )
        return self._keyword_verdict(content, tags)

    def _keyword_verdict(
        self, content: str, tags: Optional[Sequence[str]]
    ) -> ClassificationResult:
        """Rules 2 and 3 of the module docstring: keywords, tags, protection."""
        text = content or ""
        tag_set = {str(t).strip().lower() for t in (tags or ())}
        matched = {
            "core": match_keywords(text, self._core),
            "private": match_keywords(text, self._private),
            "relationship": match_keywords(text, self.relationship_keywords()),
        }
        for key, wanted in (
            ("core", CORE_TAGS),
            ("private", PRIVATE_TAGS),
            ("relationship", RELATIONSHIP_TAGS),
        ):
            matched[key].extend(f"tag:{tag}" for tag in sorted(tag_set & wanted))

        tier = DataClassification.PUBLIC
        confidence = PUBLIC_CONFIDENCE
        reason = "no privacy-sensitive terms matched"
        for candidate in (
            DataClassification.CORE,
            DataClassification.PRIVATE,
            DataClassification.RELATIONSHIP,
        ):
            hits = matched[candidate.value]
            if hits:
                base, step, ceiling = _TIER_SCORING[candidate]
                tier = candidate
                confidence = min(ceiling, base + step * len(hits))
                reason = f"{candidate.value} terms matched: {hits[:3]}"
                break

        if confidence < self.protective_confidence and tier is not _ORDER[-1]:
            tier = _ORDER[tier.rank + 1]
            reason += (
                f" (raised one tier: confidence {confidence:.2f} is below "
                f"{self.protective_confidence:.2f})"
            )
        return ClassificationResult(
            classification=tier, confidence=confidence, matched=matched, reason=reason
        )


@dataclass(frozen=True)
class StagePolicy:
    """Day thresholds of the access stages; stage N starts at index N.

    The default thresholds are 0, 90, 365, 1095, 2555, 4380 and 6570 days,
    giving stages 0 to 6. The first threshold must be 0 and the thresholds
    must increase.
    """

    thresholds_days: Tuple[int, ...] = (0, 90, 365, 1095, 2555, 4380, 6570)

    def __post_init__(self) -> None:
        values = tuple(int(v) for v in self.thresholds_days)
        if not values or values[0] != 0:
            raise ValueError("thresholds_days must start at 0")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("thresholds_days must strictly increase")
        object.__setattr__(self, "thresholds_days", values)

    @property
    def stages(self) -> int:
        """How many stages the policy has."""
        return len(self.thresholds_days)

    @property
    def final_stage(self) -> int:
        return self.stages - 1

    def stage_for_age(self, age_days: int) -> Optional[int]:
        """The stage at ``age_days``; None for a negative age."""
        if age_days < 0:
            return None
        return sum(1 for t in self.thresholds_days if age_days >= t) - 1


def _birth_date(settings: Settings) -> Optional[date]:
    raw = (settings.SYSTEM_BIRTH_DATE or "").strip()
    return date.fromisoformat(raw) if raw else None


def _utc_day(now: Optional[Union[datetime, date]]) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    if isinstance(now, datetime):
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(timezone.utc).date()
    return now


def age_days(
    now: Optional[Union[datetime, date]] = None, *, settings: Optional[Settings] = None
) -> Optional[int]:
    """Whole days since ``SYSTEM_BIRTH_DATE``, or None when it is not set."""
    active = settings if settings is not None else get_settings()
    born = _birth_date(active)
    if born is None:
        return None
    return (_utc_day(now) - born).days


def access_stage(
    now: Optional[Union[datetime, date]] = None,
    *,
    settings: Optional[Settings] = None,
    policy: Optional[StagePolicy] = None,
) -> Optional[int]:
    """The access stage (0 to the policy's final stage) at ``now``.

    None when ``SYSTEM_BIRTH_DATE`` is empty or ``now`` is before it. A naive
    ``now`` raises ValueError.
    """
    days = age_days(now, settings=settings)
    if days is None:
        return None
    return (policy or StagePolicy()).stage_for_age(days)


def stage_info(
    now: Optional[Union[datetime, date]] = None,
    *,
    settings: Optional[Settings] = None,
    policy: Optional[StagePolicy] = None,
) -> Dict[str, Any]:
    """The access stage with its arithmetic, in neutral words."""
    active = settings if settings is not None else get_settings()
    rules = policy or StagePolicy()
    days = age_days(now, settings=active)
    stage = None if days is None else rules.stage_for_age(days)
    born = _birth_date(active)
    if stage is None:
        description = (
            "stage unknown: no birth date is configured"
            if born is None
            else "stage unknown: the clock is before the birth date"
        )
    else:
        description = (
            f"stage {stage} of {rules.final_stage}: "
            f"{rules.thresholds_days[stage]} days or more since the birth date"
        )
    return {
        "stage": stage,
        "age_days": days,
        "stages": rules.stages,
        "thresholds_days": list(rules.thresholds_days),
        "birth_date": born.isoformat() if born is not None else None,
        "description": description,
    }
