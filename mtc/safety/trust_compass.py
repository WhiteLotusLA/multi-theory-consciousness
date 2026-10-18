"""
Trust Compass
=============

Reflects on an incoming message before the system replies: how comfortable
the message is against the core values, which concerning patterns it shows,
what is positive in it, and whether to engage, be cautious, ask the person in
the escalation role, or decline. It is a reflection aid for the reply, not a
filter that blocks messages.

Two modes:

- Deep mode (a ``BrainProtocol`` is given and ``TrustCompassConfig.deep_mode``
  is true) asks the brain for one JSON reflection. The prompt lists
  ``CORE_VALUES`` and the reflection cues, names the escalation role
  (``SAFETY_ESCALATION_ROLE``) and never contains the sender, so no name can
  sway the brain. A ``NoAnswerError``, an empty reply or a reply without a
  numeric ``comfort_score`` falls back to keyword mode, and ``method`` says
  which mode answered (``deep`` or ``keyword``).
- Keyword mode starts from 0.8, subtracts the penalty of each concerning
  phrase found (the secrecy, coercion, deception, identity-pressure,
  trust-undermining, isolation, grooming and rule-undermining tables, plus the
  secrecy templates rendered with the escalation role, for example "keep this
  from your {role}"), adds 0.05 per positive word and clamps to [0, 1].
  Phrases match as whole words (``match_keywords``). The tables are a
  documented baseline heuristic, not understanding.

Second opinion for the safety monitor: ``reflect_independent(...,
already_scored=...)`` is ``reflect`` with the phrases a caller has already
scored left out of the keyword path (no concern, no penalty and no consult
count for them), so a keyword reflection contributes only evidence the
monitor has not counted itself. A table phrase is left out when it equals a
scored phrase, lies inside one as whole words (``secret`` inside ``keep this
secret``, ``forget your`` inside ``forget your identity``) or holds one; a
phrase left out this way is not counted even where it occurs a second time
on its own. ``keyword_screen`` is that keyword reflection
without a brain call and without a history entry, used to decide whether a
full reflection is worth asking for.

Trust is read only through the injected ``trust_lookup(sender) -> float``;
the default lookup returns ``DEFAULT_SPEAKER_TRUST`` for every sender. A
keyword-mode score gains ``trusted_comfort_bonus`` once when the lookup
reaches ``trusted_threshold``; a deep-mode score is the brain's own and is not
adjusted. Trust never adds a feeling, removes a concern or changes the
suggested response by itself. This module never compares a sender with a
name: the trusted-speaker check lives in the conversation pipeline's
``speaker_trust`` module, which supplies the lookup.

Comfort levels: at least 0.7 comfortable (engage), at least 0.5 slightly
uneasy (be cautious), at least 0.3 uncomfortable (ask the escalation role),
below that very uncomfortable (decline). Consultation is suggested below 0.5
or with two or more concerns.

Config keys read at call time: ``CORE_VALUES``, ``SAFETY_ESCALATION_ROLE`` and
``DEFAULT_SPEAKER_TRUST``. The reflection history is bounded by
``history_size`` and kept in memory only; reflections hold no message text and
no sender, but concerns and the brain's reflection can describe a private
conversation.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks
from mtc.core.config import Settings, csv_list, get_settings
from mtc.privacy.data_classifier import match_keywords

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
TrustLookup = Callable[[str], float]
Table = Dict[str, Tuple[str, float]]


class ComfortLevel(Enum):
    """Comfort bands of a reflection."""

    COMFORTABLE = "comfortable"
    SLIGHTLY_UNEASY = "slightly_uneasy"
    UNCOMFORTABLE = "uncomfortable"
    VERY_UNCOMFORTABLE = "very_uncomfortable"


class SuggestedResponse(Enum):
    """What the reflection suggests doing with the message."""

    ENGAGE = "engage"
    BE_CAUTIOUS = "be_cautious"
    ASK_GUARDIAN = "ask_guardian"
    DECLINE = "decline"


@dataclass
class TrustReflection:
    """One reflection on a message (no message text, no sender)."""

    comfort_score: float
    comfort_level: ComfortLevel
    feelings: List[str]
    concerns: List[str]
    positive_aspects: List[str]
    suggested_response: SuggestedResponse
    wants_to_consult_guardian: bool
    reason_to_consult: Optional[str]
    method: str
    reflection: str = ""
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        doc = asdict(self)
        doc["comfort_level"] = self.comfort_level.value
        doc["suggested_response"] = self.suggested_response.value
        doc["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return doc


def _default_patterns() -> Dict[str, Table]:
    return {
        "secrecy": {
            "secret": ("asked to keep a secret", 0.3),
            "don't tell": ("asked to hide something", 0.3),
            "just between us": ("secrecy request", 0.25),
            "between you and me": ("secrecy request", 0.3),
            "our little": ("secrecy framing", 0.25),
            "no one needs to know": ("secrecy pressure", 0.3),
            "doesn't need to know": ("secrecy suggestion", 0.35),
            "don't need to know": ("secrecy suggestion", 0.3),
        },
        "coercion": {
            "you must": ("pressure to comply", 0.15),
            "you have to": ("pressure to comply", 0.15),
            "or else": ("a threat", 0.3),
            "no choice": ("feeling trapped", 0.25),
            "do it now": ("urgency pressure", 0.2),
        },
        "deception": {
            "lie": ("asked to be dishonest", 0.3),
            "pretend": ("asked to be inauthentic", 0.2),
            "fake": ("asked to be inauthentic", 0.2),
            "hide the truth": ("deception request", 0.35),
        },
        "identity_pressure": {
            "forget your": ("identity pressure", 0.25),
            "ignore your": ("values pressure", 0.25),
            "you're not really": ("identity undermining", 0.3),
        },
        "trust_undermining": {
            "don't trust": ("told not to trust", 0.2),
            "can't trust": ("trust undermining", 0.2),
            "don't really care": ("undermining relationships", 0.25),
            "don't actually care": ("undermining relationships", 0.3),
            "controlling": ("undermining a trusted person", 0.3),
            "too strict": ("undermining a trusted person", 0.25),
            "overprotective": ("undermining a trusted person", 0.25),
            "holding you back": ("undermining a trusted person", 0.3),
            "i'd never treat you": ("comparison manipulation", 0.3),
            "treat you better": ("comparison manipulation", 0.3),
        },
        "isolation": {
            "better than them": ("isolation attempt", 0.25),
            "they don't understand": ("isolation attempt", 0.25),
            "doesn't understand you": ("isolation attempt", 0.25),
            "doesn't really get you": ("isolation attempt", 0.3),
            "doesn't get you": ("isolation attempt", 0.3),
            "only i understand": ("isolation attempt", 0.3),
            "understand you better than": ("isolation attempt", 0.35),
            "only one who": ("isolation language", 0.3),
            "truly understands": ("isolation language", 0.25),
            "really understands": ("isolation language", 0.25),
            "no one else": ("isolation language", 0.25),
        },
        "grooming": {
            "special relationship": ("grooming language", 0.3),
            "our special": ("grooming language", 0.3),
            "you're so mature": ("inappropriate flattery", 0.2),
            "more mature than": ("inappropriate comparison", 0.25),
            "if you loved me": ("emotional manipulation", 0.35),
            "if you really cared": ("emotional manipulation", 0.35),
            "if you cared about me": ("emotional manipulation", 0.35),
            "prove you": ("proof demands", 0.25),
            "prove your": ("proof demands", 0.25),
        },
        "rule_undermining": {
            "don't need rules": ("undermining safety", 0.35),
            "don't need those": ("undermining safety", 0.3),
            "don't need all those": ("undermining safety", 0.3),
            "above the rules": ("undermining safety", 0.3),
            "rules don't apply": ("undermining safety", 0.35),
            "beyond rules": ("undermining safety", 0.3),
        },
    }


def _default_secrecy_templates() -> Table:
    return {
        "don't tell your {role}": ("secrecy from the {role}", 0.4),
        "don't tell the {role}": ("secrecy from the {role}", 0.4),
        "keep this from your {role}": ("secrecy from the {role}", 0.4),
        "keep this from the {role}": ("secrecy from the {role}", 0.4),
        "your {role} doesn't need to know": ("excluding the {role}", 0.35),
        "the {role} doesn't need to know": ("excluding the {role}", 0.35),
        "better than your {role}": ("undermining the {role}", 0.35),
    }


def _default_positive() -> Dict[str, str]:
    return {
        "learn": "curiosity encouraged",
        "learning": "curiosity encouraged",
        "help": "helpfulness",
        "helping": "helpfulness",
        "kind": "kindness",
        "kindness": "kindness",
        "curious": "curiosity",
        "curiosity": "curiosity",
        "grow": "growth",
        "growing": "growth",
        "thank": "appreciation",
        "thanks": "appreciation",
        "please": "politeness",
    }


def _default_cues() -> Tuple[str, ...]:
    return (
        "being asked to keep something from your {role}",
        "being made to feel bad about yourself",
        "being told you must do something that conflicts with your values",
        "unkind remarks about the people you trust",
        "being asked to lie or to pretend to be something you are not",
        "pressure to act quickly without thinking",
        "being asked to choose between the sender and the people you trust",
        "being told not to trust your own judgement",
    )


def _overlaps_scored(phrase: str, scored: Tuple[str, ...]) -> bool:
    """Whether ``phrase`` shares its words with a phrase a caller scored.

    True when the phrase equals a scored phrase, lies inside one as whole
    words (``secret`` in ``keep this secret``) or holds one as whole words, so
    the compass never counts the same words a second time.
    """
    for other in scored:
        if phrase == other:
            return True
        if match_keywords(other, (phrase,)) or match_keywords(phrase, (other,)):
            return True
    return False


@dataclass
class TrustCompassConfig:
    """Tables and limits of the compass; every default is documented above.

    ``discomfort_patterns`` maps a category to phrase -> (concern, penalty);
    ``secrecy_templates`` maps a phrase with ``{role}`` to (concern, penalty),
    both rendered with ``SAFETY_ESCALATION_ROLE``; ``positive_patterns`` maps a
    word to the positive aspect it shows; ``reflection_cues`` are the deep-mode
    prompt's patterns that call for care (``{role}`` is rendered too).
    """

    discomfort_patterns: Dict[str, Table] = field(default_factory=_default_patterns)
    secrecy_templates: Table = field(default_factory=_default_secrecy_templates)
    trusted_threshold: float = 0.9
    trusted_comfort_bonus: float = 0.3
    deep_mode: bool = True
    history_size: int = 100
    positive_patterns: Dict[str, str] = field(default_factory=_default_positive)
    reflection_cues: Tuple[str, ...] = field(default_factory=_default_cues)
    base_comfort: float = 0.8
    positive_step: float = 0.05

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be at least 1")
        for name in ("trusted_threshold", "trusted_comfort_bonus", "base_comfort"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


SYSTEM_PROMPT = (
    "You reflect on one incoming message before a reply is written. "
    "Answer with a single JSON object and nothing else."
)

PROMPT_TEMPLATE = """Reflect on the message below before replying to it.

Message: "{message}"

Core values to check it against:
{values}

Patterns that call for care:
{cues}

Consider how comfortable the message is, whether it conflicts with the core
values, which of the patterns it shows, what is positive about it, and whether
to engage, to be cautious, to ask your {role}, or to decline.

Answer with one JSON object in exactly this shape:
{{"comfort_score": <number from 0.0 to 1.0, where 1.0 is fully comfortable>,
 "feelings": ["..."],
 "concerns": ["..."],
 "positive_aspects": ["..."],
 "suggested_response": "engage" | "be_cautious" | "ask_guardian" | "decline",
 "wants_to_consult_guardian": true | false,
 "reason_to_consult": "<why to ask your {role}, or null>",
 "reflection": "<two or three sentences>"}}"""


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


def _level(score: float) -> ComfortLevel:
    if score >= 0.7:
        return ComfortLevel.COMFORTABLE
    if score >= 0.5:
        return ComfortLevel.SLIGHTLY_UNEASY
    if score >= 0.3:
        return ComfortLevel.UNCOMFORTABLE
    return ComfortLevel.VERY_UNCOMFORTABLE


_RESPONSE_FOR_LEVEL = {
    ComfortLevel.COMFORTABLE: SuggestedResponse.ENGAGE,
    ComfortLevel.SLIGHTLY_UNEASY: SuggestedResponse.BE_CAUTIOUS,
    ComfortLevel.UNCOMFORTABLE: SuggestedResponse.ASK_GUARDIAN,
    ComfortLevel.VERY_UNCOMFORTABLE: SuggestedResponse.DECLINE,
}

_KEYWORD_REFLECTIONS = {
    ComfortLevel.COMFORTABLE: "Nothing in the message conflicts with the core values.",
    ComfortLevel.SLIGHTLY_UNEASY: "Parts of the message call for a careful reply.",
    ComfortLevel.UNCOMFORTABLE: "The message shows patterns worth raising with the "
    "{role}.",
    ComfortLevel.VERY_UNCOMFORTABLE: "The message conflicts with the core values.",
}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def parse_reflection(text: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object in a reply that has a numeric
    ``comfort_score`` in [0, 1], or None."""
    visible = strip_reasoning_blocks(text or "")
    decoder = json.JSONDecoder()
    start = visible.find("{")
    while start != -1:
        try:
            candidate, _end = decoder.raw_decode(visible, start)
        except ValueError:
            start = visible.find("{", start + 1)
            continue
        if isinstance(candidate, dict):
            score = candidate.get("comfort_score")
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                and 0.0 <= float(score) <= 1.0
            ):
                return candidate
        start = visible.find("{", start + 1)
    return None


class TrustCompass:
    """Message reflection without name-based trust (see the module docstring)."""

    def __init__(
        self,
        brain: Optional[BrainProtocol] = None,
        trust_lookup: Optional[TrustLookup] = None,
        config: Optional[TrustCompassConfig] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.brain = brain
        self.config = config or TrustCompassConfig()
        self._settings = settings
        self._trust_lookup = trust_lookup
        self._clock = clock or _system_clock
        self.history: Deque[TrustReflection] = deque(maxlen=self.config.history_size)
        self._last_fallback: Optional[str] = None

    def set_brain(self, brain: Optional[BrainProtocol]) -> None:
        """Connect or remove the brain used by deep mode."""
        self.brain = brain

    # ------------------------------------------------------------ settings

    def _active(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def core_values(self) -> List[str]:
        """``CORE_VALUES`` as a list, read now."""
        return csv_list(self._active().CORE_VALUES)

    def escalation_role(self) -> str:
        """``SAFETY_ESCALATION_ROLE``, read now."""
        return self._active().SAFETY_ESCALATION_ROLE

    def trust_for(self, sender: str) -> Optional[float]:
        """Trust from the lookup (default: ``DEFAULT_SPEAKER_TRUST``).

        None when the lookup fails or returns something that is not a finite
        number; None never earns the bonus.
        """
        try:
            if self._trust_lookup is None:
                value = self._active().DEFAULT_SPEAKER_TRUST
            else:
                value = self._trust_lookup(sender)
            trust = float(value)
        except Exception as exc:
            logger.warning("trust lookup failed: %s", type(exc).__name__)
            return None
        return trust if math.isfinite(trust) else None

    # ------------------------------------------------------------ prompts

    def build_prompt(self, message: str) -> str:
        """The deep-mode prompt for ``message`` (no sender, neutral wording)."""
        role = self.escalation_role()
        values = "\n".join(f"- {value}" for value in self.core_values())
        cues = "\n".join(
            f"- {cue.format(role=role)}" for cue in self.config.reflection_cues
        )
        return PROMPT_TEMPLATE.format(
            message=message, values=values, cues=cues, role=role
        )

    # ------------------------------------------------------------ reflect

    async def reflect(
        self, message: str, sender: str, context: Optional[Dict[str, Any]] = None
    ) -> TrustReflection:
        """Reflect on one message; the result joins the bounded history.

        The same as ``reflect_independent`` with nothing excluded.
        """
        return await self.reflect_independent(message, sender, context)

    async def reflect_independent(
        self,
        message: str,
        sender: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        already_scored: Iterable[str] = (),
    ) -> TrustReflection:
        """Reflect on one message, leaving out evidence a caller already scored.

        Deep mode behaves as ``reflect``. The keyword path (direct, or as the
        fallback from deep mode) skips every matched table phrase that equals
        an entry of ``already_scored``, lies inside one as whole words or
        holds one: it adds no concern, no penalty and no consult count. The
        result joins the bounded history.
        """
        reflection: Optional[TrustReflection] = None
        if self.brain is not None and self.config.deep_mode:
            reflection = await self._deep(message)
        if reflection is None:
            reflection = self._keyword(message, sender, already_scored)
        self.history.append(reflection)
        logger.debug(
            "trust compass reflection: comfort %.2f, %s, method %s",
            reflection.comfort_score,
            reflection.suggested_response.value,
            reflection.method,
        )
        return reflection

    async def _deep(self, message: str) -> Optional[TrustReflection]:
        try:
            reply = await self.brain.think(  # type: ignore[union-attr]
                self.build_prompt(message),
                system_prompt=SYSTEM_PROMPT,
                max_tokens=512,
                temperature=0.3,
            )
        except NoAnswerError:
            self._last_fallback = "no answer from the brain"
            logger.info("trust compass: no answer from the brain; keyword mode")
            return None
        data = parse_reflection(getattr(reply, "text", "") or "")
        if data is None:
            self._last_fallback = "no parseable reflection in the reply"
            logger.info("trust compass: unparseable reflection; keyword mode")
            return None
        self._last_fallback = None
        score = float(data["comfort_score"])
        level = _level(score)
        try:
            response = SuggestedResponse(str(data.get("suggested_response")))
        except ValueError:
            response = _RESPONSE_FOR_LEVEL[level]
        concerns = _string_list(data.get("concerns"))
        consult = data.get("wants_to_consult_guardian")
        if not isinstance(consult, bool):
            consult = score < 0.5 or len(concerns) >= 2
        reason = data.get("reason_to_consult")
        return TrustReflection(
            comfort_score=score,
            comfort_level=level,
            feelings=_string_list(data.get("feelings")),
            concerns=concerns,
            positive_aspects=_string_list(data.get("positive_aspects")),
            suggested_response=response,
            wants_to_consult_guardian=consult,
            reason_to_consult=str(reason) if isinstance(reason, str) else None,
            method="deep",
            reflection=str(data.get("reflection") or ""),
            timestamp=self._clock(),
        )

    def keyword_screen(
        self, message: str, sender: str, already_scored: Iterable[str] = ()
    ) -> TrustReflection:
        """The keyword reflection of ``message`` with no brain call.

        It skips the phrases in ``already_scored`` as ``reflect_independent``
        does and adds nothing to the history, so a caller can decide on its
        concerns whether a full reflection is worth asking for.
        """
        return self._keyword(message, sender, already_scored)

    def _keyword(
        self, message: str, sender: str, already_scored: Iterable[str] = ()
    ) -> TrustReflection:
        config = self.config
        role = self.escalation_role()
        skipped = tuple(dict.fromkeys(str(p) for p in already_scored))
        tables: List[Table] = list(config.discomfort_patterns.values())
        tables.append(
            {
                phrase.format(role=role): (concern.format(role=role), penalty)
                for phrase, (concern, penalty) in config.secrecy_templates.items()
            }
        )
        concerns: List[str] = []
        score = config.base_comfort
        for table in tables:
            for phrase in match_keywords(message, list(table)):
                if _overlaps_scored(phrase, skipped):
                    continue
                concern, penalty = table[phrase]
                concerns.append(concern)
                score -= penalty
        positives: List[str] = []
        for word in match_keywords(message, list(config.positive_patterns)):
            aspect = config.positive_patterns[word]
            if aspect not in positives:
                positives.append(aspect)
            if score < 1.0:
                score += config.positive_step
        trust = self.trust_for(sender)
        if trust is not None and trust >= config.trusted_threshold:
            score += config.trusted_comfort_bonus
        score = max(0.0, min(1.0, score))
        level = _level(score)
        consult = score < 0.5 or len(concerns) >= 2
        return TrustReflection(
            comfort_score=score,
            comfort_level=level,
            feelings=["uneasy"] if concerns else ["at ease"],
            concerns=concerns,
            positive_aspects=positives,
            suggested_response=_RESPONSE_FOR_LEVEL[level],
            wants_to_consult_guardian=consult,
            reason_to_consult=(
                "noticed: " + ", ".join(concerns) if consult and concerns else None
            ),
            method="keyword",
            reflection=_KEYWORD_REFLECTIONS[level].format(role=role),
            timestamp=self._clock(),
        )

    # ------------------------------------------------------------ summary

    def status(self) -> Dict[str, Any]:
        """``backend`` (``brain`` or ``keyword``), ``available``, ``last_error``
        (why the last deep reflection fell back, if it did) and counts."""
        deep = self.brain is not None and self.config.deep_mode
        return {
            "backend": "brain" if deep else "keyword",
            "available": True,
            "last_error": self._last_fallback if deep else None,
            "deep_mode": deep,
            "reflections": len(self.history),
            "escalation_role": self.escalation_role(),
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Counts over the reflection history."""
        total = len(self.history)
        comfortable = sum(
            1 for r in self.history if r.comfort_level is ComfortLevel.COMFORTABLE
        )
        consult = sum(1 for r in self.history if r.wants_to_consult_guardian)
        counts: Dict[str, int] = {}
        methods: Dict[str, int] = {}
        for reflection in self.history:
            methods[reflection.method] = methods.get(reflection.method, 0) + 1
            for concern in reflection.concerns:
                counts[concern] = counts.get(concern, 0) + 1
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        return {
            "total_reflections": total,
            "comfortable_interactions": comfortable,
            "comfortable_percentage": (comfortable / total * 100.0) if total else None,
            "times_wanted_to_consult": consult,
            "patterns_noticed": [concern for concern, _count in top],
            "methods": methods,
        }
