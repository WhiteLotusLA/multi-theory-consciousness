"""
Theory of Mind
==============

Keeps an inferred model of each speaker the system actually talks with: their
typical affect, communication style, topics, likely goals and values, and a
set of relationship indices. The models are heuristics computed from
observable signals of each turn (message length, questions, keywords, the
supplied emotion and topics); they make no language-model call and claim no
access to anyone's mind.

Speakers and trust:

- No speaker is seeded and none is ever created except by a turn from that
  speaker (``update_from_turn``). A forgotten speaker stays forgotten:
  neither ``initialize`` nor a lookup brings it back.
- ``relationship.trust`` is set from ``trust_lookup(speaker)`` on every update
  and on every load, and never by a heuristic. The default lookup ignores the
  name and returns ``DEFAULT_SPEAKER_TRUST`` (read from settings at call
  time); a lookup that fails or returns a non-numeric value falls back to it,
  and values are clipped to [0, 1]. The Phase 6 pipeline passes a lookup
  backed by its token-checked speaker-trust module.

Smoothing: warmth, reciprocity, depth, the communication style and the
emotional profile move by an exponential moving average with weight
``ToMConfig.smoothing`` (0.85) on the old value. Style and relationship
indices start from documented neutral priors (``ToMConfig``); the emotional
profile starts unknown (None) and takes its first observed value directly. A
missing emotion or conversation depth leaves the parts that need it
unchanged. Familiarity is 1 - 1 / (1 + 0.05 * interactions).

Narration: ``get_context_for_prompt`` returns an empty string unless the
constructor's ``narrate_speaker_model`` is True; the block is neutral third
person and labels trust as configured, not inferred.

Persistence: models are stored through a ``SpeakerModelStore``
(``mtc.memory.protocols``; None means ``InMemorySpeakerModelStore``) keyed by
speaker name. Speaker models hold real people's names at runtime and are
private data; log lines never include a speaker name. Timestamps are
timezone-aware UTC ISO strings from the injectable clock.
"""

import logging
import math
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from mtc.core.config import get_settings
from mtc.memory.inmemory import InMemorySpeakerModelStore
from mtc.memory.protocols import SpeakerModelStore

logger = logging.getLogger(__name__)

Phrases = Tuple[str, ...]


def _default_goal_keywords() -> Dict[str, Phrases]:
    return {
        "connecting": ("hello", "hi", "hey", "how are you", "checking in"),
        "seeking_help": ("help", "problem", "stuck", "issue", "trouble"),
        "making_something": ("create", "make", "plan", "write", "design"),
        "exploring_ideas": ("think", "wonder", "curious", "what if", "imagine"),
        "verifying": ("check", "confirm", "verify", "make sure"),
    }


def _default_value_topics() -> Dict[str, Phrases]:
    return {
        "understanding the mind": ("consciousness", "awareness", "experience", "mind"),
        "growth and learning": ("growth", "learning", "progress", "improvement"),
        "creative expression": ("art", "music", "creative", "beauty", "writing"),
        "knowledge and ideas": ("science", "philosophy", "history", "ideas"),
        "wellbeing": ("health", "rest", "sleep", "wellbeing"),
    }


def _default_mood_keywords() -> Dict[str, Phrases]:
    return {
        "enthusiastic": ("excited", "amazing", "love", "great", "awesome"),
        "weary": ("tired", "exhausted", "rough", "hard"),
        "anxious": ("worried", "concerned", "afraid", "nervous"),
    }


@dataclass
class ToMConfig:
    """Smoothing, caps, priors and the keyword tables of the heuristics.

    Keyword tables match whole words and phrases, case-folded. Priors are the
    neutral starting points of the moving averages, not observations.
    """

    smoothing: float = 0.85
    topic_cap: int = 150
    recent_topics: int = 10
    max_goals: int = 3
    max_values: int = 5
    values_every_turns: int = 5
    min_topic_count_for_value: int = 3
    deep_conversation_turns: int = 5
    deep_message_chars: float = 300.0
    short_message_chars: int = 20
    # Priors of the moving averages
    message_length_prior: float = 100.0
    question_ratio_prior: float = 0.3
    directness_prior: float = 0.5
    depth_preference_prior: float = 0.5
    warmth_prior: float = 0.5
    depth_prior: float = 0.3
    reciprocity_prior: float = 0.5
    # Keyword tables
    depth_topics: Phrases = (
        "consciousness",
        "philosophy",
        "identity",
        "experience",
        "meaning",
    )
    direct_openers: Phrases = (
        "do",
        "make",
        "tell",
        "show",
        "give",
        "send",
        "find",
        "start",
        "stop",
        "please",
    )
    warmth_phrases: Phrases = (
        "how are you",
        "how do you feel",
        "what do you think",
        "what is on your mind",
        "how does it feel",
        "are you okay",
    )
    continue_phrases: Phrases = ("continue", "keep going", "next", "proceed", "go on")
    perspective_phrases: Phrases = ("what do you think", "how do you feel", "tell me")
    goal_keywords: Dict[str, Phrases] = field(default_factory=_default_goal_keywords)
    value_topics: Dict[str, Phrases] = field(default_factory=_default_value_topics)
    mood_keywords: Dict[str, Phrases] = field(default_factory=_default_mood_keywords)


@dataclass
class EmotionalProfile:
    """The speaker's observed affect; None until an emotion is supplied.

    Valence lies in [-1, 1] and arousal in [0, 1], as supplied by the caller.
    """

    avg_valence: Optional[float] = None
    avg_arousal: Optional[float] = None
    recent_valence_trend: Optional[float] = None


@dataclass
class CommunicationStyle:
    """Observed communication patterns (moving averages from neutral priors)."""

    avg_message_length: float = 100.0
    question_ratio: float = 0.3
    directness: float = 0.5
    depth_preference: float = 0.5


@dataclass
class RelationshipState:
    """Relationship indices; ``trust`` comes only from the trust lookup."""

    warmth: float = 0.5
    trust: float = 0.5
    depth: float = 0.3
    reciprocity: float = 0.5
    familiarity: float = 0.0


@dataclass
class SpeakerModel:
    """The inferred model of one speaker (private data at runtime)."""

    speaker_name: str
    first_seen: str = ""
    last_seen: str = ""
    interaction_count: int = 0
    emotional_profile: EmotionalProfile = field(default_factory=EmotionalProfile)
    communication_style: CommunicationStyle = field(default_factory=CommunicationStyle)
    relationship: RelationshipState = field(default_factory=RelationshipState)
    topic_frequencies: Dict[str, int] = field(default_factory=dict)
    recent_topics: List[str] = field(default_factory=list)
    knowledge_domains: Dict[str, float] = field(default_factory=dict)
    likely_goals: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    predicted_mood: str = "unknown"
    predicted_intent: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_name": self.speaker_name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "emotional_profile": asdict(self.emotional_profile),
            "communication_style": asdict(self.communication_style),
            "relationship": asdict(self.relationship),
            "topic_frequencies": dict(self.topic_frequencies),
            "recent_topics": list(self.recent_topics),
            "knowledge_domains": dict(self.knowledge_domains),
            "likely_goals": list(self.likely_goals),
            "values": list(self.values),
            "predicted_mood": self.predicted_mood,
            "predicted_intent": self.predicted_intent,
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "SpeakerModel":
        def part(kind: type, data: Any) -> Any:
            names = {f.name for f in fields(kind)}
            items = data.items() if isinstance(data, Mapping) else ()
            return kind(**{k: v for k, v in items if k in names})

        return cls(
            speaker_name=str(doc.get("speaker_name", "")),
            first_seen=str(doc.get("first_seen", "")),
            last_seen=str(doc.get("last_seen", "")),
            interaction_count=int(doc.get("interaction_count", 0) or 0),
            emotional_profile=part(EmotionalProfile, doc.get("emotional_profile")),
            communication_style=part(
                CommunicationStyle, doc.get("communication_style")
            ),
            relationship=part(RelationshipState, doc.get("relationship")),
            topic_frequencies={
                str(k): int(v) for k, v in (doc.get("topic_frequencies") or {}).items()
            },
            recent_topics=[str(t) for t in doc.get("recent_topics") or []],
            knowledge_domains={
                str(k): float(v)
                for k, v in (doc.get("knowledge_domains") or {}).items()
            },
            likely_goals=[str(g) for g in doc.get("likely_goals") or []],
            values=[str(v) for v in doc.get("values") or []],
            predicted_mood=str(doc.get("predicted_mood", "unknown")),
            predicted_intent=str(doc.get("predicted_intent", "unknown")),
        )


def _ema(old: float, new: float, alpha: float) -> float:
    """Exponential moving average with weight ``alpha`` on the old value."""
    return alpha * old + (1.0 - alpha) * new


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _phrase_pattern(phrases: Sequence[str]) -> Optional["re.Pattern[str]"]:
    words = [re.escape(p.lower()) for p in phrases if p and p.strip()]
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


def _mentions(text: str, phrases: Sequence[str]) -> bool:
    """True when the case-folded text holds one of the phrases as whole words."""
    pattern = _phrase_pattern(phrases)
    return bool(pattern and pattern.search(text))


def default_trust_lookup(speaker: str) -> float:
    """The configured default trust; the speaker's name is never consulted."""
    return float(get_settings().DEFAULT_SPEAKER_TRUST)


def _fmt(value: Optional[float]) -> str:
    return "not observed" if value is None else f"{value:.2f}"


class TheoryOfMind:
    """Speaker models updated from conversation turns.

    Args:
        store: Speaker model store; None means a new in-memory store.
        trust_lookup: Returns the trust for a speaker; None means
            ``default_trust_lookup`` (``DEFAULT_SPEAKER_TRUST`` for everyone).
        narrate_speaker_model: When False (the default) the prompt block is
            empty.
        config: Smoothing, priors and keyword tables.
        clock: Returns the current aware UTC time.
    """

    def __init__(
        self,
        store: Optional[SpeakerModelStore] = None,
        trust_lookup: Optional[Callable[[str], float]] = None,
        narrate_speaker_model: bool = False,
        config: Optional[ToMConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.store: SpeakerModelStore = (
            store if store is not None else InMemorySpeakerModelStore()
        )
        self._trust_lookup = trust_lookup or default_trust_lookup
        self.narrate_speaker_model = narrate_speaker_model
        self.config = config or ToMConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.models: Dict[str, SpeakerModel] = {}

    # ======== TRUST ========

    def _trust(self, speaker: str) -> float:
        """The lookup's trust for the speaker, clipped to [0, 1]."""
        try:
            value = _finite(self._trust_lookup(speaker))
        except Exception as e:
            logger.warning(f"Trust lookup failed, using the configured default: {e}")
            value = None
        if value is None:
            value = float(get_settings().DEFAULT_SPEAKER_TRUST)
        return min(1.0, max(0.0, value))

    # ======== LOADING ========

    async def initialize(self) -> None:
        """Load the stored models, refreshing each model's trust from the lookup.

        Only stored models load; nothing is seeded. A store failure is logged
        and leaves the in-memory models as they are.
        """
        try:
            docs = await self.store.load_models()
        except Exception as e:
            logger.warning(f"Speaker models could not be loaded: {e}")
            return
        loaded: Dict[str, SpeakerModel] = {}
        for doc in docs:
            try:
                model = SpeakerModel.from_dict(doc)
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipped an unreadable speaker model: {e}")
                continue
            if not model.speaker_name.strip():
                continue
            model.relationship.trust = self._trust(model.speaker_name)
            loaded[model.speaker_name] = model
        self.models = loaded
        logger.info(f"Theory of mind loaded {len(loaded)} speaker models")

    # ======== UPDATES ========

    def update_from_turn(
        self,
        speaker: str,
        message: str,
        emotion: Optional[Dict[str, float]] = None,
        topics: Optional[List[str]] = None,
        *,
        conversation_depth: Optional[int] = None,
    ) -> SpeakerModel:
        """Update (or create) the model of the speaker of this turn.

        Args:
            speaker: The speaker label as given; it must not be blank.
            message: The speaker's message.
            emotion: The speaker's inferred ``valence`` in [-1, 1] and
                ``arousal`` in [0, 1]; a missing key leaves that axis as is.
            topics: Topics of the turn.
            conversation_depth: Turns in the current conversation; depth moves
                only when it is supplied.
        """
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker must be a non-empty label")
        cfg = self.config
        alpha = cfg.smoothing
        now = self._clock().astimezone(timezone.utc).isoformat()
        text = str(message or "")
        lowered = text.lower()
        topic_list = [str(t) for t in (topics or []) if str(t).strip()]

        model = self.models.get(speaker)
        if model is None:
            model = SpeakerModel(
                speaker_name=speaker,
                first_seen=now,
                last_seen=now,
                communication_style=CommunicationStyle(
                    avg_message_length=cfg.message_length_prior,
                    question_ratio=cfg.question_ratio_prior,
                    directness=cfg.directness_prior,
                    depth_preference=cfg.depth_preference_prior,
                ),
                relationship=RelationshipState(
                    warmth=cfg.warmth_prior,
                    depth=cfg.depth_prior,
                    reciprocity=cfg.reciprocity_prior,
                ),
            )
            self.models[speaker] = model
        model.last_seen = now
        model.interaction_count += 1

        # Emotional profile: only the axes supplied.
        emotion = emotion or {}
        valence = _finite(emotion.get("valence"))
        arousal = _finite(emotion.get("arousal"))
        emo = model.emotional_profile
        if valence is not None:
            emo.avg_valence = (
                valence
                if emo.avg_valence is None
                else _ema(emo.avg_valence, valence, alpha)
            )
            emo.recent_valence_trend = valence - emo.avg_valence
        if arousal is not None:
            emo.avg_arousal = (
                arousal
                if emo.avg_arousal is None
                else _ema(emo.avg_arousal, arousal, alpha)
            )

        # Communication style.
        style = model.communication_style
        length = len(text)
        style.avg_message_length = _ema(style.avg_message_length, length, alpha)
        has_question = "?" in text
        style.question_ratio = _ema(
            style.question_ratio, 1.0 if has_question else 0.0, alpha
        )
        depth_signal = min(1.0, length / cfg.deep_message_chars)
        if _mentions(" ".join(topic_list).lower(), cfg.depth_topics):
            depth_signal = min(1.0, depth_signal + 0.3)
        style.depth_preference = _ema(style.depth_preference, depth_signal, alpha)
        opener = _phrase_pattern(cfg.direct_openers)
        is_direct = bool(opener and opener.match(lowered.lstrip()))
        style.directness = _ema(style.directness, 1.0 if is_direct else 0.0, alpha)

        # Relationship indices (trust is never touched here).
        rel = model.relationship
        rel.familiarity = 1.0 - 1.0 / (1.0 + model.interaction_count * 0.05)
        warmth_signal = 0.8 if _mentions(lowered, cfg.warmth_phrases) else 0.5
        if valence is not None and valence > 0.3:
            warmth_signal = min(1.0, warmth_signal + 0.1)
        rel.warmth = _ema(rel.warmth, warmth_signal, alpha)
        rel.reciprocity = _ema(rel.reciprocity, 0.8 if has_question else 0.3, alpha)
        if (
            conversation_depth is not None
            and conversation_depth > cfg.deep_conversation_turns
        ):
            rel.depth = _ema(rel.depth, 0.8, alpha)
        rel.trust = self._trust(speaker)

        # Topics.
        for topic in topic_list:
            count = model.topic_frequencies.pop(topic, 0) + 1
            model.topic_frequencies[topic] = count  # reinserted as most recent
        if len(model.topic_frequencies) > cfg.topic_cap:
            ranked = sorted(
                enumerate(model.topic_frequencies.items()),
                key=lambda item: (item[1][1], item[0]),
                reverse=True,
            )[: cfg.topic_cap]
            model.topic_frequencies = dict(pair for _, pair in sorted(ranked))
        model.recent_topics = (model.recent_topics + topic_list)[-cfg.recent_topics :]

        # Heuristic inferences.
        model.likely_goals = self._infer_goals(lowered, has_question, style, rel)
        if model.interaction_count % cfg.values_every_turns == 0:
            model.values = self._infer_values(model)
        model.predicted_mood = self._predict_mood(emo, lowered, length)
        model.predicted_intent = self._predict_intent(lowered, has_question, style)
        return model

    def record_knowledge(self, speaker: str, domain: str, level: float) -> bool:
        """Record a caller-assessed knowledge level in [0, 1] for a known speaker.

        Returns False (and creates nothing) when the speaker has no model.
        """
        model = self.models.get(speaker)
        value = _finite(level)
        if model is None or value is None or not str(domain).strip():
            return False
        model.knowledge_domains[str(domain)] = min(1.0, max(0.0, value))
        return True

    def _infer_goals(
        self,
        lowered: str,
        has_question: bool,
        style: CommunicationStyle,
        rel: RelationshipState,
    ) -> List[str]:
        goals = [
            goal
            for goal, phrases in self.config.goal_keywords.items()
            if _mentions(lowered, phrases)
        ]
        if style.depth_preference > 0.6:
            goals.append("going_deep")
        if rel.warmth > 0.6 and has_question:
            goals.append("curiosity")
        return goals[: self.config.max_goals]

    def _infer_values(self, model: SpeakerModel) -> List[str]:
        cfg = self.config
        values: List[str] = []
        ranked = sorted(
            model.topic_frequencies.items(), key=lambda item: item[1], reverse=True
        )
        for topic, count in ranked[:5]:
            if count < cfg.min_topic_count_for_value:
                continue
            bucket = next(
                (
                    value
                    for value, members in cfg.value_topics.items()
                    if topic.lower() in members
                ),
                topic,
            )
            values.append(bucket)
        if model.relationship.warmth > 0.7:
            values.append("connection")
        if model.communication_style.depth_preference > 0.6:
            values.append("depth of conversation")
        return list(dict.fromkeys(values))[: cfg.max_values]

    def _predict_mood(self, emo: EmotionalProfile, lowered: str, length: int) -> str:
        trend = emo.recent_valence_trend
        if trend is not None and trend > 0.2:
            return "upbeat"
        if trend is not None and trend < -0.2:
            return "subdued"
        for mood, phrases in self.config.mood_keywords.items():
            if _mentions(lowered, phrases):
                return mood
        if length < self.config.short_message_chars:
            return "focused"
        return "engaged"

    def _predict_intent(
        self, lowered: str, has_question: bool, style: CommunicationStyle
    ) -> str:
        cfg = self.config
        if _mentions(lowered, cfg.continue_phrases):
            return "continue"
        if _mentions(lowered, cfg.perspective_phrases):
            return "hear_perspective"
        if has_question and style.question_ratio > 0.4:
            return "understand_something"
        if style.directness > 0.6:
            return "get_something_done"
        if style.depth_preference > 0.6:
            return "explore_together"
        return "connect"

    # ======== READING ========

    def get_model(self, speaker: str) -> Optional[SpeakerModel]:
        """The speaker's model, or None; a lookup never creates one."""
        return self.models.get(speaker)

    def get_context_for_prompt(self, speaker: str) -> str:
        """A neutral third-person block about the speaker, or "".

        Empty unless ``narrate_speaker_model`` is True and the speaker has a
        model.
        """
        if not self.narrate_speaker_model:
            return ""
        model = self.models.get(speaker)
        if model is None:
            return ""
        emo = model.emotional_profile
        style = model.communication_style
        rel = model.relationship
        topics = sorted(
            model.topic_frequencies.items(), key=lambda item: item[1], reverse=True
        )[:5]
        lines = [
            f"Speaker model for {model.speaker_name} "
            f"(heuristic estimates from {model.interaction_count} turns):",
            f"- Observed affect: valence {_fmt(emo.avg_valence)}, "
            f"arousal {_fmt(emo.avg_arousal)}",
            f"- Style indices: message length about "
            f"{style.avg_message_length:.0f} characters, question ratio "
            f"{style.question_ratio:.2f}, depth preference "
            f"{style.depth_preference:.2f}",
            f"- Relationship indices: warmth {rel.warmth:.2f}, depth {rel.depth:.2f}, "
            f"reciprocity {rel.reciprocity:.2f}, familiarity {rel.familiarity:.2f}",
            f"- Trust: {rel.trust:.2f} (configured by the trust settings, "
            "not inferred from the conversation)",
        ]
        if topics:
            lines.append(
                "- Frequent topics: " + ", ".join(f"{t} ({c})" for t, c in topics)
            )
        if model.likely_goals:
            lines.append("- Likely goals: " + ", ".join(model.likely_goals))
        if model.values:
            lines.append("- Recurring themes: " + ", ".join(model.values))
        lines.append(
            f"- Predicted mood: {model.predicted_mood}; "
            f"predicted intent: {model.predicted_intent}"
        )
        return "\n".join(lines)

    # ======== PERSISTENCE ========

    async def persist_model(self, speaker: str) -> None:
        """Write the speaker's model to the store; failures are logged."""
        model = self.models.get(speaker)
        if model is None:
            return
        try:
            await self.store.upsert_model(speaker, model.to_dict())
        except Exception as e:
            logger.warning(f"Speaker model persistence failed: {e}")

    async def forget_speaker(self, speaker: str) -> bool:
        """Delete the speaker's model from memory and the store.

        Returns True when a model existed in either place.
        """
        existed = self.models.pop(speaker, None) is not None
        try:
            existed = await self.store.delete_model(speaker) or existed
        except Exception as e:
            logger.warning(f"Speaker model deletion failed: {e}")
        return existed
