"""
Pattern Learner
===============

Detects recurring patterns in the experience window and keeps them as learned
beliefs, for example "conversations with a given speaker tend to deepen",
"evening interactions are more playful" or "long conversations correlate with
deepening". Beliefs carry a confidence that grows with reinforcement and
decays with a half-life (``half_life_days``, default 7) when not reinforced.
New beliefs start weak.

Six pattern families are scanned: speaker, topic, temporal, emotional, depth
and consciousness-metric patterns. Consciousness metrics (ignition, loop
quality, epistemic depth, binding, protoself stability) belong to the system's
own cycle and are never attributed to a speaker: the consciousness beliefs are
global, each names its metric, and stored beliefs of the retired per-speaker
form are skipped on load. Evidence that is unknown (None) is skipped:
an experience without an outcome or valence shift never counts towards a group
and never enters an average. ``evidence_count`` counts reinforcing learning
passes, not distinct experiences, because a periodic pass re-reads an
overlapping window.

Config keys: ``LEARNING_RATE`` (default reinforcement rate when a pass is not
given one). Persisted state: one document per belief through a
``LearningStore`` (``upsert_belief``); beliefs pruned in memory are not deleted
from the store and are pruned again on the next pass if still weak. Belief ids
and descriptions embed speaker names at runtime and are private data.
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.consciousness.learning.experience_tracker import Experience
from mtc.core.config import get_settings
from mtc.memory.protocols import LearningStore

logger = logging.getLogger(__name__)

POSITIVE_OUTCOMES = ("positive", "deepening", "playful")
# Ids of the retired per-speaker consciousness beliefs, skipped on load.
RETIRED_SPEAKER_CREDIT_PREFIX = "consciousness_speaker_"


@dataclass
class LearnedBelief:
    """A learned pattern with a confidence in [0, 1].

    Categories: ``speaker``, ``topic``, ``temporal``, ``emotional``, ``social``
    and ``consciousness``. ``behavioral_effect`` maps an effect name (for
    example ``warmth`` or ``curiosity``) to a signed amount the behavioral
    adapter may apply.
    """

    belief_id: str
    category: str
    description: str
    pattern: Dict[str, Any]
    confidence: float
    evidence_count: int
    counter_evidence: int
    first_observed: str
    last_reinforced: str
    last_contradicted: Optional[str] = None
    behavioral_effect: Dict[str, float] = field(default_factory=dict)

    def strength(self, now: datetime, half_life_days: float = 7.0) -> float:
        """Confidence weighted by recency (exponential decay since last reinforcement)."""
        last = datetime.fromisoformat(self.last_reinforced)
        age_hours = max(0.0, (now - last).total_seconds() / 3600.0)
        recency = math.exp(-age_hours / (half_life_days * 24.0) * math.log(2))
        return self.confidence * recency

    def reinforce(self, now: datetime, amount: float = 0.05) -> None:
        """Strengthen with new evidence (a saturating increase)."""
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + amount * (1 - self.confidence))
        self.last_reinforced = now.isoformat()

    def contradict(self, now: datetime, amount: float = 0.1) -> None:
        """Weaken with counter-evidence."""
        self.counter_evidence += 1
        self.confidence = max(0.0, self.confidence - amount * self.confidence)
        self.last_contradicted = now.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "LearnedBelief":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in doc.items() if k in names})


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class PatternLearner:
    """Detects patterns in experiences and maintains learned beliefs."""

    def __init__(
        self,
        store: LearningStore,
        min_evidence: int = 3,
        half_life_days: float = 7.0,
        max_beliefs: int = 500,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._store = store
        self.min_evidence = min_evidence
        self.half_life_days = half_life_days
        self.max_beliefs = max_beliefs
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.beliefs: Dict[str, LearnedBelief] = {}

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the clock must return timezone-aware datetimes")
        return moment.astimezone(timezone.utc)

    def strength(self, belief: LearnedBelief) -> float:
        """Current recency-weighted strength of a belief."""
        return belief.strength(self._now(), self.half_life_days)

    def attach_store(self, store: LearningStore) -> None:
        """Replace the store; call ``initialize`` afterwards to reload beliefs."""
        self._store = store

    async def initialize(self) -> None:
        """Replace the in-memory beliefs with those held by the store.

        A stored belief whose id starts with ``consciousness_speaker_`` (the
        retired family that credited a speaker with the system's own metrics)
        is skipped, so an existing store stops crediting anyone.
        """
        try:
            docs = await self._store.load_beliefs(self.max_beliefs)
        except Exception as exc:
            logger.warning("Could not load beliefs: %s", exc)
            return
        self.beliefs = {}
        dropped = 0
        for doc in docs:
            try:
                belief = LearnedBelief.from_dict(doc)
            except TypeError:
                logger.warning("Skipped a malformed stored belief")
                continue
            if belief.belief_id.startswith(RETIRED_SPEAKER_CREDIT_PREFIX):
                dropped += 1
                continue
            self.beliefs[belief.belief_id] = belief
        if dropped:
            logger.debug(
                "Skipped %d stored consciousness beliefs that credited a speaker",
                dropped,
            )
        logger.info("PatternLearner loaded %d beliefs", len(self.beliefs))

    async def learn_from_experiences(
        self, experiences: Sequence[Experience], rate: Optional[float] = None
    ) -> List[LearnedBelief]:
        """Scan experiences for patterns and update beliefs.

        ``rate`` is the reinforcement rate (an affect-modulated rate when the
        caller has one); the default is ``Settings.LEARNING_RATE``. Returns the
        beliefs created or reinforced. Call periodically or during
        consolidation.
        """
        if len(experiences) < self.min_evidence:
            return []
        step = float(rate) if rate is not None else float(get_settings().LEARNING_RATE)
        changed: List[LearnedBelief] = []
        changed.extend(self._learn_speaker_patterns(experiences, step))
        changed.extend(self._learn_topic_patterns(experiences, step))
        changed.extend(self._learn_temporal_patterns(experiences, step))
        changed.extend(self._learn_emotional_patterns(experiences, step))
        changed.extend(self._learn_depth_patterns(experiences, step))
        changed.extend(self._learn_consciousness_patterns(experiences, step))
        self.prune()
        await self._persist_beliefs(changed)
        return changed

    # Pattern families ----------------------------------------------------

    def _learn_speaker_patterns(
        self, experiences: Sequence[Experience], step: float
    ) -> List[LearnedBelief]:
        changed: List[LearnedBelief] = []
        by_speaker: Dict[str, List[Experience]] = defaultdict(list)
        for item in experiences:
            by_speaker[item.speaker].append(item)
        for speaker, group in by_speaker.items():
            with_outcome = [e for e in group if e.outcome is not None]
            if len(with_outcome) >= self.min_evidence:
                outcome, count = Counter(e.outcome for e in with_outcome).most_common(
                    1
                )[0]
                ratio = count / len(with_outcome)
                if ratio > 0.4:
                    belief = self.make_belief(
                        f"speaker_{speaker}_{outcome}",
                        "speaker",
                        f"Conversations with {speaker} tend to be {outcome} "
                        f"({ratio:.0%} of the time)",
                        {
                            "speaker": speaker,
                            "dominant_outcome": outcome,
                            "ratio": ratio,
                        },
                    )
                    belief.reinforce(self._now(), step * ratio)
                    changed.append(belief)
            shifts = [e.valence_shift for e in group if e.valence_shift is not None]
            if len(shifts) >= self.min_evidence:
                avg = _mean(shifts)
                if abs(avg) > 0.05:
                    direction = "positive" if avg > 0 else "negative"
                    belief = self.make_belief(
                        f"speaker_{speaker}_valence_{direction}",
                        "speaker",
                        f"Conversations with {speaker} tend to shift emotional "
                        f"valence {direction}ly ({avg:+.2f})",
                        {"speaker": speaker, "avg_valence_shift": avg},
                        {"warmth": 0.1 if avg > 0 else -0.05},
                    )
                    belief.reinforce(self._now(), step)
                    changed.append(belief)
        return changed

    def _learn_topic_patterns(
        self, experiences: Sequence[Experience], step: float
    ) -> List[LearnedBelief]:
        changed: List[LearnedBelief] = []
        by_topic: Dict[str, List[Experience]] = defaultdict(list)
        for item in experiences:
            if item.outcome is None:
                continue
            if item.topic:
                by_topic[item.topic].append(item)
            for extra in item.topics_discussed:
                if extra and extra != item.topic:
                    by_topic[extra].append(item)
        for topic, group in by_topic.items():
            if len(group) < self.min_evidence or topic == "general":
                continue
            outcomes = Counter(e.outcome for e in group)
            total = len(group)
            deepening = outcomes.get("deepening", 0) / total
            if deepening > 0.3:
                belief = self.make_belief(
                    f"topic_{topic}_deepens",
                    "topic",
                    f"Discussing '{topic}' tends to deepen conversations "
                    f"({deepening:.0%} of the time)",
                    {"topic": topic, "deepening_ratio": deepening},
                    {"curiosity": 0.1, "verbosity": 0.05},
                )
                belief.reinforce(self._now(), step * deepening)
                changed.append(belief)
            playful = outcomes.get("playful", 0) / total
            if playful > 0.3:
                belief = self.make_belief(
                    f"topic_{topic}_playful",
                    "topic",
                    f"Discussing '{topic}' tends to be playful "
                    f"({playful:.0%} of the time)",
                    {"topic": topic, "playful_ratio": playful},
                    {"warmth": 0.05, "playfulness": 0.1},
                )
                belief.reinforce(self._now(), step * playful)
                changed.append(belief)
        return changed

    def _learn_temporal_patterns(
        self, experiences: Sequence[Experience], step: float
    ) -> List[LearnedBelief]:
        changed: List[LearnedBelief] = []
        by_time: Dict[str, List[Experience]] = defaultdict(list)
        for item in experiences:
            if item.time_of_day and item.outcome is not None:
                by_time[item.time_of_day].append(item)
        for period, group in by_time.items():
            shifts = [e.valence_shift for e in group if e.valence_shift is not None]
            if len(group) < self.min_evidence or not shifts:
                continue
            avg = _mean(shifts)
            dominant = Counter(e.outcome for e in group).most_common(1)[0][0]
            belief = self.make_belief(
                f"temporal_{period}",
                "temporal",
                f"{period.title()} interactions tend to be {dominant} "
                f"(avg valence shift: {avg:+.2f})",
                {
                    "time_of_day": period,
                    "dominant_outcome": dominant,
                    "avg_valence_shift": avg,
                },
            )
            belief.reinforce(self._now(), step)
            changed.append(belief)
        return changed

    def _learn_emotional_patterns(
        self, experiences: Sequence[Experience], step: float
    ) -> List[LearnedBelief]:
        changed: List[LearnedBelief] = []
        measured = [e for e in experiences if e.arousal_shift is not None and e.outcome]
        groups = [
            ("high_arousal", [e for e in measured if e.arousal_shift > 0.2]),
            ("calming", [e for e in measured if e.arousal_shift < -0.2]),
        ]
        for label, group in groups:
            if len(group) < self.min_evidence:
                continue
            outcomes = Counter(e.outcome for e in group)
            positive = outcomes.get("positive", 0) + outcomes.get("deepening", 0)
            ratio = positive / len(group)
            belief = self.make_belief(
                f"emotional_{label}_positive",
                "emotional",
                f"When arousal {label.replace('_', ' ')}, interactions tend to be "
                f"positive ({ratio:.0%} of the time)",
                {"arousal_pattern": label, "positive_ratio": ratio},
            )
            belief.reinforce(self._now(), step * ratio)
            changed.append(belief)
        return changed

    def _learn_depth_patterns(
        self, experiences: Sequence[Experience], step: float
    ) -> List[LearnedBelief]:
        deep = [e for e in experiences if e.conversation_depth > 5 and e.outcome]
        if len(deep) < self.min_evidence:
            return []
        ratio = Counter(e.outcome for e in deep).get("deepening", 0) / len(deep)
        scores = [
            e.consciousness_score for e in deep if e.consciousness_score is not None
        ]
        avg_score = _mean(scores)
        text = f"Longer conversations (>5 turns) deepen {ratio:.0%} of the time"
        if avg_score is not None:
            text += f" (avg consciousness: {avg_score:.2f})"
        belief = self.make_belief(
            "depth_long_conversations",
            "social",
            text,
            {
                "min_depth": 5,
                "deepening_ratio": ratio,
                "avg_consciousness": avg_score,
            },
            {"patience": 0.1, "curiosity": 0.05},
        )
        belief.reinforce(self._now(), step)
        return [belief]

    def _learn_consciousness_patterns(
        self, experiences: Sequence[Experience], step: float
    ) -> List[LearnedBelief]:
        """Correlate consciousness metrics with outcomes.

        Only experiences that carry real metric data (a positive ignition count
        or loop quality) are considered; a metric that is None never counts.
        The metrics are properties of the system's own cycle, so every belief
        of this family is global: it names its ``metric`` and never a speaker.
        """
        changed: List[LearnedBelief] = []
        rich = [
            e
            for e in experiences
            if e.outcome
            and (
                (e.gwt_ignition_count is not None and e.gwt_ignition_count > 0)
                or (
                    e.beautiful_loop_quality is not None
                    and e.beautiful_loop_quality > 0
                )
            )
        ]
        if len(rich) < self.min_evidence:
            return changed

        high_ignition = [
            e
            for e in rich
            if e.gwt_ignition_count is not None and e.gwt_ignition_count >= 2
        ]
        if len(high_ignition) >= self.min_evidence:
            ratio = sum(e.outcome in POSITIVE_OUTCOMES for e in high_ignition) / len(
                high_ignition
            )
            belief = self.make_belief(
                "consciousness_ignition_quality",
                "consciousness",
                f"When workspace ignition is high (>=2 events), {ratio:.0%} of "
                "interactions are positive or deepening",
                {
                    "metric": "gwt_ignition_count",
                    "threshold": 2,
                    "positive_ratio": ratio,
                    "sample_size": len(high_ignition),
                },
                {"curiosity": 0.05},
            )
            belief.reinforce(self._now(), step * ratio)
            changed.append(belief)

        deep_epistemic = [
            e for e in rich if e.epistemic_depth is not None and e.epistemic_depth >= 2
        ]
        if len(deep_epistemic) >= self.min_evidence:
            ratio = sum(e.outcome == "deepening" for e in deep_epistemic) / len(
                deep_epistemic
            )
            if ratio > 0.2:
                belief = self.make_belief(
                    "consciousness_epistemic_deepening",
                    "consciousness",
                    f"Higher epistemic depth (>=2) correlates with conversation "
                    f"deepening ({ratio:.0%} of the time)",
                    {
                        "metric": "epistemic_depth",
                        "threshold": 2,
                        "deepening_ratio": ratio,
                    },
                )
                belief.reinforce(self._now(), step * ratio)
                changed.append(belief)

        bound = [e for e in rich if e.binding_quality is not None]
        avg_binding = _mean([e.binding_quality for e in bound])
        if avg_binding is not None and avg_binding > 0:
            high_binding = [e for e in bound if e.binding_quality > avg_binding]
            if len(high_binding) >= self.min_evidence:
                ratio = sum(
                    e.outcome in ("positive", "deepening") for e in high_binding
                ) / len(high_binding)
                belief = self.make_belief(
                    "consciousness_binding_engagement",
                    "consciousness",
                    f"Above-average binding quality ({avg_binding:.2f}) correlates "
                    f"with positive engagement ({ratio:.0%})",
                    {
                        "metric": "binding_quality",
                        "avg": avg_binding,
                        "positive_ratio": ratio,
                    },
                )
                belief.reinforce(self._now(), step * ratio)
                changed.append(belief)

        grounded = [
            e
            for e in rich
            if e.protoself_stability is not None and e.protoself_stability > 0.5
        ]
        if len(grounded) >= self.min_evidence:
            ratio = sum(e.outcome in POSITIVE_OUTCOMES for e in grounded) / len(
                grounded
            )
            belief = self.make_belief(
                "consciousness_somatic_grounding",
                "consciousness",
                f"When somatically grounded (protoself stability > 0.5), "
                f"{ratio:.0%} of interactions are positive",
                {
                    "metric": "protoself_stability",
                    "threshold": 0.5,
                    "positive_ratio": ratio,
                    "sample_size": len(grounded),
                },
                {"warmth": 0.05, "patience": 0.05},
            )
            belief.reinforce(self._now(), step * ratio)
            changed.append(belief)

        return changed

    # Belief bookkeeping --------------------------------------------------

    def make_belief(
        self,
        belief_id: str,
        category: str,
        description: str,
        pattern: Dict[str, Any],
        behavioral_effect: Optional[Dict[str, float]] = None,
    ) -> LearnedBelief:
        """Return the belief under ``belief_id``, creating a weak one if new.

        An existing belief takes the latest description, pattern and effect.
        """
        existing = self.beliefs.get(belief_id)
        if existing is not None:
            existing.description = description
            existing.pattern = pattern
            if behavioral_effect:
                existing.behavioral_effect = behavioral_effect
            return existing
        now = self._now().isoformat()
        belief = LearnedBelief(
            belief_id=belief_id,
            category=category,
            description=description,
            pattern=pattern,
            confidence=0.1,
            evidence_count=0,
            counter_evidence=0,
            first_observed=now,
            last_reinforced=now,
            behavioral_effect=dict(behavioral_effect or {}),
        )
        self.beliefs[belief_id] = belief
        logger.info("New belief in category %s", category)
        return belief

    def contradict(self, belief_id: str, amount: float = 0.1) -> None:
        """Record counter-evidence against a belief; unknown ids are ignored."""
        belief = self.beliefs.get(belief_id)
        if belief is not None:
            belief.contradict(self._now(), amount)

    def prune(self) -> None:
        """Drop decayed weak beliefs, then the weakest beyond ``max_beliefs``."""
        for belief_id in [
            bid
            for bid, belief in self.beliefs.items()
            if self.strength(belief) < 0.01
            and belief.evidence_count < self.min_evidence
        ]:
            del self.beliefs[belief_id]
        excess = len(self.beliefs) - self.max_beliefs
        if excess > 0:
            ranked = sorted(
                self.beliefs.items(), key=lambda item: self.strength(item[1])
            )
            for belief_id, _ in ranked[:excess]:
                del self.beliefs[belief_id]

    async def _persist_beliefs(self, changed: Sequence[LearnedBelief]) -> None:
        for belief in changed:
            if belief.belief_id not in self.beliefs:
                continue
            try:
                await self._store.upsert_belief(belief.belief_id, belief.to_dict())
            except Exception as exc:
                logger.warning("Failed to persist belief %s: %s", belief.belief_id, exc)

    # Queries ---------------------------------------------------------------

    def get_active_beliefs(self, min_strength: float = 0.1) -> List[LearnedBelief]:
        """Beliefs whose current strength is at least ``min_strength``."""
        return [b for b in self.beliefs.values() if self.strength(b) >= min_strength]

    def get_beliefs_for_context(
        self,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        time_of_day: Optional[str] = None,
    ) -> List[LearnedBelief]:
        """Active beliefs relevant to a context, strongest first.

        Emotional and social beliefs are general and always apply. Speaker
        matching compares the stored pattern's speaker value with the supplied
        value; no name is treated specially.
        """
        relevant = []
        for belief in self.get_active_beliefs():
            pattern = belief.pattern
            if speaker and pattern.get("speaker") == speaker:
                relevant.append(belief)
            elif topic and pattern.get("topic") == topic:
                relevant.append(belief)
            elif time_of_day and pattern.get("time_of_day") == time_of_day:
                relevant.append(belief)
            elif belief.category in ("emotional", "social"):
                relevant.append(belief)
        return sorted(relevant, key=self.strength, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Summary of the belief set."""
        if not self.beliefs:
            return {"total_beliefs": 0, "active_beliefs": 0, "categories": {}}
        strongest = max(self.beliefs.values(), key=self.strength)
        return {
            "total_beliefs": len(self.beliefs),
            "active_beliefs": len(self.get_active_beliefs()),
            "categories": dict(Counter(b.category for b in self.beliefs.values())),
            "strongest": strongest.description,
            "total_evidence": sum(b.evidence_count for b in self.beliefs.values()),
        }
