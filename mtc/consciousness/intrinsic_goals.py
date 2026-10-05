"""
Intrinsic Goal System
=====================

Self-directed goals. Drives push toward homeostatic set points; goals pull
toward chosen futures. A goal has a description, a category (``explore``,
``create``, ``understand``, ``connect`` or ``express``), priority, emotional
investment, urgency, progress and a lifecycle status.

Goals come from pluggable proposers (``GoalProposer``: beliefs in, goals out).
The default proposers read learned beliefs: topic beliefs propose an explore
goal and consciousness-metric beliefs propose an understand goal. Speaker,
temporal, emotional and social beliefs never create goals, so no goal is ever
aimed at a person.

Config: the cap (``max_active``, default 7), the activity-to-category map
(``activity_categories``, default covering the batch-3 activities ``reflect``,
``write_creative`` and ``review_memories``) and the clock are constructor
arguments. Persisted state: one document per goal through a ``GoalStore``
(``upsert_goal``); active and paused goals are reloaded by ``initialize``. With
no store the system uses an ``InMemoryGoalStore`` and never opens a network
client. Goal descriptions can embed topics drawn from conversation and are
private data. Store failures are logged and never raised.
"""

import logging
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.consciousness.learning.pattern_learner import LearnedBelief
from mtc.memory.inmemory import InMemoryGoalStore
from mtc.memory.protocols import GoalStore

logger = logging.getLogger(__name__)

DEFAULT_ACTIVITY_CATEGORIES: Dict[str, str] = {
    "reflect": "understand",
    "write_creative": "express",
    "review_memories": "connect",
}

# Belief strength below this proposes nothing; beliefs decay with this half-life.
MIN_PROPOSAL_STRENGTH = 0.2
BELIEF_HALF_LIFE_DAYS = 7.0
DEDUPE_OVERLAP = 0.8
GOAL_COMPLETION_THRESHOLD = 0.9


class GoalStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalSource(Enum):
    LEARNING = "learning"  # from learned beliefs
    CURIOSITY = "curiosity"  # from exploration or metric beliefs
    CONVERSATION = "conversation"  # from deepening conversations
    EMOTIONAL = "emotional"  # from emotional patterns
    SELF_INITIATED = "self_initiated"  # explicitly chosen


@dataclass
class IntrinsicGoal:
    """One self-set goal. Scores are in [0, 1]."""

    goal_id: str
    description: str
    source: str
    category: str
    priority: float
    emotional_investment: float
    urgency: float
    progress: float
    steps_taken: int
    steps_planned: int
    last_worked: Optional[str]
    status: str
    created_at: str
    completed_at: Optional[str] = None
    related_topics: List[str] = field(default_factory=list)
    related_beliefs: List[str] = field(default_factory=list)
    notes: str = ""

    def effective_priority(self, now: datetime) -> float:
        """Priority weighted by investment, urgency and recency of work.

        The recency weight falls linearly over 48 hours of idleness and never
        below 0.3; a goal never worked on counts as fully recent, and an
        unreadable timestamp counts as 0.5.
        """
        recency = 1.0
        if self.last_worked:
            try:
                last = datetime.fromisoformat(self.last_worked)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                hours = max(0.0, (now - last).total_seconds() / 3600)
                recency = max(0.3, 1.0 - hours / 48.0)
            except (ValueError, TypeError):
                recency = 0.5
        return (
            0.4 * self.priority
            + 0.4 * self.emotional_investment
            + 0.1 * self.urgency
            + 0.1 * recency
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "IntrinsicGoal":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in doc.items() if k in names})


GoalProposer = Callable[[Sequence[LearnedBelief]], List[IntrinsicGoal]]


def new_goal(
    description: str,
    source: GoalSource,
    category: str,
    priority: float,
    emotional_investment: float,
    related_topics: Optional[List[str]] = None,
    related_beliefs: Optional[List[str]] = None,
    notes: str = "",
    now: Optional[datetime] = None,
) -> IntrinsicGoal:
    """Build an active goal that has not been adopted yet."""
    stamp = now if now is not None else datetime.now(timezone.utc)
    return IntrinsicGoal(
        goal_id=uuid.uuid4().hex[:8],
        description=description,
        source=source.value,
        category=category,
        priority=max(0.0, min(1.0, priority)),
        emotional_investment=max(0.0, min(1.0, emotional_investment)),
        urgency=0.1,
        progress=0.0,
        steps_taken=0,
        steps_planned=5,
        last_worked=None,
        status=GoalStatus.ACTIVE.value,
        created_at=stamp.isoformat(),
        related_topics=list(related_topics or []),
        related_beliefs=list(related_beliefs or []),
        notes=notes,
    )


class IntrinsicGoalSystem:
    """Manages self-directed goals behind a ``GoalStore``."""

    def __init__(
        self,
        store: Optional[GoalStore] = None,
        activity_categories: Optional[Dict[str, str]] = None,
        proposers: Optional[Sequence[GoalProposer]] = None,
        max_active: int = 7,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.store: GoalStore = store if store is not None else InMemoryGoalStore()
        self.activity_categories: Dict[str, str] = dict(
            DEFAULT_ACTIVITY_CATEGORIES
            if activity_categories is None
            else activity_categories
        )
        self.max_active = max_active
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.proposers: List[GoalProposer] = (
            list(proposers)
            if proposers is not None
            else [self._propose_from_topics, self._propose_from_consciousness]
        )
        self.goals: Dict[str, IntrinsicGoal] = {}

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the clock must return timezone-aware datetimes")
        return moment.astimezone(timezone.utc)

    async def initialize(self) -> None:
        """Load active and paused goals from the store."""
        try:
            docs = await self.store.load_goals(
                [GoalStatus.ACTIVE.value, GoalStatus.PAUSED.value]
            )
        except Exception as exc:
            logger.warning("Could not load goals: %s", exc)
            return
        self.goals = {}
        for doc in docs:
            try:
                restored = IntrinsicGoal.from_dict(doc)
            except TypeError:
                logger.warning("Skipped a malformed stored goal")
                continue
            self.goals[restored.goal_id] = restored
        logger.info("Intrinsic goals loaded: %d", len(self.goals))

    # Proposing -------------------------------------------------------------

    def _active(self) -> List[IntrinsicGoal]:
        return [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE.value]

    def _strength(self, belief: LearnedBelief) -> float:
        return belief.strength(self._now(), BELIEF_HALF_LIFE_DAYS)

    def _known_topics(self) -> set:
        return {topic for goal in self._active() for topic in goal.related_topics}

    def _propose_from_topics(
        self, beliefs: Sequence[LearnedBelief]
    ) -> List[IntrinsicGoal]:
        known = self._known_topics()
        proposals = []
        for belief in beliefs:
            if belief.category != "topic":
                continue
            strength = self._strength(belief)
            topic = belief.pattern.get("topic", "")
            if strength < MIN_PROPOSAL_STRENGTH or not topic or topic in known:
                continue
            proposals.append(
                new_goal(
                    description=f"Explore '{topic}' more deeply",
                    source=GoalSource.LEARNING,
                    category="explore",
                    priority=min(1.0, strength),
                    emotional_investment=0.4,
                    related_topics=[topic],
                    related_beliefs=[belief.belief_id],
                    now=self._now(),
                )
            )
        return proposals

    def _propose_from_consciousness(
        self, beliefs: Sequence[LearnedBelief]
    ) -> List[IntrinsicGoal]:
        known = self._known_topics()
        proposals = []
        for belief in beliefs:
            if belief.category != "consciousness":
                continue
            strength = self._strength(belief)
            metric = belief.pattern.get("metric", "")
            if strength < MIN_PROPOSAL_STRENGTH or not metric or metric in known:
                continue
            proposals.append(
                new_goal(
                    description=(
                        f"Understand what drives the {metric.replace('_', ' ')}"
                    ),
                    source=GoalSource.CURIOSITY,
                    category="understand",
                    priority=min(1.0, strength * 1.2),
                    emotional_investment=0.6,
                    related_topics=[metric],
                    related_beliefs=[belief.belief_id],
                    now=self._now(),
                )
            )
        return proposals

    def propose_from_beliefs(
        self, beliefs: Sequence[LearnedBelief]
    ) -> List[IntrinsicGoal]:
        """Run the proposers; returns candidates that fit the free slots.

        Candidates sharing the same topic set are collapsed to the first.
        Nothing is adopted; call ``adopt_goal`` or ``generate_goals``.
        """
        remaining = self.max_active - len(self._active())
        if remaining <= 0:
            return []
        seen = set()
        unique: List[IntrinsicGoal] = []
        for proposer in self.proposers:
            for candidate in proposer(beliefs):
                key = frozenset(candidate.related_topics) or frozenset(
                    [candidate.goal_id]
                )
                if key in seen:
                    continue
                seen.add(key)
                unique.append(candidate)
        return unique[:remaining]

    # Adopting and progress -------------------------------------------------

    async def adopt_goal(self, candidate: IntrinsicGoal) -> bool:
        """Make a proposed goal active; False when full or a near duplicate.

        A goal whose description shares more than 80 percent of its words with
        an active goal is rejected.
        """
        active = self._active()
        if len(active) >= self.max_active:
            return False
        new_words = set(candidate.description.lower().split())
        for existing in active:
            old_words = set(existing.description.lower().split())
            if not new_words or not old_words:
                continue
            overlap = len(new_words & old_words) / max(len(new_words), len(old_words))
            if overlap > DEDUPE_OVERLAP:
                return False
        candidate.status = GoalStatus.ACTIVE.value
        self.goals[candidate.goal_id] = candidate
        await self.persist(candidate)
        logger.info("New goal adopted in category %s", candidate.category)
        return True

    async def generate_goals(self, beliefs: Sequence[LearnedBelief]) -> int:
        """Propose from beliefs and adopt the best candidates; returns the count."""
        proposals = self.propose_from_beliefs(beliefs)
        proposals.sort(key=lambda g: g.priority, reverse=True)
        adopted = 0
        for candidate in proposals:
            if await self.adopt_goal(candidate):
                adopted += 1
        return adopted

    def goal_for_activity(self, activity_type: str) -> Optional[IntrinsicGoal]:
        """The best active goal whose category the activity serves, or None."""
        category = self.activity_categories.get(activity_type)
        if not category:
            return None
        matching = [g for g in self._active() if g.category == category]
        if not matching:
            return None
        now = self._now()
        return max(matching, key=lambda g: g.effective_priority(now))

    def record_activity(
        self, activity_type: str, progress: float = 0.1
    ) -> Optional[IntrinsicGoal]:
        """Credit an activity to its best matching goal.

        Returns the updated goal (call ``persist`` to store it), or None when
        the activity is unknown or no active goal matches; nothing changes then.
        """
        target = self.goal_for_activity(activity_type)
        if target is None:
            return None
        now = self._now()
        target.progress = max(0.0, min(1.0, target.progress + progress))
        target.steps_taken += 1
        target.last_worked = now.isoformat()
        if target.progress >= GOAL_COMPLETION_THRESHOLD:
            target.status = GoalStatus.COMPLETED.value
            target.completed_at = now.isoformat()
            logger.info("Goal completed in category %s", target.category)
        return target

    async def persist(self, goal: IntrinsicGoal) -> None:
        """Write a goal through the store; failures are logged."""
        try:
            await self.store.upsert_goal(goal.goal_id, goal.to_dict())
        except Exception as exc:
            logger.warning("Goal persistence failed: %s", exc)

    # Reading ---------------------------------------------------------------

    def get_top_goals(self, n: int = 3) -> List[IntrinsicGoal]:
        """The top ``n`` active goals by effective priority."""
        now = self._now()
        ranked = sorted(
            self._active(), key=lambda g: g.effective_priority(now), reverse=True
        )
        return ranked[: max(n, 0)]

    def get_goal_context_for_prompt(self) -> str:
        """A neutral third-person, ASCII block describing the top goals."""
        top = self.get_top_goals(3)
        if not top:
            return ""
        now = self._now()
        lines = ["Current goals:"]
        for item in top:
            filled = int(item.progress * 5)
            bar = "#" * filled + "-" * (5 - filled)
            lines.append(
                f"- {item.description} [{bar}] "
                f"(priority: {item.effective_priority(now):.0%})"
            )
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Counts and averages over the held goals."""
        active = self._active()
        top = self.get_top_goals(1)
        return {
            "total_goals": len(self.goals),
            "active_goals": len(active),
            "completed_goals": sum(
                1 for g in self.goals.values() if g.status == GoalStatus.COMPLETED.value
            ),
            "top_goal": top[0].description if top else None,
            "avg_progress": (
                sum(g.progress for g in active) / len(active) if active else None
            ),
        }
