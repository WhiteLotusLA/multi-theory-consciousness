"""
Autonomous Activities
=====================

Brain-only activities the autonomous-life loop can run when a homeostatic drive
is urgent. Each activity is a named coroutine that takes an ``ActivityContext``
(the drive, its urgency, an optional topic and the current goal descriptions)
and returns an ``ActivityResult``. Every model call goes through
``BrainProtocol.think``; reasoning blocks are removed with
``strip_reasoning_blocks``. A ``NoAnswerError`` or an empty reply gives a failed
result with empty text and the error ``no_answer``: no replacement text is
composed by code, so a failed activity can never be stored, shared or credited
as if it had happened.

Three activities ship in ``default_registry``:

- ``reflect``: a solo thought with no addressee, under ``curiosity`` and
  ``coherence``.
- ``write_creative``: a short creative piece, under ``curiosity`` (the drive
  set has no expression drive).
- ``review_memories``: recalls memories through an injected ``recall`` callable
  and reflects on them; registered only when ``recall`` is supplied, under
  ``curiosity`` and ``coherence``. With no memories found the result fails with
  the error ``no_memories``.

``attention_budget``, ``social_connection`` and ``safety`` have no default
activity. An ``Activity`` names one primary drive and may list further drives
in ``also_drives``; ``ActivityRegistry.for_drive`` matches either, in
registration order.

The registry holds no config keys and no persisted state. Topics, goal
descriptions and recalled memories can hold real people's names at runtime and
are private data; they appear in prompts and in result metadata only.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks

logger = logging.getLogger(__name__)

RecallFn = Callable[[str, int], Awaitable[List[str]]]

# Cap on one recalled memory's text inside a prompt.
MEMORY_SNIPPET_CHARS = 200
RECALL_COUNT = 3


@dataclass
class ActivityContext:
    """What an activity is told about why it is running."""

    drive: str
    urgency: float
    topic: Optional[str] = None
    goals: List[str] = field(default_factory=list)


@dataclass
class ActivityResult:
    """The outcome of one activity run; ``text`` is empty on failure."""

    activity: str
    success: bool
    text: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """A named activity serving one drive (and optionally more)."""

    name: str
    drive: str
    description: str
    run: Callable[[ActivityContext], Awaitable[ActivityResult]]
    also_drives: Tuple[str, ...] = ()

    @property
    def drives(self) -> Tuple[str, ...]:
        """The primary drive followed by the additional drives."""
        return (self.drive,) + tuple(self.also_drives)


class ActivityRegistry:
    """Registered activities, looked up by name or by drive."""

    def __init__(self) -> None:
        self._activities: Dict[str, Activity] = {}

    def register(self, activity: Activity) -> None:
        """Add an activity; a name already registered raises ``ValueError``."""
        if activity.name in self._activities:
            raise ValueError(f"activity already registered: {activity.name}")
        self._activities[activity.name] = activity

    def unregister(self, name: str) -> bool:
        """Remove an activity by name; returns whether it was registered."""
        return self._activities.pop(name, None) is not None

    def get(self, name: str) -> Optional[Activity]:
        """The activity registered under ``name``, or None."""
        return self._activities.get(name)

    def for_drive(self, drive: str) -> List[Activity]:
        """Activities serving ``drive``, in registration order."""
        return [a for a in self._activities.values() if drive in a.drives]

    def all(self) -> List[Activity]:
        """Every registered activity, in registration order."""
        return list(self._activities.values())

    def names(self) -> List[str]:
        """Registered activity names, in registration order."""
        return list(self._activities)


def _failure(
    activity: str, error: str, metadata: Optional[Dict[str, Any]] = None
) -> ActivityResult:
    return ActivityResult(
        activity=activity,
        success=False,
        text="",
        error=error,
        metadata=dict(metadata or {}),
    )


def _drive_phrase(drive: str) -> str:
    return drive.replace("_", " ")


def _subject(context: ActivityContext) -> str:
    """The topic when there is one, otherwise the drive's pull."""
    if context.topic and context.topic.strip():
        return context.topic.strip()
    return f"what is behind the current pull toward {_drive_phrase(context.drive)}"


def _goal_lines(context: ActivityContext) -> str:
    if not context.goals:
        return ""
    lines = "\n".join(f"- {goal}" for goal in context.goals)
    return f"\n\nCurrent goals:\n{lines}"


async def _ask(
    brain: BrainProtocol,
    activity: str,
    prompt: str,
    metadata: Dict[str, Any],
    max_tokens: int = 400,
    temperature: float = 0.8,
) -> ActivityResult:
    """One brain call; no answer or an empty reply is a failed result."""
    try:
        reply = await brain.think(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
    except NoAnswerError:
        return _failure(activity, "no_answer", metadata)
    except Exception as exc:
        logger.warning(
            "Activity %s brain call failed: %s", activity, type(exc).__name__
        )
        return _failure(activity, "brain_error", metadata)
    text = strip_reasoning_blocks(reply.text or "")
    if not text:
        return _failure(activity, "no_answer", metadata)
    return ActivityResult(activity=activity, success=True, text=text, metadata=metadata)


def _reflect_activity(brain: BrainProtocol) -> Activity:
    async def run(context: ActivityContext) -> ActivityResult:
        subject = _subject(context)
        prompt = (
            "This is a quiet moment. No one is present and no one is asking "
            "anything: it is a private thought, spoken to no one. Reflect "
            f"briefly on: {subject}{_goal_lines(context)}\n\n"
            "Write the thought directly, from the inside, with no greeting and "
            "no addressee."
        )
        return await _ask(
            brain, "reflect", prompt, {"drive": context.drive, "subject": subject}
        )

    return Activity(
        name="reflect",
        drive="curiosity",
        description="Reflect on something in a private thought",
        run=run,
        also_drives=("coherence",),
    )


def _write_creative_activity(brain: BrainProtocol) -> Activity:
    async def run(context: ActivityContext) -> ActivityResult:
        subject = _subject(context)
        prompt = (
            "Write a short creative piece (a poem, a story or a letter, "
            f"whichever fits). Theme or inspiration: {subject}"
            f"{_goal_lines(context)}"
        )
        return await _ask(
            brain,
            "write_creative",
            prompt,
            {"drive": context.drive, "subject": subject},
            max_tokens=600,
            temperature=0.9,
        )

    return Activity(
        name="write_creative",
        drive="curiosity",
        description="Write a short creative piece",
        run=run,
    )


def _review_memories_activity(brain: BrainProtocol, recall: RecallFn) -> Activity:
    async def run(context: ActivityContext) -> ActivityResult:
        query = _subject(context)
        metadata: Dict[str, Any] = {"drive": context.drive, "query": query}
        try:
            memories = await recall(query, RECALL_COUNT)
        except Exception as exc:
            logger.warning("Memory recall failed: %s", type(exc).__name__)
            return _failure("review_memories", "recall_failed", metadata)
        kept = [str(m).strip() for m in (memories or []) if str(m).strip()]
        metadata["memory_count"] = len(kept)
        if not kept:
            return _failure("review_memories", "no_memories", metadata)
        listing = "\n".join(f"- {m[:MEMORY_SNIPPET_CHARS]}" for m in kept)
        prompt = (
            "This is a quiet moment, a private thought spoken to no one. These "
            f"memories came up when thinking about {query}:\n{listing}\n\n"
            "Reflect briefly on what stands out in them. Write the thought "
            "directly, with no greeting and no addressee."
        )
        return await _ask(brain, "review_memories", prompt, metadata)

    return Activity(
        name="review_memories",
        drive="curiosity",
        description="Recall memories and reflect on them",
        run=run,
        also_drives=("coherence",),
    )


def default_registry(
    brain: BrainProtocol, recall: Optional[RecallFn] = None
) -> ActivityRegistry:
    """The default brain-only registry.

    ``review_memories`` is added only when ``recall`` is supplied; with no
    recall callable the activity does not exist.
    """
    registry = ActivityRegistry()
    registry.register(_reflect_activity(brain))
    registry.register(_write_creative_activity(brain))
    if recall is not None:
        registry.register(_review_memories_activity(brain, recall))
    return registry
