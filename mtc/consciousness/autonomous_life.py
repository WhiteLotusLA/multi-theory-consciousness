"""
Autonomous Life
===============

A lean background loop that lets the system act on its own drives between
conversations. Each ``step()`` reads the homeostatic drive state, picks an
activity for the most urgent drive, runs it through the brain, turns a
successful result into an internal workspace candidate, credits the drive and
the matching goal, optionally asks the brain whether to share the result, and
now and then runs thought consolidation. ``tick()`` is an alias of ``step()``.

Selection: ``DriveBasedSelector`` picks the first registered activity of the
chosen drive; ``BrainActivitySelector`` lists the options in a neutral prompt
and accepts only one exact option name found among the reply's words, otherwise
it falls back to the drive-based choice and records the selection source as
``drive_fallback``. Drives are considered most urgent first; only drives above
``urgency_threshold`` that have a registered activity count, so a drive with no
activity never blocks the next one. With probability ``exploration_probability``
the loop instead picks any registered activity from the seeded ``rng`` (source
``exploration``).

A failed attempt (no answer, an empty reply, an error) is counted under its
error in ``get_status()`` and nothing else happens: no candidate, no share, no
drive credit and no goal progress. Only successful results enter ``history``.

The loop never runs during a turn. ``notify_turn_started`` and
``notify_turn_finished`` bracket a turn (nesting is counted); while one is open
``step()`` does nothing and returns None, and a share decision is skipped when a
turn opens in the middle of a step. Night rest is the injected ``is_rest_time``
callable (the sleep scheduler's ``is_resting`` in the full system).

Tempo: ``next_interval_s`` scales with homeostatic free energy. Above
``high_free_energy`` the interval falls in the lowest third of the span between
``min_interval_s`` and ``max_interval_s``, above ``medium_free_energy`` in the
middle third, otherwise in the top third; with no free-energy reading yet it is
``max_interval_s`` (a documented prior, not a measurement).

Config (``AutonomousLifeConfig``): ``min_interval_s``, ``max_interval_s``,
``urgency_threshold``, ``exploration_probability``, ``consolidate_every`` (steps
between consolidation calls; 0 disables), ``history_size``,
``high_free_energy``, ``medium_free_energy`` and ``activity_types`` (activity
name to ``{"type", "intensity"}`` passed to ``update_homeostasis``; the default
maps ``reflect`` to ``sensemaking``, ``write_creative`` to ``exploring`` and
``review_memories`` to ``understanding``; an activity without a mapping credits
no drive).

Collaborators are injected: ``drives_source`` exposes ``homeostatic_drives``
(``get_drive_state``, ``get_free_energy``) and ``update_homeostasis``;
``experience_sink`` receives each candidate (without one the loop keeps a
bounded list in ``recent_candidates``); ``goals`` is an ``IntrinsicGoalSystem``;
``share`` receives a result the brain chose to share; ``consolidate_thoughts``
returns a count. ``start()`` and ``stop()`` own one task; tests call ``step()``
directly.

Persisted state (``to_state_dict``): step and completion counts, the last
activity and its time (aware UTC), the last free energy, the step of the last
consolidation, failure counts by error and the bounded success history. Result
summaries and goal text can hold real people's names at runtime and are private
data.
"""

import asyncio
import logging
import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks
from mtc.consciousness.activities import (
    Activity,
    ActivityContext,
    ActivityRegistry,
    ActivityResult,
    default_registry,
)
from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.intrinsic_goals import IntrinsicGoalSystem

logger = logging.getLogger(__name__)

SOURCE_DRIVE = "drive"
SOURCE_BRAIN = "brain"
SOURCE_DRIVE_FALLBACK = "drive_fallback"
SOURCE_EXPLORATION = "exploration"
SOURCE_SINGLE = "single_option"

SHARE_KEEP = "keep"
SHARE_SHARE = "share"
SHARE_DISCUSS = "discuss"
_SHARE_WORDS = (SHARE_KEEP, SHARE_SHARE, SHARE_DISCUSS)

_WORD = re.compile(r"[A-Za-z0-9_]+")
SUMMARY_CHARS = 100
GOAL_PROGRESS = 0.1
TOP_GOALS = 3


def _default_activity_types() -> Dict[str, Dict[str, Any]]:
    return {
        "reflect": {"type": "sensemaking", "intensity": 0.6},
        "write_creative": {"type": "exploring", "intensity": 0.6},
        "review_memories": {"type": "understanding", "intensity": 0.5},
    }


@dataclass
class AutonomousLifeConfig:
    """Pacing, thresholds and the activity-to-drive-vocabulary map."""

    min_interval_s: float = 60.0
    max_interval_s: float = 1800.0
    urgency_threshold: float = 0.3
    exploration_probability: float = 0.1
    consolidate_every: int = 10
    history_size: int = 100
    high_free_energy: float = 2.0
    medium_free_energy: float = 1.0
    activity_types: Dict[str, Dict[str, Any]] = field(
        default_factory=_default_activity_types
    )

    def __post_init__(self) -> None:
        if self.min_interval_s < 0 or self.max_interval_s < self.min_interval_s:
            raise ValueError("intervals need 0 <= min_interval_s <= max_interval_s")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be within [0, 1]")
        if self.consolidate_every < 0:
            raise ValueError("consolidate_every must be >= 0")
        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")
        if self.medium_free_energy > self.high_free_energy:
            raise ValueError("medium_free_energy must not exceed high_free_energy")


@dataclass
class Selection:
    """A chosen activity and how it was chosen."""

    activity: Activity
    source: str


@runtime_checkable
class ActivitySelector(Protocol):
    """Chooses one activity from the options for a drive."""

    async def select(
        self, context: ActivityContext, options: List[Activity]
    ) -> Selection:
        """Return a selection among a non-empty ``options`` list."""
        ...


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def most_urgent_drive(
    drive_state: Dict[str, Dict[str, Any]],
    registry: ActivityRegistry,
    threshold: float,
) -> Optional[Tuple[str, float]]:
    """The most urgent drive above ``threshold`` that has a registered activity.

    ``drive_state`` is the ``get_drive_state`` mapping (drive name to a dict
    with an ``urgency`` value). Drives with a missing or non-finite urgency are
    skipped; ties break by name. Returns ``(drive, urgency)`` or None.
    """
    ranked: List[Tuple[float, str]] = []
    for name, state in (drive_state or {}).items():
        urgency = _finite((state or {}).get("urgency"))
        if urgency is None or urgency <= threshold:
            continue
        if not registry.for_drive(name):
            continue
        ranked.append((urgency, name))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    urgency, name = ranked[0]
    return name, urgency


class DriveBasedSelector:
    """Picks the first registered activity of the drive."""

    async def select(
        self, context: ActivityContext, options: List[Activity]
    ) -> Selection:
        return Selection(options[0], SOURCE_DRIVE)


class BrainActivitySelector:
    """Asks the brain to name one option; anything else falls back.

    The reply is stripped of reasoning blocks and split into words; it is
    accepted only when exactly one distinct option name appears among them.
    A missing, empty, unparseable or ambiguous answer returns the first option
    with the source ``drive_fallback``.
    """

    def __init__(self, brain: BrainProtocol) -> None:
        self.brain = brain
        self._fallback = DriveBasedSelector()

    async def select(
        self, context: ActivityContext, options: List[Activity]
    ) -> Selection:
        if len(options) == 1:
            return Selection(options[0], SOURCE_SINGLE)
        listing = "\n".join(f"- {a.name}: {a.description}" for a in options)
        goals = ""
        if context.goals:
            goals = "\nCurrent goals: " + "; ".join(context.goals)
        prompt = (
            f"The pull toward {context.drive.replace('_', ' ')} is currently "
            f"strongest (urgency {context.urgency:.2f}).{goals}\n\n"
            f"Options:\n{listing}\n\n"
            "Reply with exactly one option name."
        )
        try:
            reply = await self.brain.think(prompt, max_tokens=64, temperature=0.3)
            text = strip_reasoning_blocks(reply.text or "")
        except NoAnswerError:
            text = ""
        except Exception as exc:
            logger.warning("Activity choice failed: %s", type(exc).__name__)
            text = ""
        by_name = {a.name.lower(): a for a in options}
        found = {w.lower() for w in _WORD.findall(text) if w.lower() in by_name}
        if len(found) == 1:
            return Selection(by_name[next(iter(found))], SOURCE_BRAIN)
        fallback = await self._fallback.select(context, options)
        return Selection(fallback.activity, SOURCE_DRIVE_FALLBACK)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AutonomousLife:
    """The lean autonomous-life loop; see the module docstring."""

    def __init__(
        self,
        brain: BrainProtocol,
        drives_source: Any,
        registry: Optional[ActivityRegistry] = None,
        selector: Optional[ActivitySelector] = None,
        experience_sink: Optional[
            Callable[[WorkspaceCandidate], Awaitable[Any]]
        ] = None,
        goals: Optional[IntrinsicGoalSystem] = None,
        share: Optional[Callable[[ActivityResult], Awaitable[None]]] = None,
        is_rest_time: Optional[Callable[[], bool]] = None,
        consolidate_thoughts: Optional[Callable[[], Awaitable[int]]] = None,
        config: Optional[AutonomousLifeConfig] = None,
        rng: Optional[random.Random] = None,
        clock: Optional[Callable[[], datetime]] = None,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> None:
        self.brain = brain
        self.drives_source = drives_source
        self.registry = registry if registry is not None else default_registry(brain)
        self.selector: ActivitySelector = selector or DriveBasedSelector()
        self.experience_sink = experience_sink
        self.goals = goals
        self.share = share
        self.is_rest_time = is_rest_time
        self.consolidate_thoughts = consolidate_thoughts
        self.config = config or AutonomousLifeConfig()
        self.rng = rng or random.Random()
        self._clock = clock or _utc_now
        self._sleep = sleep or asyncio.sleep

        self.steps_run = 0
        self.completed = 0
        self.failures: Dict[str, int] = {}
        self.last_activity: Optional[str] = None
        self.last_activity_time: Optional[datetime] = None
        self.last_free_energy: Optional[float] = None
        self.last_consolidation_step: Optional[int] = None
        self.last_consolidated: Optional[int] = None
        self.last_skip: Optional[str] = None
        self.sink_errors = 0
        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.config.history_size)
        self.recent_candidates: Deque[WorkspaceCandidate] = deque(
            maxlen=self.config.history_size
        )

        self._turns_open = 0
        self._stepping = False
        self._running = False
        self._task: Optional["asyncio.Task[None]"] = None

    # -- turns -----------------------------------------------------------------

    def notify_turn_started(self) -> None:
        """A turn began; steps do nothing until it finishes."""
        self._turns_open += 1

    def notify_turn_finished(self) -> None:
        """A turn ended (never goes below zero)."""
        self._turns_open = max(0, self._turns_open - 1)

    # -- pacing ----------------------------------------------------------------

    def next_interval_s(self, free_energy: Optional[float] = None) -> float:
        """Seconds to wait after a step, scaled by free energy.

        Higher free energy gives a shorter wait, inside the configured bounds.
        With no reading (argument or last seen) the maximum is returned.
        """
        cfg = self.config
        reading = _finite(free_energy)
        if reading is None:
            reading = self.last_free_energy
        if reading is None:
            return cfg.max_interval_s
        third = (cfg.max_interval_s - cfg.min_interval_s) / 3.0
        if reading > cfg.high_free_energy:
            low, high = cfg.min_interval_s, cfg.min_interval_s + third
        elif reading > cfg.medium_free_energy:
            low = cfg.min_interval_s + third
            high = cfg.min_interval_s + 2.0 * third
        else:
            low, high = cfg.min_interval_s + 2.0 * third, cfg.max_interval_s
        return self.rng.uniform(low, high)

    # -- the step --------------------------------------------------------------

    async def tick(self) -> Optional[ActivityResult]:
        """Alias of ``step``."""
        return await self.step()

    async def step(self) -> Optional[ActivityResult]:
        """Run at most one activity; None when nothing ran.

        Returns the ``ActivityResult`` of an attempt (successful or failed) and
        None when the step was skipped (turn in progress, rest, no drives, no
        urgent drive with an activity, or another step already running).
        """
        if self._stepping:
            return self._skip("busy")
        if self._turns_open > 0:
            return self._skip("turn_in_progress")
        self._stepping = True
        try:
            return await self._step()
        finally:
            self._stepping = False

    def _skip(self, reason: str) -> None:
        self.last_skip = reason
        return None

    def _resting(self) -> Optional[bool]:
        if self.is_rest_time is None:
            return False
        try:
            return bool(self.is_rest_time())
        except Exception as exc:
            logger.warning("Rest check failed: %s", type(exc).__name__)
            return None

    def _read_drives(self) -> Optional[Dict[str, Dict[str, Any]]]:
        drives = getattr(self.drives_source, "homeostatic_drives", None)
        if drives is None:
            return None
        try:
            state = drives.get_drive_state()
            free_energy = _finite(drives.get_free_energy())
        except Exception as exc:
            logger.warning("Drive read failed: %s", type(exc).__name__)
            return None
        if free_energy is not None:
            self.last_free_energy = free_energy
        return state

    async def _step(self) -> Optional[ActivityResult]:
        rest = self._resting()
        if rest is None or rest:
            return self._skip("rest" if rest else "rest_check_failed")
        drive_state = self._read_drives()
        if drive_state is None:
            return self._skip("no_drives")

        picked = self._pick(drive_state)
        if picked is None:
            return self._skip("no_urgent_drive")
        context, options, forced = picked
        if forced is not None:
            selection = forced
        else:
            selection = await self.selector.select(context, options)
        if self._turns_open > 0:
            return self._skip("turn_in_progress")

        self.last_skip = None
        activity = selection.activity
        try:
            result = await activity.run(context)
        except Exception as exc:
            logger.warning("Activity %s raised: %s", activity.name, type(exc).__name__)
            result = ActivityResult(
                activity=activity.name,
                success=False,
                error="activity_error",
            )
        if result.success and not (result.text or "").strip():
            result = ActivityResult(
                activity=activity.name,
                success=False,
                error="empty_text",
                metadata=dict(result.metadata),
            )
        self.steps_run += 1
        self.last_activity = activity.name
        self.last_activity_time = self._clock()

        if not result.success:
            reason = result.error or "failed"
            self.failures[reason] = self.failures.get(reason, 0) + 1
        else:
            await self._record_success(activity, context, selection, result)

        await self._maybe_consolidate()
        return result

    def _pick(
        self, drive_state: Dict[str, Dict[str, Any]]
    ) -> Optional[Tuple[ActivityContext, List[Activity], Optional[Selection]]]:
        cfg = self.config
        if cfg.exploration_probability > 0 and (
            self.rng.random() < cfg.exploration_probability
        ):
            everything = self.registry.all()
            if everything:
                chosen = self.rng.choice(everything)
                urgency = _finite((drive_state.get(chosen.drive) or {}).get("urgency"))
                context = self._context(chosen.drive, urgency or 0.0)
                return context, [chosen], Selection(chosen, SOURCE_EXPLORATION)
        urgent = most_urgent_drive(drive_state, self.registry, cfg.urgency_threshold)
        if urgent is None:
            return None
        drive, urgency = urgent
        return self._context(drive, urgency), self.registry.for_drive(drive), None

    def _context(self, drive: str, urgency: float) -> ActivityContext:
        descriptions: List[str] = []
        if self.goals is not None:
            try:
                descriptions = [
                    str(g.description) for g in self.goals.get_top_goals(TOP_GOALS)
                ]
            except Exception as exc:
                logger.warning("Goal read failed: %s", type(exc).__name__)
        topic = descriptions[0] if descriptions else None
        return ActivityContext(
            drive=drive, urgency=urgency, topic=topic, goals=descriptions
        )

    # -- after a successful activity ----------------------------------------------

    async def _record_success(
        self,
        activity: Activity,
        context: ActivityContext,
        selection: Selection,
        result: ActivityResult,
    ) -> None:
        self.completed += 1
        now = self._clock()
        text = result.text.strip()
        self.history.append(
            {
                "activity": activity.name,
                "drive": context.drive,
                "source": selection.source,
                "timestamp": now.isoformat(),
                "summary": text[:SUMMARY_CHARS],
            }
        )
        await self._experience(activity, context, selection, text, now)
        await self._credit_drive(activity)
        await self._credit_goal(activity)
        await self._maybe_share(result)

    async def _experience(
        self,
        activity: Activity,
        context: ActivityContext,
        selection: Selection,
        text: str,
        now: datetime,
    ) -> None:
        candidate = WorkspaceCandidate(
            content=text,
            content_type="autonomous_activity",
            summary=" ".join(text.split())[:80],
            source=WorkspaceCandidateSource.INTERNAL,
            source_module="autonomous_life",
            activation_level=0.5,
            timestamp=now.timestamp(),
            metadata={
                "activity": activity.name,
                "drive": context.drive,
                "selection_source": selection.source,
            },
        )
        if self.experience_sink is None:
            self.recent_candidates.append(candidate)
            return
        try:
            await self.experience_sink(candidate)
        except Exception as exc:
            self.sink_errors += 1
            logger.warning("Experience sink failed: %s", type(exc).__name__)

    async def _credit_drive(self, activity: Activity) -> None:
        mapped = self.config.activity_types.get(activity.name)
        if not mapped:
            return
        payload = {"type": mapped.get("type"), "intensity": mapped.get("intensity")}
        try:
            await self.drives_source.update_homeostasis(payload)
        except Exception as exc:
            logger.warning("Drive update failed: %s", type(exc).__name__)

    async def _credit_goal(self, activity: Activity) -> None:
        if self.goals is None:
            return
        try:
            goal = self.goals.record_activity(activity.name, GOAL_PROGRESS)
            if goal is not None:
                await self.goals.persist(goal)
        except Exception as exc:
            logger.warning("Goal credit failed: %s", type(exc).__name__)

    async def _maybe_share(self, result: ActivityResult) -> None:
        if self.share is None or self._turns_open > 0:
            return
        prompt = (
            "You just did something on your own.\n\n"
            f"Activity: {result.activity}\n"
            f"What came of it: {result.text[:300]}\n\n"
            "Would you like to keep it private, share it, or talk about it "
            "the next time you are in conversation? Reply with one word: "
            "KEEP, SHARE or DISCUSS."
        )
        try:
            reply = await self.brain.think(prompt, max_tokens=32, temperature=0.3)
        except NoAnswerError:
            return
        except Exception as exc:
            logger.warning("Share decision failed: %s", type(exc).__name__)
            return
        text = strip_reasoning_blocks(reply.text or "")
        found = {w.lower() for w in _WORD.findall(text) if w.lower() in _SHARE_WORDS}
        if len(found) != 1:
            return
        choice = next(iter(found))
        if choice == SHARE_KEEP:
            return
        if self._turns_open > 0:
            return
        shared = ActivityResult(
            activity=result.activity,
            success=True,
            text=result.text,
            metadata={**result.metadata, "share_choice": choice},
        )
        try:
            await self.share(shared)
        except Exception as exc:
            logger.warning("Share callback failed: %s", type(exc).__name__)

    async def _maybe_consolidate(self) -> None:
        every = self.config.consolidate_every
        if every <= 0 or self.consolidate_thoughts is None:
            return
        if self.steps_run % every != 0:
            return
        self.last_consolidation_step = self.steps_run
        try:
            self.last_consolidated = await self.consolidate_thoughts()
        except Exception as exc:
            logger.warning("Thought consolidation failed: %s", type(exc).__name__)

    # -- background task -------------------------------------------------------

    def start(self) -> None:
        """Start the loop task; a second call while it runs does nothing."""
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        """Stop the loop task and wait for it to end."""
        self._running = False
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while self._running:
            try:
                result = await self.step()
                wait = (
                    self.config.min_interval_s
                    if result is None
                    else self.next_interval_s()
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Life loop step failed: %s", type(exc).__name__)
                wait = self.config.min_interval_s
            await self._sleep(wait)

    # -- status and persistence --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Counts, the last activity, the last skip reason and what is wired."""
        return {
            "running": self._task is not None and not self._task.done(),
            "turn_in_progress": self._turns_open > 0,
            "steps_run": self.steps_run,
            "completed": self.completed,
            "failed": sum(self.failures.values()),
            "failures": dict(self.failures),
            "last_activity": self.last_activity,
            "last_activity_time": (
                self.last_activity_time.isoformat() if self.last_activity_time else None
            ),
            "last_skip": self.last_skip,
            "last_free_energy": self.last_free_energy,
            "last_consolidation_step": self.last_consolidation_step,
            "history_size": len(self.history),
            "recent_candidates": len(self.recent_candidates),
            "sink_errors": self.sink_errors,
            "activities": self.registry.names(),
            "has_sink": self.experience_sink is not None,
            "has_share": self.share is not None,
            "has_goals": self.goals is not None,
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Return ``{"scalars", "arrays", "histories"}`` for the persistence hub."""
        return {
            "scalars": {
                "steps_run": self.steps_run,
                "completed": self.completed,
                "last_activity": self.last_activity,
                "last_activity_time": (
                    self.last_activity_time.isoformat()
                    if self.last_activity_time
                    else None
                ),
                "last_free_energy": self.last_free_energy,
                "last_consolidation_step": self.last_consolidation_step,
            },
            "arrays": {},
            "histories": {
                "history": [dict(entry) for entry in self.history],
                "failures": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(self.failures.items())
                ],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a state dict, rebuilding the bounded history."""
        scalars = state.get("scalars", {})
        histories = state.get("histories", {})
        self.steps_run = int(scalars.get("steps_run", 0))
        self.completed = int(scalars.get("completed", 0))
        self.last_activity = scalars.get("last_activity")
        stamp = scalars.get("last_activity_time")
        self.last_activity_time = datetime.fromisoformat(stamp) if stamp else None
        reading = scalars.get("last_free_energy")
        self.last_free_energy = None if reading is None else float(reading)
        step = scalars.get("last_consolidation_step")
        self.last_consolidation_step = None if step is None else int(step)
        self.history = deque(
            (dict(e) for e in histories.get("history", [])),
            maxlen=self.config.history_size,
        )
        self.failures = {
            str(item["reason"]): int(item["count"])
            for item in histories.get("failures", [])
        }
