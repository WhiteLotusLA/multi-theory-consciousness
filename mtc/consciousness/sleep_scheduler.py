"""
Sleep Scheduler
===============

A three-state machine (``awake``, ``napping``, ``dreaming``) driven by the
configured local clock and by inactivity:

- **Night sleep.** Inside the night window (``SLEEP_HOUR`` up to
  ``WAKE_HOUR`` in ``TIMEZONE``, wrapping midnight) the system goes to sleep
  after ``night_reentry_idle_minutes`` of quiet, so a conversation at night
  keeps it awake and it settles again once the conversation stops. It wakes
  when the window ends. While asleep at night a dream is due every
  ``dream_interval_minutes``.
- **Naps.** Outside the window, ``nap_idle_minutes`` without activity start a
  nap of ``nap_minutes``; a nap that reaches the night window becomes night
  sleep. After a nap ends on its own, no new nap starts for
  ``min_awake_minutes_after_nap``: without that guard the idle time measured
  from the last conversation is still past the threshold when the nap ends,
  and the machine would re-enter a nap on the very next tick, thrashing
  between the two states.
- **Activity.** ``notify_activity`` records a conversation turn and wakes the
  system from a nap or from night sleep at once.

``TIMEZONE``, ``WAKE_HOUR`` and ``SLEEP_HOUR`` are read from the settings at
call time (``settings`` or ``get_settings()``); stored timestamps are aware
UTC and local time is used only for the window and the status display. Every
clock read goes through the injectable ``clock``.

Sleep entry (a nap, night sleep, or a nap turning into night sleep) runs, in
order, while the system stays asleep: the thought stream is paused and set to
its sleep mode; the consolidator runs (any sleep); the memory pruner runs
(night only, with ``prune_dry_run``); one dream is attempted. Waking resets
the thought-stream mode and resumes it. If activity wakes the system while
this work is in flight, the remaining steps are skipped. A missing seam skips
its work, which ``get_status()`` reports under ``missing_seams`` and
``skipped_work``; a seam that raises is logged and counted under ``errors``,
never raised.

The thought-stream vocabulary is ``awake``, ``drowsy``, ``dreaming`` and
``deep_sleep`` (the thought machine's modes, recorded here for it). The
scheduler sends ``dreaming`` for a nap and for night sleep, because naps use
the dream strategies too, and ``awake`` on waking. A thought mode is not the
sleep state: the sleep state is this scheduler's state, which
``is_resting()`` and ``get_status()`` report.

Dreams go through ``BrainProtocol.think`` with the prompt the dream engine
builds from the theme, affect and goal sources. The request asks for the
dream's token cap plus a small allowance so the last sentence can close; a
reply stopped by the cap is cut back to its last sentence end. Reasoning
blocks are removed. ``NoAnswerError`` or an empty reply stores nothing,
injects nothing and is counted under ``no_answer``; no dream text is ever
composed by code. A reply that arrives after activity has woken the
system is discarded (neither stored nor injected) and counted under
``woken``. A produced dream goes to the dream store (in memory when
none is given) as a record with ``content``, ``dream_type``, ``timestamp``,
``state``, ``parameters``, ``requested_max_tokens``, ``finish_reason``,
``cap_hit``, ``trimmed_chars``, ``trim_result`` and ``output_tokens``, and is
injected into the thought stream as a ``dream_<type>`` thought. Dream records
can echo recent themes and goals, which hold real people's names at runtime,
and are private data.

Affect in the dream context: the latest PAD state counts as a reading only
when its measured inputs cover pleasure and arousal
(``measured_dimensions``); a state of unknown provenance (no measured list)
names no emotion and does not trigger the gentle floor. A reading names its
label as the dream's undertone when the label is not Neutral and the state
lies far enough from neutral (``emotion_min_strength``).

The gentle floor is an optional welfare setting, off by default. When
enabled and the latest measured pleasure is at or below
``gentle_floor_valence_threshold``, no emotion word reaches the prompt and
the dream engine is asked to bias its draw toward comforting types; no
configured word stands in for an emotion that was not measured.

``tick()`` performs one evaluation and is what tests call; ``start()`` runs
it every ``tick_seconds`` in one task and ``stop()`` cancels that task.
Persisted state (``to_state_dict``): the sleep state, the activity, nap and
dream instants, the counters and the dream engine's recent dream types.
"""

import asyncio
import inspect
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)
from zoneinfo import ZoneInfo

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks
from mtc.consciousness.dream_engine import (
    DreamContext,
    DreamEngine,
    dream_sentence_allowance,
    trim_to_sentence_end,
)
from mtc.consciousness.pad_affect import PADState, measured_dimensions
from mtc.consciousness.temporal_context import in_night_window
from mtc.core.config import get_settings
from mtc.memory.inmemory import InMemoryDreamStore
from mtc.memory.protocols import DreamStore

logger = logging.getLogger(__name__)


class SleepState(str, Enum):
    """The scheduler's sleep state."""

    AWAKE = "awake"
    NAPPING = "napping"
    DREAMING = "dreaming"


THOUGHT_MODES = ("awake", "drowsy", "dreaming", "deep_sleep")
THOUGHT_MODE_BY_STATE: Dict[SleepState, str] = {
    SleepState.AWAKE: "awake",
    SleepState.NAPPING: "dreaming",
    SleepState.DREAMING: "dreaming",
}
# Finish reasons that mean the reply stopped at the token limit.
CAP_FINISH_REASONS = frozenset({"length", "max_tokens"})
TRIM_NOT_NEEDED = "not_needed"
SEAM_NAMES = (
    "brain",
    "thought_stream",
    "consolidator",
    "pruner",
    "themes",
    "affect",
    "goals",
)
_ERROR_KEYS = (
    "consolidation",
    "pruning",
    "dream",
    "dream_store",
    "thought_stream",
    "context",
)
_SKIP_KEYS = ("consolidation", "pruning", "thought_stream", "dream")
_DREAM_KEYS = ("attempted", "stored", "no_answer", "woken")
# PAD dimensions a state must have measured inputs for to count as a reading.
_READING_DIMENSIONS = frozenset({"pleasure", "arousal"})


# ----------------------------------------------------------------------------
# Configuration and seams
# ----------------------------------------------------------------------------


@dataclass
class SleepSchedulerConfig:
    """Intervals, thresholds and switches of the sleep scheduler.

    ``min_awake_minutes_after_nap`` is the thrash guard (by default as long
    as the nap idle threshold); ``prune_dry_run`` False prunes for real at
    night. ``emotion_min_strength`` is the distance of the latest PAD state
    from neutral (``|P| + |A - 0.5| + |D - 0.5|``, capped at 1) above which
    its label becomes the dream's undertone; only a state whose measured
    inputs cover pleasure and arousal and whose label is not Neutral is
    named. ``gentle_floor_enabled`` with ``gentle_floor_valence_threshold``
    passes the comfort bias, and no emotion, for a measured pleasure at or
    below the threshold. ``context_items`` is how many themes and goals are
    read for a dream.
    """

    tick_seconds: float = 60.0
    nap_idle_minutes: float = 30.0
    nap_minutes: float = 15.0
    night_reentry_idle_minutes: float = 5.0
    dream_interval_minutes: float = 90.0
    min_awake_minutes_after_nap: float = 30.0
    prune_dry_run: bool = False
    gentle_floor_enabled: bool = False
    gentle_floor_valence_threshold: float = -0.4
    emotion_min_strength: float = 0.3
    context_items: int = 3

    def __post_init__(self) -> None:
        positive = (
            "tick_seconds",
            "nap_idle_minutes",
            "nap_minutes",
            "dream_interval_minutes",
        )
        for name in positive:
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be > 0")
        for name in ("night_reentry_idle_minutes", "min_awake_minutes_after_nap"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not -1.0 <= float(self.gentle_floor_valence_threshold) <= 1.0:
            raise ValueError("gentle_floor_valence_threshold must be within [-1, 1]")
        if not 0.0 <= float(self.emotion_min_strength) <= 1.0:
            raise ValueError("emotion_min_strength must be within [0, 1]")
        if int(self.context_items) < 1:
            raise ValueError("context_items must be at least 1")


@runtime_checkable
class ThoughtStreamControl(Protocol):
    """The thought stream as the scheduler drives it.

    Methods may be plain or async; an awaitable result is awaited (from
    ``notify_activity`` it is scheduled on the running loop).
    """

    def set_consciousness_state(self, mode: str) -> Any: ...

    def pause(self) -> Any: ...

    def resume(self) -> Any: ...

    def inject_thought(
        self, text: str, thought_type: str, emotional_tone: str
    ) -> Any: ...


@runtime_checkable
class Consolidator(Protocol):
    """Runs a consolidation pass (for example the learning system's)."""

    async def consolidate(self) -> Dict[str, int]: ...


@runtime_checkable
class MemoryPruner(Protocol):
    """Runs a forgetting pass over stored memories."""

    async def prune(self, dry_run: bool) -> Dict[str, int]: ...


@runtime_checkable
class ThemeSource(Protocol):
    """Recent themes (short gists) of the thought stream, newest first."""

    def get_recent_themes(self, n: int) -> List[str]: ...


@runtime_checkable
class AffectSource(Protocol):
    """The latest PAD affect state, or None before one exists."""

    def get_latest(self) -> Optional[PADState]: ...


@runtime_checkable
class GoalSource(Protocol):
    """The top goals; each item exposes a ``description`` string."""

    def get_top_goals(self, n: int) -> List[Any]: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _minutes(value: float) -> timedelta:
    return timedelta(minutes=float(value))


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.isoformat() if moment is not None else None


def _parse(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("persisted sleep timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


# ----------------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------------


class SleepScheduler:
    """Sleep and wake transitions, sleep-entry work and dreams."""

    def __init__(
        self,
        brain: Optional[BrainProtocol] = None,
        dream_engine: Optional[DreamEngine] = None,
        dream_store: Optional[DreamStore] = None,
        thought_stream: Optional[ThoughtStreamControl] = None,
        consolidator: Optional[Consolidator] = None,
        pruner: Optional[MemoryPruner] = None,
        themes: Optional[ThemeSource] = None,
        affect: Optional[AffectSource] = None,
        goals: Optional[GoalSource] = None,
        config: Optional[SleepSchedulerConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
        settings: Any = None,
    ) -> None:
        self.brain = brain
        self.dream_store: DreamStore = (
            dream_store if dream_store is not None else InMemoryDreamStore()
        )
        self.thought_stream = thought_stream
        self.consolidator = consolidator
        self.pruner = pruner
        self.themes = themes
        self.affect = affect
        self.goals = goals
        self.config = config if config is not None else SleepSchedulerConfig()
        self._dream_engine = dream_engine
        self._restored_dream_types: List[str] = []
        self._clock = clock if clock is not None else _utc_now
        self._settings = settings

        self._state = SleepState.AWAKE
        self._last_activity = self._instant(None)
        self._nap_start: Optional[datetime] = None
        self._last_nap_end: Optional[datetime] = None
        self._dream_clock: Optional[datetime] = None
        self._last_dream_at: Optional[datetime] = None
        self._dreams = {key: 0 for key in _DREAM_KEYS}
        self._errors = {key: 0 for key in _ERROR_KEYS}
        self._skipped = {key: 0 for key in _SKIP_KEYS}
        self._mode_dirty = False
        self._task: Optional[asyncio.Task] = None
        self._pending: Set[asyncio.Task] = set()

    # -- clock and locale ---------------------------------------------------

    def _instant(self, now: Optional[datetime]) -> datetime:
        moment = self._clock() if now is None else now
        if (
            not isinstance(moment, datetime)
            or moment.tzinfo is None
            or moment.utcoffset() is None
        ):
            raise ValueError("the sleep scheduler needs timezone-aware datetimes")
        return moment.astimezone(timezone.utc)

    def _locale(self) -> Tuple[ZoneInfo, int, int]:
        cfg = self._settings if self._settings is not None else get_settings()
        return ZoneInfo(str(cfg.TIMEZONE)), int(cfg.WAKE_HOUR), int(cfg.SLEEP_HOUR)

    def _is_night(self, instant: datetime) -> bool:
        zone, wake, sleep = self._locale()
        local = instant.astimezone(zone)
        hour = local.hour + local.minute / 60.0 + local.second / 3600.0
        return in_night_window(hour, wake, sleep)

    # -- public state -------------------------------------------------------

    @property
    def state(self) -> SleepState:
        return self._state

    def is_resting(self) -> bool:
        """True while napping or asleep at night."""
        return self._state is not SleepState.AWAKE

    @property
    def dream_engine(self) -> DreamEngine:
        """The dream engine; the default one (``config/dreams.yaml``) on first use."""
        if self._dream_engine is None:
            self._dream_engine = DreamEngine()
            if self._restored_dream_types:
                self._dream_engine.from_state_dict(
                    {"histories": {"dream_types": self._restored_dream_types}}
                )
                self._restored_dream_types = []
        return self._dream_engine

    # -- transitions --------------------------------------------------------

    async def tick(self, now: Optional[datetime] = None) -> SleepState:
        """Evaluate the clock and inactivity once and run any due work."""
        instant = self._instant(now)
        night = self._is_night(instant)
        if self._mode_dirty:
            self._mode_dirty = False
            await self._await_all(self._thought_mode_calls(self._state, False))
        cfg = self.config
        idle = instant - self._last_activity
        state = self._state
        if state is SleepState.AWAKE:
            if night and idle >= _minutes(cfg.night_reentry_idle_minutes):
                await self._enter_sleep(SleepState.DREAMING, instant)
            elif (
                not night
                and idle >= _minutes(cfg.nap_idle_minutes)
                and self._nap_guard_clear(instant)
            ):
                await self._enter_sleep(SleepState.NAPPING, instant)
        elif state is SleepState.NAPPING:
            if night:
                await self._enter_sleep(SleepState.DREAMING, instant)
            elif self._nap_start is None or instant - self._nap_start >= _minutes(
                cfg.nap_minutes
            ):
                await self._await_all(self._become_awake(instant, "nap ended"))
        elif state is SleepState.DREAMING:
            if not night:
                await self._await_all(self._become_awake(instant, "night ended"))
            elif self._dream_clock is None or instant - self._dream_clock >= _minutes(
                cfg.dream_interval_minutes
            ):
                self._dream_clock = instant
                await self._dream(instant)
        return self._state

    def notify_activity(self, now: Optional[datetime] = None) -> None:
        """Record a conversation turn; wake at once from a nap or night sleep."""
        instant = self._instant(now)
        self._last_activity = instant
        if self._state is not SleepState.AWAKE:
            self._schedule(self._become_awake(instant, "activity"))

    def _nap_guard_clear(self, instant: datetime) -> bool:
        if self._last_nap_end is None:
            return True
        guard = _minutes(self.config.min_awake_minutes_after_nap)
        return instant - self._last_nap_end >= guard

    async def _enter_sleep(self, new_state: SleepState, instant: datetime) -> None:
        old = self._state
        self._state = new_state
        self._nap_start = instant if new_state is SleepState.NAPPING else None
        self._dream_clock = instant
        logger.info("sleep state: %s -> %s", old.value, new_state.value)
        await self._await_all(self._thought_mode_calls(new_state))
        if self._state is not new_state:
            return
        await self._consolidate()
        if self._state is not new_state:
            return
        if new_state is SleepState.DREAMING:
            await self._prune()
            if self._state is not new_state:
                return
        await self._dream(instant)

    def _become_awake(self, instant: datetime, reason: str) -> List[Awaitable[Any]]:
        old = self._state
        self._state = SleepState.AWAKE
        if old is SleepState.NAPPING:
            self._last_nap_end = instant
        self._nap_start = None
        logger.info("sleep state: %s -> awake (%s)", old.value, reason)
        return self._thought_mode_calls(SleepState.AWAKE)

    # -- thought stream -----------------------------------------------------

    def _thought_mode_calls(
        self, state: SleepState, count_skip: bool = True
    ) -> List[Awaitable[Any]]:
        """Set the stream's mode and pause or resume it; return awaitables."""
        stream = self.thought_stream
        if stream is None:
            if count_skip:
                self._skipped["thought_stream"] += 1
            return []
        pending: List[Awaitable[Any]] = []
        try:
            for result in (
                stream.set_consciousness_state(THOUGHT_MODE_BY_STATE[state]),
                stream.resume() if state is SleepState.AWAKE else stream.pause(),
            ):
                if inspect.isawaitable(result):
                    pending.append(result)
        except Exception as exc:
            self._errors["thought_stream"] += 1
            logger.warning("thought stream mode change failed: %s", exc)
        return pending

    async def _await_all(self, pending: List[Awaitable[Any]]) -> None:
        for item in pending:
            try:
                await item
            except Exception as exc:
                self._errors["thought_stream"] += 1
                logger.warning("thought stream call failed: %s", exc)

    def _schedule(self, pending: List[Awaitable[Any]]) -> None:
        if not pending:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            for item in pending:
                if inspect.iscoroutine(item):
                    item.close()
            self._errors["thought_stream"] += 1
            logger.warning("thought stream call dropped: no running event loop")
            return
        task = loop.create_task(self._await_all(pending))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Wait for thought-stream calls scheduled by ``notify_activity``."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    # -- sleep-entry work ---------------------------------------------------

    async def _consolidate(self) -> None:
        if self.consolidator is None:
            self._skipped["consolidation"] += 1
            return
        try:
            result = await self.consolidator.consolidate()
            logger.info("sleep consolidation finished: %s", result)
        except Exception as exc:
            self._errors["consolidation"] += 1
            logger.warning("sleep consolidation failed: %s", exc)

    async def _prune(self) -> None:
        if self.pruner is None:
            self._skipped["pruning"] += 1
            return
        try:
            result = await self.pruner.prune(dry_run=bool(self.config.prune_dry_run))
            logger.info(
                "sleep pruning finished (dry_run=%s): %s",
                self.config.prune_dry_run,
                result,
            )
        except Exception as exc:
            self._errors["pruning"] += 1
            logger.warning("sleep pruning failed: %s", exc)

    # -- dreams -------------------------------------------------------------

    def _dream_context(self, mode: str) -> Tuple[DreamContext, bool]:
        """Build the dream context; each source failing degrades only itself."""
        cfg = self.config
        count = int(cfg.context_items)
        topics: List[str] = []
        if self.themes is not None:
            try:
                themes = self.themes.get_recent_themes(count) or []
                topics = [
                    theme.strip()
                    for theme in themes
                    if isinstance(theme, str) and theme.strip()
                ][:count]
            except Exception as exc:
                self._errors["context"] += 1
                logger.debug("dream context: themes unavailable: %s", exc)

        emotion: Optional[str] = None
        gentle = False
        if self.affect is not None:
            try:
                pad = self.affect.get_latest()
                if pad is not None and _READING_DIMENSIONS <= measured_dimensions(pad):
                    values = (
                        float(pad.pleasure),
                        float(pad.arousal),
                        float(pad.dominance),
                    )
                    if all(math.isfinite(value) for value in values):
                        pleasure, arousal, dominance = values
                        gentle = bool(cfg.gentle_floor_enabled) and (
                            pleasure <= cfg.gentle_floor_valence_threshold
                        )
                        if not gentle:
                            strength = min(
                                1.0,
                                abs(pleasure)
                                + abs(arousal - 0.5)
                                + abs(dominance - 0.5),
                            )
                            label = (getattr(pad, "label", "") or "").strip().lower()
                            if (
                                label
                                and label != "neutral"
                                and strength > cfg.emotion_min_strength
                            ):
                                emotion = label
            except Exception as exc:
                self._errors["context"] += 1
                logger.debug("dream context: affect unavailable: %s", exc)

        question: Optional[str] = None
        if self.goals is not None:
            try:
                for goal in self.goals.get_top_goals(count) or []:
                    description = getattr(goal, "description", None)
                    if isinstance(description, str) and description.strip():
                        question = description.strip()
                        break
            except Exception as exc:
                self._errors["context"] += 1
                logger.debug("dream context: goals unavailable: %s", exc)

        return DreamContext(topics, emotion, question, mode), gentle

    async def _dream(self, instant: datetime) -> Optional[Dict[str, Any]]:
        """Attempt one dream; return the stored record or None."""
        if self.brain is None:
            self._skipped["dream"] += 1
            return None
        state = self._state
        mode = "nap" if state is SleepState.NAPPING else "night"
        try:
            engine = self.dream_engine
            context, gentle = self._dream_context(mode)
            prompt = engine.build_prompt(context, comfort_bias=True if gentle else None)
        except Exception as exc:
            self._errors["dream"] += 1
            logger.warning("dream prompt could not be built: %s", exc)
            return None
        requested = prompt.max_tokens + dream_sentence_allowance(prompt.max_tokens)
        self._dreams["attempted"] += 1
        try:
            reply = await self.brain.think(
                prompt=prompt.prompt,
                system_prompt=prompt.system_prompt,
                max_tokens=requested,
                temperature=prompt.temperature,
                thinking_mode=False,
            )
        except NoAnswerError:
            self._dreams["no_answer"] += 1
            logger.info("dream attempt got no answer; nothing stored")
            return None
        except Exception as exc:
            self._errors["dream"] += 1
            logger.warning("dream generation failed: %s", exc)
            return None
        if self._state is not state:
            # Activity woke the system while the brain was answering: the
            # stream is already awake for the conversation, so the dream is
            # neither stored nor injected.
            self._dreams["woken"] += 1
            logger.info(
                "dream reply arrived after waking (%s -> %s); discarded",
                state.value,
                self._state.value,
            )
            return None

        text = strip_reasoning_blocks(getattr(reply, "text", "") or "")
        if not text:
            self._dreams["no_answer"] += 1
            logger.info("dream reply was empty; nothing stored")
            return None
        finish_reason = getattr(reply, "finish_reason", None)
        output_tokens = getattr(reply, "output_tokens", None)
        cap_hit = finish_reason in CAP_FINISH_REASONS or (
            finish_reason is None
            and output_tokens is not None
            and output_tokens >= requested
        )
        trimmed_chars = 0
        trim_result = TRIM_NOT_NEEDED
        if cap_hit:
            trim = trim_to_sentence_end(text)
            text, trimmed_chars, trim_result = (
                trim.text,
                trim.trimmed_chars,
                trim.result,
            )
            logger.info(
                "dream reached its cap (%d tokens requested, cap %d): %s, %d chars cut",
                requested,
                prompt.max_tokens,
                trim_result,
                trimmed_chars,
            )

        record: Dict[str, Any] = {
            "content": text,
            "dream_type": prompt.dream_type,
            "timestamp": instant,
            "state": state.value,
            "parameters": {
                "temperature": prompt.temperature,
                "max_tokens": prompt.max_tokens,
                "mode": mode,
                "gentle_floor": gentle,
            },
            "requested_max_tokens": requested,
            "finish_reason": finish_reason,
            "cap_hit": cap_hit,
            "trimmed_chars": trimmed_chars,
            "trim_result": trim_result,
            "output_tokens": output_tokens,
        }
        self._last_dream_at = instant
        try:
            await self.dream_store.add_dream(record)
            self._dreams["stored"] += 1
            logger.info(
                "dream stored: type=%s chars=%d state=%s",
                prompt.dream_type,
                len(text),
                state.value,
            )
        except Exception as exc:
            self._errors["dream_store"] += 1
            logger.warning("dream storage failed: %s", exc)

        if self.thought_stream is not None:
            try:
                result = self.thought_stream.inject_thought(
                    text=text,
                    thought_type=f"dream_{prompt.dream_type}",
                    emotional_tone=context.dominant_emotion or "",
                )
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._errors["thought_stream"] += 1
                logger.warning(
                    "dream injection into the thought stream failed: %s", exc
                )
        return record

    # -- background loop ----------------------------------------------------

    def start(self) -> None:
        """Run ``tick()`` every ``tick_seconds`` in one task on the running loop."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("sleep scheduler tick failed: %s", exc)
            await asyncio.sleep(float(self.config.tick_seconds))

    async def stop(self) -> None:
        """Cancel the loop task and wait for scheduled thought-stream calls."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.drain()

    # -- status and persistence ---------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Report the state, the local clock, counters and the missing seams."""
        instant = self._instant(None)
        status: Dict[str, Any] = {
            "state": self._state.value,
            "is_resting": self.is_resting(),
            "idle_seconds": (instant - self._last_activity).total_seconds(),
            "nap_remaining_seconds": None,
            "last_dream_at": _iso(self._last_dream_at),
            "dreams": dict(self._dreams),
            "errors": dict(self._errors),
            "skipped_work": dict(self._skipped),
            "missing_seams": [
                name for name in SEAM_NAMES if getattr(self, name) is None
            ],
            "running": self._task is not None and not self._task.done(),
            "config": asdict(self.config),
        }
        if self._state is SleepState.NAPPING and self._nap_start is not None:
            elapsed = (instant - self._nap_start).total_seconds()
            status["nap_remaining_seconds"] = max(
                0.0, float(self.config.nap_minutes) * 60.0 - elapsed
            )
        try:
            zone, _, _ = self._locale()
            status["timezone"] = str(zone.key)
            status["local_time"] = instant.astimezone(zone).isoformat()
            status["is_night"] = self._is_night(instant)
        except Exception as exc:
            logger.warning("sleep status: locale unavailable: %s", exc)
            status["timezone"] = None
            status["local_time"] = None
            status["is_night"] = None
        return status

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for the persistence hub (instants as ISO strings)."""
        scalars: Dict[str, Any] = {
            "state": self._state.value,
            "last_activity": _iso(self._last_activity),
            "nap_start": _iso(self._nap_start),
            "last_nap_end": _iso(self._last_nap_end),
            "dream_clock": _iso(self._dream_clock),
            "last_dream_at": _iso(self._last_dream_at),
        }
        for key, value in self._dreams.items():
            scalars[f"dreams_{key}"] = value
        for key, value in self._errors.items():
            scalars[f"errors_{key}"] = value
        for key, value in self._skipped.items():
            scalars[f"skipped_{key}"] = value
        if self._dream_engine is not None:
            dream_types = self._dream_engine.recent_types
        else:
            dream_types = list(self._restored_dream_types)
        return {
            "scalars": scalars,
            "arrays": {},
            "histories": {"dream_types": dream_types},
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore typed state; the next tick re-applies the thought-stream mode."""
        scalars = (state or {}).get("scalars", {})
        histories = (state or {}).get("histories", {})
        self._state = SleepState(scalars.get("state", SleepState.AWAKE.value))
        self._last_activity = _parse(scalars.get("last_activity")) or self._instant(
            None
        )
        self._nap_start = _parse(scalars.get("nap_start"))
        self._last_nap_end = _parse(scalars.get("last_nap_end"))
        self._dream_clock = _parse(scalars.get("dream_clock"))
        self._last_dream_at = _parse(scalars.get("last_dream_at"))
        self._dreams = {
            key: int(scalars.get(f"dreams_{key}", 0)) for key in _DREAM_KEYS
        }
        self._errors = {
            key: int(scalars.get(f"errors_{key}", 0)) for key in _ERROR_KEYS
        }
        self._skipped = {
            key: int(scalars.get(f"skipped_{key}", 0)) for key in _SKIP_KEYS
        }
        dream_types = [str(name) for name in histories.get("dream_types", [])]
        if self._dream_engine is not None:
            self._dream_engine.from_state_dict(
                {"histories": {"dream_types": dream_types}}
            )
        else:
            self._restored_dream_types = dream_types
        self._mode_dirty = True
