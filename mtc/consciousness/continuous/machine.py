"""
Continuous Thought Machine
==========================

Runs the background thought loop: generate a thought from the seed packs,
measure its novelty, decide whether it may surface, optionally give a surfacing
thought a model-written form, broadcast it, store it with dedup-on-write, and
now and then write a gist of the recent window.

One step is ``tick()``, which tests call directly; ``start()`` creates the one
loop task (tick, then wait ``next_delay()``) and returns, ``stop()`` cancels it
and waits for it. One tick runs at a time. A tick, in order:

1. While paused, nothing happens (``quiet_reason`` ``paused``).
2. Rate limit: with ``max_thoughts_per_minute`` thoughts generated in the last
   60 seconds, nothing happens (``rate_limited``) and the loop backs off for
   ``RATE_LIMIT_BACKOFF_S``.
3. The consciousness level is refreshed from the injected provider at the first
   tick and every ``consciousness_modulation.refresh_every`` thoughts. It scales
   the cadence by ``base + slope * level`` (0.4 + 1.2 * level by default). A
   provider that is absent or returns None leaves the cadence unmodulated, and
   ``get_status()["modulation"]`` says why (``no_provider``, ``unavailable``,
   ``disabled``); no level is assumed.
4. The generator makes a thought; None is a miss (``no_seeds`` when it had no
   seed text to work with).
5. Novelty (``NoveltyFilter``, off the event loop): should-express comes only
   from measured novelty, a unique thought novel enough under the filter's
   thresholds; the generator's provisional flag is replaced. With no embedder
   every thought stays quiet (``no_embedder``), with a non-semantic embedder and
   no opt-in too (``non_semantic_embedder``). Thoughts of a type starting with a
   ``surfacing.excluded_type_prefixes`` entry (dreams) never surface.
6. LLM form: a surfacing thought is rewritten by the enhancer only when the
   contention guard says the brain is free. No guard counts as contended, so
   the model is never taken from a conversation by default; a skipped upgrade
   reads ``contended``. The config's ``contention_window_s`` belongs to the
   guard the caller builds (for example a conversation flight that reports
   contention for that many seconds after a turn). An enhancer miss keeps the
   template thought and is counted (``llm_misses``); nothing is composed in the
   model's place. The model's line is measured as well, and a near-duplicate of
   a recent thought is dropped in favour of the novel template thought
   (``llm_duplicates``).
7. Dedup-on-write: the final thought is linked after the last stored thought.
   A near-duplicate is not stored; the store counts one more access of the
   earlier thought instead. A unique thought gets its memory associations
   (only with a ``vector_search``, off the loop), the current burst's
   ``stream_id`` (a new burst after ``burst_gap_s`` of silence or a change of
   state; a thought with no recorded state changes no burst) and is inserted;
   the parent's child link is recorded by ``thought_id``.
8. Only a stored thought joins the buffer, so every buffered record and every
   continuity link in the buffer names a thought the store holds (a
   near-duplicate or a failed insert stays out). Then an expressed thought is
   broadcast (``spontaneous_thought``) with the stream summary.
9. Every ``gist.every`` stored thoughts, outside the ``cooldown_s`` and only when
   at least ``min_new_fraction`` of the window is new, a gist is composed from
   the window's own thought texts (no model call) with ``source_count`` and the
   configured starting retention. Its ``dominant_emotion`` is the emotion that
   most often scored above ``GIST_EMOTION_FLOOR`` in the window, or None when
   no thought in it carried one.
10. Every ``resource.check_every`` thoughts the process CPU is sampled without
    blocking (psutil ``cpu_percent(interval=None)``, optional); above the limit
    the cadence is halved, and it is restored once a later sample is under it.

Hooks: ``on_thought_generated`` and ``on_thought_persisted`` receive the record
(sync or async). Every step that fails is counted by step name in
``get_status()["failures"]``, logged at WARNING the first time and every
hundredth time, and re-raised when ``Settings.DEBUG`` is true.

``inject_thought`` (the sleep scheduler's ``ThoughtStreamControl``) stores an
external thought, such as a dream, unexpressed, even while paused.
``set_consciousness_state`` takes the scheduler's vocabulary (``awake``,
``drowsy``, ``dreaming``, ``deep_sleep``).

Persisted state (``to_state_dict``): counters, the state and level, the burst,
the last stored thought id, the gist window and the recent thought times. The
generator and the buffer persist themselves. Thought text, memories and
emotions are private runtime data; this module logs none of them. Config: the
CTM config (``config/ctm/ctm.yaml``) and ``DEBUG``.
"""

import asyncio
import inspect
import logging
import math
import random
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.novelty import NoveltyFilter, NoveltyResult
from mtc.consciousness.continuous.schemas import (
    CONSCIOUSNESS_STATES,
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
    new_thought_id,
)
from mtc.core.config import Settings, get_settings
from mtc.memory.thought_gist import ThoughtGist

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORY_BY_TYPE",
    "ContinuousThoughtMachine",
    "QUIET_REASONS",
    "TickResult",
]

QUIET_REASONS = (
    "no_embedder",
    "non_semantic_embedder",
    "contended",
    "paused",
    "rate_limited",
    "no_seeds",
)

RATE_WINDOW_S = 60.0
RATE_LIMIT_BACKOFF_S = 10.0
PAUSED_POLL_S = 1.0
LOOP_ERROR_BACKOFF_S = 5.0
IDLE_DELAY_S = 60.0
FAILURE_LOG_EVERY = 100
RESOURCE_THROTTLE = 0.5
GIST_PREVIEW_ITEMS = 3
GIST_PREVIEW_CHARS = 200
GIST_EMOTION_FLOOR = 0.3
INJECTED_SALIENCE = 0.5

# Thought type -> the enhancer's category for its model-written form.
CATEGORY_BY_TYPE: Dict[str, str] = {
    ThoughtType.MEMORY_REPLAY.value: "memory",
    ThoughtType.QUESTION.value: "question",
    ThoughtType.EMOTION.value: "emotion",
    ThoughtType.CREATIVE.value: "creative",
}


@dataclass
class TickResult:
    """What one tick did.

    ``quiet_reason`` names the first reason the tick fell short of a stored,
    expressed, model-formed thought (``QUIET_REASONS``), or None when nothing
    held it back (a thought that is simply not novel enough is quiet by
    measurement, with no reason). ``thought_id`` and ``duplicate`` are
    additions for callers that follow the thought.
    """

    generated: bool = False
    expressed: bool = False
    persisted: bool = False
    upgraded: bool = False
    quiet_reason: Optional[str] = None
    thought_id: Optional[str] = None
    duplicate: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _call_hook(hook: Optional[Callable[..., Any]], *args: Any) -> None:
    if hook is None:
        return
    result = hook(*args)
    if inspect.isawaitable(result):
        await result


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.isoformat() if moment is not None else None


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None or moment.utcoffset() is None:
        return None
    return moment.astimezone(timezone.utc)


def _count(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class ContinuousThoughtMachine:
    """The background thought loop (see the module docstring)."""

    def __init__(
        self,
        generator: Any,
        buffer: Any,
        store: Any,
        gist_store: Any = None,
        novelty: Optional[NoveltyFilter] = None,
        enhancer: Any = None,
        config: Optional[CTMConfig] = None,
        broadcaster: Optional[Callable[[Dict[str, Any]], Any]] = None,
        contention_guard: Optional[Callable[[], bool]] = None,
        consciousness_level_provider: Optional[
            Callable[[], Awaitable[Optional[float]]]
        ] = None,
        vector_search: Any = None,
        on_thought_generated: Optional[Callable[[ThoughtRecord], Any]] = None,
        on_thought_persisted: Optional[Callable[[ThoughtRecord], Any]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        rng: Optional[random.Random] = None,
        *,
        settings: Optional[Settings] = None,
        process: Any = None,
    ) -> None:
        self.generator = generator
        self.buffer = buffer
        self.store = store
        self.gist_store = gist_store
        self.novelty = novelty
        self.enhancer = enhancer
        self.config = config if config is not None else CTMConfig()
        self.broadcaster = broadcaster
        self.contention_guard = contention_guard
        self.consciousness_level_provider = consciousness_level_provider
        self.vector_search = vector_search
        self.on_thought_generated = on_thought_generated
        self.on_thought_persisted = on_thought_persisted
        self.clock = clock or _utcnow
        self.rng = rng if rng is not None else random.Random()
        self._settings = settings

        self.consciousness_state = "awake"
        self.paused = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_lock = asyncio.Lock()
        self.started_at: Optional[datetime] = None

        self.consciousness_level: Optional[float] = None
        self._level_status = (
            "no_provider" if consciousness_level_provider is None else "pending"
        )
        self._level_refreshed_at: Optional[int] = None

        self._process = process
        self._process_checked = process is not None
        if process is None:
            self._acquire_process()  # primes the reading at construction
        self._resource_factor = 1.0
        self._resource_status: Dict[str, Any] = {
            "available": None,
            "last_cpu_percent": None,
            "throttled": False,
            "checks": 0,
            "breaches": 0,
        }

        self._thought_times: List[datetime] = []
        self._stream_id = uuid.uuid4().hex
        self._last_stream_time: Optional[datetime] = None
        self._last_stream_state: Optional[str] = None
        self._last_persisted_id: Optional[str] = None
        # template text -> stored text of a thought whose form was upgraded, so
        # a later near-duplicate of the template bumps the stored thought
        self._stored_text: "OrderedDict[str, str]" = OrderedDict()

        self._since_gist = 0
        self._last_gist_at: Optional[datetime] = None
        self._last_gist_window: List[str] = []

        self.counters: Counter = Counter()
        self.failures: Counter = Counter()
        self.last_quiet_reason: Optional[str] = None
        self._ticked = False
        reason = self._static_quiet_reason()
        if reason is not None:
            logger.warning(
                "continuous thoughts will stay quiet (%s): no measured novelty can "
                "let a thought surface",
                reason,
            )

    # -- helpers ------------------------------------------------------------

    def _debug(self) -> bool:
        settings = self._settings if self._settings is not None else get_settings()
        return bool(getattr(settings, "DEBUG", False))

    def _failure(self, step: str, exc: BaseException) -> None:
        """Count a failed step, log the first and every hundredth, raise in DEBUG."""
        self.failures[step] += 1
        count = self.failures[step]
        if count == 1 or count % FAILURE_LOG_EVERY == 0:
            logger.warning(
                "continuous thought step %s failed (%d so far): %s",
                step,
                count,
                type(exc).__name__,
            )
        if self._debug():
            raise exc

    @property
    def running(self) -> bool:
        """Whether the loop task is running."""
        return bool(self._running and self._task is not None and not self._task.done())

    @property
    def current_frequency(self) -> float:
        """Thoughts per second for the state, after modulation and throttling."""
        base = self.config.cadence.for_state(self.consciousness_state)
        return base * self.modulation_factor() * self._resource_factor

    def modulation_factor(self) -> float:
        """``base + slope * level``, or 1.0 without a measured level."""
        modulation = self.config.consciousness_modulation
        if not modulation.enabled or self.consciousness_level is None:
            return 1.0
        return modulation.base + modulation.slope * self.consciousness_level

    def next_delay(self, result: Optional[TickResult] = None) -> float:
        """Seconds to wait before the next tick (jittered, floored)."""
        if result is not None and result.quiet_reason == "rate_limited":
            return RATE_LIMIT_BACKOFF_S
        if self.paused:
            return PAUSED_POLL_S
        frequency = self.current_frequency
        if frequency <= 0 or not math.isfinite(frequency):
            return IDLE_DELAY_S
        base = 1.0 / frequency
        spread = base * self.config.jitter
        delay = base + self.rng.uniform(-spread, spread)
        return max(self.config.min_sleep_s, delay)

    def _static_quiet_reason(self) -> Optional[str]:
        if self.paused:
            return "paused"
        novelty = self.novelty
        if novelty is None or novelty.embedder is None:
            return "no_embedder"
        if not novelty.expression_possible:
            return "non_semantic_embedder"
        return None

    def _excluded(self, thought: ThoughtRecord) -> bool:
        prefixes = tuple(self.config.surfacing.excluded_type_prefixes)
        return bool(prefixes) and thought.thought_type.value.startswith(prefixes)

    def _contended(self) -> bool:
        guard = self.contention_guard
        if guard is None:
            return True
        try:
            return bool(guard())
        except Exception as exc:
            self._failure("contention_guard", exc)
            return True

    def _rate_limited(self, now: datetime) -> bool:
        floor = now - timedelta(seconds=RATE_WINDOW_S)
        self._thought_times = [t for t in self._thought_times if t > floor]
        return len(self._thought_times) >= self.config.max_thoughts_per_minute

    # -- level and resources -------------------------------------------------

    async def _refresh_level(self) -> None:
        provider = self.consciousness_level_provider
        if provider is None:
            return
        generated = self.counters["generated"]
        every = self.config.consciousness_modulation.refresh_every
        last = self._level_refreshed_at
        if last is not None and generated - last < every:
            return
        self._level_refreshed_at = generated
        try:
            value = await provider()
        except Exception as exc:
            self.consciousness_level = None
            self._level_status = "unavailable"
            self._failure("consciousness_level", exc)
            return
        number = None
        if value is not None and not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = None
        if number is None or not math.isfinite(number):
            self.consciousness_level = None
            self._level_status = "unavailable"
            return
        self.consciousness_level = max(0.0, min(1.0, number))
        self._level_status = "measured"

    def _acquire_process(self) -> Any:
        if self._process_checked:
            return self._process
        self._process_checked = True
        try:
            import psutil
        except ImportError:
            self._resource_status["available"] = False
            self._resource_status["reason"] = "psutil is not installed"
            return None
        process = psutil.Process()
        process.cpu_percent(interval=None)  # primes the non-blocking reading
        self._process = process
        return process

    def _check_resources(self) -> None:
        process = self._acquire_process()
        if process is None:
            self._resource_factor = 1.0
            return
        try:
            cpu = float(process.cpu_percent(interval=None))
        except Exception as exc:
            self._resource_status["available"] = False
            self._failure("resource_check", exc)
            return
        status = self._resource_status
        status["available"] = True
        status["checks"] += 1
        status["last_cpu_percent"] = cpu
        if cpu > self.config.resource.cpu_percent_limit:
            if not status["throttled"]:
                logger.warning(
                    "continuous thought cadence halved: CPU %.1f%% above %.1f%%",
                    cpu,
                    self.config.resource.cpu_percent_limit,
                )
            status["breaches"] += 1
            status["throttled"] = True
            self._resource_factor = RESOURCE_THROTTLE
        else:
            status["throttled"] = False
            self._resource_factor = 1.0

    # -- tick ---------------------------------------------------------------

    async def tick(self) -> TickResult:
        """Run one step of the loop (see the module docstring)."""
        async with self._tick_lock:
            result = await self._tick()
        self._ticked = True
        self.last_quiet_reason = result.quiet_reason
        return result

    async def _tick(self) -> TickResult:
        self.counters["ticks"] += 1
        if self.paused:
            return TickResult(quiet_reason="paused")
        now = self.clock()
        if self._rate_limited(now):
            self.counters["rate_limited"] += 1
            return TickResult(quiet_reason="rate_limited")
        await self._refresh_level()

        try:
            thought = await self.generator.generate_thought(self.consciousness_state)
        except Exception as exc:
            self._failure("generate", exc)
            return TickResult()
        if thought is None:
            self.counters["missed"] += 1
            reason = getattr(self.generator, "last_miss_reason", None)
            return TickResult(quiet_reason="no_seeds" if reason == "no_seeds" else None)

        self.counters["generated"] += 1
        self._thought_times.append(now)
        result = TickResult(generated=True)
        try:
            await _call_hook(self.on_thought_generated, thought)
        except Exception as exc:
            self._failure("hook_generated", exc)

        measured = await self._measure(thought)
        express = self._express_decision(thought, measured)
        result.quiet_reason = self._static_quiet_reason()

        final = thought
        if express and self.enhancer is not None:
            if self._contended():
                self.counters["contended"] += 1
                result.quiet_reason = result.quiet_reason or "contended"
            else:
                upgraded = await self._upgrade(thought)
                if upgraded is not None:
                    final = upgraded
                    result.upgraded = True

        if express:
            final.should_express = True
            final.expressed_at = self.clock()
        final.previous_thought_id = self._last_persisted_id

        persisted, duplicate = await self._persist(final, measured, result.upgraded)
        result.persisted = persisted
        result.duplicate = duplicate
        result.thought_id = final.thought_id
        if persisted:
            await self._add_to_buffer(final)

        if express:
            result.expressed = True
            self.counters["expressed"] += 1
            await self._broadcast(final)

        if persisted:
            await self._maybe_gist()
        if self.counters["generated"] % self.config.resource.check_every == 0:
            self._check_resources()
        return result

    async def _add_to_buffer(self, thought: ThoughtRecord) -> None:
        """Add a stored thought to the buffer (only stored thoughts join it)."""
        try:
            await self.buffer.add_thought(thought)
        except Exception as exc:
            self._failure("buffer", exc)

    async def _measure(self, thought: ThoughtRecord) -> Optional[NoveltyResult]:
        return await self._measure_text(thought.content)

    async def _measure_text(self, text: str) -> Optional[NoveltyResult]:
        novelty = self.novelty
        if novelty is None or novelty.embedder is None:
            return None
        try:
            return await asyncio.to_thread(novelty.find_near_duplicate, text)
        except Exception as exc:
            self._failure("novelty", exc)
            return None

    def _express_decision(
        self, thought: ThoughtRecord, measured: Optional[NoveltyResult]
    ) -> bool:
        context = thought.processing_context
        context["provisional_express"] = bool(thought.should_express)
        if measured is not None:
            context["novelty"] = round(measured.novelty, 4)
        allowed = (
            self.novelty is not None
            and self.novelty.express_allowed(measured)
            and not self._excluded(thought)
        )
        context["express_basis"] = "novelty" if measured is not None else "unmeasured"
        thought.should_express = bool(allowed)
        return bool(allowed)

    async def _upgrade(self, thought: ThoughtRecord) -> Optional[ThoughtRecord]:
        category = CATEGORY_BY_TYPE.get(thought.thought_type.value, "reflection")
        context = {}
        llm_context = getattr(self.generator, "llm_context", None)
        if callable(llm_context):
            context = dict(llm_context())
        if not context.get("emotions") and thought.emotional_context:
            context["emotions"] = dict(thought.emotional_context)
        try:
            upgraded = await self.enhancer.generate(
                category, context, self.consciousness_state
            )
        except Exception as exc:
            self._failure("upgrade", exc)
            upgraded = None
        if upgraded is None:
            self.counters["llm_misses"] += 1
            return None
        # the model's line is measured too: a rehash of a recent thought is
        # dropped and the novel template thought stays
        check = await self._measure_text(upgraded.content)
        if check is not None and not check.is_unique:
            self.counters["llm_duplicates"] += 1
            return None
        self.counters["llm_upgrades"] += 1
        merged = dict(thought.processing_context)
        merged.update(upgraded.processing_context or {})
        merged["upgraded_from"] = thought.thought_id
        merged["express_basis"] = "novelty"
        upgraded.processing_context = merged
        upgraded.consciousness_state = thought.consciousness_state
        if not upgraded.associated_memories:
            upgraded.associated_memories = list(thought.associated_memories)
        if not upgraded.emotional_context:
            upgraded.emotional_context = dict(thought.emotional_context)
        if upgraded.triggered_by is None:
            upgraded.triggered_by = thought.triggered_by
        self._remember_stored_text(thought.content, upgraded.content)
        return upgraded

    def _remember_stored_text(self, template: str, stored: str) -> None:
        self._stored_text[template] = stored
        self._stored_text.move_to_end(template)
        limit = max(1, self.config.novelty.cache)
        while len(self._stored_text) > limit:
            self._stored_text.popitem(last=False)

    async def _broadcast(self, thought: ThoughtRecord) -> None:
        if self.broadcaster is None:
            return
        try:
            summary = self.buffer.get_stream_summary()
        except Exception:
            summary = {}
        message = {
            "type": "spontaneous_thought",
            "thought": {
                "id": thought.thought_id,
                "type": thought.thought_type.value,
                "content": thought.content,
                "intensity": thought.intensity.value,
                "salience": thought.salience_score,
                "timestamp": thought.timestamp.isoformat(),
                "emotional_context": dict(thought.emotional_context),
            },
            "stream_state": summary,
        }
        try:
            await _call_hook(self.broadcaster, message)
            self.counters["broadcast"] += 1
        except Exception as exc:
            self._failure("broadcast", exc)

    def _assign_stream(self, thought: ThoughtRecord) -> None:
        moment = thought.timestamp
        # A thought with no recorded state (None) cannot show a change of state.
        state = thought.consciousness_state
        if self._last_stream_time is not None:
            gap = (moment - self._last_stream_time).total_seconds()
            changed = (
                state is not None
                and self._last_stream_state is not None
                and state != self._last_stream_state
            )
            if gap > self.config.burst_gap_s or changed:
                self._stream_id = uuid.uuid4().hex
                self.counters["bursts"] += 1
        self._last_stream_time = moment
        if state is not None:
            self._last_stream_state = state
        thought.stream_id = self._stream_id

    async def _associate(self, thought: ThoughtRecord) -> None:
        search = self.vector_search
        if search is None:
            return
        settings = self.config.association
        try:
            hits = await asyncio.to_thread(
                search.search_similar,
                thought.content,
                settings.top_k,
                settings.min_score,
            )
        except Exception as exc:
            self._failure("associate", exc)
            return
        for hit in hits or []:
            point = str(hit[0])
            if point not in thought.associated_memories:
                thought.associated_memories.append(point)

    async def _persist(
        self,
        thought: ThoughtRecord,
        measured: Optional[NoveltyResult],
        upgraded: bool,
    ) -> Tuple[bool, bool]:
        if measured is not None and not measured.is_unique and not upgraded:
            matched = measured.matched_text or ""
            text = self._stored_text.get(matched, matched)
            self.counters["duplicates_filtered"] += 1
            try:
                found = await self.store.bump_access(text)
            except Exception as exc:
                self._failure("bump_access", exc)
                return False, True
            if found is None:
                self.counters["access_unmatched"] += 1
            return False, True
        await self._associate(thought)
        return await self._insert(thought), False

    async def _insert(self, thought: ThoughtRecord) -> bool:
        self._assign_stream(thought)
        try:
            await self.store.insert(thought)
        except Exception as exc:
            self._failure("persist", exc)
            return False
        self.counters["persisted"] += 1
        parent = thought.previous_thought_id
        self._last_persisted_id = thought.thought_id
        if parent:
            try:
                linked = await self.store.link_child(parent, thought.thought_id)
            except Exception as exc:
                self._failure("link_child", exc)
                linked = False
            self.counters["links_resolved" if linked else "links_unresolved"] += 1
        try:
            await _call_hook(self.on_thought_persisted, thought)
        except Exception as exc:
            self._failure("hook_persisted", exc)
        return True

    # -- gists --------------------------------------------------------------

    async def _maybe_gist(self) -> None:
        if self.gist_store is None:
            return
        settings = self.config.gist
        self._since_gist += 1
        if self._since_gist < settings.every:
            return
        now = self.clock()
        if self._last_gist_at is not None:
            if (now - self._last_gist_at).total_seconds() < settings.cooldown_s:
                return
        self._since_gist = 0
        self._last_gist_at = now
        try:
            window = await self.store.recent(settings.every)
        except Exception as exc:
            self._failure("gist_window", exc)
            return
        if not window:
            return
        ids = [t.thought_id for t in window]
        new = len(set(ids) - set(self._last_gist_window))
        needed = max(1, math.ceil(len(ids) * settings.min_new_fraction))
        if self._last_gist_window and new < needed:
            self.counters["gists_skipped"] += 1
            return
        gist = self._compose_gist(window, now)
        try:
            await self.gist_store.add_gist(gist)
        except Exception as exc:
            self._failure("gist_store", exc)
            return
        self._last_gist_window = ids
        self.counters["gists"] += 1

    def _compose_gist(self, window: List[ThoughtRecord], now: datetime) -> ThoughtGist:
        texts = [t.content for t in window if t.content]
        types = Counter(t.thought_type.value for t in window)
        dominant_type = types.most_common(1)[0][0]
        emotions: Counter = Counter()
        for thought in window:
            for name, value in (thought.emotional_context or {}).items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value > GIST_EMOTION_FLOOR
                ):
                    emotions[name] += 1
        # None when no thought in the window carried a measured emotion: the
        # prompt then shows no mood for the gist.
        dominant_emotion = emotions.most_common(1)[0][0] if emotions else None
        preview = "; ".join(texts[:GIST_PREVIEW_ITEMS])
        if len(preview) > GIST_PREVIEW_CHARS:
            preview = preview[: GIST_PREVIEW_CHARS - 3].rstrip() + "..."
        salience = [t.salience_score for t in window]
        average = sum(salience) / len(salience)
        stamps = [t.timestamp for t in window]
        return ThoughtGist(
            gist_id=f"gist_{uuid.uuid4().hex}",
            created_at=now,
            summary=f"{dominant_type} thoughts ({len(texts)}): {preview}",
            source_count=len(texts),
            source_thought_ids=[t.thought_id for t in window],
            dominant_emotion=dominant_emotion,
            time_span={
                "start": min(stamps).isoformat(),
                "end": max(stamps).isoformat(),
            },
            avg_salience=average,
            total_access_count=sum(t.access_count for t in window),
            content_variations=texts[:GIST_PREVIEW_ITEMS],
            importance_score=average,
            retention_strength=self.config.gist.retention,
        )

    # -- control ------------------------------------------------------------

    async def start(self) -> None:
        """Create the loop task on the running loop and return."""
        if self.running:
            logger.info("continuous thought machine already running")
            return
        self._running = True
        self.started_at = self.clock()
        self._task = asyncio.get_running_loop().create_task(
            self._loop(), name="continuous-thought"
        )
        logger.info(
            "continuous thought machine started (state %s)", self.consciousness_state
        )

    async def stop(self) -> None:
        """Cancel the loop task and wait for it to end."""
        self._running = False
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "continuous thought loop ended with %s", type(exc).__name__
                )
        logger.info(
            "continuous thought machine stopped (%d thoughts generated)",
            self.counters["generated"],
        )

    async def _loop(self) -> None:
        while self._running:
            try:
                result = await self.tick()
                delay = self.next_delay(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failure("loop", exc)
                delay = LOOP_ERROR_BACKOFF_S
            await asyncio.sleep(delay)

    def pause(self) -> None:
        """Pause generation; ``inject_thought`` still stores."""
        self.paused = True

    def resume(self) -> None:
        """Resume generation."""
        self.paused = False

    def set_consciousness_state(self, mode: str) -> None:
        """Switch the thought mode (``awake``, ``drowsy``, ``dreaming``,
        ``deep_sleep``); an unknown mode raises ValueError."""
        if mode not in CONSCIOUSNESS_STATES:
            raise ValueError(
                f"unknown thought mode {mode!r}; expected one of "
                f"{CONSCIOUSNESS_STATES}"
            )
        self.consciousness_state = mode

    async def inject_thought(
        self, text: str, thought_type: str, emotional_tone: str
    ) -> Optional[ThoughtRecord]:
        """Store an external thought (for example a dream) unexpressed.

        A ``thought_type`` starting with ``dream`` is stored as a dream; another
        known type keeps its type; anything else is stored as spontaneous. The
        original label is kept in ``triggered_by`` and the tone word in
        ``processing_context["emotional_tone"]``. Returns the record, or None
        for an empty text (nothing is stored). Like a generated thought, the
        record joins the buffer only once it is stored; a failed insert is
        counted (``persist``) and leaves the buffer unchanged.
        """
        content = (text or "").strip()
        if not content:
            return None
        label = str(thought_type or "")
        if label.startswith(ThoughtType.DREAM.value):
            kind = ThoughtType.DREAM
        else:
            try:
                kind = ThoughtType(label)
            except ValueError:
                kind = ThoughtType.SPONTANEOUS
        record = ThoughtRecord(
            thought_id=new_thought_id(),
            timestamp=self.clock(),
            thought_type=kind,
            content=content,
            intensity=ThoughtIntensity.MODERATE,
            salience_score=INJECTED_SALIENCE,
            triggered_by=label or None,
            consciousness_state=self.consciousness_state,
            processing_context={
                "injected": True,
                "emotional_tone": emotional_tone or None,
                "salience_basis": "injected_default",
                "express_basis": "injected",
            },
            should_express=False,
        )
        record.previous_thought_id = self._last_persisted_id
        self.counters["injected"] += 1
        if await self._insert(record):
            await self._add_to_buffer(record)
        return record

    # -- status and persistence ----------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Counters, state, frequency, quiet reason, failures and the parts."""
        uptime = None
        if self.started_at is not None:
            uptime = (self.clock() - self.started_at).total_seconds()
        if self._ticked:
            quiet = self.last_quiet_reason
        else:
            quiet = self._static_quiet_reason()
        if self.paused:
            quiet = "paused"
        modulation = self._level_status
        if not self.config.consciousness_modulation.enabled:
            modulation = "disabled"
        try:
            summary = self.buffer.get_stream_summary()
        except Exception:
            summary = {}
        generator_status = getattr(self.generator, "status", None)
        store_status = getattr(self.store, "status", None)
        return {
            "running": self.running,
            "paused": self.paused,
            "consciousness_state": self.consciousness_state,
            "consciousness_level": self.consciousness_level,
            "modulation": modulation,
            "modulation_factor": self.modulation_factor(),
            "resource_factor": self._resource_factor,
            "current_frequency": self.current_frequency,
            "quiet_reason": quiet,
            "total_thoughts_generated": self.counters["generated"],
            "total_thoughts_expressed": self.counters["expressed"],
            "total_thoughts_missed": self.counters["missed"],
            "counters": dict(self.counters),
            "failures": dict(self.failures),
            "uptime_seconds": uptime,
            "recent_thought_rate": len(self._thought_times),
            "contention_guard": self.contention_guard is not None,
            "enhancer": self.enhancer is not None,
            "vector_search": self.vector_search is not None,
            "gist_store": self.gist_store is not None,
            "novelty": (
                self.novelty.status()
                if self.novelty is not None
                else {"embedder": "none", "expression_possible": False}
            ),
            "resource": dict(self._resource_status),
            "generator": generator_status() if callable(generator_status) else {},
            "store": store_status() if callable(store_status) else {},
            "stream_summary": summary,
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Counters, state, burst, gist window and recent thought times."""
        return {
            "scalars": {
                "counters": {k: int(v) for k, v in self.counters.items()},
                "consciousness_state": self.consciousness_state,
                "consciousness_level": self.consciousness_level,
                "since_gist": self._since_gist,
                "last_gist_at": _iso(self._last_gist_at),
                "stream_id": self._stream_id,
                "last_stream_time": _iso(self._last_stream_time),
                "last_stream_state": self._last_stream_state,
                "last_persisted_id": self._last_persisted_id,
            },
            "arrays": {},
            "histories": {
                "thought_times": [t.isoformat() for t in self._thought_times],
                "last_gist_window": list(self._last_gist_window),
            },
        }

    def from_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore what ``to_state_dict`` wrote; malformed entries are skipped."""
        scalars = state.get("scalars") or {}
        histories = state.get("histories") or {}
        counters = scalars.get("counters")
        if isinstance(counters, Mapping):
            restored: Counter = Counter()
            for name, value in counters.items():
                number = _count(value)
                if isinstance(name, str) and number is not None:
                    restored[name] = number
            self.counters = restored
        mode = scalars.get("consciousness_state")
        if mode in CONSCIOUSNESS_STATES:
            self.consciousness_state = mode
        level = scalars.get("consciousness_level")
        if level is None:
            self.consciousness_level = None
        elif isinstance(level, (int, float)) and not isinstance(level, bool):
            if math.isfinite(level):
                self.consciousness_level = max(0.0, min(1.0, float(level)))
        since = _count(scalars.get("since_gist"))
        if since is not None:
            self._since_gist = since
        self._last_gist_at = _parse_time(scalars.get("last_gist_at"))
        stream_id = scalars.get("stream_id")
        if isinstance(stream_id, str) and stream_id:
            self._stream_id = stream_id
        self._last_stream_time = _parse_time(scalars.get("last_stream_time"))
        last_state = scalars.get("last_stream_state")
        if last_state is None or last_state in CONSCIOUSNESS_STATES:
            self._last_stream_state = last_state
        persisted = scalars.get("last_persisted_id")
        if persisted is None or isinstance(persisted, str):
            self._last_persisted_id = persisted
        times = histories.get("thought_times")
        if isinstance(times, list):
            self._thought_times = [
                moment for moment in (_parse_time(t) for t in times) if moment
            ]
        window = histories.get("last_gist_window")
        if isinstance(window, list):
            self._last_gist_window = [w for w in window if isinstance(w, str)]
