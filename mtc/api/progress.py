"""
Bounded Generation with Heartbeats
==================================

``run_with_heartbeat`` awaits one generation (a conversation turn) under two
bounds while reporting progress during silence, so a stalled generation never
holds the turn lock forever and a long but healthy one never leaves the
client without news.

- The first-token bound (``GENERATION_TIMEOUT_S``) counts from the start of
  the wait until the first streamed chunk, reasoning included. Time spent
  queued behind another turn or in a slow prefill counts against it and
  against nothing else.
- The stall bound (``GENERATION_STALL_TIMEOUT_S``) applies once the first
  chunk has arrived: the generation is cut when no chunk has arrived for that
  long. A long reply that keeps streaming is never cut.
- A heartbeat callback runs after every ``THINKING_HEARTBEAT_S`` of silence
  (no chunk and no earlier heartbeat in that time) with the seconds elapsed
  since the start. A failing heartbeat is logged at DEBUG and never aborts
  the work.

The operation is built by ``coro_factory(mark)``: the runner passes a fresh
``FirstTokenMark`` and the factory hands ``mark.wrap(callback)`` to the brain
as its stream callback, so every chunk passes through the mark. The brain's
retry sentinel and empty chunks are forwarded but are not tokens.

When a bound fires the operation is cancelled first and ``GenerationTimeout``
names the bound (``first_token`` or ``stall``). Any other outcome of the
operation (its result or its exception) propagates unchanged.

Config keys read at call time when an argument is None:
``GENERATION_TIMEOUT_S``, ``GENERATION_STALL_TIMEOUT_S`` and
``THINKING_HEARTBEAT_S``. Nothing is persisted. Streamed chunks are private
conversation text; they pass through untouched and are never logged.
"""

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Literal, Optional, Tuple, TypeVar

from mtc.brain.brain import STREAM_RETRY_SENTINEL
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

Bound = Literal["first_token", "stall"]
BOUNDS: Tuple[str, ...] = ("first_token", "stall")

HeartbeatFn = Callable[[float], Awaitable[Any]]
StreamSink = Callable[[str], Any]


class FirstTokenMark:
    """When the first and the latest streamed chunk arrived, on the loop clock.

    ``first_at`` and ``last_at`` are None until the first real chunk.
    """

    def __init__(self) -> None:
        self.first_at: Optional[float] = None
        self.last_at: Optional[float] = None
        self.chunks = 0
        self._first = asyncio.Event()

    @property
    def is_set(self) -> bool:
        return self.first_at is not None

    def mark(self) -> None:
        """Record one real chunk now."""
        now = asyncio.get_running_loop().time()
        if self.first_at is None:
            self.first_at = now
            self._first.set()
        self.last_at = now
        self.chunks += 1

    async def wait_first(self) -> None:
        """Return once the first real chunk has arrived."""
        await self._first.wait()

    def wrap(self, callback: Optional[StreamSink] = None) -> Callable[[str], Any]:
        """The stream callback to hand the brain: marks real chunks and passes
        every chunk on to ``callback`` (sync or async), or nowhere when None."""

        async def _marking(chunk: str) -> None:
            if chunk and chunk != STREAM_RETRY_SENTINEL:
                self.mark()
            if callback is not None:
                result = callback(chunk)
                if inspect.isawaitable(result):
                    await result

        return _marking


class GenerationTimeout(asyncio.TimeoutError):
    """A bound fired and the generation was cancelled.

    Attributes:
        bound: ``first_token`` or ``stall``.
        limit_s: The bound in seconds.
        elapsed_s: Seconds from the start of the wait to the cancellation.
        first_token_s: Seconds from the start to the first chunk, or None.
    """

    def __init__(
        self,
        bound: Bound,
        limit_s: float,
        elapsed_s: float,
        first_token_s: Optional[float] = None,
    ) -> None:
        if bound not in BOUNDS:
            raise ValueError(f"bound must be one of {BOUNDS}, got {bound!r}")
        self.bound = bound
        self.limit_s = float(limit_s)
        self.elapsed_s = float(elapsed_s)
        self.first_token_s = first_token_s
        super().__init__(self.detail())

    def describe(self) -> str:
        """The short phrase a client is shown."""
        if self.bound == "first_token":
            return f"no first token within {self.limit_s:g}s"
        return f"no streamed text for {self.limit_s:g}s after the first token"

    def detail(self) -> str:
        """The phrase plus when the first token came, for the log."""
        if self.first_token_s is None:
            when = "no token streamed"
        else:
            when = f"first token at {self.first_token_s:.1f}s"
        return f"{self.describe()} ({when}, cancelled at {self.elapsed_s:.1f}s)"


def _positive(name: str, value: Any) -> float:
    number = float(value)
    if not number > 0.0 or number != number or number == float("inf"):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


def resolve_bounds(
    *,
    first_token_timeout_s: Optional[float] = None,
    stall_timeout_s: Optional[float] = None,
    heartbeat_s: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> Tuple[float, float, float]:
    """The ``(first_token, stall, heartbeat)`` seconds in force: each argument,
    or its Settings key when None. Every value must be positive."""
    cfg = settings if settings is not None else get_settings()
    first = (
        cfg.GENERATION_TIMEOUT_S
        if first_token_timeout_s is None
        else first_token_timeout_s
    )
    stall = (
        cfg.GENERATION_STALL_TIMEOUT_S if stall_timeout_s is None else stall_timeout_s
    )
    beat = cfg.THINKING_HEARTBEAT_S if heartbeat_s is None else heartbeat_s
    return (
        _positive("first_token_timeout_s", first),
        _positive("stall_timeout_s", stall),
        _positive("heartbeat_s", beat),
    )


async def _cancel(task: "asyncio.Future[Any]") -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001 - the bound is what is reported
        logger.debug("cancelled generation ended with %s", type(exc).__name__)


async def run_with_heartbeat(
    coro_factory: Callable[[FirstTokenMark], Awaitable[T]],
    *,
    on_heartbeat: Optional[HeartbeatFn],
    first_token_timeout_s: Optional[float] = None,
    stall_timeout_s: Optional[float] = None,
    heartbeat_s: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> T:
    """Await ``coro_factory(mark)`` under the first-token and stall bounds.

    Raises:
        GenerationTimeout: a bound elapsed; the operation was cancelled first.
        ValueError: a bound that is not a positive number.
    """
    first_bound, stall_bound, beat = resolve_bounds(
        first_token_timeout_s=first_token_timeout_s,
        stall_timeout_s=stall_timeout_s,
        heartbeat_s=heartbeat_s,
        settings=settings,
    )
    loop = asyncio.get_running_loop()
    mark = FirstTokenMark()
    task: "asyncio.Future[T]" = asyncio.ensure_future(coro_factory(mark))
    start = loop.time()
    last_beat = start
    # The first chunk switches the bound from the first-token deadline to the
    # stall deadline, which may be earlier, so the wait wakes on it.
    first_waiter: "asyncio.Future[None]" = asyncio.ensure_future(mark.wait_first())
    try:
        while True:
            if task.done():
                return task.result()
            now = loop.time()
            if mark.last_at is None:
                deadline = start + first_bound
                bound: Bound = "first_token"
                limit = first_bound
            else:
                deadline = mark.last_at + stall_bound
                bound = "stall"
                limit = stall_bound
            if now >= deadline:
                await _cancel(task)
                first_token_s = (
                    mark.first_at - start if mark.first_at is not None else None
                )
                raise GenerationTimeout(bound, limit, now - start, first_token_s)
            silent_since = max(last_beat, mark.last_at or start)
            next_beat = silent_since + beat
            if now >= next_beat:
                last_beat = now
                if on_heartbeat is not None:
                    try:
                        await on_heartbeat(now - start)
                    except Exception as exc:  # noqa: BLE001 - progress only
                        logger.debug(
                            "heartbeat callback failed: %s", type(exc).__name__
                        )
                continue
            # Wake for the earlier of the deadline and the next heartbeat, or on
            # the first chunk; any later chunk only moves both later, and they
            # are recomputed on waking.
            waiters = {task} if first_waiter.done() else {task, first_waiter}
            await asyncio.wait(
                waiters,
                timeout=max(0.0, min(deadline, next_beat) - now),
                return_when=asyncio.FIRST_COMPLETED,
            )
    finally:
        if not task.done():
            task.cancel()
        if not first_waiter.done():
            first_waiter.cancel()
            try:
                await first_waiter
            except asyncio.CancelledError:
                pass
