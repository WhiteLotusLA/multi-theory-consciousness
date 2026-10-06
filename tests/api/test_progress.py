"""Generation bounds and heartbeats: a first-token bound, a stall bound after
the first token, and heartbeats during silence.

No test sleeps: every operation waits on an event that the runner's own
heartbeat or bound resolves, with bounds of a few hundredths of a second.
"""

import asyncio

import pytest

from mtc.api.progress import (
    FirstTokenMark,
    GenerationTimeout,
    resolve_bounds,
    run_with_heartbeat,
)
from mtc.brain.brain import STREAM_RETRY_SENTINEL
from mtc.core.config import Settings


async def _no_beat(elapsed: float) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_first_token_bound():
    cancelled = asyncio.Event()

    async def never_streams(mark: FirstTokenMark) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    with pytest.raises(GenerationTimeout) as info:
        await run_with_heartbeat(
            never_streams,
            on_heartbeat=_no_beat,
            first_token_timeout_s=0.05,
            stall_timeout_s=30.0,
            heartbeat_s=30.0,
        )
    error = info.value
    assert error.bound == "first_token"
    assert error.limit_s == pytest.approx(0.05)
    assert error.first_token_s is None
    assert error.elapsed_s >= 0.05
    assert isinstance(error, asyncio.TimeoutError)
    assert cancelled.is_set(), "the operation is cancelled before the error"
    assert "first token" in error.describe()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_stall_bound_after_first_token():
    async def streams_then_stalls(mark: FirstTokenMark) -> str:
        send = mark.wrap(None)
        await send("Hello")
        await asyncio.Event().wait()
        return "unreachable"

    with pytest.raises(GenerationTimeout) as info:
        await run_with_heartbeat(
            streams_then_stalls,
            on_heartbeat=_no_beat,
            first_token_timeout_s=30.0,
            stall_timeout_s=0.05,
            heartbeat_s=30.0,
        )
    error = info.value
    assert error.bound == "stall"
    assert error.limit_s == pytest.approx(0.05)
    assert error.first_token_s is not None
    # The first chunk wakes the wait: the stall bound fires soon after it, not
    # at the (much later) first-token deadline.
    assert error.elapsed_s < 5.0
    assert "after the first token" in error.describe()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_slow_first_token_is_not_cut_by_the_stall_bound():
    """Waiting longer than the stall bound before the first token is fine:
    only the first-token bound applies until a chunk arrives."""
    beats = []
    release = asyncio.Event()

    async def beat(elapsed: float) -> None:
        beats.append(elapsed)
        if len(beats) >= 4:
            release.set()

    async def slow_start(mark: FirstTokenMark) -> str:
        send = mark.wrap(None)
        await release.wait()
        await send("late but fine")
        return "answered"

    result = await run_with_heartbeat(
        slow_start,
        on_heartbeat=beat,
        first_token_timeout_s=30.0,
        stall_timeout_s=0.03,
        heartbeat_s=0.02,
    )
    assert result == "answered"
    assert beats[-1] > 0.03, "the wait outlasted the stall bound"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_heartbeat_every_interval():
    beats = []
    release = asyncio.Event()

    async def beat(elapsed: float) -> None:
        beats.append(elapsed)
        if len(beats) == 3:
            release.set()

    async def waits(mark: FirstTokenMark) -> str:
        await release.wait()
        return "done"

    result = await run_with_heartbeat(
        waits,
        on_heartbeat=beat,
        first_token_timeout_s=30.0,
        stall_timeout_s=30.0,
        heartbeat_s=0.02,
    )
    assert result == "done"
    assert len(beats) == 3
    assert beats == sorted(beats)
    for index, elapsed in enumerate(beats, start=1):
        assert elapsed >= 0.02 * index * 0.9


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_failed_heartbeat_never_aborts_the_work():
    release = asyncio.Event()
    calls = []

    async def broken_beat(elapsed: float) -> None:
        calls.append(elapsed)
        if len(calls) >= 2:
            release.set()
        raise RuntimeError("the client went away")

    async def waits(mark: FirstTokenMark) -> str:
        await release.wait()
        return "done"

    assert (
        await run_with_heartbeat(
            waits,
            on_heartbeat=broken_beat,
            first_token_timeout_s=30.0,
            stall_timeout_s=30.0,
            heartbeat_s=0.02,
        )
        == "done"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_operation_error_propagates_unchanged():
    async def fails(mark: FirstTokenMark) -> str:
        raise ValueError("inner failure")

    with pytest.raises(ValueError, match="inner failure"):
        await run_with_heartbeat(
            fails,
            on_heartbeat=_no_beat,
            first_token_timeout_s=30.0,
            stall_timeout_s=30.0,
            heartbeat_s=30.0,
        )


@pytest.mark.asyncio
async def test_mark_forwards_every_chunk_and_skips_the_retry_sentinel():
    seen = []

    async def sink(chunk: str) -> None:
        seen.append(chunk)

    mark = FirstTokenMark()
    send = mark.wrap(sink)
    await send("")
    await send(STREAM_RETRY_SENTINEL)
    assert not mark.is_set, "neither an empty chunk nor the sentinel is a token"
    await send("first")
    first = mark.first_at
    await send("second")
    assert mark.is_set and mark.first_at == first
    assert mark.last_at >= first
    assert seen == ["", STREAM_RETRY_SENTINEL, "first", "second"]


@pytest.mark.asyncio
async def test_mark_accepts_a_sync_sink():
    seen = []
    mark = FirstTokenMark()
    await mark.wrap(seen.append)("chunk")
    assert seen == ["chunk"] and mark.is_set


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_no_task_is_left_after_a_timeout():
    before = len(asyncio.all_tasks())

    async def never(mark: FirstTokenMark) -> None:
        await asyncio.Event().wait()

    with pytest.raises(GenerationTimeout):
        await run_with_heartbeat(
            never,
            on_heartbeat=_no_beat,
            first_token_timeout_s=0.03,
            stall_timeout_s=30.0,
            heartbeat_s=30.0,
        )
    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) == before


def test_bounds_from_settings():
    cfg = Settings(
        _env_file=None,
        GENERATION_TIMEOUT_S=120.0,
        GENERATION_STALL_TIMEOUT_S=45.0,
        THINKING_HEARTBEAT_S=7.5,
    )
    assert resolve_bounds(settings=cfg) == (120.0, 45.0, 7.5)
    # An explicit value wins over the setting.
    assert resolve_bounds(
        first_token_timeout_s=3.0, stall_timeout_s=2.0, heartbeat_s=1.0, settings=cfg
    ) == (3.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="first_token_timeout_s"):
        resolve_bounds(first_token_timeout_s=0.0, settings=cfg)
    with pytest.raises(ValueError, match="stall_timeout_s"):
        resolve_bounds(stall_timeout_s=-1.0, settings=cfg)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_run_reads_its_bounds_from_settings():
    cfg = Settings(
        _env_file=None,
        GENERATION_TIMEOUT_S=0.05,
        GENERATION_STALL_TIMEOUT_S=30.0,
        THINKING_HEARTBEAT_S=30.0,
    )

    async def never(mark: FirstTokenMark) -> None:
        await asyncio.Event().wait()

    with pytest.raises(GenerationTimeout) as info:
        await run_with_heartbeat(never, on_heartbeat=_no_beat, settings=cfg)
    assert info.value.bound == "first_token"
    assert info.value.limit_s == pytest.approx(0.05)
