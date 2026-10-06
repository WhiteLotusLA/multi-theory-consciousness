"""Stage profiler: sync and async stages, percentiles, honest empties, bounds."""

import asyncio

import pytest

from mtc.core.profiler import StageProfiler


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_sync_and_async_stages_recorded():
    clock = _Clock()
    profiler = StageProfiler(clock=clock)
    with profiler.stage("sync"):
        clock.advance(0.010)

    async def run():
        async with profiler.stage("async"):
            clock.advance(0.020)

    asyncio.run(run())
    summary = profiler.summary()
    assert summary["sync"]["count"] == 1
    assert summary["sync"]["mean"] == pytest.approx(10.0)
    assert summary["async"]["count"] == 1
    assert summary["async"]["max"] == pytest.approx(20.0)


def test_summary_percentiles():
    clock = _Clock()
    profiler = StageProfiler(clock=clock)
    for ms in range(1, 101):
        with profiler.stage("cycle"):
            clock.advance(ms / 1000.0)
    stats = profiler.summary()["cycle"]
    assert stats["count"] == 100
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["p50"] == pytest.approx(50.0)
    assert stats["p95"] == pytest.approx(95.0)
    assert stats["max"] == pytest.approx(100.0)


def test_empty_stage_is_none_not_zero():
    clock = _Clock()
    profiler = StageProfiler(clock=clock)
    with profiler.stage("once"):
        clock.advance(0.005)
    profiler.reset()
    stats = profiler.summary()["once"]
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["p50"] is None
    assert stats["p95"] is None
    assert stats["max"] is None
    assert profiler.summary(["never"])["never"]["mean"] is None


def test_bounded_samples():
    clock = _Clock()
    profiler = StageProfiler(max_samples=5, clock=clock)
    for ms in range(1, 11):
        with profiler.stage("bounded"):
            clock.advance(ms / 1000.0)
    stats = profiler.summary()["bounded"]
    assert stats["count"] == 5
    # Only the five most recent samples (6 to 10 ms) are kept.
    assert stats["mean"] == pytest.approx(8.0)
    assert stats["max"] == pytest.approx(10.0)


def test_failing_stage_still_recorded():
    clock = _Clock()
    profiler = StageProfiler(clock=clock)
    with pytest.raises(RuntimeError):
        with profiler.stage("failing"):
            clock.advance(0.003)
            raise RuntimeError("boom")
    assert profiler.summary()["failing"]["count"] == 1
    with pytest.raises(ValueError):
        StageProfiler(max_samples=0)
