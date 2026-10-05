"""Sleep scheduler: state machine, configured clock, sleep-entry work, seams."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from mtc.consciousness import sleep_scheduler as module
from mtc.consciousness.sleep_scheduler import (
    THOUGHT_MODES,
    Consolidator,
    MemoryPruner,
    SleepScheduler,
    SleepSchedulerConfig,
    SleepState,
    ThoughtStreamControl,
)

UTC = timezone.utc
ZONES = ("UTC", "Europe/Berlin", "Asia/Tokyo")


def at(hour, minute=0, zone="UTC", day=1):
    """A fixed instant on 2000-01-01 given as local wall time in ``zone``."""
    local = datetime(2000, 1, day, hour, minute, tzinfo=ZoneInfo(zone))
    return local.astimezone(UTC)


def locale(zone="UTC", wake=6, sleep=22):
    return SimpleNamespace(TIMEZONE=zone, WAKE_HOUR=wake, SLEEP_HOUR=sleep)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, minutes=0.0):
        self.now = self.now + timedelta(minutes=minutes)
        return self.now


class FakeStream:
    def __init__(self):
        self.calls = []

    def set_consciousness_state(self, mode):
        self.calls.append(("mode", mode))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def inject_thought(self, text, thought_type, emotional_tone):
        self.calls.append(("inject", text, thought_type, emotional_tone))


class FakeConsolidator:
    def __init__(self, on_call=None):
        self.calls = 0
        self.on_call = on_call

    async def consolidate(self):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        return {"experiences": 3, "beliefs": 1}


class FakePruner:
    def __init__(self):
        self.calls = []

    async def prune(self, dry_run):
        self.calls.append(dry_run)
        return {"deleted": 0}


def make(clock, zone="UTC", wake=6, sleep=22, **kwargs):
    return SleepScheduler(clock=clock, settings=locale(zone, wake, sleep), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("zone", ZONES)
async def test_night_window_uses_configured_zone_and_hours(zone):
    clock = Clock(at(23, 0, zone))
    scheduler = make(clock, zone)
    clock.advance(5)
    assert await scheduler.tick() is SleepState.DREAMING
    clock.now = at(6, 0, zone, day=2)
    assert await scheduler.tick() is SleepState.AWAKE

    # Custom hours: night from 21:00 to 09:00 local.
    clock.now = at(8, 30, zone, day=3)
    late_waker = make(clock, zone, wake=9, sleep=21)
    clock.advance(5)
    assert await late_waker.tick() is SleepState.DREAMING
    clock.now = at(9, 30, zone, day=3)
    assert await late_waker.tick() is SleepState.AWAKE
    assert late_waker.get_status()["is_night"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zone,expected",
    [
        ("UTC", SleepState.AWAKE),
        ("Europe/Berlin", SleepState.DREAMING),
        ("Asia/Tokyo", SleepState.AWAKE),
    ],
)
async def test_same_instant_reads_differently_per_zone(zone, expected):
    # 21:30 UTC is 22:30 in Berlin (night) and 06:30 the next day in Tokyo.
    clock = Clock(datetime(2000, 1, 1, 21, 25, tzinfo=UTC))
    scheduler = make(clock, zone)
    clock.advance(5)
    assert await scheduler.tick() is expected


@pytest.mark.asyncio
async def test_settings_are_read_at_call_time(monkeypatch):
    current = locale("UTC")
    monkeypatch.setattr(module, "get_settings", lambda: current)
    clock = Clock(datetime(2000, 1, 1, 21, 25, tzinfo=UTC))
    scheduler = SleepScheduler(clock=clock)
    clock.advance(5)
    assert await scheduler.tick() is SleepState.AWAKE
    current = locale("Europe/Berlin")
    assert await scheduler.tick() is SleepState.DREAMING


@pytest.mark.asyncio
async def test_nap_after_idle_and_wake_on_activity():
    clock = Clock(at(12, 0))
    scheduler = make(clock)
    clock.advance(29)
    assert await scheduler.tick() is SleepState.AWAKE
    clock.advance(1)
    assert await scheduler.tick() is SleepState.NAPPING
    assert scheduler.get_status()["nap_remaining_seconds"] == pytest.approx(900.0)
    clock.advance(5)
    scheduler.notify_activity()
    assert scheduler.state is SleepState.AWAKE
    assert scheduler.get_status()["idle_seconds"] == pytest.approx(0.0)

    # A nap ends on its own after nap_minutes.
    clock.advance(30)
    assert await scheduler.tick() is SleepState.NAPPING
    clock.advance(15)
    assert await scheduler.tick() is SleepState.AWAKE


@pytest.mark.asyncio
async def test_night_reentry_after_short_idle():
    clock = Clock(at(23, 0))
    scheduler = make(clock)
    clock.advance(4)
    assert await scheduler.tick() is SleepState.AWAKE
    clock.advance(1)
    assert await scheduler.tick() is SleepState.DREAMING
    clock.advance(5)
    scheduler.notify_activity()
    assert scheduler.state is SleepState.AWAKE
    clock.advance(4)
    assert await scheduler.tick() is SleepState.AWAKE
    clock.advance(1)
    assert await scheduler.tick() is SleepState.DREAMING


@pytest.mark.asyncio
async def test_napping_turns_into_night_sleep():
    clock = Clock(at(21, 20))
    scheduler = make(clock)
    clock.advance(30)
    assert await scheduler.tick() is SleepState.NAPPING
    clock.advance(10)
    assert await scheduler.tick() is SleepState.DREAMING


@pytest.mark.asyncio
async def test_no_thrash_between_states():
    clock = Clock(at(10, 0))
    config = SleepSchedulerConfig()
    assert config.min_awake_minutes_after_nap == 30.0
    scheduler = make(clock, config=config)
    states = []
    for _ in range(121):
        states.append((clock.now, await scheduler.tick()))
        clock.advance(1)
    entries = [
        moment
        for (moment, state), (_, before) in zip(states[1:], states[:-1])
        if state is SleepState.NAPPING and before is SleepState.AWAKE
    ]
    wakes = [
        moment
        for (moment, state), (_, before) in zip(states[1:], states[:-1])
        if state is SleepState.AWAKE and before is SleepState.NAPPING
    ]
    assert entries == [at(10, 30), at(11, 15), at(12, 0)]
    assert wakes == [at(10, 45), at(11, 30)]
    for woke, slept in zip(wakes, entries[1:]):
        assert slept - woke >= timedelta(minutes=config.min_awake_minutes_after_nap)


@pytest.mark.asyncio
async def test_is_resting_matches_state():
    clock = Clock(at(12, 0))
    scheduler = make(clock)
    assert scheduler.is_resting() is False
    clock.advance(30)
    await scheduler.tick()
    assert scheduler.state is SleepState.NAPPING
    assert scheduler.is_resting() is True
    scheduler.notify_activity()
    assert scheduler.is_resting() is False
    clock.now = at(23, 0)
    scheduler.notify_activity()
    clock.advance(5)
    await scheduler.tick()
    assert scheduler.state is SleepState.DREAMING
    assert scheduler.is_resting() is True


@pytest.mark.asyncio
async def test_tick_needs_no_sleeping(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("tick must not sleep")

    monkeypatch.setattr(module.asyncio, "sleep", forbidden)
    clock = Clock(at(9, 0))
    scheduler = make(clock)
    clock.advance(180)
    assert await scheduler.tick() is SleepState.NAPPING
    assert await scheduler.tick(now=at(23, 0)) is SleepState.DREAMING
    assert await scheduler.tick(now=at(7, 0, day=2)) is SleepState.AWAKE
    with pytest.raises(ValueError):
        await scheduler.tick(now=datetime(2000, 1, 2, 8, 0))


@pytest.mark.asyncio
async def test_stop_cancels_cleanly():
    ticked = asyncio.Event()

    class Probe(SleepScheduler):
        async def tick(self, now=None):
            ticked.set()
            return await super().tick(now)

    clock = Clock(at(12, 0))
    scheduler = Probe(
        clock=clock,
        settings=locale(),
        config=SleepSchedulerConfig(tick_seconds=3600.0),
    )
    scheduler.start()
    first = scheduler._task
    scheduler.start()
    assert scheduler._task is first
    assert scheduler.get_status()["running"] is True
    await asyncio.wait_for(ticked.wait(), timeout=5.0)
    await scheduler.stop()
    assert first.done()
    assert scheduler._task is None
    assert scheduler.get_status()["running"] is False
    await scheduler.stop()


@pytest.mark.asyncio
async def test_state_dict_round_trip():
    clock = Clock(at(12, 0))
    stream = FakeStream()
    scheduler = make(clock, thought_stream=stream)
    clock.advance(30)
    await scheduler.tick()
    state = scheduler.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}

    fresh_stream = FakeStream()
    restored = make(Clock(clock.now), thought_stream=fresh_stream)
    restored.from_state_dict(state)
    assert restored.state is SleepState.NAPPING
    assert isinstance(restored._nap_start, datetime)
    assert restored._nap_start.tzinfo is not None
    assert restored._last_activity == scheduler._last_activity
    assert restored.get_status()["skipped_work"] == (
        scheduler.get_status()["skipped_work"]
    )
    assert restored.to_state_dict() == state

    # The next tick re-applies the thought-stream mode for the restored state.
    assert await restored.tick() is SleepState.NAPPING
    assert ("pause",) in fresh_stream.calls


@pytest.mark.asyncio
async def test_consolidation_on_sleep_entry():
    clock = Clock(at(12, 0))
    consolidator = FakeConsolidator()
    scheduler = make(clock, consolidator=consolidator)
    assert isinstance(consolidator, Consolidator)
    clock.advance(30)
    await scheduler.tick()
    assert consolidator.calls == 1
    clock.advance(15)
    await scheduler.tick()
    assert scheduler.state is SleepState.AWAKE
    assert consolidator.calls == 1
    clock.now = at(23, 0)
    await scheduler.tick()
    assert scheduler.state is SleepState.DREAMING
    assert consolidator.calls == 2
    clock.advance(30)
    await scheduler.tick()
    assert consolidator.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
async def test_pruning_only_at_night(dry_run):
    clock = Clock(at(12, 0))
    pruner = FakePruner()
    assert isinstance(pruner, MemoryPruner)
    config = SleepSchedulerConfig(prune_dry_run=dry_run)
    scheduler = make(clock, pruner=pruner, config=config)
    clock.advance(30)
    await scheduler.tick()
    assert scheduler.state is SleepState.NAPPING
    assert pruner.calls == []
    clock.now = at(23, 0)
    await scheduler.tick()
    assert scheduler.state is SleepState.DREAMING
    assert pruner.calls == [dry_run]


def test_reference_prune_choice_is_the_default():
    assert SleepSchedulerConfig().prune_dry_run is False


@pytest.mark.asyncio
async def test_thought_stream_paused_and_resumed_with_mode_names():
    clock = Clock(at(12, 0))
    stream = FakeStream()
    assert isinstance(stream, ThoughtStreamControl)
    scheduler = make(clock, thought_stream=stream)
    clock.advance(30)
    await scheduler.tick()
    assert stream.calls == [("mode", "dreaming"), ("pause",)]
    stream.calls.clear()
    scheduler.notify_activity()
    assert stream.calls == [("mode", "awake"), ("resume",)]
    stream.calls.clear()
    clock.now = at(23, 0)
    await scheduler.tick()
    assert stream.calls == [("mode", "dreaming"), ("pause",)]
    stream.calls.clear()
    clock.now = at(6, 0, day=2)
    await scheduler.tick()
    assert stream.calls == [("mode", "awake"), ("resume",)]
    assert set(THOUGHT_MODES) == {"awake", "drowsy", "dreaming", "deep_sleep"}
    assert set(module.THOUGHT_MODE_BY_STATE.values()) <= set(THOUGHT_MODES)


@pytest.mark.asyncio
async def test_async_thought_stream_methods_are_awaited():
    calls = []

    class AsyncStream:
        def set_consciousness_state(self, mode):
            calls.append(("mode", mode))

        async def pause(self):
            calls.append(("pause",))

        async def resume(self):
            calls.append(("resume",))

        async def inject_thought(self, text, thought_type, emotional_tone):
            calls.append(("inject", thought_type))

    clock = Clock(at(12, 0))
    scheduler = make(clock, thought_stream=AsyncStream())
    clock.advance(30)
    await scheduler.tick()
    assert calls == [("mode", "dreaming"), ("pause",)]
    scheduler.notify_activity()
    await scheduler.drain()
    assert calls[-2:] == [("mode", "awake"), ("resume",)]


@pytest.mark.asyncio
async def test_missing_seams_are_skipped_and_named():
    clock = Clock(at(23, 0))
    scheduler = make(clock)
    status = scheduler.get_status()
    for name in ("brain", "thought_stream", "consolidator", "pruner"):
        assert name in status["missing_seams"]
    assert "dream_store" not in status["missing_seams"]
    assert "dream_engine" not in status["missing_seams"]
    clock.advance(5)
    assert await scheduler.tick() is SleepState.DREAMING
    skipped = scheduler.get_status()["skipped_work"]
    assert skipped["consolidation"] == 1
    assert skipped["pruning"] == 1
    assert skipped["thought_stream"] == 1
    assert skipped["dream"] == 1
    assert scheduler.get_status()["dreams"]["attempted"] == 0


@pytest.mark.asyncio
async def test_activity_during_sleep_entry_stops_remaining_work():
    clock = Clock(at(23, 0))
    pruner = FakePruner()
    holder = {}
    consolidator = FakeConsolidator(on_call=lambda: holder["s"].notify_activity())
    scheduler = make(clock, consolidator=consolidator, pruner=pruner)
    holder["s"] = scheduler
    clock.advance(5)
    assert await scheduler.tick() is SleepState.AWAKE
    assert consolidator.calls == 1
    assert pruner.calls == []


@pytest.mark.asyncio
async def test_seam_failures_are_logged_not_raised():
    class Broken:
        async def consolidate(self):
            raise RuntimeError("store offline")

        async def prune(self, dry_run):
            raise RuntimeError("store offline")

    clock = Clock(at(23, 0))
    scheduler = make(clock, consolidator=Broken(), pruner=Broken())
    clock.advance(5)
    assert await scheduler.tick() is SleepState.DREAMING
    errors = scheduler.get_status()["errors"]
    assert errors["consolidation"] == 1
    assert errors["pruning"] == 1


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        SleepSchedulerConfig(tick_seconds=0)
    with pytest.raises(ValueError):
        SleepSchedulerConfig(nap_minutes=-1)
    with pytest.raises(ValueError):
        SleepSchedulerConfig(gentle_floor_valence_threshold=-2.0)
