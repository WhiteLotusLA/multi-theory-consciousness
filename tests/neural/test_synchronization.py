"""Tests for the neural clock, pairwise phase locking and the multi-system
synchronizer."""

import time
from datetime import datetime, timezone

import pytest

from mtc.neural.protocols.synchronization import (
    ClockReference,
    NeuralSynchronizer,
    PhaseLocker,
    SyncState,
)


class TestClockReference:
    def test_default_and_custom_frequency(self):
        assert ClockReference().frequency == 40.0
        assert ClockReference(base_frequency=60.0).frequency == 60.0

    def test_time_advances(self):
        clock = ClockReference()
        first = clock.get_current_time()
        time.sleep(0.01)

        assert clock.get_current_time() > first

    def test_ticks_follow_frequency(self):
        clock = ClockReference(base_frequency=40.0)
        time.sleep(0.06)

        assert clock.get_tick_count() >= 2

    def test_phase_in_unit_interval(self):
        clock = ClockReference()

        assert 0.0 <= clock.get_phase() < 1.0

    def test_adjust_frequency(self):
        clock = ClockReference()
        clock.adjust_frequency(45.0)

        assert clock.frequency == 45.0


class TestPhaseLocker:
    def test_construction(self):
        locker = PhaseLocker(target_phase=0.25, lock_strength=0.8)

        assert locker.target_phase == 0.25
        assert locker.lock_strength == 0.8

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "target, phase_a, phase_b, locked",
        [
            (0.0, 0.5, 0.5, True),
            (0.0, 0.0, 0.5, False),
            (0.25, 0.5, 0.75, True),
            (0.0, 0.98, 0.02, True),  # wraps around the cycle
        ],
    )
    async def test_lock_decision(self, target, phase_a, phase_b, locked):
        locker = PhaseLocker(target_phase=target)

        assert await locker.apply_phase_lock(phase_a, phase_b) is locked

    def test_coherence_without_history_is_zero(self):
        assert PhaseLocker().get_phase_coherence() == 0.0

    @pytest.mark.asyncio
    async def test_coherence_tracks_error_history(self):
        aligned = PhaseLocker()
        misaligned = PhaseLocker()
        for _ in range(30):
            await aligned.apply_phase_lock(0.4, 0.4)
            await misaligned.apply_phase_lock(0.0, 0.5)

        assert aligned.get_phase_coherence() > 0.9
        assert misaligned.get_phase_coherence() < 0.5

    @pytest.mark.asyncio
    async def test_history_is_bounded(self):
        locker = PhaseLocker()
        for _ in range(250):
            await locker.apply_phase_lock(0.0, 0.0)

        assert len(locker._phase_history) == 100


class TestSyncState:
    def _state(self, errors):
        return SyncState(
            timestamp=datetime.now(timezone.utc),
            phase_differences={k: 0.0 for k in errors},
            sync_errors=dict(errors),
        )

    def test_defaults(self):
        state = self._state({"a-b": 0.05})

        assert state.is_synchronized is False
        assert state.sync_quality == 0.0

    def test_low_errors_synchronize(self):
        state = self._state({"a-b": 0.01, "a-c": 0.02})
        state.update_sync_status()

        assert state.sync_quality > 0.7
        assert state.is_synchronized is True

    def test_one_large_error_breaks_synchronization(self):
        state = self._state({"a-b": 0.01, "a-c": 0.15})
        state.update_sync_status()

        assert state.is_synchronized is False

    def test_high_errors_degrade_quality(self):
        state = self._state({"a-b": 0.5})
        state.update_sync_status()

        assert state.sync_quality < 0.1
        assert state.is_synchronized is False

    def test_no_errors_is_not_synchronized(self):
        state = self._state({})
        state.update_sync_status()

        assert state.sync_quality == 0.0
        assert state.is_synchronized is False


class TestNeuralSynchronizer:
    @pytest.mark.asyncio
    async def test_register_and_unregister(self):
        sync = NeuralSynchronizer()
        assert sync.get_current_phases() == {}

        await sync.register_system("snn", phase_offset=0.0)
        await sync.register_system("lsm", phase_offset=0.25)
        assert set(sync.get_current_phases()) == {"snn", "lsm"}
        for phase in sync.get_current_phases().values():
            assert 0.0 <= phase < 1.0

        await sync.unregister_system("snn")
        assert set(sync.get_current_phases()) == {"lsm"}

    @pytest.mark.asyncio
    async def test_update_advances_phases(self):
        sync = NeuralSynchronizer()
        await sync.register_system("snn")
        await sync.register_system("lsm", phase_offset=0.25)
        before = sync.get_current_phases()

        for _ in range(10):
            await sync.update(dt=0.001)

        after = sync.get_current_phases()
        assert after["snn"] != before["snn"]
        assert after["lsm"] != before["lsm"]

    @pytest.mark.asyncio
    async def test_sync_state_covers_every_pair(self):
        sync = NeuralSynchronizer()
        for name, offset in (("snn", 0.0), ("lsm", 0.25), ("htm", 0.5)):
            await sync.register_system(name, phase_offset=offset)
        for _ in range(50):
            await sync.update(dt=0.001)

        state = sync.get_sync_state()

        assert isinstance(state, SyncState)
        assert set(state.phase_differences) == {"snn-lsm", "snn-htm", "lsm-htm"}
        assert set(state.sync_errors) == set(state.phase_differences)
        assert state.timestamp.tzinfo is not None
        assert state.timestamp.utcoffset().total_seconds() == 0
        assert 0.0 <= state.sync_quality <= 1.0

    def test_coupling_strength_is_clamped(self):
        sync = NeuralSynchronizer()

        sync.set_coupling_strength(0.8)
        assert sync._coupling_strength == 0.8
        sync.set_coupling_strength(1.5)
        assert sync._coupling_strength == 1.0
        sync.set_coupling_strength(-0.5)
        assert sync._coupling_strength == 0.0

    @pytest.mark.asyncio
    async def test_strong_coupling_converges(self):
        sync = NeuralSynchronizer()
        sync.set_coupling_strength(0.5)
        await sync.register_system("snn", phase_offset=0.0)
        await sync.register_system("lsm", phase_offset=0.0)

        for _ in range(500):
            await sync.update(dt=0.001)

        state = sync.get_sync_state()
        assert state.sync_errors["snn-lsm"] < 0.05
        assert state.sync_quality > 0.5
