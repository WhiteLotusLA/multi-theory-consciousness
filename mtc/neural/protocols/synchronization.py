"""
Neural Synchronization -- Phase-Locked Substrate Coordination
================================================================

Implements phase-locked loop synchronization across SNN, LSM, and
HTM substrates. In biological brains, neural synchronization at
gamma frequencies (~40Hz) is strongly correlated with conscious
awareness. This module provides:

- ClockReference: global neural clock at configurable frequency
- PhaseLocker: pairwise phase-locking between systems
- NeuralSynchronizer: multi-system coordination
- SyncState: observable synchronization snapshot

Created: March 7, 2026
Authors: Multi-Theory Consciousness Contributors
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import time
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyncState:
    """
    Snapshot of synchronization state across all registered systems.

    phase_differences: pairwise phase offsets (e.g., "SNN-LSM": 0.25)
    sync_errors: pairwise error magnitudes
    sync_quality: 0.0 (desynchronized) to 1.0 (perfectly locked)
    """

    timestamp: datetime
    phase_differences: Dict[str, float]
    sync_errors: Dict[str, float]
    is_synchronized: bool = False
    sync_quality: float = 0.0

    def update_sync_status(self) -> None:
        """Recompute sync quality and status from current errors."""
        if not self.sync_errors:
            self.sync_quality = 0.0
            self.is_synchronized = False
            return

        mean_error = sum(self.sync_errors.values()) / len(self.sync_errors)
        # Quality decays exponentially with error
        self.sync_quality = max(0.0, math.exp(-mean_error * 10.0))
        self.is_synchronized = self.sync_quality > 0.7 and all(
            e < 0.1 for e in self.sync_errors.values()
        )


class ClockReference:
    """
    Global neural clock -- the metronome for substrate coordination.

    Runs at a configurable base frequency (default 40Hz, gamma band).
    All substrates reference this clock for phase calculations.
    """

    def __init__(self, base_frequency: float = 40.0):
        self.frequency = base_frequency
        self._start_time = time.monotonic()
        self._tick_count = 0
        self._last_tick_time = self._start_time

    def get_current_time(self) -> float:
        """Elapsed seconds since clock start."""
        return time.monotonic() - self._start_time

    def get_tick_count(self) -> int:
        """Total ticks elapsed at current frequency."""
        elapsed = self.get_current_time()
        return int(elapsed * self.frequency)

    def get_phase(self) -> float:
        """Current phase within the cycle, 0.0 to 1.0."""
        elapsed = self.get_current_time()
        cycle_position = (elapsed * self.frequency) % 1.0
        return cycle_position

    def adjust_frequency(self, new_frequency: float) -> None:
        """Smoothly adjust the clock frequency."""
        self.frequency = new_frequency


class PhaseLocker:
    """
    Pairwise phase-locking mechanism.

    Computes whether two systems are phase-locked (oscillating in
    a stable phase relationship) and applies corrective coupling
    to bring them into alignment.
    """

    def __init__(
        self,
        target_phase: float = 0.0,
        lock_strength: float = 0.5,
    ):
        self.target_phase = target_phase
        self.lock_strength = lock_strength
        self._phase_history: list = []

    async def apply_phase_lock(self, phase_a: float, phase_b: float) -> bool:
        """
        Attempt to lock phase_b to phase_a + target_phase.

        Returns True if phase error is within tolerance.
        """
        desired = (phase_a + self.target_phase) % 1.0
        error = abs(desired - phase_b)
        # Wrap-around handling
        if error > 0.5:
            error = 1.0 - error

        self._phase_history.append(error)
        if len(self._phase_history) > 100:
            self._phase_history = self._phase_history[-100:]

        return error < 0.1  # locked if < 10% of cycle

    def get_phase_coherence(self) -> float:
        """Average phase error over recent history."""
        if not self._phase_history:
            return 0.0
        mean_error = sum(self._phase_history) / len(self._phase_history)
        return max(0.0, 1.0 - mean_error * 2.0)


class NeuralSynchronizer:
    """
    Multi-system synchronization coordinator.

    Manages phase relationships between all registered neural
    substrates. Implements a simplified phase-locked loop:
    each update step nudges phases toward their target offsets.
    """

    def __init__(self, clock: Optional[ClockReference] = None):
        self._clock = clock or ClockReference(base_frequency=40.0)
        self._systems: Dict[str, float] = {}  # system_id -> phase_offset
        self._phases: Dict[str, float] = {}  # system_id -> current_phase
        self._coupling_strength: float = 0.1
        self._sync_state: Optional[SyncState] = None

    async def register_system(self, system_id: str, phase_offset: float = 0.0) -> None:
        """Register a neural system with its target phase offset."""
        self._systems[system_id] = phase_offset
        # Start at a random phase -- systems must converge via coupling
        import random

        self._phases[system_id] = random.random()
        logger.debug(f"Registered {system_id} with phase offset {phase_offset:.2f}")

    async def unregister_system(self, system_id: str) -> None:
        """Remove a system from synchronization."""
        self._systems.pop(system_id, None)
        self._phases.pop(system_id, None)

    async def update(self, dt: float = 0.001) -> None:
        """
        Advance synchronization by dt seconds.

        Each system's phase advances by (frequency * dt) plus a
        coupling term that pulls it toward its target offset
        relative to the global clock.
        """
        global_phase = self._clock.get_phase()

        for sys_id, target_offset in self._systems.items():
            current = self._phases.get(sys_id, 0.0)
            target = (global_phase + target_offset) % 1.0

            # Phase error (with wrap-around)
            error = target - current
            if error > 0.5:
                error -= 1.0
            elif error < -0.5:
                error += 1.0

            # Coupling nudge
            correction = error * self._coupling_strength
            new_phase = (current + self._clock.frequency * dt + correction) % 1.0
            self._phases[sys_id] = new_phase

    def get_sync_state(self) -> SyncState:
        """Build a SyncState snapshot from current phases."""
        phase_diffs = {}
        sync_errors = {}

        system_ids = list(self._systems.keys())
        for i in range(len(system_ids)):
            for j in range(i + 1, len(system_ids)):
                a, b = system_ids[i], system_ids[j]
                key = f"{a}-{b}"
                diff = abs(self._phases.get(a, 0) - self._phases.get(b, 0))
                if diff > 0.5:
                    diff = 1.0 - diff
                phase_diffs[key] = diff

                # Error = difference from expected offset
                expected = abs(self._systems[a] - self._systems[b])
                if expected > 0.5:
                    expected = 1.0 - expected
                sync_errors[key] = abs(diff - expected)

        state = SyncState(
            timestamp=datetime.utcnow(),
            phase_differences=phase_diffs,
            sync_errors=sync_errors,
        )
        state.update_sync_status()
        self._sync_state = state
        return state

    def get_current_phases(self) -> Dict[str, float]:
        """Return current phase for each registered system."""
        return dict(self._phases)

    def set_coupling_strength(self, strength: float) -> None:
        """Adjust how strongly systems are pulled toward sync."""
        self._coupling_strength = max(0.0, min(1.0, strength))
