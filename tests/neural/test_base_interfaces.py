"""Tests for the neural base interfaces: configs, state snapshots, the
abstract system contract and the registry."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from mtc.neural.base_interfaces import (
    CTMState,
    HTMState,
    LSMState,
    NeuralState,
    NeuralSystem,
    NeuralSystemConfig,
    NeuralSystemRegistry,
    NeuralSystemType,
    ResourceAllocation,
    SNNState,
    SystemHealth,
)
from mtc.neural.protocols.message_format import (
    MessagePriority,
    NeuralMessage,
    NeuralMessageType,
)


class StubSystem(NeuralSystem[SNNState]):
    """Minimal concrete system that records what the contract asks of it."""

    def __init__(self, config: NeuralSystemConfig):
        super().__init__(config)
        self._state = SNNState(system_id=config.system_id, neuron_count=128)
        self.saved_to: Optional[str] = None

    async def initialize(self) -> None:
        self.health = SystemHealth.HEALTHY

    async def shutdown(self) -> None:
        self.health = SystemHealth.OFFLINE

    async def process_message(self, message: NeuralMessage) -> Optional[NeuralMessage]:
        self._message_count += 1
        return None

    async def update(self, dt: float) -> None:
        self._state.timestamp = datetime.now(timezone.utc)
        self._state.sequence_number += 1

    def get_state(self) -> SNNState:
        return self._state

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": self.health.value,
            "message_count": self._message_count,
            "memory_usage_gb": self._state.get_memory_usage(),
        }

    async def save_state(self, path: Optional[str] = None) -> None:
        self.saved_to = path or self.config.state_save_path

    async def load_state(self, path: Optional[str] = None) -> None:
        return None


def _config(system_id: str = "snn_a", system_type=NeuralSystemType.SNN):
    return NeuralSystemConfig(
        system_type=system_type,
        system_id=system_id,
        resources=ResourceAllocation(memory_gb=2.0, cpu_cores=2),
    )


def test_config_defaults():
    config = NeuralSystemConfig(system_type=NeuralSystemType.LSM, system_id="lsm_a")

    assert config.resources == ResourceAllocation()
    assert config.neural_params == {}
    assert config.state_save_path is None
    assert config.max_throughput_msgs_sec == 1000.0


def test_resource_allocation_fields():
    allocation = ResourceAllocation(
        memory_gb=4.0,
        cpu_cores=4,
        gpu_memory_gb=2.0,
        priority=1,
        max_memory_gb=6.0,
        min_memory_gb=1.0,
        burst_memory_gb=8.0,
    )

    assert allocation.min_memory_gb < allocation.memory_gb < allocation.max_memory_gb
    assert allocation.burst_memory_gb > allocation.max_memory_gb
    assert allocation.priority == 1


def test_state_timestamp_is_aware_utc():
    before = datetime.now(timezone.utc)
    state = SNNState(system_id="snn_a")
    after = datetime.now(timezone.utc)

    assert state.timestamp.tzinfo is not None
    assert state.timestamp.utcoffset().total_seconds() == 0
    assert before <= state.timestamp <= after


def test_state_snapshots_and_memory_estimates():
    snn = SNNState(system_id="snn_a", neuron_count=1000)
    lsm = LSMState(system_id="lsm_a", reservoir_size=500)
    htm = HTMState(system_id="htm_a", column_count=2048, cell_count=65536)
    ctm = CTMState(system_id="ctm_a", thoughts_generated=3)

    assert snn.get_memory_usage() > 0
    assert lsm.get_memory_usage() == pytest.approx(500**2 * 8 / 1024**3)
    assert htm.get_memory_usage() == pytest.approx(65536 * 32 / 1024**3)
    assert ctm.get_memory_usage() == pytest.approx(0.1)
    assert ctm.thoughts_generated == 3


def test_neural_state_is_abstract():
    with pytest.raises(TypeError):
        NeuralState(system_id="x")  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_system_lifecycle_and_message_count():
    system = StubSystem(_config())
    assert system.system_id == "snn_a"
    assert system.system_type == NeuralSystemType.SNN
    assert system.health == SystemHealth.INITIALIZING

    await system.initialize()
    assert system.health == SystemHealth.HEALTHY

    message = NeuralMessage(
        source="orchestrator",
        target="snn_a",
        message_type=NeuralMessageType.SPIKE_TRAIN,
        priority=MessagePriority.HIGH,
    )
    assert await system.process_message(message) is None
    assert system.get_health_status()["message_count"] == 1

    system.health = SystemHealth.DEGRADED
    assert system.get_health_status()["status"] == "degraded"

    await system.shutdown()
    assert system.health == SystemHealth.OFFLINE


@pytest.mark.asyncio
async def test_save_state_uses_configured_path(tmp_path):
    config = _config()
    config.state_save_path = str(tmp_path / "snn_state")
    system = StubSystem(config)

    await system.save_state()
    await system.load_state()

    assert system.saved_to == str(tmp_path / "snn_state")


@pytest.mark.asyncio
async def test_concurrent_updates_advance_each_system():
    systems = [StubSystem(_config(f"snn_{i}")) for i in range(3)]
    await asyncio.gather(*(s.initialize() for s in systems))

    await asyncio.gather(*(s.update(0.001) for _ in range(10) for s in systems))

    for system in systems:
        assert system.get_state().sequence_number == 10
        assert system.get_state().timestamp.tzinfo is not None


def test_registry_lookup_by_id_and_type():
    registry = NeuralSystemRegistry()
    snn = StubSystem(_config("snn_a"))
    lsm = StubSystem(_config("lsm_a", NeuralSystemType.LSM))
    registry.register(snn)
    registry.register(lsm)

    assert registry.get_system("snn_a") is snn
    assert registry.get_systems_by_type(NeuralSystemType.SNN) == [snn]
    assert registry.get_systems_by_type(NeuralSystemType.HTM) == []
    assert len(registry.get_all_systems()) == 2
    assert registry.get_health_summary() == {
        "snn_a": "initializing",
        "lsm_a": "initializing",
    }

    registry.unregister("snn_a")
    registry.unregister("missing")
    assert registry.get_system("snn_a") is None
    assert len(registry.get_all_systems()) == 1
