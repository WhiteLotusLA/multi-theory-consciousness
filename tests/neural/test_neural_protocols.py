"""End-to-end checks across the neural protocol layer: build a message, route
it, serialize it in each format, and keep the substrates' phases in step."""

import asyncio
import json
from datetime import timezone

import numpy as np
import pytest

from mtc.neural.protocols import (
    ClockReference,
    MessagePriority,
    NeuralMessage,
    NeuralMessageType,
    NeuralRouter,
    NeuralSerializer,
    NeuralSynchronizer,
    PhaseLocker,
    RouteType,
    RoutingRule,
    SerializationFormat,
    SpikeTrain,
)


def _spike_message() -> NeuralMessage:
    return NeuralMessage(
        source="snn_primary",
        target="ctm",
        message_type=NeuralMessageType.SPIKE_TRAIN,
        priority=MessagePriority.HIGH,
        payload=SpikeTrain(
            neuron_ids=list(range(100)),
            spike_times=[0.001 * i for i in range(100)],
            strengths=[0.5 + 0.5 * float(np.sin(i / 10)) for i in range(100)],
        ),
        metadata={"processing_time": 0.5, "confidence": 0.95},
    )


def test_package_exports_the_protocol_surface():
    import mtc.neural.protocols as protocols

    for name in (
        "NeuralMessage",
        "NeuralRouter",
        "NeuralSynchronizer",
        "NeuralSerializer",
        "SyncState",
    ):
        assert hasattr(protocols, name), name


def test_routed_message_survives_both_formats():
    router = NeuralRouter()
    router.add_rule(
        RoutingRule(
            source_pattern="snn*",
            target_pattern="ctm",
            route_type=RouteType.DIRECT,
            priority_threshold=MessagePriority.NORMAL,
        )
    )
    serializer = NeuralSerializer()
    message = _spike_message()

    rule = router.get_route(message)
    assert rule is not None and rule.route_type == RouteType.DIRECT

    for fmt in (SerializationFormat.JSON, SerializationFormat.BINARY):
        encoded = serializer.serialize(message, fmt)
        decoded = serializer.deserialize(encoded, fmt)
        assert decoded.source == message.source
        assert decoded.target == message.target
        assert decoded.priority == message.priority
        assert decoded.payload.neuron_ids == message.payload.neuron_ids
        assert decoded.timestamp == message.timestamp
        assert decoded.timestamp.tzinfo is not None
        assert decoded.timestamp.utcoffset() == timezone.utc.utcoffset(None)
        # The decoded message routes exactly like the original
        assert router.get_route(decoded) is rule

    parsed = json.loads(serializer.serialize(message, SerializationFormat.JSON))
    assert parsed["metadata"] == {"processing_time": 0.5, "confidence": 0.95}


def test_broadcast_rule_lists_its_target():
    router = NeuralRouter()
    router.add_rule(
        RoutingRule(
            source_pattern="lsm",
            target_pattern="htm",
            route_type=RouteType.BROADCAST,
            transform_func=lambda m: m,
        )
    )
    message = NeuralMessage(
        source="lsm", target="htm", message_type=NeuralMessageType.CREATIVE_OUTPUT
    )

    assert router.get_route(message).route_type == RouteType.BROADCAST
    assert router.route_message(message) == ["htm"]


@pytest.mark.asyncio
async def test_synchronizer_and_phase_locker_together():
    sync = NeuralSynchronizer(clock=ClockReference(base_frequency=40.0))
    for name, offset in (("snn", 0.0), ("lsm", 0.25), ("htm", 0.5)):
        await sync.register_system(name, phase_offset=offset)

    for _ in range(10):
        await sync.update(dt=0.001)
    phases = sync.get_current_phases()
    assert set(phases) == {"snn", "lsm", "htm"}

    locker = PhaseLocker(target_phase=0.25, lock_strength=0.8)
    locked = await locker.apply_phase_lock(phases["snn"], phases["lsm"])
    assert isinstance(locked, bool)
    assert 0.0 <= locker.get_phase_coherence() <= 1.0


@pytest.mark.asyncio
async def test_clock_ticks_while_awaiting():
    clock = ClockReference(base_frequency=40.0)
    start = clock.get_current_time()

    await asyncio.sleep(0.1)

    assert clock.get_current_time() > start
    assert clock.get_tick_count() >= 3
    assert 0.0 <= clock.get_phase() < 1.0
