"""Throughput and latency measurements for the neural protocol layer.

Marked ``benchmark`` and ``slow``: the timings depend on the machine and are
skipped by default runs (``-m "not slow"``). Run them with
``-m benchmark``. The thresholds are loose floors that catch an order of
magnitude regression, not targets.
"""

import asyncio
import statistics
import time

import numpy as np
import pytest

from mtc.neural.protocols.message_format import (
    MessagePriority,
    NeuralMessage,
    NeuralMessageType,
    SpikeTrain,
)
from mtc.neural.protocols.routing import NeuralRouter, RouteType, RoutingRule
from mtc.neural.protocols.serialization import NeuralSerializer, SerializationFormat
from mtc.neural.protocols.synchronization import (
    ClockReference,
    NeuralSynchronizer,
    PhaseLocker,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

SUBSTRATES = ("snn", "lsm", "htm", "ctm")


def _rate(operation, seconds: float = 0.5) -> float:
    """Calls per second of ``operation`` over a fixed wall-clock window."""
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        operation()
        count += 1
    return count / (time.perf_counter() - start)


def _spike_train(size: int = 100) -> SpikeTrain:
    return SpikeTrain(
        neuron_ids=list(range(size)),
        spike_times=[i * 0.001 for i in range(size)],
        strengths=[0.8] * size,
    )


def test_message_creation_throughput():
    rate = _rate(
        lambda: NeuralMessage(
            source="snn",
            target="htm",
            message_type=NeuralMessageType.SPIKE_TRAIN,
            priority=MessagePriority.NORMAL,
        )
    )
    assert rate > 10_000


def test_spike_train_creation_throughput():
    assert _rate(_spike_train) > 10_000


def test_cached_routing_throughput():
    router = NeuralRouter()
    for src in SUBSTRATES:
        for tgt in SUBSTRATES:
            if src != tgt:
                router.add_rule(
                    RoutingRule(
                        source_pattern=f"{src}*",
                        target_pattern=f"{tgt}*",
                        route_type=RouteType.DIRECT,
                    )
                )
    message = NeuralMessage(
        source="snn_primary",
        target="htm_main",
        message_type=NeuralMessageType.SPIKE_TRAIN,
    )

    assert _rate(lambda: router.get_route(message)) > 100_000


def test_uncached_routing_over_many_rules():
    router = NeuralRouter()
    for i in range(100):
        router.add_rule(
            RoutingRule(
                source_pattern=f"system_{i}", target_pattern=f"system_{(i + 1) % 100}"
            )
        )
    rng = np.random.default_rng(0)
    messages = [
        NeuralMessage(
            source=f"system_{rng.integers(0, 100)}",
            target=f"system_{rng.integers(0, 100)}",
            message_type=NeuralMessageType.STATE_UPDATE,
        )
        for _ in range(1000)
    ]

    start = time.perf_counter()
    for message in messages:
        router.get_route(message)
    assert time.perf_counter() - start < 1.0


def test_clock_resolution_is_sub_millisecond():
    clock = ClockReference()
    readings = [clock.get_current_time() for _ in range(1000)]
    steps = [b - a for a, b in zip(readings, readings[1:]) if b > a]

    assert steps and min(steps) < 0.001


def test_json_round_trip_latency():
    serializer = NeuralSerializer()
    message = NeuralMessage(
        source="snn",
        target="htm",
        message_type=NeuralMessageType.SPIKE_TRAIN,
        payload=_spike_train(),
    )

    latencies_ms = []
    for _ in range(500):
        start = time.perf_counter()
        encoded = serializer.serialize(message, SerializationFormat.JSON)
        serializer.deserialize(encoded, SerializationFormat.JSON)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    assert statistics.mean(latencies_ms) < 1.0


def test_binary_round_trip_latency():
    serializer = NeuralSerializer()
    message = NeuralMessage(
        source="snn",
        target="htm",
        message_type=NeuralMessageType.SPIKE_TRAIN,
        payload=_spike_train(),
    )

    latencies_ms = []
    for _ in range(500):
        start = time.perf_counter()
        encoded = serializer.serialize(message, SerializationFormat.BINARY)
        serializer.deserialize(encoded, SerializationFormat.BINARY)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    assert statistics.mean(latencies_ms) < 1.0


@pytest.mark.asyncio
async def test_concurrent_phase_lockers_complete():
    lockers = [PhaseLocker(target_phase=i * 0.25, lock_strength=0.5) for i in range(4)]

    async def run(locker):
        return [await locker.apply_phase_lock(0.5, 0.5) for _ in range(200)]

    results = await asyncio.gather(*(run(locker) for locker in lockers))

    assert [len(r) for r in results] == [200, 200, 200, 200]
    assert all(results[0])  # target 0.0: identical phases lock


@pytest.mark.asyncio
async def test_synchronizer_update_throughput():
    sync = NeuralSynchronizer()
    for name, offset in (("snn", 0.0), ("lsm", 0.25), ("htm", 0.5)):
        await sync.register_system(name, phase_offset=offset)

    start = time.perf_counter()
    for _ in range(1000):
        await sync.update(dt=0.001)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert 0.0 <= sync.get_sync_state().sync_quality <= 1.0
