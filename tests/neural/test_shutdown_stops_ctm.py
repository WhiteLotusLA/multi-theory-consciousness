"""Shutdown must really stop the continuous thought loop, tolerate a partly
built orchestrator, run without torch, and close sync or async data managers."""

import asyncio
import gc
import sys
import warnings
from types import SimpleNamespace

from mtc.neural.orchestrator import NeuralOrchestrator


class _FakeThoughtMachine:
    def __init__(self):
        self.stopped = False
        self.running = True

    async def stop(self):
        await asyncio.sleep(0)
        self.running = False
        self.stopped = True


def _bare_orchestrator(machine=None, manager=None):
    orchestrator = NeuralOrchestrator.__new__(NeuralOrchestrator)
    orchestrator.continuous_thought_machine = machine
    orchestrator.experience_buffer = []
    orchestrator.neural_data_manager = manager
    orchestrator.state = SimpleNamespace(is_awake=True)
    return orchestrator


def test_shutdown_awaits_ctm_stop():
    machine = _FakeThoughtMachine()
    orchestrator = _bare_orchestrator(machine)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(orchestrator.shutdown())
        gc.collect()

    assert machine.stopped is True
    assert machine.running is False
    assert orchestrator.continuous_thought_machine is None
    assert orchestrator.state.is_awake is False
    assert not [w for w in caught if "never awaited" in str(w.message)]


def test_shutdown_tolerates_partially_built_instance():
    orchestrator = NeuralOrchestrator.__new__(NeuralOrchestrator)
    asyncio.run(orchestrator.shutdown())


def test_shutdown_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    machine = _FakeThoughtMachine()
    orchestrator = _bare_orchestrator(machine)
    asyncio.run(orchestrator.shutdown())
    assert machine.stopped is True
    assert orchestrator.state.is_awake is False


class _SyncManager:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _AsyncManager:
    def __init__(self):
        self.closed = False

    async def close(self):
        await asyncio.sleep(0)
        self.closed = True


def test_close_handles_sync_and_async_managers():
    sync_manager = _SyncManager()
    asyncio.run(_bare_orchestrator(manager=sync_manager).shutdown())
    assert sync_manager.closed is True

    async_manager = _AsyncManager()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_bare_orchestrator(manager=async_manager).shutdown())
        gc.collect()
    assert async_manager.closed is True
    assert not [w for w in caught if "never awaited" in str(w.message)]
