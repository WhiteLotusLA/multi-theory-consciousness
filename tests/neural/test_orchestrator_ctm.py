"""The neural orchestrator and the continuous thought machine: the guarded
import resolves, construction builds no machine, an attached machine feeds the
measurement read, the controls delegate, CTM_ENABLED is honoured, shutdown
awaits the stop, and the machine sees the emotion bridge once it is built."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mtc.consciousness.continuous as package
import mtc.neural.orchestrator as orch_mod
from mtc.consciousness.continuous import BridgeEmotionSource
from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryThoughtGistStore
from mtc.neural.orchestrator import NeuralOrchestrator
from tests.consciousness.continuous.test_machine_cadence import (
    Clock,
    make_machine,
    run,
)

ROOT = Path(__file__).resolve().parents[2]


def _settings(tmp_path, **values) -> Settings:
    values.setdefault("CTM_CONFIG_DIR", str(ROOT / "config" / "ctm"))
    values.setdefault("CTM_STORE", "memory")
    return Settings(PROJECT_ROOT=str(tmp_path), DEBUG=False, **values)


def _orchestrator(**kwargs) -> NeuralOrchestrator:
    return NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False, **kwargs)


def test_guarded_import_succeeds_and_flag_true():
    assert orch_mod._HAS_CTM is True
    assert orch_mod.ContinuousThoughtMachine is package.ContinuousThoughtMachine
    assert orch_mod.ThoughtGenerator is package.ThoughtGenerator
    assert orch_mod.ThoughtStream is package.ThoughtStream
    assert _orchestrator().available_components()["continuous_thought"] is True


def test_constructing_orchestrator_builds_no_machine(monkeypatch):
    def refuse(**_kwargs):
        raise AssertionError("construction must not build a machine")

    monkeypatch.setattr(package, "build_thought_machine", refuse)
    orchestrator = _orchestrator()
    assert orchestrator.continuous_thought_machine is None
    assert orchestrator.thought_generator is None
    assert orchestrator.thought_stream is None
    assert orchestrator.recent_thought_texts() == []

    machine = make_machine(["river"])
    injected = _orchestrator(thought_machine=machine)
    assert injected.continuous_thought_machine is machine
    assert injected.thought_generator is machine.generator
    assert injected.thought_stream is machine.buffer


def test_attached_machine_feeds_measurement_thoughts():
    clock = Clock()
    machine = make_machine(["river", "garden"], clock=clock)
    orchestrator = _orchestrator()
    orchestrator.attach_thought_machine(machine)
    run(machine.tick())
    clock.advance(40)
    run(machine.tick())
    assert orchestrator.recent_thought_texts() == ["garden", "river"]

    captured = {}

    class RecordingFramework:
        async def measure_consciousness(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(phi=0.0, integration=0.0, self_model_accuracy=0.0)

    orchestrator.consciousness_framework = RecordingFramework()
    orchestrator.experience_buffer = [
        SimpleNamespace(
            snn_result={"firing_rates": {"input": [0.1, 0.2], "output": [0.3]}},
            subconscious_state=SimpleNamespace(state_trajectory=np.zeros((3, 4))),
        )
    ]
    assert run(orchestrator.measure_consciousness()) is not None
    assert captured["thoughts"] == ["garden", "river"]


def test_start_stop_status_delegate(tmp_path):
    settings = _settings(tmp_path)

    async def scenario():
        orchestrator = _orchestrator()
        assert orchestrator.get_continuous_thought_status(settings) == {
            "running": False,
            "status": "not_attached",
            "orchestrator_failures": {},
        }
        await orchestrator.start_continuous_thought_machine(
            consciousness_state="drowsy", settings=settings
        )
        machine = orchestrator.continuous_thought_machine
        assert isinstance(machine, package.ContinuousThoughtMachine)
        assert isinstance(machine.store, package.InMemoryThoughtStore)
        status = orchestrator.get_continuous_thought_status(settings)
        assert status["status"] == "running"
        assert status["consciousness_state"] == "drowsy"
        orchestrator.pause_continuous_thoughts()
        assert orchestrator.get_continuous_thought_status(settings)["status"] == (
            "paused"
        )
        orchestrator.resume_continuous_thoughts()
        orchestrator.set_consciousness_state("awake")
        assert machine.consciousness_state == "awake"
        await orchestrator.start_continuous_thought_machine(settings=settings)
        assert orchestrator.continuous_thought_machine is machine  # no second one
        await orchestrator.stop_continuous_thought_machine()
        assert machine.running is False
        status = orchestrator.get_continuous_thought_status(settings)
        assert status["status"] == "stopped"

    asyncio.run(scenario())


def test_start_builds_in_memory_whatever_ctm_store_says(tmp_path, monkeypatch):
    import mtc.neural.mongodb_schemas as schemas

    def refuse(*_args, **_kwargs):
        raise AssertionError("start must not build a data manager")

    monkeypatch.setattr(schemas, "NeuralDataManager", refuse)
    settings = _settings(tmp_path, CTM_STORE="mongo")

    async def scenario():
        orchestrator = _orchestrator()
        await orchestrator.start_continuous_thought_machine(settings=settings)
        machine = orchestrator.continuous_thought_machine
        await machine.stop()
        return machine

    machine = asyncio.run(scenario())
    assert isinstance(machine.store, package.InMemoryThoughtStore)
    assert isinstance(machine.gist_store, InMemoryThoughtGistStore)


def test_start_respects_ctm_enabled(tmp_path):
    settings = _settings(tmp_path, CTM_ENABLED=False)

    async def scenario():
        orchestrator = _orchestrator()
        await orchestrator.start_continuous_thought_machine(settings=settings)
        assert orchestrator.continuous_thought_machine is None
        return orchestrator.get_continuous_thought_status(settings)

    assert asyncio.run(scenario()) == {
        "running": False,
        "status": "disabled",
        "orchestrator_failures": {},
    }


def test_shutdown_awaits_stop():
    async def scenario():
        machine = make_machine(["river"])
        orchestrator = _orchestrator(thought_machine=machine)
        await machine.start()
        assert machine.running is True
        await orchestrator.shutdown()
        return orchestrator, machine

    orchestrator, machine = asyncio.run(scenario())
    assert machine.running is False
    assert orchestrator.continuous_thought_machine is None


def test_shutdown_leaves_the_detached_machine_state_to_the_machine(tmp_path):
    async def scenario():
        machine = package.build_thought_machine(settings=_settings(tmp_path))
        orchestrator = _orchestrator(thought_machine=machine)
        for _ in range(2):
            await machine.tick()
        held = [t.thought_id for t in machine.buffer.current_thoughts]
        types = list(machine.generator.recent_thought_types)
        await orchestrator.shutdown()
        return orchestrator, machine, held, types

    orchestrator, machine, held, types = asyncio.run(scenario())
    assert len(held) == 2 and len(types) == 2
    assert orchestrator.continuous_thought_machine is None
    assert orchestrator.thought_generator is None
    assert orchestrator.thought_stream is None
    # The machine persists itself: shutdown stops and detaches it and clears
    # nothing of its buffer or its generator's history.
    assert [t.thought_id for t in machine.buffer.current_thoughts] == held
    assert list(machine.generator.recent_thought_types) == types


def _broken(*_args, **_kwargs):
    raise RuntimeError("read failed")


def test_ctm_read_failures_counted_logged_and_reported(tmp_path, caplog, monkeypatch):
    settings = _settings(tmp_path)
    machine = make_machine(["river"])
    orchestrator = _orchestrator(thought_machine=machine)
    monkeypatch.setattr(machine.buffer, "get_recent_thoughts", _broken)
    with caplog.at_level("WARNING", logger=orch_mod.__name__):
        for _ in range(200):
            assert orchestrator.recent_thought_texts() == []
    lines = [r for r in caplog.records if "recent_thoughts" in r.getMessage()]
    assert len(lines) == 3  # the first, the hundredth and the two hundredth
    assert orchestrator.ctm_failures == {"recent_thoughts": 200}
    status = orchestrator.get_continuous_thought_status(settings)
    assert status["orchestrator_failures"] == {"recent_thoughts": 200}
    entry = orchestrator.get_system_metrics()["continuous_thought"]
    assert entry["orchestrator_failures"] == {"recent_thoughts": 200}

    monkeypatch.setattr(machine, "get_status", _broken)
    entry = orchestrator.get_system_metrics()["continuous_thought"]
    assert entry["attached"] is True
    assert entry["quiet_reason"] is None and entry["generated"] is None
    assert entry["orchestrator_failures"] == {
        "recent_thoughts": 200,
        "thought_status": 1,
    }


def test_ctm_read_failure_raises_under_debug(monkeypatch):
    import mtc.core.config as config_module

    machine = make_machine(["river"])
    orchestrator = _orchestrator(thought_machine=machine)
    monkeypatch.setattr(config_module, "get_settings", lambda: Settings(DEBUG=True))
    monkeypatch.setattr(machine.buffer, "get_recent_thoughts", _broken)
    with pytest.raises(RuntimeError):
        orchestrator.recent_thought_texts()
    monkeypatch.setattr(machine, "get_status", _broken)
    with pytest.raises(RuntimeError):
        orchestrator.get_system_metrics()
    assert orchestrator.ctm_failures == {"recent_thoughts": 1, "thought_status": 1}


def test_unmeasured_stream_values_read_none_in_status(tmp_path):
    settings = _settings(tmp_path)
    clock = Clock()
    machine = make_machine(["river", "garden"], clock=clock)
    orchestrator = _orchestrator(thought_machine=machine)
    summary = orchestrator.get_continuous_thought_status(settings)["stream_summary"]
    assert summary["current_intensity"] is None
    assert summary["thought_frequency"] is None
    run(machine.tick())
    summary = orchestrator.get_continuous_thought_status(settings)["stream_summary"]
    assert summary["current_intensity"] == pytest.approx(0.5)
    assert summary["thought_frequency"] is None  # one thought has no span
    clock.advance(40)
    run(machine.tick())
    summary = orchestrator.get_continuous_thought_status(settings)["stream_summary"]
    assert summary["thought_frequency"] == pytest.approx(1 / 40)


def test_component_map_reports_continuous_thought():
    orchestrator = _orchestrator()
    metrics = orchestrator.get_system_metrics()
    assert metrics["components"]["continuous_thought"] is True
    assert metrics["continuous_thought"] == {
        "available": True,
        "attached": False,
        "running": False,
        "orchestrator_failures": {},
    }
    machine = make_machine(["river"], embedder=None)
    orchestrator.attach_thought_machine(machine)
    run(machine.tick())
    entry = orchestrator.get_system_metrics()["continuous_thought"]
    assert entry["attached"] is True and entry["running"] is False
    assert entry["quiet_reason"] == "no_embedder"
    assert entry["generated"] == 1


def test_machine_sees_bridge_built_on_first_use(tmp_path):
    settings = _settings(tmp_path)

    async def scenario():
        orchestrator = _orchestrator()
        await orchestrator.start_continuous_thought_machine(settings=settings)
        machine = orchestrator.continuous_thought_machine
        await machine.stop()
        return orchestrator, machine

    orchestrator, machine = asyncio.run(scenario())
    source = machine.generator.emotion_source
    assert isinstance(source, BridgeEmotionSource)
    assert source.get_affect() is None  # no bridge yet: nothing measured
    state = SimpleNamespace(
        intuitive_valence=0.4, intuitive_arousal=0.6, readout_trained=True
    )
    # what _ensure_components does on the first experience
    orchestrator.lsm_bridge = SimpleNamespace(
        readout_trained=lambda: True, state_history=[state]
    )
    assert source.get_affect() == {"valence": 0.4, "arousal": 0.6}


def test_attaching_over_a_running_machine_is_refused():
    async def scenario():
        first = make_machine(["river"])
        orchestrator = _orchestrator(thought_machine=first)
        await first.start()
        try:
            with pytest.raises(RuntimeError):
                orchestrator.attach_thought_machine(make_machine(["garden"]))
        finally:
            await first.stop()

    asyncio.run(scenario())
