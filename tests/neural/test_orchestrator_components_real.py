"""The orchestrator's optional components resolve to real modules.

Small bridge and consolidator classes (hidden 32 by 32, reservoir 50, a 64
column HTM, all on the CPU) stand in for the default-size ones so these cases
stay fast and never place a tensor on an accelerator. No store client is built
against a host: the one case that builds a data manager from Settings replaces
the client factory with a fake.
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import mtc.neural.orchestrator as orch_mod
from mtc.neural.htm import (
    EpisodicMemory,
    MemoryConsolidator,
    ConsolidatorConfig,
)
from mtc.neural.htm.production_htm import HTMConfig, ProductionHTM
from mtc.neural.liquid.emotion_lsm_bridge import (
    EmotionLSMBridge,
    EmotionLSMBridgeConfig,
)
from mtc.neural.mongodb_schemas import NeuralDataManager
from mtc.neural.orchestrator import NeuralOrchestrator
from tests.memory.test_mongo_stores import FakeMongoClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bridge_config(**changes) -> EmotionLSMBridgeConfig:
    values = dict(
        snn_hidden_layers=(32, 32),
        reservoir_size=50,
        lsm_input_dim=20,
        lsm_output_dim=6,
        numpy_hidden_neurons=20,
        numpy_reservoir_size=40,
        device="cpu",
        seed=1,
    )
    values.update(changes)
    return EmotionLSMBridgeConfig(**values)


def _small_htm() -> ProductionHTM:
    return ProductionHTM(
        HTMConfig(
            num_columns=64,
            cells_per_column=4,
            input_bits=128,
            input_sparsity=0.05,
            potential_radius=64,
            local_area_density=0.05,
            activation_threshold=2,
            max_segments_per_cell=16,
            max_synapses_per_segment=8,
        )
    )


class SmallBridge(EmotionLSMBridge):
    """The real bridge with a small config, built the way the orchestrator does."""

    def __init__(self, **kwargs):
        super().__init__(config=_bridge_config(), **kwargs)


class SmallConsolidator(MemoryConsolidator):
    def __init__(self, **kwargs):
        super().__init__(htm=_small_htm(), **kwargs)


@pytest.fixture
def neural_extra():
    pytest.importorskip("torch")
    pytest.importorskip("snntorch")
    pytest.importorskip("reservoirpy")


@pytest.fixture
def small_defaults(monkeypatch, neural_extra):
    monkeypatch.setattr(orch_mod, "EmotionLSMBridge", SmallBridge)
    monkeypatch.setattr(orch_mod, "MemoryConsolidator", SmallConsolidator)


def test_hollow_imports_resolved():
    from mtc.memory import redis_manager
    from mtc.neural import mongodb_schemas
    from mtc.neural.htm import memory_consolidator
    from mtc.neural.liquid import emotion_lsm_bridge

    assert orch_mod._HAS_EMOTION_BRIDGE is True
    assert orch_mod._HAS_MEMORY_CONSOLIDATOR is True
    assert orch_mod._HAS_NEURAL_DATA is True
    assert orch_mod._HAS_REDIS is True
    assert orch_mod.EmotionLSMBridge is emotion_lsm_bridge.EmotionLSMBridge
    assert orch_mod.MemoryConsolidator is memory_consolidator.MemoryConsolidator
    assert orch_mod.NeuralDataManager is mongodb_schemas.NeuralDataManager
    assert orch_mod.RedisManager is redis_manager.RedisManager
    assert orch_mod.EpisodicMemory is memory_consolidator.EpisodicMemory
    assert orch_mod.SemanticPattern is memory_consolidator.SemanticPattern
    assert orch_mod.SubconsciousState is emotion_lsm_bridge.SubconsciousState
    assert orch_mod.EmotionState is emotion_lsm_bridge.EmotionState


def test_default_orchestrator_builds_no_store_client():
    code = (
        "import sys\n"
        "from mtc.neural.orchestrator import NeuralOrchestrator\n"
        "o = NeuralOrchestrator()\n"
        "assert o.neural_data_manager is None and o.redis is None\n"
        "assert o.lsm_bridge is None and o.memory_consolidator is None\n"
        "clients = ('pymongo', 'motor', 'redis', 'asyncpg', 'neo4j', 'qdrant_client')\n"
        "loaded = [c for c in clients if c in sys.modules]\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    env = {**os.environ, "PYTHONPATH": ROOT, "PYPHI_WELCOME_OFF": "yes"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_enable_mongodb_defaults_to_false():
    import inspect

    default = inspect.signature(NeuralOrchestrator.__init__).parameters[
        "enable_mongodb"
    ]
    assert default.default is False
    assert NeuralOrchestrator().neural_data_manager is None


def test_components_built_lazily_on_first_experience(small_defaults):
    orchestrator = NeuralOrchestrator()
    assert orchestrator.lsm_bridge is None and orchestrator.memory_consolidator is None
    assert orchestrator.substrate_sizes() == {
        "snn": None,
        "lsm": None,
        "htm": None,
        "htm_columns": None,
    }
    bare_framework = orchestrator.consciousness_framework
    assert (bare_framework.snn_neurons, bare_framework.lsm_neurons) == (0, 0)

    orchestrator.process_experience("observation", {"joy": 0.7})
    assert isinstance(orchestrator.lsm_bridge, SmallBridge)
    assert isinstance(orchestrator.memory_consolidator, SmallConsolidator)
    framework = orchestrator.consciousness_framework
    assert framework is not bare_framework
    assert framework.snn_neurons == 10 + 32 + 32 + 10
    assert framework.lsm_neurons == 50 and framework.htm_columns == 64
    assert orchestrator.substrate_sizes()["htm"] == 64 * 4

    bridge = orchestrator.lsm_bridge
    orchestrator.process_experience("observation", {"joy": 0.5})
    assert orchestrator.lsm_bridge is bridge  # built once
    assert orchestrator.consciousness_framework is framework


def test_injected_components_used(neural_extra):
    bridge = EmotionLSMBridge(config=_bridge_config())
    consolidator = MemoryConsolidator(htm=_small_htm())
    orchestrator = NeuralOrchestrator(
        emotion_bridge=bridge, memory_consolidator=consolidator
    )
    assert orchestrator.lsm_bridge is bridge
    assert orchestrator.memory_consolidator is consolidator
    assert orchestrator.substrate_sizes()["lsm"] == 50  # sized at construction
    orchestrator.process_experience("observation", {"joy": 0.7})
    assert orchestrator.lsm_bridge is bridge and bridge.states_processed == 1
    assert len(consolidator.episodic_memories) == 1

    left_out = NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False)
    left_out.process_experience("observation", {"joy": 0.7})
    assert left_out.lsm_bridge is None and left_out.memory_consolidator is None
    assert left_out.latest_neural_signals is None
    assert left_out.sleep() == {"status": "no_consolidator"}


def test_enable_mongodb_builds_manager_from_settings_without_connecting(
    monkeypatch, small_defaults
):
    from mtc.neural import mongodb_schemas

    client = FakeMongoClient()
    seen = []

    def factory(settings, **options):
        seen.append((settings, options))
        return client

    monkeypatch.setattr(mongodb_schemas, "create_mongo_client", factory)
    orchestrator = NeuralOrchestrator(enable_mongodb=True)
    assert isinstance(orchestrator.neural_data_manager, NeuralDataManager)
    assert len(seen) == 1 and seen[0][0] is not None
    assert seen[0][1] == {"connect": False}  # the driver must not connect either
    assert client.calls == []  # nothing connected, nothing written
    assert orchestrator.lsm_bridge is None  # components still wait for first use
    orchestrator._ensure_components()
    assert (
        orchestrator.lsm_bridge.neural_data_manager is orchestrator.neural_data_manager
    )
    assert orchestrator.lsm_bridge.store_states is True
    assert (
        orchestrator.memory_consolidator.neural_data_manager
        is orchestrator.neural_data_manager
    )
    assert client.calls == []

    injected = NeuralDataManager(client=FakeMongoClient())
    with_manager = NeuralOrchestrator(neural_data_manager=injected)
    assert with_manager.neural_data_manager is injected
    with_manager._ensure_components()
    assert with_manager.lsm_bridge.store_states is False  # enable_mongodb is off


def test_process_experience_end_to_end_small(small_defaults):
    orchestrator = NeuralOrchestrator()
    experience = orchestrator.process_experience(
        "conversation", {"joy": 0.7, "fear": 0.2}, context="a synthetic sentence"
    )
    snn_result = experience.snn_result
    assert len(snn_result["firing_rates"]["output"]) == 10
    assert isinstance(snn_result["total_spikes"], int)
    state = experience.subconscious_state
    assert state.readout_trained is False and state.emotional_coherence is None
    assert state.intuitive_valence is None and state.intuitive_arousal is None
    assert isinstance(experience.episodic_memory, EpisodicMemory)
    assert experience.episodic_memory.consciousness_level_at_encoding is None
    assert set(orchestrator.latest_neural_signals) == {"snn", "lsm", "htm"}
    assert orchestrator._last_binding_result is not None
    assert orchestrator.state.episodic_memories_count == 1

    blocks = orchestrator._verification_neural_data()
    assert blocks["snn"]["is_real"] is True  # the SNN produced a synchrony
    assert blocks["lsm"]["is_real"] is False  # an untrained readout is no coherence
    assert blocks["lsm"]["coherence"] is None
    assert blocks["lsm"]["intuitive_valence"] is None
    metrics = orchestrator.get_system_metrics()
    assert metrics["intuition"]["intuitive_feelings"]["valence"] is None
    assert metrics["htm_metrics"] is not None
    assert metrics["memory"]["consolidation_strength"] is None  # nothing replayed
    assert all(
        metrics["components"][n]
        for n in ("emotion_lsm_bridge", "memory_consolidator", "neural_data_manager")
    )


def test_sleep_consolidates_through_real_consolidator(small_defaults):
    orchestrator = NeuralOrchestrator()
    for emotion in ("joy", "fear", "joy", "love"):
        orchestrator.process_experience("observation", {emotion: 0.8})
    assert len(orchestrator.experience_buffer) == 4
    report = orchestrator.sleep(duration="short")
    assert report["status"] == "completed" and report["cycles"] == 10
    assert report["memories_processed"] == 4
    assert "patterns_discovered" in report and "anomaly_reduction" in report
    assert orchestrator.experience_buffer == []
    assert orchestrator.total_sleep_cycles == 1
    consolidator = orchestrator.memory_consolidator
    assert orchestrator.state.semantic_patterns_count == len(
        consolidator.semantic_patterns
    )
    assert (
        orchestrator.get_semantic_understanding()
        == consolidator.get_semantic_patterns()
    )
    strength = orchestrator.get_system_metrics()["memory"]["consolidation_strength"]
    assert strength is None or 0.0 <= strength <= 1.0
    assert orchestrator.sleep() == {"status": "no_experiences"}


def test_sleep_after_a_restore_without_episodes_does_not_fail(small_defaults):
    source = NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False)
    for _ in range(2):
        source.process_experience("observation", {"joy": 0.4})
    saved = source.to_state_dict()
    restored = NeuralOrchestrator()
    restored.from_state_dict(saved)
    assert len(restored.experience_buffer) == 2
    report = restored.sleep()
    assert report == {"status": "no_memories"}
    assert restored.memory_consolidator is not None  # built by the sleep


def test_available_components_report_real_modules():
    orchestrator = NeuralOrchestrator()
    components = orchestrator.available_components()
    for name, path in (
        ("emotion_lsm_bridge", "mtc.neural.liquid.emotion_lsm_bridge"),
        ("memory_consolidator", "mtc.neural.htm.memory_consolidator"),
        ("neural_data_manager", "mtc.neural.mongodb_schemas"),
        ("redis_manager", "mtc.memory.redis_manager"),
    ):
        assert importlib.util.find_spec(path) is not None
        assert components[name] is True, name
    missing = {name for name, ok in components.items() if not ok}
    assert missing <= {"continuous_thought"}  # the last hollow import (Phase 6)
    assert orchestrator.get_system_metrics()["components"] == components


def test_construction_log_names_only_the_continuous_machine(caplog):
    with caplog.at_level("INFO", logger="mtc.neural.orchestrator"):
        NeuralOrchestrator()
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Optional components not installed" in r.getMessage()
    ]
    for line in lines:
        for name in (
            "emotion_lsm_bridge",
            "memory_consolidator",
            "neural_data_manager",
            "redis_manager",
        ):
            assert name not in line


def test_shutdown_cache_clear_through_device_manager(monkeypatch):
    calls = []

    class Manager:
        def __init__(self, device: str, enabled: bool = True) -> None:
            self.device = SimpleNamespace(type=device)
            self.enabled = enabled

        def empty_cache(self) -> None:
            calls.append(self.device.type)

    def install(manager) -> None:
        monkeypatch.setitem(
            sys.modules,
            "mtc.neural.mps_utils",
            SimpleNamespace(mps_manager=manager),
        )

    pytest.importorskip("torch")
    orchestrator = NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False)

    install(Manager("mps"))
    orchestrator.process_experience("observation", {"joy": 0.5})
    assert calls == ["mps"]  # the per-experience clear
    asyncio.run(orchestrator.shutdown())
    assert calls == ["mps", "mps"]

    calls.clear()
    install(Manager("cuda"))
    orchestrator.process_experience("observation", {"joy": 0.5})
    assert calls == []  # a CUDA cache is released at shutdown only
    asyncio.run(orchestrator.shutdown())
    assert calls == ["cuda"]

    calls.clear()
    for manager in (Manager("cpu"), Manager("mps", enabled=False)):
        install(manager)
        asyncio.run(orchestrator.shutdown())
    assert calls == []


def test_orchestrator_source_names_no_accelerator_backend():
    import inspect

    source = inspect.getsource(orch_mod)
    for needle in ("torch.mps", "torch.cuda", "backends.mps"):
        assert needle not in source, needle


def test_simulate_day_is_reproducible_and_leaves_global_random_alone(small_defaults):
    first = NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False)
    second = NeuralOrchestrator(emotion_bridge=False, memory_consolidator=False)
    np.random.seed(5)
    expected = np.random.random()
    np.random.seed(5)
    first.simulate_day(n_experiences=4, sleep_at_end=False, seed=11)
    assert np.random.random() == expected
    second.simulate_day(n_experiences=4, sleep_at_end=False, seed=11)
    assert [e.emotions for e in first.experience_buffer] == [
        e.emotions for e in second.experience_buffer
    ]
    assert all(
        e.experience_type == "simulated_experience" for e in first.experience_buffer
    )
    with pytest.raises(ValueError):
        first.simulate_day(n_experiences=0)


def test_simulate_day_runs_the_substrates_and_sleeps(small_defaults):
    orchestrator = NeuralOrchestrator()
    summary = orchestrator.simulate_day(
        n_experiences=6, sleep_at_end=True, sleep_duration="short", seed=3
    )
    assert summary["n_experiences"] == 6
    assert summary["consolidation"]["status"] in ("completed", "no_experiences")
    assert summary["final_metrics"]["consciousness"]["experiences_processed"] == 6
    assert orchestrator.total_sleep_cycles >= 1
    assert summary["avg_experience_time_sec"] == pytest.approx(
        summary["elapsed_time_sec"] / 6
    )


def test_consolidator_config_is_reachable_from_the_package():
    assert ConsolidatorConfig().store_context_text is False
