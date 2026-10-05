"""The orchestrator's optional components resolve to their real targets and are
reported honestly: present ones as available, missing ones as missing, and the
measurement framework sized from the live substrates instead of constants."""

import importlib
import importlib.util
import logging
from types import SimpleNamespace

import mtc.neural.orchestrator as orch_mod
from mtc.neural.orchestrator import NeuralOrchestrator

PORT_TARGETS = {
    "emotion_lsm_bridge": "mtc.neural.liquid.emotion_lsm_bridge",
    "memory_consolidator": "mtc.neural.htm.memory_consolidator",
    "neural_data_manager": "mtc.neural.mongodb_schemas",
    "redis_manager": "mtc.memory.redis_manager",
    "verification_framework": "mtc.consciousness.verification_framework",
    "continuous_thought": "mtc.consciousness.continuous",
    "measurement_framework": "mtc.assessment.framework",
}


def _module_missing(path: str) -> bool:
    try:
        return importlib.util.find_spec(path) is None
    except ModuleNotFoundError:
        return True


def test_measurement_framework_resolves_from_assessment_package():
    framework_mod = importlib.import_module("mtc.assessment.framework")
    assert orch_mod._HAS_MEASUREMENT is True
    assert (
        orch_mod.ConsciousnessMeasurementFramework
        is framework_mod.ConsciousnessMeasurementFramework
    )
    assert orch_mod.ConsciousnessMetrics is framework_mod.ConsciousnessMetrics
    assert orch_mod.OPTIONAL_COMPONENTS["measurement_framework"][1] is None


def test_optional_imports_point_at_port_targets():
    registry = orch_mod.OPTIONAL_COMPONENTS
    for name, target in PORT_TARGETS.items():
        assert registry[name][0] == target, name
    with open(orch_mod.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "mtc.consciousness.measurement_framework" not in source
    for target in PORT_TARGETS.values():
        assert f"from {target} import" in source, target


def test_available_components_reports_missing_modules():
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    components = orchestrator.available_components()

    assert set(PORT_TARGETS) <= set(components)
    assert all(isinstance(value, bool) for value in components.values())
    assert components["measurement_framework"] is True
    for name, target in PORT_TARGETS.items():
        if _module_missing(target):
            assert components[name] is False, name

    metrics = orchestrator.get_system_metrics()
    assert metrics["components"] == components


def test_missing_components_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="mtc.neural.orchestrator"):
        orchestrator = NeuralOrchestrator(enable_mongodb=False)

    missing = [
        name for name, ok in orchestrator.available_components().items() if not ok
    ]
    lines = [
        record.getMessage()
        for record in caplog.records
        if "Optional components not installed" in record.getMessage()
    ]
    if not missing:
        assert lines == []
        return
    assert len(lines) == 1
    for name in missing:
        assert name in lines[0]
        phase = orch_mod.OPTIONAL_COMPONENTS[name][1]
        if phase:
            assert phase in lines[0]


class _FakeSNN:
    def _count_neurons(self) -> int:
        return 123


class _FakeBridge:
    def __init__(self, **kwargs):
        self.snn = _FakeSNN()
        self.lsm = SimpleNamespace(config=SimpleNamespace(reservoir_size=77))


class _FakeConsolidator:
    def __init__(self, **kwargs):
        self.htm = SimpleNamespace(
            config=SimpleNamespace(num_columns=16, cells_per_column=4)
        )
        self.episodic_memories = []
        self.semantic_patterns = []


class _RecordingFramework:
    def __init__(self, snn_neurons, lsm_neurons, htm_columns):
        self.snn_neurons = snn_neurons
        self.lsm_neurons = lsm_neurons
        self.htm_columns = htm_columns


def test_framework_gets_detected_sizes_not_constants(monkeypatch):
    # A recording stand-in for the framework class pins what the orchestrator
    # passes, independent of the framework's own construction.
    monkeypatch.setattr(
        orch_mod, "ConsciousnessMeasurementFramework", _RecordingFramework
    )
    bare = NeuralOrchestrator(enable_mongodb=False)
    assert bare.substrate_sizes() == {
        "snn": None,
        "lsm": None,
        "htm": None,
        "htm_columns": None,
    }
    framework = bare.consciousness_framework
    assert (framework.snn_neurons, framework.lsm_neurons, framework.htm_columns) == (
        0,
        0,
        0,
    )

    monkeypatch.setattr(orch_mod, "_HAS_EMOTION_BRIDGE", True)
    monkeypatch.setattr(orch_mod, "EmotionLSMBridge", _FakeBridge)
    monkeypatch.setattr(orch_mod, "_HAS_MEMORY_CONSOLIDATOR", True)
    monkeypatch.setattr(orch_mod, "MemoryConsolidator", _FakeConsolidator)
    sized = NeuralOrchestrator(enable_mongodb=False)

    sizes = sized.substrate_sizes()
    assert sizes == {"snn": 123, "lsm": 77, "htm": 64, "htm_columns": 16}
    framework = sized.consciousness_framework
    assert framework.snn_neurons == 123
    assert framework.lsm_neurons == 77
    assert framework.htm_columns == 16
    # No constant from the old construction survives.
    assert 5000 not in (framework.snn_neurons, framework.lsm_neurons)
