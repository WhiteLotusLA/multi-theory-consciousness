"""The orchestrator's optional components resolve to their real targets and are
reported honestly: present ones as available, missing ones as missing, and the
measurement framework sized from the live substrates instead of constants."""

import importlib
import importlib.util
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest

import mtc.neural.orchestrator as orch_mod
from mtc.core.config import Settings
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
    assert orch_mod.OPTIONAL_COMPONENTS["measurement_framework"][1] == "core"


def test_optional_imports_point_at_port_targets():
    registry = orch_mod.OPTIONAL_COMPONENTS
    for name, target in PORT_TARGETS.items():
        assert registry[name][0] == target, name
    with open(orch_mod.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "mtc.consciousness.measurement_framework" not in source
    for target in PORT_TARGETS.values():
        assert f"from {target} import" in source, target


def test_available_components_reports_missing_modules(monkeypatch):
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    components = orchestrator.available_components()

    assert set(PORT_TARGETS) <= set(components)
    assert all(isinstance(value, bool) for value in components.values())
    assert components["measurement_framework"] is True
    for name, target in PORT_TARGETS.items():
        assert components[name] is (not _module_missing(target)), name
    assert orchestrator.get_system_metrics()["components"] == components

    # One component that cannot be imported is reported missing; every other
    # one stays present.
    module = _orchestrator_module_without(
        monkeypatch, "mtc.consciousness.verification_framework"
    )
    reduced = module.NeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    seen = reduced.available_components()
    assert seen["verification_framework"] is False
    others = {name: ok for name, ok in seen.items() if name != "verification_framework"}
    assert set(others) == set(module.OPTIONAL_COMPONENTS) - {"verification_framework"}
    assert all(ok is True for ok in others.values())
    assert reduced.consciousness_verifier is None
    assert reduced.get_system_metrics()["components"] == seen


def test_missing_components_logged_once(monkeypatch, caplog):
    module = _orchestrator_module_without(monkeypatch, "mtc.consciousness.continuous")

    def lines():
        return [
            record.getMessage()
            for record in caplog.records
            if "Optional components could not be imported" in record.getMessage()
        ]

    with caplog.at_level(logging.INFO):
        orchestrator = module.NeuralOrchestrator(
            enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
        )
        orchestrator.available_components()
        orchestrator.get_system_metrics()

    assert orchestrator.available_components()["continuous_thought"] is False
    assert lines() == [
        "Optional components could not be imported: continuous_thought "
        "(part of the core install)"
    ]
    for name in module.OPTIONAL_COMPONENTS:
        if name != "continuous_thought":
            assert name not in lines()[0]


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


class _RecordingBridge(_FakeBridge):
    build_on_first_use = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs


class _RecordingConsolidator(_FakeConsolidator):
    build_on_first_use = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs


def _recording_substrates(monkeypatch):
    monkeypatch.setattr(orch_mod, "_HAS_EMOTION_BRIDGE", True)
    monkeypatch.setattr(orch_mod, "EmotionLSMBridge", _RecordingBridge)
    monkeypatch.setattr(orch_mod, "_HAS_MEMORY_CONSOLIDATOR", True)
    monkeypatch.setattr(orch_mod, "MemoryConsolidator", _RecordingConsolidator)


def test_layer_flags_follow_components():
    bare = NeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    assert bare.get_system_metrics()["layers"] == {
        "snn_active": False,
        "lsm_active": False,
        "htm_active": False,
    }

    wired = NeuralOrchestrator(
        enable_mongodb=False,
        emotion_bridge=_FakeBridge(),
        memory_consolidator=_FakeConsolidator(),
    )
    wired.lsm_bridge.get_metrics = lambda: {}
    assert wired.get_system_metrics()["layers"] == {
        "snn_active": True,
        "lsm_active": True,
        "htm_active": True,
    }

    half = NeuralOrchestrator(
        enable_mongodb=False,
        emotion_bridge=SimpleNamespace(snn=_FakeSNN(), lsm=None, get_metrics=dict),
        memory_consolidator=SimpleNamespace(htm=None),
    )
    assert half.get_system_metrics()["layers"] == {
        "snn_active": True,
        "lsm_active": False,
        "htm_active": False,
    }


def test_deferred_components_are_not_active_until_built(monkeypatch):
    _recording_substrates(monkeypatch)
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    assert orchestrator.get_system_metrics()["layers"] == {
        "snn_active": False,
        "lsm_active": False,
        "htm_active": False,
    }
    orchestrator._ensure_components()
    orchestrator.lsm_bridge.get_metrics = lambda: {}
    assert set(orchestrator.get_system_metrics()["layers"].values()) == {True}


def test_orchestrator_reads_the_settings_it_was_given(monkeypatch):
    """Given a Settings object, the orchestrator reads DEBUG, CTM_ENABLED and
    the device from it and hands it to the substrates it builds; the process
    Settings are never read."""
    from mtc.core import config as config_module

    reads = []

    def forbidden():
        reads.append("get_settings")
        raise AssertionError("the process Settings were read")

    _recording_substrates(monkeypatch)
    cfg = Settings(_env_file=None, DEVICE="cpu", CTM_ENABLED=False, DEBUG=True)
    monkeypatch.setattr(config_module, "get_settings", forbidden)
    orchestrator = NeuralOrchestrator(enable_mongodb=False, settings=cfg)
    orchestrator._ensure_components()
    assert orchestrator.memory_consolidator.kwargs["settings"] is cfg
    assert orchestrator.lsm_bridge.kwargs["config"].device == "cpu"
    assert orchestrator.get_continuous_thought_status()["status"] == "disabled"
    # DEBUG comes from the orchestrator's own Settings: the failure is raised
    with pytest.raises(RuntimeError, match="read failed"):
        orchestrator._ctm_failure("recent_thoughts", RuntimeError("read failed"))
    released = orch_mod._release_accelerator_cache(
        include_cuda=False, manager_source=orchestrator._accelerator_manager
    )
    assert released is None  # the CPU has no cache to release
    assert reads == []


def test_accelerator_release_uses_a_manager_bound_to_the_settings():
    pytest.importorskip("torch")
    from mtc.neural import mps_utils

    cfg = Settings(_env_file=None, DEVICE="cpu")
    bound = NeuralOrchestrator(
        enable_mongodb=False,
        emotion_bridge=False,
        memory_consolidator=False,
        settings=cfg,
    )
    manager = bound._accelerator_manager()
    assert isinstance(manager, mps_utils.MPSManager)
    assert manager is not mps_utils.mps_manager
    assert manager._settings is cfg
    assert bound._accelerator_manager() is manager  # built once
    assert manager.device.type == "cpu"
    # without Settings the shared manager serves, as before
    plain = NeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    assert plain._accelerator_manager() is mps_utils.mps_manager


def test_orchestrator_without_settings_builds_substrates_as_before(monkeypatch):
    _recording_substrates(monkeypatch)
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    orchestrator._ensure_components()
    assert "settings" not in orchestrator.memory_consolidator.kwargs
    assert "config" not in orchestrator.lsm_bridge.kwargs


def test_simple_workspace_is_gone():
    orchestrator = NeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    assert "global_workspace" not in orchestrator.available_components()
    assert "global_workspace" not in orch_mod.OPTIONAL_COMPONENTS
    assert not hasattr(orchestrator, "global_workspace")
    assert not hasattr(orch_mod, "_HAS_GWT")
    assert importlib.util.find_spec("mtc.consciousness.global_workspace") is None
    with open(orch_mod.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "mtc.consciousness.global_workspace" not in source
    # The enhanced workspace stays
    assert orchestrator.available_components()["enhanced_workspace"] is True


def _orchestrator_module_without(monkeypatch, hidden: str):
    """A fresh copy of the orchestrator module loaded while ``hidden`` cannot
    be imported, so its guarded import of that component fails for real."""
    import sys

    monkeypatch.setitem(sys.modules, hidden, None)
    name = "_orchestrator_without_component"
    spec = importlib.util.spec_from_file_location(name, orch_mod.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_missing_component_log_names_its_extra(monkeypatch, caplog):
    module = _orchestrator_module_without(monkeypatch, "mtc.memory.redis_manager")
    with caplog.at_level(logging.INFO):
        module.NeuralOrchestrator(
            enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
        )
    lines = [
        record.getMessage()
        for record in caplog.records
        if "Optional components could not be imported" in record.getMessage()
    ]
    assert lines == [
        "Optional components could not be imported: redis_manager "
        "(uses the redis extra)"
    ]
    assert "phase" not in lines[0].lower()


def test_component_modules_import_without_the_extra_they_use(tmp_path):
    """The label names what a component uses at run time, not a cause of a
    failed import: every module imports with every one of those extras hidden."""
    import os
    import tomllib
    from pathlib import Path

    root = Path(orch_mod.__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    used = {
        provider
        for _path, provider in orch_mod.OPTIONAL_COMPONENTS.values()
        if provider != orch_mod.CORE_INSTALL
    }
    assert used, "no component names an extra"
    hidden = {"pymdp"}
    for extra in used:
        for requirement in extras[extra]:
            name = requirement.split(">")[0].split("=")[0].split("[")[0].strip()
            hidden.add("pymdp" if name == "inferactively-pymdp" else name)
    modules = sorted(path for path, _ in orch_mod.OPTIONAL_COMPONENTS.values())
    script = tmp_path / "import_without_extras.py"
    script.write_text(
        "import importlib, sys\n"
        f"for name in {sorted(hidden)!r}:\n"
        "    sys.modules[name] = None\n"
        f"for module in {modules!r}:\n"
        "    importlib.import_module(module)\n"
    )
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MTC_") or key == "MTC_HOME"
    }
    child_env.update(
        {
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "PYPHI_WELCOME_OFF": "yes",
            "PYTHONPATH": str(root),
            "MTC_POSTGRES_ENABLED": "false",
            "MTC_MONGODB_ENABLED": "false",
            "MTC_REDIS_ENABLED": "false",
            "MTC_QDRANT_ENABLED": "false",
            "MTC_NEO4J_ENABLED": "false",
            "MTC_API_PORT": "0",
        }
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
        env=child_env,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_component_labels_name_an_extra_or_the_core_install():
    import tomllib
    from pathlib import Path

    root = Path(orch_mod.__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as handle:
        extras = set(tomllib.load(handle)["project"]["optional-dependencies"])
    for name, (_path, provider) in orch_mod.OPTIONAL_COMPONENTS.items():
        assert provider == orch_mod.CORE_INSTALL or provider in extras, name
        assert "phase" not in provider.lower(), name
