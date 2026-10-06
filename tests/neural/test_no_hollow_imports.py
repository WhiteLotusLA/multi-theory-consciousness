"""No hollow imports: every optional component the neural orchestrator lists
and every guarded import in the GPU integration layer resolves to a module that
exists and defines the names imported from it.

A guarded import is a ``from ... import ...`` inside a module-level ``try``.
The check reads source with ``ast`` and locates modules with
``importlib.util.find_spec``, so it imports none of the optional libraries the
guarded modules may need.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

import mtc.neural.orchestrator as orch_mod

ROOT = Path(__file__).resolve().parents[2]
GUARDED_SOURCES = ("mtc/neural/orchestrator.py", "mtc/neural/gpu_integration.py")


def guarded_imports(source: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """``(module, names)`` of each ``from`` import inside a module-level try."""
    found = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Try):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                names = tuple(alias.name for alias in statement.names)
                found.append((statement.module, names))
    return found


def defined_names(path: Path) -> Set[str]:
    """Names a module defines or imports at its top level."""
    names: Set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _spec_path(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, f"{module} does not exist"
    return Path(spec.origin)


def _all_guarded() -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
    return {
        relative: guarded_imports((ROOT / relative).read_text(encoding="utf-8"))
        for relative in GUARDED_SOURCES
    }


def test_optional_component_paths_exist_and_export_names():
    guarded = _all_guarded()
    orchestrator_modules = {module for module, _ in guarded[GUARDED_SOURCES[0]]}
    for name, (path, _phase) in orch_mod.OPTIONAL_COMPONENTS.items():
        assert importlib.util.find_spec(path) is not None, f"{name}: {path}"
        assert path in orchestrator_modules, f"{name}: no guarded import of {path}"
    hollow = []
    for relative, imports in guarded.items():
        assert imports, f"{relative} has no guarded import to check"
        for module, names in imports:
            exported = defined_names(_spec_path(module))
            for imported in names:
                if imported not in exported:
                    hollow.append(f"{relative}: {module}.{imported}")
    assert not hollow, "guarded imports that resolve to nothing: " + ", ".join(hollow)


def test_component_flags_cover_the_table_and_the_machine_is_available():
    flags = orch_mod._component_flags()
    assert set(flags) == set(orch_mod.OPTIONAL_COMPONENTS)
    assert flags["continuous_thought"] is True


def test_detector_finds_a_hollow_import():
    source = (
        "try:\n"
        "    from mtc.consciousness.continuous.novelty import NoSuchFilter\n"
        "except ImportError:\n"
        "    NoSuchFilter = None\n"
    )
    ((module, names),) = guarded_imports(source)
    assert module == "mtc.consciousness.continuous.novelty"
    assert "NoSuchFilter" not in defined_names(_spec_path(module))
    assert names == ("NoSuchFilter",)


class LookupEmbedder:
    semantic = True
    dim = 8

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim))
        out[:, 0] = 1.0
        return out


def _block_torch(monkeypatch) -> None:
    """Make ``import torch`` fail from here on, as on an install without it."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", None)


def test_cpu_novelty_filter_needs_no_torch(monkeypatch):
    # runs with or without torch installed: the CPU filter is the one left
    from mtc.consciousness.continuous.novelty import (
        AcceleratedNoveltyFilter,
        NoveltyFilter,
    )

    _block_torch(monkeypatch)
    with pytest.raises(ImportError, match="neural"):
        AcceleratedNoveltyFilter(LookupEmbedder())
    cpu = NoveltyFilter(LookupEmbedder())
    assert cpu.status()["embedder"] == "semantic"
    assert cpu.find_near_duplicate("a first thought").is_unique is True
    assert cpu.find_near_duplicate("a second thought").is_unique is False


def test_gpu_factory_creates_novelty_filter(monkeypatch):
    # the GPU integration layer itself imports torch (through the device
    # manager), so the factory is exercised only where torch is installed;
    # torch is then blocked to show its CPU branch never needs it
    pytest.importorskip("torch")
    from mtc.consciousness.continuous.novelty import (
        AcceleratedNoveltyFilter,
        NoveltyFilter,
    )
    from mtc.neural import gpu_integration
    from mtc.neural.gpu_integration import GPUComponentFactory, GPUIntegrationConfig

    assert gpu_integration._HAS_THOUGHT_PROCESSOR is True
    cpu = GPUComponentFactory(GPUIntegrationConfig.cpu_only_config())
    plain = cpu.create_thought_processor(embedder=LookupEmbedder())
    assert type(plain) is NoveltyFilter
    assert plain.status()["embedder"] == "semantic"

    # no accelerator (or no accelerated torch): the CPU filter
    idle = SimpleNamespace(enabled=False, device=SimpleNamespace(type="cpu"))
    no_accel = GPUComponentFactory(GPUIntegrationConfig(manager=idle))
    assert (
        type(no_accel.create_thought_processor(embedder=LookupEmbedder()))
        is NoveltyFilter
    )

    # the embedder may come from a vector coordinator
    coordinator = SimpleNamespace(embedder=LookupEmbedder())
    from_coordinator = cpu.create_thought_processor(vector_coordinator=coordinator)
    assert from_coordinator.embedder is coordinator.embedder

    # the accelerated branch, exercised on the CPU device
    cpu_device = SimpleNamespace(enabled=True, device="cpu")
    accelerated = GPUComponentFactory(GPUIntegrationConfig(manager=cpu_device))
    built = accelerated.create_thought_processor(
        embedder=LookupEmbedder(), cache_size=4
    )
    assert isinstance(built, AcceleratedNoveltyFilter)
    assert built.status()["device"] == "cpu" and built.cache_size == 4

    # without torch the factory's CPU branch still builds a working filter
    _block_torch(monkeypatch)
    for factory in (cpu, no_accel):
        built = factory.create_thought_processor(embedder=LookupEmbedder())
        assert type(built) is NoveltyFilter
        assert built.find_near_duplicate("a first thought").is_unique is True
        assert built.find_near_duplicate("a second thought").is_unique is False

    source = Path(gpu_integration.__file__).read_text(encoding="utf-8")
    for claim in ("faster", "zero-copy", "ZERO-COPY", "200x"):
        assert claim not in source, claim
