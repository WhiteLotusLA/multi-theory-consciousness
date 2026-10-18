"""The continuous package exports the names the neural orchestrator imports and
a factory that builds an in-memory machine from Settings and the CTM files."""

import ast
import asyncio
from pathlib import Path

import pytest

import mtc.consciousness.continuous as package
import mtc.neural.orchestrator as orch_mod
from mtc.brain.protocol import FakeBrain
from mtc.consciousness.continuous import (
    ContinuousThoughtMachine,
    InMemoryThoughtStore,
    LLMThoughtEnhancer,
    MemoryWeightRule,
    MongoThoughtStore,
    ThoughtBuffer,
    ThoughtGenerator,
    ThoughtStream,
    build_thought_machine,
)
from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryThoughtGistStore

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = ROOT / "mtc" / "consciousness" / "continuous"
GUARDED = ("ThoughtGenerator", "ThoughtStream", "ContinuousThoughtMachine")
PLANNED = GUARDED + (
    "ThoughtBuffer",
    "ThoughtRecord",
    "ThoughtType",
    "ThoughtIntensity",
    "CTMConfig",
    "SeedLibrary",
    "CTMPersona",
    "ThoughtStore",
    "InMemoryThoughtStore",
    "MongoThoughtStore",
    "NoveltyFilter",
    "LLMThoughtEnhancer",
    "build_thought_machine",
)
# Optional libraries no module of the package may import at module level.
LAZY = ("torch", "psutil", "pymongo", "sentence_transformers", "httpx")


def _settings(tmp_path, **values) -> Settings:
    values.setdefault("CTM_CONFIG_DIR", str(ROOT / "config" / "ctm"))
    values.setdefault("CTM_STORE", "memory")
    return Settings(PROJECT_ROOT=str(tmp_path), DEBUG=False, **values)


def test_exports_exact_guarded_names():
    for name in PLANNED:
        assert hasattr(package, name), name
        assert name in package.__all__, name
    assert ThoughtStream is ThoughtBuffer
    assert orch_mod._HAS_CTM is True
    assert orch_mod.ThoughtGenerator is ThoughtGenerator
    assert orch_mod.ThoughtStream is ThoughtStream
    assert orch_mod.ContinuousThoughtMachine is ContinuousThoughtMachine


def test_package_modules_import_no_optional_library_at_module_level():
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in LAZY, f"{path.name}: {name}"


def test_build_thought_machine_defaults_in_memory(tmp_path):
    machine = build_thought_machine(settings=_settings(tmp_path))
    assert isinstance(machine, ContinuousThoughtMachine)
    assert isinstance(machine.store, InMemoryThoughtStore)
    assert isinstance(machine.gist_store, InMemoryThoughtGistStore)
    assert isinstance(machine.generator, ThoughtGenerator)
    assert isinstance(machine.buffer, ThoughtBuffer)
    assert machine.enhancer is None
    assert machine.config.source == "file" and machine.config.profile == "default"
    assert machine.generator.seeds.source == "file"
    status = machine.get_status()
    assert status["novelty"]["embedder"] == "none"
    assert status["quiet_reason"] == "no_embedder"
    result = asyncio.run(machine.tick())
    assert result.generated is True and result.expressed is False

    with_brain = build_thought_machine(settings=_settings(tmp_path), brain=FakeBrain())
    assert isinstance(with_brain.enhancer, LLMThoughtEnhancer)


def _write_config(directory: Path) -> None:
    (directory / "seeds").mkdir(parents=True)
    (directory / "ctm.yaml").write_text(
        "buffer_size: 7\nprofiles:\n  demo:\n    cadence:\n      awake: 0.25\n",
        encoding="utf-8",
    )
    (directory / "seeds" / "default.yaml").write_text(
        'spontaneous_lines:\n  - "A plain test line."\n', encoding="utf-8"
    )


def test_build_reads_profile_config_dir_and_store(tmp_path):
    demo = build_thought_machine(settings=_settings(tmp_path, CTM_PROFILE="demo"))
    assert demo.config.profile == "demo"
    assert demo.config.cadence.awake >= 0.2
    assert demo.config.max_thoughts_per_minute <= 30
    assert demo.config.gist.every == 5

    custom = tmp_path / "custom_ctm"
    _write_config(custom)
    settings = _settings(tmp_path, CTM_CONFIG_DIR="custom_ctm", CTM_PROFILE="demo")
    machine = build_thought_machine(settings=settings)
    assert machine.config.path == custom / "ctm.yaml"
    assert machine.config.buffer_size == 7 and machine.buffer.max_buffer_size == 7
    assert machine.config.cadence.awake == 0.25
    thought = asyncio.run(machine.generator.generate_thought())
    assert thought.content == "A plain test line."

    from mtc.neural.mongodb_schemas import NeuralDataManager
    from tests.memory.test_mongo_stores import FakeMongoClient

    client = FakeMongoClient()
    mongo_settings = _settings(tmp_path, CTM_STORE="mongo")
    manager = NeuralDataManager(
        database_name="mtc_test_unit", client=client, settings=mongo_settings
    )
    mongo = build_thought_machine(settings=mongo_settings, data_manager=manager)
    assert isinstance(mongo.store, MongoThoughtStore)
    assert mongo.gist_store is manager
    result = asyncio.run(mongo.tick())
    assert result.persisted is True
    assert client.calls  # the write went to the recording fake, not a server


def test_mood_gate_reads_the_passed_settings(tmp_path, monkeypatch):
    import mtc.core.narration as narration

    on = _settings(tmp_path, NARRATE_THOUGHT_MOOD=True, NARRATE_ALL=False)
    off = _settings(tmp_path, NARRATE_THOUGHT_MOOD=False, NARRATE_ALL=False)
    everything = _settings(tmp_path, NARRATE_THOUGHT_MOOD=False, NARRATE_ALL=True)

    # the process settings say the opposite of the passed ones each time
    monkeypatch.setattr(narration, "get_settings", lambda: off)
    machine = build_thought_machine(settings=on, brain=FakeBrain())
    assert machine.enhancer._mood_allowed() is True
    machine = build_thought_machine(settings=everything, brain=FakeBrain())
    assert machine.enhancer._mood_allowed() is True

    monkeypatch.setattr(narration, "get_settings", lambda: on)
    machine = build_thought_machine(settings=off, brain=FakeBrain())
    assert machine.enhancer._mood_allowed() is False


def test_enhancer_debug_reads_the_passed_settings(tmp_path, monkeypatch):
    from mtc.consciousness.continuous import enhancer as enhancer_module

    loud = Settings(
        PROJECT_ROOT=str(tmp_path),
        CTM_CONFIG_DIR=str(ROOT / "config" / "ctm"),
        CTM_STORE="memory",
        DEBUG=True,
    )
    quiet = _settings(tmp_path)
    # the process settings say the opposite of the passed ones
    monkeypatch.setattr(enhancer_module, "get_settings", lambda: quiet)
    machine = build_thought_machine(
        settings=loud, brain=FakeBrain([RuntimeError("boom")])
    )
    assert machine.enhancer._debug() is True
    with pytest.raises(RuntimeError):
        asyncio.run(machine.enhancer.generate("reflection", {}, "awake"))


def test_weight_rules_from_settings(tmp_path):
    settings = _settings(
        tmp_path,
        SENSITIVE_MEMORY_PHRASES="first topic, second topic",
        SENSITIVE_MEMORY_REPLAY_WEIGHT=0.5,
    )
    machine = build_thought_machine(settings=settings)
    (rule,) = machine.generator.weight_rules
    assert isinstance(rule, MemoryWeightRule)
    assert rule.phrases == ("first topic", "second topic")
    assert rule.weight == 0.5
    assert rule.matches("about the First Topic today")
    assert not rule.matches("nothing related")
    assert (
        build_thought_machine(settings=_settings(tmp_path)).generator.weight_rules == []
    )


def test_allow_non_semantic_is_explicit(tmp_path):
    from mtc.core.embeddings import HashEmbedder

    quiet = build_thought_machine(
        settings=_settings(tmp_path), embedder=HashEmbedder(dim=32)
    )
    assert quiet.get_status()["quiet_reason"] == "non_semantic_embedder"
    opted = build_thought_machine(
        settings=_settings(tmp_path),
        embedder=HashEmbedder(dim=32),
        allow_non_semantic=True,
    )
    assert opted.get_status()["novelty"]["non_semantic_opt_in"] is True
    assert opted.get_status()["quiet_reason"] is None


@pytest.mark.parametrize("name", GUARDED)
def test_guarded_names_are_classes(name):
    assert isinstance(getattr(package, name), type)
