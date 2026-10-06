"""Each loader of a shipped default falls back to its packaged copy when the
project has none (an empty temporary PROJECT_ROOT stands in for a wheel
install), and a project copy wins over the packaged one.

Loaders whose shipped step names a file next to the code (the persona
example, the transition prompts, the dreams config) are pointed at a missing
file of the same name, which is what an install from a wheel sees."""

import pytest

from mtc.brain import model_transition, prompt
from mtc.brain.prompt import PromptAssembler
from mtc.consciousness import dream_engine
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.persona import CTMPersona
from mtc.consciousness.continuous.seeds import SeedLibrary
from mtc.core.config import Settings
from mtc.core.param_overrides import load_overrides
from mtc.core.resources import packaged_path, resolve_shipped
from mtc.pipeline.runtime import load_pipeline_overrides


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=str(project),
        PERSONA_PATH=str(project / "config" / "persona.yaml"),
        SYSTEM_NAME="Testagent",
    )
    from mtc.core import config as config_module

    monkeypatch.setattr(config_module, "_settings", settings)
    monkeypatch.chdir(project)
    return settings


def _wheel_layout(tmp_path, name):
    """A missing file named like the shipped one, outside any checkout."""
    return tmp_path / "site-packages" / "config" / name


def test_ctm_config_falls_back_to_package(empty_project):
    config = CTMConfig.load(settings=empty_project)
    assert config.source == "package"
    assert config.path == packaged_path("config/ctm/ctm.yaml")
    # An operator's own directory is never filled from the package.
    own = CTMConfig.load(
        empty_project.PROJECT_ROOT + "/elsewhere", settings=empty_project
    )
    assert own.source == "defaults"


def test_seed_library_falls_back_to_package(empty_project):
    library = SeedLibrary.from_config_dir(settings=empty_project)
    assert library.source == "package"
    assert library.packs == [packaged_path("config/ctm/seeds/default.yaml")]
    assert library.keys()


def test_ctm_persona_falls_back_to_package(empty_project):
    persona = CTMPersona.load(settings=empty_project)
    assert persona.source == "package"
    assert persona.path == packaged_path("config/ctm/persona.example.yaml")


def test_persona_example_falls_back_to_package(empty_project, tmp_path, monkeypatch):
    monkeypatch.setattr(
        prompt,
        "PERSONA_EXAMPLE_PATH",
        _wheel_layout(tmp_path, "persona.example.yaml"),
    )
    assembler = PromptAssembler(settings=empty_project, identity_statement="")
    assert assembler.persona_source == "package"
    assert assembler.persona["speaker_templates"]["unverified"]
    assert "package" in prompt.PERSONA_SOURCES


def test_transition_prompts_fall_back_to_package(empty_project, tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_transition,
        "DEFAULT_PROMPTS_PATH",
        _wheel_layout(tmp_path, "transition_prompts.json"),
    )
    protocol = model_transition.ModelTransitionProtocol(assess=None, brain=None)
    prompts = protocol.load_prompts()
    assert prompts
    assert protocol.prompts_source == "package"


def test_dreams_fall_back_to_package(empty_project, tmp_path, monkeypatch):
    monkeypatch.setattr(
        dream_engine, "DEFAULT_CONFIG_PATH", _wheel_layout(tmp_path, "dreams.yaml")
    )
    config = dream_engine.load_dream_config()
    assert config.source == "package"
    assert config.templates
    engine = dream_engine.DreamEngine()
    assert engine.config.source == "package"


def test_runtime_preset_and_schema_fall_back_to_package(empty_project):
    for source in ("experiments/presets", "experiments/param_schema.toml"):
        assert resolve_shipped(source, settings=empty_project)[1] == "package"
    chosen = empty_project.model_copy(update={"PIPELINE_PRESET": "legacy_v04"})
    overrides = load_pipeline_overrides(chosen)
    expected = load_overrides(
        str(packaged_path("experiments/presets/legacy_v04.toml")),
        str(packaged_path("experiments/param_schema.toml")),
    )
    assert overrides == expected and overrides


def test_project_copy_wins_over_package(empty_project, tmp_path, monkeypatch):
    project = tmp_path / "project"
    ctm_dir = project / "config" / "ctm"
    ctm_dir.mkdir(parents=True)
    (ctm_dir / "ctm.yaml").write_text("contention_window_s: 7\n", encoding="utf-8")
    config = CTMConfig.load(settings=empty_project)
    assert config.source == "file"
    assert config.contention_window_s == 7
    (project / "config" / "dreams.yaml").write_text(
        packaged_path("config/dreams.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dream_engine, "DEFAULT_CONFIG_PATH", _wheel_layout(tmp_path, "dreams.yaml")
    )
    assert dream_engine.load_dream_config().source == "file"
    path, where = resolve_shipped("config/dreams.yaml", settings=empty_project)
    assert where == "project" and path == project / "config" / "dreams.yaml"
