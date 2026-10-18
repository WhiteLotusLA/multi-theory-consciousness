"""Phase 6 Settings keys: providers, generation bounds, prompt caps, narration."""

import os
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtc.core.config import Settings, csv_list
from mtc.core.env_example import render_env_example

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_brain_defaults(clean_env):
    s = Settings()
    assert s.LLM_PROVIDER == "local"
    assert s.LLM_TEMPLATE_FAMILY == "generic"
    assert s.LLM_CONNECT_TIMEOUT_S == 10.0
    assert s.LLM_REQUEST_TIMEOUT_S == 600.0
    assert s.LLM_MAX_TOKENS == 4096
    assert s.GENERATION_TIMEOUT_S == 900.0
    assert s.LLM_MODEL == ""
    # Stored relative to LLM_BASE_URL (path defaults stay relative).
    assert s.LLM_COMPLETIONS_PATH == "v1/chat/completions"
    assert csv_list(s.LLM_HEALTH_PATHS) == ["/health", "/v1/models"]
    assert s.LLM_EXTRA_BODY == {}
    assert s.ANTHROPIC_MAX_TOKENS == 8192
    assert s.ANTHROPIC_THINKING_BUDGET == 0
    assert s.ANTHROPIC_SAMPLING_ENABLED is False
    assert s.HISTORY_TOKEN_BUDGET == 6000
    assert s.CHARS_PER_TOKEN == 3.5
    # The reference values, read from its configuration by grep.
    assert s.THINKING_LOOP_MAX_RETRIES == 2
    assert s.THINKING_LOOP_PENALTY_STEP == pytest.approx(0.2)


def test_base_url_derives_from_host_and_port(clean_env, monkeypatch):
    assert Settings().LLM_BASE_URL == Settings().LLM_URL
    monkeypatch.setenv("MTC_LLM_HOST", "host.invalid")
    monkeypatch.setenv("MTC_LLM_PORT", "9001")
    assert Settings().LLM_BASE_URL == "http://host.invalid:9001"
    monkeypatch.setenv("MTC_LLM_BASE_URL", "http://host.invalid:7000/")
    assert Settings().LLM_BASE_URL == "http://host.invalid:7000"


def test_provider_rejects_other_values(clean_env, monkeypatch):
    for choice in ("local", "anthropic", "mock"):
        monkeypatch.setenv("MTC_LLM_PROVIDER", choice)
        assert Settings().LLM_PROVIDER == choice
    for other in ("remote", "cloud", "Local", ""):
        monkeypatch.setenv("MTC_LLM_PROVIDER", other)
        with pytest.raises(ValidationError):
            Settings()
    # The accepted set is exactly the three supported providers.
    annotation = Settings.model_fields["LLM_PROVIDER"].annotation
    assert set(typing.get_args(annotation)) == {"local", "anthropic", "mock"}


def test_template_family_rejects_other_values(clean_env, monkeypatch):
    for family in ("generic", "qwen", "gemma"):
        monkeypatch.setenv("MTC_LLM_TEMPLATE_FAMILY", family)
        assert Settings().LLM_TEMPLATE_FAMILY == family
    monkeypatch.setenv("MTC_LLM_TEMPLATE_FAMILY", "other")
    with pytest.raises(ValidationError):
        Settings()


def test_extra_body_must_be_json_object(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LLM_EXTRA_BODY", '{"seed": 7, "cache_prompt": true}')
    assert Settings().LLM_EXTRA_BODY == {"seed": 7, "cache_prompt": True}
    monkeypatch.setenv("MTC_LLM_EXTRA_BODY", "")
    assert Settings().LLM_EXTRA_BODY == {}
    for bad in ("[1, 2]", '"text"', "not json", "{broken"):
        monkeypatch.setenv("MTC_LLM_EXTRA_BODY", bad)
        with pytest.raises(ValidationError, match="LLM_EXTRA_BODY"):
            Settings()
    assert Settings(LLM_EXTRA_BODY={"a": 1}).LLM_EXTRA_BODY == {"a": 1}


def test_thinking_budget_bounds(clean_env, monkeypatch):
    assert Settings(ANTHROPIC_THINKING_BUDGET=0).ANTHROPIC_THINKING_BUDGET == 0
    assert Settings(ANTHROPIC_THINKING_BUDGET=4096).ANTHROPIC_THINKING_BUDGET == 4096
    # A positive budget at or above ANTHROPIC_MAX_TOKENS is rejected.
    for bad in (8192, 9000):
        with pytest.raises(ValidationError, match="ANTHROPIC_THINKING_BUDGET"):
            Settings(ANTHROPIC_THINKING_BUDGET=bad)
    with pytest.raises(ValidationError, match="ANTHROPIC_THINKING_BUDGET"):
        Settings(ANTHROPIC_THINKING_BUDGET=2048, ANTHROPIC_MAX_TOKENS=2048)
    # The vendor's documented minimum for a fixed budget.
    with pytest.raises(ValidationError, match="1024"):
        Settings(ANTHROPIC_THINKING_BUDGET=512)
    with pytest.raises(ValidationError):
        Settings(ANTHROPIC_THINKING_BUDGET=-1)


def test_vision_key_is_gone_and_ignored(clean_env, monkeypatch):
    # The key had no reader; an env file that still sets it is ignored.
    assert "VISION_ENABLED" not in Settings.model_fields
    monkeypatch.setenv("MTC_VISION_ENABLED", "true")
    settings = Settings()
    assert not hasattr(settings, "VISION_ENABLED")
    assert "MTC_VISION_ENABLED" not in render_env_example(Settings)


def test_provider_state_path_derives_under_data_directory(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_DATA_DIRECTORY", str(clean_env / "data"))
    s = Settings()
    assert Path(s.PROVIDER_STATE_PATH) == clean_env / "data" / "provider_state.json"
    monkeypatch.setenv("MTC_PROVIDER_STATE_PATH", str(clean_env / "elsewhere.json"))
    assert Settings().PROVIDER_STATE_PATH == str(clean_env / "elsewhere.json")


def test_transition_dir_derives_under_results_dir(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_RESULTS_DIR", "lab_results")
    s = Settings()
    assert Path(s.TRANSITION_HISTORY_DIR) == Path("lab_results") / "model_transitions"
    monkeypatch.setenv("MTC_TRANSITION_HISTORY_DIR", "custom/transitions")
    assert Settings().TRANSITION_HISTORY_DIR == "custom/transitions"


def test_prompt_caps_positive(clean_env, monkeypatch):
    s = Settings()
    assert (
        s.PROMPT_MAX_THOUGHTS,
        s.PROMPT_MAX_GISTS,
        s.PROMPT_MAX_MEMORIES,
        s.PROMPT_MAX_MEMORY_CHARS,
        s.PROMPT_MAX_CONCEPTS,
    ) == (5, 5, 5, 250, 3)
    for key in (
        "PROMPT_MAX_THOUGHTS",
        "PROMPT_MAX_GISTS",
        "PROMPT_MAX_MEMORIES",
        "PROMPT_MAX_MEMORY_CHARS",
        "PROMPT_MAX_CONCEPTS",
        "HISTORY_TOKEN_BUDGET",
        "LLM_MAX_TOKENS",
        "ANTHROPIC_MAX_TOKENS",
    ):
        for bad in ("0", "-3"):
            monkeypatch.setenv(f"MTC_{key}", bad)
            with pytest.raises(ValidationError):
                Settings()
        monkeypatch.delenv(f"MTC_{key}")
    for key in (
        "CHARS_PER_TOKEN",
        "LLM_CONNECT_TIMEOUT_S",
        "LLM_REQUEST_TIMEOUT_S",
        "GENERATION_TIMEOUT_S",
    ):
        for bad in ("0", "-1.5", "nan"):
            monkeypatch.setenv(f"MTC_{key}", bad)
            with pytest.raises(ValidationError):
                Settings()
        monkeypatch.delenv(f"MTC_{key}")


def test_thinking_loop_bounds(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_THINKING_LOOP_MAX_RETRIES", "-1")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("MTC_THINKING_LOOP_MAX_RETRIES", "0")
    assert Settings().THINKING_LOOP_MAX_RETRIES == 0
    monkeypatch.setenv("MTC_THINKING_LOOP_PENALTY_STEP", "-0.1")
    with pytest.raises(ValidationError):
        Settings()


def test_llm_paths_are_url_paths(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LLM_COMPLETIONS_PATH", "/api/chat")
    assert Settings().LLM_COMPLETIONS_PATH == "api/chat"
    for bad in ("http://host.invalid/v1", "", "/", "v1 chat"):
        monkeypatch.setenv("MTC_LLM_COMPLETIONS_PATH", bad)
        with pytest.raises(ValidationError, match="LLM_COMPLETIONS_PATH"):
            Settings()
    monkeypatch.delenv("MTC_LLM_COMPLETIONS_PATH")
    monkeypatch.setenv("MTC_LLM_HEALTH_PATHS", "/health,models")
    assert csv_list(Settings().LLM_HEALTH_PATHS) == ["/health", "models"]
    monkeypatch.setenv("MTC_LLM_HEALTH_PATHS", "/health,http://host.invalid/x")
    with pytest.raises(ValidationError, match="LLM_HEALTH_PATHS"):
        Settings()


def test_disabled_sections_parse_with_csv_list(clean_env, monkeypatch):
    assert csv_list(Settings().PROMPT_DISABLED_SECTIONS) == []
    monkeypatch.setenv("MTC_PROMPT_DISABLED_SECTIONS", " memories, ,concepts ")
    assert csv_list(Settings().PROMPT_DISABLED_SECTIONS) == ["memories", "concepts"]


def test_narration_keys_default_false(clean_env):
    s = Settings()
    names = [name for name in Settings.model_fields if name.startswith("NARRATE_")]
    assert len(names) == 11
    assert all(getattr(s, name) is False for name in names)


def test_model_type_removed(clean_env):
    assert "MODEL_TYPE" not in Settings.model_fields
    assert "MTC_MODEL_TYPE" not in render_env_example(Settings)
    assert "MODEL_TYPE" not in (ROOT / "docs" / "CONFIGURATION.md").read_text()


def test_generated_env_example_parses(clean_env):
    (clean_env / ".env").write_text(render_env_example(Settings), encoding="utf-8")
    s = Settings()
    assert s.LLM_PROVIDER == "local"
    assert s.LLM_EXTRA_BODY == {}
    assert s.LLM_MAX_TOKENS == 4096
    assert s.HISTORY_TOKEN_BUDGET == 6000
    assert s.CHARS_PER_TOKEN == 3.5
    assert s.ANTHROPIC_MAX_TOKENS == 8192
    assert s.NARRATE_ALL is False
    assert Path(s.PROVIDER_STATE_PATH).name == "provider_state.json"


def test_phase6_keys_land_in_their_sections():
    rendered = render_env_example(Settings)
    assert "# === Other ===" not in rendered
    sections = {}
    current = ""
    for line in rendered.splitlines():
        if line.startswith("# === "):
            current = line[6:-4]
        elif line and not line.startswith("#"):
            sections[line.split("=", 1)[0]] = current
    for key in ("LLM_PROVIDER", "LLM_EXTRA_BODY", "ANTHROPIC_THINKING_BUDGET"):
        assert sections[f"MTC_{key}"] == "LLM", key
    for key in (
        "GENERATION_TIMEOUT_S",
        "THINKING_LOOP_MAX_RETRIES",
        "THINKING_LOOP_PENALTY_STEP",
        "PROVIDER_STATE_PATH",
        "HISTORY_TOKEN_BUDGET",
        "CHARS_PER_TOKEN",
        "PROMPT_MAX_THOUGHTS",
        "PROMPT_DISABLED_SECTIONS",
        "TRANSITION_HISTORY_DIR",
    ):
        assert sections[f"MTC_{key}"] == "Brain and prompt", key
    assert sections["MTC_NARRATE_ALL"] == "Narration"
    assert sections["MTC_NARRATE_SOMATIC"] == "Narration"


def test_numeric_token_keys_render_their_defaults():
    # Keys whose names contain TOKEN but hold numbers are not secrets.
    rendered = render_env_example(Settings)
    assert "MTC_HISTORY_TOKEN_BUDGET=6000" in rendered
    assert "MTC_CHARS_PER_TOKEN=3.5" in rendered
    assert "MTC_LLM_MAX_TOKENS=4096" in rendered
    assert "MTC_ANTHROPIC_MAX_TOKENS=8192" in rendered
    assert "ANTHROPIC_API_KEY=" in rendered.splitlines()
