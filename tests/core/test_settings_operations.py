"""The operations Settings keys: the service manager, backups, the LLM launcher
backend, the MongoDB authentication database, the changed defaults, the safety
gate, emergency shutdown, trust compass and sleep dry-run keys and a generated
example that parses with no key left under Other."""

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtc.core.config import Settings
from mtc.core.env_example import SECTIONS, _section_for, render_env_example

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_service_manager_choices(clean_env):
    assert Settings().SERVICE_MANAGER == "none"
    for value in ("none", "brew", "docker", "systemd"):
        assert Settings(SERVICE_MANAGER=value).SERVICE_MANAGER == value
    for bad in ("", "launchd", "Docker", "podman"):
        with pytest.raises(ValidationError):
            Settings(SERVICE_MANAGER=bad)


def test_llm_backend_choices_and_empty_default(clean_env):
    assert Settings().LLM_BACKEND == ""
    for value in ("", "mlx_lm", "llama_cpp"):
        assert Settings(LLM_BACKEND=value).LLM_BACKEND == value
    for bad in ("ollama", "mlx", "llama.cpp", "vllm"):
        with pytest.raises(ValidationError):
            Settings(LLM_BACKEND=bad)


def test_backup_defaults_and_bounds(clean_env):
    settings = Settings()
    assert settings.BACKUP_RETENTION_DAYS == 30
    assert settings.BACKUP_KEEP_MINIMUM == 5
    assert Settings(BACKUP_RETENTION_DAYS=1).BACKUP_RETENTION_DAYS == 1
    assert Settings(BACKUP_KEEP_MINIMUM=1).BACKUP_KEEP_MINIMUM == 1
    for key in ("BACKUP_RETENTION_DAYS", "BACKUP_KEEP_MINIMUM"):
        for bad in (0, -3):
            with pytest.raises(ValidationError):
                Settings(**{key: bad})


def test_backup_location_relative_and_gitignored(clean_env):
    location = Settings().BACKUP_LOCATION
    assert location == "backups"
    assert not Path(location).is_absolute()
    probe = f"{location}/20000101T000000Z/MANIFEST.txt"
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", probe],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    if result.returncode == 128:
        pytest.skip("not inside a git checkout")
    assert result.returncode == 0, f"{probe} is not ignored by .gitignore"


def test_mongodb_auth_source_default_empty(clean_env):
    assert Settings().MONGODB_AUTH_SOURCE == ""
    assert Settings(MONGODB_AUTH_SOURCE="admin").MONGODB_AUTH_SOURCE == "admin"


def test_changed_defaults(clean_env):
    settings = Settings()
    assert settings.GPU_LAYERS == -1
    assert settings.THREADS == 0
    assert settings.USE_MLOCK is False
    assert settings.CONTEXT_SIZE == 32768
    assert settings.DCM_CACHE_DIR == "data/dcm_cache"
    # The history budget stays the smaller term of the brain's budget.
    assert settings.HISTORY_TOKEN_BUDGET < settings.CONTEXT_SIZE
    with pytest.raises(ValidationError):
        Settings(GPU_LAYERS=-2)
    with pytest.raises(ValidationError):
        Settings(THREADS=-1)


def test_safety_shutdown_compass_and_sleep_dry_run_keys(clean_env, monkeypatch):
    settings = Settings()
    assert settings.SLEEP_PRUNE_DRY_RUN is False
    assert settings.SAFETY_GATE_FAIL_CLOSED is True
    assert settings.TRUST_COMPASS_DEEP_MODE is True
    assert settings.EMERGENCY_SHUTDOWN_ENABLED is False
    fields = Settings.model_fields
    for name in (
        "SLEEP_PRUNE_DRY_RUN",
        "SAFETY_GATE_FAIL_CLOSED",
        "TRUST_COMPASS_DEEP_MODE",
        "EMERGENCY_SHUTDOWN_ENABLED",
    ):
        assert fields[name].annotation is bool, name
        assert fields[name].description, name
    emergency = fields["EMERGENCY_SHUTDOWN_ENABLED"].description
    assert "suspends every untrusted conversation" in emergency
    assert "denies the message" in emergency
    assert _section_for("SLEEP_PRUNE_DRY_RUN") == "Runtime"
    assert _section_for("SAFETY_GATE_FAIL_CLOSED") == "Safety and API"
    assert _section_for("TRUST_COMPASS_DEEP_MODE") == "Safety and API"
    for name in (
        "TRUSTED_SPEAKERS",
        "TRUST_ACTIVATION_BONUS",
        "TRUSTED_EMOTION_TRUST",
        "SPEAKER_TOKENS",
        "DEFAULT_SPEAKER_TRUST",
    ):
        assert _section_for(name) == "Speakers and trust", name
    for name in ("SLEEP_PRUNE_DRY_RUN", "TRUST_COMPASS_DEEP_MODE"):
        monkeypatch.setenv(f"MTC_{name}", "true")
    monkeypatch.setenv("MTC_SAFETY_GATE_FAIL_CLOSED", "false")
    monkeypatch.setenv("MTC_EMERGENCY_SHUTDOWN_ENABLED", "true")
    loaded = Settings()
    assert loaded.SLEEP_PRUNE_DRY_RUN is True
    assert loaded.SAFETY_GATE_FAIL_CLOSED is False
    assert loaded.EMERGENCY_SHUTDOWN_ENABLED is True
    monkeypatch.setenv("MTC_TRUST_COMPASS_DEEP_MODE", "false")
    assert Settings().TRUST_COMPASS_DEEP_MODE is False


def test_phi_idle_token_budget_and_safety_gate_descriptions():
    fields = Settings.model_fields
    phi = fields["PHI_IDLE_SAMPLE_S"].description
    assert "idle samples repeat the last state" in phi
    assert "pull the proxy toward its floor" in phi
    budget = fields["ANTHROPIC_MAX_TOKENS"].description
    assert "direct provider call" in budget
    assert "conversation turns use LLM_MAX_TOKENS" in budget
    # The gate state is unchecked whichever value the key has; only the
    # outcome of the turn depends on it.
    gate = fields["SAFETY_GATE_FAIL_CLOSED"].description
    assert "the turn record says unchecked either way" in gate
    assert "false processes the message and the turn record says" not in gate


def test_no_key_lands_under_other():
    assert "Operations" in [title for title, _ in SECTIONS]
    other = [name for name in Settings.model_fields if _section_for(name) == "Other"]
    assert other == []
    assert _section_for("SERVICE_MANAGER") == "Operations"
    for name in ("BACKUP_LOCATION", "BACKUP_RETENTION_DAYS", "BACKUP_KEEP_MINIMUM"):
        assert _section_for(name) == "Operations", name
    assert _section_for("LLM_BACKEND") == "LLM"
    assert _section_for("MONGODB_AUTH_SOURCE") == "Databases"
    assert "# === Other ===" not in render_env_example(Settings)


def test_generated_env_example_parses(clean_env, tmp_path):
    rendered = render_env_example(Settings)
    for line in (
        "MTC_SERVICE_MANAGER=none",
        "MTC_LLM_BACKEND=",
        "MTC_BACKUP_LOCATION=backups",
        "MTC_BACKUP_RETENTION_DAYS=30",
        "MTC_BACKUP_KEEP_MINIMUM=5",
        "MTC_MONGODB_AUTH_SOURCE=",
        "MTC_GPU_LAYERS=-1",
        "MTC_THREADS=0",
        "MTC_USE_MLOCK=false",
        "MTC_CONTEXT_SIZE=32768",
        "MTC_DCM_CACHE_DIR=data/dcm_cache",
    ):
        assert line in rendered.splitlines(), line
    env_file = tmp_path / "generated.env"
    env_file.write_text(rendered, encoding="utf-8")
    loaded = Settings(_env_file=str(env_file))
    defaults = Settings(_env_file=None)
    for key in (
        "SERVICE_MANAGER",
        "LLM_BACKEND",
        "BACKUP_LOCATION",
        "BACKUP_RETENTION_DAYS",
        "BACKUP_KEEP_MINIMUM",
        "MONGODB_AUTH_SOURCE",
        "GPU_LAYERS",
        "THREADS",
        "USE_MLOCK",
        "CONTEXT_SIZE",
        "DCM_CACHE_DIR",
    ):
        assert getattr(loaded, key) == getattr(defaults, key), key
