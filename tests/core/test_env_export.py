"""Settings export for shell scripts: quoted values, derived names, refused
secrets, exit 2 on an unknown field and a bash round trip."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.core import env_export
from mtc.core.config import Settings
from mtc.core.env_export import (
    SecretRefused,
    UnknownField,
    export_lines,
    is_secret_field,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_exports_resolved_values_with_shell_quoting(clean_env):
    settings = Settings(
        SYSTEM_NAME="Two words; and $more",
        POSTGRES_ENABLED=True,
        GPU_LAYERS=-1,
        LLM_EXTRA_BODY={"b": 1, "a": True},
    )
    lines = export_lines(
        [
            "SYSTEM_NAME",
            "POSTGRES_ENABLED",
            "MONGODB_ENABLED",
            "GPU_LAYERS",
            "LLM_EXTRA_BODY",
            "LLM_URL",
        ],
        settings=settings,
    )
    assert lines[0] == "export MTC_SYSTEM_NAME='Two words; and $more'"
    assert lines[1] == "export MTC_POSTGRES_ENABLED=true"
    assert lines[2] == "export MTC_MONGODB_ENABLED=false"
    assert lines[3] == "export MTC_GPU_LAYERS=-1"
    assert lines[4] == """export MTC_LLM_EXTRA_BODY='{"a": true, "b": 1}'"""
    # A computed field resolves from the loaded values.
    assert lines[5] == f"export MTC_LLM_URL={settings.LLM_URL}"


def test_derived_store_names_exported(clean_env):
    settings = Settings(DB_PREFIX="lab")
    lines = export_lines(
        [
            "POSTGRES_DB",
            "POSTGRES_USER",
            "MONGODB_DB",
            "QDRANT_COLLECTION",
            "REDIS_KEY_PREFIX",
        ],
        settings=settings,
    )
    assert lines == [
        "export MTC_POSTGRES_DB=lab_consciousness",
        "export MTC_POSTGRES_USER=lab",
        "export MTC_MONGODB_DB=lab_neural",
        "export MTC_QDRANT_COLLECTION=lab_memories",
        "export MTC_REDIS_KEY_PREFIX=lab:",
    ]


def test_secrets_refused_without_flag(clean_env, monkeypatch, capsys):
    secret = "pw" + "7" * 30
    settings = Settings(POSTGRES_PASSWORD=secret, API_KEY=secret)
    for name in (
        "POSTGRES_PASSWORD",
        "API_KEY",
        "SPEAKER_TOKENS",
        "ANTHROPIC_API_KEY",
        "POSTGRES_URL",
        "MONGODB_URL",
        "REDIS_URL",
    ):
        assert is_secret_field(name), name
        with pytest.raises(SecretRefused):
            export_lines(["POSTGRES_DB", name], settings=settings)
    # Numeric fields whose names hold a marker are not secrets.
    for name in ("HISTORY_TOKEN_BUDGET", "LLM_MAX_TOKENS", "ANTHROPIC_MAX_TOKENS"):
        assert not is_secret_field(name), name
    allowed = export_lines(["POSTGRES_PASSWORD"], settings=settings, secrets=True)
    assert allowed == [f"export MTC_POSTGRES_PASSWORD={secret}"]
    monkeypatch.setenv("MTC_POSTGRES_PASSWORD", secret)
    code = env_export.main(["POSTGRES_DB", "POSTGRES_PASSWORD"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert secret not in captured.err
    assert "POSTGRES_PASSWORD" in captured.err and "--secrets" in captured.err


def test_unknown_field_exits_2(clean_env, capsys):
    with pytest.raises(UnknownField):
        export_lines(["NOT_A_FIELD"], settings=Settings())
    assert env_export.main(["POSTGRES_DB", "NOT_A_FIELD"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "NOT_A_FIELD" in captured.err
    assert env_export.main(["POSTGRES_DB"]) == 0
    assert capsys.readouterr().out == "export MTC_POSTGRES_DB=mtc_consciousness\n"


def test_invalid_settings_exit_2_naming_the_field_only(clean_env, monkeypatch, capsys):
    monkeypatch.setenv("MTC_SERVICE_MANAGER", "not-a-manager-value")
    assert env_export.main(["POSTGRES_DB"]) == 2
    captured = capsys.readouterr()
    assert "SERVICE_MANAGER" in captured.err
    assert "not-a-manager-value" not in captured.err


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_bash_eval_round_trip(clean_env):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MTC_") and key != "ANTHROPIC_API_KEY"
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_SYSTEM_NAME": "It's a $name with spaces",
            "MTC_DB_PREFIX": "roundtrip",
        }
    )
    script = (
        'eval "$("$PYTHON" -m mtc.core.env_export SYSTEM_NAME POSTGRES_DB '
        'POSTGRES_ENABLED)"; '
        'printf "%s|%s|%s" "$MTC_SYSTEM_NAME" "$MTC_POSTGRES_DB" '
        '"$MTC_POSTGRES_ENABLED"'
    )
    env["PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=clean_env,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout == "It's a $name with spaces|roundtrip_consciousness|false"
