"""Path resolution over the path-like Settings keys."""

import os
from pathlib import Path

import pytest

from mtc.core import paths
from mtc.core.config import Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_relative_paths_resolve_under_project_root(clean_env, monkeypatch):
    root = clean_env / "root"
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(root))
    settings = Settings()
    assert paths.project_root(settings) == root
    assert paths.log_dir(settings) == root / "logs"
    assert paths.data_dir(settings) == root / "data"
    assert paths.results_dir(settings) == root / "results"
    assert paths.agent_space_dir(settings) == root / "agent_space"
    assert paths.state_dir(settings) == root / "agent_space" / "consciousness_state"
    assert paths.shadow_log_dir(settings) == root / "logs" / "shadow_signals"
    assert paths.resolve_path("sub/dir", settings=settings) == root / "sub" / "dir"


def test_absolute_paths_unchanged(clean_env, monkeypatch):
    absolute = clean_env / "elsewhere"
    monkeypatch.setenv("MTC_RESULTS_DIR", str(absolute))
    settings = Settings()
    assert paths.results_dir(settings) == absolute
    assert paths.resolve_path(absolute, settings=settings) == absolute
    home = paths.resolve_path("~/x", settings=settings)
    assert home == Path.home() / "x"


@pytest.fixture
def fresh_singleton():
    from mtc.core import config

    config.reset_settings()
    yield config
    config.reset_settings()


def test_settings_read_at_call_time(clean_env, monkeypatch, fresh_singleton):
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env / "one"))
    fresh_singleton.reset_settings()
    assert paths.log_dir() == clean_env / "one" / "logs"
    # The process-wide instance is built once: without a reset the helpers
    # keep reading it.
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env / "two"))
    monkeypatch.setenv("MTC_LOG_DIRECTORY", "var/log")
    assert paths.log_dir() == clean_env / "one" / "logs"
    fresh_singleton.reset_settings()
    assert paths.log_dir() == clean_env / "two" / "var" / "log"


def test_display_path(clean_env, monkeypatch, fresh_singleton):
    root = clean_env / "root"
    settings = Settings(PROJECT_ROOT=str(root))
    inside = root / "data" / "privacy" / "privacy_audit.jsonl"
    cases = {
        inside: "data/privacy/privacy_audit.jsonl",
        str(inside): "data/privacy/privacy_audit.jsonl",
        "data/consent.json": "data/consent.json",
        clean_env / "elsewhere" / "consent.json": "consent.json",
        root / ".." / "escape" / "f.txt": "f.txt",
        "~/x/home_file.txt": "home_file.txt",
    }
    for given, expected in cases.items():
        shown = paths.display_path(given, settings=settings)
        assert isinstance(shown, str), given
        assert shown == expected, given
        assert not os.path.isabs(shown), given
    # Without a Settings object the process-wide instance supplies the root.
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(root))
    fresh_singleton.reset_settings()
    assert paths.display_path(root / "logs" / "a.log") == "logs/a.log"
    assert paths.display_path(clean_env / "b.log") == "b.log"


def test_resolution_creates_no_directory(clean_env, monkeypatch):
    root = clean_env / "untouched"
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(root))
    settings = Settings()
    for fn in (
        paths.log_dir,
        paths.data_dir,
        paths.results_dir,
        paths.agent_space_dir,
        paths.state_dir,
        paths.shadow_log_dir,
    ):
        fn(settings)
    assert not root.exists()
    made = paths.ensure_dir(root / "a" / "b")
    assert made.is_dir()
    assert paths.ensure_dir(made) == made


def test_path_defaults_are_relative(clean_env):
    settings = Settings()
    checked = []
    for name in type(settings).model_fields:
        if not (
            name.endswith("_DIR")
            or name.endswith("_DIRECTORY")
            or name.endswith("_PATH")
        ):
            continue
        value = getattr(settings, name)
        if name == "PROJECT_ROOT":
            continue
        checked.append(name)
        assert not os.path.isabs(value), name
        assert "~" not in value, name
    assert "LOG_DIRECTORY" in checked and "STATE_DIR" in checked
