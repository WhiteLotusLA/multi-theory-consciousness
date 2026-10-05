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


def test_settings_read_at_call_time(clean_env, monkeypatch):
    monkeypatch.setattr(paths, "get_settings", lambda: Settings())
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env / "one"))
    assert paths.log_dir() == clean_env / "one" / "logs"
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env / "two"))
    monkeypatch.setenv("MTC_LOG_DIRECTORY", "var/log")
    assert paths.log_dir() == clean_env / "two" / "var" / "log"


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
        if name == "TEMP_DIRECTORY":
            continue
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
