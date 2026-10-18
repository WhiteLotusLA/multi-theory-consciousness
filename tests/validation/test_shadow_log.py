"""Shadow signal log: append-only JSONL that never raises."""

import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from mtc.validation import shadow_log
from mtc.validation.shadow_log import ShadowSignalLogger


def test_appends_jsonl_with_aware_utc_timestamps(tmp_path):
    log = ShadowSignalLogger("sig_a", tmp_path / "out")
    assert log.log({"values": {"x": 1.0}}) is True
    assert log.log({"values": {"x": 2.0}}) is True
    lines = (tmp_path / "out" / "sig_a.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["signal_id"] == "sig_a"
    assert first["values"] == {"x": 1.0}
    ts = datetime.fromisoformat(first["ts"])
    assert ts.utcoffset() == timedelta(0)


def test_never_raises_on_unwritable_directory(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    log = ShadowSignalLogger("sig", blocker / "child")
    with caplog.at_level("WARNING"):
        assert log.log({"a": 1}) is False
    assert "shadow log write failed" in caplog.text


def test_non_json_or_non_finite_record_returns_false(tmp_path):
    log = ShadowSignalLogger("sig", tmp_path)
    assert log.log({"a": float("nan")}) is False
    assert not (tmp_path / "sig.jsonl").exists()
    assert log.log({"a": {1, 2}}) is True  # sets fall back to their string form


def test_directory_from_settings_at_call_time(tmp_path, monkeypatch):
    # Outside pytest the settings directory is written; lift the guard here.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    first = tmp_path / "first"
    second = tmp_path / "second"
    current = {"dir": str(first)}
    monkeypatch.setattr(
        shadow_log,
        "get_settings",
        lambda: SimpleNamespace(
            SHADOW_LOG_DIR=current["dir"], LOG_DIRECTORY=str(tmp_path / "logs")
        ),
    )
    log = ShadowSignalLogger("sig")
    assert log.log({"n": 1})
    current["dir"] = str(second)
    assert log.log({"n": 2})
    assert (first / "sig.jsonl").exists()
    assert (second / "sig.jsonl").exists()


def test_empty_setting_derives_from_log_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shadow_log,
        "get_settings",
        lambda: SimpleNamespace(
            SHADOW_LOG_DIR="", LOG_DIRECTORY=str(tmp_path / "logs")
        ),
    )
    assert ShadowSignalLogger("sig").directory == tmp_path / "logs" / "shadow_signals"


def test_no_directory_created_until_first_write(tmp_path):
    target = tmp_path / "later"
    log = ShadowSignalLogger("sig", target)
    assert not target.exists()
    assert log.directory == target
    assert not target.exists()
    log.log({"a": 1})
    assert target.is_dir()


def test_signal_id_must_be_a_plain_stem(tmp_path):
    for bad in ("", "a/b", "..", "a\\b"):
        with pytest.raises(ValueError):
            ShadowSignalLogger(bad, tmp_path)


def test_default_directory_skipped_under_pytest(tmp_path, monkeypatch, caplog):
    """Under pytest only an explicit directory is written: the gates read the
    settings directory as live evidence, so test runs must not add to it."""
    target = tmp_path / "live"
    monkeypatch.setattr(
        shadow_log,
        "get_settings",
        lambda: SimpleNamespace(
            SHADOW_LOG_DIR=str(target), LOG_DIRECTORY=str(tmp_path / "logs")
        ),
    )
    assert "PYTEST_CURRENT_TEST" in shadow_log.os.environ
    with caplog.at_level("WARNING"):
        assert ShadowSignalLogger("sig").log({"n": 1}) is False
    assert not target.exists()
    assert "shadow log write failed" not in caplog.text
    explicit = tmp_path / "explicit"
    assert ShadowSignalLogger("sig", explicit).log({"n": 1}) is True
    assert (explicit / "sig.jsonl").exists()
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert ShadowSignalLogger("sig").log({"n": 2}) is True
    assert (target / "sig.jsonl").exists()


@pytest.fixture
def fresh_settings():
    """Rebuild the process-wide Settings around the test, never keep its copy."""
    from mtc.core.config import reset_settings

    reset_settings()
    yield reset_settings
    reset_settings()


def test_default_directory_is_resolved_against_project_root(
    tmp_path, monkeypatch, fresh_settings
):
    from mtc.core.paths import shadow_log_dir

    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    root = tmp_path / "root"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(root))
    monkeypatch.chdir(elsewhere)
    fresh_settings()
    directory = ShadowSignalLogger("sig").directory
    assert directory == root / "logs" / "shadow_signals"
    assert directory == shadow_log_dir()
    assert directory.is_absolute()
    # a relative SHADOW_LOG_DIR joins PROJECT_ROOT, not the working directory
    monkeypatch.setenv("MTC_SHADOW_LOG_DIR", "custom/shadow")
    fresh_settings()
    assert ShadowSignalLogger("sig").directory == root / "custom" / "shadow"
