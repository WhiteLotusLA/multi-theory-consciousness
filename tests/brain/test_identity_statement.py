"""Identity statement loader: call-time read, neutral template, size guard."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

from mtc.brain import identity_statement
from mtc.brain.identity_statement import load_identity_statement
from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "config" / "identity_statement.example.md"


def test_missing_file_returns_empty(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=identity_statement.__name__):
        assert load_identity_statement(tmp_path / "absent.md") == ""
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_reads_configured_path_at_call_time(tmp_path, monkeypatch):
    target = tmp_path / "statement.md"
    state = {"path": str(target)}
    monkeypatch.setattr(
        identity_statement,
        "get_settings",
        lambda: SimpleNamespace(IDENTITY_STATEMENT_PATH=state["path"]),
    )
    assert load_identity_statement() == ""
    target.write_text("  first version \n", encoding="utf-8")
    assert load_identity_statement() == "first version"
    target.write_text("second version", encoding="utf-8")
    assert load_identity_statement() == "second version"
    other = tmp_path / "other.md"
    other.write_text("other text", encoding="utf-8")
    state["path"] = str(other)
    assert load_identity_statement() == "other text"
    assert load_identity_statement(target) == "second version"


def test_default_path_under_agent_space(tmp_path, monkeypatch):
    from mtc.core.config import Settings

    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings(AGENT_SPACE_DIR=str(tmp_path / "space"))
    assert settings.IDENTITY_STATEMENT_PATH == str(
        tmp_path / "space" / "identity_statement.md"
    )
    (tmp_path / "space").mkdir()
    (tmp_path / "space" / "identity_statement.md").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(identity_statement, "get_settings", lambda: settings)
    assert load_identity_statement() == "hello"


def test_oversized_file_rejected_with_warning(tmp_path, caplog):
    big = tmp_path / "big.md"
    big.write_text("x" * 101, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=identity_statement.__name__):
        assert load_identity_statement(big, max_chars=100) == ""
    assert [r for r in caplog.records if r.levelno == logging.WARNING]
    exact = tmp_path / "exact.md"
    exact.write_text("y" * 100, encoding="utf-8")
    assert load_identity_statement(exact, max_chars=100) == "y" * 100


def test_undecodable_file_returns_empty(tmp_path):
    broken = tmp_path / "broken.md"
    broken.write_bytes(b"\xff\xfe\xfa bad bytes")
    assert load_identity_statement(broken) == ""


def test_example_template_is_neutral():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "{system_name}" in text and "{creator_name}" in text
    lowered = " ".join(text.lower().split())
    assert "ai research system" in lowered
    assert "do not claim certainty about your own consciousness" in lowered
    denylist = load_denylist(ROOT / "config" / "privacy" / "denylist.txt")
    allowlist = load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt")
    assert run_audit([EXAMPLE], denylist, allowlist, root=ROOT) == []
    assert load_identity_statement(EXAMPLE).startswith("# Identity statement")


def test_log_lines_carry_the_file_name_never_its_directory(tmp_path, caplog):
    """The statement's directory can hold the machine's home path or a user
    name; log lines name the file and the key only."""
    absent = tmp_path / "space" / "absent_statement.md"
    with caplog.at_level(logging.INFO, logger=identity_statement.__name__):
        assert load_identity_statement(absent) == ""
    (info,) = caplog.records
    assert "absent_statement.md" in info.getMessage()
    assert "IDENTITY_STATEMENT_PATH" in info.getMessage()
    assert str(tmp_path) not in info.getMessage()

    caplog.clear()
    large = tmp_path / "space" / "large_statement.md"
    large.parent.mkdir(parents=True, exist_ok=True)
    large.write_text("x" * 50, encoding="utf-8")
    with caplog.at_level(logging.INFO, logger=identity_statement.__name__):
        assert load_identity_statement(large, max_chars=10) == ""
    (warning,) = caplog.records
    assert "large_statement.md" in warning.getMessage()
    assert str(tmp_path) not in warning.getMessage()
