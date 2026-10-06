"""The .env helpers: atomic upserts that keep comments, mode 0600, values that
python-dotenv reads back unchanged and never appear in a log."""

import logging
import os
import stat

import pytest

from mtc.core import envfile
from mtc.core.envfile import (
    UnsafeEnvValue,
    env_mode_ok,
    read_env,
    render_value,
    upsert_env,
)

EXISTING = """# Project settings
MTC_SYSTEM_NAME=Tessera

# Stores
MTC_POSTGRES_ENABLED=false
UNRELATED_TOOL_FLAG=1
# trailing comment
"""


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_upsert_preserves_comments_and_unrelated_lines(tmp_path):
    path = tmp_path / "settings.env"
    path.write_text(EXISTING, encoding="utf-8")
    written = upsert_env(
        path,
        {"MTC_POSTGRES_ENABLED": "true", "MTC_TIMEZONE": "UTC"},
        header="Added by the test",
    )
    assert written == ["MTC_POSTGRES_ENABLED", "MTC_TIMEZONE"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:7] == [
        "# Project settings",
        "MTC_SYSTEM_NAME=Tessera",
        "",
        "# Stores",
        "MTC_POSTGRES_ENABLED=true",
        "UNRELATED_TOOL_FLAG=1",
        "# trailing comment",
    ]
    assert lines[7:] == ["", "# Added by the test", "MTC_TIMEZONE=UTC"]
    # A second identical upsert changes no byte.
    before = path.read_bytes()
    upsert_env(path, {"MTC_POSTGRES_ENABLED": "true", "MTC_TIMEZONE": "UTC"})
    assert path.read_bytes() == before


def test_upsert_creates_file_mode_0600(tmp_path):
    path = tmp_path / "fresh.env"
    upsert_env(path, {"MTC_API_PORT": "8999"})
    assert _mode(path) == 0o600
    assert env_mode_ok(path)
    # A file that was readable by others ends at 0600 after an upsert.
    loose = tmp_path / "loose.env"
    loose.write_text("A=1\n", encoding="utf-8")
    os.chmod(loose, 0o644)
    assert not env_mode_ok(loose)
    upsert_env(loose, {"B": "2"})
    assert _mode(loose) == 0o600
    assert env_mode_ok(loose)
    assert not env_mode_ok(tmp_path / "absent.env")


def test_keep_existing_never_overwrites(tmp_path):
    path = tmp_path / "keep.env"
    path.write_text("MTC_API_KEY=already-there\nMTC_EMPTY_KEY=\n", encoding="utf-8")
    written = upsert_env(
        path,
        {
            "MTC_API_KEY": "new-value",
            "MTC_EMPTY_KEY": "filled",
            "MTC_FRESH_KEY": "added",
        },
        keep_existing={"MTC_API_KEY", "MTC_EMPTY_KEY", "MTC_FRESH_KEY"},
    )
    values = read_env(path)
    assert values["MTC_API_KEY"] == "already-there"
    # An empty existing value counts as absent and is filled.
    assert values["MTC_EMPTY_KEY"] == "filled"
    assert values["MTC_FRESH_KEY"] == "added"
    assert "MTC_API_KEY" not in written
    assert set(written) == {"MTC_EMPTY_KEY", "MTC_FRESH_KEY"}


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain",
        "a1_b2.c3:d4/e5,f6@g7+h8=i9-j0",
        "with spaces inside",
        " leading and trailing ",
        "hash # inside",
        "single 'quotes' kept",
        "semi;colon&amp*star?",
        "accented caf" + chr(0xE9),
    ],
)
def test_values_round_trip_through_dotenv(tmp_path, value):
    path = tmp_path / "round.env"
    upsert_env(path, {"MTC_VALUE": value, "MTC_AFTER": "next"})
    values = read_env(path)
    assert values["MTC_VALUE"] == value
    assert values["MTC_AFTER"] == "next"
    from dotenv import dotenv_values

    # python-dotenv's own reader with its default interpolation agrees.
    assert dotenv_values(path)["MTC_VALUE"] == value


@pytest.mark.parametrize("character", ["$", '"', "\\", "`", "\n", "\r", "\t", chr(0)])
def test_unsafe_values_rejected(tmp_path, character):
    path = tmp_path / "unsafe.env"
    path.write_text("MTC_KEEP=1\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(UnsafeEnvValue):
        render_value(f"abc{character}def")
    with pytest.raises(UnsafeEnvValue):
        upsert_env(path, {"MTC_SAFE": "fine", "MTC_BAD": f"abc{character}def"})
    assert path.read_bytes() == before


def test_spaces_are_quoted():
    assert render_value("two words") == '"two words"'
    assert render_value("bare-value_1.2:3/4,5@6+7=8") == "bare-value_1.2:3/4,5@6+7=8"
    assert render_value("") == ""


def test_failed_write_leaves_old_file(tmp_path, monkeypatch):
    path = tmp_path / "stable.env"
    path.write_text("MTC_A=old\n", encoding="utf-8")
    os.chmod(path, 0o600)
    before = path.read_bytes()

    def broken_replace(src, dst):
        raise OSError("simulated failure")

    monkeypatch.setattr(envfile.os, "replace", broken_replace)
    with pytest.raises(OSError):
        upsert_env(path, {"MTC_A": "new"})
    assert path.read_bytes() == before
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "stable.env"]
    assert leftovers == []


def test_values_never_logged(tmp_path, caplog):
    path = tmp_path / "quiet.env"
    secret = "s3cretvalue" + "9" * 20
    with caplog.at_level(logging.DEBUG):
        upsert_env(path, {"MTC_API_KEY": secret, "MTC_OTHER": "visible-value"})
        read_env(path)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in text
    assert "visible-value" not in text
    assert any("key(s) written" in record.getMessage() for record in caplog.records)


def test_invalid_key_rejected_and_multiline_value_refused(tmp_path):
    path = tmp_path / "keys.env"
    with pytest.raises(ValueError):
        upsert_env(path, {"BAD KEY": "x"})
    path.write_text('MTC_LONG="first line\nsecond line"\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        upsert_env(path, {"MTC_LONG": "short"})
    assert path.read_bytes() == before


def test_read_env_missing_file_is_empty(tmp_path):
    assert read_env(tmp_path / "nothing.env") == {}
    path = tmp_path / "literal.env"
    path.write_text("MTC_A=x\nMTC_B=${MTC_A}\n", encoding="utf-8")
    # No interpolation: the reader returns what the file says.
    assert read_env(path)["MTC_B"] == "${MTC_A}"
