"""The three helpers local tools use: where the speakers file lives, the
token a local client presents for a configured speaker, and the two
environment entries the setup wizard writes for the creator."""

import logging
import os
import secrets
from pathlib import Path

import pytest

from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.pipeline import speaker_trust as module
from mtc.pipeline.speaker_trust import (
    SpeakerTrust,
    env_entries,
    speakers_path,
    token_for,
)

PREFIX = Settings.model_config["env_prefix"]
TOKENS_KEY = f"{PREFIX}SPEAKER_TOKENS"
TRUSTED_KEY = f"{PREFIX}TRUSTED_SPEAKERS"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    # The process settings are a singleton an earlier test may have built
    # somewhere else: each test here starts without one, drops the one it
    # built, and monkeypatch then puts the earlier one back.
    monkeypatch.setattr(config_module, "_settings", None)
    yield tmp_path
    config_module._settings = None


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def captured_log():
    handler = _Capture()
    logger = logging.getLogger(module.__name__)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


def _write_speakers(root: Path, text: str) -> Path:
    path = root / "config" / "speakers.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


SPEAKERS = """\
speakers:
  - name: Alice
    role: creator
    trust: 0.9
    token_env: HELPER_TOKEN_ALICE
  - name: Operator
    role: companion
    trust: 0.7
  - name: Guest
    role: guest
    trust: 0.3
"""


def test_token_for_reads_token_env_then_speaker_tokens(clean_env, monkeypatch):
    _write_speakers(clean_env, SPEAKERS)
    from_env = secrets.token_hex(16)
    from_pair = secrets.token_hex(16)
    other_pair = secrets.token_hex(16)
    cfg = Settings(SPEAKER_TOKENS=f"Alice:{other_pair},Operator:{from_pair}")

    # The entry names a variable: only that variable counts, and only once it
    # is exported; the pair under the same name is not a second chance.
    assert token_for("Alice", cfg) is None
    monkeypatch.setenv("HELPER_TOKEN_ALICE", from_env)
    assert token_for("Alice", cfg) == from_env
    # No variable named: the pair is the token.
    assert token_for("Operator", cfg) == from_pair
    assert token_for("  Operator ", cfg) == from_pair
    # An entry with neither has no token to present.
    assert token_for("Guest", cfg) is None

    # What the helper returns is what the trust source accepts.
    trust = SpeakerTrust.from_settings(cfg)
    assert trust.verify("Alice", token_for("Alice", cfg)).authenticated is True
    assert trust.verify("Operator", token_for("Operator", cfg)).authenticated is True


def test_token_for_unknown_name_is_none(clean_env):
    stray = secrets.token_hex(16)
    cfg = Settings(SPEAKER_TOKENS=f"Stranger:{stray}")
    # No speakers file at all.
    assert token_for("Stranger", cfg) is None
    assert token_for("", cfg) is None
    _write_speakers(clean_env, SPEAKERS)
    # A pair without an entry belongs to nobody: the trust source would not
    # authenticate it, so the helper has nothing to hand a client.
    assert token_for("Stranger", cfg) is None
    assert token_for("alice", cfg) is None


def test_env_entries_for_creator():
    token = secrets.token_hex(32)
    entries = env_entries("Alice", token, trusted=True)
    assert entries == {TOKENS_KEY: f"Alice:{token}", TRUSTED_KEY: "Alice"}

    untrusted = env_entries("Alice", token, trusted=False)
    assert untrusted == {TOKENS_KEY: f"Alice:{token}", TRUSTED_KEY: ""}

    # The written pair authenticates through the trust source and a bare
    # name still gets nothing.
    cfg = Settings(
        SPEAKER_TOKENS=entries[TOKENS_KEY], TRUSTED_SPEAKERS=entries[TRUSTED_KEY]
    )
    trust = SpeakerTrust(
        [module.SpeakerEntry(name="Alice", role="creator", trust=0.9)], settings=cfg
    )
    assert trust.verify("Alice", token).trusted is True
    assert trust.verify("Alice", None).authenticated is False


def test_env_entries_keep_other_pairs_and_replace_a_renamed_creator():
    old = secrets.token_hex(32)
    new = secrets.token_hex(32)
    guest = secrets.token_hex(32)
    existing_tokens = f"Operator:{old}, Guest:{guest}"
    existing_trusted = "Operator, Guest"

    renamed = env_entries(
        "Alice",
        new,
        trusted=True,
        existing_tokens=existing_tokens,
        existing_trusted=existing_trusted,
        previous_name="Operator",
    )
    assert renamed[TOKENS_KEY] == f"Alice:{new},Guest:{guest}"
    assert renamed[TRUSTED_KEY] == "Alice,Guest"
    assert old not in renamed[TOKENS_KEY]

    # Same name, a new token: the pair is replaced, never duplicated.
    again = env_entries(
        "Alice",
        old,
        trusted=False,
        existing_tokens=renamed[TOKENS_KEY],
        existing_trusted=renamed[TRUSTED_KEY],
    )
    assert again[TOKENS_KEY] == f"Alice:{old},Guest:{guest}"
    assert again[TRUSTED_KEY] == "Guest"

    # keep_existing keeps the token already written for the same name and
    # never the token of a renamed speaker.
    kept = env_entries(
        "Alice",
        secrets.token_hex(32),
        trusted=True,
        existing_tokens=renamed[TOKENS_KEY],
        existing_trusted=renamed[TRUSTED_KEY],
        keep_existing=True,
    )
    assert kept[TOKENS_KEY] == renamed[TOKENS_KEY]
    fresh = secrets.token_hex(32)
    moved = env_entries(
        "Alice",
        fresh,
        trusted=True,
        existing_tokens=f"Operator:{old}",
        previous_name="Operator",
        keep_existing=True,
    )
    assert moved[TOKENS_KEY] == f"Alice:{fresh}"

    # A parsed environment file supplies the existing values by key.
    from_file = env_entries(
        "Alice",
        new,
        trusted=True,
        existing_env={TOKENS_KEY: existing_tokens, TRUSTED_KEY: existing_trusted},
        previous_name="Operator",
    )
    assert from_file == renamed


def test_env_entries_reject_separators_in_names():
    token = secrets.token_hex(32)
    for bad in ("Ali,ce", "Ali:ce", "Ali\nce", "Ali\rce", "", "   "):
        with pytest.raises(ValueError):
            env_entries(bad, token, trusted=True)
    with pytest.raises(ValueError):
        env_entries("Alice", token, trusted=True, previous_name="Ope,rator")
    for bad_token in ("", "a,b", "a b", "a\nb"):
        with pytest.raises(ValueError) as caught:
            env_entries("Alice", bad_token, trusted=True)
        if bad_token.strip():
            assert bad_token not in str(caught.value)


def test_speakers_path_from_settings(clean_env):
    assert speakers_path(Settings()) == clean_env / "config" / "speakers.yaml"
    moved = Settings(SPEAKERS_PATH="private/people.yaml")
    assert speakers_path(moved) == clean_env / "private" / "people.yaml"
    absolute = clean_env / "elsewhere" / "speakers.yaml"
    assert speakers_path(Settings(SPEAKERS_PATH=str(absolute))) == absolute
    # Read at call time from the process settings when none is passed.
    assert speakers_path() == speakers_path(Settings())


def test_helpers_never_log_tokens(clean_env, monkeypatch, captured_log):
    _write_speakers(clean_env, SPEAKERS)
    env_token = secrets.token_hex(16)
    pair_token = secrets.token_hex(16)
    monkeypatch.setenv("HELPER_TOKEN_ALICE", env_token)
    cfg = Settings(SPEAKER_TOKENS=f"Operator:{pair_token}")

    assert token_for("Alice", cfg) == env_token
    assert token_for("Operator", cfg) == pair_token
    assert token_for("Nobody", cfg) is None
    written = env_entries(
        "Operator",
        pair_token,
        trusted=True,
        existing_tokens=f"Guest:{env_token}",
        existing_trusted="Guest",
    )
    assert pair_token in written[TOKENS_KEY]
    with pytest.raises(ValueError) as caught:
        env_entries("Ope:rator", pair_token, trusted=True)
    speakers_path(cfg)

    assert captured_log.lines, "the trust source logs the table load"
    logged = "\n".join(captured_log.lines) + str(caught.value)
    for secret_value in (env_token, pair_token):
        assert secret_value not in logged
    for name in ("Alice", "Operator", "Guest", "Nobody"):
        assert name not in logged
