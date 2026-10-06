"""Speaker trust: a token is the only thing that authenticates a speaker."""

import ast
import logging
import os
import re
import secrets
import sys
from pathlib import Path

import pytest

from mtc.core.config import Settings
from mtc.pipeline import speaker_trust as module
from mtc.pipeline.speaker_trust import (
    ROLES,
    SpeakerEntry,
    SpeakerIdentity,
    SpeakerTrust,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mtc" / "pipeline" / "speaker_trust.py"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def _table(settings=None, **extra_entries):
    entries = [
        SpeakerEntry(name="Alice", role="creator", trust=0.9, token_env="TOKEN_A"),
        SpeakerEntry(name="Operator", role="companion", trust=0.7, token_env=None),
        SpeakerEntry(name="Guest", role="guest", trust=0.3, token_env="TOKEN_G"),
    ]
    return SpeakerTrust(entries, settings=settings)


def test_roles_are_the_three_configured_roles():
    assert ROLES == ("creator", "companion", "guest")
    with pytest.raises(ValueError, match="role"):
        SpeakerEntry(name="Alice", role="admin", trust=0.5, token_env=None)
    with pytest.raises(ValueError, match="trust"):
        SpeakerEntry(name="Alice", role="guest", trust=1.5, token_env=None)
    with pytest.raises(ValueError, match="name"):
        SpeakerEntry(name="  ", role="guest", trust=0.5, token_env=None)


def test_name_without_token_gets_nothing(clean_env, monkeypatch):
    cfg = _settings(TRUSTED_SPEAKERS="Alice,Operator")
    monkeypatch.setenv("TOKEN_A", secrets.token_hex(16))
    trust = _table(cfg)
    for name in (cfg.CREATOR_NAME, "Operator", "admin", "Alice"):
        identity = trust.verify(name, None)
        assert identity.authenticated is False
        assert identity.trusted is False
        assert identity.role is None
        assert identity.trust == pytest.approx(cfg.DEFAULT_SPEAKER_TRUST)
        assert trust.trust_bonus(identity) == 0.0
        assert trust.emotion_trust(identity) == pytest.approx(cfg.DEFAULT_EMOTION_TRUST)
        assert trust.label_for(identity) == f"[{name}]"
        assert trust.is_trusted(name, None) is False
        assert trust.is_trusted(name, "") is False
        assert trust.safety_role(identity) == "guest"


def test_wrong_token_rejected(clean_env, monkeypatch):
    monkeypatch.setenv("TOKEN_A", secrets.token_hex(16))
    trust = _table(_settings(TRUSTED_SPEAKERS="Alice"))
    identity = trust.verify("Alice", secrets.token_hex(16))
    assert identity.authenticated is False
    assert identity.trusted is False
    assert identity.trust == pytest.approx(0.5)


def test_token_of_another_speaker_rejected(clean_env, monkeypatch):
    token_a, token_g = secrets.token_hex(16), secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token_a)
    monkeypatch.setenv("TOKEN_G", token_g)
    trust = _table(_settings(TRUSTED_SPEAKERS="Alice"))
    assert trust.verify("Alice", token_g).authenticated is False
    assert trust.verify("Guest", token_a).authenticated is False
    assert trust.verify("Guest", token_g).authenticated is True


def test_correct_token_authenticates_with_configured_trust(clean_env, monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    cfg = _settings(TRUSTED_SPEAKERS="Alice")
    trust = _table(cfg)
    identity = trust.verify("Alice", token)
    assert identity.authenticated is True
    assert identity.trusted is True
    assert identity.role == "creator"
    assert identity.trust == pytest.approx(0.9)
    assert identity.is_default is False
    assert trust.trust_bonus(identity) == pytest.approx(cfg.TRUST_ACTIVATION_BONUS)
    assert trust.emotion_trust(identity) == pytest.approx(cfg.TRUSTED_EMOTION_TRUST)
    assert trust.is_trusted("Alice", token) is True
    assert trust.safety_role(identity) == "creator"
    view = identity.view()
    assert (view.name, view.role, view.authenticated) == ("Alice", "creator", True)


def test_trusted_requires_allowlist(clean_env, monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_G", token)
    cfg = _settings(TRUSTED_SPEAKERS="Alice")
    trust = _table(cfg)
    identity = trust.verify("Guest", token)
    assert identity.authenticated is True
    assert identity.trusted is False
    assert identity.trust == pytest.approx(0.3)
    assert trust.trust_bonus(identity) == 0.0
    assert trust.emotion_trust(identity) == pytest.approx(cfg.DEFAULT_EMOTION_TRUST)
    assert trust.is_trusted("Guest", token) is False


def test_compare_digest_is_the_comparator(clean_env, monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    calls = []
    real = module.secrets.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(module.secrets, "compare_digest", spy)
    trust = _table(_settings())
    assert trust.verify("Alice", token).authenticated is True
    assert len(calls) == 1
    # An empty presented token is never compared.
    trust.verify("Alice", "")
    trust.verify("Alice", None)
    assert len(calls) == 1

    # The source holds no equality test between token values.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(
            isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops
        ):
            continue
        names = [
            getattr(side, "id", getattr(side, "attr", ""))
            for side in [node.left, *node.comparators]
        ]
        if any("token" in str(name).lower() for name in names):
            offenders.append(node.lineno)
    assert offenders == []


def test_token_env_read_at_call_time(clean_env, monkeypatch):
    trust = _table(_settings())
    first = secrets.token_hex(16)
    assert trust.verify("Alice", first).authenticated is False
    monkeypatch.setenv("TOKEN_A", first)
    assert trust.verify("Alice", first).authenticated is True
    second = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", second)
    assert trust.verify("Alice", first).authenticated is False
    assert trust.verify("Alice", second).authenticated is True


def test_speaker_tokens_split_on_first_colon(clean_env):
    token = secrets.token_hex(8) + ":with:colons"
    cfg = _settings(SPEAKER_TOKENS=f"Operator:{token}, Guest:{secrets.token_hex(8)}")
    trust = _table(cfg)
    identity = trust.verify("Operator", token)
    assert identity.authenticated is True
    assert identity.role == "companion"
    assert identity.trust == pytest.approx(0.7)
    assert trust.verify("Operator", token[:-1]).authenticated is False


def test_entry_without_token_never_authenticates(clean_env):
    trust = _table(_settings())
    for presented in (secrets.token_hex(16), "", None):
        assert trust.verify("Operator", presented).authenticated is False
    # A name with no table entry authenticates with no token either, whatever
    # is presented.
    bare = SpeakerTrust([], settings=_settings())
    assert bare.verify("Guest", secrets.token_hex(16)).authenticated is False


def test_label_rules(clean_env, monkeypatch):
    token_a, token_g = secrets.token_hex(16), secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token_a)
    monkeypatch.setenv("TOKEN_G", token_g)
    trust = _table(_settings())
    default = trust.verify(None, None)
    assert default.is_default is True
    assert default.name == "User"
    assert trust.label_for(default) == ""
    creator = trust.verify("Alice", token_a)
    assert trust.label_for(creator) == ""
    assert trust.label_for(trust.verify("Alice", None)) == "[Alice]"
    assert trust.label_for(trust.verify("Guest", token_g)) == "[Guest]"
    # A name with no table entry is labelled too
    bare = SpeakerTrust([], settings=_settings())
    assert bare.label_for(bare.verify("Guest", None)) == "[Guest]"
    assert trust.verify("", None).is_default is True
    assert trust.verify("   ", None).name == "User"


def test_session_key_separates_authenticated_and_not(clean_env, monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    trust = _table(_settings())
    verified = trust.session_key(trust.verify("Alice", token))
    claimed = trust.session_key(trust.verify("Alice", None))
    assert verified != claimed
    assert token not in verified
    assert trust.session_key(trust.verify("Alice", token)) == verified
    assert trust.session_key(trust.verify(None, None)) != claimed


def test_trust_lookup_reads_current_identity_only(clean_env, monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    cfg = _settings()
    trust = _table(cfg)
    default = cfg.DEFAULT_SPEAKER_TRUST
    assert trust.trust_lookup("Alice") == pytest.approx(default)
    identity = trust.verify("Alice", token)
    with trust.use_identity(identity):
        assert trust.trust_lookup("Alice") == pytest.approx(0.9)
        assert trust.trust_lookup("Guest") == pytest.approx(default)
    assert trust.trust_lookup("Alice") == pytest.approx(default)
    with trust.use_identity(trust.verify("Alice", None)):
        assert trust.trust_lookup("Alice") == pytest.approx(default)


def test_status_never_contains_names_or_tokens(clean_env, monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    trust = _table(
        _settings(TRUSTED_SPEAKERS="Alice", SPEAKER_TOKENS=f"Operator:{token}")
    )
    trust.verify("Alice", token)
    status = trust.status()
    text = repr(status)
    for secret_or_name in ("Alice", "Operator", "Guest", token, "TOKEN_A"):
        assert secret_or_name not in text
    assert status["entries"] == 3
    assert status["by_role"] == {"creator": 1, "companion": 1, "guest": 1}
    assert status["allowlisted"] == 1


def test_missing_file_is_empty_table(clean_env):
    cfg = _settings(SPEAKERS_PATH=str(clean_env / "absent.yaml"))
    trust = SpeakerTrust.from_settings(cfg)
    assert trust.status()["entries"] == 0
    assert trust.verify("Alice", secrets.token_hex(16)).authenticated is False


def test_from_settings_loads_entries(clean_env, monkeypatch):
    path = clean_env / "speakers.yaml"
    path.write_text(
        "speakers:\n"
        "  - name: Alice\n"
        "    role: creator\n"
        "    trust: 0.9\n"
        "    token_env: TOKEN_A\n"
        "  - name: Guest\n"
        "    role: guest\n"
        "    trust: 0.2\n",
        encoding="utf-8",
    )
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    trust = SpeakerTrust.from_settings(_settings(SPEAKERS_PATH=str(path)))
    assert trust.status()["entries"] == 2
    assert trust.verify("Alice", token).authenticated is True
    bad = clean_env / "bad.yaml"
    bad.write_text(
        "speakers:\n  - name: Alice\n    role: admin\n    trust: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="role"):
        SpeakerTrust.from_settings(_settings(SPEAKERS_PATH=str(bad)))
    # A missing trust is an error, never a silent 0.0 for an authenticated entry
    no_trust = clean_env / "no_trust.yaml"
    no_trust.write_text(
        "speakers:\n  - name: Alice\n    role: creator\n    token_env: TOKEN_A\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trust"):
        SpeakerTrust.from_settings(_settings(SPEAKERS_PATH=str(no_trust)))


@pytest.mark.parametrize("shape", ["unclosed_quote", "unclosed_flow"])
def test_yaml_error_names_file_and_line_only(clean_env, caplog, shape):
    token = secrets.token_hex(16)
    if shape == "unclosed_quote":
        # The parser's own message quotes the token on the failing line
        text = 'speakers:\n  - name: Alice\n    role: creator\n    note: "%s\n' % token
    else:
        # The parser's own message quotes the line that holds the name
        text = "speakers:\n  - {name: Alice, role: creator, trust: 0.9, note: %s\n" % (
            token
        )
    path = clean_env / "speakers.yaml"
    path.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="mtc.pipeline.speaker_trust"):
        with pytest.raises(ValueError) as caught:
            SpeakerTrust.from_settings(_settings(SPEAKERS_PATH=str(path)))
    message = str(caught.value)
    assert "speakers.yaml" in message
    assert re.search(r"line \d+", message)
    logged = [record.getMessage() for record in caplog.records]
    assert any("speakers.yaml" in line for line in logged)
    for secret_or_name in (token, "Alice"):
        assert secret_or_name not in message
        assert all(secret_or_name not in line for line in logged)
    # The parser's error, which quotes the line, is neither chained nor shown
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_entry_errors_echo_no_value(clean_env):
    token = secrets.token_hex(16)
    # A key that is a name, and a role field holding a token
    path = clean_env / "speakers.yaml"
    path.write_text(f"speakers:\n  - Alice: {token}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field") as caught:
        SpeakerTrust.from_settings(_settings(SPEAKERS_PATH=str(path)))
    assert "Alice" not in str(caught.value) and token not in str(caught.value)
    with pytest.raises(ValueError, match="role") as caught:
        SpeakerEntry(name="Guest", role=token, trust=0.5, token_env=None)
    assert token not in str(caught.value)


def test_every_speaker_role_is_a_privacy_role():
    from mtc.privacy.access_control_enforcer import UserRole

    for role in ROLES:
        assert UserRole.from_string(role).value == role


def test_yaml_imported_lazily():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "yaml" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "yaml"
    assert "yaml" not in getattr(module, "__dict__", {}) or not isinstance(
        module.__dict__.get("yaml"), type(sys)
    )


def test_identity_is_frozen_and_view_has_no_token_fields():
    identity = SpeakerIdentity(
        name="Alice",
        role=None,
        trust=0.5,
        authenticated=False,
        trusted=False,
        is_default=False,
    )
    with pytest.raises(Exception):
        identity.trust = 1.0  # type: ignore[misc]
    view = identity.view()
    assert not hasattr(view, "token")
    assert view.label == "unverified"
