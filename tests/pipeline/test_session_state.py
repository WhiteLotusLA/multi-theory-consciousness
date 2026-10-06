"""Sessions keyed by the trust source, counters, trimming and pruning."""

import secrets
from datetime import datetime, timedelta, timezone

import pytest

from mtc.core.config import Settings
from mtc.pipeline.session_state import Session, SessionManager
from mtc.pipeline.speaker_trust import SpeakerEntry, SpeakerTrust

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
TOKEN_ENV = "MTC_TEST_SESSION_TOKEN_ALICE"


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def trust(monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv(TOKEN_ENV, token)
    entries = [SpeakerEntry("Alice", "companion", 0.8, token_env=TOKEN_ENV)]
    return SpeakerTrust(entries, settings=Settings(_env_file=None)), token


def test_unauthenticated_name_never_shares_authenticated_session(trust):
    source, token = trust
    verified = source.verify("Alice", token)
    claimed = source.verify("Alice", None)
    wrong = source.verify("Alice", secrets.token_hex(16))
    assert verified.authenticated and not claimed.authenticated
    manager = SessionManager(clock=Clock())
    session = manager.start(source.session_key(verified), "conv-verified")
    session.history.append({"role": "user", "content": "private line"})
    # a bare name (or a wrong token) maps to a different key with no session
    assert source.session_key(claimed) != source.session_key(verified)
    assert source.session_key(wrong) == source.session_key(claimed)
    assert manager.get(source.session_key(claimed)) is None
    other = manager.start(source.session_key(claimed), "conv-claimed")
    assert other is not session and other.history == []
    assert manager.get(source.session_key(verified)).history == [
        {"role": "user", "content": "private line"}
    ]
    # keys are strings from the trust source, never an identity or a name object
    with pytest.raises(TypeError):
        manager.get(verified)


def test_turn_counter_advances():
    clock = Clock()
    manager = SessionManager(clock=clock)
    manager.start("verified:Alice", "conv-1")
    assert manager.advance("verified:Alice") == 1
    clock.now = T0 + timedelta(minutes=5)
    assert manager.advance("verified:Alice") == 2
    session = manager.get("verified:Alice")
    assert session.turn_number == 2
    assert session.last_turn_at == T0 + timedelta(minutes=5)
    assert session.started_at == T0
    with pytest.raises(KeyError):
        manager.advance("claimed:Alice")


def test_trim_in_place_to_max_history():
    manager = SessionManager(max_history=3, clock=Clock())
    history = [{"role": "user", "content": str(i)} for i in range(5)]
    session = manager.start("k", "c", history=history)
    # start already trims to the newest entries and copies the input
    assert [e["content"] for e in session.history] == ["2", "3", "4"]
    assert len(history) == 5
    held = session.history
    held.extend({"role": "assistant", "content": str(i)} for i in range(5, 8))
    assert manager.trim("k") == 3
    assert session.history is held
    assert [e["content"] for e in held] == ["5", "6", "7"]
    assert manager.trim("k") == 0
    assert manager.trim("missing") == 0
    with pytest.raises(ValueError):
        SessionManager(max_history=0)


def test_prune_after_idle_hours():
    clock = Clock()
    manager = SessionManager(idle_prune_hours=48.0, clock=clock)
    manager.start("old", "c1")
    manager.start("busy", "c2")
    clock.now = T0 + timedelta(hours=40)
    manager.advance("busy")
    clock.now = T0 + timedelta(hours=49)
    removed = manager.prune()
    assert [s.key for s in removed] == ["old"]
    assert manager.get("old") is None and manager.get("busy") is not None
    later = manager.prune(now=T0 + timedelta(hours=40 + 48, minutes=1))
    assert [s.key for s in later] == ["busy"]
    assert manager.sessions() == []
    assert manager.status() == {
        "sessions": 0,
        "pruned": 2,
        "max_history": 100,
        "idle_prune_hours": 48.0,
    }


def test_default_speaker_key(trust):
    source, _ = trust
    anonymous = source.verify("", None)
    assert anonymous.is_default
    key = source.session_key(anonymous)
    assert key == source.session_key(source.verify(None, None))
    manager = SessionManager(clock=Clock())
    session = manager.start(key, "conv-anon")
    assert isinstance(session, Session)
    assert manager.get(key) is session
    # the default speaker never shares a session with a named speaker
    assert key != source.session_key(source.verify("Alice", None))
    assert key != source.session_key(source.verify("User", None))


def test_session_keeps_the_stored_speaker_label(trust):
    """The label every stored row of the session names the speaker by is kept
    on the session; a session opened without one has none."""
    source, token = trust
    manager = SessionManager(clock=Clock())
    claimed = source.verify("Alice", None)
    session = manager.start(source.session_key(claimed), "c1", speaker_label="[Alice]")
    assert session.speaker_label == "[Alice]"
    verified = source.verify("Alice", token)
    other = manager.start(source.session_key(verified), "c2", speaker_label="Alice")
    assert other.speaker_label == "Alice"
    assert manager.start("k", "c3").speaker_label is None
    with pytest.raises(TypeError):
        manager.start("k2", "c4", speaker_label="")
