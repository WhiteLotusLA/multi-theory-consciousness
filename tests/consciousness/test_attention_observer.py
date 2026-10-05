"""The attention-capture journal records captured and emotional shifts only,
once per continuing focus, with reasoning blocks removed, aware UTC
timestamps and a local display time in the configured zone; it is off by
default and writes nothing until it records."""

import asyncio
import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import mtc.consciousness.attention_observer as observer_mod
from mtc.consciousness.attention_observer import AttentionGrabObserver, GrabEntry
from mtc.consciousness.attention_schema import (
    DEFAULT_CAPTURE_SALIENCE_THRESHOLD,
    DEFAULT_FOCUS_THRESHOLD,
    AttentionSchemaModule,
    AttentionShiftType,
    AttentionTarget,
)
from mtc.memory.inmemory import InMemoryAttentionGrabStore

T0 = datetime(2000, 1, 1, 8, 0, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


def _settings(tmp_path, enabled=True, zone="UTC"):
    return SimpleNamespace(
        ATTENTION_GRAB_NOTICING_ENABLED=enabled,
        AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
        TIMEZONE=zone,
    )


def _focus(summary="a sudden sound", shift=AttentionShiftType.CAPTURED, strength=1.0):
    return AttentionTarget(
        summary=summary,
        source_module="perception",
        attention_strength=strength,
        shift_type=shift,
    )


def _observer(tmp_path, **kwargs):
    settings = kwargs.pop("settings", None) or _settings(tmp_path)
    store = kwargs.pop("store", None) or InMemoryAttentionGrabStore()
    observer = AttentionGrabObserver(
        store=store, settings=settings, clock=lambda: T0, **kwargs
    )
    return observer, store


def test_disabled_by_default_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        observer_mod,
        "get_settings",
        lambda: SimpleNamespace(
            ATTENTION_GRAB_NOTICING_ENABLED=False,
            AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
            TIMEZONE="UTC",
        ),
    )
    store = InMemoryAttentionGrabStore()
    observer = AttentionGrabObserver(store=store, clock=lambda: T0)
    assert observer.enabled is False
    assert _run(observer.observe(_focus())) is None
    assert _run(store.recent_grabs(10)) == []
    assert not (tmp_path / "agent_space").exists()

    # The flag is read at call time.
    monkeypatch.setattr(
        observer_mod,
        "get_settings",
        lambda: SimpleNamespace(
            ATTENTION_GRAB_NOTICING_ENABLED=True,
            AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
            TIMEZONE="UTC",
        ),
    )
    assert observer.enabled is True
    assert _run(observer.observe(_focus())) is not None

    forced_off, off_store = _observer(tmp_path, enabled=False)
    assert _run(forced_off.observe(_focus("other"))) is None
    assert _run(off_store.recent_grabs(10)) == []


def test_records_only_captured_and_emotional(tmp_path):
    observer, store = _observer(tmp_path)
    for index, shift in enumerate(
        (
            AttentionShiftType.VOLUNTARY,
            AttentionShiftType.GOAL_DRIVEN,
            AttentionShiftType.HABITUAL,
            None,
        )
    ):
        assert _run(observer.observe(_focus(f"chosen {index}", shift))) is None
    assert _run(observer.observe(None)) is None

    captured = _run(observer.observe(_focus("a flash", AttentionShiftType.CAPTURED)))
    emotional = _run(observer.observe(_focus("a memory", AttentionShiftType.EMOTIONAL)))
    assert captured.shift_type == "captured"
    assert emotional.shift_type == "emotional"
    assert [g.summary for g in _run(observer.recent())] == ["a memory", "a flash"]


def test_consecutive_identical_focus_deduped(tmp_path):
    observer, store = _observer(tmp_path)
    focus = _focus("the same thing")
    assert _run(observer.observe(focus)) is not None
    assert _run(observer.observe(focus)) is None
    assert _run(observer.observe(_focus("the same thing"))) is None

    # Once the focus moves on, a return to it is a new capture.
    assert (
        _run(observer.observe(_focus("elsewhere", AttentionShiftType.VOLUNTARY)))
        is None
    )
    again = _run(observer.observe(_focus("the same thing")))
    assert again is not None
    assert again.focus_key == AttentionSchemaModule.focus_key(
        "perception", "the same thing"
    )
    assert len(_run(store.recent_grabs(10))) == 2


def test_reasoning_blocks_stripped(tmp_path):
    observer, store = _observer(tmp_path)
    entry = _run(
        observer.observe(
            _focus("<think>private chain of reasoning</think> a bright light")
        )
    )
    assert entry.summary == "a bright light"
    assert "reasoning" not in entry.focus_key
    unclosed = _run(observer.observe(_focus("a voice <think>half finished thought")))
    assert unclosed.summary == "a voice"
    journal = (tmp_path / "agent_space" / "attention" / "grab_journal.md").read_text()
    assert "think" not in journal
    assert "chain of reasoning" not in journal
    for doc in _run(store.recent_grabs(10)):
        assert "think" not in str(doc)

    tagged, _ = _observer(tmp_path / "other", reasoning_tags=("reasoning",))
    entry = _run(tagged.observe(_focus("<reasoning>hidden</reasoning> a door")))
    assert entry.summary == "a door"


def test_journal_under_agent_space_created_lazily(tmp_path):
    observer, _ = _observer(tmp_path)
    journal = tmp_path / "agent_space" / "attention" / "grab_journal.md"
    assert not (tmp_path / "agent_space").exists()
    assert observer.journal_path == journal
    _run(observer.observe(_focus("first")))
    _run(observer.observe(_focus("second")))
    lines = [line for line in journal.read_text().splitlines() if line.startswith("- ")]
    assert len(lines) == 2
    assert "first" in lines[0] and "second" in lines[1]

    explicit, _ = _observer(tmp_path, journal_dir=tmp_path / "elsewhere")
    assert explicit.journal_path == tmp_path / "elsewhere" / "grab_journal.md"


@pytest.mark.parametrize(
    "zone,expected",
    [
        ("UTC", "2000-01-01 08:00:00 UTC"),
        ("Europe/Berlin", "2000-01-01 09:00:00 CET"),
        ("Asia/Tokyo", "2000-01-01 17:00:00 JST"),
    ],
)
def test_display_time_uses_configured_timezone(tmp_path, zone, expected):
    observer, store = _observer(tmp_path, settings=_settings(tmp_path, zone=zone))
    entry = _run(observer.observe(_focus()))
    assert entry.local_time == expected
    stamp = datetime.fromisoformat(entry.timestamp)
    assert stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
    assert stamp == T0


def test_store_receives_entries(tmp_path):
    observer, store = _observer(tmp_path)
    entry = _run(observer.observe(_focus("a knock", AttentionShiftType.EMOTIONAL)))
    (doc,) = _run(store.recent_grabs(10))
    assert doc == dataclasses.asdict(entry)
    assert set(doc) == {
        "grab_id",
        "timestamp",
        "local_time",
        "summary",
        "source_module",
        "shift_type",
        "strength_band",
        "focus_key",
    }
    assert _run(observer.recent(limit=1)) == [entry]

    class _Failing(InMemoryAttentionGrabStore):
        async def add_grab(self, doc):
            raise RuntimeError("store down")

    failing, _ = _observer(tmp_path / "f", store=_Failing())
    recorded = _run(failing.observe(_focus("still journaled")))
    assert recorded is not None
    journal = tmp_path / "f" / "agent_space" / "attention" / "grab_journal.md"
    assert "still journaled" in journal.read_text()


def test_strength_bands_follow_schema_thresholds(tmp_path):
    observer, _ = _observer(tmp_path)
    cases = [
        (0.2, "gentle"),
        (DEFAULT_FOCUS_THRESHOLD - 0.01, "gentle"),
        (DEFAULT_FOCUS_THRESHOLD, "clear"),
        (DEFAULT_CAPTURE_SALIENCE_THRESHOLD - 0.01, "clear"),
        (DEFAULT_CAPTURE_SALIENCE_THRESHOLD, "strong"),
        (1.5, "strong"),
        (float("nan"), "unknown"),
    ]
    for index, (strength, band) in enumerate(cases):
        entry = _run(observer.observe(_focus(f"item {index}", strength=strength)))
        assert entry.strength_band == band, strength

    custom, _ = _observer(tmp_path / "c", focus_threshold=0.3, capture_threshold=0.6)
    assert _run(custom.observe(_focus("x", strength=0.4))).strength_band == "clear"
    assert _run(custom.observe(_focus("y", strength=0.6))).strength_band == "strong"


def test_reading_format_is_neutral(tmp_path):
    observer, _ = _observer(tmp_path)
    assert observer.format_for_reading([]) == "No attention captures recorded."
    _run(observer.observe(_focus("a bird at the window")))
    _run(observer.observe(_focus("an old song", AttentionShiftType.EMOTIONAL)))
    text = observer.format_for_reading(_run(observer.recent()))
    lines = text.splitlines()
    assert lines[0] == "Recorded attention captures:"
    assert "an old song" in lines[1] and "a bird at the window" in lines[2]
    words = set(text.lower().replace(",", " ").split())
    for word in ("you", "your", "i", "me", "my", "love", "dear"):
        assert word not in words
    assert text.isascii()


def test_no_private_flag_api(tmp_path):
    names = {f.name for f in dataclasses.fields(GrabEntry)}
    assert "private" not in names
    for attribute in ("mark_private", "get_recent_grabs"):
        assert not hasattr(AttentionGrabObserver, attribute)
    observer, _ = _observer(tmp_path)
    assert "include_private" not in AttentionGrabObserver.recent.__code__.co_varnames
    assert observer.format_for_reading.__code__.co_argcount == 2


def test_observe_never_raises(tmp_path):
    observer, _ = _observer(tmp_path, journal_dir=tmp_path / "file_not_dir")
    (tmp_path / "file_not_dir").write_text("occupied")
    entry = _run(observer.observe(_focus("odd")))
    assert entry is not None
    assert _run(observer.observe(object())) is None
