"""Night seam and last-turn inference, in three time zones."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from mtc.core.config import Settings
from mtc.pipeline.context_budget import (
    NIGHT_NOTE_TEXT,
    apply_night_seam,
    build_night_note,
    find_night_boundary,
    infer_last_turn_at,
)

ZONES = ["UTC", "Europe/Berlin", "Asia/Tokyo"]


def _settings(zone, **overrides):
    values = {
        "TIMEZONE": zone,
        "WAKE_HOUR": 7,
        "SLEEP_HOUR": 23,
        "NIGHT_MIN_GAP_HOURS": 6.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _local(zone, hour, minute=0, day_offset=0):
    """An aware UTC instant for a local wall time on 2000-01-01 (plus days)."""
    base = datetime(2000, 1, 1, hour, minute, tzinfo=ZoneInfo(zone))
    return (base + timedelta(days=day_offset)).astimezone(timezone.utc)


def _entry(moment, content="line", role="user"):
    return {"role": role, "speaker": "Alice", "content": content, "timestamp": moment}


@pytest.mark.parametrize("zone", ZONES)
def test_boundary_found_across_night(zone):
    cfg = _settings(zone)
    entries = [
        _entry(_local(zone, 18)),
        _entry(_local(zone, 20)),
        _entry(_local(zone, 9, day_offset=1)),
        _entry(_local(zone, 9, 5, day_offset=1)),
    ]
    assert find_night_boundary(entries, settings=cfg) == (2, 1)
    # a late message inside the night window counts even with a short gap
    late = [
        _entry(_local(zone, 2, day_offset=1)),
        _entry(_local(zone, 7, 30, day_offset=1)),
    ]
    assert find_night_boundary(late, settings=cfg) == (1, 1)
    # two waking days apart: two nights
    long_gap = [_entry(_local(zone, 20)), _entry(_local(zone, 10, day_offset=2))]
    assert find_night_boundary(long_gap, settings=cfg) == (1, 2)
    # the newest night is the one returned
    two = [
        _entry(_local(zone, 20)),
        _entry(_local(zone, 10, day_offset=1)),
        _entry(_local(zone, 10, day_offset=2)),
    ]
    assert find_night_boundary(two, settings=cfg) == (2, 1)


@pytest.mark.parametrize("zone", ZONES)
def test_no_boundary_without_gap(zone):
    cfg = _settings(zone)
    # a chat across midnight stays in one waking day
    midnight = [
        _entry(_local(zone, 22, 30)),
        _entry(_local(zone, 23, 50)),
        _entry(_local(zone, 0, 40, day_offset=1)),
    ]
    assert find_night_boundary(midnight, settings=cfg) is None
    # a late goodnight and an early word before the wake hour: same waking day
    early = [_entry(_local(zone, 22)), _entry(_local(zone, 5, day_offset=1))]
    assert find_night_boundary(early, settings=cfg) is None
    # a long daytime gap inside one day is not a night
    daytime = [_entry(_local(zone, 8)), _entry(_local(zone, 21))]
    assert find_night_boundary(daytime, settings=cfg) is None
    # out of order and too short inputs
    backwards = [_entry(_local(zone, 9, day_offset=1)), _entry(_local(zone, 20))]
    assert find_night_boundary(backwards, settings=cfg) is None
    assert find_night_boundary([_entry(_local(zone, 9))], settings=cfg) is None
    assert find_night_boundary([], settings=cfg) is None


@pytest.mark.parametrize("zone", ZONES)
def test_note_marked_not_a_person(zone):
    note = build_night_note(1)
    assert note["role"] == "user"
    assert note["speaker"] == "clock"
    assert note["is_clock_note"] is True
    assert note["content"] == NIGHT_NOTE_TEXT
    text = NIGHT_NOTE_TEXT.lower()
    assert "not a person" in text and "night passed" in text
    assert "above this note" in text and "before that night" in text
    plural = build_night_note(3)["content"]
    assert "3 nights passed" in plural
    assert NIGHT_NOTE_TEXT.isascii() and plural.isascii()
    # inserted only when the marker is enabled
    entries = [_entry(_local(zone, 20)), _entry(_local(zone, 9, day_offset=1))]
    off = apply_night_seam(entries, settings=_settings(zone))
    assert off == entries and off is not entries
    on = apply_night_seam(
        entries, settings=_settings(zone, NIGHT_SEAM_MARKER_ENABLED=True)
    )
    assert [e.get("is_clock_note", False) for e in on] == [False, True, False]
    # a night before the current message puts the note at the end
    tail = apply_night_seam(
        entries[:1],
        now=_local(zone, 9, day_offset=1),
        settings=_settings(zone, NIGHT_SEAM_MARKER_ENABLED=True),
    )
    assert tail[-1]["is_clock_note"] is True
    # never doubled
    again = apply_night_seam(
        on, settings=_settings(zone, NIGHT_SEAM_MARKER_ENABLED=True)
    )
    assert again == on


@pytest.mark.parametrize("zone", ZONES)
def test_min_gap_from_settings(zone):
    # 21:00 to 08:00 the next morning is eleven hours: a night at six hours,
    # not at twelve (the earlier stamp is outside the night window)
    entries = [_entry(_local(zone, 21)), _entry(_local(zone, 8, day_offset=1))]
    assert find_night_boundary(entries, settings=_settings(zone)) == (1, 1)
    strict = _settings(zone, NIGHT_MIN_GAP_HOURS=12.0)
    assert find_night_boundary(entries, settings=strict) is None
    assert Settings(_env_file=None).NIGHT_MIN_GAP_HOURS == 6.0
    assert Settings(_env_file=None).NIGHT_SEAM_MARKER_ENABLED is False


@pytest.mark.parametrize("zone", ZONES)
def test_mixed_aware_naive_never_raises(zone):
    cfg = _settings(zone)
    evening = _local(zone, 20)
    morning = _local(zone, 9, day_offset=1)
    entries = [
        _entry(evening.replace(tzinfo=None)),  # naive counts as UTC
        {"role": "user", "content": "no stamp"},
        _entry("not a timestamp"),
        _entry(None),
        _entry(morning.isoformat()),
        _entry(morning),
    ]
    found = find_night_boundary(entries, settings=cfg)
    assert found == (4, 1)
    assert infer_last_turn_at(entries) == morning
    assert find_night_boundary(["garbage", 17, object()], settings=cfg) is None


@pytest.mark.parametrize("zone", ZONES)
def test_infer_last_turn_at_is_aware(zone):
    stamps = [_local(zone, 9), _local(zone, 15), _local(zone, 12)]
    entries = [_entry(s) for s in stamps]
    entries.append(build_night_note(1))
    found = infer_last_turn_at(entries)
    assert found == _local(zone, 15)
    assert found.utcoffset() == timedelta(0)
    assert infer_last_turn_at([]) is None
    assert infer_last_turn_at([{"content": "x"}]) is None
    naive = infer_last_turn_at([_entry(datetime(2000, 1, 1, 12, 0))])
    assert naive == datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
