"""Temporal context: configured zone, DST, night window, daylight gate."""

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from mtc.consciousness import temporal_context as tc
from mtc.consciousness.temporal_context import (
    build_now_block,
    build_now_line,
    in_night_window,
    part_of_day,
    solar_times,
)
from mtc.core.config import Settings
from mtc.devtools.privacy_audit import PATTERNS

UTC = timezone.utc


def make_settings(tmp_path, monkeypatch, **overrides):
    monkeypatch.chdir(tmp_path)
    return Settings(**overrides)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)

    def build(**overrides):
        return make_settings(tmp_path, monkeypatch, **overrides)

    return build


@pytest.mark.parametrize(
    "zone,expected_time,abbreviation",
    [
        ("UTC", "12:00", "UTC"),
        ("Europe/Berlin", "13:00", "CET"),
        ("Asia/Tokyo", "21:00", "JST"),
    ],
)
def test_now_line_uses_configured_timezone(cfg, zone, expected_time, abbreviation):
    now = datetime(2000, 1, 1, 12, 0, tzinfo=UTC)
    line = build_now_line(now, settings=cfg(TIMEZONE=zone))
    assert "Saturday" in line
    assert "2000-01-01" in line
    assert expected_time in line
    assert f" {abbreviation}" in line
    assert line.endswith(".")
    assert "\n" not in line


def test_dst_abbreviation_switches_on_transition_day(cfg):
    settings = cfg(TIMEZONE="Europe/Berlin")
    before = build_now_line(datetime(2000, 3, 25, 12, 0, tzinfo=UTC), settings=settings)
    after = build_now_line(datetime(2000, 3, 26, 12, 0, tzinfo=UTC), settings=settings)
    # The transition happens at 01:00 UTC on the last Sunday of March.
    before_transition = build_now_line(
        datetime(2000, 3, 26, 0, 30, tzinfo=UTC), settings=settings
    )
    assert "CET" in before and "CEST" not in before
    assert "CEST" in after
    assert "CET" in before_transition and "CEST" not in before_transition


def test_part_of_day_follows_configured_hours():
    def at(hour, minute=0):
        return datetime(2000, 1, 1, hour, minute)

    assert part_of_day(at(6), 6, 22) == "morning"
    assert part_of_day(at(11, 59), 6, 22) == "morning"
    assert part_of_day(at(12), 6, 22) == "afternoon"
    assert part_of_day(at(16, 59), 6, 22) == "afternoon"
    assert part_of_day(at(17), 6, 22) == "evening"
    assert part_of_day(at(21, 59), 6, 22) == "evening"
    assert part_of_day(at(22), 6, 22) == "night"
    assert part_of_day(at(3), 6, 22) == "night"
    # Hours that overlap the fixed boundaries clip them.
    assert part_of_day(at(13), 13, 3) == "afternoon"
    assert part_of_day(at(18), 13, 3) == "evening"
    assert part_of_day(at(1), 13, 3) == "evening"
    assert part_of_day(at(5), 13, 3) == "night"
    assert part_of_day(at(9), 6, 10) == "morning"
    assert part_of_day(at(13), 6, 14) == "afternoon"


def test_night_window_wraps_midnight():
    assert in_night_window(23.5, 6, 22)
    assert in_night_window(5.0, 6, 22)
    assert in_night_window(22.0, 6, 22)
    assert not in_night_window(6.0, 6, 22)
    assert not in_night_window(12.0, 6, 22)
    assert not in_night_window(21.99, 6, 22)
    # Sleep before wake within one calendar day.
    assert in_night_window(2.0, 10, 2)
    assert in_night_window(9.99, 10, 2)
    assert not in_night_window(10.0, 10, 2)
    assert not in_night_window(1.99, 10, 2)
    assert not in_night_window(23.0, 10, 2)


def test_waking_hours_remaining(cfg):
    settings = cfg()
    block = build_now_block(datetime(2000, 1, 1, 14, 30, tzinfo=UTC), settings=settings)
    assert block.startswith("# Current time")
    assert "06:00" in block and "22:00" in block
    assert "7.5 hours" in block
    night = build_now_block(datetime(2000, 1, 1, 23, 30, tzinfo=UTC), settings=settings)
    assert "night window" in night
    assert "hours remain" not in night


def test_daylight_lines_absent_without_coordinates(cfg):
    block = build_now_block(
        datetime(2000, 3, 20, 12, 0, tzinfo=UTC), settings=cfg(TIMEZONE="UTC")
    )
    assert "unrise" not in block and "unset" not in block


def minutes_of(moment: datetime) -> float:
    return moment.hour * 60 + moment.minute + moment.second / 60


def test_sunrise_sunset_at_equator_on_equinox():
    result = solar_times(date(2000, 3, 20), 0.0, 0.0, ZoneInfo("UTC"))
    assert result is not None
    sunrise, sunset = result
    assert abs(minutes_of(sunrise) - 6 * 60) <= 20
    assert abs(minutes_of(sunset) - 18 * 60) <= 20
    assert sunrise.tzinfo is not None


def test_daylight_lines_present_when_gate_passes(cfg):
    settings = cfg(TIMEZONE="UTC", LOCATION_LAT=0.0, LOCATION_LON=0.0)
    block = build_now_block(datetime(2000, 3, 20, 12, 0, tzinfo=UTC), settings=settings)
    assert "Sunrise" in block and "sunset" in block


def test_solar_times_follow_the_zone_for_east_and_west_longitudes():
    east = solar_times(date(2000, 3, 20), 35.0, 140.0, ZoneInfo("Asia/Tokyo"))
    west = solar_times(date(2000, 3, 20), 35.0, -120.0, ZoneInfo("Etc/GMT+8"))
    for result in (east, west):
        assert result is not None
        sunrise, sunset = result
        assert sunrise.date() == date(2000, 3, 20)
        assert sunset.date() == date(2000, 3, 20)
        assert 5 * 60 <= minutes_of(sunrise) <= 7.5 * 60
        assert 17 * 60 <= minutes_of(sunset) <= 19.5 * 60


def test_polar_day_has_no_sunrise_line(cfg):
    assert solar_times(date(2000, 6, 21), 80.0, 0.0, ZoneInfo("UTC")) is None
    assert solar_times(date(2000, 12, 21), 80.0, 0.0, ZoneInfo("UTC")) is None
    settings = cfg(TIMEZONE="UTC", LOCATION_LAT=80.0, LOCATION_LON=0.0)
    block = build_now_block(datetime(2000, 6, 21, 12, 0, tzinfo=UTC), settings=settings)
    assert "unrise" not in block and "unset" not in block


def test_implausible_combination_is_gated(cfg, caplog):
    settings = cfg(TIMEZONE="Asia/Tokyo", LOCATION_LAT=0.0, LOCATION_LON=0.0)
    with caplog.at_level(logging.WARNING, logger=tc.__name__):
        block = build_now_block(
            datetime(2000, 3, 20, 3, 0, tzinfo=UTC), settings=settings
        )
    assert "unrise" not in block and "unset" not in block
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for key in ("TIMEZONE", "LOCATION_LAT", "LOCATION_LON"):
        assert key in message


def test_gap_note_after_long_silence(cfg):
    settings = cfg()
    now = datetime(2000, 1, 1, 12, 0, tzinfo=UTC)
    long_gap = build_now_block(
        now, last_exchange_at=now - timedelta(hours=5, minutes=12), settings=settings
    )
    assert "5 hours 12 minutes" in long_gap
    assert "before that gap" in long_gap
    short_gap = build_now_block(
        now, last_exchange_at=now - timedelta(minutes=12), settings=settings
    )
    assert "12 minutes" in short_gap
    assert "before that gap" not in short_gap
    boundary = build_now_block(
        now,
        last_exchange_at=now - timedelta(hours=4),
        gap_note_hours=4.0,
        settings=settings,
    )
    assert "before that gap" in boundary
    future = build_now_block(
        now, last_exchange_at=now + timedelta(hours=1), settings=settings
    )
    assert "before that gap" not in future
    none_given = build_now_block(now, settings=settings)
    assert "last exchange" not in none_given.lower()


def test_never_raises_on_bad_input(cfg):
    naive = datetime(2000, 1, 1, 12, 0)
    line = build_now_line(naive, settings=cfg(TIMEZONE="Asia/Tokyo"))
    assert "21:00" in line  # naive instants count as UTC
    broken = SimpleNamespace(
        TIMEZONE="Not/AZone",
        WAKE_HOUR=6,
        SLEEP_HOUR=22,
        LOCATION_LAT=None,
        LOCATION_LON=None,
    )
    fallback = build_now_line(datetime(2000, 1, 1, 12, 0, tzinfo=UTC), settings=broken)
    assert "UTC" in fallback and "12:00" in fallback
    block = build_now_block(datetime(2000, 1, 1, 12, 0, tzinfo=UTC), settings=broken)
    assert block.startswith("# Current time") and "UTC" in block
    no_attrs = SimpleNamespace()
    assert "UTC" in build_now_line(naive, settings=no_attrs)
    assert "UTC" in build_now_line("not a datetime", settings=cfg())  # type: ignore
    assert isinstance(
        build_now_block(
            naive, last_exchange_at="garbage", settings=cfg()  # type: ignore
        ),
        str,
    )


def test_solar_times_never_raises_on_bad_input():
    assert solar_times(date(2000, 3, 20), float("nan"), 0.0, ZoneInfo("UTC")) is None
    assert solar_times(date(2000, 3, 20), 95.0, 0.0, ZoneInfo("UTC")) is None


def test_module_source_has_no_zone_or_coordinate_literal():
    source = Path(tc.__file__).read_text(encoding="utf-8")
    zones = re.findall(r"\b[A-Z][A-Za-z_]+/[A-Z][A-Za-z_+\-0-9]+\b", source)
    assert zones == []
    assert not PATTERNS["coords"].search(source)
    for line in source.splitlines():
        assert not PATTERNS["coords"].search(line), line
