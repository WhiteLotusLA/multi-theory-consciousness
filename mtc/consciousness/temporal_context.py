"""
Temporal Context
================

Builds the time and place lines a prompt assembler can prepend: one sentence
with the local weekday, date, time, zone abbreviation and part of day
(``build_now_line``), and a short neutral block with the configured waking
hours, optional daylight times and a note after a long silence
(``build_now_block``).

Every value comes from configuration and nothing is assumed:

- ``TIMEZONE`` (an IANA name) picks the display zone; the abbreviation comes
  from the zone database, so it follows daylight saving transitions.
- ``WAKE_HOUR`` and ``SLEEP_HOUR`` define the waking day. The night window
  runs from the sleep hour to the wake hour and wraps midnight when the sleep
  hour is the larger one.
- ``LOCATION_LAT`` and ``LOCATION_LON`` are optional. Sunrise and sunset lines
  appear only when both are set and a plausibility gate agrees that the
  coordinates and the zone describe the same part of the world.

Timestamps handed in are instants: a naive datetime counts as UTC. Display
conversion happens here and nowhere is anything stored. Both builders never
raise; any internal failure yields a line in UTC.

Sunrise and sunset use the public NOAA general solar-position equations
(fractional year, equation of time, solar declination, hour angle at a
zenith of 90.833 degrees, which includes refraction and the solar radius).
"""

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Tuple
from zoneinfo import ZoneInfo

from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Fixed boundaries between morning, afternoon and evening, in local hours.
NOON_HOUR = 12
AFTERNOON_END_HOUR = 17

# Sun-disc zenith used for sunrise and sunset, in degrees.
SUNRISE_ZENITH_DEGREES = 90.833
# Earth's axial tilt bounds the solar declination at the solstices.
SOLSTICE_DECLINATION_DEGREES = 23.44
# Solar noon must fall within this many hours of local clock noon.
NOON_TOLERANCE_HOURS = 3.0
# Allowed slack on the day length bounds, in minutes.
DAY_LENGTH_SLACK_MINUTES = 15.0

# Equation of time series terms (minutes = 229.18 * series).
EOT_SCALE = 229.18
EOT_CONST = 0.000075
EOT_COS1 = 0.001868
EOT_SIN1 = 0.032077
EOT_COS2 = 0.014615
EOT_SIN2 = 0.040849

# Solar declination series terms (radians).
DECL_CONST = 0.006918
DECL_COS1 = 0.399912
DECL_SIN1 = 0.070257
DECL_COS2 = 0.006758
DECL_SIN2 = 0.000907
DECL_COS3 = 0.002697
DECL_SIN3 = 0.00148


# ----------------------------------------------------------------------------
# Clock helpers
# ----------------------------------------------------------------------------


def in_night_window(local_hour: float, wake_hour: int, sleep_hour: int) -> bool:
    """Return True when a local hour falls inside the configured night window.

    The window runs from ``sleep_hour`` up to ``wake_hour`` and wraps midnight
    when ``sleep_hour`` is greater than ``wake_hour``.
    """
    if sleep_hour > wake_hour:
        return local_hour >= sleep_hour or local_hour < wake_hour
    return sleep_hour <= local_hour < wake_hour


def _hour_of(moment: datetime) -> float:
    return moment.hour + moment.minute / 60.0 + moment.second / 3600.0


def part_of_day(local: datetime, wake_hour: int, sleep_hour: int) -> str:
    """Name the part of day for a local time.

    ``night`` inside the night window; otherwise ``morning`` from the wake
    hour to 12:00, ``afternoon`` to 17:00 and ``evening`` to the sleep hour.
    The fixed boundaries are clipped to the waking day when the configured
    hours overlap them, so a late waking hour leaves no morning.
    """
    hour = _hour_of(local)
    if in_night_window(hour, wake_hour, sleep_hour):
        return "night"
    position = hour if hour >= wake_hour else hour + 24.0
    day_end = float(sleep_hour if sleep_hour > wake_hour else sleep_hour + 24)
    morning_end = min(max(float(NOON_HOUR), float(wake_hour)), day_end)
    afternoon_end = min(max(float(AFTERNOON_END_HOUR), morning_end), day_end)
    if position < morning_end:
        return "morning"
    if position < afternoon_end:
        return "afternoon"
    return "evening"


def _as_aware_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _clock(hour: int) -> str:
    return f"{hour:02d}:00"


# ----------------------------------------------------------------------------
# Solar position
# ----------------------------------------------------------------------------


def _solar_terms(day: date) -> Tuple[float, float]:
    """Return ``(equation_of_time_minutes, declination_radians)`` for a day."""
    days_in_year = 366 if _is_leap(day.year) else 365
    day_of_year = day.timetuple().tm_yday
    gamma = 2.0 * math.pi / days_in_year * (day_of_year - 1)
    eqtime = EOT_SCALE * (
        EOT_CONST
        + EOT_COS1 * math.cos(gamma)
        - EOT_SIN1 * math.sin(gamma)
        - EOT_COS2 * math.cos(2 * gamma)
        - EOT_SIN2 * math.sin(2 * gamma)
    )
    declination = (
        DECL_CONST
        - DECL_COS1 * math.cos(gamma)
        + DECL_SIN1 * math.sin(gamma)
        - DECL_COS2 * math.cos(2 * gamma)
        + DECL_SIN2 * math.sin(2 * gamma)
        - DECL_COS3 * math.cos(3 * gamma)
        + DECL_SIN3 * math.sin(3 * gamma)
    )
    return eqtime, declination


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _hour_angle_degrees(lat: float, declination: float) -> Optional[float]:
    """Sunrise hour angle in degrees, or None for polar day or night."""
    lat_rad = math.radians(lat)
    zenith = math.radians(SUNRISE_ZENITH_DEGREES)
    denominator = math.cos(lat_rad) * math.cos(declination)
    if abs(denominator) < 1e-12:
        return None
    cosine = math.cos(zenith) / denominator - math.tan(lat_rad) * math.tan(declination)
    if cosine <= -1.0 or cosine >= 1.0:
        return None
    return math.degrees(math.acos(cosine))


def solar_times(
    day: date, lat: float, lon: float, tz: ZoneInfo
) -> Optional[Tuple[datetime, datetime]]:
    """Return ``(sunrise, sunset)`` for a local calendar day, in ``tz``.

    Uses the NOAA general solar-position equations. Returns None for polar day
    or polar night and for inputs that are not finite or are out of range.
    """
    try:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            return None
        eqtime, declination = _solar_terms(day)
        hour_angle = _hour_angle_degrees(lat, declination)
        if hour_angle is None:
            return None
        midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        sunrise_minutes = 720.0 - 4.0 * (lon + hour_angle) - eqtime
        sunset_minutes = 720.0 - 4.0 * (lon - hour_angle) - eqtime
        sunrise = midnight + timedelta(minutes=sunrise_minutes)
        sunset = midnight + timedelta(minutes=sunset_minutes)
        return sunrise.astimezone(tz), sunset.astimezone(tz)
    except (ValueError, OverflowError, ArithmeticError):
        return None


def _day_length_minutes(lat: float, declination: float) -> float:
    lat_rad = math.radians(lat)
    zenith = math.radians(SUNRISE_ZENITH_DEGREES)
    denominator = math.cos(lat_rad) * math.cos(declination)
    if abs(denominator) < 1e-12:
        return 0.0
    cosine = math.cos(zenith) / denominator - math.tan(lat_rad) * math.tan(declination)
    if cosine >= 1.0:
        return 0.0
    if cosine <= -1.0:
        return 1440.0
    return 8.0 * math.degrees(math.acos(cosine))


def _daylight_is_plausible(
    day: date, lat: float, lon: float, tz: ZoneInfo, sunrise: datetime, sunset: datetime
) -> bool:
    """Check that the coordinates and the zone describe the same place.

    Local solar noon must fall within three hours of 12:00 local time, and the
    day length must sit within 15 minutes of the bounds the solstice
    declinations allow at this latitude.
    """
    eqtime, _ = _solar_terms(day)
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    noon_utc = midnight + timedelta(minutes=720.0 - 4.0 * lon - eqtime)
    noon_local = noon_utc.astimezone(tz)
    noon_hour = _hour_of(noon_local)
    if abs(noon_hour - NOON_HOUR) > NOON_TOLERANCE_HOURS:
        return False
    tilt = math.radians(SOLSTICE_DECLINATION_DEGREES)
    bounds = (_day_length_minutes(lat, tilt), _day_length_minutes(lat, -tilt))
    length = (sunset - sunrise).total_seconds() / 60.0
    low = min(bounds) - DAY_LENGTH_SLACK_MINUTES
    high = max(bounds) + DAY_LENGTH_SLACK_MINUTES
    return low <= length <= high


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------


def _line_for(local: datetime, wake_hour: int, sleep_hour: int) -> str:
    return (
        f"It is {local:%A}, {local:%Y-%m-%d}, {local:%H:%M} {local.tzname()}, "
        f"{part_of_day(local, wake_hour, sleep_hour)}."
    )


def _utc_instant(now: Any) -> datetime:
    if isinstance(now, datetime):
        return _as_aware_utc(now)
    return datetime.now(timezone.utc)


def build_now_line(
    now: Optional[datetime] = None, *, settings: Optional[Settings] = None
) -> str:
    """Return one sentence naming the local weekday, date, time and part of day.

    The abbreviation comes from the zone database and follows daylight saving
    time. Reads ``TIMEZONE``, ``WAKE_HOUR`` and ``SLEEP_HOUR``. Never raises.
    """
    instant = _utc_instant(now)
    try:
        cfg = settings if settings is not None else get_settings()
        local = instant.astimezone(ZoneInfo(cfg.TIMEZONE))
        return _line_for(local, int(cfg.WAKE_HOUR), int(cfg.SLEEP_HOUR))
    except Exception as exc:  # noqa: BLE001 - the contract is never to raise
        logger.warning("temporal context fell back to UTC: %s", type(exc).__name__)
        return _line_for(instant.astimezone(timezone.utc), 6, 22)


def _describe_gap(gap: timedelta) -> str:
    total_minutes = int(gap.total_seconds() // 60)
    days, remainder = divmod(total_minutes, 1440)
    hours, minutes = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or not parts:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts)


def _schedule_line(local: datetime, wake_hour: int, sleep_hour: int) -> str:
    hour = _hour_of(local)
    window = f"{_clock(wake_hour)} to {_clock(sleep_hour)}"
    if in_night_window(hour, wake_hour, sleep_hour):
        return (
            f"This moment is inside the configured night window "
            f"({_clock(sleep_hour)} to {_clock(wake_hour)})."
        )
    remaining = (sleep_hour - hour) % 24.0
    return f"Waking hours are {window}; {remaining:.1f} hours remain."


def _daylight_line(cfg: Any, local: datetime, tz: ZoneInfo) -> Optional[str]:
    lat = getattr(cfg, "LOCATION_LAT", None)
    lon = getattr(cfg, "LOCATION_LON", None)
    if lat is None or lon is None:
        return None
    result = solar_times(local.date(), float(lat), float(lon), tz)
    if result is None:
        return None
    sunrise, sunset = result
    if not _daylight_is_plausible(
        local.date(), float(lat), float(lon), tz, sunrise, sunset
    ):
        logger.warning(
            "daylight lines omitted: TIMEZONE, LOCATION_LAT and LOCATION_LON "
            "do not describe the same place"
        )
        return None
    return f"Sunrise is at {sunrise:%H:%M} and sunset at {sunset:%H:%M} today."


def build_now_block(
    now: Optional[datetime] = None,
    *,
    last_exchange_at: Optional[datetime] = None,
    gap_note_hours: float = 4.0,
    settings: Optional[Settings] = None,
) -> str:
    """Return a short neutral block describing the current time.

    Lines: local time, the configured waking hours with the hours remaining (or
    a note that the moment is inside the night window), sunrise and sunset only
    when both coordinates are set and the plausibility gate passes, the time
    since ``last_exchange_at`` and, at or beyond ``gap_note_hours``, a line
    saying earlier messages come from before that gap. Never raises.
    """
    instant = _utc_instant(now)
    header = "# Current time"
    try:
        cfg = settings if settings is not None else get_settings()
        tz = ZoneInfo(cfg.TIMEZONE)
        wake, sleep = int(cfg.WAKE_HOUR), int(cfg.SLEEP_HOUR)
        local = instant.astimezone(tz)
        lines = [
            header,
            _line_for(local, wake, sleep),
            _schedule_line(local, wake, sleep),
        ]
        daylight = _daylight_line(cfg, local, tz)
        if daylight:
            lines.append(daylight)
    except Exception as exc:  # noqa: BLE001 - the contract is never to raise
        logger.warning("temporal context fell back to UTC: %s", type(exc).__name__)
        lines = [header, _line_for(instant.astimezone(timezone.utc), 6, 22)]
    lines.extend(_gap_lines(instant, last_exchange_at, gap_note_hours))
    return "\n".join(lines)


def _gap_lines(instant: datetime, last: Any, gap_note_hours: float) -> list:
    if not isinstance(last, datetime):
        return []
    try:
        gap = instant - _as_aware_utc(last)
        if gap < timedelta(0):
            return []
        lines = [f"Time since the last exchange: {_describe_gap(gap)}."]
        if gap >= timedelta(hours=gap_note_hours):
            lines.append(
                "Earlier messages in this conversation come from before that gap."
            )
        return lines
    except Exception as exc:  # noqa: BLE001 - never raise
        logger.warning("gap note skipped: %s", type(exc).__name__)
        return []
