"""The night seam in a history window and the last-turn time.

The token budget runs once, in the brain (``mtc.brain.budget``); this module
keeps the other half of a history window's bookkeeping:

- ``find_night_boundary`` finds where a night fell between two adjacent
  entries. A night lies between an earlier and a later timestamp when their
  waking days differ and either the gap is at least ``NIGHT_MIN_GAP_HOURS``
  or the earlier timestamp falls inside the night window. The waking day of
  an instant is its local date in ``TIMEZONE`` after subtracting
  ``WAKE_HOUR`` hours, so a message at 00:30 or just before the wake hour
  still belongs to the day before; the night window runs from ``SLEEP_HOUR``
  to ``WAKE_HOUR`` (``mtc.consciousness.temporal_context.in_night_window``).
  A chat that runs across midnight, or a late message followed by an early
  one inside the same waking day, is not a night. The number of nights is the
  difference of the waking days. Walking from the newest pair back, the first
  night found is returned.
- ``build_night_note`` returns the clock note that goes where the night fell:
  a history entry (role ``user``, speaker ``clock``, ``is_clock_note`` true)
  that says on its face it is not a person, that a night passed and that the
  lines above it come from before the night. The brain's single budget pass
  drops a clock note left at the head of the kept window.
- ``apply_night_seam`` inserts that note when ``NIGHT_SEAM_MARKER_ENABLED`` is
  true (it is false by default) and a boundary exists.
- ``infer_last_turn_at`` recovers the newest timestamp of a rehydrated
  history, so the clock's gap lines survive a restart.

Timestamps are instants: an aware datetime or an ISO string; a naive value
counts as UTC (the project's stored timestamps are aware UTC). An entry with
no readable timestamp and every clock note are skipped, so pairs are adjacent
among the readable entries. Local time is used only for the waking-day and
night-window rules. None of the functions raises on unreadable input. Config
keys: ``TIMEZONE``, ``WAKE_HOUR``, ``SLEEP_HOUR``, ``NIGHT_MIN_GAP_HOURS`` and
``NIGHT_SEAM_MARKER_ENABLED``, read at call time. Histories hold private
conversation text; nothing here logs it.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from mtc.consciousness.temporal_context import in_night_window
from mtc.core.config import get_settings

logger = logging.getLogger(__name__)

CLOCK_SPEAKER = "clock"
NIGHT_NOTE_TEXT = (
    "[Clock note, not a person: a night passed at this point; the lines above "
    "this note come from before that night.]"
)
_NIGHTS_NOTE_TEXT = (
    "[Clock note, not a person: {nights} nights passed at this point; the lines "
    "above this note come from before those nights.]"
)


def _instant(raw: Any) -> Optional[datetime]:
    """An aware UTC instant from a datetime or ISO string, or None."""
    if raw is None:
        return None
    try:
        moment = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if moment.tzinfo is None or moment.utcoffset() is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _stamp(entry: Any) -> Optional[datetime]:
    if isinstance(entry, dict):
        if entry.get("is_clock_note"):
            return None
        return _instant(entry.get("timestamp"))
    return _instant(entry)


def _rules(settings: Any) -> Tuple[ZoneInfo, int, int, timedelta]:
    cfg = settings if settings is not None else get_settings()
    return (
        ZoneInfo(str(cfg.TIMEZONE)),
        int(cfg.WAKE_HOUR),
        int(cfg.SLEEP_HOUR),
        timedelta(hours=float(cfg.NIGHT_MIN_GAP_HOURS)),
    )


def _waking_day(local: datetime, wake_hour: int) -> date:
    wall = local.replace(tzinfo=None)
    try:
        return (wall - timedelta(hours=wake_hour)).date()
    except OverflowError:
        return wall.date()


def _local_hour(local: datetime) -> float:
    return local.hour + local.minute / 60.0 + local.second / 3600.0


def find_night_boundary(
    entries: Sequence[Any], *, settings: Any = None
) -> Optional[Tuple[int, int]]:
    """Return ``(index, nights)`` for the newest night between adjacent
    entries, or None.

    ``index`` is the position in ``entries`` of the later entry of the pair
    (the note goes immediately before it) and ``nights`` the number of nights
    that passed. Entries are history dicts (their ``timestamp`` is read) or
    bare timestamps.
    """
    if not entries:
        return None
    try:
        zone, wake_hour, sleep_hour, min_gap = _rules(settings)
    except Exception as exc:  # noqa: BLE001 - never raises; reported once
        logger.warning("night seam rules unreadable: %s", type(exc).__name__)
        return None
    later_index: Optional[int] = None
    later: Optional[datetime] = None
    for index in range(len(entries) - 1, -1, -1):
        moment = _stamp(entries[index])
        if moment is None:
            continue
        if later is not None:
            try:
                earlier_local = moment.astimezone(zone)
                later_local = later.astimezone(zone)
                nights = (
                    _waking_day(later_local, wake_hour)
                    - _waking_day(earlier_local, wake_hour)
                ).days
                at_night = in_night_window(
                    _local_hour(earlier_local), wake_hour, sleep_hour
                )
                if nights > 0 and (later - moment >= min_gap or at_night):
                    return later_index, nights
            except (OverflowError, ValueError):
                pass
        later_index, later = index, moment
    return None


def build_night_note(nights: int) -> Dict[str, Any]:
    """The clock note for ``nights`` nights (at least one)."""
    count = max(1, int(nights))
    text = NIGHT_NOTE_TEXT if count == 1 else _NIGHTS_NOTE_TEXT.format(nights=count)
    return {
        "role": "user",
        "speaker": CLOCK_SPEAKER,
        "content": text,
        "timestamp": None,
        "is_clock_note": True,
    }


def apply_night_seam(
    entries: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    settings: Any = None,
) -> List[Dict[str, Any]]:
    """Return a copy of ``entries`` with the night note inserted where the
    newest night fell, when ``NIGHT_SEAM_MARKER_ENABLED`` is true.

    ``now`` is the current message's instant: a night between the last entry
    and it puts the note at the end of the copy. Existing clock notes are
    kept and never doubled.
    """
    copied = [dict(entry) for entry in entries]
    cfg = settings if settings is not None else get_settings()
    if not bool(cfg.NIGHT_SEAM_MARKER_ENABLED):
        return copied
    stamps: List[Any] = list(copied)
    if now is not None:
        stamps.append({"timestamp": now})
    found = find_night_boundary(stamps, settings=cfg)
    if found is None:
        return copied
    index, nights = found
    if index > 0 and copied[index - 1].get("is_clock_note"):
        return copied
    copied.insert(index, build_night_note(nights))
    return copied


def infer_last_turn_at(entries: Sequence[Any]) -> Optional[datetime]:
    """Return the newest readable timestamp of ``entries`` as aware UTC, or
    None. Clock notes and unreadable stamps are skipped; never raises."""
    newest: Optional[datetime] = None
    for entry in entries or ():
        moment = _stamp(entry)
        if moment is not None and (newest is None or moment > newest):
            newest = moment
    return newest


__all__ = (
    "CLOCK_SPEAKER",
    "NIGHT_NOTE_TEXT",
    "apply_night_seam",
    "build_night_note",
    "find_night_boundary",
    "infer_last_turn_at",
)
