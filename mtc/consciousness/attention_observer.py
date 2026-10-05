"""
Attention-Capture Journal
=========================

Observe-only record of the moments attention was pulled rather than chosen:
the attention schema's ``captured`` and ``emotional`` shifts. Recording
changes nothing in the attention dynamics. The Phase 6 orchestrator calls
``observe`` with the schema's current focus after each cycle when the journal
is enabled.

Config keys, read at call time from ``settings`` (or ``get_settings()``):

- ``ATTENTION_GRAB_NOTICING_ENABLED`` (default off) gates recording unless the
  constructor's ``enabled`` is given; disabled, ``observe`` is a no-op.
- ``AGENT_SPACE_DIR``: the journal is ``<AGENT_SPACE_DIR>/attention/
  grab_journal.md`` unless ``journal_dir`` is given; created on the first
  recorded entry.
- ``TIMEZONE``: the local display time of each entry.

Each recorded focus is reduced to a ``GrabEntry``: the summary with reasoning
blocks removed (``strip_reasoning_blocks`` over ``reasoning_tags``) and
whitespace collapsed, the source module, the shift type, a strength band and
the focus key (``AttentionSchemaModule.focus_key`` over the cleaned summary,
the schema's own key whenever the summary holds no reasoning block). A focus
identical to the previous observed focus is not recorded again; once the
focus moves on, a return to it is a new entry. Strength bands come from the
schema's thresholds on its 0 to 1.5 strength scale: ``gentle`` below the focus
threshold, ``clear`` from it up to the capture threshold, ``strong`` at or
above it, ``unknown`` for a non-finite strength.

Entries go to an ``AttentionGrabStore`` (``mtc.memory.protocols``; None means
``InMemoryAttentionGrabStore``) with an aware UTC ISO timestamp, and as one
line to the journal. Both writes are independent and fail soft: ``observe``
never raises. Grab entries can hold real people's names at runtime and are
private data; the journal lives in the gitignored agent space.
"""

import logging
import math
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from mtc.brain.protocol import strip_reasoning_blocks
from mtc.consciousness.attention_schema import (
    DEFAULT_CAPTURE_SALIENCE_THRESHOLD,
    DEFAULT_FOCUS_THRESHOLD,
    AttentionSchemaModule,
    AttentionTarget,
)
from mtc.core.config import get_settings
from mtc.memory.inmemory import InMemoryAttentionGrabStore
from mtc.memory.protocols import AttentionGrabStore

logger = logging.getLogger(__name__)

GRAB_SHIFT_TYPES = frozenset({"captured", "emotional"})
SUMMARY_MAX_CHARS = 140
JOURNAL_NAME = "grab_journal.md"
_JOURNAL_HEADER = "# Attention-capture journal\n\n"


@dataclass
class GrabEntry:
    """One recorded attention capture.

    ``timestamp`` is an aware UTC ISO string; ``local_time`` is the display
    time in ``TIMEZONE``.
    """

    grab_id: str
    timestamp: str
    local_time: str
    summary: str
    source_module: str
    shift_type: str
    strength_band: str
    focus_key: str


def _shift_value(focus: Any) -> Optional[str]:
    shift = getattr(focus, "shift_type", None)
    value = getattr(shift, "value", shift)
    return value.lower() if isinstance(value, str) else None


class AttentionGrabObserver:
    """Records captured and emotional attention shifts.

    Args:
        store: Journal store; None means a new in-memory store.
        journal_dir: Folder of the journal file; None derives
            ``<AGENT_SPACE_DIR>/attention`` at write time.
        enabled: Overrides ``ATTENTION_GRAB_NOTICING_ENABLED`` when not None.
        reasoning_tags: Tags whose blocks are removed from summaries.
        settings: Settings object to read; None reads ``get_settings()``.
        clock: Returns the current aware UTC time.
        focus_threshold: Strength at which a capture is ``clear``.
        capture_threshold: Strength at which a capture is ``strong``.
    """

    def __init__(
        self,
        store: Optional[AttentionGrabStore] = None,
        journal_dir: Optional[Path] = None,
        enabled: Optional[bool] = None,
        reasoning_tags: Sequence[str] = ("think",),
        settings: Any = None,
        clock: Optional[Callable[[], datetime]] = None,
        *,
        focus_threshold: float = DEFAULT_FOCUS_THRESHOLD,
        capture_threshold: float = DEFAULT_CAPTURE_SALIENCE_THRESHOLD,
    ) -> None:
        self.store: AttentionGrabStore = (
            store if store is not None else InMemoryAttentionGrabStore()
        )
        self._journal_dir = Path(journal_dir) if journal_dir is not None else None
        self._enabled = enabled
        self.reasoning_tags = tuple(reasoning_tags)
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.focus_threshold = float(focus_threshold)
        self.capture_threshold = float(capture_threshold)
        self._last_focus_key: Optional[str] = None

    def _config(self) -> Any:
        return self._settings if self._settings is not None else get_settings()

    @property
    def enabled(self) -> bool:
        """The constructor override, else the setting read now."""
        if self._enabled is not None:
            return bool(self._enabled)
        return bool(getattr(self._config(), "ATTENTION_GRAB_NOTICING_ENABLED", False))

    @property
    def journal_path(self) -> Path:
        """The journal file, resolved now."""
        folder = self._journal_dir
        if folder is None:
            folder = Path(self._config().AGENT_SPACE_DIR) / "attention"
        return folder / JOURNAL_NAME

    def strength_band(self, strength: Any) -> str:
        """``gentle``, ``clear``, ``strong`` or ``unknown`` for a strength."""
        try:
            value = float(strength)
        except (TypeError, ValueError):
            return "unknown"
        if not math.isfinite(value):
            return "unknown"
        if value >= self.capture_threshold:
            return "strong"
        if value >= self.focus_threshold:
            return "clear"
        return "gentle"

    def _clean_summary(self, summary: Any) -> str:
        text = strip_reasoning_blocks(str(summary or ""), self.reasoning_tags)
        return " ".join(text.split())[:SUMMARY_MAX_CHARS]

    async def observe(self, focus: Optional[AttentionTarget]) -> Optional[GrabEntry]:
        """Record the focus when it is a new capture; never raises.

        Returns the recorded entry, or None when disabled, when there is no
        focus, when the shift was chosen or habitual, or when the focus is
        the one observed last.
        """
        if not self.enabled:
            return None
        try:
            if focus is None:
                self._last_focus_key = None
                return None
            summary = self._clean_summary(getattr(focus, "summary", ""))
            source = str(getattr(focus, "source_module", "") or "")
            key = AttentionSchemaModule.focus_key(source, summary)
            previous, self._last_focus_key = self._last_focus_key, key
            kind = _shift_value(focus)
            if kind not in GRAB_SHIFT_TYPES or key == previous:
                return None
            moment = self._clock().astimezone(timezone.utc)
            zone = ZoneInfo(str(getattr(self._config(), "TIMEZONE", "UTC") or "UTC"))
            entry = GrabEntry(
                grab_id=uuid.uuid4().hex,
                timestamp=moment.isoformat(),
                local_time=moment.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z"),
                summary=summary,
                source_module=source,
                shift_type=kind,
                strength_band=self.strength_band(
                    getattr(focus, "attention_strength", None)
                ),
                focus_key=key,
            )
        except Exception as e:
            logger.warning(f"Attention capture observation failed: {e}")
            return None
        await self._persist(entry)
        logger.info(
            f"Attention capture recorded ({entry.shift_type}, {entry.strength_band}) "
            f"from {entry.source_module or 'an unnamed source'}"
        )
        return entry

    async def _persist(self, entry: GrabEntry) -> None:
        """Write the entry to the store and to the journal, independently."""
        try:
            await self.store.add_grab(asdict(entry))
        except Exception as e:
            logger.warning(f"Attention capture store write failed: {e}")
        try:
            path = self.journal_path
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            with open(path, "a", encoding="utf-8") as handle:
                if new_file:
                    handle.write(_JOURNAL_HEADER)
                handle.write(self._line(entry) + "\n")
        except Exception as e:
            logger.warning(f"Attention capture journal write failed: {e}")

    @staticmethod
    def _line(entry: GrabEntry) -> str:
        source = f" (source: {entry.source_module})" if entry.source_module else ""
        return (
            f"- {entry.local_time} [{entry.shift_type}, {entry.strength_band}] "
            f"{entry.summary}{source}"
        )

    async def recent(self, limit: int = 20) -> List[GrabEntry]:
        """The latest entries from the store, newest first."""
        names = {f.name for f in fields(GrabEntry)}
        entries = []
        for doc in await self.store.recent_grabs(limit):
            try:
                entries.append(GrabEntry(**{k: doc[k] for k in names}))
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipped an unreadable attention capture entry: {e}")
        return entries

    def format_for_reading(self, entries: Sequence[GrabEntry]) -> str:
        """A neutral plain-text listing of the entries in the order given."""
        if not entries:
            return "No attention captures recorded."
        return "Recorded attention captures:\n" + "\n".join(
            self._line(entry) for entry in entries
        )
