"""
Shadow signal log
=================

An append-only JSON Lines writer for signals that run in shadow: measured and
recorded, never narrated and never fed back into behavior. One file per
signal, ``<directory>/<signal_id>.jsonl``, one JSON object per line holding
``ts`` (timezone-aware UTC, ISO 8601), ``signal_id`` and the caller's record.

A shadow instrument must not break the live pipeline, so ``log`` never raises:
a failed write (unwritable directory, a record that is not JSON, a non-finite
number) returns False and logs one warning line. The directory is created on
the first write, never earlier.

Config: ``Settings.SHADOW_LOG_DIR`` (empty derives ``<LOG_DIRECTORY>/
shadow_signals``), read at call time when no directory is passed to the
constructor. Records may hold conversation-derived text and are private data.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mtc.core.config import get_settings

logger = logging.getLogger(__name__)


def resolve_shadow_directory() -> Path:
    """The configured shadow log directory, read from settings at call time."""
    settings = get_settings()
    configured = getattr(settings, "SHADOW_LOG_DIR", "") or ""
    if configured:
        return Path(configured)
    return Path(settings.LOG_DIRECTORY) / "shadow_signals"


class ShadowSignalLogger:
    """Append-only JSONL writer for one shadow signal."""

    def __init__(self, signal_id: str, directory: Optional[Path] = None) -> None:
        if not signal_id or "/" in signal_id or "\\" in signal_id or ".." in signal_id:
            raise ValueError("signal_id must be a plain file stem")
        self.signal_id = signal_id
        self._directory = Path(directory) if directory is not None else None

    @property
    def directory(self) -> Path:
        """Explicit directory, or the settings directory resolved now."""
        return self._directory or resolve_shadow_directory()

    def log(self, record: Dict[str, Any]) -> bool:
        """Append one record; True when written, False on any failure."""
        try:
            directory = self.directory
            line = json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "signal_id": self.signal_id,
                    **record,
                },
                default=str,
                allow_nan=False,
            )
            directory.mkdir(parents=True, exist_ok=True)
            with open(
                directory / f"{self.signal_id}.jsonl", "a", encoding="utf-8"
            ) as fh:
                fh.write(line + "\n")
            return True
        except Exception as exc:  # a shadow instrument never breaks the caller
            logger.warning("shadow log write failed for %s: %s", self.signal_id, exc)
            return False
