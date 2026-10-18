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
constructor and resolved against ``PROJECT_ROOT`` (``mtc.core.paths``), so
the writer and the gate runner name the same directory whatever the working
directory. Records may hold conversation-derived text and are private data.

Test hygiene: the acceptance gates read the settings directory as live shadow
evidence, and test runs exercise the same taps. So under pytest (the
``PYTEST_CURRENT_TEST`` environment variable is set) a logger built without a
directory writes nothing and ``log`` returns False; a logger given an explicit
directory still writes.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mtc.core.config import get_settings
from mtc.core.paths import log_dir, shadow_log_dir

logger = logging.getLogger(__name__)


def resolve_shadow_directory() -> Path:
    """The configured shadow log directory, read from settings at call time.

    The same directory ``mtc.core.paths.shadow_log_dir`` gives the gate
    runner: a relative setting joins ``PROJECT_ROOT``, never the working
    directory. An empty ``SHADOW_LOG_DIR`` derives
    ``<LOG_DIRECTORY>/shadow_signals``, resolved the same way.
    """
    settings = get_settings()
    configured = getattr(settings, "SHADOW_LOG_DIR", "") or ""
    if configured:
        return shadow_log_dir(settings)
    return log_dir(settings) / "shadow_signals"


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
        """Append one record; True when written, False on any failure.

        Under pytest a logger without an explicit directory writes nothing.
        """
        if self._directory is None and "PYTEST_CURRENT_TEST" in os.environ:
            logger.debug("shadow log write skipped under pytest: %s", self.signal_id)
            return False
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
