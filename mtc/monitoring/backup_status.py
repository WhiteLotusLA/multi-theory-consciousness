"""
Backup Status
=============

Reads the newest backup ``scripts/ops/backup.sh`` wrote and says whether it
is fresh and complete. It never makes, changes or deletes a backup; it only
looks at file names and sizes and at the manifest's status lines, so it reads
no conversation text.

A backup is a directory named with a UTC timestamp (``YYYYMMDDTHHMMSSZ``)
under ``BACKUP_LOCATION`` (relative paths join ``PROJECT_ROOT``). For each
enabled store it holds one file: ``postgres.dump``, ``mongo.archive.gz``,
``qdrant-<QDRANT_COLLECTION>.snapshot`` and ``redis.rdb``, plus
``MANIFEST.txt``. Neo4j is not expected: it is dumped only offline and on
request.

``check_backups`` reports with the shared check vocabulary:

- ``unreachable``: ``BACKUP_LOCATION`` does not exist (no backup was ever
  taken, or its disk is not mounted);
- ``failed``: no backup directory in it, the newest one is older than
  ``max_age_hours``, a store's file is missing or smaller than ``min_bytes``,
  the manifest is missing, or the manifest records a failed dump;
- ``ok``: the newest backup is fresh and holds every expected file.

``backup_report`` is the async callable the safety integration takes; a
``failed`` status there raises a backup alert. ``scripts/ops/backup_watchdog.py``
wraps ``check_backups`` for a timer.

Config keys read: ``BACKUP_LOCATION``, ``PROJECT_ROOT``, the five store enable
keys and ``QDRANT_COLLECTION``. Backups themselves hold private data
(conversation text, names, thoughts and beliefs); this module reports names,
ages and sizes only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mtc.core.checks import CheckStatus
from mtc.core.config import Settings, get_settings
from mtc.core.paths import resolve_path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.txt"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
BACKUP_DIR_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
DEFAULT_MAX_AGE_HOURS = 26.0


@dataclass
class BackupStatus:
    """What the newest backup looks like from the outside."""

    status: CheckStatus
    newest: Optional[str]
    age_hours: Optional[float]
    total_bytes: int
    missing: List[str] = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-ready mapping; ``errors`` lists what makes it fail."""
        data = asdict(self)
        data["status"] = self.status.value
        data["errors"] = [self.detail] if self.status is CheckStatus.FAILED else []
        return data


def expected_files(settings: Optional[Settings] = None) -> List[str]:
    """The files a complete backup holds for the enabled stores."""
    cfg = settings if settings is not None else get_settings()
    names = []
    if cfg.POSTGRES_ENABLED:
        names.append("postgres.dump")
    if cfg.MONGODB_ENABLED:
        names.append("mongo.archive.gz")
    if cfg.QDRANT_ENABLED:
        names.append(f"qdrant-{cfg.QDRANT_COLLECTION}.snapshot")
    if cfg.REDIS_ENABLED:
        names.append("redis.rdb")
    names.append(MANIFEST_NAME)
    return names


def backup_directories(location: Path) -> List[Path]:
    """Backup directories under ``location``, oldest first."""
    return sorted(
        path
        for path in location.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and BACKUP_DIR_PATTERN.match(path.name)
    )


def _failed_stores(manifest: Path) -> List[str]:
    """Stores the manifest records as failed (``status <store> failed``)."""
    failed = []
    try:
        lines = manifest.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return failed
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "status" and parts[2] == "failed":
            failed.append(parts[1])
    return failed


def check_backups(
    settings: Optional[Settings] = None,
    *,
    now: Optional[datetime] = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    min_bytes: int = 1,
) -> BackupStatus:
    """Judge the newest backup (see the module docstring)."""
    cfg = settings if settings is not None else get_settings()
    location = resolve_path(cfg.BACKUP_LOCATION, settings=cfg)
    if not location.is_dir():
        return BackupStatus(
            CheckStatus.UNREACHABLE,
            None,
            None,
            0,
            detail="BACKUP_LOCATION does not exist: no backup has been taken "
            "there, or its disk is not attached",
        )
    backups = backup_directories(location)
    if not backups:
        return BackupStatus(
            CheckStatus.FAILED,
            None,
            None,
            0,
            detail="no backup found in BACKUP_LOCATION",
        )
    newest = backups[-1]
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    taken = datetime.strptime(newest.name, TIMESTAMP_FORMAT).replace(
        tzinfo=timezone.utc
    )
    age_hours = round((moment - taken).total_seconds() / 3600.0, 2)
    total = sum(path.stat().st_size for path in newest.rglob("*") if path.is_file())
    missing = []
    for name in expected_files(cfg):
        path = newest / name
        if not path.is_file() or path.stat().st_size < int(min_bytes):
            missing.append(name)
    problems = []
    if missing:
        problems.append("missing or too small: " + ", ".join(missing))
    failed = _failed_stores(newest / MANIFEST_NAME)
    if failed:
        problems.append("the manifest records a failed dump: " + ", ".join(failed))
    if age_hours > float(max_age_hours):
        problems.append(
            f"the newest backup is {age_hours:g} hours old (limit {max_age_hours:g})"
        )
    if problems:
        return BackupStatus(
            CheckStatus.FAILED,
            newest.name,
            age_hours,
            total,
            missing=missing,
            detail="; ".join(problems),
        )
    return BackupStatus(
        CheckStatus.OK,
        newest.name,
        age_hours,
        total,
        detail=f"newest backup {newest.name}, {age_hours:g} hours old, {total} bytes",
    )


async def backup_report(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """The backup status as the mapping the safety integration expects.

    The ``status`` key holds the check word; ``failed`` makes the integration
    raise a backup alert with the ``errors`` it finds here.
    """
    report = check_backups(settings).as_dict()
    logger.debug("backup status: %s", report["status"])
    return report
