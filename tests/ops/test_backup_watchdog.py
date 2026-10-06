"""The backup watchdog: fresh and complete, or it says what is wrong."""

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from mtc.core import config as config_module
from mtc.core.checks import CheckStatus
from mtc.core.config import Settings
from mtc.monitoring import alert_manager
from mtc.monitoring.alert_manager import AlertManager, AlertType
from mtc.monitoring.backup_status import (
    BackupStatus,
    check_backups,
    expected_files,
)

from .conftest import OPS_DIR

NOW = datetime(2000, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
FORMAT = "%Y%m%dT%H%M%SZ"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "backup_watchdog_script", OPS_DIR / "backup_watchdog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watchdog = _load_script()


@pytest.fixture
def project(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_settings", None)
    yield tmp_path
    config_module._settings = None


def settings_for(project: Path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        PROJECT_ROOT=str(project),
        DB_PREFIX="acme",
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        QDRANT_ENABLED=True,
        REDIS_ENABLED=True,
    )
    values.update(overrides)
    return Settings(**values)


def write_backup(
    project: Path,
    taken: datetime,
    *,
    files=None,
    manifest: str = "",
    location="backups",
) -> Path:
    directory = project / location / taken.strftime(FORMAT)
    directory.mkdir(parents=True)
    names = (
        files
        if files is not None
        else (
            "postgres.dump",
            "mongo.archive.gz",
            "qdrant-acme_memories.snapshot",
            "redis.rdb",
        )
    )
    for name in names:
        (directory / name).write_bytes(b"x" * 10)
    (directory / "MANIFEST.txt").write_text(
        "mtc-backup-manifest 1\n" + manifest, encoding="utf-8"
    )
    return directory


def test_fresh_backup_ok(project):
    cfg = settings_for(project)
    write_backup(project, NOW - timedelta(days=3))
    newest = write_backup(project, NOW - timedelta(hours=5))
    status = check_backups(cfg, now=NOW)
    assert isinstance(status, BackupStatus)
    assert status.status is CheckStatus.OK
    assert status.newest == newest.name
    assert status.age_hours == pytest.approx(5.0)
    assert status.missing == []
    assert status.total_bytes == 4 * 10 + len("mtc-backup-manifest 1\n")
    assert "5 hours old" in status.detail
    assert expected_files(cfg) == [
        "postgres.dump",
        "mongo.archive.gz",
        "qdrant-acme_memories.snapshot",
        "redis.rdb",
        "MANIFEST.txt",
    ]
    # Only the enabled stores are expected; Neo4j never is.
    small = settings_for(
        project, MONGODB_ENABLED=False, QDRANT_ENABLED=False, NEO4J_ENABLED=True
    )
    assert expected_files(small) == ["postgres.dump", "redis.rdb", "MANIFEST.txt"]
    report = status.as_dict()
    assert report["status"] == "ok" and report["errors"] == []


@pytest.mark.parametrize("zone", ["UTC", "Europe/Berlin", "Asia/Tokyo"])
def test_stale_backup_fails(project, zone):
    cfg = settings_for(project, TIMEZONE=zone)
    write_backup(project, NOW - timedelta(hours=27))
    stale = check_backups(cfg, now=NOW)
    assert stale.status is CheckStatus.FAILED
    assert stale.age_hours == pytest.approx(27.0)
    assert "27 hours old (limit 26)" in stale.detail
    assert stale.as_dict()["errors"] == [stale.detail]
    # The limit is the caller's: the same backup passes a longer one, and a
    # backup exactly at the limit is still fresh. Ages are UTC, whatever the
    # display zone.
    assert check_backups(cfg, now=NOW, max_age_hours=28).status is CheckStatus.OK
    edge = project / "edge"
    write_backup(edge, NOW - timedelta(hours=26))
    at_limit = check_backups(settings_for(edge, TIMEZONE=zone), now=NOW)
    assert at_limit.status is CheckStatus.OK
    # A naive "now" is read as UTC.
    naive = check_backups(cfg, now=NOW.replace(tzinfo=None))
    assert naive.age_hours == pytest.approx(27.0)


def test_missing_store_file_fails(project):
    cfg = settings_for(project)
    backup = write_backup(project, NOW - timedelta(hours=1))
    (backup / "mongo.archive.gz").unlink()
    (backup / "redis.rdb").write_bytes(b"")
    status = check_backups(cfg, now=NOW)
    assert status.status is CheckStatus.FAILED
    assert status.missing == ["mongo.archive.gz", "redis.rdb"]
    assert "missing or too small: mongo.archive.gz, redis.rdb" in status.detail
    # A size floor makes a truncated dump count as missing.
    assert check_backups(cfg, now=NOW, min_bytes=11).missing == [
        "postgres.dump",
        "mongo.archive.gz",
        "qdrant-acme_memories.snapshot",
        "redis.rdb",
    ]
    # A manifest that records a failed dump fails the check on its own.
    other = project / "other"
    write_backup(other, NOW - timedelta(hours=1), manifest="status postgres failed\n")
    failed = check_backups(settings_for(other), now=NOW)
    assert failed.status is CheckStatus.FAILED
    assert "records a failed dump: postgres" in failed.detail
    # No manifest at all is incomplete.
    bare = project / "bare"
    directory = write_backup(bare, NOW - timedelta(hours=1))
    (directory / "MANIFEST.txt").unlink()
    assert check_backups(settings_for(bare), now=NOW).missing == ["MANIFEST.txt"]


def test_absent_location_is_unreachable(project):
    cfg = settings_for(project)
    status = check_backups(cfg, now=NOW)
    assert status.status is CheckStatus.UNREACHABLE
    assert status.newest is None and status.age_hours is None
    assert "BACKUP_LOCATION does not exist" in status.detail
    # A location that exists and holds no backup is a failure, not a pass.
    (project / "backups").mkdir()
    (project / "backups" / "notes").mkdir()
    (project / "backups" / "20000101T000000Z.txt").write_text("a file\n")
    empty = check_backups(cfg, now=NOW)
    assert empty.status is CheckStatus.FAILED
    assert empty.detail == "no backup found in BACKUP_LOCATION"
    # An absolute location elsewhere is read where it is.
    away = project.parent / f"{project.name}-vault"
    write_backup(away, NOW - timedelta(hours=1), location=".")
    moved = check_backups(settings_for(project, BACKUP_LOCATION=str(away)), now=NOW)
    assert moved.status is CheckStatus.OK


def _fresh(project: Path) -> Path:
    return write_backup(project, datetime.now(timezone.utc) - timedelta(minutes=5))


def _env(monkeypatch) -> None:
    for key, value in (
        ("MTC_DB_PREFIX", "acme"),
        ("MTC_POSTGRES_ENABLED", "true"),
        ("MTC_MONGODB_ENABLED", "true"),
        ("MTC_QDRANT_ENABLED", "true"),
        ("MTC_REDIS_ENABLED", "true"),
    ):
        monkeypatch.setenv(key, value)


def test_alert_through_alert_manager(project, monkeypatch, capsys):
    _env(monkeypatch)
    delivered: List[dict] = []

    class FakeNotifier:
        name = "fake"

        async def send(self, body, channel, priority, title, activity_type):
            delivered.append(
                {
                    "body": body,
                    "priority": priority,
                    "title": title,
                    "type": activity_type,
                }
            )
            return True

    built: List[AlertManager] = []

    def from_settings(settings=None, store=None):
        manager = AlertManager(notifiers=[FakeNotifier()], settings=settings)
        built.append(manager)
        return manager

    monkeypatch.setattr(alert_manager.AlertManager, "from_settings", from_settings)

    # Nothing to alert about: a fresh, complete backup raises nothing.
    _fresh(project)
    assert watchdog.main(["--alert"]) == 0
    assert delivered == [] and built == []

    # Stale: one backup alert goes out, and the exit code still says failed.
    assert watchdog.main(["--alert", "--max-age-hours", "0.01"]) == 1
    assert len(delivered) == 1
    assert delivered[0]["title"].endswith("Backup check did not pass")
    assert "hours old" in delivered[0]["body"]
    (alert,) = built[0].get_alerts_by_type(AlertType.BACKUP)
    assert alert.title == "Backup check did not pass"
    assert alert.context["status"] == "failed"
    # Without --alert the same failure is only reported.
    assert watchdog.main(["--max-age-hours", "0.01"]) == 1
    assert len(delivered) == 1
    capsys.readouterr()


def test_json_output(project, monkeypatch, capsys):
    _env(monkeypatch)
    # No location yet: unreachable, exit 3.
    assert watchdog.main(["--json"]) == 3
    absent = json.loads(capsys.readouterr().out)
    assert absent["status"] == "unreachable" and absent["exit_code"] == 3
    assert absent["newest"] is None

    newest = _fresh(project)
    assert watchdog.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert sorted(report) == [
        "age_hours",
        "detail",
        "errors",
        "exit_code",
        "missing",
        "newest",
        "status",
        "total_bytes",
    ]
    assert report["status"] == "ok" and report["exit_code"] == 0
    assert report["newest"] == newest.name
    assert report["missing"] == [] and report["errors"] == []
    assert 0 <= report["age_hours"] < 1

    (newest / "postgres.dump").unlink()
    assert watchdog.main(["--json"]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "failed" and failed["missing"] == ["postgres.dump"]
    assert failed["errors"] == [failed["detail"]]
    # The plain form is one line, and --min-bytes is passed through.
    (newest / "postgres.dump").write_bytes(b"x")
    assert watchdog.main([]) == 0
    assert capsys.readouterr().out.startswith("backup: ok: newest backup ")
    assert watchdog.main(["--min-bytes", "5"]) == 1
    assert "postgres.dump" in capsys.readouterr().out
    assert watchdog.main(["--max-age-hours", "0"]) == 2
    monkeypatch.setenv("MTC_WAKE_HOUR", "99")
    assert watchdog.main([]) == 2
    capsys.readouterr()
