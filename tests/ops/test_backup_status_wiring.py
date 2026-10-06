"""The runtime hands the backup status to the safety integration."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.monitoring.alert_manager import AlertType
from mtc.monitoring.backup_status import backup_report
from mtc.pipeline.runtime import build_runtime

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "%Y%m%dT%H%M%SZ"


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_settings", None)
    yield tmp_path
    config_module._settings = None
    reset_logging()


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        PROJECT_ROOT=str(ROOT),
        DATA_DIRECTORY=str(tmp_path / "data"),
        STATE_DIR=str(tmp_path / "state"),
        LOG_DIRECTORY=str(tmp_path / "logs"),
        RESULTS_DIR=str(tmp_path / "results"),
        AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
        SPEAKERS_PATH=str(tmp_path / "speakers.yaml"),
        BACKUP_LOCATION=str(tmp_path / "backups"),
        LLM_PROVIDER="mock",
        CTM_PROFILE="demo",
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        DEVICE="cpu",
        SLEEP_SCHEDULER_ENABLED=False,
        ALERT_NOTIFIERS="",
    )
    values.update(overrides)
    return Settings(**values)


def _write_backup(location: Path, age: timedelta) -> Path:
    taken = datetime.now(timezone.utc) - age
    directory = location / taken.strftime(FORMAT)
    directory.mkdir(parents=True)
    (directory / "MANIFEST.txt").write_text("mtc-backup-manifest 1\n", encoding="utf-8")
    return directory


@pytest.mark.asyncio
async def test_runtime_passes_backup_report_to_safety_integration(clean):
    settings = _settings(clean)
    runtime = await build_runtime(
        settings,
        embedder=HashEmbedder(dim=128),
        allow_non_semantic=True,
        substrates=False,
    )
    try:
        safety = runtime.safety
        assert safety.status()["backup"] == {"configured": True}

        # No backup location yet: the integration reports what the backup
        # module measured, for the runtime's own settings.
        absent = await safety.backup_report()
        assert absent["status"] == "unreachable"
        assert absent["result"] == await backup_report(settings)
        assert absent["result"]["newest"] is None

        # A fresh backup (no store is enabled, so the manifest is all there is).
        fresh = _write_backup(clean / "backups", timedelta(minutes=10))
        healthy = await safety.backup_report()
        assert healthy["status"] == "ok"
        assert healthy["result"]["newest"] == fresh.name

        # A stale one fails, and the failure raises a backup alert.
        fresh.rename(clean / "backups" / "20000101T000000Z")
        stale = await safety.backup_report()
        assert stale["status"] == "failed"
        assert "hours old" in stale["result"]["detail"]
        alerts = safety._alerts.get_alerts_by_type(AlertType.BACKUP)
        assert len(alerts) == 1
        assert "hours old" in alerts[0].message
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_report_without_backups_says_none_found(clean):
    settings = _settings(clean)
    (clean / "backups").mkdir()
    report = await backup_report(settings)
    assert report["status"] == "failed"
    assert report["detail"] == "no backup found in BACKUP_LOCATION"
    assert report["errors"] == ["no backup found in BACKUP_LOCATION"]
    assert report["newest"] is None and report["age_hours"] is None
    assert report["total_bytes"] == 0 and report["missing"] == []
    # The report follows the settings it is given, not the process settings.
    elsewhere = _settings(clean, BACKUP_LOCATION=str(clean / "nowhere"))
    assert (await backup_report(elsewhere))["status"] == "unreachable"
    _write_backup(clean / "backups", timedelta(hours=1))
    assert (await backup_report(settings))["status"] == "ok"
    assert (await backup_report(elsewhere))["status"] == "unreachable"
