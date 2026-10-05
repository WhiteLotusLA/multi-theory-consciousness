"""Alert manager: ring, suppression, notifiers, failure visibility and the store."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryAlertStore
from mtc.monitoring import alert_manager as am
from mtc.monitoring.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
    ConsoleNotifier,
    FileNotifier,
    Notifier,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingNotifier:
    def __init__(self, min_severity=AlertSeverity.INFO, result=True, raises=None):
        self.min_severity = min_severity
        self.result = result
        self.raises = raises
        self.calls = []

    async def send(self, body, channel, priority, title, activity_type) -> bool:
        self.calls.append(
            {
                "body": body,
                "channel": channel,
                "priority": priority,
                "title": title,
                "activity_type": activity_type,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.result


class FailingStore:
    def __init__(self) -> None:
        self.calls = 0

    async def add_alert(self, alert) -> None:
        self.calls += 1
        raise RuntimeError("store down")

    async def acknowledge(self, alert_id) -> bool:
        raise RuntimeError("store down")

    async def recent_alerts(self, limit=100, alert_type=None, min_severity=None):
        raise RuntimeError("store down")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _manager(tmp_path, **kwargs):
    kwargs.setdefault("failure_log", tmp_path / "failed.log")
    kwargs.setdefault("clock", Clock())
    return AlertManager(**kwargs)


async def _send(manager, severity=AlertSeverity.WARNING, kind=AlertType.SYSTEM):
    return await manager.send_alert(kind, severity, "Title", "Message", {"k": 1})


def test_severity_orders_by_rank():
    assert AlertSeverity.INFO < AlertSeverity.WARNING < AlertSeverity.URGENT
    assert AlertSeverity.CRITICAL >= AlertSeverity.URGENT
    assert not AlertSeverity.INFO >= AlertSeverity.WARNING
    assert [s.value for s in sorted(AlertSeverity, reverse=True)][0] == "critical"


@pytest.mark.asyncio
async def test_ring_buffer_bounded(tmp_path):
    manager = _manager(tmp_path, notifiers=[], ring_size=3)
    for index in range(5):
        await manager.send_alert(AlertType.SYSTEM, AlertSeverity.INFO, f"t{index}", "m")
    titles = [a.title for a in manager.get_alerts_by_type(AlertType.SYSTEM)]
    assert titles == ["t2", "t3", "t4"]
    assert manager.get_summary()["total_stored"] == 3


@pytest.mark.asyncio
async def test_suppression_per_type(tmp_path):
    clock = Clock()
    notifier = RecordingNotifier()
    manager = _manager(tmp_path, notifiers=[notifier], clock=clock)
    manager.suppress(AlertType.PRIVACY, seconds=60)
    assert await _send(manager, kind=AlertType.PRIVACY) is None
    assert await _send(manager, kind=AlertType.SYSTEM) is not None
    assert len(notifier.calls) == 1
    clock.now = T0 + timedelta(seconds=61)
    assert await _send(manager, kind=AlertType.PRIVACY) is not None
    assert manager.get_summary()["suppressed"] == 1


@pytest.mark.asyncio
async def test_acknowledge(tmp_path):
    store = InMemoryAlertStore()
    manager = _manager(tmp_path, notifiers=[], store=store)
    alert = await _send(manager)
    assert [a.alert_id for a in manager.get_unacknowledged()] == [alert.alert_id]
    assert await manager.acknowledge(alert.alert_id, "Operator") is True
    assert manager.get_unacknowledged() == []
    assert alert.acknowledged_by == "Operator"
    assert alert.acknowledged_at == T0
    stored = await store.recent_alerts()
    assert stored[0]["acknowledged"] is True
    assert await manager.acknowledge("missing") is False


@pytest.mark.asyncio
async def test_notifiers_receive_at_or_above_their_severity(tmp_path):
    loud = RecordingNotifier(min_severity=AlertSeverity.URGENT)
    quiet = RecordingNotifier(min_severity=AlertSeverity.INFO)
    manager = _manager(tmp_path, notifiers=[loud, quiet])
    await _send(manager, AlertSeverity.INFO)
    await _send(manager, AlertSeverity.WARNING)
    await _send(manager, AlertSeverity.URGENT)
    await _send(manager, AlertSeverity.CRITICAL)
    assert len(quiet.calls) == 4
    assert [c["priority"] for c in loud.calls] == ["normal", "urgent"]
    assert all(c["channel"] == "auto" for c in quiet.calls)
    assert all(c["activity_type"] == "alert" for c in quiet.calls)


@pytest.mark.asyncio
async def test_notifier_failure_counted_and_logged_locally(tmp_path, caplog):
    broken = RecordingNotifier(raises=RuntimeError("boom"))
    refusing = RecordingNotifier(result=False)
    healthy = RecordingNotifier()
    manager = _manager(tmp_path, notifiers=[broken, refusing, healthy])
    with caplog.at_level("ERROR"):
        alert = await _send(manager)
    assert alert is not None
    assert manager.notify_failures == 2
    assert len(healthy.calls) == 1
    assert manager.status()["available"] is False
    assert "undelivered" in manager.status()["last_error"]
    lines = (tmp_path / "failed.log").read_text().splitlines()
    assert len(lines) == 2
    assert "Title: Message" in lines[0] and "boom" in lines[0]
    assert any("delivery failed" in r.getMessage() for r in caplog.records)
    await _send(manager)  # a second alert fails the same two notifiers
    assert manager.notify_failures == 4


@pytest.mark.asyncio
async def test_status_recovers_after_a_clean_alert(tmp_path):
    flaky = RecordingNotifier(raises=RuntimeError("boom"))
    manager = _manager(tmp_path, notifiers=[flaky])
    await _send(manager)
    assert manager.status()["available"] is False
    flaky.raises = None
    await _send(manager)
    assert manager.status()["available"] is True


@pytest.mark.asyncio
async def test_notifier_timeout_is_a_failure(tmp_path, monkeypatch):
    import asyncio

    class Slow:
        async def send(self, body, channel, priority, title, activity_type):
            await asyncio.sleep(5)
            return True

    monkeypatch.setattr(am, "NOTIFIER_TIMEOUT_S", 0.05)
    manager = _manager(tmp_path, notifiers=[Slow()])
    assert await _send(manager) is not None
    assert manager.notify_failures == 1


def test_failure_log_under_log_directory(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env))
    monkeypatch.setenv("MTC_LOG_DIRECTORY", str(clean_env / "logs"))
    settings = Settings()
    manager = AlertManager(notifiers=[], settings=settings)
    assert manager.failure_log_path == clean_env / "logs" / "alerts_failed.log"
    assert ".." not in str(manager.failure_log_path)
    assert "Library" not in str(manager.failure_log_path)


@pytest.mark.asyncio
async def test_failure_log_is_written_there(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env))
    monkeypatch.setenv("MTC_LOG_DIRECTORY", str(clean_env / "logs"))
    settings = Settings()
    manager = AlertManager(
        notifiers=[RecordingNotifier(raises=RuntimeError("x"))], settings=settings
    )
    await _send(manager)
    assert (clean_env / "logs" / "alerts_failed.log").exists()


def test_from_settings_builds_configured_notifiers(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env))
    monkeypatch.setenv("MTC_ALERT_NOTIFIERS", "console,file")
    manager = AlertManager.from_settings(Settings())
    kinds = [type(n) for n in manager.notifiers]
    assert kinds == [ConsoleNotifier, FileNotifier]
    monkeypatch.setenv("MTC_ALERT_NOTIFIERS", "file")
    assert [type(n) for n in AlertManager.from_settings(Settings()).notifiers] == [
        FileNotifier
    ]
    assert all(isinstance(n, Notifier) for n in manager.notifiers)


def test_unknown_notifier_name_is_rejected():
    with pytest.raises(ValueError, match="console, file"):
        am.build_notifiers(["console", "pager"])


@pytest.mark.asyncio
async def test_titles_name_system_from_settings(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_SYSTEM_NAME", "Atlas")
    settings = Settings()
    notifier = RecordingNotifier()
    manager = AlertManager(
        notifiers=[notifier],
        failure_log=clean_env / "f.log",
        settings=settings,
    )
    await manager.send_alert(
        AlertType.SYSTEM, AlertSeverity.WARNING, "Disk", "Nearly full"
    )
    call = notifier.calls[0]
    assert call["title"] == "Atlas alert: Disk"
    assert call["body"] == "[WARNING] system: Nearly full"


@pytest.mark.asyncio
async def test_store_failure_does_not_block_delivery(tmp_path):
    notifier = RecordingNotifier()
    store = FailingStore()
    manager = _manager(tmp_path, notifiers=[notifier], store=store)
    alert = await _send(manager)
    assert alert is not None
    assert store.calls == 1
    assert len(notifier.calls) == 1
    assert manager.persist_failures == 1
    assert "store down" in (tmp_path / "failed.log").read_text()
    assert manager.get_alerts_by_type(AlertType.SYSTEM) == [alert]


@pytest.mark.asyncio
async def test_store_receives_aware_records(tmp_path):
    store = InMemoryAlertStore()
    manager = _manager(tmp_path, notifiers=[], store=store)
    await _send(manager)
    record = (await store.recent_alerts())[0]
    assert record["timestamp"] == T0
    assert record["alert_type"] == "system"
    assert record["severity"] == "warning"
    assert manager.status()["backend"] == "store"


@pytest.mark.asyncio
async def test_file_notifier_appends_json_lines(tmp_path):
    target = tmp_path / "out" / "alerts.jsonl"
    notifier = FileNotifier(target, clock=Clock())
    assert await notifier.send("body", "auto", "low", "title", "alert") is True
    assert await notifier.send("again", "auto", "low", "title", "alert") is True
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert [r["body"] for r in rows] == ["body", "again"]
    assert rows[0]["timestamp"] == T0.isoformat()


def test_file_notifier_default_path_is_under_log_directory(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env))
    notifier = FileNotifier(settings=Settings())
    assert notifier.path == clean_env / "logs" / "alerts.jsonl"
    assert not (clean_env / "logs").exists()


@pytest.mark.asyncio
async def test_console_notifier_logs_by_priority(caplog):
    notifier = ConsoleNotifier()
    assert notifier.min_severity is AlertSeverity.WARNING
    with caplog.at_level("INFO"):
        await notifier.send("body", "auto", "urgent", "title", "alert")
        await notifier.send("body", "auto", "low", "title", "alert")
    levels = [r.levelname for r in caplog.records if r.name == am.__name__]
    assert levels == ["ERROR", "INFO"]


def test_no_external_channel_exists():
    classes = {
        name
        for name, value in vars(am).items()
        if isinstance(value, type) and hasattr(value, "send")
    }
    assert classes == {"Notifier", "ConsoleNotifier", "FileNotifier"}
    source = open(am.__file__, encoding="utf-8").read()
    channels = ("carrierpigeon", "carrierpigeon", "smtp", "http://", "https://")
    for word in channels + ("carrierpigeon",):
        assert word not in source.lower()


@pytest.mark.asyncio
async def test_default_manager_has_a_console_notifier(tmp_path):
    manager = _manager(tmp_path)
    assert [type(n) for n in manager.notifiers] == [ConsoleNotifier]
    assert _manager(tmp_path, notifiers=[]).notifiers == []


@pytest.mark.asyncio
async def test_summary_and_history(tmp_path):
    clock = Clock()
    manager = _manager(tmp_path, notifiers=[], clock=clock)
    await _send(manager, AlertSeverity.WARNING, AlertType.PRIVACY)
    clock.now = T0 + timedelta(hours=30)
    await _send(manager, AlertSeverity.CRITICAL, AlertType.SAFETY)
    summary = manager.get_summary()
    assert summary["total_24h"] == 1
    assert summary["by_type"] == {"safety": 1}
    assert summary["unacknowledged"] == 2
    assert [a.severity for a in manager.get_alert_history(hours=48)] == [
        AlertSeverity.CRITICAL,
        AlertSeverity.WARNING,
    ]
    only_critical = manager.get_unacknowledged(AlertSeverity.URGENT)
    assert [a.alert_type for a in only_critical] == [AlertType.SAFETY]


def test_alert_to_dict_is_json_safe():
    alert = Alert(
        AlertType.SAFETY,
        AlertSeverity.URGENT,
        "t",
        "m",
        context={"when": T0},
        timestamp=T0,
    )
    data = alert.to_dict()
    json.dumps(data)
    assert data["timestamp"] == T0.isoformat()
    assert data["acknowledged_at"] is None


def test_singleton_builds_from_settings(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(clean_env))
    am.reset_alert_manager()
    try:
        first = am.get_alert_manager(Settings())
        assert am.get_alert_manager() is first
    finally:
        am.reset_alert_manager()


# MongoAlertStore over the fake driver (the db contract entry checks a server).


def _mongo_store(clock=None):
    from mtc.monitoring.stores import MongoAlertStore
    from tests.memory.test_mongo_stores import FakeMongoClient

    client = FakeMongoClient()
    store = MongoAlertStore(database=client["fake_db"], clock=clock)
    return store, client


def _doc(alert_id, offset=0, alert_type="system", severity="info"):
    return {
        "alert_id": alert_id,
        "timestamp": T0 + timedelta(minutes=offset),
        "alert_type": alert_type,
        "severity": severity,
        "title": "t",
        "message": "m",
        "context": {"k": [1, 2]},
        "acknowledged": False,
    }


def test_alert_indexes_match_artifact():
    from mtc.monitoring.stores import ALERT_INDEXES

    store, client = _mongo_store()
    assert store.ensure_indexes() == 4
    built = [
        (tuple(i["keys"]), i["unique"]) for i in client["fake_db"]["alerts"].indexes
    ]
    assert built == [
        ((("alert_id", 1),), True),
        ((("timestamp", -1),), False),
        ((("alert_type", 1),), False),
        ((("severity", 1),), False),
    ]
    assert list(ALERT_INDEXES) == ["alerts"]
    assert store.ensure_indexes() == 4
    assert len(client["fake_db"]["alerts"].indexes) == 4


@pytest.mark.asyncio
async def test_mongo_alert_store_reads_filters_and_copies():
    store, _ = _mongo_store(clock=Clock(T0 + timedelta(days=1)))
    await store.add_alert(_doc("a", 0, "system", "info"))
    await store.add_alert(_doc("b", 1, "privacy", "warning"))
    await store.add_alert(_doc("c", 2, "system", "critical"))
    await store.add_alert(_doc("d", 2, "system", "warning"))
    rows = await store.recent_alerts()
    assert [r["alert_id"] for r in rows] == ["d", "c", "b", "a"]  # ties: later first
    assert all("_id" not in r for r in rows)
    assert [r["alert_id"] for r in await store.recent_alerts(alert_type="privacy")] == [
        "b"
    ]
    assert [
        r["alert_id"] for r in await store.recent_alerts(min_severity="warning")
    ] == ["d", "c", "b"]
    assert await store.recent_alerts(limit=0) == []
    with pytest.raises(ValueError):
        await store.recent_alerts(min_severity="loud")
    rows[0]["context"]["k"].append(9)
    assert (await store.recent_alerts(limit=1))[0]["context"] == {"k": [1, 2]}


@pytest.mark.asyncio
async def test_mongo_alert_store_validation_and_acknowledge():
    store, _ = _mongo_store(clock=Clock(T0 + timedelta(days=1)))
    with pytest.raises(ValueError):
        await store.add_alert({"alert_id": "a", "alert_type": "system"})
    with pytest.raises(ValueError):
        await store.add_alert(_doc("a", severity="loud"))
    with pytest.raises(ValueError):
        await store.add_alert({**_doc("a"), "timestamp": datetime(2000, 1, 1)})
    document = _doc("a")
    store.ensure_indexes()
    await store.add_alert(document)
    assert "_id" not in document  # the caller's dict is not mutated
    with pytest.raises(ValueError, match="already stored"):
        await store.add_alert(_doc("a"))
    del document["timestamp"]
    document["alert_id"] = "stamped"
    await store.add_alert(document)
    stamped = [r for r in await store.recent_alerts() if r["alert_id"] == "stamped"]
    assert stamped[0]["timestamp"] == T0 + timedelta(days=1)
    assert await store.acknowledge("a") is True
    assert await store.acknowledge("missing") is False
    by_id = {r["alert_id"]: r for r in await store.recent_alerts()}
    assert by_id["a"]["acknowledged"] is True
    assert by_id["a"]["acknowledged_at"] == T0 + timedelta(days=1)
    assert by_id["stamped"]["acknowledged"] is False


@pytest.mark.asyncio
async def test_mongo_alert_store_unreachable_is_store_unavailable():
    from mtc.memory.protocols import StoreUnavailable

    store, client = _mongo_store()
    assert store.status() == {
        "backend": "mongodb",
        "available": True,
        "last_error": None,
    }
    client.unreachable = True
    with pytest.raises(StoreUnavailable):
        await store.recent_alerts()
    assert store.status()["available"] is False
    client.unreachable = False
    assert await store.recent_alerts() == []
    assert store.status()["available"] is True


@pytest.mark.asyncio
async def test_manager_over_the_mongo_store(tmp_path):
    store, _ = _mongo_store()
    manager = _manager(tmp_path, notifiers=[], store=store)
    alert = await _send(manager, AlertSeverity.URGENT, AlertType.PRIVACY)
    [row] = await store.recent_alerts(alert_type="privacy", min_severity="urgent")
    assert row["alert_id"] == alert.alert_id and row["timestamp"] == T0
    assert row["context"] == {"k": 1}
    assert await manager.acknowledge(alert.alert_id, "Operator") is True
    assert (await store.recent_alerts())[0]["acknowledged"] is True


def test_mongo_alert_store_connects_nowhere_on_construction():
    store, client = _mongo_store()
    assert client.calls == []
