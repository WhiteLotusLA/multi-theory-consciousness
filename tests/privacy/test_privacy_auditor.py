"""Privacy auditor: bounded log, lazy JSONL, store writes, violations, reports."""

import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

import pytest

from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryPrivacyAuditStore
from mtc.privacy.data_classifier import DataClassification as DC
from mtc.privacy.privacy_auditor import (
    EventType,
    PrivacyAuditor,
    PrivacyEvent,
    ViolationType,
)

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, moment: datetime = T0) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, PROJECT_ROOT=str(tmp_path), DATA_DIRECTORY="data")


def _event(
    user_id: str = "Alice",
    success: bool = True,
    event_type: EventType = EventType.DATA_READ,
    timestamp=None,
    **kwargs,
) -> PrivacyEvent:
    return PrivacyEvent(
        event_type=event_type,
        user_role="guest",
        user_id=user_id,
        data_accessed="synthetic record",
        data_classification=kwargs.pop("classification", DC.PUBLIC),
        success=success,
        timestamp=timestamp,
        **kwargs,
    )


def _auditor(tmp_path, **kwargs) -> PrivacyAuditor:
    kwargs.setdefault("clock", Clock())
    return PrivacyAuditor(settings=_settings(tmp_path), **kwargs)


@pytest.mark.asyncio
async def test_events_append_to_bounded_memory(tmp_path):
    auditor = _auditor(tmp_path, max_events=5)
    for i in range(8):
        await auditor.log_event(_event(timestamp=T0 + timedelta(seconds=i)))
    assert auditor.event_count() == 5
    held = auditor.events_for_user("Alice")
    assert [e.timestamp for e in held] == [
        T0 + timedelta(seconds=i) for i in range(3, 8)
    ]
    with pytest.raises(ValueError):
        PrivacyAuditor(max_events=0, settings=_settings(tmp_path))


@pytest.mark.asyncio
async def test_jsonl_under_data_directory_created_lazily(tmp_path):
    auditor = _auditor(tmp_path)
    expected = tmp_path / "data" / "privacy" / "privacy_audit.jsonl"
    assert auditor.jsonl_path == expected
    assert not (tmp_path / "data").exists()
    event_id = await auditor.log_event(_event(user_id="Operator"))
    lines = expected.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_id"] == event_id
    assert record["user_id"] == "Operator"
    assert record["timestamp"] == T0.isoformat()
    assert record["data_classification"] == "public"

    explicit = tmp_path / "elsewhere" / "audit.jsonl"
    other = _auditor(tmp_path, jsonl_path=explicit)
    await other.log_event(_event())
    assert explicit.exists()

    silent = _auditor(tmp_path / "quiet", jsonl_enabled=False)
    await silent.log_event(_event())
    assert silent.jsonl_path is None
    assert not (tmp_path / "quiet").exists()


@pytest.mark.asyncio
async def test_store_receives_events(tmp_path):
    store = InMemoryPrivacyAuditStore()
    auditor = _auditor(tmp_path, store=store)
    event_id = await auditor.log_event(
        _event(user_id="Guest", success=False, event_type=EventType.ACCESS_DENIED)
    )
    [doc] = await store.events()
    assert doc["event_id"] == event_id
    assert doc["event_type"] == "access_denied"
    assert doc["user_id"] == "Guest"
    assert doc["success"] is False
    assert doc["timestamp"] == T0
    default = _auditor(tmp_path)
    assert isinstance(default.store, InMemoryPrivacyAuditStore)
    await default.log_event(_event())
    assert len(await default.store.events()) == 1


@pytest.mark.asyncio
async def test_violation_rules(tmp_path):
    clock = Clock(T0 + timedelta(hours=2))
    auditor = _auditor(tmp_path, clock=clock)
    await auditor.log_event(
        _event(user_id="Guest", success=False, event_type=EventType.ACCESS_DENIED)
    )
    first = auditor.detect_violations()
    assert [v.violation_type for v in first] == [ViolationType.UNAUTHORIZED_ACCESS]
    assert first[0].severity == "medium"
    assert auditor.detect_violations() == []  # reported once

    # 50 events in the last hour (spread out) is the limit, 51 is bulk.
    spread = _auditor(tmp_path, clock=clock)
    start = clock.moment - timedelta(minutes=55)
    for i in range(50):
        await spread.log_event(
            _event(user_id="Operator", timestamp=start + timedelta(minutes=i))
        )
    assert spread.detect_violations() == []
    await spread.log_event(_event(user_id="Operator", timestamp=clock.moment))
    bulk = spread.detect_violations()
    assert [v.violation_type for v in bulk] == [ViolationType.BULK_ACCESS]
    assert bulk[0].severity == "high" and bulk[0].user_id == "Operator"
    assert spread.detect_violations() == []  # still the same condition

    # 20 events in five minutes is the limit, 21 is a rapid-access anomaly.
    rapid = _auditor(tmp_path, clock=clock)
    for i in range(20):
        await rapid.log_event(
            _event(timestamp=clock.moment - timedelta(seconds=10 * i))
        )
    assert rapid.detect_violations() == []
    await rapid.log_event(_event(timestamp=clock.moment))
    anomaly = rapid.detect_violations()
    assert [v.violation_type for v in anomaly] == [ViolationType.PATTERN_ANOMALY]

    # Old events fall out of the windows.
    clock.moment += timedelta(hours=2)
    assert rapid.detect_violations() == []
    assert [v.violation_type for v in rapid.violations()] == [
        ViolationType.PATTERN_ANOMALY
    ]
    violation_id = anomaly[0].violation_id
    assert rapid.resolve_violation(violation_id, "synthetic load test") is True
    assert rapid.resolve_violation("missing", "x") is False


@pytest.mark.asyncio
async def test_health_score_range(tmp_path):
    clock = Clock()
    auditor = _auditor(tmp_path, clock=clock)
    assert auditor.health_score() == 100.0
    for _ in range(3):
        await auditor.log_event(_event(success=False))
    auditor.detect_violations()
    score = auditor.health_score()
    assert 0.0 <= score < 100.0
    for _ in range(40):
        await auditor.log_event(_event(success=False))
    auditor.detect_violations()
    assert auditor.health_score() == 0.0


@pytest.mark.asyncio
async def test_reports_have_compliance_blocks_and_neutral_summary(tmp_path):
    clock = Clock(T0 + timedelta(days=3))
    auditor = _auditor(tmp_path, clock=clock)
    day = T0.date()
    await auditor.log_event(_event(user_id="Alice", timestamp=T0))
    await auditor.log_event(
        _event(user_id="Guest", success=False, timestamp=T0 + timedelta(hours=1))
    )
    await auditor.log_event(
        _event(
            user_id="Operator",
            event_type=EventType.CONSENT_GRANTED,
            timestamp=T0 + timedelta(days=1),
            classification=DC.PRIVATE,
        )
    )
    auditor.detect_violations()

    daily = auditor.daily_report(day)
    assert daily["report_type"] == "daily"
    assert daily["total_events"] == 2
    assert daily["events_by_type"] == {"data_read": 2}
    assert daily["events_by_classification"] == {"public": 2}
    assert daily["violations_detected"] == 1
    assert 0.0 <= daily["health_score"] <= 100.0
    block = daily["compliance"]
    assert block["denied_access"] == 1
    assert block["consent_grants"] == 0
    assert "note" in block

    weekly = auditor.weekly_report(date(1999, 12, 30))
    assert weekly["report_type"] == "weekly"
    assert weekly["total_events"] == 3
    assert weekly["unique_users"] == 3
    assert weekly["compliance"]["consent_grants"] == 1
    summary = weekly["summary"]
    assert "3 times by 3 users" in summary
    assert not re.search(r"\b(parent|dad|child|family|her|his)\b", summary.lower())
    for text in (summary, json.dumps(daily), json.dumps(weekly, default=str)):
        assert text.isascii()

    empty = auditor.weekly_report(date(2001, 1, 1))
    assert empty["total_events"] == 0
    assert "no data access" in empty["summary"].lower()


@pytest.mark.asyncio
async def test_subject_consent_field(tmp_path):
    store = InMemoryPrivacyAuditStore()
    auditor = _auditor(tmp_path, store=store)
    event = _event(subject_consent=True)
    assert event.subject_consent is True
    await auditor.log_event(event)
    [doc] = await store.events()
    assert doc["subject_consent"] is True
    assert set(doc) >= {
        "event_id",
        "timestamp",
        "event_type",
        "user_role",
        "user_id",
        "data_accessed",
        "data_classification",
        "subject_consent",
        "success",
        "metadata",
    }
    assert _event().subject_consent is None


class _BrokenStore:
    async def add_event(self, event):
        raise RuntimeError("synthetic store outage")

    async def events(self, since=None, limit=100, user_id=None, event_type=None):
        return []


@pytest.mark.asyncio
async def test_store_failure_logged_not_raised(tmp_path, caplog):
    auditor = _auditor(tmp_path, store=_BrokenStore())
    with caplog.at_level(logging.WARNING):
        event_id = await auditor.log_event(_event())
        await auditor.log_event(_event())
    assert event_id
    assert auditor.event_count() == 2
    status = auditor.status()
    assert status["store_errors"] == 2
    assert "synthetic store outage" in status["last_error"]
    assert caplog.text.count("privacy audit store write failed") == 1
    assert (tmp_path / "data" / "privacy" / "privacy_audit.jsonl").exists()


@pytest.mark.asyncio
async def test_naive_event_timestamp_rejected(tmp_path):
    auditor = _auditor(tmp_path)
    with pytest.raises(ValueError):
        await auditor.log_event(_event(timestamp=datetime(2000, 1, 1)))


def test_status_reports_no_absolute_path(tmp_path):
    inside = _auditor(tmp_path)
    status = inside.status()
    assert status["jsonl_path"] == "data/privacy/privacy_audit.jsonl"
    assert not os.path.isabs(status["jsonl_path"])
    # the property keeps the real path for the writer
    assert inside.jsonl_path == tmp_path / "data" / "privacy" / "privacy_audit.jsonl"
    assert inside.jsonl_path.is_absolute()
    outside = _auditor(
        tmp_path / "root", jsonl_path=tmp_path / "elsewhere" / "trail.jsonl"
    )
    assert outside.status()["jsonl_path"] == "trail.jsonl"
    assert _auditor(tmp_path, jsonl_enabled=False).status()["jsonl_path"] is None
