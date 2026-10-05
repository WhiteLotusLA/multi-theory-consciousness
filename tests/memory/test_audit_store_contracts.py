"""Audit store contract tests: safety audits, alerts and privacy events.

Every case runs against each entry of ``BACKENDS``, in the shape of
``tests/memory/test_store_contracts.py``: the in-memory twins are the default,
and a real store joins by appending one ``pytest.param`` entry (carrying
``pytest.mark.db``) whose factory takes its throwaway target first and builds
the kinds it implements, falling back to the in-memory twin for the others. A
store-backed entry names its fixture and an async ``prepare`` hook that
creates the schema there.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

import pytest
import pytest_asyncio

from mtc.memory import inmemory, protocols
from mtc.monitoring.stores import MongoAlertStore
from mtc.privacy.stores import MongoPrivacyAuditStore
from mtc.safety.stores import PostgresSafetyAuditStore
from tests.memory.test_mongo_stores import FakeMongoClient

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

KINDS = ("safety", "alert", "privacy")

PROTOCOLS = {
    "safety": protocols.SafetyAuditStore,
    "alert": protocols.AlertStore,
    "privacy": protocols.PrivacyAuditStore,
}


@dataclass(frozen=True)
class Backend:
    """A named factory building a store of each kind."""

    name: str
    factory: Callable[..., Any]
    fixture: Optional[str] = None
    prepare: Optional[Callable[[Any], Awaitable[None]]] = None


def _fixed_clock() -> datetime:
    return T0 + timedelta(days=1)


def _inmemory_factory(kind: str) -> Any:
    classes = {
        "safety": inmemory.InMemorySafetyAuditStore,
        "alert": inmemory.InMemoryAlertStore,
        "privacy": inmemory.InMemoryPrivacyAuditStore,
    }
    return classes[kind](clock=_fixed_clock)


# MongoDB classes by kind; each needs ``(database=...)`` and ``ensure_indexes()``.
_MONGO_CLASSES: Dict[str, Any] = {}
_MONGO_CLASSES["alert"] = MongoAlertStore
_MONGO_CLASSES["privacy"] = MongoPrivacyAuditStore


def _mongo_factory(database: Any, kind: str) -> Any:
    if kind in _MONGO_CLASSES:
        return _MONGO_CLASSES[kind](database=database)
    return _inmemory_factory(kind)


async def _mongo_prepare(database: Any) -> None:
    for cls in _MONGO_CLASSES.values():
        await asyncio.to_thread(cls(database=database).ensure_indexes)


@pytest.fixture
def fake_mongo() -> Any:
    """A fake pymongo database: checks the store logic without a server."""
    return FakeMongoClient()["audit_contract"]


# Append a pytest.param(Backend(...), marks=pytest.mark.db, id="...") entry per
# real store; the cases below do not change.
def _postgres_factory(pool: Any, kind: str) -> Any:
    if kind == "safety":
        return PostgresSafetyAuditStore(pool)
    return _inmemory_factory(kind)


async def _postgres_prepare(pool: Any) -> None:
    await PostgresSafetyAuditStore(pool).ensure_schema()


BACKENDS = [
    pytest.param(Backend("inmemory", _inmemory_factory), id="inmemory"),
    pytest.param(
        Backend(
            "postgres",
            _postgres_factory,
            fixture="throwaway_postgres",
            prepare=_postgres_prepare,
        ),
        marks=pytest.mark.db,
        id="postgres",
    ),
    # The mongo-fake entry runs the MongoDB classes over the fake driver of
    # tests/memory/test_mongo_stores.py (a model of the driver, not of a
    # server); the db entry is what checks real MongoDB.
    pytest.param(
        Backend(
            "mongo-fake", _mongo_factory, fixture="fake_mongo", prepare=_mongo_prepare
        ),
        id="mongo-fake",
    ),
    pytest.param(
        Backend(
            "mongo",
            _mongo_factory,
            fixture="throwaway_mongo",
            prepare=_mongo_prepare,
        ),
        marks=pytest.mark.db,
        id="mongo",
    ),
]


@pytest.fixture(params=BACKENDS)
def backend(request) -> Backend:
    return request.param


@pytest.fixture
def store_target(request, backend) -> Any:
    if backend.fixture is None:
        return None
    return request.getfixturevalue(backend.fixture)


@pytest_asyncio.fixture
async def prepared_target(backend, store_target) -> Any:
    if backend.prepare is not None:
        await backend.prepare(store_target)
    return store_target


@pytest.fixture
def make_store(backend, prepared_target):
    def build(kind: str) -> Any:
        if backend.fixture is not None:
            return backend.factory(prepared_target, kind)
        return backend.factory(kind)

    return build


def _audit(offset_minutes: int, review: bool = False, decision: str = "allow"):
    return {
        "timestamp": T0 + timedelta(minutes=offset_minutes),
        "action_proposed": f"synthetic action {offset_minutes}",
        "safety_level": "caution" if review else "safe",
        "checks_performed": {"manipulation_score": 0.1, "actor": "Guest"},
        "decision": decision,
        "reasoning": "synthetic reasoning",
        "requires_review": review,
    }


def _alert(
    offset_minutes: int,
    alert_type: str = "system",
    severity: str = "info",
    alert_id: Optional[str] = None,
):
    return {
        "alert_id": alert_id or f"alert-{offset_minutes}-{alert_type}-{severity}",
        "timestamp": T0 + timedelta(minutes=offset_minutes),
        "alert_type": alert_type,
        "severity": severity,
        "title": "synthetic title",
        "message": "synthetic message",
        "context": {"k": [1, 2]},
        "acknowledged": False,
    }


def _event(
    offset_minutes: int,
    user_id: str = "Alice",
    event_type: str = "data_read",
    event_id: Optional[str] = None,
):
    return {
        "event_id": event_id or f"event-{offset_minutes}-{user_id}-{event_type}",
        "timestamp": T0 + timedelta(minutes=offset_minutes),
        "event_type": event_type,
        "user_role": "guest",
        "user_id": user_id,
        "data_accessed": "synthetic record",
        "data_classification": "public",
        "success": True,
        "subject_consent": None,
        "metadata": {"k": 1},
    }


def _aware_utc(moment: Any) -> bool:
    return isinstance(moment, datetime) and moment.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_audit_round_trip(make_store):
    audits = make_store("safety")
    row_id = await audits.add_audit(_audit(0, review=True, decision="deny"))
    assert isinstance(row_id, str) and row_id
    [row] = await audits.recent()
    assert row["id"] == row_id
    assert row["action_proposed"] == "synthetic action 0"
    assert row["safety_level"] == "caution"
    assert row["checks_performed"] == {"manipulation_score": 0.1, "actor": "Guest"}
    assert row["decision"] == "deny"
    assert row["reasoning"] == "synthetic reasoning"
    assert row["requires_review"] is True
    assert row["timestamp"] == T0 and _aware_utc(row["timestamp"])

    minimal = await audits.add_audit(
        {"action_proposed": "x", "safety_level": "safe", "decision": "allow"}
    )
    newest = (await audits.recent(limit=1))[0]
    assert newest["id"] == minimal
    assert newest["checks_performed"] == {}
    assert newest["reasoning"] == ""
    assert newest["requires_review"] is False
    assert _aware_utc(newest["timestamp"])

    alerts = make_store("alert")
    await alerts.add_alert(_alert(0, alert_id="a-1"))
    [alert] = await alerts.recent_alerts()
    assert alert["alert_id"] == "a-1"
    assert alert["context"] == {"k": [1, 2]}
    assert alert["timestamp"] == T0 and _aware_utc(alert["timestamp"])

    privacy = make_store("privacy")
    await privacy.add_event(_event(0, event_id="e-1"))
    [event] = await privacy.events()
    assert event["event_id"] == "e-1"
    assert event["user_id"] == "Alice"
    assert event["metadata"] == {"k": 1}
    assert event["subject_consent"] is None
    assert event["timestamp"] == T0 and _aware_utc(event["timestamp"])


@pytest.mark.asyncio
async def test_audit_record_fields_checked(make_store):
    audits = make_store("safety")
    with pytest.raises(ValueError):
        await audits.add_audit({"action_proposed": "x", "safety_level": "safe"})
    with pytest.raises(ValueError):
        await audits.add_audit(
            {
                "action_proposed": "x",
                "safety_level": "safe",
                "decision": "allow",
                "speaker": "Alice",
            }
        )
    assert await audits.recent() == []
    alerts = make_store("alert")
    with pytest.raises(ValueError):
        await alerts.add_alert({"alert_id": "a", "alert_type": "system"})
    with pytest.raises(ValueError):
        await alerts.add_alert(_alert(0, severity="loud"))
    privacy = make_store("privacy")
    with pytest.raises(ValueError):
        await privacy.add_event({"event_id": "e", "event_type": "data_read"})


@pytest.mark.asyncio
async def test_pending_review_filters(make_store):
    audits = make_store("safety")
    for offset, review in ((0, True), (1, False), (2, True), (3, False), (4, True)):
        await audits.add_audit(_audit(offset, review=review))
    pending = await audits.pending_review()
    assert [r["action_proposed"] for r in pending] == [
        "synthetic action 4",
        "synthetic action 2",
        "synthetic action 0",
    ]
    assert all(r["requires_review"] is True for r in pending)
    assert len(await audits.pending_review(limit=2)) == 2
    assert await audits.pending_review(limit=0) == []
    assert len(await audits.recent()) == 5
    assert len(await audits.recent(limit=3)) == 3


@pytest.mark.asyncio
async def test_alerts_filter_by_type_and_severity(make_store):
    alerts = make_store("alert")
    await alerts.add_alert(_alert(0, "system", "info"))
    await alerts.add_alert(_alert(1, "privacy", "warning"))
    await alerts.add_alert(_alert(2, "system", "urgent"))
    await alerts.add_alert(_alert(3, "wellbeing", "critical"))
    await alerts.add_alert(_alert(4, "system", "warning"))

    system = await alerts.recent_alerts(alert_type="system")
    assert [a["severity"] for a in system] == ["warning", "urgent", "info"]
    urgent_up = await alerts.recent_alerts(min_severity="urgent")
    assert [a["severity"] for a in urgent_up] == ["critical", "urgent"]
    both = await alerts.recent_alerts(alert_type="system", min_severity="warning")
    assert [a["severity"] for a in both] == ["warning", "urgent"]
    assert len(await alerts.recent_alerts(min_severity="info")) == 5
    assert len(await alerts.recent_alerts(limit=2)) == 2
    assert await alerts.recent_alerts(alert_type="backup") == []
    with pytest.raises(ValueError):
        await alerts.recent_alerts(min_severity="loud")


@pytest.mark.asyncio
async def test_acknowledge_reports_existence(make_store):
    alerts = make_store("alert")
    await alerts.add_alert(_alert(0, alert_id="a-1"))
    await alerts.add_alert(_alert(1, alert_id="a-2"))
    assert await alerts.acknowledge("a-1") is True
    assert await alerts.acknowledge("missing") is False
    by_id = {a["alert_id"]: a for a in await alerts.recent_alerts()}
    assert by_id["a-1"]["acknowledged"] is True
    assert _aware_utc(by_id["a-1"]["acknowledged_at"])
    assert by_id["a-2"]["acknowledged"] is False
    assert await alerts.acknowledge("a-1") is True  # still exists


@pytest.mark.asyncio
async def test_privacy_events_filter(make_store):
    privacy = make_store("privacy")
    await privacy.add_event(_event(0, "Alice", "data_read"))
    await privacy.add_event(_event(10, "Operator", "access_denied"))
    await privacy.add_event(_event(20, "Alice", "access_denied"))
    await privacy.add_event(_event(30, "Guest", "data_read"))

    alice = await privacy.events(user_id="Alice")
    assert [e["timestamp"] for e in alice] == [
        T0 + timedelta(minutes=20),
        T0,
    ]
    denied = await privacy.events(event_type="access_denied")
    assert [e["user_id"] for e in denied] == ["Alice", "Operator"]
    since = await privacy.events(since=T0 + timedelta(minutes=10))
    assert [e["user_id"] for e in since] == ["Guest", "Alice", "Operator"]
    combined = await privacy.events(
        since=T0 + timedelta(minutes=5), user_id="Alice", event_type="access_denied"
    )
    assert len(combined) == 1
    assert len(await privacy.events(limit=2)) == 2
    assert await privacy.events(limit=0) == []
    assert await privacy.events(user_id="Nobody") == []


@pytest.mark.asyncio
async def test_newest_first(make_store):
    offsets = (5, 1, 9, 3)
    audits = make_store("safety")
    alerts = make_store("alert")
    privacy = make_store("privacy")
    for offset in offsets:
        await audits.add_audit(_audit(offset))
        await alerts.add_alert(_alert(offset))
        await privacy.add_event(_event(offset))
    expected = [T0 + timedelta(minutes=m) for m in sorted(offsets, reverse=True)]
    assert [r["timestamp"] for r in await audits.recent()] == expected
    assert [a["timestamp"] for a in await alerts.recent_alerts()] == expected
    assert [e["timestamp"] for e in await privacy.events()] == expected

    # Equal instants: the later write comes first.
    await privacy.add_event(_event(20, event_id="first"))
    await privacy.add_event(_event(20, event_id="second"))
    top = await privacy.events(limit=2)
    assert [e["event_id"] for e in top] == ["second", "first"]


@pytest.mark.asyncio
async def test_returned_documents_are_copies(make_store):
    audits = make_store("safety")
    record = _audit(0)
    await audits.add_audit(record)
    record["checks_performed"]["manipulation_score"] = 0.99
    [row] = await audits.recent()
    row["checks_performed"]["actor"] = "Operator"
    [again] = await audits.recent()
    assert again["checks_performed"] == {"manipulation_score": 0.1, "actor": "Guest"}

    alerts = make_store("alert")
    alert = _alert(0, alert_id="a-1")
    await alerts.add_alert(alert)
    alert["context"]["k"].append(3)
    [got] = await alerts.recent_alerts()
    got["context"]["k"].append(4)
    assert (await alerts.recent_alerts())[0]["context"] == {"k": [1, 2]}

    privacy = make_store("privacy")
    event = _event(0, event_id="e-1")
    await privacy.add_event(event)
    event["metadata"]["k"] = 2
    [stored] = await privacy.events()
    stored["metadata"]["k"] = 3
    assert (await privacy.events())[0]["metadata"] == {"k": 1}


@pytest.mark.asyncio
async def test_naive_timestamps_are_rejected(make_store):
    naive = datetime(2000, 1, 1, 12, 0, 0)
    audits = make_store("safety")
    with pytest.raises(ValueError):
        await audits.add_audit({**_audit(0), "timestamp": naive})
    alerts = make_store("alert")
    with pytest.raises(ValueError):
        await alerts.add_alert({**_alert(0), "timestamp": naive})
    privacy = make_store("privacy")
    with pytest.raises(ValueError):
        await privacy.add_event({**_event(0), "timestamp": naive})
    with pytest.raises(ValueError):
        await privacy.events(since=naive)


@pytest.mark.parametrize("kind", KINDS)
def test_implementations_satisfy_protocols(make_store, kind):
    store = make_store(kind)
    assert isinstance(store, PROTOCOLS[kind])
    report = store.status()
    assert {"backend", "available", "last_error"} <= set(report)


def test_severity_order_and_filter_helper():
    assert protocols.ALERT_SEVERITY_ORDER == ("info", "warning", "urgent", "critical")
    assert protocols.severities_at_or_above("warning") == (
        "warning",
        "urgent",
        "critical",
    )
    with pytest.raises(ValueError):
        protocols.severities_at_or_above("WARNING")


@pytest.mark.asyncio
async def test_inmemory_cap_drops_oldest_write():
    store = inmemory.InMemoryPrivacyAuditStore(max_items=2, clock=_fixed_clock)
    for offset in range(3):
        await store.add_event(_event(offset))
    assert [e["timestamp"] for e in await store.events()] == [
        T0 + timedelta(minutes=2),
        T0 + timedelta(minutes=1),
    ]
    with pytest.raises(ValueError):
        await store.add_event(_event(2))  # the same event_id again


@pytest.mark.asyncio
async def test_inmemory_clock_stamps_missing_timestamps():
    store = inmemory.InMemoryAlertStore(clock=_fixed_clock)
    alert = _alert(0, alert_id="a-1")
    del alert["timestamp"]
    await store.add_alert(alert)
    [got] = await store.recent_alerts()
    assert got["timestamp"] == _fixed_clock()
    await store.acknowledge("a-1")
    assert (await store.recent_alerts())[0]["acknowledged_at"] == _fixed_clock()
