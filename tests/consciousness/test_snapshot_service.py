"""The snapshot service stores assessment snapshots through the snapshot store
with the assessment's own status and coverage: an insufficient overall is
null and never crosses a threshold, events come from measured changes only,
and the JSON backup lands under the results directory."""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtc.consciousness.snapshot_service as snapshot_service
from mtc.assessment.assessment import (
    ConsciousnessReport,
    ConsciousnessTheory,
    IndicatorResult,
    PhiMeasurement,
)
from mtc.consciousness.snapshot_service import SnapshotConfig, SnapshotService
from mtc.memory.inmemory import InMemorySnapshotStore
from mtc.memory.protocols import SnapshotRecord

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


class _Clock:
    def __init__(self, start=T0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + timedelta(**kwargs)


def _report(
    overall=0.6,
    status="measured",
    coverage=1.0,
    theories=None,
    phi=None,
    passing=3,
):
    indicator = IndicatorResult(
        name="global_broadcast",
        theory=ConsciousnessTheory.GWT,
        score=overall if overall is not None else 0.0,
        confidence=0.8,
        threshold=0.5,
        passes_threshold=bool(overall is not None and overall >= 0.5),
        evidence={"cycles": 3},
        measurement_time_ms=1.5,
        status="insufficient_data" if status == "insufficient_data" else "measured",
    )
    phi_measurement = None
    if phi is not None:
        phi_measurement = PhiMeasurement(
            phi=phi,
            phi_normalized=min(1.0, phi / 10.0),
            subsystem_size=4,
            computation_time_ms=2.0,
            is_approximate=True,
        )
    return ConsciousnessReport(
        session_id="session-a",
        timestamp=T0,
        overall_score=overall,
        architecture_functional=True,
        confidence=0.5,
        indicator_results={"global_broadcast": indicator},
        passing_count=passing,
        total_indicators=30,
        theory_scores=dict(theories if theories is not None else {"GWT": 0.6}),
        phi_measurement=phi_measurement,
        processing_time_ms=12.0,
        status=status,
        coverage=coverage,
        insufficient_indicators=(
            ["global_broadcast"] if status == "insufficient_data" else []
        ),
    )


def _service(tmp_path, clock=None, **kwargs):
    store = kwargs.pop("store", None) or InMemorySnapshotStore()
    service = SnapshotService(
        store=store,
        backup_dir=kwargs.pop("backup_dir", tmp_path / "backup"),
        clock=clock or _Clock(),
        **kwargs,
    )
    return service, store


def test_take_snapshot_stores_record_and_backup(tmp_path):
    service, store = _service(tmp_path)
    record = _run(service.take_snapshot("manual", _report(0.62, phi=1.5)))

    assert isinstance(record, SnapshotRecord)
    assert record.id == 1
    assert record.trigger == "manual"
    assert record.timestamp == T0
    assert record.overall_score == pytest.approx(0.62)
    assert record.phi_value == pytest.approx(1.5)
    assert record.status == "measured" and record.coverage == 1.0
    assert record.passing_count == 3
    assert record.theory_scores == {"GWT": 0.6}
    assert record.indicator_results["global_broadcast"]["status"] == "measured"
    assert record.metadata["session_id"] == "session-a"
    assert _run(store.latest_snapshot()) == record

    files = list((tmp_path / "backup").glob("*.json"))
    assert len(files) == 1
    backup = json.loads(files[0].read_text())
    assert backup["snapshot_id"] == 1
    assert backup["trigger"] == "manual"
    assert backup["overall_score"] == pytest.approx(0.62)
    assert backup["status"] == "measured"


def test_backup_under_results_dir_with_snapshot_prefix(tmp_path, monkeypatch):
    results = tmp_path / "results"
    monkeypatch.setattr(
        snapshot_service,
        "get_settings",
        lambda: SimpleNamespace(RESULTS_DIR=str(results)),
    )
    service = SnapshotService(clock=_Clock())
    assert not results.exists()
    _run(service.take_snapshot("periodic", _report()))

    folder = results / "consciousness_snapshots"
    names = [p.name for p in folder.iterdir()]
    assert len(names) == 1
    assert re.fullmatch(r"snapshot_periodic_20000101T120000\d{6}Z\.json", names[0])

    # A trigger can never steer the file outside the folder.
    _run(service.take_snapshot("../../escape", _report()))
    assert all(p.parent == folder for p in folder.iterdir())
    assert not (tmp_path / "escape").exists()


def test_insufficient_overall_stored_as_null(tmp_path):
    service, store = _service(tmp_path)
    report = _report(0.2, status="insufficient_data", coverage=0.0)
    record = _run(service.take_snapshot("manual", report))
    assert record.overall_score is None
    assert record.status == "insufficient_data"
    assert record.coverage == 0.0
    assert record.indicator_results["global_broadcast"]["score"] is None
    backup = json.loads(next((tmp_path / "backup").glob("*.json")).read_text())
    assert backup["overall_score"] is None


def test_no_crossing_event_from_insufficient_snapshot(tmp_path):
    clock = _Clock()
    service, store = _service(tmp_path, clock)
    _run(service.take_snapshot("manual", _report(0.4)))
    clock.advance(minutes=5)
    _run(
        service.take_snapshot(
            "manual", _report(0.7, status="insufficient_data", coverage=0.0)
        )
    )
    assert _run(store.events()) == []

    # The next measured snapshot compares with the last measured value.
    clock.advance(minutes=5)
    record = _run(service.take_snapshot("manual", _report(0.6)))
    events = _run(store.events())
    assert [e.event_type for e in events] == ["score_crossed_up"]
    assert events[0].old_value == pytest.approx(0.4)
    assert events[0].new_value == pytest.approx(0.6)
    assert events[0].snapshot_id == record.id


def test_events_for_crossing_theory_change_and_phi_shift(tmp_path):
    clock = _Clock()
    service, store = _service(tmp_path, clock)
    _run(service.take_snapshot("manual", _report(0.6, theories={"GWT": 0.5}, phi=1.0)))
    assert _run(store.events()) == []

    clock.advance(minutes=1)
    _run(
        service.take_snapshot(
            "manual",
            _report(0.55, theories={"GWT": 0.55, "IIT": 0.9}, phi=1.5),
        )
    )
    assert _run(store.events()) == []  # changes below every threshold

    clock.advance(minutes=1)
    _run(service.take_snapshot("manual", _report(0.4, theories={"GWT": 0.75}, phi=3.0)))
    kinds = sorted(e.event_type for e in _run(store.events()))
    assert kinds == ["phi_shift", "score_crossed_down", "theory_change"]
    for event in _run(store.events()):
        assert event.timestamp == clock.now
        assert event.description.isascii()

    filtered = _run(service.get_events(event_type="phi_shift"))
    assert [e.event_type for e in filtered] == ["phi_shift"]


def test_custom_thresholds_come_from_config(tmp_path):
    config = SnapshotConfig(crossing_threshold=0.8, theory_change_threshold=0.5)
    clock = _Clock()
    service, store = _service(tmp_path, clock, config=config)
    _run(service.take_snapshot("manual", _report(0.7, theories={"GWT": 0.1})))
    clock.advance(minutes=1)
    _run(service.take_snapshot("manual", _report(0.85, theories={"GWT": 0.4})))
    assert [e.event_type for e in _run(store.events())] == ["score_crossed_up"]


def test_maybe_snapshot_respects_interval(tmp_path):
    clock = _Clock()
    reports = {"next": _report(0.6, phi=1.0)}
    calls = []

    async def assess():
        calls.append(clock.now)
        return reports["next"]

    service, store = _service(tmp_path, clock, assess=assess)
    first = _run(service.maybe_snapshot())
    assert first is not None and first.trigger == "periodic"

    clock.advance(hours=1)
    assert _run(service.maybe_snapshot()) is None

    reports["next"] = _report(0.4, phi=1.0)
    clock.advance(minutes=10)
    crossing = _run(service.maybe_snapshot())
    assert crossing.trigger == "threshold_crossing"

    reports["next"] = _report(0.4, phi=4.0)
    clock.advance(minutes=10)
    shift = _run(service.maybe_snapshot())
    assert shift.trigger == "phi_change"

    clock.advance(hours=1, minutes=59)
    assert _run(service.maybe_snapshot()) is None
    clock.advance(minutes=1)
    periodic = _run(service.maybe_snapshot())
    assert periodic.trigger == "periodic"
    assert periodic.timestamp == clock.now

    # An explicit report skips the assessment; an explicit instant is used.
    before = len(calls)
    later = clock.now + timedelta(hours=3)
    given = _run(service.maybe_snapshot(now=later, report=_report(0.4, phi=4.0)))
    assert given.trigger == "periodic" and given.timestamp == later
    assert len(calls) == before

    # No assess hook and no report: nothing to snapshot.
    bare, _ = _service(tmp_path, _Clock())
    assert _run(bare.maybe_snapshot()) is None

    async def failing():
        raise RuntimeError("assessment unavailable")

    broken, broken_store = _service(tmp_path, _Clock(), assess=failing)
    assert _run(broken.maybe_snapshot()) is None
    assert _run(broken_store.latest_snapshot()) is None


def test_history_and_trends_from_store(tmp_path):
    clock = _Clock(datetime(2000, 1, 1, 8, 0, tzinfo=timezone.utc))
    service, store = _service(tmp_path, clock)
    _run(service.take_snapshot("manual", _report(0.4, theories={"GWT": 0.2}, phi=1.0)))
    clock.advance(hours=4)
    _run(service.take_snapshot("manual", _report(0.6, theories={"GWT": 0.4}, phi=3.0)))
    clock.advance(hours=1)
    _run(
        service.take_snapshot(
            "manual", _report(0.1, status="insufficient_data", coverage=0.0)
        )
    )
    clock.advance(hours=20)  # 2000-01-02 09:00
    _run(service.take_snapshot("manual", _report(0.8, theories={"GWT": 0.9})))

    history = _run(service.get_history(hours=24))
    assert [r.overall_score for r in history] == [
        pytest.approx(0.8),
        None,
        pytest.approx(0.6),
    ]
    assert len(_run(service.get_history(hours=48))) == 4
    assert len(_run(service.get_history(hours=48, limit=2))) == 2

    trends = _run(service.get_trends(hours=72))
    days = trends["days"]
    assert [d["day"] for d in days] == ["2000-01-01", "2000-01-02"]
    first = days[0]
    assert first["snapshot_count"] == 3
    assert first["measured_count"] == 2
    assert first["avg_score"] == pytest.approx(0.5)
    assert first["min_score"] == pytest.approx(0.4)
    assert first["max_score"] == pytest.approx(0.6)
    assert first["avg_phi"] == pytest.approx(2.0)
    assert first["theory_scores"] == {"GWT": pytest.approx(0.3)}
    assert days[1]["avg_phi"] is None


def test_insufficient_only_day_has_no_averages(tmp_path):
    service, _ = _service(tmp_path)
    _run(
        service.take_snapshot(
            "manual", _report(0.1, status="insufficient_data", coverage=0.0)
        )
    )
    (day,) = _run(service.get_trends())["days"]
    assert day["measured_count"] == 0
    assert day["avg_score"] is None
    assert day["min_score"] is None and day["max_score"] is None


def test_delta_none_without_older_snapshot(tmp_path):
    clock = _Clock()
    service, store = _service(tmp_path, clock)
    assert _run(service.get_delta()) is None
    current = _run(service.get_current())
    assert current == {"latest": None, "delta_24h": None}

    _run(service.take_snapshot("manual", _report(0.5, phi=1.0, passing=2)))
    clock.advance(hours=3)
    _run(service.take_snapshot("manual", _report(0.7, phi=2.0, passing=4)))
    assert _run(service.get_delta(hours=24)) is None

    clock.advance(hours=24, minutes=1)
    _run(service.take_snapshot("manual", _report(0.65, phi=None, passing=5)))
    delta = _run(service.get_delta(hours=24))
    assert delta["overall_score"] == pytest.approx(0.65 - 0.7)
    assert delta["phi_value"] is None
    assert delta["passing_count"] == pytest.approx(1.0)
    current = _run(service.get_current())
    assert current["latest"].overall_score == pytest.approx(0.65)
    assert current["delta_24h"] == delta


def test_delta_has_no_passing_count_against_an_insufficient_snapshot(tmp_path):
    clock = _Clock()
    service, store = _service(tmp_path, clock)
    _run(service.take_snapshot("manual", _report(0.6, passing=12, coverage=0.8)))
    clock.advance(hours=25)
    _run(
        service.take_snapshot(
            "manual",
            _report(0.0, status="insufficient_data", coverage=0.0, passing=0),
        )
    )
    delta = _run(service.get_delta(hours=24))
    assert delta["passing_count"] is None
    assert delta["overall_score"] is None
    assert delta["coverage"] == pytest.approx(0.0 - 0.8)

    # The other way round: an insufficient older snapshot, a measured latest.
    clock.advance(hours=25)
    _run(service.take_snapshot("manual", _report(0.7, passing=9, coverage=0.9)))
    delta = _run(service.get_delta(hours=24))
    assert delta["passing_count"] is None
    assert delta["overall_score"] is None
    assert delta["coverage"] == pytest.approx(0.9)


def test_developmental_hook_optional_and_stored_in_metadata(tmp_path):
    async def hook():
        return {"status": "measured", "stage": 3}

    service, _ = _service(tmp_path, developmental_hook=hook)
    record = _run(service.take_snapshot("manual", _report()))
    assert record.metadata["developmental"] == {"status": "measured", "stage": 3}

    plain, _ = _service(tmp_path)
    assert (
        "developmental" not in _run(plain.take_snapshot("manual", _report())).metadata
    )

    async def nothing():
        return None

    async def broken():
        raise RuntimeError("assessor down")

    for bad in (nothing, broken):
        service, store = _service(tmp_path, developmental_hook=bad)
        record = _run(service.take_snapshot("manual", _report()))
        assert "developmental" not in record.metadata
        assert _run(store.latest_snapshot()) is not None


def test_none_store_is_in_memory(tmp_path):
    service = SnapshotService(backup_dir=tmp_path)
    assert isinstance(service.store, InMemorySnapshotStore)
    source = Path(snapshot_service.__file__).read_text(encoding="utf-8")
    for driver in ("asyncpg", "psycopg", "motor", "pymongo", "sqlalchemy"):
        assert driver not in source


def test_take_snapshot_needs_a_report_or_an_assessment(tmp_path):
    service, _ = _service(tmp_path)
    with pytest.raises(ValueError):
        _run(service.take_snapshot("manual"))

    async def assess():
        return _report(0.55)

    assessed, _ = _service(tmp_path, assess=assess)
    assert _run(assessed.take_snapshot("manual")).overall_score == pytest.approx(0.55)


def test_initialize_seeds_last_values_from_store(tmp_path):
    clock = _Clock()
    store = InMemorySnapshotStore()
    first, _ = _service(tmp_path, clock, store=store)
    _run(first.take_snapshot("manual", _report(0.6)))

    clock.advance(minutes=30)
    second, _ = _service(tmp_path, clock, store=store, assess=None)
    _run(second.initialize())
    # The restored last snapshot is 30 minutes old: not yet periodic, but a
    # crossing against its score is detected.
    assert _run(second.maybe_snapshot(report=_report(0.6))) is None
    crossed = _run(second.maybe_snapshot(report=_report(0.3)))
    assert crossed.trigger == "threshold_crossing"


def test_service_reads_no_agent_space(tmp_path, monkeypatch):
    agent_space = tmp_path / "agent_space"
    monkeypatch.setattr(
        snapshot_service,
        "get_settings",
        lambda: SimpleNamespace(
            RESULTS_DIR=str(tmp_path / "results"), AGENT_SPACE_DIR=str(agent_space)
        ),
    )
    service = SnapshotService(clock=_Clock())
    _run(service.take_snapshot("manual", _report()))
    _run(service.get_trends())
    _run(service.get_current())
    assert not agent_space.exists()
    source = Path(snapshot_service.__file__).read_text(encoding="utf-8")
    assert "AGENT_SPACE_DIR" not in source
    assert "SYSTEM_BIRTH_DATE" not in source
