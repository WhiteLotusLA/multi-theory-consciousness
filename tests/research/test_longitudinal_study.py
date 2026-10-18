"""Longitudinal study manager: session records and reports."""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtc.research import longitudinal_study as ls
from mtc.research.longitudinal_study import LongitudinalStudyManager

T0 = datetime(2000, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


def fake_report(
    overall, status="measured", coverage=1.0, indicators=None, theories=None
):
    indicators = indicators or {"alpha_signal": (overall, "measured")}
    return SimpleNamespace(
        status=status,
        coverage=coverage,
        overall_score=overall,
        theory_scores=theories or {"GWT": overall if overall is not None else 0.0},
        insufficient_indicators=[
            n for n, (_, s) in indicators.items() if s != "measured"
        ],
        indicator_results={
            n: SimpleNamespace(status=s, score=v) for n, (v, s) in indicators.items()
        },
    )


class Sequence:
    """An assess callable that returns queued reports, then raises."""

    def __init__(self, *items):
        self.items = list(items)

    async def __call__(self):
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Ticker:
    """A clock that advances once per session (two reads: start and end)."""

    def __init__(self, start=T0, step=timedelta(hours=1)):
        self.start, self.step, self.calls = start, step, 0

    def __call__(self):
        value = self.start + self.step * (self.calls // 2)
        self.calls += 1
        return value


def manager(assess, tmp_path, **kw):
    kw.setdefault("clock", Ticker())
    return LongitudinalStudyManager("study-1", assess, output_dir=tmp_path, **kw)


def test_session_appends_jsonl_under_output_dir(tmp_path):
    mgr = manager(Sequence(fake_report(0.5), fake_report(0.6)), tmp_path / "out")
    assert not (tmp_path / "out").exists()
    run(mgr.run_session("morning"))
    run(mgr.run_session())
    path = tmp_path / "out" / "study-1" / "sessions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["session_number"] for r in rows] == [1, 2]
    assert rows[0]["label"] == "morning" and rows[1]["label"] == "scheduled"
    assert rows[0]["overall_score"] == 0.5
    assert datetime.fromisoformat(rows[0]["timestamp"]).utcoffset() == timedelta(0)


def test_output_dir_defaults_to_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ls,
        "get_settings",
        lambda: SimpleNamespace(LONGITUDINAL_OUTPUT_DIR=str(tmp_path / "cfg")),
    )
    mgr = LongitudinalStudyManager("s", Sequence(fake_report(0.5)), clock=Ticker())
    run(mgr.run_session())
    assert (tmp_path / "cfg" / "s" / "sessions.jsonl").exists()


def test_default_directory_resolves_under_project_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(
        ls,
        "get_settings",
        lambda: SimpleNamespace(
            LONGITUDINAL_OUTPUT_DIR="results/longitudinal_studies",
            PROJECT_ROOT=str(project),
        ),
    )
    mgr = LongitudinalStudyManager("s", Sequence(fake_report(0.5)), clock=Ticker())
    assert mgr.study_dir == project / "results" / "longitudinal_studies" / "s"
    run(mgr.run_session())
    assert (
        project / "results" / "longitudinal_studies" / "s" / "sessions.jsonl"
    ).exists()
    assert not (elsewhere / "results").exists()


def test_failed_assessment_records_error_not_scores(tmp_path):
    mgr = manager(Sequence(RuntimeError("module offline")), tmp_path)
    record = run(mgr.run_session())
    assert record.status == "error"
    assert record.overall_score is None and record.theory_scores == {}
    assert "module offline" in record.error
    assert mgr.sessions()[0].error == record.error


def test_insufficient_sessions_excluded_from_trend(tmp_path):
    mgr = manager(
        Sequence(
            fake_report(0.5),
            fake_report(
                None,
                status="insufficient_data",
                coverage=0.0,
                indicators={"a": (0.0, "insufficient_data")},
            ),
            fake_report(0.6),
            RuntimeError("x"),
            fake_report(0.7),
        ),
        tmp_path,
    )
    for _ in range(5):
        run(mgr.run_session())
    final = mgr.final_report()
    assert final["n_sessions"] == 5
    assert final["n_measured"] == 3
    assert final["n_insufficient"] == 1 and final["n_errors"] == 1
    assert final["trajectory"] == [0.5, 0.6, 0.7]
    assert final["trend"]["n"] == 3
    assert final["trend"]["slope"] > 0
    assert final["mean_overall"] == pytest.approx(0.6)


def test_daily_and_weekly_reports_aggregate_measured_sessions(tmp_path):
    clock = Ticker(
        start=datetime(2000, 1, 3, 20, 0, tzinfo=timezone.utc), step=timedelta(hours=6)
    )
    mgr = manager(
        Sequence(
            fake_report(0.4),  # Jan 3 20:00
            fake_report(0.6),  # Jan 4 02:00
            fake_report(
                None,
                status="insufficient_data",
                coverage=0.0,
                indicators={"a": (0.0, "insufficient_data")},
            ),  # Jan 4 08:00
            fake_report(0.8),  # Jan 4 14:00
        ),
        tmp_path,
        clock=clock,
    )
    for _ in range(4):
        run(mgr.run_session())
    day = mgr.daily_report(date(2000, 1, 4))
    assert day["status"] == "measured"
    assert day["n_sessions"] == 3 and day["n_measured"] == 2
    assert day["mean_overall"] == pytest.approx(0.7)
    assert day["change_from_previous_day"] == pytest.approx(0.3)
    first = mgr.daily_report(date(2000, 1, 3))
    assert first["change_from_previous_day"] is None
    week = mgr.weekly_report(date(2000, 1, 3))
    assert week["n_sessions"] == 4 and week["n_measured"] == 3
    assert week["week_end"] == "2000-01-09"
    empty = mgr.daily_report(date(2000, 2, 1))
    assert empty["status"] == "no_sessions" and empty["mean_overall"] is None
    other_week = mgr.weekly_report(date(2000, 1, 10))
    assert other_week["n_sessions"] == 0


def test_days_are_utc_days(tmp_path):
    late = datetime(
        2000, 1, 1, 23, 30, tzinfo=timezone(timedelta(hours=-5))
    )  # Jan 2 04:30 UTC
    mgr = manager(Sequence(fake_report(0.5)), tmp_path, clock=Ticker(start=late))
    run(mgr.run_session())
    assert mgr.daily_report(date(2000, 1, 2))["n_sessions"] == 1
    assert mgr.daily_report(date(2000, 1, 1))["n_sessions"] == 0


def test_trend_reports_slope_and_p_value(tmp_path):
    scores = [0.30, 0.38, 0.52, 0.61, 0.70, 0.79]
    mgr = manager(
        Sequence(*[fake_report(s) for s in scores]),
        tmp_path,
        clock=Ticker(step=timedelta(days=1)),
    )
    for _ in scores:
        run(mgr.run_session())
    final = mgr.final_report()
    trend = final["trend"]
    assert trend["slope"] == pytest.approx(0.0925, abs=0.01)  # per day
    assert trend["p_value"] < 0.05 and trend["significant"] is True
    assert final["significant_positive_trend"] is True
    assert "says nothing about experience" in final["interpretation"]
    assert final["change"] == pytest.approx(0.49)


def test_trend_flag_needs_five_measured_sessions(tmp_path):
    scores = [0.3, 0.5, 0.7, 0.9]
    mgr = manager(Sequence(*[fake_report(s) for s in scores]), tmp_path)
    for _ in scores:
        run(mgr.run_session())
    final = mgr.final_report()
    assert final["trend"]["significant"] is True
    assert final["significant_positive_trend"] is False


def test_trend_none_below_three_measured(tmp_path):
    mgr = manager(Sequence(fake_report(0.4), fake_report(0.5)), tmp_path)
    for _ in range(2):
        run(mgr.run_session())
    assert mgr.final_report()["trend"] is None


def test_no_random_or_mock_snapshots(tmp_path):
    mgr = manager(Sequence(fake_report(0.5)), tmp_path)
    record = run(mgr.run_session())
    assert record.battery is None and record.snapshot is None
    assert record.battery_error is None and record.snapshot_error is None
    row = json.loads(mgr.sessions_path.read_text())
    assert row["battery"] is None and row["snapshot"] is None


def test_suite_and_snapshot_hook_are_recorded_and_failures_isolated(tmp_path):
    calls = []

    class Suite:
        async def run_battery(self):
            return SimpleNamespace(
                status="measured",
                overall_score=0.7,
                summaries={
                    "false_belief": SimpleNamespace(
                        status="measured", n_completed=5, n_errors=1, mean_score=0.7
                    )
                },
            )

    async def snap(label):
        calls.append(label)
        return {"id": "snap-1", "n": float("nan")}

    mgr = manager(Sequence(fake_report(0.5)), tmp_path, suite=Suite(), snapshot=snap)
    record = run(mgr.run_session("evening"))
    assert calls == ["evening"]
    assert record.battery["experiments"]["false_belief"]["n_errors"] == 1
    assert record.snapshot == {"id": "snap-1", "n": None}

    async def bad_snap(label):
        raise OSError("disk")

    class BadSuite:
        async def run_battery(self):
            raise RuntimeError("endpoint")

    mgr = manager(
        Sequence(fake_report(0.5)), tmp_path / "b", suite=BadSuite(), snapshot=bad_snap
    )
    record = run(mgr.run_session())
    assert record.status == "measured" and record.overall_score == 0.5
    assert "endpoint" in record.battery_error and "disk" in record.snapshot_error


def test_indicator_names_come_from_the_report(tmp_path):
    inds = {"zeta_probe": (0.4, "measured"), "eta_probe": (0.0, "insufficient_data")}
    mgr = manager(
        Sequence(
            fake_report(0.5, indicators=inds),
            fake_report(0.7, indicators={"zeta_probe": (0.8, "measured")}),
        ),
        tmp_path,
    )
    for _ in range(2):
        run(mgr.run_session())
    first = mgr.sessions()[0]
    assert set(first.indicators) == {"zeta_probe", "eta_probe"}
    assert first.indicators["eta_probe"] == {
        "status": "insufficient_data",
        "score": None,
    }
    final = mgr.final_report()
    assert set(final["indicators"]) == {"zeta_probe"}
    assert final["indicators"]["zeta_probe"]["mean_score"] == pytest.approx(0.6)
    assert final["indicators"]["zeta_probe"]["n_measured"] == 2


def test_numbering_continues_across_managers_and_malformed_lines_are_skipped(tmp_path):
    first = manager(Sequence(fake_report(0.5)), tmp_path)
    run(first.run_session())
    with open(first.sessions_path, "a") as fh:
        fh.write("not json\n")
    second = manager(Sequence(fake_report(0.6)), tmp_path)
    record = run(second.run_session())
    assert record.session_number == 2
    assert len(second.sessions()) == 2


def test_unmeasured_assessment_report_records_no_overall(tmp_path):
    from mtc.assessment.assessment import ConsciousnessAssessment

    async def assess():
        # No module supplied: every indicator is absent, nothing is measured
        return await ConsciousnessAssessment().run_full_assessment(normalize=False)

    mgr = manager(assess, tmp_path)
    record = run(mgr.run_session())
    assert record.status == "insufficient_data"
    assert record.overall_score is None
    assert record.theory_scores == {}
    assert record.is_measured is False
    assert all(info["score"] is None for info in record.indicators.values())
    row = json.loads(mgr.sessions_path.read_text())
    assert row["overall_score"] is None
    final = mgr.final_report()
    assert final["n_measured"] == 0 and final["n_insufficient"] == 1
    assert final["status"] == "insufficient_data"


def test_study_id_must_be_plain(tmp_path):
    for bad in ("", "../x", "a/b"):
        with pytest.raises(ValueError):
            LongitudinalStudyManager(bad, Sequence(), output_dir=tmp_path)
