"""The shared check vocabulary: exit codes, strict mode, ASCII text, JSON and
a summary that counts every status."""

import json
from pathlib import Path

import pytest

from mtc.core.checks import (
    CheckResult,
    CheckStatus,
    exit_code_for,
    render_text,
    results_from_json,
    summarize,
    to_json,
)


def _r(status, required=True, name="check"):
    return CheckResult(name=name, status=CheckStatus(status), required=required)


def test_exit_code_rules():
    assert exit_code_for([]) == 0
    assert exit_code_for([_r("ok"), _r("skipped")]) == 0
    assert exit_code_for([_r("ok"), _r("failed")]) == 1
    assert exit_code_for([_r("unreachable")]) == 3
    # A required failure wins over a required unreachable store.
    assert exit_code_for([_r("unreachable"), _r("failed")]) == 1
    # Advisory checks never change the code.
    advisory = [
        _r("failed", required=False),
        _r("unreachable", required=False),
        _r("warning"),
        _r("insufficient_data"),
    ]
    assert exit_code_for(advisory) == 0
    # A skipped check never fails, even when required.
    assert exit_code_for([_r("skipped")], strict=True) == 0


@pytest.mark.parametrize("status", ["warning", "insufficient_data"])
def test_strict_counts_warnings(status):
    assert exit_code_for([_r("ok"), _r(status)]) == 0
    assert exit_code_for([_r("ok"), _r(status)], strict=True) == 1
    assert exit_code_for([_r(status, required=False)], strict=True) == 1
    assert exit_code_for([_r("unreachable", required=False)], strict=True) == 3
    assert exit_code_for([_r("failed", required=False)], strict=True) == 1


def test_render_text_is_ascii():
    results = [
        CheckResult("store:postgres", CheckStatus.OK, "schema matches"),
        CheckResult(
            "store:redis",
            CheckStatus.WARNING,
            "policy caf" + chr(0xE9) + " " + chr(0x2014) + " noeviction\nsecond",
            required=False,
        ),
    ]
    text = render_text(results, title="Database check " + chr(0x2013))
    text.encode("ascii")
    lines = text.splitlines()
    assert lines[0].startswith("Database check")
    assert "[ok]" in lines[1] and "store:postgres" in lines[1]
    assert "(advisory)" in lines[2] and "second" in lines[2]
    assert len(lines) == 4
    assert lines[-1].startswith("summary: ok=1 warning=1 failed=0")


def test_json_round_trip():
    results = [
        CheckResult(
            "store:qdrant",
            CheckStatus.FAILED,
            "dimension 8192, expected 384",
            data={"expected": 384, "found": 8192, "path": Path("x")},
        ),
        CheckResult("store:neo4j", CheckStatus.SKIPPED, "disabled", required=False),
    ]
    payload = json.loads(json.dumps(to_json(results)))
    assert payload["exit_code"] == 1
    assert payload["summary"]["failed"] == 1 and payload["summary"]["total"] == 2
    rebuilt = results_from_json(payload)
    assert [r.name for r in rebuilt] == ["store:qdrant", "store:neo4j"]
    assert [r.status for r in rebuilt] == [CheckStatus.FAILED, CheckStatus.SKIPPED]
    assert rebuilt[1].required is False
    assert rebuilt[0].data == {"expected": 384, "found": 8192, "path": "x"}
    assert to_json([_r("warning")], strict=True)["exit_code"] == 1


def test_summary_counts_every_status():
    statuses = [status.value for status in CheckStatus]
    assert statuses == [
        "ok",
        "warning",
        "failed",
        "unreachable",
        "skipped",
        "insufficient_data",
    ]
    results = [_r(status) for status in statuses] + [_r("ok")]
    counts = summarize(results)
    assert counts == {
        "ok": 2,
        "warning": 1,
        "failed": 1,
        "unreachable": 1,
        "skipped": 1,
        "insufficient_data": 1,
        "total": 7,
    }
    assert summarize([]) == {status: 0 for status in statuses} | {"total": 0}
    assert isinstance(CheckStatus.OK, str) and CheckStatus("ok") is CheckStatus.OK
