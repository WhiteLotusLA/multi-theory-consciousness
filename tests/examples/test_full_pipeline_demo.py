"""The full pipeline demo runs offline, end to end, and meets its exit
criterion: a subprocess under a socket guard that fails any connection, the
summary parsed and every clause asserted, then a privacy audit over the log
directory it wrote, at the default log level, that finds nothing at all.

The log directory sits outside the repository, where ``scripts/privacy_audit.py
--paths`` cannot follow (it relativizes every path to the repository root), so
the audit runs through the module the script loads, over text copies of the
log files, with the public denylist and the private extra list the script
would read and no allowlist: stricter than the script, never looser."""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from mtc.devtools.privacy_audit import load_denylist, run_audit

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "full_pipeline_demo.py"
DENYLIST = ROOT / "config" / "privacy" / "denylist.txt"
LOCAL_EXTRA = ROOT / "config" / "privacy" / "denylist.local.txt"
STORE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _extra_denylist() -> List[str]:
    configured = os.environ.get("PRIVACY_DENYLIST_EXTRA")
    if configured and Path(configured).is_file():
        return load_denylist(Path(configured))
    if LOCAL_EXTRA.is_file():
        return load_denylist(LOCAL_EXTRA)
    return []


def audit_log_directory(log_dir: Path, scratch: Path) -> list:
    """Every hit of the privacy audit over the log files, as text copies."""
    copies = []
    for source in sorted(p for p in log_dir.rglob("*") if p.is_file()):
        copy = scratch / (source.name + ".txt")
        shutil.copyfile(source, copy)
        copies.append(copy)
    assert copies, "the demo wrote no log file"
    return run_audit(
        copies, load_denylist(DENYLIST), [], extra_denylist=_extra_denylist()
    )


RUNNER = """
import runpy
import socket
import sys

attempts = []


def refuse(*args, **kwargs):
    attempts.append(1)
    raise OSError("outgoing connections are disabled in this test")


socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
example = sys.argv[1]
sys.argv = [example] + sys.argv[2:]
code = 0
try:
    runpy.run_path(example, run_name="__main__")
except SystemExit as stop:
    code = stop.code if isinstance(stop.code, int) else 1
print("NETWORK_ATTEMPTS", len(attempts))
sys.exit(code)
"""


@pytest.mark.slow
def test_demo_runs_offline_and_meets_exit_criterion(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    state = tmp_path / "state"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTC_")}
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_API_PORT": str(_unused_port()),
        }
    )
    env.update({key: "false" for key in STORE_KEYS})
    completed = subprocess.run(
        [sys.executable, str(runner), str(EXAMPLE), "--state-dir", str(state)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    summaries = [line for line in lines if line.startswith("DEMO_SUMMARY ")]
    assert len(summaries) == 1, completed.stdout[-2000:]
    summary = json.loads(summaries[0][len("DEMO_SUMMARY ") :])
    assert summary["ok"] is True
    assert summary["turns"] == 5
    assert summary["cycle_ok_all"] is True
    assert summary["step_failures"] == 0
    assert summary["unexplained_none_fields"] == []
    assert summary["replies_stored"] == 5
    assert summary["provider"] == "mock"
    assert summary["trust"] == {"spoofed_trusted": False, "authenticated_trusted": True}
    # The trust outcome is read back from the runtime for every turn.
    outcome_lines = [line for line in lines if "read from the runtime" in line]
    assert len(outcome_lines) == 5
    assert any("token matched True" in line for line in outcome_lines)
    assert sum("safety gate bypassed" in line for line in outcome_lines) == 1
    assert summary["ctm"]["ticks"] == 10
    assert summary["ctm"]["generated"] >= 1
    assert summary["elapsed_s"] < 600
    assert "NETWORK_ATTEMPTS 0" in lines
    assert any("NON-SEMANTIC" in line for line in lines)
    # nothing landed in the repository; everything sits under the state dir
    assert (state / "state" / "manifest.json").is_file()
    assert not (tmp_path / "data").exists()
    # The audit over the log directory finds nothing: no allowance, every hit
    # would fail the run.
    log_audit = summary["log_audit"]
    assert log_audit["files"] >= 1
    assert log_audit["hits"] == 0
    assert log_audit["unexplained_hits"] == []
    assert "known_frozen_module_log_hits" not in log_audit
    assert not [line for line in lines if "known" in line and "log audit" in line]
    scratch = tmp_path / "audit_copies"
    scratch.mkdir()
    hits = audit_log_directory(state / "logs", scratch)
    assert hits == [], [f"{Path(hit.path).name}:{hit.line}:{hit.kind}" for hit in hits]


@pytest.mark.slow
def test_demo_honours_a_relative_state_dir(tmp_path):
    """Run from another directory with a relative ``--state-dir`` (the CI
    install smoke's invocation), every file sits under that directory and
    nothing under the repository, and the run meets its exit criterion: the
    speaker table the demo writes is the one the runtime reads, and the log
    audit finds the log the runtime wrote."""
    stray = ROOT / "relative_state_dir"
    assert not stray.exists(), "a leftover relative_state_dir sits in the repository"
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTC_")}
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_API_PORT": str(_unused_port()),
        }
    )
    env.update({key: "false" for key in STORE_KEYS})
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(runner),
                str(EXAMPLE),
                "--state-dir",
                "relative_state_dir/demo",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        landed_in_repository = stray.exists()
    finally:
        shutil.rmtree(stray, ignore_errors=True)
    assert not landed_in_repository, "the demo wrote under the repository"
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    summaries = [line for line in lines if line.startswith("DEMO_SUMMARY ")]
    assert len(summaries) == 1, completed.stdout[-2000:]
    summary = json.loads(summaries[0][len("DEMO_SUMMARY ") :])
    assert summary["ok"] is True
    assert summary["trust"]["authenticated_trusted"] is True
    assert summary["log_audit"]["files"] >= 1
    state = tmp_path / "relative_state_dir" / "demo"
    assert (state / "speakers.yaml").is_file()
    assert (state / "state" / "manifest.json").is_file()
    assert any(p.is_file() for p in (state / "logs").rglob("*"))
    assert "NETWORK_ATTEMPTS 0" in lines


def test_audit_helper_detects_a_planted_hit(tmp_path):
    # The planted token is read from the public denylist at run time, so this
    # file spells no denied word and no pattern of its own.
    planted = load_denylist(DENYLIST)[0]
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "mtc.log").write_text(
        f"a log line that names {planted} by mistake\n", encoding="utf-8"
    )
    scratch = tmp_path / "copies"
    scratch.mkdir()
    hits = audit_log_directory(log_dir, scratch)
    assert hits, "the audit helper must see a planted denylist token"
    assert all(hit.kind == "token" for hit in hits)


def _demo():
    import runpy

    return runpy.run_path(str(EXAMPLE), run_name="full_pipeline_demo")


def test_demo_audit_has_no_allowance(tmp_path):
    """A hit on a record of any logger, the assessment framework included,
    is unexplained: the demo's audit keeps no list of known hits."""
    demo = _demo()
    assert "KNOWN_FROZEN_LOG_SOURCES" not in demo
    planted = load_denylist(DENYLIST)[0]
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    record = {
        "logger": "mtc.assessment.framework",
        "line": 147,
        "message": f"a record that names {planted}",
    }
    (log_dir / "mtc.log").write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = demo["audit_logs"](log_dir)
    assert report["files"] == 1
    assert report["hits"] == len(report["unexplained_hits"]) >= 1
    assert set(report) == {"files", "hits", "unexplained_hits"}


def test_unexplained_none_check_covers_the_measured_fields():
    """A measured state field that reads None (the integration level, the
    broadcast coverage, the binding quality, the protoself stability, the
    feeling of knowing, the autobiographic continuity) counts as unexplained
    unless the turn names it in absent_signals."""
    from types import SimpleNamespace

    demo = _demo()
    measured = (
        "integration_level",
        "broadcast_coverage",
        "binding_quality",
        "protoself_stability",
        "feeling_of_knowing",
        "autobiographic_continuity",
    )
    state = SimpleNamespace(**{name: None for name in demo["OPTIONAL_FIELDS"]})
    absent = {name: "no_input" for name in demo["OPTIONAL_FIELDS"]}

    def runtime(named):
        return SimpleNamespace(
            orchestrator=SimpleNamespace(_last_state=state),
            pipeline=SimpleNamespace(
                status=lambda: {"last_turn": {"absent_signals": named}}
            ),
        )

    assert demo["_unexplained"](runtime(absent)) == []
    for name in measured:
        named = {key: value for key, value in absent.items() if key != name}
        assert demo["_unexplained"](runtime(named)) == [name], name
