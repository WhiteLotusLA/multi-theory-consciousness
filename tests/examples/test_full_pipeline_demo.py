"""The full pipeline demo runs offline, end to end, and meets its exit
criterion: a subprocess under a socket guard that fails any connection, the
summary parsed and every clause asserted, then a privacy audit over the log
directory it wrote, at the default log level, that finds nothing beyond the
frozen assessment framework's known hits, which the summary names.

The log directory sits outside the repository, where ``scripts/privacy_audit.py
--paths`` cannot follow (it relativizes every path to the repository root), so
the audit runs through the module the script loads, over text copies of the
log files, with the public denylist and the private extra list the script
would read and no allowlist: stricter than the script, never looser."""

import json
import os
import shutil
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
# The one frozen module (closed until Phase 9) whose log text may carry an
# absolute module path; the demo reports those hits by name.
FROZEN_LOGGER = "mtc.assessment.framework"


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
        {"PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes", "MTC_DEVICE": "cpu"}
    )
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
    # The audit over the log directory: no hit the demo cannot explain, and
    # every hit the frozen assessment framework causes is reported by name in
    # the summary, never filtered out of the log.
    log_audit = summary["log_audit"]
    assert log_audit["files"] >= 1
    assert log_audit["unexplained_hits"] == []
    known = log_audit["known_frozen_module_log_hits"]
    scratch = tmp_path / "audit_copies"
    scratch.mkdir()
    hits = audit_log_directory(state / "logs", scratch)
    found = {(Path(hit.path).name[: -len(".txt")], hit.line) for hit in hits}
    assert found == {(entry["file"], entry["line"]) for entry in known}, [
        f"{Path(hit.path).name}:{hit.line}:{hit.kind}" for hit in hits
    ]
    for entry in known:
        assert entry["logger"] == FROZEN_LOGGER
        lines = (state / "logs" / entry["file"]).read_text(encoding="utf-8")
        record = json.loads(lines.splitlines()[entry["line"] - 1])
        assert record["logger"] == FROZEN_LOGGER
        assert record["line"] == entry["source_line"]


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
