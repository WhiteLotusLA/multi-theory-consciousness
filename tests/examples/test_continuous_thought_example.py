"""The continuous thought example runs offline, end to end, with the mock brain.

The example runs in a subprocess under a small runner that makes every outgoing
connection fail and counts the attempts, with a temporary state directory, so a
pass shows it needs no network, no store and no model server.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "continuous_thought.py"

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
def test_example_runs_offline_and_reports_a_summary(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    state = tmp_path / "state"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("MTC_")
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            str(EXAMPLE),
            "--ticks",
            "12",
            "--state-dir",
            str(state),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = completed.stdout.splitlines()
    summary_lines = [line for line in lines if line.startswith("CTM_SUMMARY ")]
    assert len(summary_lines) == 1
    summary = json.loads(summary_lines[0][len("CTM_SUMMARY ") :])
    assert summary["ok"] is True
    assert summary["ticks"] == 12
    assert summary["generated"] >= 1
    assert summary["provider"] == "mock"
    assert summary["embedder"] == "non_semantic"
    assert summary["non_semantic_opt_in"] is True
    assert summary["failures"] == {}
    assert "NETWORK_ATTEMPTS 0" in lines
    assert any("NON-SEMANTIC" in line for line in lines)  # the opt-in is labelled
    # the provider state went to the temporary state directory
    assert (state / "data" / "provider_state.json").is_file()
    assert not (tmp_path / "data").exists()
