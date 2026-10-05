"""The legacy_v04 preset must reproduce the recorded v0.4.0 runner numbers."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "fixtures" / "baseline_v040_experiment.txt"


def _metric(text: str, name: str) -> float:
    match = re.search(rf"^{name}:\s*([0-9.]+)", text, re.M)
    assert match, f"{name} missing from runner output"
    return float(match.group(1))


@pytest.mark.slow
def test_legacy_v04_preset_reproduces_baseline():
    baseline = BASELINE.read_text()
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--preset", "legacy_v04"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "PYPHI_WELCOME_OFF": "yes"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert (
        abs(_metric(out, "overall_score") - _metric(baseline, "overall_score")) <= 0.01
    )
    assert _metric(out, "passing_count") == _metric(baseline, "passing_count")
