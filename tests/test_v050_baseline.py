"""The v0.5.0 defaults must reproduce the recorded v0.5.0 fixtures.

``tests/fixtures/baseline_v050_experiment.txt`` is the experiment runner's
output (stdout and stderr) with no preset, and
``tests/fixtures/baseline_v050_assessment.json`` is the seeded assessment CLI's
report; ``tests/fixtures/baseline_v050_env.txt`` records the environment they
were captured in. With PyPhi absent the tracker path falls back to the
calculator, so both tests skip without PyPhi (and without pymdp, which the
inference module needs).

A deliberate default change in a later phase regenerates both fixtures with

    PYPHI_WELCOME_OFF=yes python3 scripts/run_experiment.py \
        > tests/fixtures/baseline_v050_experiment.txt 2>&1
    PYPHI_WELCOME_OFF=yes python3 scripts/run_assessment.py --full --json \
        --seed 42 --json-path tests/fixtures/baseline_v050_assessment.json

(dropping any warning line that carries a home-directory path and keeping every
metric line verbatim), re-captures the environment file, and adds its rows to
``docs/MIGRATION_FROM_V04.md``.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXPERIMENT_BASELINE = FIXTURES / "baseline_v050_experiment.txt"
ASSESSMENT_BASELINE = FIXTURES / "baseline_v050_assessment.json"

_ENV = {**os.environ, "PYPHI_WELCOME_OFF": "yes"}


def _skip_without_optional_deps() -> None:
    os.environ.setdefault("PYPHI_WELCOME_OFF", "yes")
    pytest.importorskip("pymdp")
    pytest.importorskip("pyphi")


def _metric(text: str, name: str) -> float:
    match = re.search(rf"^{name}:\s*([0-9.]+)", text, re.M)
    assert match, f"{name} missing from runner output"
    return float(match.group(1))


@pytest.mark.slow
def test_v050_defaults_reproduce_fixture():
    """The experiment runner without a preset lands on the v0.5.0 fixture."""
    _skip_without_optional_deps()
    baseline = EXPERIMENT_BASELINE.read_text()
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=_ENV,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert (
        abs(_metric(out, "overall_score") - _metric(baseline, "overall_score")) <= 0.01
    )
    assert _metric(out, "passing_count") == _metric(baseline, "passing_count")


@pytest.mark.slow
def test_v050_assessment_fixture_reproduces(tmp_path):
    """The seeded assessment CLI lands on the v0.5.0 assessment fixture."""
    _skip_without_optional_deps()
    baseline = json.loads(ASSESSMENT_BASELINE.read_text())
    json_path = tmp_path / "assessment.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_assessment.py",
            "--full",
            "--json",
            "--seed",
            "42",
            "--json-path",
            str(json_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=_ENV,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(json_path.read_text())
    assert report["overall_score"] is not None
    assert abs(report["overall_score"] - baseline["overall_score"]) <= 0.01
    assert report["passing_count"] == baseline["passing_count"]
