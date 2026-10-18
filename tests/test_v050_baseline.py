"""The v0.5.0 defaults must reproduce the recorded v0.5.0 fixtures.

``tests/fixtures/baseline_v050_experiment.txt`` is the experiment runner's
output (stdout and stderr) with no preset, and
``tests/fixtures/baseline_v050_assessment.json`` is the seeded assessment CLI's
report; ``tests/fixtures/baseline_v050_env.txt`` records the environment they
were captured in.

What each test compares and needs:

- The experiment test holds every metric line the runner prints, as the
  printed string (only ``processing_ms`` and ``total_seconds`` are left
  out). It needs pymdp, which the inference module imports, and PyPhi:
  without PyPhi the tracker path falls back to the calculator and the
  ``phi_source_used`` line changes.
- The assessment test holds every field of the report (statuses, counts,
  keys and booleans exactly, floats within 1e-9). Left out: the session id,
  the timestamp and the timing fields. The seeded CLI pins the introspection
  cooldown at 0.0, which removes the wall-clock gate, so the three counter
  readouts are compared in full. One residual clock term is on record: on
  a machine 25 times slower one evidence value
  (``genuine_implementation.evidence.avg_loop_quality``) moves by 1.2e-9;
  the measurement is in ``docs/MIGRATION_FROM_V04.md`` under "What the
  cooldown pin removes". It needs pymdp only: a five-cycle warmup leaves
  the tracker warming up, so the report is the same with and without
  PyPhi.

When a comparison fails, the message lists every differing line or field and
then how the running environment differs from the recorded one.

A deliberate default change regenerates both fixtures with

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
import subprocess
import sys
from pathlib import Path

import pytest

from tests.baseline_compare import (
    assessment_diff,
    current_environment,
    diff_blocks,
    explain,
    metric_block,
    recorded_environment,
    runner_environment,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXPERIMENT_BASELINE = FIXTURES / "baseline_v050_experiment.txt"
ASSESSMENT_BASELINE = FIXTURES / "baseline_v050_assessment.json"
ENVIRONMENT_RECORD = FIXTURES / "baseline_v050_env.txt"


def _skip_without_pymdp() -> None:
    pytest.importorskip("pymdp")


def _skip_without_pyphi() -> None:
    os.environ.setdefault("PYPHI_WELCOME_OFF", "yes")
    pytest.importorskip("pyphi")


def _failure_message(differences) -> str:
    return explain(
        differences,
        recorded_environment(ENVIRONMENT_RECORD.read_text()),
        current_environment(),
    )


@pytest.mark.slow
def test_v050_defaults_reproduce_fixture(tmp_path):
    """The experiment runner without a preset prints the fixture's lines."""
    _skip_without_pymdp()
    _skip_without_pyphi()
    expected = metric_block(EXPERIMENT_BASELINE.read_text())
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=runner_environment(ROOT, tmp_path),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    differences = diff_blocks(expected, metric_block(result.stdout))
    assert differences == [], _failure_message(differences)


@pytest.mark.slow
def test_v050_assessment_fixture_reproduces(tmp_path):
    """The seeded assessment CLI lands on the v0.5.0 assessment fixture."""
    _skip_without_pymdp()
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
        env=runner_environment(ROOT, tmp_path),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(json_path.read_text())
    # The seeded CLI pins the introspection cooldown at 0.0, which removes the
    # wall-clock gate, so every field is compared (one recorded residual at a
    # clock 25 times slower: see the module docstring)
    differences = assessment_diff(baseline, report)
    assert differences == [], _failure_message(differences)
