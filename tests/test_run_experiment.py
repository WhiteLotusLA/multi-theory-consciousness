import subprocess
import sys


def test_run_experiment_baseline():
    """Baseline experiment (no overrides) should complete and print results."""
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "overall_score:" in result.stdout
    assert "passing_count:" in result.stdout
    assert "gwt_score:" in result.stdout


def test_run_experiment_deterministic():
    """Two runs with same params produce identical scores."""
    scores = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "scripts/run_experiment.py"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        for line in result.stdout.splitlines():
            if line.startswith("overall_score:"):
                scores.append(line.strip())
    assert len(scores) == 2
    assert scores[0] == scores[1], f"Non-deterministic: {scores}"
