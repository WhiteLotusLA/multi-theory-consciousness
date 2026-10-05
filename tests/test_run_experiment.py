import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.consciousness.attention_schema import AttentionShiftType

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_script_run_experiment", ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_warmup_profiles_set_shift_types():
    """v04 feeds captured shifts only; v05 rotates the types (fast)."""
    runner = _load_runner()
    seen = {}
    for profile, expected in (
        ("v04", AttentionShiftType.CAPTURED),
        ("v05", AttentionShiftType.GOAL_DRIVEN),
    ):
        built = runner.build_modules({})
        # Only the workspace and the attention schema are fed: the other
        # modules are left out the way an ablation leaves them out
        modules = {
            "global_workspace": built["global_workspace"],
            "attention_schema": built["attention_schema"],
        }
        # Three cycles: the last one runs at index 2, where the profiles differ
        asyncio.run(runner.warmup(modules, cycles=3, profile=profile))
        seen[profile] = modules["attention_schema"].schema.current_focus.shift_type
        assert seen[profile] == expected
    assert seen["v04"] != seen["v05"]


def test_runner_exposes_tracker_and_attached_oscillator():
    """The workspace's tracker reaches the assessment; RPT reads its binding."""
    modules = _load_runner().build_modules({})
    assert modules["phi_tracker"] is modules["global_workspace"].phi_tracker
    assert modules["rpt_measurement"].oscillatory_binding is not None


def test_runner_forwards_ast_overrides_to_the_owned_schema():
    modules = _load_runner().build_modules({"ast": {"focus_threshold": 0.55}})
    assert modules["global_workspace"].attention_schema.focus_threshold == 0.55


@pytest.mark.slow
def test_run_experiment_baseline():
    """Baseline experiment (no overrides) should complete and print results."""
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "overall_score:" in result.stdout
    assert "passing_count:" in result.stdout
    assert "gwt_score:" in result.stdout


@pytest.mark.slow
def test_run_experiment_deterministic():
    """Two runs with same params produce identical scores."""
    scores = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "scripts/run_experiment.py"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0
        for line in result.stdout.splitlines():
            if line.startswith("overall_score:"):
                scores.append(line.strip())
    assert len(scores) == 2
    assert scores[0] == scores[1], f"Non-deterministic: {scores}"
