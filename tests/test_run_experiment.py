import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.consciousness.attention_schema import AttentionShiftType
from tests.baseline_compare import (
    diff_blocks,
    format_differences,
    metric_block,
    runner_environment,
)

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


class _Report:
    """The fields print_report and phi_source_used read, with no Phi source."""

    def __init__(self, overall):
        self.overall_score = overall
        self.indicator_results = {}
        self.status = "insufficient_data" if overall is None else "measured"
        self.coverage = 0.0
        self.theory_scores = {}
        self.processing_time_ms = 1.0
        self.phi_measurement = None


class _NoneAssessment:
    """A stand-in assessment whose report measured nothing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run_full_assessment(self, **kwargs):
        return _Report(None)


def test_ablation_printer_shows_insufficient_data_for_none(capsys):
    """A None ablated score prints insufficient_data, never 0.000000, and the
    ablation cannot pass on it (K2)."""
    runner = _load_runner()
    runner.print_report(
        _Report(0.25),
        1.0,
        {
            "gwt": {
                "score_with": 0.25,
                "score_without": None,
                "drop": None,
                "genuine": None,
            },
            "ast": {
                "score_with": 0.25,
                "score_without": 0.2,
                "drop": 0.05,
                "genuine": True,
            },
        },
    )
    out = capsys.readouterr().out
    assert (
        " gwt: score_with=0.250000 score_without=insufficient_data "
        "drop=insufficient_data INSUFFICIENT" in out
    )
    assert (
        " ast: score_with=0.250000 score_without=0.200000 drop=0.050000 GENUINE" in out
    )
    assert "ablation_passed:   false" in out
    assert "0.000000" not in out.split("ablation_results:")[1]


def test_run_ablation_keeps_a_none_ablated_score(monkeypatch):
    """An ablated run that measured nothing reports None, not a full drop."""
    runner = _load_runner()

    async def no_warmup(*args, **kwargs):
        return None

    monkeypatch.setattr(runner, "build_modules", lambda overrides: {})
    monkeypatch.setattr(runner, "build_augmenter", lambda overrides, modules: None)
    monkeypatch.setattr(runner, "warmup", no_warmup)
    monkeypatch.setattr(runner, "ConsciousnessAssessment", _NoneAssessment)
    results = asyncio.run(runner.run_ablation({}, ["gwt"], 0.2, {}))
    assert results["gwt"] == {
        "score_with": 0.2,
        "score_without": None,
        "drop": None,
        "genuine": None,
    }


def test_ablation_is_skipped_when_the_baseline_is_none(monkeypatch, capsys):
    """With a baseline that measured nothing the ablation is skipped with one
    plain line instead of comparing against 0.0."""
    runner = _load_runner()
    calls = []

    async def no_warmup(*args, **kwargs):
        return None

    async def no_baseline(overrides, iterations=5):
        return {}

    async def record_ablation(*args, **kwargs):
        calls.append(args)
        return {}

    monkeypatch.setattr(runner, "build_modules", lambda overrides: {})
    monkeypatch.setattr(runner, "build_augmenter", lambda overrides, modules: None)
    monkeypatch.setattr(runner, "warmup", no_warmup)
    monkeypatch.setattr(runner, "_compute_noise_baseline", no_baseline)
    monkeypatch.setattr(runner, "ConsciousnessAssessment", _NoneAssessment)
    monkeypatch.setattr(runner, "run_ablation", record_ablation)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--validate", "gwt"])
    asyncio.run(runner.run())
    captured = capsys.readouterr()
    assert calls == []
    assert "overall_score:     insufficient_data" in captured.out
    assert "ablation skipped" in captured.err
    assert "insufficient_data" in captured.err
    assert "ablation_passed" not in captured.out


def _run_runner(tmp_path) -> "subprocess.CompletedProcess[str]":
    """The experiment runner from the repository root, on the CPU."""
    return subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=runner_environment(ROOT, tmp_path),
    )


@pytest.mark.slow
def test_run_experiment_baseline(tmp_path):
    """Baseline experiment (no overrides) should complete and print results."""
    result = _run_runner(tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "overall_score:" in result.stdout
    assert "passing_count:" in result.stdout
    assert "gwt_score:" in result.stdout


@pytest.mark.slow
def test_run_experiment_deterministic(tmp_path):
    """Two runs with the same params print the same metric lines."""
    blocks = []
    for run in range(2):
        result = _run_runner(tmp_path / f"run{run}")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        blocks.append(metric_block(result.stdout))
    assert "overall_score" in blocks[0]
    differences = diff_blocks(blocks[0], blocks[1])
    assert differences == [], "Non-deterministic:\n" + format_differences(differences)
