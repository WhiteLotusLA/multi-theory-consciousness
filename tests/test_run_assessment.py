"""Tests for scripts/run_assessment.py: warmup profiles, seeding, state input."""

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.consciousness.attention_schema import AttentionShiftType
from mtc.consciousness.state_persistence import ConsciousnessStateManager
from tests.baseline_compare import (
    assessment_diff,
    format_differences,
    runner_environment,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_cli(scratch: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "scripts/run_assessment.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=runner_environment(ROOT, scratch),
    )


def test_cli_warmup_profiles_set_shift_types():
    """The v04 profile feeds captured shifts only; v05 rotates the types."""
    script = _load_script("run_assessment")
    seen = {}
    for profile, expected in (
        ("v04", AttentionShiftType.CAPTURED),
        ("v05", AttentionShiftType.GOAL_DRIVEN),
    ):
        script._seed_everything(3)
        workspace, attention, hot, fep, _ = script._build_standalone_modules({})
        # Three cycles: the last one runs at index 2, where the profiles differ
        asyncio.run(script.warmup(workspace, attention, hot, fep, 3, profile))
        seen[profile] = attention.schema.current_focus.shift_type
        assert seen[profile] == expected
    assert seen["v04"] != seen["v05"]


@pytest.mark.slow
def test_seeded_assessment_is_reproducible(tmp_path):
    """Two runs of one seed agree in every field of the report.

    Three warmup cycles complete exactly one introspection pass (the first,
    which no cooldown can block), so no field of this report follows the
    wall clock and nothing is left out of the comparison.
    """
    reports = []
    for run in range(2):
        out = tmp_path / f"run{run}.json"
        result = _run_cli(
            tmp_path,
            "--full",
            "--warmup-cycles",
            "3",
            "--seed",
            "7",
            "--json",
            "--json-path",
            str(out),
        )
        assert result.returncode == 0, result.stderr[-2000:]
        reports.append(json.loads(out.read_text()))
    assert reports[0]["overall_score"] is not None
    assert len(reports[0]["indicator_results"]) == 30
    differences = assessment_diff(reports[0], reports[1])
    assert differences == [], format_differences(differences)


@pytest.mark.slow
def test_state_dir_restores_and_assesses(tmp_path):
    pytest.importorskip("pyphi")
    experiment = _load_script("run_experiment")
    modules = experiment.build_modules({})
    asyncio.run(experiment.warmup(modules, cycles=3))

    from mtc.assessment.causal_emergence import CausalEmergenceAnalyzer

    rpt = modules["rpt_measurement"]
    hub = ConsciousnessStateManager(
        tmp_path / "state",
        {
            "gwt": modules["global_workspace"],
            "rpt": rpt,
            "phi": modules["phi_tracker"],
            "causal_emergence": CausalEmergenceAnalyzer(),
            "pad_affect": modules["pad_affect_model"],
            "oscillatory_binding": rpt.oscillatory_binding,
        },
    )
    hub.save_all()

    # An empty state directory restores nothing, so the same indicators have no
    # evidence: that is what the restored state changes
    empty = tmp_path / "empty"
    empty.mkdir()
    fresh_out = tmp_path / "fresh.json"
    fresh = _run_cli(tmp_path, "--state-dir", str(empty), "--json-path", str(fresh_out))
    assert fresh.returncode == 0, fresh.stderr[-2000:]
    fresh_status = json.loads(fresh_out.read_text())["indicator_results"]
    assert fresh_status["global_broadcast"]["status"] == "insufficient_data"

    out = tmp_path / "report.json"
    result = _run_cli(
        tmp_path,
        "--state-dir",
        str(tmp_path / "state"),
        "--json-path",
        str(out),
        "--seed",
        "5",
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for name in ("gwt", "rpt", "phi", "pad_affect", "oscillatory_binding"):
        line = next(
            ln for ln in result.stdout.splitlines() if ln.strip()[:30].startswith(name)
        )
        assert line.rstrip().endswith("restored"), line

    report = json.loads(out.read_text())
    indicators = report["indicator_results"]
    # Indicators fed by the restored workspace and RPT history are measured;
    # affect coherence needs ten recorded affect states, so three cycles leave
    # it reporting insufficient_data rather than a number
    for name in (
        "global_broadcast",
        "ignition_dynamics",
        "recurrent_processing",
    ):
        assert indicators[name]["status"] == "measured", name
        assert indicators[name]["score"] is not None
    assert indicators["affect_coherence"]["status"] == "insufficient_data"
    assert indicators["affect_coherence"]["score"] is None
