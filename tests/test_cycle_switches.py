"""The cycle switches move the runners' numbers only by opt-in."""

import asyncio
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

from mtc.consciousness.enhanced_global_workspace import WorkspaceCandidateSource
from mtc.core.param_overrides import load_overrides
from mtc.pipeline.cycle_augmenter import CycleAugmenterConfig

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "experiments" / "param_schema.toml"
LEGACY = ROOT / "experiments" / "presets" / "legacy_v04.toml"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_script_run_experiment_cycle", ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)


def test_runner_pins_cycle_switches_off_unless_set():
    runner = _load_runner()
    modules = runner.build_modules({})
    augmenter = runner.build_augmenter({}, modules)
    assert augmenter.config == CycleAugmenterConfig.all_off()
    assert augmenter.workspace is modules["global_workspace"]
    assert augmenter.pad_model is modules["pad_affect_model"]

    opted_overrides = {"cycle": {"pad_from_state": True}}
    opted = runner.build_augmenter(
        opted_overrides, runner.build_modules(opted_overrides)
    )
    assert opted.config.pad_from_state is True
    assert opted.config.drive_candidates is False
    assert opted.config.gate_arousal == "workspace"
    # The assessment never sees the augmenter: the modules dict stays
    # splattable into run_full_assessment, and an ablated workspace has no
    # cycle to wrap.
    assert "cycle_augmenter" not in modules
    assert runner.build_augmenter({}, {**modules, "global_workspace": None}) is None


def test_legacy_preset_rows_turn_cycle_off():
    overrides = load_overrides(str(LEGACY), str(SCHEMA))
    assert overrides["cycle"] == {
        "drive_candidates": False,
        "prospective_candidates": False,
        "crossmodal_binding": False,
        "gate_arousal": "workspace",
        "pad_from_state": False,
        "candidate_cap": 0,
    }
    config = CycleAugmenterConfig.from_overrides(overrides["cycle"])
    assert config == CycleAugmenterConfig.all_off()


def test_runner_with_drive_candidates_injects_homeostatic():
    runner = _load_runner()
    _seed()
    overrides = {"cycle": {"drive_candidates": True}}
    modules = runner.build_modules(overrides)
    augmenter = runner.build_augmenter(overrides, modules)
    workspace = modules["global_workspace"]
    # A fresh drive set sits at its set points; deplete one so it is urgent
    # above the default threshold when the warmup starts.
    workspace.active_inference.homeostatic_drives.drives["safety"].current_level = 0.0
    asyncio.run(runner.warmup(modules, cycles=4, profile="v05", augmenter=augmenter))
    winning_sources = {
        source
        for event in workspace.bottleneck.competition_history
        for source in event.winning_sources
    }
    assert WorkspaceCandidateSource.HOMEOSTATIC.value in winning_sources
    assert augmenter.status()["totals"]["drive_candidates"] > 0
    # Without the switch the same urgent drive never competes
    _seed()
    plain = runner.build_modules({})
    plain["global_workspace"].active_inference.homeostatic_drives.drives[
        "safety"
    ].current_level = 0.0
    asyncio.run(
        runner.warmup(
            plain, cycles=4, profile="v05", augmenter=runner.build_augmenter({}, plain)
        )
    )
    plain_sources = {
        source
        for event in plain["global_workspace"].bottleneck.competition_history
        for source in event.winning_sources
    }
    assert WorkspaceCandidateSource.HOMEOSTATIC.value not in plain_sources


def _metrics(modules):
    """Seed-determined numbers of a warmup. Wall-clock terms (the recency
    bonus inside the salience, the ignition duration) are left out because
    two plain runs differ in them too."""
    workspace = modules["global_workspace"]
    stats = workspace.get_statistics()
    bottleneck = stats["bottleneck"]
    ignition = stats["ignition"]
    pad = modules["pad_affect_model"]
    return {
        "cycles": stats["cycle_count"],
        "competitions": bottleneck.get("total_competitions"),
        "winning_sources": bottleneck.get("source_distribution"),
        "avg_candidates": bottleneck.get("avg_candidates"),
        "avg_winners": bottleneck.get("avg_winners"),
        "ignitions": ignition.get("total_ignitions"),
        "sustained": ignition.get("sustained_count"),
        "winners": [
            event.winning_sources for event in workspace.bottleneck.competition_history
        ],
        "stream": [c.candidate.summary for c in workspace.stream_of_consciousness],
        "pad": [
            (round(s.pleasure, 12), round(s.arousal, 12), round(s.dominance, 12))
            for s in pad._history
        ],
        "arousal": workspace.bottleneck.arousal,
        "rpt": modules["rpt_measurement"].measurement_count,
        "random_tail": (random.random(), float(np.random.rand())),
    }


def test_runner_numbers_identical_with_augmenter_off():
    runner = _load_runner()
    _seed()
    with_hooks = runner.build_modules({})
    augmenter = runner.build_augmenter({}, with_hooks)
    assert augmenter.config == CycleAugmenterConfig.all_off()
    asyncio.run(runner.warmup(with_hooks, cycles=6, profile="v05", augmenter=augmenter))
    hooked = _metrics(with_hooks)
    assert augmenter.status()["before_calls"] == 6

    _seed()
    without_hooks = runner.build_modules({})
    asyncio.run(runner.warmup(without_hooks, cycles=6, profile="v05"))
    plain = _metrics(without_hooks)

    assert hooked == plain
