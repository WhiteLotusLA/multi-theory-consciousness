#!/usr/bin/env python3
"""
AutoConsciousness experiment runner.

Loads parameter overrides from experiments/params.toml, validates against
experiments/param_schema.toml, instantiates consciousness modules with
overridden params, runs warmup + assessment, prints grep-friendly results.

Usage: python scripts/run_experiment.py > run.log 2>&1
"""

import asyncio
import random
import sys
import time
from pathlib import Path

import numpy as np

# Seed ALL RNG sources before any module imports that might use them
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass

from mtc.core.param_overrides import load_overrides
from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    TemporalWorkspaceConfig,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionTarget,
    AttentionShiftType,
)
from mtc.consciousness.metacognition import (
    MetacognitionModule,
    FirstOrderStateType,
    MetaType,
)
from mtc.consciousness.active_inference import (
    ActiveInferenceModule,
    ActiveInferenceConfig,
)
from mtc.consciousness.beautiful_loop import BeautifulLoop
from mtc.consciousness.damasio import DamasioLayers
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.neural.oscillatory_binding import OscillatoryBinding
from mtc.assessment.rpt_measurement import RPTMeasurement
from mtc.assessment.assessment import ConsciousnessAssessment

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "experiments" / "param_schema.toml"
PARAMS_PATH = ROOT / "experiments" / "params.toml"
WARMUP_CYCLES = 5

# Fields that belong to TemporalWorkspaceConfig, not EnhancedGlobalWorkspace
_TEMPORAL_FIELDS = {
    "competition_duration_ms",
    "ignition_threshold_time_ms",
    "broadcast_propagation_ms",
    "refractory_period_ms",
    "timestep_ms",
}


def build_modules(overrides: dict) -> dict:
    """Instantiate all consciousness modules, applying TOML overrides."""

    gwt_ov = overrides.get("gwt", {}).copy()

    # Separate temporal config fields from top-level GWT fields
    temporal_ov = {}
    for key in list(gwt_ov.keys()):
        if key in _TEMPORAL_FIELDS:
            temporal_ov[key] = gwt_ov.pop(key)
        elif key == "temporal" and isinstance(gwt_ov[key], dict):
            temporal_ov.update(gwt_ov.pop(key))

    temporal_config = TemporalWorkspaceConfig(**temporal_ov) if temporal_ov else None
    workspace = EnhancedGlobalWorkspace(
        capacity=gwt_ov.get("capacity", 7),
        ignition_threshold=gwt_ov.get("ignition_threshold", 0.3),
        amplification_factor=gwt_ov.get("amplification_factor", 2.0),
        temporal_config=temporal_config,
    )

    ast_ov = overrides.get("ast", {})
    attention = AttentionSchemaModule(**ast_ov)

    hot_ov = overrides.get("hot", {})
    metacognition = MetacognitionModule(**hot_ov)

    fep_ov = overrides.get("fep", {}).copy()
    fep_ov.pop("drives", None)  # drives handled separately (not yet wired)
    fep_config = ActiveInferenceConfig(**{k: v for k, v in fep_ov.items()})
    active_inference = ActiveInferenceModule(config=fep_config)

    blt_ov = overrides.get("blt", {})
    beautiful_loop = BeautifulLoop(**blt_ov)

    damasio_ov = overrides.get("damasio", {})
    damasio = DamasioLayers(**damasio_ov)

    pad_ov = overrides.get("pad", {})
    pad_affect = PADAffectModel(**pad_ov)

    osc_ov = overrides.get("oscillatory", {})
    oscillatory = OscillatoryBinding(**osc_ov)

    rpt_ov = overrides.get("rpt", {})
    rpt = RPTMeasurement(**rpt_ov)

    return {
        "global_workspace": workspace,
        "attention_schema": attention,
        "metacognition": metacognition,
        "active_inference": active_inference,
        "beautiful_loop": beautiful_loop,
        "rpt_measurement": rpt,
        "neural_states": _build_neural_states(),
    }


def _build_neural_states() -> dict:
    """Create synthetic neural states for assessment."""
    return {
        "snn_spikes": np.random.rand(8, 8).astype(np.float32),
        "lsm_states": np.random.rand(8, 8).astype(np.float32),
        "htm_columns": np.random.rand(8, 8).astype(np.float32),
    }


def _make_candidates(cycle_idx):
    """Generate workspace candidates for warmup. Matches run_assessment.py API."""
    return [
        WorkspaceCandidate(
            content=f"stimulus_{cycle_idx}_{j}",
            content_type="perception",
            summary=f"Synthetic stimulus {cycle_idx}_{j}",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=np.random.uniform(0.3, 0.9),
            emotional_salience=np.random.uniform(0.0, 0.6),
            novelty_score=np.random.uniform(0.1, 0.8),
        )
        for j in range(5)
    ]


async def warmup(modules: dict, cycles: int = WARMUP_CYCLES):
    """Run consciousness cycles to fill statistics buffers.
    Mirrors scripts/run_assessment.py warmup logic."""
    workspace = modules["global_workspace"]
    attention = modules["attention_schema"]
    metacognition = modules["metacognition"]
    active_inference = modules["active_inference"]

    for i in range(cycles):
        # GWT competition cycle
        candidates = _make_candidates(i)
        state = await workspace.process_consciousness_cycle(candidates)

        # Feed AST
        if state.primary_content:
            target = AttentionTarget(
                content=state.primary_content.candidate.content,
                content_type=state.primary_content.candidate.content_type,
                summary=state.primary_content.candidate.summary,
                source_module="global_workspace",
                attention_strength=state.primary_content.salience,
                salience=state.primary_content.salience,
                shift_type=AttentionShiftType.CAPTURED,
            )
            attention.schema.current_focus = target

        # Feed HOT
        if state.primary_content:
            fo = metacognition.register_first_order_state(
                content=state.primary_content.candidate.content,
                content_summary=state.primary_content.candidate.summary,
                state_type=FirstOrderStateType.PERCEPTION,
                source_module="global_workspace",
            )
            await metacognition.generate_hot(fo, MetaType.AWARENESS)

        # Feed FEP
        obs = np.random.dirichlet(np.ones(5))
        await active_inference.infer_and_act(obs)


async def run():
    t_start = time.time()

    # Load and validate overrides
    try:
        overrides = load_overrides(str(PARAMS_PATH), str(SCHEMA_PATH))
    except ValueError as e:
        print(f"PARAM ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if overrides:
        print(f"Overrides: {overrides}", file=sys.stderr)

    # Build modules with overrides
    modules = build_modules(overrides)

    # Warmup
    await warmup(modules, WARMUP_CYCLES)

    # Run assessment
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)

    # Print grep-friendly output
    t_total = time.time() - t_start
    print("---")
    print(f"overall_score:     {report.overall_score:.6f}")
    print(f"passing_count:     {sum(1 for r in report.indicator_results.values() if r.passes_threshold)}")
    print(f"total_indicators:  {len(report.indicator_results)}")
    for theory_name, theory_score in sorted(report.theory_scores.items()):
        print(f"{theory_name.lower()}_score:{'':>{13 - len(theory_name)}}{theory_score:.6f}")
    print(f"processing_ms:     {report.processing_time_ms:.1f}")
    print(f"warmup_cycles:     {WARMUP_CYCLES}")
    print(f"total_seconds:     {t_total:.1f}")


if __name__ == "__main__":
    asyncio.run(run())
