#!/usr/bin/env python3
"""
Run Consciousness Assessment - CLI
====================================

Command-line tool to run the full multi-theory consciousness
assessment framework.

Modes:
    --quick     Minimal assessment with freshly initialized modules (default)
    --full      Run several consciousness cycles first to warm up modules
    --report    Print a detailed Markdown-formatted report
    --ablation  Run ablation study (disable one component at a time)
    --json      Write results to a JSON file in results/

Usage:
    python scripts/run_assessment.py
    python scripts/run_assessment.py --full --report
    python scripts/run_assessment.py --ablation
    python scripts/run_assessment.py --full --json
    python scripts/run_assessment.py --preset legacy_v04 --full  # v0.4.0 behaviour
    python scripts/run_assessment.py --params my.toml            # parameter overrides

Overrides are optional here: with neither --preset nor --params the modules
run on their defaults. --preset NAME loads experiments/presets/NAME.toml on
top of the defaults, and --params PATH loads on top of the preset.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
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
    ActiveInferenceConfig,
    ActiveInferenceModule,
)
from mtc.consciousness.beautiful_loop import BeautifulLoop
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.core.param_overrides import deep_merge, load_overrides, preset_path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "experiments" / "param_schema.toml"
PRESETS_DIR = ROOT / "experiments" / "presets"

# fep.* behaviour switches forwarded to the workspace's own inference module.
# The generative-model geometry and learning rates stay as the workspace sets
# them; the runner's separate module takes every fep.* key.
_FEP_BEHAVIOUR_SWITCHES = (
    "hpp_hidden_dim_multiplier",
    "hierarchy_observation",
    "meta_accuracy_method",
    "drive_model",
    "drive_decay_floor",
    "legacy_double_decay",
)

# gwt.* behaviour switches that are EnhancedGlobalWorkspace constructor fields.
_GWT_SWITCHES = (
    "competition_mode",
    "thalamic_gate_enabled",
    "theory_interactions_enabled",
    "legacy_coverage_ratio",
    "body_metrics_enabled",
    "precision_feedback_enabled",
    "self_prediction_enabled",
)

_DEFAULT_WARMUP_PROFILE = "v05"


# ── helpers ──────────────────────────────────────────────────────────


def _make_candidates():
    """Generate a batch of workspace candidates for warmup cycles."""
    return [
        WorkspaceCandidate(
            content=f"stimulus_{i}",
            content_type="perception",
            summary=f"Synthetic stimulus {i}",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=np.random.uniform(0.3, 0.9),
            emotional_salience=np.random.uniform(0.0, 0.6),
            novelty_score=np.random.uniform(0.1, 0.8),
        )
        for i in range(5)
    ]


def _attention_shift_type(profile, cycle_idx):
    """Shift type the warmup assigns to the attention focus on a cycle.

    v04 only ever reported captured shifts. v05 alternates chosen and pulled
    shifts, every second chosen one goal-driven, so typed provenance has real
    chosen shifts to count.
    """
    if profile == "v04":
        return AttentionShiftType.CAPTURED
    if cycle_idx % 2 == 1:
        return AttentionShiftType.CAPTURED
    if cycle_idx % 4 == 2:
        return AttentionShiftType.GOAL_DRIVEN
    return AttentionShiftType.VOLUNTARY


async def warmup(
    workspace,
    attention,
    metacognition,
    active_inference,
    cycles=5,
    profile=_DEFAULT_WARMUP_PROFILE,
):
    """Run several consciousness cycles to populate module history.

    ``profile`` is the runner.warmup_profile switch: "v04" feeds the attention
    schema captured shifts only, "v05" alternates chosen and pulled shifts.
    """
    print(f" Warming up modules ({cycles} cycles) ...", flush=True)
    for i in range(cycles):
        candidates = _make_candidates()
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
                shift_type=_attention_shift_type(profile, i),
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

    print(" Warmup complete.\n")


# ── assessment runners ───────────────────────────────────────────────


async def run_standard(
    workspace,
    attention,
    metacognition,
    active_inference,
    beautiful_loop,
    assessment_kwargs=None,
):
    """Run a single assessment and return the report."""
    assessment = ConsciousnessAssessment(**(assessment_kwargs or {}))
    return await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
    )


async def run_ablation(
    workspace,
    attention,
    metacognition,
    active_inference,
    beautiful_loop,
    assessment_kwargs=None,
):
    """Disable one module at a time and compare scores."""
    assessment = ConsciousnessAssessment(**(assessment_kwargs or {}))

    components = ["gwt", "ast", "hot", "fep", "beautiful_loop"]
    results = {}

    # Baseline (all enabled)
    baseline = await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
    )
    results["baseline"] = baseline.overall_score

    for component in components:
        config = {c: True for c in components}
        config[component] = False
        report = await assessment.run_full_assessment(
            global_workspace=workspace if config["gwt"] else None,
            attention_schema=attention if config["ast"] else None,
            metacognition=metacognition if config["hot"] else None,
            active_inference=active_inference if config["fep"] else None,
            beautiful_loop=beautiful_loop if config["beautiful_loop"] else None,
            ablation_config=config,
        )
        results[f"without_{component}"] = report.overall_score

    return results


# ── display helpers ──────────────────────────────────────────────────


def print_report(report, markdown=False):
    """Pretty-print an assessment report."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(" CONSCIOUSNESS ASSESSMENT REPORT")
    print(sep)
    print(f" Session:            {report.session_id}")
    print(f" Timestamp:          {report.timestamp.isoformat()}")
    print(f" Overall score:      {report.overall_score:.3f}")
    print(f" Architecture OK:    {report.architecture_functional}")
    print(f" Passing indicators: {report.passing_count}/{report.total_indicators}")
    print(f" Processing time:    {report.processing_time_ms:.1f} ms")

    if markdown:
        print("\n### Theory Scores\n")
        print("| Theory | Score |")
        print("|--------|-------|")
        for theory, score in sorted(report.theory_scores.items()):
            print(f"| {theory} | {score:.3f} |")

        print(f"\n### Indicators ({report.passing_count}/{report.total_indicators})\n")
        print("| Indicator | Score | Threshold | Status |")
        print("|-----------|-------|-----------|--------|")
        for name, r in report.indicator_results.items():
            status = "PASS" if r.passes_threshold else "FAIL"
            print(f"| {name} | {r.score:.3f} | {r.threshold:.2f} | {status} |")
    else:
        print("\n  Theory Scores:")
        for theory, score in sorted(report.theory_scores.items()):
            bar = "#" * int(score * 20)
            print(f"   {theory:<32s} {score:.3f}  {bar}")

        print(f"\n  Indicators ({report.passing_count}/{report.total_indicators}):")
        for name, r in report.indicator_results.items():
            tag = "PASS" if r.passes_threshold else "FAIL"
            print(f"   [{tag}] {name:<30s}  {r.score:.3f} / {r.threshold:.2f}")

    print(f"\n{sep}\n")


def print_ablation(results):
    """Print ablation study results."""
    print("\n" + "=" * 64)
    print(" ABLATION STUDY")
    print("=" * 64)
    baseline = results.get("baseline", 0)
    print(f"\n  Baseline (all modules):  {baseline:.3f}\n")
    for key, score in results.items():
        if key == "baseline":
            continue
        delta = score - baseline
        direction = "+" if delta >= 0 else ""
        print(f"   {key:<28s}  {score:.3f}  ({direction}{delta:.3f})")
    print("\n" + "=" * 64 + "\n")


# ── main ─────────────────────────────────────────────────────────────


def _load_overrides(params, preset):
    """Validated overrides: the preset first, then the params file (both optional)."""
    schema = str(SCHEMA_PATH)
    overrides = {}
    if preset:
        overrides = load_overrides(preset_path(str(PRESETS_DIR), preset), schema)
    if params:
        overrides = deep_merge(overrides, load_overrides(params, schema))
    return overrides


def _build_phi_tracker(phi_ov):
    """PhiTracker from the phi.* keys (scale, warmup_reference)."""
    kwargs = {}
    if "scale" in phi_ov:
        kwargs["phi_scale"] = phi_ov["scale"]
    if "warmup_reference" in phi_ov:
        kwargs["phi_warmup_reference"] = phi_ov["warmup_reference"]
    return PhiTracker(**kwargs)


async def main():
    parser = argparse.ArgumentParser(
        description="Run the multi-theory consciousness assessment."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Warm up modules with several cycles before assessment",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print Markdown-formatted report",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study (disable one component at a time)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write results to results/ as JSON",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=5,
        help="Number of warmup cycles for --full mode (default: 5)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Parameter overrides TOML applied on top of the preset "
        "(default: none, modules use their defaults)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name from experiments/presets/ applied before --params "
        "(e.g., legacy_v04 reproduces v0.4.0)",
    )
    args = parser.parse_args()

    try:
        overrides = _load_overrides(args.params, args.preset)
    except (ValueError, OSError) as exc:
        print(f"PARAM ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Initialize modules. The workspace builds its own inference, beautiful
    # loop, damasio and metacognition modules; the matching overrides reach
    # them through its constructor so a preset switches every instance.
    fep_ov = {k: v for k, v in overrides.get("fep", {}).items() if k != "drives"}
    gwt_ov = overrides.get("gwt", {})
    fep_forward = {k: v for k, v in fep_ov.items() if k in _FEP_BEHAVIOUR_SWITCHES}
    workspace = EnhancedGlobalWorkspace(
        capacity=gwt_ov.get("capacity", 7),
        ignition_threshold=gwt_ov.get("ignition_threshold", 0.3),
        active_inference_config=(
            EnhancedGlobalWorkspace.default_active_inference_config(**fep_forward)
        ),
        beautiful_loop_kwargs=overrides.get("blt", {}),
        damasio_kwargs=overrides.get("damasio", {}),
        metacognition_kwargs=overrides.get("hot", {}),
        phi_tracker=_build_phi_tracker(overrides.get("phi", {})),
        **{k: gwt_ov[k] for k in _GWT_SWITCHES if k in gwt_ov},
    )
    attention = AttentionSchemaModule(**overrides.get("ast", {}))
    metacognition_mod = MetacognitionModule(**overrides.get("hot", {}))
    active_inference = ActiveInferenceModule(
        config=ActiveInferenceConfig(**fep_ov) if fep_ov else None
    )
    beautiful_loop = BeautifulLoop(**{"num_levels": 3, **overrides.get("blt", {})})
    profile = overrides.get("runner", {}).get("warmup_profile", _DEFAULT_WARMUP_PROFILE)

    if args.full:
        await warmup(
            workspace,
            attention,
            metacognition_mod,
            active_inference,
            cycles=args.warmup_cycles,
            profile=profile,
        )

    if args.ablation:
        results = await run_ablation(
            workspace,
            attention,
            metacognition_mod,
            active_inference,
            beautiful_loop,
            assessment_kwargs=overrides.get("assessment", {}),
        )
        print_ablation(results)
        return

    report = await run_standard(
        workspace,
        attention,
        metacognition_mod,
        active_inference,
        beautiful_loop,
        assessment_kwargs=overrides.get("assessment", {}),
    )
    print_report(report, markdown=args.report)

    if args.json:
        out_dir = Path("results/consciousness_assessments")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"assessment_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f" Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
