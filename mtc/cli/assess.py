"""
Consciousness Assessment Runner (``mtc assess``)
================================================

Runs the full multi-theory consciousness assessment. This is the body of
``scripts/run_assessment.py``, moved into the package so an installed ``mtc``
command runs it from any directory; the script is now a shim over ``main``
and both give the same results for the same seed.

Modes:
    --quick     Minimal assessment with freshly initialized modules (default)
    --full      Run several consciousness cycles first to warm up modules
    --report    Print a detailed Markdown-formatted report
    --ablation  Run ablation study (disable one component at a time)
    --json      Write results to a JSON file under ASSESSMENT_OUTPUT_DIR
    --json-path Write the JSON report to this path instead (implies --json)
    --seed      Seed random, numpy and torch before any module is built (42)
    --state-dir Restore persisted module state from this directory, then assess

State directory (--state-dir): the modules are built, registered with the
ConsciousnessStateManager under six names, restored with restore_all() and then
assessed, using the workspace's own attention schema, metacognition, active
inference and beautiful loop as the AST, HOT, FEP and BLT inputs (gwt, rpt,
phi, causal_emergence, pad_affect, oscillatory_binding).

Without --state-dir (the standalone path) the workspace's PhiTracker reaches
the assessment in every mode, so a tracker that has not reported yet reads
insufficient_data rather than absent. The beautiful loop it assesses is
chosen by ``runner.assess_blt_source``: the workspace's own loop, which the
warmup drives (workspace, the default), or the separate loop nothing drives
(standalone, the behaviour before v0.5.0). The introspection cooldown of
every metacognition module built here is pinned at 0.0 unless
``hot.introspection_cooldown`` is set, because it is a wall-clock gate and a
seeded run must not follow the speed of the machine.

Noise baselines: this runner passes none, so the assessment normalises
against its own stub pass; ``assessment.noise_baseline_matching`` says what
happens when that baseline was made by a different Phi producer than the
result. scripts/run_experiment.py computes its baselines from real modules.

Overrides are optional: with neither --preset nor --params the modules run on
their defaults. --preset NAME loads experiments/presets/NAME.toml on top of
the defaults, and --params PATH loads on top of the preset. The preset
directory and experiments/param_schema.toml resolve through
``mtc.core.resources.resolve_shipped``: the project copy under PROJECT_ROOT,
else the packaged copy.

On an install without the fep extra the workspace cannot be built; the
command prints one line naming the extra on standard error, with no
traceback, and exits with the usage code 2.

Config keys read: ``ASSESSMENT_OUTPUT_DIR`` (the default JSON directory,
resolved under ``PROJECT_ROOT``) and ``PROJECT_ROOT``. Files written: the JSON
report when asked. The report holds measurement values only.
"""

import argparse
import asyncio
import inspect
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionTarget,
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
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.consciousness.state_persistence import ConsciousnessStateManager
from mtc.neural.oscillatory_binding import OscillatoryBinding
from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.assessment.causal_emergence import CausalEmergenceAnalyzer
from mtc.assessment.rpt_measurement import RPTMeasurement
from mtc.cli.missing_extra import explain_missing_extra
from mtc.core.config import get_settings
from mtc.core.param_overrides import deep_merge, load_overrides, preset_path
from mtc.core.paths import resolve_path
from mtc.core.resources import resolve_shipped
from mtc.core.runner_support import (
    WARMUP_PROFILE_V05,
    attention_shift_type,
    build_cycle_augmenter,
    build_phi_tracker,
    build_workspace,
)

logger = logging.getLogger(__name__)

SCHEMA_SOURCE = "experiments/param_schema.toml"
PRESETS_SOURCE = "experiments/presets"

_DEFAULT_WARMUP_PROFILE = WARMUP_PROFILE_V05
_DEFAULT_SEED = 42

# runner.assess_blt_source: which beautiful loop the standalone path assesses
BLT_SOURCE_WORKSPACE = "workspace"
BLT_SOURCE_STANDALONE = "standalone"
BLT_SOURCES = (BLT_SOURCE_WORKSPACE, BLT_SOURCE_STANDALONE)
_DEFAULT_BLT_SOURCE = BLT_SOURCE_WORKSPACE

# The introspection cooldown is a wall-clock gate; the seeded CLI pins it at
# 0.0 on every metacognition module it builds unless hot.introspection_cooldown
# is set (the legacy preset sets 1.0), so two runs of one seed agree.
_INTROSPECTION_COOLDOWN_PIN = 0.0

# The exit code of a configuration error, shared by the mtc commands
# (mtc.cli.main.EXIT_USAGE); a missing extra is one.
EXIT_USAGE = 2

# The PhiTracker reports no Phi before this many transitions. The CLI keeps the
# tracker's own default: a short warmup leaves it warming up, and only a
# restored long-lived tracker (--state-dir) reports.
_PHI_MIN_TRANSITIONS = (
    inspect.signature(PhiTracker).parameters["min_transitions"].default
)


# -- helpers -----------------------------------------------------


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


async def warmup(
    workspace,
    attention,
    metacognition,
    active_inference,
    cycles=5,
    profile=_DEFAULT_WARMUP_PROFILE,
    augmenter=None,
):
    """Run several consciousness cycles to populate module history.

    ``profile`` is the runner.warmup_profile switch: "v04" feeds the attention
    schema captured shifts only, "v05" alternates chosen and pulled shifts.
    ``augmenter`` is the cycle augmenter whose hooks wrap each cycle; None
    (or one with every cycle.* switch off) leaves the cycles as they were.
    """
    print(f" Warming up modules ({cycles} cycles) ...", flush=True)
    for i in range(cycles):
        candidates = _make_candidates()
        if augmenter is not None:
            candidates = augmenter.before_cycle(candidates)
        state = await workspace.process_consciousness_cycle(candidates)
        if augmenter is not None:
            augmenter.after_cycle(state)

        # Feed AST
        if state.primary_content:
            target = AttentionTarget(
                content=state.primary_content.candidate.content,
                content_type=state.primary_content.candidate.content_type,
                summary=state.primary_content.candidate.summary,
                source_module="global_workspace",
                attention_strength=state.primary_content.salience,
                salience=state.primary_content.salience,
                shift_type=attention_shift_type(i, profile),
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


# -- assessment runners -----------------------------------------


async def run_standard(
    workspace,
    attention,
    metacognition,
    active_inference,
    beautiful_loop,
    assessment_kwargs=None,
    extra_modules=None,
):
    """Run a single assessment and return the report.

    ``extra_modules`` are further run_full_assessment keywords (phi_tracker,
    rpt_measurement, pad_affect_model, causal_emergence_analyzer).
    """
    assessment = ConsciousnessAssessment(**(assessment_kwargs or {}))
    return await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
        **(extra_modules or {}),
    )


async def run_ablation(
    workspace,
    attention,
    metacognition,
    active_inference,
    beautiful_loop,
    assessment_kwargs=None,
    extra_modules=None,
):
    """Disable one module at a time and compare scores.

    The workspace's PhiTracker only has data through the workspace, so
    ablating gwt also withholds it.
    """
    assessment = ConsciousnessAssessment(**(assessment_kwargs or {}))
    extra = dict(extra_modules or {})

    components = ["gwt", "ast", "hot", "fep", "beautiful_loop"]
    results = {}

    # Baseline (all enabled)
    baseline = await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
        **extra,
    )
    results["baseline"] = baseline.overall_score

    for component in components:
        config = {c: True for c in components}
        config[component] = False
        kwargs = dict(extra)
        if component == "gwt":
            kwargs.pop("phi_tracker", None)
        report = await assessment.run_full_assessment(
            global_workspace=workspace if config["gwt"] else None,
            attention_schema=attention if config["ast"] else None,
            metacognition=metacognition if config["hot"] else None,
            active_inference=active_inference if config["fep"] else None,
            beautiful_loop=beautiful_loop if config["beautiful_loop"] else None,
            ablation_config=config,
            **kwargs,
        )
        results[f"without_{component}"] = report.overall_score

    return results


# -- display helpers --------------------------------------------


def _fmt(score, spec=".3f"):
    """A score, or insufficient_data when the assessment has none."""
    return "insufficient_data" if score is None else format(score, spec)


def _indicator_score(result):
    return None if result.status == "insufficient_data" else result.score


def print_report(report, markdown=False):
    """Pretty-print an assessment report."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(" CONSCIOUSNESS ASSESSMENT REPORT")
    print(sep)
    print(f" Session:            {report.session_id}")
    print(f" Timestamp:          {report.timestamp.isoformat()}")
    print(f" Overall score:      {_fmt(report.overall_score)}")
    print(f" Status:             {report.status}")
    print(f" Coverage:           {report.coverage:.3f}")
    print(f" Architecture OK:    {report.architecture_functional}")
    print(f" Passing indicators: {report.passing_count}/{report.total_indicators}")
    if report.insufficient_indicators:
        print(f" Insufficient data:  {', '.join(report.insufficient_indicators)}")
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
            if r.status == "insufficient_data":
                status = "INSUFFICIENT"
            print(
                f"| {name} | {_fmt(_indicator_score(r))} | {r.threshold:.2f} "
                f"| {status} |"
            )
    else:
        print("\n  Theory Scores:")
        for theory, score in sorted(report.theory_scores.items()):
            bar = "#" * int(score * 20)
            print(f"   {theory:<32s} {score:.3f}  {bar}")

        print(f"\n  Indicators ({report.passing_count}/{report.total_indicators}):")
        for name, r in report.indicator_results.items():
            tag = "PASS" if r.passes_threshold else "FAIL"
            if r.status == "insufficient_data":
                tag = "NODATA"
            print(
                f"   [{tag}] {name:<30s}  {_fmt(_indicator_score(r))} "
                f"/ {r.threshold:.2f}"
            )

    print(f"\n{sep}\n")


def print_ablation(results):
    """Print ablation study results."""
    print("\n" + "=" * 64)
    print(" ABLATION STUDY")
    print("=" * 64)
    baseline = results.get("baseline")
    print(f"\n  Baseline (all modules):  {_fmt(baseline)}\n")
    for key, score in results.items():
        if key == "baseline":
            continue
        if score is None or baseline is None:
            print(f"   {key:<28s}  {_fmt(score)}")
            continue
        delta = score - baseline
        direction = "+" if delta >= 0 else ""
        print(f"   {key:<28s}  {score:.3f}  ({direction}{delta:.3f})")
    print("\n" + "=" * 64 + "\n")


# -- main ---------------------------------------------------------


def _load_overrides(params, preset):
    """Validated overrides: the preset first, then the params file (both optional)."""
    schema = str(resolve_shipped(SCHEMA_SOURCE)[0])
    overrides = {}
    if preset:
        presets_dir = resolve_shipped(PRESETS_SOURCE)[0]
        overrides = load_overrides(preset_path(str(presets_dir), preset), schema)
    if params:
        overrides = deep_merge(overrides, load_overrides(params, schema))
    return overrides


def _seed_everything(seed):
    """Seed random, numpy and torch (when installed) before any module exists."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _build_workspace(overrides):
    """The workspace with its PhiTracker, built from the layered overrides.

    Machine metrics and the previous cycle's duration reach the protoself when
    body metrics are on, so the seeded CLI pins them off unless
    gwt.body_metrics_enabled is set.
    """
    tracker = build_phi_tracker(overrides.get("phi", {}), _PHI_MIN_TRANSITIONS)
    return build_workspace(
        overrides,
        tracker,
        body_metrics_default=False,
        introspection_cooldown_default=_INTROSPECTION_COOLDOWN_PIN,
    )


def _blt_source(overrides) -> str:
    """The runner.assess_blt_source switch: workspace unless the overrides say
    standalone."""
    source = overrides.get("runner", {}).get("assess_blt_source", _DEFAULT_BLT_SOURCE)
    if source not in BLT_SOURCES:
        raise ValueError(
            f"runner.assess_blt_source must be one of {BLT_SOURCES}, got {source!r}"
        )
    return source


def _build_standalone_modules(overrides):
    """Workspace plus the separate AST, HOT, FEP and BLT modules.

    The workspace builds its own inference, beautiful loop, damasio,
    metacognition and attention schema modules; the matching overrides reach
    them through its constructor so a preset switches every instance. The
    separate metacognition module gets the same cooldown pin as the
    workspace's own.
    """
    fep_ov = {k: v for k, v in overrides.get("fep", {}).items() if k != "drives"}
    workspace = _build_workspace(overrides)
    attention = AttentionSchemaModule(**overrides.get("ast", {}))
    metacognition = MetacognitionModule(
        **{
            "introspection_cooldown": _INTROSPECTION_COOLDOWN_PIN,
            **overrides.get("hot", {}),
        }
    )
    active_inference = ActiveInferenceModule(
        config=ActiveInferenceConfig(**fep_ov) if fep_ov else None
    )
    beautiful_loop = BeautifulLoop(**{"num_levels": 3, **overrides.get("blt", {})})
    return workspace, attention, metacognition, active_inference, beautiful_loop


def _build_persisted_modules(overrides):
    """Every module a state directory can hold, keyed by its persistence name."""
    workspace = _build_workspace(overrides)
    rpt = RPTMeasurement(**overrides.get("rpt", {}))
    oscillatory = OscillatoryBinding(**overrides.get("oscillatory", {}))
    rpt.attach_oscillatory_binding(oscillatory)
    return {
        "gwt": workspace,
        "rpt": rpt,
        "phi": workspace.phi_tracker,
        "causal_emergence": CausalEmergenceAnalyzer(
            **overrides.get("causal_emergence", {})
        ),
        "pad_affect": PADAffectModel(**overrides.get("pad", {})),
        "oscillatory_binding": oscillatory,
    }


def _restore_state(state_dir, persisted):
    """Register the six modules, restore them and print which came back."""
    hub = ConsciousnessStateManager(state_dir, dict(persisted))
    restored = hub.restore_all()
    print(f" Restoring state from {state_dir}")
    for name in persisted:
        print(f"   {name:<22s} {'restored' if restored.get(name) else 'fresh'}")
    print()
    return restored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtc assess",
        description="Run the multi-theory consciousness assessment.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Assess freshly initialized modules without warmup (the default "
        "mode; naming it changes nothing)",
    )
    mode.add_argument(
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
        help="Write results as JSON under ASSESSMENT_OUTPUT_DIR",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Write the JSON report to this file instead of a timestamped file "
        "under ASSESSMENT_OUTPUT_DIR (implies --json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Seed for random, numpy and torch (default: {_DEFAULT_SEED})",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Restore persisted module state (gwt, rpt, phi, causal_emergence, "
        "pad_affect, oscillatory_binding) from this directory, then assess",
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the runner's flags, run the assessment and return the exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s  %(name)s  %(message)s",
    )
    try:
        overrides = _load_overrides(args.params, args.preset)
    except (ValueError, OSError) as exc:
        print(f"PARAM ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        asyncio.run(_run(args, overrides))
    except ImportError as exc:
        # A missing extra is a configuration error: one plain line, no
        # traceback, the usage exit code. Any other import error propagates.
        line = explain_missing_extra(exc)
        if line is None:
            raise
        print(line, file=sys.stderr)
        return EXIT_USAGE
    return 0


async def _run(args: argparse.Namespace, overrides: dict) -> None:

    # Seed before any module is built: the modules draw from these streams
    _seed_everything(args.seed)

    extra_modules = {}
    if args.state_dir:
        persisted = _build_persisted_modules(overrides)
        _restore_state(args.state_dir, persisted)
        workspace = persisted["gwt"]
        # The workspace's own modules are the AST, HOT, FEP and BLT inputs
        attention = workspace.attention_schema
        metacognition_mod = workspace.metacognition
        active_inference = workspace.active_inference
        beautiful_loop = workspace.beautiful_loop
        extra_modules = {
            "phi_tracker": persisted["phi"],
            "rpt_measurement": persisted["rpt"],
            "pad_affect_model": persisted["pad_affect"],
            "causal_emergence_analyzer": persisted["causal_emergence"],
        }
    else:
        (
            workspace,
            attention,
            metacognition_mod,
            active_inference,
            beautiful_loop,
        ) = _build_standalone_modules(overrides)
        # The workspace's tracker reaches the assessment in every mode: with
        # no cycles run it is warming up and integrated_information reads
        # insufficient_data, not absent
        extra_modules = {"phi_tracker": workspace.phi_tracker}
        # The separate loop is built either way (the seeded stream stays the
        # same); the switch says which loop the assessment reads
        if _blt_source(overrides) == BLT_SOURCE_WORKSPACE:
            beautiful_loop = workspace.beautiful_loop
    profile = overrides.get("runner", {}).get("warmup_profile", _DEFAULT_WARMUP_PROFILE)
    # Every cycle.* switch stays off unless the overrides set it
    augmenter = build_cycle_augmenter(
        overrides, workspace, pad_model=extra_modules.get("pad_affect_model")
    )

    if args.full:
        await warmup(
            workspace,
            attention,
            metacognition_mod,
            active_inference,
            cycles=args.warmup_cycles,
            profile=profile,
            augmenter=augmenter,
        )

    if args.ablation:
        results = await run_ablation(
            workspace,
            attention,
            metacognition_mod,
            active_inference,
            beautiful_loop,
            assessment_kwargs=overrides.get("assessment", {}),
            extra_modules=extra_modules,
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
        extra_modules=extra_modules,
    )
    print_report(report, markdown=args.report)

    if args.json or args.json_path:
        if args.json_path:
            out_path = Path(args.json_path)
        else:
            settings = get_settings()
            out_dir = resolve_path(settings.ASSESSMENT_OUTPUT_DIR, settings=settings)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"assessment_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f" Results written to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
