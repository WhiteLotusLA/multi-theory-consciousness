#!/usr/bin/env python3
"""
AutoConsciousness experiment runner.

Loads parameter overrides from experiments/params.toml (or --params PATH),
optionally layered over a preset from experiments/presets/ (--preset NAME),
validates against experiments/param_schema.toml, instantiates consciousness
modules with overridden params, runs warmup + assessment, prints
grep-friendly results.

Usage:
    python scripts/run_experiment.py                    # standard run
    python scripts/run_experiment.py --preset legacy_v04  # reproduce v0.4.0
    python scripts/run_experiment.py --params my.toml   # alternate params file
    python scripts/run_experiment.py --validate gwt     # run + ablation check on GWT
    python scripts/run_experiment.py --validate ast,hot # check multiple modules
"""

import argparse
import asyncio
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

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

from mtc.core.param_overrides import load_layered_overrides  # noqa: E402
from mtc.core.runner_support import (  # noqa: E402
    WARMUP_PROFILE_V05,
    attention_shift_type,
    build_phi_tracker,
    build_workspace,
)
from mtc.consciousness.enhanced_global_workspace import (  # noqa: E402
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.attention_schema import (  # noqa: E402
    AttentionSchemaModule,
    AttentionTarget,
)
from mtc.consciousness.metacognition import (  # noqa: E402
    MetacognitionModule,
    FirstOrderStateType,
    MetaType,
)
from mtc.consciousness.active_inference import (  # noqa: E402
    ActiveInferenceModule,
    ActiveInferenceConfig,
)
from mtc.consciousness.beautiful_loop import BeautifulLoop  # noqa: E402
from mtc.consciousness.pad_affect import PADAffectModel, PADSources  # noqa: E402
from mtc.neural.oscillatory_binding import OscillatoryBinding  # noqa: E402
from mtc.assessment.rpt_measurement import RPTMeasurement  # noqa: E402
from mtc.assessment.assessment import (  # noqa: E402
    INSUFFICIENT_DATA,
    ConsciousnessAssessment,
)
from mtc.assessment.causal_emergence import CausalEmergenceAnalyzer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "experiments" / "param_schema.toml"
PARAMS_PATH = ROOT / "experiments" / "params.toml"
PRESETS_DIR = ROOT / "experiments" / "presets"
WARMUP_CYCLES = 25


class _WarmupSelfModel:
    """Lightweight self-model for BLT epistemic depth during warmup.

    Starts naive (0.5 across the board) and drifts via random walk.
    No pre-calibration - the model earns its confidence through cycles.
    """

    def __init__(self):
        self.update_count = 0
        self.confidence = 0.5
        self.cognitive_load = 0.5
        self.prediction_accuracy = 0.5
        self.model_confidence = 0.5
        self.self_calibration_score = 0.5
        self.prediction_accuracy_history = []

    def step(self):
        self.update_count += 1
        self.prediction_accuracy += np.random.uniform(-0.08, 0.08)
        self.prediction_accuracy = np.clip(self.prediction_accuracy, 0.1, 0.9)
        self.prediction_accuracy_history.append(self.prediction_accuracy)
        self.model_confidence = 0.4 + 0.2 * min(1.0, self.update_count / 10)


class _WarmupMetaState:
    """Lightweight meta-state for BLT epistemic depth during warmup.

    Independent tracking with its own noise - not locked to self-model.
    """

    def __init__(self):
        self.total_predictions = 0
        self.predicted_accuracy = 0.5
        self.prediction_accuracy = 0.5
        self.model_confidence = 0.5

    def step(self, self_model):
        self.total_predictions += 1
        # Track self-model loosely, not perfectly - independent noise
        self.prediction_accuracy = self_model.prediction_accuracy + np.random.uniform(
            -0.1, 0.1
        )
        self.prediction_accuracy = np.clip(self.prediction_accuracy, 0.1, 0.9)
        self.predicted_accuracy = self.prediction_accuracy + np.random.uniform(
            -0.1, 0.1
        )
        self.model_confidence = 0.4 + 0.2 * min(1.0, self.total_predictions / 10)


# A 25-cycle warmup records 24 transitions, below the tracker's default
# reporting minimum, so the runner lowers it to what the causal emergence
# analysis has always needed here.
_PHI_MIN_TRANSITIONS = 10

_DEFAULT_WARMUP_PROFILE = WARMUP_PROFILE_V05


def _warmup_profile(overrides: dict) -> str:
    return overrides.get("runner", {}).get("warmup_profile", _DEFAULT_WARMUP_PROFILE)


def build_modules(overrides: dict) -> dict:
    """Instantiate all consciousness modules, applying TOML overrides."""

    # Machine metrics and the previous cycle's wall-clock duration reach the
    # protoself when body metrics are on, so two runs of the same seed would
    # differ. The runner pins them off unless gwt.body_metrics_enabled is set.
    # The workspace builds its own inference, beautiful loop, damasio,
    # metacognition and attention schema modules; the matching overrides reach
    # them through its constructor so a preset switches every instance in the
    # run.
    phi_tracker = build_phi_tracker(overrides.get("phi", {}), _PHI_MIN_TRANSITIONS)
    workspace = build_workspace(overrides, phi_tracker, body_metrics_default=False)

    ast_ov = overrides.get("ast", {})
    attention = AttentionSchemaModule(**ast_ov)

    hot_ov = overrides.get("hot", {})
    metacognition = MetacognitionModule(**hot_ov)

    fep_ov = overrides.get("fep", {}).copy()
    fep_ov.pop("drives", None)  # drives handled separately (not yet wired)
    # Every fep.* key (including the world_model, hpp_hidden_dim_multiplier,
    # meta_accuracy_method, drive_model, drive_decay_floor and
    # legacy_double_decay switches) is an ActiveInferenceConfig field.
    fep_config = ActiveInferenceConfig(**{k: v for k, v in fep_ov.items()})
    active_inference = ActiveInferenceModule(config=fep_config)

    # blt.* keys (loop_quality_formula, precision_floor_enabled,
    # require_recursive_confidence, ...) are BeautifulLoop constructor kwargs.
    blt_ov = overrides.get("blt", {})
    beautiful_loop = BeautifulLoop(**blt_ov)

    pad_ov = overrides.get("pad", {})
    pad_affect = PADAffectModel(**pad_ov)

    osc_ov = overrides.get("oscillatory", {})
    oscillatory = OscillatoryBinding(**osc_ov)

    rpt_ov = overrides.get("rpt", {})
    rpt = RPTMeasurement(**rpt_ov)

    # Wire oscillatory binding into RPT for oscillatory_binding_coherence indicator
    rpt.attach_oscillatory_binding(oscillatory)

    return {
        "global_workspace": workspace,
        "attention_schema": attention,
        "metacognition": metacognition,
        "active_inference": active_inference,
        "beautiful_loop": beautiful_loop,
        "rpt_measurement": rpt,
        "pad_affect_model": pad_affect,
        "neural_states": _build_neural_states(),
        "phi_tracker": phi_tracker,
    }


def _build_neural_states() -> dict:
    """Create neutral synthetic neural states for assessment.

    Uses unbiased random data so the Phi approximation measures whatever
    integration structure emerges naturally. No engineered connectivity
    or shared-signal injection - scores reflect the framework's baseline
    response to uninstructed substrate data.
    """
    n = 16  # Moderate size - enough to exercise Phi without inflating size_factor
    snn = np.random.rand(n, n).astype(np.float32)
    lsm = np.random.rand(n, n).astype(np.float32)
    htm = np.random.rand(n, n).astype(np.float32)

    combined_state = np.mean(
        [snn.mean(axis=1), lsm.mean(axis=1), htm.mean(axis=1)], axis=0
    )

    # Random symmetric connectivity - no engineered spectral structure
    raw = np.random.rand(n, n).astype(np.float32)
    connectivity = (raw + raw.T) / 2.0  # Symmetric
    np.fill_diagonal(connectivity, 1.0)

    return {
        "snn_spikes": snn,
        "lsm_states": lsm,
        "htm_columns": htm,
        "combined_state": combined_state,
        "connectivity": connectivity,
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
        for j in range(7)
    ]


async def warmup(
    modules: dict,
    cycles: int = WARMUP_CYCLES,
    profile: str = _DEFAULT_WARMUP_PROFILE,
    ce_overrides: Optional[dict] = None,
):
    """Run consciousness cycles to fill statistics buffers.
    Mirrors scripts/run_assessment.py warmup logic.
    Handles None modules gracefully (for ablation runs).

    ``profile`` is the runner.warmup_profile switch: "v04" feeds the attention
    schema captured shifts only, "v05" alternates chosen and pulled shifts.
    The PhiTracker the workspace records into is read back for the causal
    emergence analysis, which is built from the causal_emergence overrides
    ``ce_overrides``; without a workspace nothing is recorded."""
    workspace = modules.get("global_workspace")
    attention = modules.get("attention_schema")
    metacognition = modules.get("metacognition")
    active_inference = modules.get("active_inference")
    beautiful_loop = modules.get("beautiful_loop")
    rpt = modules.get("rpt_measurement")
    pad_affect = modules.get("pad_affect_model")
    oscillatory = rpt.oscillatory_binding if rpt is not None else None
    phi_tracker = getattr(workspace, "phi_tracker", None)

    # BLT epistemic depth needs self-model and meta-state
    self_model = _WarmupSelfModel() if beautiful_loop is not None else None
    meta_state = _WarmupMetaState() if beautiful_loop is not None else None

    for i in range(cycles):
        state = None
        hot_obj = None
        fep_result = None

        # GWT competition cycle
        if workspace is not None:
            candidates = _make_candidates(i)
            state = await workspace.process_consciousness_cycle(candidates)

        # Feed AST
        if attention is not None and state is not None and state.primary_content:
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
        if metacognition is not None and state is not None and state.primary_content:
            fo = metacognition.register_first_order_state(
                content=state.primary_content.candidate.content,
                content_summary=state.primary_content.candidate.summary,
                state_type=FirstOrderStateType.PERCEPTION,
                source_module="global_workspace",
            )
            hot_obj = await metacognition.generate_hot(fo, MetaType.AWARENESS)

        # Feed FEP
        if active_inference is not None:
            obs = np.random.dirichlet(np.ones(5))
            fep_result = await active_inference.infer_and_act(obs)

        # Feed BLT - run a conscious moment with full cross-module data
        if beautiful_loop is not None:
            self_model.step()
            meta_state.step(self_model)
            # Synthetic prediction errors (one per hierarchy level) for precision control
            pred_errors = [np.random.uniform(0.1, 0.4) for _ in range(3)]
            if fep_result is not None and hasattr(fep_result, "prediction_error"):
                pred_errors[0] = float(fep_result.prediction_error)
            blt_kwargs = {
                "context": f"warmup_cycle_{i}",
                "self_model": self_model,
                "meta_state": meta_state,
                "prediction_errors": pred_errors,
            }
            if state is not None:
                blt_kwargs["workspace_winners"] = state.workspace_contents or []
            if fep_result is not None:
                blt_kwargs["inference_result"] = fep_result
            if hot_obj is not None:
                blt_kwargs["higher_order_thoughts"] = [hot_obj]
            if attention is not None:
                blt_kwargs["attention_schema_state"] = getattr(
                    attention, "schema", None
                )
            await beautiful_loop.process_conscious_moment(**blt_kwargs)

        # Feed RPT - neutral substrate states for recurrence measurement
        if rpt is not None:
            # SNN: full range - hidden can amplify OR attenuate
            num_neurons = 100
            input_spikes = np.random.randint(1, 20, size=num_neurons)
            hidden_spikes = np.random.randint(1, 20, size=num_neurons)
            snn_state = {
                "total_spikes": int(input_spikes.sum() + hidden_spikes.sum()),
                "num_neurons": num_neurons,
                "spike_counts": [input_spikes, hidden_spikes],
                "activation_pattern": np.random.rand(num_neurons),
            }
            # LSM: full spectral radius range, not locked at edge-of-chaos
            lsm_state = {
                "spectral_radius": np.random.uniform(0.3, 1.1),
                "reservoir_state": np.random.rand(50),
                "active_neuron_ratio": np.random.uniform(0.1, 0.9),
                "edge_of_chaos_metric": np.random.uniform(0.2, 1.0),
            }
            ws_state = None
            if workspace is not None:
                gw_stats = (
                    workspace.get_statistics()
                    if hasattr(workspace, "get_statistics")
                    else {}
                )
                ws_state = {
                    "broadcast": gw_stats.get("broadcast", {}),
                    "cycle_count": gw_stats.get("cycle_count", i + 1),
                    "reentry_count": i + 1,
                    "source_module_diversity": 3,
                    "ignition": gw_stats.get("ignition", {}),
                }
            rpt.measure_full(
                snn_state=snn_state,
                lsm_state=lsm_state,
                workspace_state=ws_state,
            )

        # Feed oscillatory binding (for RPT oscillatory_binding_coherence)
        if oscillatory is not None:
            osc_metrics = {
                "attention_focus": np.random.uniform(0.0, 1.0),
                "valence": np.random.uniform(-1.0, 1.0),
                "novelty": np.random.uniform(0.0, 1.0),
                "arousal": np.random.uniform(0.0, 1.0),
                "goal_relevance": np.random.uniform(0.0, 1.0),
                "temporal_urgency": np.random.uniform(0.0, 1.0),
            }
            oscillatory.bind(osc_metrics)

        # Feed PAD affect model - full valid domain per PADSources spec
        if pad_affect is not None:
            pad_sources = PADSources(
                somatic_valence=np.random.uniform(-1.0, 1.0),
                homeostatic_satisfaction=np.random.uniform(-1.0, 1.0),
                snn_valence=np.random.uniform(-1.0, 1.0),
                protoself_arousal=np.random.uniform(0.0, 1.0),
                snn_spike_rate=np.random.uniform(0.0, 1.0),
                ignition_rate=np.random.uniform(0.0, 1.0) if state else 0.0,
                prediction_error=np.random.uniform(0.0, 1.0),
                agency_score=np.random.uniform(0.0, 1.0),
                voluntary_attention_ratio=np.random.uniform(0.0, 1.0),
                epistemic_depth_normalized=np.random.uniform(0.0, 1.0),
            )
            pad_affect.compute(pad_sources)

    # Post-warmup: run causal emergence analysis
    if phi_tracker is not None and not phi_tracker.warming_up:
        ce_analyzer = CausalEmergenceAnalyzer(**(ce_overrides or {}))
        ce_analyzer.analyze_modules(phi_tracker)
        modules["causal_emergence_analyzer"] = ce_analyzer


async def _compute_noise_baseline(
    overrides: dict, iterations: int = 5
) -> Dict[str, float]:
    """Compute noise baseline by running real modules with different random seeds.

    This measures what the modules produce on random data - the floor score
    that any input would achieve. Scores above this floor represent genuine
    signal from structured input.
    """
    accumulated = defaultdict(list)

    for i in range(iterations):
        # Different seed each iteration - but deterministic for reproducibility
        seed = SEED + 1000 + i
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass

        modules = build_modules(overrides)
        await warmup(
            modules,
            WARMUP_CYCLES,
            _warmup_profile(overrides),
            overrides.get("causal_emergence", {}),
        )
        assessment = ConsciousnessAssessment(**overrides.get("assessment", {}))
        report = await assessment.run_full_assessment(
            **modules, warmup_cycles=WARMUP_CYCLES, normalize=False
        )
        for name, result in report.indicator_results.items():
            # An indicator with no evidence has no score to take a floor from
            if result.status == INSUFFICIENT_DATA:
                continue
            accumulated[name].append(result.score)

    # Restore original seed
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except ImportError:
        pass

    return {name: float(np.mean(scores)) for name, scores in accumulated.items()}


def phi_source_used(report) -> str:
    """Where the report's Phi came from.

    One of phi_tracker_iit4, calculator_iit4_sia or calculator_spectral_proxy,
    read from the integrated_information evidence, else from the report's
    phi measurement; ``none`` when no Phi was measured.
    """
    result = report.indicator_results.get("integrated_information")
    if result is not None:
        source = (result.evidence or {}).get("phi_source_used")
        if source:
            return str(source)
    measurement = report.phi_measurement
    if measurement is None:
        return "none"
    if (measurement.cause_repertoire or {}).get("iit_version") == "4.0":
        return "phi_tracker_iit4"
    return f"calculator_{measurement.method}"


def _format_score(score: Optional[float]) -> str:
    return "insufficient_data" if score is None else f"{score:.6f}"


def print_report(report, t_total: float, ablation_results: dict = None):
    """Print grep-friendly output block."""
    print("---")
    print(f"overall_score:     {_format_score(report.overall_score)}")
    passing = sum(1 for r in report.indicator_results.values() if r.passes_threshold)
    print(f"passing_count:     {passing}")
    print(f"total_indicators:  {len(report.indicator_results)}")
    print(f"assessment_status: {report.status}")
    print(f"coverage:          {report.coverage:.3f}")
    print(f"phi_source_used:   {phi_source_used(report)}")
    for theory_name, theory_score in sorted(report.theory_scores.items()):
        print(
            f"{theory_name.lower()}_score:{'':>{13 - len(theory_name)}}{theory_score:.6f}"
        )
    print(f"processing_ms:     {report.processing_time_ms:.1f}")
    print(f"warmup_cycles:     {WARMUP_CYCLES}")
    print(f"total_seconds:     {t_total:.1f}")

    if ablation_results:
        print("---")
        print("ablation_results:")
        all_pass = True
        for module_name, result in sorted(ablation_results.items()):
            status = "GENUINE" if result["genuine"] else "COSMETIC"
            print(
                f" {module_name}: score_with={result['score_with']:.6f} "
                f"score_without={result['score_without']:.6f} "
                f"drop={result['drop']:.6f} {status}"
            )
            if not result["genuine"]:
                all_pass = False
        print(f"ablation_passed:   {'true' if all_pass else 'false'}")


# Module name -> assessment kwarg mapping for ablation
_ABLATION_MAP = {
    "gwt": "global_workspace",
    "ast": "attention_schema",
    "hot": "metacognition",
    "fep": "active_inference",
    "blt": "beautiful_loop",
    "rpt": "rpt_measurement",
}


async def run_ablation(
    overrides: dict,
    modules_to_check: list[str],
    baseline_score: float,
    noise_baselines: dict = None,
) -> dict:
    """Run ablation validation: disable each specified module and check score drops.

    Returns dict of {module_name: {score_with, score_without, drop, genuine}}.
    A change is 'genuine' if disabling the module causes a score drop > 0.001.
    """
    results = {}
    for module_name in modules_to_check:
        if module_name not in _ABLATION_MAP:
            print(
                f"WARNING: Unknown module '{module_name}' for ablation, skipping",
                file=sys.stderr,
            )
            continue

        kwarg_name = _ABLATION_MAP[module_name]

        # Re-seed and rebuild without the target module
        random.seed(SEED)
        np.random.seed(SEED)
        try:
            import torch

            torch.manual_seed(SEED)
        except ImportError:
            pass

        modules = build_modules(overrides)
        modules[kwarg_name] = None  # Disable this module
        await warmup(
            modules,
            WARMUP_CYCLES,
            _warmup_profile(overrides),
            overrides.get("causal_emergence", {}),
        )
        assessment = ConsciousnessAssessment(**overrides.get("assessment", {}))
        ablated_report = await assessment.run_full_assessment(
            **modules,
            warmup_cycles=WARMUP_CYCLES,
            normalize=True,
            noise_baselines=noise_baselines,
        )

        # A None overall means nothing was measured or absent; the module
        # contributed everything, so that counts as a full drop
        without = ablated_report.overall_score
        drop = baseline_score - (without if without is not None else 0.0)
        results[module_name] = {
            "score_with": baseline_score,
            "score_without": without if without is not None else 0.0,
            "drop": drop,
            "genuine": drop > 0.001,  # Must cause measurable score drop
        }

    return results


async def run():
    parser = argparse.ArgumentParser(description="AutoConsciousness experiment runner")
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Comma-separated module names to ablation-check (e.g., gwt,ast,hot)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=str(PARAMS_PATH),
        help="Parameter overrides TOML (default: experiments/params.toml)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name from experiments/presets/ applied before --params "
        "(e.g., legacy_v04 reproduces v0.4.0)",
    )
    args = parser.parse_args()

    t_start = time.time()

    # Load and validate overrides: defaults, then preset, then params file
    try:
        overrides = load_layered_overrides(
            args.params,
            str(SCHEMA_PATH),
            preset=args.preset,
            presets_dir=str(PRESETS_DIR),
        )
    except (ValueError, OSError) as e:
        print(f"PARAM ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if overrides:
        print(f"Overrides: {overrides}", file=sys.stderr)

    # Compute noise baseline once (used for all assessments)
    print("Computing noise baseline...", file=sys.stderr)
    noise_baselines = await _compute_noise_baseline(overrides, iterations=5)

    # Run main assessment (re-seed for determinism)
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except ImportError:
        pass

    modules = build_modules(overrides)
    await warmup(
        modules,
        WARMUP_CYCLES,
        _warmup_profile(overrides),
        overrides.get("causal_emergence", {}),
    )
    assessment = ConsciousnessAssessment(**overrides.get("assessment", {}))
    report = await assessment.run_full_assessment(
        **modules,
        warmup_cycles=WARMUP_CYCLES,
        normalize=True,
        noise_baselines=noise_baselines,
    )

    # Run ablation validation if requested
    ablation_results = None
    if args.validate:
        modules_to_check = [m.strip() for m in args.validate.split(",")]
        ablation_results = await run_ablation(
            overrides,
            modules_to_check,
            report.overall_score if report.overall_score is not None else 0.0,
            noise_baselines,
        )

    t_total = time.time() - t_start
    print_report(report, t_total, ablation_results)


if __name__ == "__main__":
    asyncio.run(run())
