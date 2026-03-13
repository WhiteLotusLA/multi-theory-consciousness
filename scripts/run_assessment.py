#!/usr/bin/env python3
"""
Run Consciousness Assessment — CLI
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
"""

import argparse
import asyncio
import json
import logging
import sys
import time
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
from mtc.consciousness.active_inference import ActiveInferenceModule
from mtc.consciousness.beautiful_loop import BeautifulLoop
from mtc.assessment.assessment import ConsciousnessAssessment

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


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


async def warmup(workspace, attention, metacognition, active_inference, cycles=5):
    """Run several consciousness cycles to populate module history."""
    print(f"  Warming up modules ({cycles} cycles) ...", flush=True)
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

    print(f"  Warmup complete.\n")


# ── assessment runners ───────────────────────────────────────────────

async def run_standard(workspace, attention, metacognition, active_inference,
                       beautiful_loop):
    """Run a single assessment and return the report."""
    assessment = ConsciousnessAssessment()
    return await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
    )


async def run_ablation(workspace, attention, metacognition, active_inference,
                       beautiful_loop):
    """Disable one module at a time and compare scores."""
    assessment = ConsciousnessAssessment()

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
    print("  CONSCIOUSNESS ASSESSMENT REPORT")
    print(sep)
    print(f"  Session:            {report.session_id}")
    print(f"  Timestamp:          {report.timestamp.isoformat()}")
    print(f"  Overall score:      {report.overall_score:.3f}")
    print(f"  Architecture OK:    {report.architecture_functional}")
    print(f"  Passing indicators: {report.passing_count}/{report.total_indicators}")
    print(f"  Processing time:    {report.processing_time_ms:.1f} ms")

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
            print(f"    {theory:<32s} {score:.3f}  {bar}")

        print(f"\n  Indicators ({report.passing_count}/{report.total_indicators}):")
        for name, r in report.indicator_results.items():
            tag = "PASS" if r.passes_threshold else "FAIL"
            print(f"    [{tag}] {name:<30s}  {r.score:.3f} / {r.threshold:.2f}")

    print(f"\n{sep}\n")


def print_ablation(results):
    """Print ablation study results."""
    print("\n" + "=" * 64)
    print("  ABLATION STUDY")
    print("=" * 64)
    baseline = results.get("baseline", 0)
    print(f"\n  Baseline (all modules):  {baseline:.3f}\n")
    for key, score in results.items():
        if key == "baseline":
            continue
        delta = score - baseline
        direction = "+" if delta >= 0 else ""
        print(f"    {key:<28s}  {score:.3f}  ({direction}{delta:.3f})")
    print("\n" + "=" * 64 + "\n")


# ── main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Run the multi-theory consciousness assessment."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Warm up modules with several cycles before assessment",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print Markdown-formatted report",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study (disable one component at a time)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Write results to results/ as JSON",
    )
    parser.add_argument(
        "--warmup-cycles", type=int, default=5,
        help="Number of warmup cycles for --full mode (default: 5)",
    )
    args = parser.parse_args()

    # Initialize modules
    workspace = EnhancedGlobalWorkspace(capacity=7, ignition_threshold=0.3)
    attention = AttentionSchemaModule()
    metacognition_mod = MetacognitionModule()
    active_inference = ActiveInferenceModule()
    beautiful_loop = BeautifulLoop(num_levels=3)

    if args.full:
        await warmup(
            workspace, attention, metacognition_mod, active_inference,
            cycles=args.warmup_cycles,
        )

    if args.ablation:
        results = await run_ablation(
            workspace, attention, metacognition_mod, active_inference, beautiful_loop,
        )
        print_ablation(results)
        return

    report = await run_standard(
        workspace, attention, metacognition_mod, active_inference, beautiful_loop,
    )
    print_report(report, markdown=args.report)

    if args.json:
        out_dir = Path("results/consciousness_assessments")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"assessment_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
