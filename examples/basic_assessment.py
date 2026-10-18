#!/usr/bin/env python3
"""
Basic Consciousness Assessment
==============================

Runs the assessment once on freshly built modules and prints what it
measured. The assessment defines 30 positive indicators and 3 negative
indicators over eight theories (GWT, IIT, AST, HOT, FEP, RPT, BLT and Agency
and Embodiment).

Every indicator reports a status:

- ``measured``: the module produced the evidence and the indicator was scored
  0 to 1 against its threshold;
- ``absent``: the module was not supplied, so the indicator scores 0.0;
- ``insufficient_data``: the module was supplied but has not produced the
  evidence yet (for example a workspace that has run no cycle). Such an
  indicator is left out of every average and never passes.

The modules here have run no cycle, so most indicators read
``insufficient_data`` and the report's status says so; run cycles first (see
``multi_theory_pipeline.py``) or use ``mtc assess --full`` for a warmed-up run.
Passing indicators would show functional architecture, not experience.

Needs the ``fep`` extra (pymdp) for the active inference module; without it
the example prints one line that names the extra and exits 2. Opens no
network connection.

Usage:
    python examples/basic_assessment.py
"""

import asyncio
import logging
import sys

from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.cli.missing_extra import explain_missing_extra
from mtc.consciousness.active_inference import ActiveInferenceModule
from mtc.consciousness.attention_schema import AttentionSchemaModule
from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
from mtc.consciousness.metacognition import MetacognitionModule

logging.basicConfig(level=logging.WARNING)


def _score(value):
    return "none" if value is None else f"{value:.3f}"


async def run():
    # 1. Build the modules the assessment reads.
    workspace = EnhancedGlobalWorkspace(capacity=7, ignition_threshold=0.3)
    attention = AttentionSchemaModule()
    metacognition = MetacognitionModule()
    active_inference = ActiveInferenceModule()

    # 2. Assess them. Modules not passed (RPT, PAD, Phi inputs, the beautiful
    #    loop) report their indicators as absent.
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
    )

    # 3. Print what was measured and what was not.
    print_report(
        report,
        positive=len(assessment.indicator_configs),
        negative=len(assessment.negative_indicator_configs),
    )


def print_report(report, *, positive, negative):
    """Print the report's status, scores and per-indicator outcomes."""
    print("\n" + "=" * 60)
    print("  CONSCIOUSNESS ASSESSMENT REPORT")
    print("=" * 60)
    print(f"  Indicators defined:   {positive} positive, {negative} negative")
    print(f"  Status:               {report.status}")
    print(f"  Coverage:             {report.coverage:.3f}")
    print(f"  Overall score:        {_score(report.overall_score)}")
    print(f"  Passing indicators:   {report.passing_count}/{report.total_indicators}")
    print(f"  Architecture active:  {report.architecture_functional}")
    print(f"  Insufficient data:    {len(report.insufficient_indicators)}")

    print(
        "\n  Theory scores (absent indicators count as 0.0; theories with no "
        "measured indicator are not listed):"
    )
    for theory, score in sorted(report.theory_scores.items()):
        print(f"    {theory:<6s} {_score(score)}")

    print("\n  Indicators:")
    for name, result in report.indicator_results.items():
        if result.status == "insufficient_data":
            shown = "insufficient_data"
        else:
            verdict = "pass" if result.passes_threshold else "fail"
            shown = f"{result.status:<9s} {result.score:.3f} ({verdict})"
        print(f"    {name:<32s} {shown}  threshold {result.threshold:.2f}")

    print("\n  Negative indicators:")
    for name, entry in report.negative_indicators.items():
        print(
            f"    {name:<32s} {entry['status']:<17s} "
            f"score {_score(entry['score'])}  penalty {entry['penalty']:.3f}"
        )
    print("\n" + "=" * 60)


def main() -> int:
    """Run the example; on an install without the ``fep`` extra print the one
    line that names it to stderr and return 2 (any other ImportError is
    raised)."""
    try:
        asyncio.run(run())
    except ImportError as exc:
        line = explain_missing_extra(exc)
        if line is None:
            raise
        print(line, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
