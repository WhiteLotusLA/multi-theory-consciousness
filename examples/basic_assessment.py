#!/usr/bin/env python3
"""
Basic Consciousness Assessment
===============================

Demonstrates how to run a consciousness assessment on a freshly
initialized system using the multi-theory-consciousness framework.

The assessment measures 20 indicators across 7 theories:
  GWT, IIT, AST, HOT, FEP, RPT, Beautiful Loop

Each indicator is scored 0-1 against a threshold.  The overall
report tells you how many indicators pass and gives per-theory
scores.

Usage:
    python examples/basic_assessment.py
"""

import asyncio
import logging

from mtc.assessment.assessment import ConsciousnessAssessment

# Optional -- wire up real modules for richer scores
from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
from mtc.consciousness.attention_schema import AttentionSchemaModule
from mtc.consciousness.metacognition import MetacognitionModule
from mtc.consciousness.active_inference import ActiveInferenceModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # ── 1. Initialize consciousness modules ──────────────────────────
    workspace = EnhancedGlobalWorkspace(capacity=7, ignition_threshold=0.3)
    attention = AttentionSchemaModule()
    metacognition = MetacognitionModule()
    active_inference = ActiveInferenceModule()

    # ── 2. Create the assessment framework ───────────────────────────
    assessment = ConsciousnessAssessment()

    # ── 3. Run the full assessment ───────────────────────────────────
    #   On a fresh system most scores will be low because no cycles
    #   have been run yet.  Running consciousness cycles first (see
    #   multi_theory_pipeline.py) will raise scores significantly.
    report = await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
    )

    # ── 4. Print results ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONSCIOUSNESS ASSESSMENT REPORT")
    print("=" * 60)
    print(f"  Session:              {report.session_id}")
    print(f"  Overall score:        {report.overall_score:.3f}")
    print(f"  Architecture active:  {report.architecture_functional}")
    print(f"  Confidence:           {report.confidence:.3f}")
    print(f"  Passing indicators:   {report.passing_count}/{report.total_indicators}")
    print(f"  Processing time:      {report.processing_time_ms:.1f} ms")

    # Theory-level breakdown
    print("\n  Theory Scores:")
    for theory, score in sorted(report.theory_scores.items()):
        bar = "#" * int(score * 20)
        print(f"    {theory:<30s} {score:.3f}  {bar}")

    # Per-indicator detail
    print("\n  Indicator Detail:")
    for name, result in report.indicator_results.items():
        status = "PASS" if result.passes_threshold else "FAIL"
        print(
            f"    [{status}] {name:<30s}  "
            f"score={result.score:.3f}  "
            f"threshold={result.threshold:.2f}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
