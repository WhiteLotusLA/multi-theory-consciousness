#!/usr/bin/env python3
"""
Multi-Theory Consciousness Pipeline
=====================================

Demonstrates all 7 consciousness theories working together in a
single processing cycle:

  1. GWT  — Global Workspace Theory (competition & broadcast)
  2. IIT  — Integrated Information Theory (measured via assessment)
  3. AST  — Attention Schema Theory (self-model of attention)
  4. HOT  — Higher-Order Thought Theory (thoughts about thoughts)
  5. FEP  — Free Energy Principle (active inference)
  6. BLT  — Beautiful Loop Theory (recursive self-reference)
  7. Damasio — Three-layer consciousness (protoself/core/extended)

The pipeline mirrors what happens inside a full consciousness cycle:
candidates enter the workspace, theories enrich the state, and the
result is an integrated conscious moment.

Usage:
    python examples/multi_theory_pipeline.py
"""

import asyncio
import logging
import numpy as np

# -- Consciousness modules --
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
    ActiveInferenceModule,
    HomeostaticDrives,
)
from mtc.consciousness.beautiful_loop import BeautifulLoop
from mtc.consciousness.damasio import DamasioLayers

# -- Assessment --
from mtc.assessment.assessment import ConsciousnessAssessment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # ══════════════════════════════════════════════════════════════════
    # 1.  Initialize all theory modules
    # ══════════════════════════════════════════════════════════════════
    workspace = EnhancedGlobalWorkspace(capacity=7, ignition_threshold=0.3)
    attention = AttentionSchemaModule()
    metacognition = MetacognitionModule()
    active_inference = ActiveInferenceModule()
    beautiful_loop = BeautifulLoop(num_levels=3)
    damasio = DamasioLayers()

    print("\n" + "=" * 64)
    print("  MULTI-THEORY CONSCIOUSNESS PIPELINE")
    print("=" * 64)

    # ══════════════════════════════════════════════════════════════════
    # 2.  Create a stimulus to process
    # ══════════════════════════════════════════════════════════════════
    candidates = [
        WorkspaceCandidate(
            content="novel pattern detected in input stream",
            content_type="perception",
            summary="Unexpected pattern in sensory data",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.75,
            emotional_salience=0.5,
            novelty_score=0.9,
        ),
        WorkspaceCandidate(
            content="recalled similar pattern from past experience",
            content_type="memory",
            summary="Memory of a similar pattern",
            source=WorkspaceCandidateSource.MEMORY,
            activation_level=0.6,
            emotional_salience=0.3,
            novelty_score=0.2,
        ),
        WorkspaceCandidate(
            content="internal prediction about what happens next",
            content_type="prediction",
            summary="Predictive model output",
            source=WorkspaceCandidateSource.LLM,
            activation_level=0.5,
            goal_relevance=0.7,
            novelty_score=0.4,
        ),
    ]

    # ══════════════════════════════════════════════════════════════════
    # 3.  Run the GWT consciousness cycle (steps 1-8 internally)
    #     This also triggers AST, HOT, FEP, Beautiful Loop, and Damasio
    #     if they are wired into the workspace.
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 1: Running GWT consciousness cycle ...")
    state = await workspace.process_consciousness_cycle(candidates)
    print(f"    Conscious: {state.is_conscious}")
    print(f"    Ignitions: {state.ignition_events}")
    if state.primary_content:
        print(f"    Winner:    {state.primary_content.candidate.summary}")

    # ══════════════════════════════════════════════════════════════════
    # 4.  AST — Update the attention schema
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 2: Updating Attention Schema (AST) ...")
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
    report = await attention.report_attention()
    print(f"    Attention report: {report}")

    # ══════════════════════════════════════════════════════════════════
    # 5.  HOT — Generate higher-order thoughts about workspace content
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 3: Generating Higher-Order Thoughts (HOT) ...")
    if state.primary_content:
        first_order = metacognition.register_first_order_state(
            content=state.primary_content.candidate.content,
            content_summary=state.primary_content.candidate.summary,
            state_type=FirstOrderStateType.PERCEPTION,
            source_module="global_workspace",
            confidence=0.7,
        )
        hot = await metacognition.generate_hot(
            first_order,
            meta_type=MetaType.AWARENESS,
            trigger="workspace_winner",
        )
        if hot:
            print(f"    HOT generated: {hot.summary}")
            print(f"    Meta-level:    {hot.meta_level.name}")

    # ══════════════════════════════════════════════════════════════════
    # 6.  FEP — Run active inference on an observation
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 4: Running Active Inference (FEP) ...")
    observation = np.random.dirichlet(np.ones(5))  # 5 observation channels
    inference_result = await active_inference.infer_and_act(observation)
    print(f"    Prediction error:  {inference_result.prediction_error:.4f}")
    print(f"    Free energy:       {inference_result.variational_free_energy:.4f}")
    print(f"    Selected action:   {inference_result.selected_action}")
    print(f"    Action confidence: {inference_result.action_confidence:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # 7.  Beautiful Loop — Measure recursive self-reference
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 5: Processing Beautiful Loop (BLT) ...")
    moment = await beautiful_loop.process_conscious_moment(
        workspace_winners=state.workspace_contents,
        inference_result=inference_result,
    )
    print(f"    Loop quality:       {moment.loop_quality:.3f}")
    print(f"    Epistemic depth:    {moment.epistemic_depth}")
    print(f"    Binding quality:    {moment.binding_quality:.3f}")
    print(f"    Field-evidencing:   {moment.is_field_evidencing}")

    # ══════════════════════════════════════════════════════════════════
    # 8.  Damasio — Three-layer consciousness enrichment
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 6: Running Damasio Three-Layer Model ...")
    homeostatic_drives = active_inference.homeostatic_drives
    damasio_state = await damasio.process(
        homeostatic_drives=homeostatic_drives,
        workspace_winners=state.workspace_contents,
    )
    print(f"    Protoself stability:       {damasio_state['protoself_stability']:.3f}")
    print(f"    Self-world binding:        {damasio_state['self_world_binding']:.3f}")
    print(f"    Feeling of knowing:        {damasio_state['feeling_of_knowing']:.3f}")
    print(
        f"    Autobiographic continuity: "
        f"{damasio_state['autobiographic_continuity']:.3f}"
    )

    # ══════════════════════════════════════════════════════════════════
    # 9.  Run the full consciousness assessment
    # ══════════════════════════════════════════════════════════════════
    print("\n  Step 7: Running consciousness assessment ...")
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
    )
    print(f"    Overall score:       {report.overall_score:.3f}")
    print(f"    Passing indicators:  {report.passing_count}/{report.total_indicators}")
    print(f"    Architecture active: {report.architecture_functional}")

    print("\n" + "=" * 64)
    print("  Pipeline complete. All 7 theories participated.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
