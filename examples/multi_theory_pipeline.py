#!/usr/bin/env python3
"""
Multi-Theory Consciousness Pipeline
===================================

Wires the theory modules together by hand for one stimulus, then assesses
them. The framework implements eight theories; six have modules wired below,
and the assessment measures the other two (IIT and Recurrent Processing) and
Agency and Embodiment from module statistics:

1. GWT: the enhanced global workspace (competition, ignition, broadcast)
2. AST: the attention schema (a model of attention)
3. HOT: metacognition (higher-order thoughts about first-order states)
4. FEP: active inference (prediction error and free energy)
5. BLT: the beautiful loop (precision, binding and epistemic depth)
6. Damasio's three layers (protoself, core and extended consciousness), which
   has no indicator of its own

The assessment defines 30 positive and 3 negative indicators. After a single
cycle some of them still read ``insufficient_data`` (for example the error
trend needs 10 prediction errors) and others ``absent`` (no RPT, PAD or Phi
input is passed here); the script prints how many of each. The live runtime
(``mtc/pipeline/runtime.py``) runs this wiring every conversation turn; see
``examples/full_pipeline_demo.py``. Scores describe functional architecture,
not experience.

Needs the ``fep`` extra (pymdp). Opens no network connection.

Usage:
    python examples/multi_theory_pipeline.py
"""

import asyncio
import logging

import numpy as np

from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.consciousness.active_inference import ActiveInferenceModule
from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionShiftType,
    AttentionTarget,
)
from mtc.consciousness.beautiful_loop import BeautifulLoop
from mtc.consciousness.damasio import DamasioLayers
from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.metacognition import (
    FirstOrderStateType,
    MetacognitionModule,
    MetaType,
)

logging.basicConfig(level=logging.WARNING)


async def main():
    rng = np.random.default_rng(7)

    # 1. The theory modules.
    workspace = EnhancedGlobalWorkspace(capacity=7, ignition_threshold=0.3)
    attention = AttentionSchemaModule()
    metacognition = MetacognitionModule()
    active_inference = ActiveInferenceModule()
    beautiful_loop = BeautifulLoop(num_levels=3)
    damasio = DamasioLayers()

    print("\n" + "=" * 64)
    print("  MULTI-THEORY PIPELINE: ONE STIMULUS")
    print("=" * 64)

    # 2. The stimulus, as workspace candidates.
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

    # 3. GWT: one workspace cycle.
    print("\n  GWT: one workspace cycle")
    state = await workspace.process_consciousness_cycle(candidates)
    print(f"    Ignition events:  {state.ignition_events}")
    if state.primary_content:
        print(f"    Primary content:  {state.primary_content.candidate.summary}")

    # 4. AST: point the schema at the winner and ask for its report.
    print("\n  AST: attention schema")
    if state.primary_content:
        winner = state.primary_content
        attention.schema.current_focus = AttentionTarget(
            content=winner.candidate.content,
            content_type=winner.candidate.content_type,
            summary=winner.candidate.summary,
            source_module="global_workspace",
            attention_strength=winner.salience,
            salience=winner.salience,
            shift_type=AttentionShiftType.CAPTURED,
        )
    print(f"    Report: {await attention.report_attention()}")

    # 5. HOT: a higher-order thought about the winner.
    print("\n  HOT: metacognition")
    if state.primary_content:
        first_order = metacognition.register_first_order_state(
            content=state.primary_content.candidate.content,
            content_summary=state.primary_content.candidate.summary,
            state_type=FirstOrderStateType.PERCEPTION,
            source_module="global_workspace",
            confidence=0.7,
        )
        hot = await metacognition.generate_hot(
            first_order, meta_type=MetaType.AWARENESS, trigger="workspace_winner"
        )
        if hot is not None:
            print(f"    Higher-order thought: {hot.meta_content}")
            print(f"    Level:                {hot.level.name}")

    # 6. FEP: one active inference step on an observation.
    print("\n  FEP: active inference")
    observation = rng.dirichlet(np.ones(5))  # five observation channels
    inference = await active_inference.infer_and_act(observation)
    print(f"    Prediction error:   {inference.prediction_error:.4f}")
    print(f"    Free energy:        {inference.variational_free_energy:.4f}")
    print(f"    Selected action:    {inference.selected_action}")

    # 7. BLT: one conscious moment of the beautiful loop.
    print("\n  BLT: beautiful loop")
    moment = await beautiful_loop.process_conscious_moment(
        workspace_winners=state.workspace_contents,
        inference_result=inference,
    )
    print(f"    Loop quality:       {moment.loop_quality:.3f}")
    print(f"    Epistemic depth:    {moment.epistemic_depth}")
    print(f"    Binding quality:    {moment.binding_quality:.3f}")

    # 8. Damasio: the three layers over the drives and the winners.
    print("\n  Damasio: three layers")
    layers = await damasio.process(
        homeostatic_drives=active_inference.homeostatic_drives,
        workspace_winners=state.workspace_contents,
    )
    print(f"    Protoself stability:       {layers['protoself_stability']:.3f}")
    print(f"    Self-world binding:        {layers['self_world_binding']:.3f}")
    print(f"    Autobiographic continuity: {layers['autobiographic_continuity']:.3f}")

    # 9. The assessment over the wired modules.
    print("\n  Assessment")
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        global_workspace=workspace,
        attention_schema=attention,
        metacognition=metacognition,
        active_inference=active_inference,
        beautiful_loop=beautiful_loop,
    )
    statuses = {}
    for result in report.indicator_results.values():
        statuses[result.status] = statuses.get(result.status, 0) + 1
    positive = len(assessment.indicator_configs)
    negative = len(assessment.negative_indicator_configs)
    overall = report.overall_score
    print(f"    Indicators defined:  {positive} positive, {negative} negative")
    print(f"    Status:              {report.status}")
    print(f"    Coverage:            {report.coverage:.3f}")
    print(f"    By status:           {dict(sorted(statuses.items()))}")
    print(f"    Passing:             {report.passing_count}/{report.total_indicators}")
    print(
        "    Overall score:       " + ("none" if overall is None else f"{overall:.3f}")
    )
    print("\n" + "=" * 64 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
