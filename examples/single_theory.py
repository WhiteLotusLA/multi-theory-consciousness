#!/usr/bin/env python3
"""
Single Theory Example: Global Workspace Theory (GWT)
=====================================================

Demonstrates how to use ONE consciousness theory module in
isolation.  This example uses the Enhanced Global Workspace,
which implements Baars' Global Workspace Theory (1988):

  1. Workspace candidates compete for conscious access.
  2. Winners that cross the ignition threshold are amplified.
  3. Ignited content is broadcast to all cognitive modules.

Usage:
    python examples/single_theory.py
"""

import asyncio
import logging
import numpy as np

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # ── 1. Create a Global Workspace ─────────────────────────────────
    workspace = EnhancedGlobalWorkspace(
        capacity=7,               # 7 +/- 2 items (Miller's law)
        ignition_threshold=0.3,   # Salience needed to ignite
        amplification_factor=2.0, # Post-ignition boost
    )

    # ── 2. Build candidates competing for conscious access ───────────
    candidates = [
        WorkspaceCandidate(
            content="visual scene of a sunset",
            content_type="perception",
            summary="Sunset over the ocean",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.8,
            emotional_salience=0.7,
            novelty_score=0.3,
        ),
        WorkspaceCandidate(
            content="recalled memory of a conversation",
            content_type="memory",
            summary="Yesterday's interesting discussion",
            source=WorkspaceCandidateSource.MEMORY,
            activation_level=0.5,
            emotional_salience=0.4,
            novelty_score=0.2,
        ),
        WorkspaceCandidate(
            content="background noise detection",
            content_type="perception",
            summary="Faint sound of distant traffic",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.2,
            emotional_salience=0.0,
            novelty_score=0.1,
        ),
        WorkspaceCandidate(
            content="goal: finish writing report",
            content_type="intention",
            summary="Task deadline approaching",
            source=WorkspaceCandidateSource.LLM,
            activation_level=0.6,
            goal_relevance=0.9,
            novelty_score=0.1,
        ),
    ]

    # ── 3. Run a consciousness cycle ─────────────────────────────────
    state = await workspace.process_consciousness_cycle(candidates)

    # ── 4. Inspect results ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GLOBAL WORKSPACE — CONSCIOUSNESS CYCLE RESULT")
    print("=" * 60)
    print(f"  Conscious?           {state.is_conscious}")
    print(f"  Ignition events:     {state.ignition_events}")
    print(f"  Integration level:   {state.integration_level:.3f}")
    print(f"  Broadcast coverage:  {state.broadcast_coverage:.1%}")
    print(f"  Attention focus:     {state.attention_focus}")

    if state.primary_content:
        pc = state.primary_content
        print(f"\n  Primary content:")
        print(f"    Summary:   {pc.candidate.summary}")
        print(f"    Source:    {pc.candidate.source.value}")
        print(f"    Salience:  {pc.salience:.3f}")
        print(f"    Rank:      {pc.competition_rank}")

    print(f"\n  All workspace contents ({len(state.workspace_contents)}):")
    for wc in state.workspace_contents:
        print(f"    - {wc.candidate.summary} (salience={wc.salience:.3f})")

    print("\n  Attention distribution:")
    for source, weight in state.attention_distribution.items():
        bar = "#" * int(weight * 30)
        print(f"    {source:<12s} {weight:.3f}  {bar}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
