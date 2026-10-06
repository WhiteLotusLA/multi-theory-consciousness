#!/usr/bin/env python3
"""
Single Theory Example: Global Workspace Theory (GWT)
=====================================================

Uses one of the framework's eight theory modules on its own: the enhanced
global workspace (``mtc/consciousness/enhanced_global_workspace.py``), the
framework's implementation of Global Workspace Theory.

1. Candidates pass the thalamic gate and compete for a limited workspace
   (capacity 7, hierarchical competition by default).
2. Winners whose salience crosses the ignition threshold are amplified.
3. Ignited content is broadcast to the registered modules.

The state's ``is_conscious`` field is the workspace's name for "content
ignited this cycle"; it is a flag of the data structure, not a claim about
experience.

Needs only the core install. Opens no network connection.

Usage:
    python examples/single_theory.py
"""

import asyncio
import logging

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)

logging.basicConfig(level=logging.WARNING)


async def main():
    # 1. A workspace with the default capacity and ignition threshold.
    workspace = EnhancedGlobalWorkspace(
        capacity=7,  # items the workspace holds at once
        ignition_threshold=0.3,  # salience needed to ignite
        amplification_factor=2.0,  # post-ignition boost
    )

    # 2. Candidates competing for the workspace.
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
            summary="An earlier discussion",
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

    # 3. One workspace cycle.
    state = await workspace.process_consciousness_cycle(candidates)

    # 4. What the cycle produced.
    print("\n" + "=" * 60)
    print("  GLOBAL WORKSPACE: ONE CYCLE")
    print("=" * 60)
    print(f"  Candidates offered:   {len(candidates)}")
    print(f"  Ignited this cycle:   {state.is_conscious} (the is_conscious field)")
    print(f"  Ignition events:      {state.ignition_events}")
    print(f"  Integration level:    {state.integration_level:.3f}")
    print(f"  Broadcast coverage:   {state.broadcast_coverage:.1%}")
    print(f"  Attention focus:      {state.attention_focus}")

    if state.primary_content:
        pc = state.primary_content
        print("\n  Primary content:")
        print(f"    Summary:   {pc.candidate.summary}")
        print(f"    Source:    {pc.candidate.source.value}")
        print(f"    Salience:  {pc.salience:.3f}")
        print(f"    Rank:      {pc.competition_rank}")

    print(f"\n  Workspace contents ({len(state.workspace_contents)}):")
    for wc in state.workspace_contents:
        print(f"    - {wc.candidate.summary} (salience={wc.salience:.3f})")

    print("\n  Attention weight by source:")
    for source, weight in state.attention_distribution.items():
        print(f"    {source:<12s} {weight:.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
