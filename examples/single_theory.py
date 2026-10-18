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
experience. The integration level needs neural signals from at least two
sources and the broadcast coverage needs a registered receiver; this example
passes neither, so both print as not measured rather than as a number.

Needs the ``fep`` extra (pymdp), because the workspace builds the active
inference module; without it the example prints one line that names the
extra and exits 2. Opens no network connection.

Usage:
    python examples/single_theory.py
"""

import asyncio
import logging
import sys

from mtc.cli.missing_extra import explain_missing_extra
from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)

logging.basicConfig(level=logging.WARNING)


async def run():
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
    integration = state.integration_level
    coverage = state.broadcast_coverage
    print(
        "  Integration level:    "
        + (
            "not measured (no neural signals were passed)"
            if integration is None
            else f"{integration:.3f}"
        )
    )
    print(
        "  Broadcast coverage:   "
        + (
            "unmeasured (no receiver is registered)"
            if coverage is None
            else f"{coverage:.1%}"
        )
    )
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
