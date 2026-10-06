#!/usr/bin/env python3
"""
Continuous Thought Machine, Offline
===================================

Runs the continuous thought machine for a fixed number of ticks with nothing
outside this process: an in-memory thought store and gist store, an
``LLMBrain`` on the deterministic mock provider (its replies start with
``[mock]``; no model runs), and the hash embedder passed with the explicit
non-semantic opt-in, so novelty here measures word overlap, not meaning. The
demo cadence profile (``CTM_PROFILE=demo``) keeps the rhythm visible; time is
simulated, so each tick advances a clock by the machine's own next delay
instead of waiting for it.

The run prints each tick, the surfaced thoughts (the ones the novelty gate let
through, as the conversation prompt would see them), the machine's status and
a final ``CTM_SUMMARY {json}`` line. The provider state file goes to a
temporary state directory (``--state-dir`` to choose one); nothing is written
into the repository. Thought text here comes from the shipped neutral seed
pack and from the mock provider.

Usage:
    python examples/continuous_thought.py [--ticks 12] [--state-dir DIR]
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mtc.brain.brain import LLMBrain
from mtc.consciousness.continuous import build_thought_machine
from mtc.consciousness.continuous.surfacing import get_surfaced_thoughts
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2000, 1, 1, 9, 0, tzinfo=timezone.utc)


class SimulatedClock:
    """Wall-clock stand-in that moves only when the example advances it."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _settings(state_dir: Path) -> Settings:
    return Settings(
        PROJECT_ROOT=str(ROOT),
        DATA_DIRECTORY=str(state_dir / "data"),
        PROVIDER_STATE_PATH=str(state_dir / "data" / "provider_state.json"),
        LOG_DIRECTORY=str(state_dir / "logs"),
        LLM_PROVIDER="mock",
        CTM_PROFILE="demo",
        CTM_STORE="memory",
        CTM_CONFIG_DIR="config/ctm",
        DEBUG=False,
    )


async def run_example(ticks: int, state_dir: Path) -> Dict[str, Any]:
    started = time.monotonic()
    settings = _settings(state_dir)
    brain = LLMBrain.from_settings(settings)
    health = await brain.initialize()
    clock = SimulatedClock()
    machine = build_thought_machine(
        settings=settings,
        brain=brain,
        embedder=HashEmbedder(dim=128),
        allow_non_semantic=True,
        # nothing else uses the brain in this example, so it is never contended
        contention_guard=lambda: False,
        clock=clock,
        rng=random.Random(7),
    )

    print("continuous thought machine, offline example")
    print(
        f"brain provider: mock (healthy: {health.ok}); replies start with [mock] "
        "and are test text, not a model's answer"
    )
    print(
        "embedder: hash, NON-SEMANTIC, explicit opt-in for this example; "
        "novelty measures word overlap, not meaning"
    )
    config = machine.config
    print(
        f"profile: {config.profile} (awake cadence {config.cadence.awake} Hz, "
        f"gist every {config.gist.every} stored thoughts); store: in memory"
    )
    print()

    results = []
    for number in range(1, ticks + 1):
        result = await machine.tick()
        results.append(result)
        flags = [
            name
            for name in ("generated", "expressed", "persisted", "upgraded", "duplicate")
            if getattr(result, name)
        ]
        elapsed = (clock() - START).total_seconds()
        print(
            f"tick {number:2d} at +{elapsed:7.1f}s: {', '.join(flags) or 'nothing'}"
            f"{'' if result.quiet_reason is None else f' (quiet: {result.quiet_reason})'}"
        )
        clock.advance(machine.next_delay(result))

    surfaced = await get_surfaced_thoughts(
        machine.store,
        limit=5,
        exclude_type_prefixes=tuple(config.surfacing.excluded_type_prefixes),
    )
    print()
    print("surfaced thoughts (newest first):")
    for thought in surfaced:
        print(f"  [{thought.thought_type}] {thought.content}")
    if not surfaced:
        print("  none")

    status = machine.get_status()
    counters = status["counters"]
    print()
    print(f"quiet reason: {status['quiet_reason']}")
    print(f"current frequency: {status['current_frequency']:.4f} Hz")
    print(f"counters: {json.dumps(counters, sort_keys=True)}")
    print(f"failures: {json.dumps(status['failures'], sort_keys=True)}")
    await brain.shutdown()

    summary = {
        "ok": not status["failures"] and counters.get("generated", 0) > 0,
        "ticks": ticks,
        "generated": counters.get("generated", 0),
        "expressed": counters.get("expressed", 0),
        "persisted": counters.get("persisted", 0),
        "upgraded": counters.get("llm_upgrades", 0),
        "duplicates": counters.get("duplicates_filtered", 0),
        "gists": counters.get("gists", 0),
        "surfaced": len(surfaced),
        "quiet_reason": status["quiet_reason"],
        "embedder": status["novelty"]["embedder"],
        "non_semantic_opt_in": status["novelty"]["non_semantic_opt_in"],
        "provider": "mock",
        "failures": status["failures"],
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ticks", type=int, default=12, help="ticks to run")
    parser.add_argument(
        "--state-dir", type=Path, default=None, help="where state files go"
    )
    args = parser.parse_args(argv)
    if args.ticks < 1:
        parser.error("--ticks must be at least 1")
    logging.basicConfig(level=logging.WARNING)
    if args.state_dir is not None:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        summary = asyncio.run(run_example(args.ticks, args.state_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="mtc_ctm_example_") as tmp:
            summary = asyncio.run(run_example(args.ticks, Path(tmp)))
    print("CTM_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
