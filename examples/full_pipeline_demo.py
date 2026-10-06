#!/usr/bin/env python3
"""
Full Pipeline, Offline
======================

Builds the whole runtime from Settings with nothing outside this process: the
deterministic mock provider (its replies start with ``[mock]``; no model runs),
every store in process memory, a temporary state and log directory, a
temporary speaker table holding one ``Operator`` entry whose token is generated
when the demo starts, the hash embedder passed with the explicit non-semantic
opt-in (labelled below: novelty then measures word overlap, not meaning) and
the continuous thought machine's demo cadence profile. The neural
orchestrator's emotion bridge and memory consolidator are left out (a default
build costs tens of seconds and hundreds of megabytes), which the output says.

The run ticks the thought machine ten times, sends five turns (three from
``Alice`` without a token, one from ``Operator`` with the token, one from
``Operator`` without it), prints for each turn the four modulation layers as
shipped (narration off: the narrative holds the ungated workspace lines only)
and the trust outcome, then the first-run report, the assessment status and
coverage, the surfaced thoughts with the machine's quiet reason and the status
counters, stops the runtime, runs the privacy audit over its own log directory
and prints one ``DEMO_SUMMARY {json}`` line. The audit's hits on records of
frozen modules known to log an absolute module path are reported by name under
``log_audit.known_frozen_module_log_hits`` (they are not filtered out of the
log); any other hit is unexplained and makes ``ok`` false.

Nothing is written into the repository: every directory sits under
``--state-dir`` (a temporary directory by default). The texts here come from the
shipped neutral seed packs and the mock provider.

Usage:
    python examples/full_pipeline_demo.py [--state-dir DIR] [--ticks 10]
"""

import argparse
import asyncio
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mtc.consciousness.continuous.surfacing import get_surfaced_thoughts
from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.pipeline.runtime import build_runtime

ROOT = Path(__file__).resolve().parents[1]
TOKEN_ENV = "MTC_DEMO_OPERATOR_TOKEN"
DENYLIST = ROOT / "config" / "privacy" / "denylist.txt"
LOCAL_EXTRA_DENYLIST = ROOT / "config" / "privacy" / "denylist.local.txt"
# Log records (logger, source line) of modules closed to edits until Phase 9
# that put an absolute module path into their text: the assessment framework's
# warning that the native HTM interface is absent quotes the ImportError, which
# names the file of the package it imported from. Their audit hits are
# reported under known_frozen_module_log_hits; any other hit fails the run.
KNOWN_FROZEN_LOG_SOURCES: Tuple[Tuple[str, int], ...] = (
    ("mtc.assessment.framework", 147),
)
OPTIONAL_FIELDS = tuple(
    item.name for item in fields(ConsciousnessState) if item.default is None
)
TURNS: Tuple[Tuple[str, str, bool], ...] = (
    ("Alice", "Hello. What do you notice when a conversation begins?", False),
    ("Alice", "Which of your modules took the message just now?", False),
    ("Alice", "Does anything carry over from the earlier questions?", False),
    ("Operator", "This is the operator checking in on the run.", True),
    ("Operator", "I am the operator again, no token this time.", False),
)


def _settings(base: Path) -> Settings:
    speakers = base / "speakers.yaml"
    speakers.write_text(
        "speakers:\n"
        "  - name: Operator\n"
        "    role: creator\n"
        "    trust: 0.9\n"
        f"    token_env: {TOKEN_ENV}\n",
        encoding="utf-8",
    )
    return Settings(
        _env_file=None,
        PROJECT_ROOT=str(ROOT),
        DATA_DIRECTORY=str(base / "data"),
        STATE_DIR=str(base / "state"),
        LOG_DIRECTORY=str(base / "logs"),
        RESULTS_DIR=str(base / "results"),
        AGENT_SPACE_DIR=str(base / "agent_space"),
        SPEAKERS_PATH=str(speakers),
        # The shipped neutral persona: the demo never reads an operator's
        # private persona file.
        PERSONA_PATH="config/persona.example.yaml",
        CREATOR_NAME="Operator",
        TRUSTED_SPEAKERS="Operator",
        LLM_PROVIDER="mock",
        CTM_PROFILE="demo",
        CTM_STORE="memory",
        CTM_CONFIG_DIR="config/ctm",
        DEVICE="cpu",
        DEBUG=False,
    )


def _extra_denylist() -> List[str]:
    """The private extra denylist the audit script reads, when present."""
    from mtc.devtools.privacy_audit import load_denylist

    configured = os.environ.get("PRIVACY_DENYLIST_EXTRA")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise SystemExit(
                "PRIVACY_DENYLIST_EXTRA names a file that does not exist; the "
                "log audit does not run without the private list"
            )
        return load_denylist(path)
    if LOCAL_EXTRA_DENYLIST.is_file():
        return load_denylist(LOCAL_EXTRA_DENYLIST)
    return []


def _log_record(path: Path, line: int) -> Optional[Dict[str, Any]]:
    """The JSON log record on ``line`` (1-based) of ``path``; None otherwise."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for number, text in enumerate(handle, start=1):
            if number == line:
                try:
                    record = json.loads(text)
                except ValueError:
                    return None
                return record if isinstance(record, dict) else None
    return None


def audit_logs(log_dir: Path) -> Dict[str, Any]:
    """The privacy audit over the run's log directory (the exit criterion).

    Every file is scanned as a text copy with the public denylist and the
    private extra list the audit script reads, and no allowlist. A hit on a
    record from a frozen module known to log an absolute module path
    (``KNOWN_FROZEN_LOG_SOURCES``) is reported under
    ``known_frozen_module_log_hits``; every other hit is unexplained and fails
    the run. Hits are reported by file, line and kind; the matched text is
    never printed.
    """
    from mtc.devtools.privacy_audit import load_denylist, run_audit

    files = sorted(p for p in log_dir.rglob("*") if p.is_file())
    known: List[Dict[str, Any]] = []
    unexplained: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mtc_demo_log_audit_") as scratch:
        root = Path(scratch)
        sources: Dict[str, Path] = {}
        for source in files:
            copy = root / f"{source.name}.txt"
            shutil.copyfile(source, copy)
            sources[copy.name] = source
        hits = run_audit(
            sorted(root.iterdir()),
            load_denylist(DENYLIST),
            [],
            extra_denylist=_extra_denylist(),
            root=root,
        )
        for hit in hits:
            source = sources.get(Path(hit.path).name)
            entry: Dict[str, Any] = {
                "file": source.name if source is not None else hit.path,
                "line": hit.line,
                "kind": hit.kind,
            }
            record = _log_record(source, hit.line) if source is not None else None
            site = (
                (record.get("logger"), record.get("line"))
                if record is not None
                else None
            )
            if site in KNOWN_FROZEN_LOG_SOURCES:
                known.append({**entry, "logger": site[0], "source_line": site[1]})
            else:
                unexplained.append(entry)
    return {
        "files": len(files),
        "hits": len(known) + len(unexplained),
        "unexplained_hits": unexplained,
        "known_frozen_module_log_hits": known,
    }


def _step_failures(runtime: Any) -> Dict[str, int]:
    pipeline = runtime.pipeline.status()
    counts: Dict[str, int] = {}
    for label, failures in (
        ("orchestrator", runtime.orchestrator.status()["failures"]),
        ("pipeline", pipeline["failures"]),
        ("post_turn", pipeline["post_turn"]["failures"]),
        ("persistence", pipeline["persistence"]["failures"]),
        ("ctm", runtime.ctm.get_status()["failures"]),
        ("background", {t.name: t.failures for t in runtime.background if t.failures}),
    ):
        for step, count in dict(failures).items():
            counts[f"{label}.{step}"] = int(count)
    return counts


def _unexplained(runtime: Any) -> List[str]:
    state = runtime.orchestrator._last_state
    absent = runtime.pipeline.status()["last_turn"].get("absent_signals") or {}
    if state is None:
        return ["<no state>"]
    return [
        name
        for name in OPTIONAL_FIELDS
        if getattr(state, name, None) is None and name not in absent
    ]


def _print_layers(snapshot: Dict[str, Any]) -> None:
    narrative = snapshot.get("narrative") or ""
    print("    layer 1 narrative (narration off, ungated lines only):")
    for line in narrative.splitlines() or ["      (empty)"]:
        print(f"      {line}")
    params = snapshot.get("generation_params") or {}
    print(f"    layer 2 temperature: {params.get('temperature')}")
    directives = snapshot.get("directives") or []
    print(f"    layer 3 directives ({len(directives)}):")
    for directive in directives:
        print(f"      - {directive}")
    awareness = snapshot.get("self_awareness") or ""
    print(
        "    layer 4 self-awareness: "
        + (awareness if awareness else "(empty: NARRATE_INNER_STATE is off)")
    )
    absent = snapshot.get("absent_signals") or {}
    print(f"    absent signals named: {json.dumps(absent, sort_keys=True)}")


async def run_demo(base: Path, ticks: int) -> Dict[str, Any]:
    started = time.monotonic()
    token = secrets.token_hex(16)
    os.environ[TOKEN_ENV] = token
    cfg = _settings(base)
    print("full pipeline, offline demo")
    print(
        "brain provider: mock; replies start with [mock] and are test text, not a "
        "model's answer"
    )
    print(
        "embedder: hash, NON-SEMANTIC, explicit opt-in for this demo; novelty "
        "measures word overlap, not meaning"
    )
    print(
        "neural substrates: the emotion bridge and memory consolidator are left "
        "out of this run (substrates=False)"
    )
    print("stores: every store in process memory (every *_ENABLED key is false)")
    print(
        "persona: the shipped neutral example (config/persona.example.yaml); "
        "the file log under the state directory keeps the default level and is "
        "audited after the stop"
    )
    runtime = await build_runtime(
        cfg,
        embedder=HashEmbedder(dim=128),
        allow_non_semantic=True,
        substrates=False,
    )
    await runtime.start()
    stores = {name: report["backend"] for name, report in runtime.stores.items()}
    print(f"store backends: {json.dumps(stores, sort_keys=True)}")
    print(f"background tasks: {[task.name for task in runtime.background]}")
    print()

    print(f"continuous thought: {ticks} ticks on the demo profile")
    for number in range(1, ticks + 1):
        result = await runtime.ctm.tick()
        flags = [
            name
            for name in ("generated", "expressed", "persisted", "upgraded")
            if getattr(result, name)
        ]
        quiet = (
            "" if result.quiet_reason is None else f" (quiet: {result.quiet_reason})"
        )
        print(f"  tick {number:2d}: {', '.join(flags) or 'nothing'}{quiet}")
    print()

    cycle_ok_all = True
    unexplained: List[str] = []
    turns_sent = 0
    spoofed_trusted: Optional[bool] = None
    authenticated_trusted: Optional[bool] = None
    authenticated_before = runtime.speaker_trust.status()["authenticated"]
    conversation_ids: List[str] = []
    for index, (speaker, message, with_token) in enumerate(TURNS, start=1):
        result = await runtime.pipeline.send_message(
            message, speaker=speaker, speaker_token=token if with_token else None
        )
        turns_sent += 1
        cycle_ok_all = cycle_ok_all and bool(result.cycle_ok)
        unexplained.extend(_unexplained(runtime))
        if result.conversation_id and result.conversation_id not in conversation_ids:
            conversation_ids.append(result.conversation_id)
        label = result.speaker_label
        # The trust outcome is read back from the runtime: the pipeline records
        # what SpeakerTrust verified for this turn, and the trust source's
        # authenticated count moves only when a token matched.
        last_turn = runtime.pipeline.status()["last_turn"]
        authenticated = bool(last_turn["authenticated"])
        trusted = bool(last_turn["trusted"])
        authenticated_count = runtime.speaker_trust.status()["authenticated"]
        token_matched = authenticated_count - authenticated_before == 1
        authenticated_before = authenticated_count
        if speaker == "Operator":
            if with_token:
                authenticated_trusted = trusted and token_matched
            else:
                spoofed_trusted = trusted or token_matched
        print(
            f"turn {index}: speaker given {speaker!r}, "
            f"{'with' if with_token else 'without'} token"
        )
        print(
            f"    trust outcome (read from the runtime): label {label!r}, "
            f"authenticated {authenticated}, token matched {token_matched}, "
            f"trusted {trusted}, safety gate {last_turn['safety_gate']}"
        )
        print(
            f"    result: success {result.success}, cycle_ok {result.cycle_ok}, "
            f"provider {result.provider_id}, error {result.error_code}"
        )
        print(f"    reply: {(result.response or '')[:100]}")
        turns = await runtime.turn_store.get_conversation_turns(result.conversation_id)
        assistant = [t for t in turns if t["role"] == "assistant"]
        snapshot = (
            assistant[-1].get("metadata", {}).get("consciousness_snapshot", {})
            if assistant
            else {}
        )
        _print_layers(snapshot)
        print()

    print("first-run report:")
    print(json.dumps(runtime.orchestrator.first_run_report(), indent=2, default=str))
    baseline = runtime.orchestrator.baseline.as_dict()
    print(
        f"assessment status: {baseline.get('status')}; coverage: "
        f"{baseline.get('coverage')}; overall: {baseline.get('overall_score')}"
    )
    print()

    ctm_status = runtime.ctm.get_status()
    surfaced = await get_surfaced_thoughts(
        runtime.ctm.store,
        limit=5,
        exclude_type_prefixes=tuple(
            runtime.ctm.config.surfacing.excluded_type_prefixes
        ),
    )
    print(f"surfaced thoughts (quiet reason: {ctm_status['quiet_reason']}):")
    for thought in surfaced:
        print(f"  [{thought.thought_type}] {thought.content}")
    if not surfaced:
        print("  none")
    print()

    failures = _step_failures(runtime)
    print("status counters:")
    print(f"  orchestrator cycles: {runtime.orchestrator.status()['cycles']}")
    print(f"  pipeline turns: {runtime.pipeline.status()['turns']}")
    print(f"  pipeline errors: {json.dumps(runtime.pipeline.status()['errors'])}")
    print(f"  ctm counters: {json.dumps(ctm_status['counters'], sort_keys=True)}")
    print(f"  step failures: {json.dumps(failures, sort_keys=True)}")
    print(f"  flight: {json.dumps(runtime.flight.status()['counts'])}")

    replies_stored = 0
    for conversation_id in conversation_ids:
        turns = await runtime.turn_store.get_conversation_turns(conversation_id)
        replies_stored += sum(1 for t in turns if t["role"] == "assistant")

    await runtime.stop()
    print(f"runtime stopped: {runtime.last_stop_order}")

    log_audit = audit_logs(Path(cfg.LOG_DIRECTORY))
    known = log_audit["known_frozen_module_log_hits"]
    print(
        f"log audit: {log_audit['files']} file(s), {log_audit['hits']} hit(s): "
        f"{len(known)} from frozen modules (known, reported for Phase 9), "
        f"{len(log_audit['unexplained_hits'])} unexplained"
    )
    for entry in known:
        print(
            f"  known: {entry['file']} line {entry['line']} ({entry['kind']}), "
            f"logged by {entry['logger']} at line {entry['source_line']}"
        )
    for entry in log_audit["unexplained_hits"]:
        print(f"  UNEXPLAINED: {entry['file']} line {entry['line']} ({entry['kind']})")

    counters = ctm_status["counters"]
    summary = {
        "turns": turns_sent,
        "cycle_ok_all": cycle_ok_all,
        "step_failures": sum(failures.values()),
        "unexplained_none_fields": sorted(set(unexplained)),
        "provider": runtime.brain.registry.active_id,
        "replies_stored": replies_stored,
        "trust": {
            "spoofed_trusted": spoofed_trusted,
            "authenticated_trusted": authenticated_trusted,
        },
        "assessment_status": baseline.get("status"),
        "ctm": {
            "ticks": ticks,
            "generated": int(counters.get("generated", 0)),
            "expressed": int(counters.get("expressed", 0)),
            "quiet_reason": ctm_status["quiet_reason"],
        },
        "embedder": "non_semantic",
        "non_semantic_opt_in": True,
        "substrates": False,
        "log_audit": log_audit,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    summary["ok"] = bool(
        summary["turns"] == len(TURNS)
        and summary["cycle_ok_all"]
        and summary["step_failures"] == 0
        and not summary["unexplained_none_fields"]
        and summary["replies_stored"] == summary["turns"]
        and summary["provider"] == "mock"
        and summary["trust"]["spoofed_trusted"] is False
        and summary["trust"]["authenticated_trusted"] is True
        and summary["ctm"]["generated"] >= 1
        and log_audit["files"] >= 1
        and not log_audit["unexplained_hits"]
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--state-dir", type=Path, default=None, help="where every file goes"
    )
    parser.add_argument("--ticks", type=int, default=10, help="thought machine ticks")
    args = parser.parse_args(argv)
    if args.ticks < 1:
        parser.error("--ticks must be at least 1")
    if args.state_dir is not None:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        summary = asyncio.run(run_demo(args.state_dir, args.ticks))
    else:
        with tempfile.TemporaryDirectory(prefix="mtc_pipeline_demo_") as tmp:
            summary = asyncio.run(run_demo(Path(tmp), args.ticks))
    print("DEMO_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
