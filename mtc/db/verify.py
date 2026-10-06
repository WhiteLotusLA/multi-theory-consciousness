"""
Database Verification (``mtc db check``)
========================================

Checks every enabled store against the schema registry without changing
anything: the PostgreSQL tables and indexes the migrations create, the
MongoDB indexes by keys and uniqueness, the Qdrant collection with its HNSW
parameters and payload indexes, the Neo4j constraints by label, properties and
type, and Redis's ping, eviction policy and key families. A disabled store is
``skipped``; a store that refuses the connection is ``unreachable``.

``probe_store`` is the quick reachability check ``mtc doctor`` runs for one
store (a two-second timeout; ``skipped`` when the store is disabled).

Usage: ``python3 -m mtc.db.verify [--only STORE...] [--json] [--strict]
[--env-file PATH]``. Exit codes: 0 every required check passed; 1 a required
check failed (with ``--strict`` also any warning or ``insufficient_data``); 2
a usage or configuration error (including a missing client extra); 3 a
required store was unreachable; 130 interrupted.

Config keys read: the five store enable keys and the connection keys the
store modules read. Nothing is written; no stored row, document, point or key
value is read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from mtc.core.checks import (
    CheckResult,
    CheckStatus,
    describe_error,
    render_text,
    status_for_error,
    to_json,
)
from mtc.core.config import Settings, get_settings
from mtc.db.bootstrap import (
    ENABLE_KEYS,
    EXIT_INTERRUPTED,
    EXIT_USAGE,
    STORE_ORDER,
    command_exit_code,
    configured_secrets,
    disabled_result,
    enabled_stores,
    load_settings,
    missing_extras,
    selected_stores,
    settings_error,
    store_module,
)

PROBE_TIMEOUT_S = 2.0


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


async def probe_store(
    store: str,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> CheckResult:
    """One store's quick reachability check; ``skipped`` when it is disabled."""
    if store not in STORE_ORDER:
        raise ValueError(f"unknown store {store!r}; known: {', '.join(STORE_ORDER)}")
    active = _settings(settings)
    if not getattr(active, ENABLE_KEYS[store]):
        return disabled_result(store)
    module = store_module(store)
    try:
        return await module.probe(active, timeout_s=timeout_s)
    except Exception as exc:  # a probe reports, it never raises
        detail = describe_error(exc, secrets=configured_secrets(active))
        return CheckResult(f"store:{store}", status_for_error(exc), detail)


async def verify(
    settings: Optional[Settings] = None, *, only: Optional[Sequence[str]] = None
) -> List[CheckResult]:
    """Check every selected, enabled store; a disabled one is ``skipped``."""
    active = _settings(settings)
    enabled = set(enabled_stores(active))
    results: List[CheckResult] = []
    for store in selected_stores(only):
        if store not in enabled:
            results.append(disabled_result(store))
            continue
        module = store_module(store)
        try:
            results.extend(await module.check(settings=active))
        except Exception as exc:  # each store in its own try
            detail = describe_error(exc, secrets=configured_secrets(active))
            results.append(CheckResult(f"store:{store}", status_for_error(exc), detail))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mtc.db.verify",
        description="Check the enabled stores against the schema registry.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=STORE_ORDER,
        metavar="STORE",
        help=f"limit to these stores ({', '.join(STORE_ORDER)})",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="count warnings and insufficient_data as failures",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="environment file to read (default: .env in the working directory)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        active = load_settings(args.env_file)
    except Exception as exc:
        print(f"mtc db check: {settings_error(exc)}", file=sys.stderr)
        return EXIT_USAGE
    try:
        results = asyncio.run(verify(active, only=args.only))
    except KeyboardInterrupt:
        print("mtc db check: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    code = command_exit_code(results, strict=args.strict)
    if args.json:
        payload = to_json(results, strict=args.strict)
        payload["exit_code"] = code
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(results, title="mtc db check"))
        for extra in missing_extras(results):
            print(
                f"install the {extra!r} extra: "
                f"pip install multi-theory-consciousness[{extra}]"
            )
        if not any(r.status is not CheckStatus.SKIPPED for r in results):
            print("no store is enabled; everything runs in process memory")
    return code


if __name__ == "__main__":
    sys.exit(main())
