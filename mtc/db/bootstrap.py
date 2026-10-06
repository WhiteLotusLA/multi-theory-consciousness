"""
Database Bootstrap (``mtc db init``)
====================================

Initializes every enabled store, records the install date and seeds the
stores, idempotently: a second run creates nothing and changes nothing.

Order: each enabled store's ``init`` in ``STORE_ORDER`` (PostgreSQL, MongoDB,
Qdrant, Neo4j, Redis), each in its own try so one failure does not stop the
next; Redis has no schema and gets its checks only. Then
``record_birth_date`` writes ``MTC_SYSTEM_BIRTH_DATE`` once when an
environment file is given (an existing date is kept), then ``seed_stores``
seeds core memory (PostgreSQL) and the graph (Neo4j) for the stores that were
selected and initialized without error. A disabled store is ``skipped``,
never ``ok``.

Usage: ``python3 -m mtc.db.bootstrap [--only STORE...] [--no-seed]
[--env-file PATH] [--json]``. Exit codes: 0 every required step succeeded; 1
a required step failed; 2 a usage or configuration error (including a missing
client extra for an enabled store); 3 a required store was unreachable; 130
interrupted.

Config keys read: the five store enable keys and everything the store modules
and the seeding read (connection keys, ``TIMEZONE``, ``PERSONA_PATH`` and the
identity keys). Files and stores written: the stores' schema artifacts, the
install-date line of ``.env`` (which also holds secrets; this module never
reads them back or prints them), the ``core_memory`` table and the graph's
person and concept nodes, which hold real names. Nothing is dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.core.checks import (
    CheckResult,
    CheckStatus,
    describe_error,
    exit_code_for,
    render_text,
    status_for_error,
    to_json,
)
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

STORE_ORDER = ("postgres", "mongodb", "qdrant", "neo4j", "redis")
ENABLE_KEYS: Dict[str, str] = {
    "postgres": "POSTGRES_ENABLED",
    "mongodb": "MONGODB_ENABLED",
    "qdrant": "QDRANT_ENABLED",
    "neo4j": "NEO4J_ENABLED",
    "redis": "REDIS_ENABLED",
}
STORE_MODULES: Dict[str, str] = {
    "postgres": "mtc.db.postgres_store",
    "mongodb": "mtc.db.mongo_store",
    "qdrant": "mtc.db.qdrant_store",
    "neo4j": "mtc.db.neo4j_store",
    "redis": "mtc.db.redis_store",
}
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def enabled_stores(settings: Optional[Settings] = None) -> List[str]:
    """The stores whose enable key is true, in ``STORE_ORDER``."""
    active = _settings(settings)
    return [store for store in STORE_ORDER if getattr(active, ENABLE_KEYS[store])]


def selected_stores(only: Optional[Sequence[str]]) -> List[str]:
    """The stores ``only`` names (all when empty), in ``STORE_ORDER``."""
    if not only:
        return list(STORE_ORDER)
    unknown = sorted(set(only) - set(STORE_ORDER))
    if unknown:
        raise ValueError(
            f"unknown store {', '.join(unknown)}; known: {', '.join(STORE_ORDER)}"
        )
    return [store for store in STORE_ORDER if store in only]


def configured_secrets(settings: Optional[Settings] = None) -> List[str]:
    """Every configured secret value (the store passwords, the keys), so an
    error detail built from a driver message can mask any of them."""
    from mtc.core.env_export import is_secret_field

    active = _settings(settings)
    values = []
    for name in type(active).model_fields:
        if is_secret_field(name):
            value = getattr(active, name, "")
            if isinstance(value, str) and value:
                values.append(value)
    return values


def store_module(store: str) -> Any:
    """The setup module of a store, imported now."""
    return importlib.import_module(STORE_MODULES[store])


def disabled_result(store: str) -> CheckResult:
    return CheckResult(
        f"store:{store}",
        CheckStatus.SKIPPED,
        f"{ENABLE_KEYS[store]} is false",
        required=False,
    )


@dataclass
class BootstrapResult:
    """Every step's result, the install date and the seed report."""

    results: List[CheckResult] = field(default_factory=list)
    birth_date: Optional[str] = None
    birth_date_written: bool = False
    seed: Optional[Any] = None


async def _init_store(store: str, active: Settings) -> List[CheckResult]:
    module = store_module(store)
    try:
        if store == "redis":
            return list(await module.check(settings=active))
        return [await module.init(settings=active)]
    except Exception as exc:  # each store in its own try
        detail = describe_error(exc, secrets=configured_secrets(active))
        return [CheckResult(f"store:{store}", status_for_error(exc), detail)]


def _ok(results: List[CheckResult], store: str) -> bool:
    mine = [r for r in results if r.name == f"store:{store}"]
    return bool(mine) and all(r.status is CheckStatus.OK for r in mine)


def _seed_detail(report: Any) -> str:
    parts = []
    if report.core_blocks:
        written = [name for name, flag in report.core_blocks.items() if flag]
        kept = [name for name, flag in report.core_blocks.items() if not flag]
        parts.append(
            f"core memory: {len(written)} blocks written, {len(kept)} kept as they were"
        )
    if report.graph:
        rendered = ", ".join(f"{key} {value}" for key, value in report.graph.items())
        parts.append(f"graph: {rendered}")
    for name, reason in report.skipped.items():
        parts.append(f"{name}: {reason}")
    return "; ".join(parts)


async def bootstrap(
    settings: Optional[Settings] = None,
    *,
    env_path: Optional[Path] = None,
    only: Optional[Sequence[str]] = None,
    seed: bool = True,
    clock: Optional[Callable[[], datetime]] = None,
) -> BootstrapResult:
    """Initialize, record the install date and seed (see the module docstring)."""
    active = _settings(settings)
    chosen = selected_stores(only)
    enabled = set(enabled_stores(active))
    outcome = BootstrapResult()
    for store in chosen:
        if store not in enabled:
            outcome.results.append(disabled_result(store))
            continue
        outcome.results.extend(await _init_store(store, active))

    if env_path is not None:
        from mtc.onboarding.seed_db import record_birth_date

        try:
            day, written = record_birth_date(
                Path(env_path), settings=active, clock=clock
            )
        except Exception as exc:
            detail = describe_error(exc, secrets=configured_secrets(active))
            outcome.results.append(
                CheckResult("birth_date", CheckStatus.FAILED, detail)
            )
        else:
            outcome.birth_date, outcome.birth_date_written = day, written
            verb = "recorded" if written else "kept"
            outcome.results.append(
                CheckResult(
                    "birth_date",
                    CheckStatus.OK,
                    f"install date {verb}: {day}",
                    required=False,
                    data={"date": day, "written": written},
                )
            )
    else:
        outcome.birth_date = active.SYSTEM_BIRTH_DATE or None
        outcome.results.append(
            CheckResult(
                "birth_date",
                CheckStatus.SKIPPED,
                "no environment file given; the install date was not recorded",
                required=False,
            )
        )

    if seed:
        outcome.results.append(await _seed(active, chosen, enabled, outcome))
    else:
        outcome.results.append(
            CheckResult("seed", CheckStatus.SKIPPED, "--no-seed given", required=False)
        )
    return outcome


async def _seed(
    active: Settings,
    chosen: List[str],
    enabled: set,
    outcome: BootstrapResult,
) -> CheckResult:
    from mtc.onboarding.seed_db import NO_STORE_REASON, seed_stores

    seed_postgres = "postgres" in chosen and "postgres" in enabled
    seed_graph = "neo4j" in chosen and "neo4j" in enabled
    reasons = []
    if seed_postgres and not _ok(outcome.results, "postgres"):
        seed_postgres = False
        reasons.append("PostgreSQL did not initialize, so core memory was not seeded")
    if seed_graph and not _ok(outcome.results, "neo4j"):
        seed_graph = False
        reasons.append("Neo4j did not initialize, so the graph was not seeded")
    if not seed_postgres and not seed_graph:
        if not reasons:
            if enabled & {"postgres", "neo4j"}:
                reasons.append("no seeded store was selected")
            else:
                reasons.append(NO_STORE_REASON)
        return CheckResult(
            "seed", CheckStatus.SKIPPED, "; ".join(reasons), required=False
        )
    scoped = active.model_copy(
        update={"POSTGRES_ENABLED": seed_postgres, "NEO4J_ENABLED": seed_graph}
    )
    try:
        report = await seed_stores(
            scoped, birth_date=outcome.birth_date, graph=seed_graph
        )
    except Exception as exc:
        detail = describe_error(exc, secrets=configured_secrets(active))
        return CheckResult("seed", status_for_error(exc), detail)
    outcome.seed = report
    detail = _seed_detail(report)
    if reasons:
        detail = f"{detail}; {'; '.join(reasons)}" if detail else "; ".join(reasons)
    return CheckResult("seed", CheckStatus.OK, detail, data=asdict(report))


def missing_extras(results: Sequence[CheckResult]) -> List[str]:
    """The client extras that enabled stores need and that are not installed."""
    return sorted(
        {
            str(result.data["missing_extra"])
            for result in results
            if isinstance(result.data, dict) and result.data.get("missing_extra")
        }
    )


def command_exit_code(results: Sequence[CheckResult], *, strict: bool = False) -> int:
    """The shared exit code, with a missing client extra as a usage error (2)."""
    if missing_extras(results):
        return EXIT_USAGE
    return exit_code_for(results, strict=strict)


def load_settings(env_file: Optional[Path]) -> Settings:
    """Settings from the process environment and ``env_file`` (or ``.env``)."""
    if env_file is not None:
        return Settings(_env_file=str(env_file))
    return Settings()


def settings_error(exc: Exception) -> str:
    """A one-line description of a Settings validation error, by field only."""
    names = []
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            names = sorted({str(e["loc"][0]) for e in errors() if e.get("loc")})
        except Exception:
            names = []
    return "settings could not be loaded" + (
        f" (fields: {', '.join(names)})" if names else ""
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mtc.db.bootstrap",
        description="Initialize the enabled stores, record the install date and "
        "seed core memory and the graph.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=STORE_ORDER,
        metavar="STORE",
        help=f"limit to these stores ({', '.join(STORE_ORDER)})",
    )
    parser.add_argument("--no-seed", action="store_true", help="skip the seeding")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="environment file to read and to record the install date in "
        "(default: .env in the working directory)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        active = load_settings(args.env_file)
    except Exception as exc:
        print(f"mtc db init: {settings_error(exc)}", file=sys.stderr)
        return EXIT_USAGE
    env_path = args.env_file
    if env_path is None:
        env_path = Path(active.PROJECT_ROOT) / ".env"
    try:
        outcome = asyncio.run(
            bootstrap(active, env_path=env_path, only=args.only, seed=not args.no_seed)
        )
    except KeyboardInterrupt:
        print("mtc db init: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    code = command_exit_code(outcome.results)
    if args.json:
        payload = to_json(outcome.results)
        payload["exit_code"] = code
        payload["birth_date"] = outcome.birth_date
        payload["birth_date_written"] = outcome.birth_date_written
        payload["seed"] = asdict(outcome.seed) if outcome.seed is not None else None
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(outcome.results, title="mtc db init"))
        for extra in missing_extras(outcome.results):
            print(
                f"install the {extra!r} extra: pip install multi-theory-consciousness[{extra}]"
            )
    return code


if __name__ == "__main__":
    sys.exit(main())
