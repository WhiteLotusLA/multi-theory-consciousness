"""
Neo4j Setup and Check
=====================

Probes, initializes and checks the knowledge graph's schema against the
registry's ``CONSTRAINTS`` (uniqueness constraints and the person-name index).

- ``probe`` builds a driver through ``Neo4jConfig.create_driver`` with a short
  connection timeout and calls ``verify_connectivity``.
- ``init`` applies the constraints through
  ``KnowledgeGraphManager.bootstrap_constraints`` (every statement is
  ``CREATE ... IF NOT EXISTS``) and reports which constraints and indexes it
  created.
- ``check`` reads ``SHOW CONSTRAINTS`` and ``SHOW INDEXES`` and matches each
  registry entry by label, properties and type (a uniqueness constraint, or a
  range index for a plain index); names are not compared.

The community edition holds one graph, so the init applies schema only and
seeds nothing (the bootstrap's seeding step merges the system, creator and
seed-concept nodes when Neo4j is enabled). The driver is synchronous; calls
run in ``asyncio.to_thread``. Config keys read at call time: ``NEO4J_URI``,
``NEO4J_USER``, ``NEO4J_PASSWORD`` and ``NEO4J_DATABASE``. The graph holds
people's names and what was learned from them; no node is read here, and
nothing is dropped or deleted.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

from mtc.core.checks import CheckResult, CheckStatus, describe_error, status_for_error
from mtc.core.config import Settings, get_settings
from mtc.core.schema_registry import load_artifact

STORE = "neo4j"
EXTRA = "neo4j"
CHECK_NAME = "store:neo4j"
CONSTRAINTS_PATH = "mtc.knowledge.graph_manager.CONSTRAINTS"

_CONSTRAINT = re.compile(
    r"CREATE\s+CONSTRAINT\s+(\w+)\s+IF\s+NOT\s+EXISTS\s+FOR\s+\(\w+:(\w+)\)\s+"
    r"REQUIRE\s+(.+?)\s+IS\s+UNIQUE",
    re.IGNORECASE,
)
_INDEX = re.compile(
    r"CREATE\s+INDEX\s+(\w+)\s+IF\s+NOT\s+EXISTS\s+FOR\s+\(\w+:(\w+)\)\s+ON\s+"
    r"\((.+?)\)\s*$",
    re.IGNORECASE,
)
_SHOW_CONSTRAINT_QUERY = (
    "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties"
)
_SHOW_INDEX_QUERY = (
    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties"
)


@dataclass(frozen=True)
class SchemaItem:
    """One registry schema entry: a uniqueness constraint or a plain index."""

    kind: str  # "unique" or "index"
    label: str
    properties: Tuple[str, ...]
    name: str

    def describe(self) -> str:
        props = ", ".join(self.properties)
        return f"{self.kind} on :{self.label}({props})"


def _properties(text: str) -> Tuple[str, ...]:
    text = text.strip().strip("()")
    return tuple(part.strip().split(".", 1)[-1] for part in text.split(","))


def registry_items() -> List[SchemaItem]:
    """Parse the registry's constraint and index statements."""
    items = []
    for statement in load_artifact(CONSTRAINTS_PATH):
        constraint = _CONSTRAINT.match(statement.strip())
        if constraint:
            name, label, props = constraint.groups()
            items.append(SchemaItem("unique", label, _properties(props), name))
            continue
        index = _INDEX.match(statement.strip())
        if index:
            name, label, props = index.groups()
            items.append(SchemaItem("index", label, _properties(props), name))
            continue
        raise ValueError(f"unrecognized graph schema statement: {statement[:80]}")
    return items


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def create_driver(settings: Optional[Settings] = None, *, timeout_s: float = 2.0):
    """A driver for the configured URI (it connects lazily)."""
    from mtc.knowledge.config import Neo4jConfig

    return Neo4jConfig.from_settings(_settings(settings)).create_driver(
        connection_timeout=timeout_s
    )


def _error(exc: BaseException, active: Settings) -> CheckResult:
    if isinstance(exc, ImportError):
        return CheckResult(
            CHECK_NAME, CheckStatus.FAILED, f"{exc}", data={"missing_extra": EXTRA}
        )
    status = status_for_error(exc)
    detail = describe_error(exc, secrets=[active.NEO4J_PASSWORD])
    if status is CheckStatus.UNREACHABLE:
        hint = f"Neo4j at {active.NEO4J_URI} did not answer"
    elif "auth" in detail.lower():
        hint = "authentication was refused; check MTC_NEO4J_USER and MTC_NEO4J_PASSWORD"
    else:
        hint = f"Neo4j at {active.NEO4J_URI} refused the operation"
    return CheckResult(CHECK_NAME, status, f"{hint} ({detail})")


def _close(driver: Any) -> None:
    closer = getattr(driver, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # closing is best effort after the result is known
            pass


async def probe(
    settings: Optional[Settings] = None, *, timeout_s: float = 2.0
) -> CheckResult:
    """Quick reachability check: ``verify_connectivity``."""
    active = _settings(settings)
    driver = None
    try:
        driver = create_driver(active, timeout_s=timeout_s)
        await asyncio.wait_for(
            asyncio.to_thread(driver.verify_connectivity), timeout=timeout_s + 1.0
        )
    except Exception as exc:
        return _error(exc, active)
    finally:
        if driver is not None:
            _close(driver)
    return CheckResult(CHECK_NAME, CheckStatus.OK, f"answered at {active.NEO4J_URI}")


def _rows(driver: Any, database: str, query: str) -> List[dict]:
    with driver.session(database=database) as session:
        return [dict(record) for record in session.run(query)]


def _present(driver: Any, database: str) -> Set[Tuple[str, str, Tuple[str, ...]]]:
    """``(kind, label, properties)`` for the node constraints and indexes now."""
    found = set()
    for row in _rows(driver, database, _SHOW_CONSTRAINT_QUERY):
        if str(row.get("entityType", "")).upper() != "NODE":
            continue
        if "UNIQUENESS" not in str(row.get("type", "")).upper():
            continue
        for label in row.get("labelsOrTypes") or []:
            found.add(("unique", str(label), tuple(row.get("properties") or ())))
    for row in _rows(driver, database, _SHOW_INDEX_QUERY):
        if str(row.get("entityType", "")).upper() != "NODE":
            continue
        if str(row.get("type", "")).upper() != "RANGE":
            continue
        for label in row.get("labelsOrTypes") or []:
            found.add(("index", str(label), tuple(row.get("properties") or ())))
    return found


def _key(item: SchemaItem) -> Tuple[str, str, Tuple[str, ...]]:
    return (item.kind, item.label, item.properties)


def _init_sync(driver: Any, active: Settings) -> CheckResult:
    from mtc.knowledge.graph_manager import KnowledgeGraphManager

    database = active.NEO4J_DATABASE
    items = registry_items()
    before = _present(driver, database)
    manager = KnowledgeGraphManager(driver=driver, settings=active)
    applied = manager.bootstrap_constraints()
    after = _present(driver, database)
    created = [
        item.describe()
        for item in items
        if _key(item) in after and _key(item) not in before
    ]
    if created:
        detail = f"applied {applied} statements: created {len(created)} entries"
    else:
        detail = f"applied {applied} statements: all present, nothing created"
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        detail,
        data={"applied": applied, "created": created},
    )


async def init(
    driver: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 10.0,
) -> CheckResult:
    """Apply the registry constraints through the graph manager."""
    active = _settings(settings)
    owned = driver is None
    try:
        if owned:
            driver = create_driver(active, timeout_s=timeout_s)
        return await asyncio.to_thread(_init_sync, driver, active)
    except Exception as exc:
        return _error(exc, active)
    finally:
        if owned and driver is not None:
            _close(driver)


def _check_sync(driver: Any, active: Settings) -> List[CheckResult]:
    present = _present(driver, active.NEO4J_DATABASE)
    items = registry_items()
    missing = [item.describe() for item in items if _key(item) not in present]
    if missing:
        return [
            CheckResult(
                "neo4j:constraints",
                CheckStatus.FAILED,
                f"{len(missing)} of {len(items)} missing: {', '.join(missing)}; "
                "run mtc db init",
                data={"missing": missing},
            )
        ]
    return [
        CheckResult(
            "neo4j:constraints",
            CheckStatus.OK,
            f"all {len(items)} constraints and indexes present",
            data={"missing": []},
        )
    ]


async def check(
    driver: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 10.0,
) -> List[CheckResult]:
    """Match every registry entry by label, properties and type."""
    active = _settings(settings)
    owned = driver is None
    try:
        if owned:
            driver = create_driver(active, timeout_s=timeout_s)
        return await asyncio.to_thread(_check_sync, driver, active)
    except Exception as exc:
        return [_error(exc, active)]
    finally:
        if owned and driver is not None:
            _close(driver)
