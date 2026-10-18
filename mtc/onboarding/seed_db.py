"""
Store Seeding and the Install Date
==================================

What the bootstrap does after the schema exists.

- ``record_birth_date`` writes ``MTC_SYSTEM_BIRTH_DATE`` into ``.env`` once:
  the calendar day of the first initialization in ``TIMEZONE``. An existing
  non-empty value (in the file or in the loaded Settings) is kept and never
  overwritten.
- ``render_seed_blocks`` renders the three core memory blocks with the memory
  package's ``render_core_blocks`` unchanged and appends one sentence to the
  identity block when an install date exists ("This system was first initialized on
  YYYY-MM-DD."), which records the install date without claiming a birthday.
  The blocks carry no trust value, bond or other date.
- ``seed_stores`` seeds core memory through
  ``EnhancedMemoryManager.seed_core_memory`` (existing blocks are kept) on a
  PostgreSQL-only memory stack when PostgreSQL is enabled, and merges the
  system and creator person nodes and the neutral seed concepts through
  ``KnowledgeGraphManager.seed_core_entities(include_creator=True)`` only when
  Neo4j is enabled and graph seeding is asked for. With no persistent store it
  seeds nothing and says that core memory is seeded at run time from the
  persona.

Config keys read: ``TIMEZONE``, ``SYSTEM_BIRTH_DATE``, ``POSTGRES_ENABLED``,
``NEO4J_ENABLED``, ``PERSONA_PATH`` (through ``load_persona``), the keys the
core blocks read (``SYSTEM_NAME``, ``CREATOR_NAME``, ``CREATOR_RELATIONSHIP``,
``CORE_VALUES`` and the Big Five seeds) and the store connection keys. Files
and stores written: ``.env`` (one key), the ``core_memory`` table and the
graph's person and concept nodes. The identity and human blocks and the person
nodes hold real names; ``.env`` also holds secrets, which this module never
reads back or logs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from mtc.core.config import Settings, get_settings
from mtc.core.envfile import read_env, upsert_env

logger = logging.getLogger(__name__)

BIRTH_DATE_KEY = "MTC_SYSTEM_BIRTH_DATE"
BIRTH_DATE_HEADER = "Install date, recorded once by the bootstrap"
INSTALL_DATE_SENTENCE = "This system was first initialized on {date}."
NO_STORE_REASON = (
    "no persistent store is enabled; core memory is seeded at run time from "
    "the persona"
)

Clock = Callable[[], datetime]


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def record_birth_date(
    env_path: Path,
    *,
    settings: Optional[Settings] = None,
    clock: Optional[Clock] = None,
) -> Tuple[str, bool]:
    """Return ``(install date, written)``; write it only when none exists."""
    active = _settings(settings)
    existing = read_env(Path(env_path)).get(BIRTH_DATE_KEY, "").strip()
    if not existing:
        existing = (active.SYSTEM_BIRTH_DATE or "").strip()
    if existing:
        date.fromisoformat(existing)
        return existing, False
    moment = (clock or _utcnow)()
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("the bootstrap clock must return aware datetimes")
    day = moment.astimezone(ZoneInfo(active.TIMEZONE)).date().isoformat()
    upsert_env(
        Path(env_path),
        {BIRTH_DATE_KEY: day},
        keep_existing={BIRTH_DATE_KEY},
        header=BIRTH_DATE_HEADER,
    )
    logger.info("install date recorded in the environment file")
    return day, True


def render_seed_blocks(
    persona: Optional[Mapping[str, Any]],
    birth_date: Optional[str],
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, str]:
    """The core blocks, with the install-date sentence when a date exists."""
    from mtc.memory.core_blocks import render_core_blocks

    blocks = render_core_blocks(persona, settings=_settings(settings))
    if birth_date:
        day = date.fromisoformat(birth_date).isoformat()
        sentence = INSTALL_DATE_SENTENCE.format(date=day)
        blocks["identity"] = f"{blocks['identity']} {sentence}"
    return blocks


@dataclass
class SeedReport:
    """Which core blocks were written, graph nodes created and what was not."""

    core_blocks: Dict[str, bool] = field(default_factory=dict)
    graph: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)


async def _build_stack(settings: Settings) -> Any:
    """A PostgreSQL-only memory stack (no vectors, no graph)."""
    from mtc.memory.factory import build_memory_manager

    return await build_memory_manager(
        settings, postgres=True, vectors=False, graph=False
    )


def _build_graph(settings: Settings) -> Any:
    from mtc.knowledge.graph_manager import KnowledgeGraphManager

    return KnowledgeGraphManager(settings=settings)


def _seed_graph(settings: Settings) -> Dict[str, int]:
    manager = _build_graph(settings)
    try:
        manager.connect()
        return dict(manager.seed_core_entities(include_creator=True))
    finally:
        manager.close()


async def seed_stores(
    settings: Optional[Settings] = None,
    *,
    persona: Optional[Mapping[str, Any]] = None,
    birth_date: Optional[str] = None,
    graph: bool = True,
) -> SeedReport:
    """Seed core memory and the graph where their stores are enabled.

    A store error propagates to the caller (the bootstrap reports it).
    """
    active = _settings(settings)
    if birth_date and active.SYSTEM_BIRTH_DATE != birth_date:
        active = active.model_copy(update={"SYSTEM_BIRTH_DATE": birth_date})
    if persona is None:
        from mtc.core.persona import load_persona

        persona = load_persona(settings=active)
    report = SeedReport()
    if active.POSTGRES_ENABLED:
        blocks = render_seed_blocks(persona, birth_date, settings=active)
        stack = await _build_stack(active)
        try:
            written = await stack.manager.seed_core_memory(
                blocks=blocks, overwrite=False
            )
        finally:
            await stack.close()
        report.core_blocks = {name: bool(flag) for name, flag in written.items()}
    elif active.NEO4J_ENABLED and graph:
        report.skipped["core_memory"] = (
            "PostgreSQL is disabled; core memory is seeded at run time from the "
            "persona"
        )
    else:
        report.skipped["core_memory"] = NO_STORE_REASON
    if active.NEO4J_ENABLED and graph:
        counts = await asyncio.to_thread(_seed_graph, active)
        report.graph = {name: int(value) for name, value in counts.items()}
    elif not active.NEO4J_ENABLED:
        report.skipped["graph"] = "Neo4j is disabled; no graph nodes are seeded"
    else:
        report.skipped["graph"] = "graph seeding was not requested"
    logger.info(
        "seed: %d core blocks written, graph %s",
        sum(report.core_blocks.values()),
        report.graph or "not seeded",
    )
    return report
