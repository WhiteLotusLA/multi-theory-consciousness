"""
PostgreSQL Setup and Check
==========================

Probes, initializes and checks the PostgreSQL database against the rendered
migrations (``mtc.db.migrations``).

- ``probe`` opens a one-connection pool through
  ``mtc.memory.clients.create_postgres_pool`` and runs ``SELECT 1`` within
  ``timeout_s``.
- ``apply_migrations`` runs the migration statements, in file order, in one
  transaction through ``apply_schema``; every statement is
  ``CREATE ... IF NOT EXISTS``, so a second run changes nothing.
- ``check_schema`` compares the tables and indexes in the current schema with
  the names the same statements create.
- ``init`` and ``check`` wrap them and return ``CheckResult`` values; ``init``
  reports which tables and indexes it created.

A missing database, a refused authentication or a permission error is
``failed`` with a plain hint (the role and database come from
``scripts/db/init_postgres.sh`` or the Compose file); a refused connection or a
timeout is ``unreachable``; a missing ``asyncpg`` is ``failed`` naming the
``postgres`` extra. The password is never echoed.

Config keys read at call time (through the client factory): ``POSTGRES_HOST``,
``POSTGRES_PORT``, ``POSTGRES_DB``, ``POSTGRES_USER``, ``POSTGRES_PASSWORD`` and
the ``DB_POOL_*`` keys. The database holds conversation text, memories and the
safety audit; nothing here reads or prints rows, and nothing drops anything.
"""

from __future__ import annotations

import asyncio
from typing import Any, FrozenSet, List, Optional, Sequence, Tuple

from mtc.core.checks import CheckResult, CheckStatus, describe_error, status_for_error
from mtc.core.config import Settings, get_settings
from mtc.db.migrations import expected_names, statements_in_order
from mtc.memory.clients import apply_schema, create_postgres_pool

STORE = "postgres"
EXTRA = "postgres"
CHECK_NAME = "store:postgres"
ROLE_SCRIPT_HINT = (
    "create the role and database with scripts/db/init_postgres.sh (or the "
    "Compose file), then run mtc db init"
)
CREDENTIALS_HINT = "check MTC_POSTGRES_USER and MTC_POSTGRES_PASSWORD in .env"

_TABLES_SQL = (
    "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = current_schema()"
)
_INDEXES_SQL = (
    "SELECT indexname FROM pg_catalog.pg_indexes WHERE schemaname = current_schema()"
)


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def _where(active: Settings) -> str:
    return (
        f"{active.POSTGRES_HOST}:{active.POSTGRES_PORT} database "
        f"{active.POSTGRES_DB} as {active.POSTGRES_USER}"
    )


def error_result(
    exc: BaseException, active: Settings, name: str = CHECK_NAME
) -> CheckResult:
    """The ``CheckResult`` for an exception raised while reaching PostgreSQL."""
    if isinstance(exc, ImportError):
        return CheckResult(
            name,
            CheckStatus.FAILED,
            f"{exc}",
            data={"missing_extra": EXTRA},
        )
    status = status_for_error(exc)
    detail = describe_error(exc, secrets=[active.POSTGRES_PASSWORD])
    names = " ".join(cls.__name__ for item in _chain(exc) for cls in type(item).__mro__)
    if status is CheckStatus.UNREACHABLE:
        hint = f"PostgreSQL at {_where(active)} did not answer"
    elif "InvalidCatalogName" in names:
        hint = f"the database {active.POSTGRES_DB} does not exist; {ROLE_SCRIPT_HINT}"
    elif "Password" in names or "Auth" in names:
        hint = f"authentication was refused; {CREDENTIALS_HINT}"
    elif "Privilege" in names or "Permission" in names:
        hint = f"the role {active.POSTGRES_USER} lacks a permission; {ROLE_SCRIPT_HINT}"
    else:
        hint = f"PostgreSQL at {_where(active)} refused the operation"
    return CheckResult(name, status, f"{hint} ({detail})")


def _chain(exc: BaseException) -> List[BaseException]:
    seen: List[BaseException] = []
    current: Any = exc
    while isinstance(current, BaseException) and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


async def _open_pool(active: Settings, timeout_s: float) -> Any:
    return await asyncio.wait_for(
        create_postgres_pool(active, min_size=1, max_size=1, timeout=timeout_s),
        timeout=timeout_s + 1.0,
    )


async def _close(pool: Any) -> None:
    closer = getattr(pool, "close", None)
    if callable(closer):
        try:
            await asyncio.wait_for(closer(), timeout=5.0)
        except Exception:  # closing is best effort after the result is known
            pass


async def probe(
    settings: Optional[Settings] = None, *, timeout_s: float = 2.0
) -> CheckResult:
    """Quick reachability check: connect and run ``SELECT 1``."""
    active = _settings(settings)
    pool = None
    try:
        pool = await _open_pool(active, timeout_s)
        await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=timeout_s)
    except Exception as exc:
        return error_result(exc, active)
    finally:
        if pool is not None:
            await _close(pool)
    return CheckResult(CHECK_NAME, CheckStatus.OK, f"answered at {_where(active)}")


async def existing_names(pool: Any) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """The table and index names in the current schema."""
    tables = await pool.fetch(_TABLES_SQL)
    indexes = await pool.fetch(_INDEXES_SQL)
    return (
        frozenset(str(row[0]).lower() for row in tables),
        frozenset(str(row[0]).lower() for row in indexes),
    )


async def apply_migrations(
    pool: Any, statements: Optional[Sequence[str]] = None
) -> int:
    """Run the migration statements in file order in one transaction."""
    chosen = list(statements) if statements is not None else statements_in_order()
    return await apply_schema(pool, chosen)


async def check_schema(
    pool: Any, statements: Optional[Sequence[str]] = None
) -> List[CheckResult]:
    """Compare the schema's tables and indexes with the migrations' names."""
    tables, indexes = expected_names(statements)
    found_tables, found_indexes = await existing_names(pool)
    results = []
    for kind, expected, found in (
        ("tables", tables, found_tables),
        ("indexes", indexes, found_indexes),
    ):
        missing = sorted(expected - found)
        name = f"postgres:{kind}"
        if missing:
            results.append(
                CheckResult(
                    name,
                    CheckStatus.FAILED,
                    f"{len(missing)} of {len(expected)} {kind} missing: "
                    f"{', '.join(missing)}; run mtc db init",
                    data={"missing": missing, "expected": len(expected)},
                )
            )
        else:
            results.append(
                CheckResult(
                    name,
                    CheckStatus.OK,
                    f"all {len(expected)} {kind} present",
                    data={"missing": [], "expected": len(expected)},
                )
            )
    return results


async def init(
    pool: Any = None,
    settings: Optional[Settings] = None,
    *,
    statements: Optional[Sequence[str]] = None,
    timeout_s: float = 10.0,
) -> CheckResult:
    """Apply the migrations; report the tables and indexes that were created."""
    active = _settings(settings)
    owned = pool is None
    chosen = list(statements) if statements is not None else statements_in_order()
    tables, indexes = expected_names(chosen)
    try:
        if owned:
            pool = await _open_pool(active, timeout_s)
        before_tables, before_indexes = await existing_names(pool)
        count = await apply_migrations(pool, chosen)
        after_tables, after_indexes = await existing_names(pool)
    except Exception as exc:
        return error_result(exc, active)
    finally:
        if owned and pool is not None:
            await _close(pool)
    created_tables = sorted((after_tables - before_tables) & tables)
    created_indexes = sorted((after_indexes - before_indexes) & indexes)
    if created_tables or created_indexes:
        detail = (
            f"applied {count} statements: created {len(created_tables)} tables "
            f"and {len(created_indexes)} indexes"
        )
    else:
        detail = f"applied {count} statements: schema already present, nothing created"
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        detail,
        data={
            "statements": count,
            "created_tables": created_tables,
            "created_indexes": created_indexes,
        },
    )


async def check(
    pool: Any = None,
    settings: Optional[Settings] = None,
    *,
    statements: Optional[Sequence[str]] = None,
    timeout_s: float = 10.0,
) -> List[CheckResult]:
    """Check the schema; an unreachable server is one ``unreachable`` result."""
    active = _settings(settings)
    owned = pool is None
    try:
        if owned:
            pool = await _open_pool(active, timeout_s)
        return await check_schema(pool, statements)
    except Exception as exc:
        return [error_result(exc, active)]
    finally:
        if owned and pool is not None:
            await _close(pool)
