"""
Re-embed Backfill (PostgreSQL to Qdrant)
========================================

Gives every PostgreSQL episodic row a Qdrant point: after Qdrant is enabled on
an install that already holds memories, after a lost collection, or (with
``--all``) after a change of ``EMBEDDING_MODEL``.

- ``episodic_memory`` is read by keyset pagination on ``id`` (``WHERE id >
  last ORDER BY id LIMIT batch``), so every row is visited once even while
  rows are added. Rows with ``memory_type = 'archived'`` are skipped, as recall
  hides them.
- For each batch, the point ids ``point_id_for_memory(id)`` that already exist
  are looked up; only the missing rows are embedded (every row with
  ``all_rows``), through ``VectorCoordinator.store_memories``: one embedding
  path, the ``EMBEDDING_MODEL`` seam, with the payload ``postgres_id``,
  ``memory_type``, ``importance`` and ``created_at``. The upsert is idempotent
  because the point id is derived from the row id, so a second run finds
  nothing missing.
- ``dry_run`` counts the missing rows and writes nothing.
- A non-semantic embedder (the hash embedder) is refused unless
  ``allow_non_semantic`` is given; an embedder that cannot load stops the run
  before anything is written for that batch and the command exits non-zero.

The collection must exist (``mtc db init`` creates it at ``VECTOR_DIMENSIONS``,
384 for the default model). A collection of another width, such as an
8192-dimension collection from v0.4.0, is never reused or recreated here: set
``QDRANT_COLLECTION`` to a new name, run ``mtc db init`` and then this
backfill with ``--all``.

Usage: ``python3 -m mtc.db.reembed [--dry-run] [--all] [--batch-size N]
[--limit N] [--json] [--env-file PATH]``. ``--limit`` stops after N rows were
embedded (or counted as missing in a dry run). Exit codes: 0 done; 1 the
embedder or a write failed; 2 a usage or configuration error (a missing
extra, a missing or mismatched collection); 3 a store was unreachable; 130
interrupted.

Config keys read: ``POSTGRES_*`` (through the pool factory),
``QDRANT_HOST``, ``QDRANT_PORT``, ``QDRANT_COLLECTION``, ``VECTOR_DIMENSIONS``
and ``EMBEDDING_MODEL``. The rows read are conversation memories; their text
is embedded and stored in the collection's payload, and never logged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mtc.core.checks import CheckStatus, describe_error, status_for_error
from mtc.core.config import Settings, get_settings
from mtc.memory.clients import create_postgres_pool
from mtc.memory.protocols import point_id_for_memory

logger = logging.getLogger(__name__)

ARCHIVED = "archived"
PAGE_SQL = (
    "SELECT id, content, memory_type, importance, created_at FROM episodic_memory "
    "WHERE id > $1 ORDER BY id LIMIT $2"
)
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3
EXIT_INTERRUPTED = 130


@dataclass
class ReembedResult:
    """Counts of one backfill run and the error that stopped it, if any."""

    scanned: int = 0
    missing: int = 0
    embedded: int = 0
    skipped_archived: int = 0
    errors: Dict[str, str] = field(default_factory=dict)
    unreachable: bool = False


class ReembedRefused(RuntimeError):
    """The run cannot start: a refused embedder or an unusable collection."""


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def _default_embedder(active: Settings) -> Any:
    from mtc.core.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(expected_dim=int(active.VECTOR_DIMENSIONS))


async def _open_pool(active: Settings) -> Any:
    return await create_postgres_pool(active, min_size=1, max_size=2)


def _build_coordinator(active: Settings, embedder: Any, allow_non_semantic: bool):
    from mtc.neural.vectors.coordinator import VectorCoordinator

    return VectorCoordinator(
        embedder=embedder,
        allow_non_semantic=allow_non_semantic,
        create_collection=False,
        settings=active,
    )


def _check_collection(coordinator: Any, active: Settings) -> None:
    from mtc.db.qdrant_store import describe_collection

    client = coordinator.client
    name = coordinator.collection_name
    names = [item.name for item in client.get_collections().collections]
    if name not in names:
        raise ReembedRefused(f"collection {name} does not exist; run mtc db init first")
    size = describe_collection(client.get_collection(name))["size"]
    if size != int(active.VECTOR_DIMENSIONS):
        raise ReembedRefused(
            f"collection {name} holds {size}-dimension vectors but "
            f"VECTOR_DIMENSIONS is {active.VECTOR_DIMENSIONS}; set "
            "MTC_QDRANT_COLLECTION to a new name, run mtc db init, then run "
            "this backfill with --all"
        )


def _existing(coordinator: Any, point_ids: List[str]) -> set:
    records = coordinator.client.retrieve(
        collection_name=coordinator.collection_name,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(record.id) for record in records}


def _created_at(value: Any) -> Optional[str]:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


async def reembed(
    settings: Optional[Settings] = None,
    *,
    batch_size: int = 64,
    limit: Optional[int] = None,
    dry_run: bool = False,
    all_rows: bool = False,
    embedder: Any = None,
    allow_non_semantic: bool = False,
) -> ReembedResult:
    """Backfill missing points (see the module docstring).

    Raises ReembedRefused for a non-semantic embedder without the opt-in, a
    missing collection or one of another width; store and embedding errors
    end the run and are recorded in ``errors``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")
    active = _settings(settings)
    chosen = embedder if embedder is not None else _default_embedder(active)
    if not getattr(chosen, "semantic", False) and not allow_non_semantic:
        raise ReembedRefused(
            "the embedder is not semantic; vector recall compares meanings, so "
            "pass allow_non_semantic=True only for tests and demonstrations"
        )
    result = ReembedResult()
    coordinator = _build_coordinator(active, chosen, allow_non_semantic)
    try:
        await asyncio.to_thread(_check_collection, coordinator, active)
        pool = await _open_pool(active)
        try:
            await _backfill(
                pool, coordinator, result, batch_size, limit, dry_run, all_rows
            )
        finally:
            closer = getattr(pool, "close", None)
            if callable(closer):
                await closer()
    finally:
        closer = getattr(coordinator, "close", None)
        if callable(closer):
            closer()
    logger.info(
        "re-embed: scanned %d, missing %d, embedded %d, archived skipped %d",
        result.scanned,
        result.missing,
        result.embedded,
        result.skipped_archived,
    )
    return result


def _embedding_error(exc: BaseException) -> bool:
    current: Any = exc
    while isinstance(current, BaseException):
        if any("Embedding" in cls.__name__ for cls in type(current).__mro__):
            return True
        current = current.__cause__
    return False


def _record(result: ReembedResult, where: str, exc: BaseException) -> None:
    result.errors[where] = describe_error(exc)
    if status_for_error(exc) is CheckStatus.UNREACHABLE:
        result.unreachable = True


async def _backfill(
    pool: Any,
    coordinator: Any,
    result: ReembedResult,
    batch_size: int,
    limit: Optional[int],
    dry_run: bool,
    all_rows: bool,
) -> None:
    last_id = 0
    while limit is None or result.missing < limit:
        try:
            rows = await pool.fetch(PAGE_SQL, last_id, batch_size)
        except Exception as exc:
            _record(result, "postgres", exc)
            return
        if not rows:
            return
        last_id = int(rows[-1]["id"])
        result.scanned += len(rows)
        live = []
        for row in rows:
            if row["memory_type"] == ARCHIVED:
                result.skipped_archived += 1
            else:
                live.append(row)
        if not live:
            continue
        ids = [point_id_for_memory(int(row["id"])) for row in live]
        if all_rows:
            wanted = list(zip(live, ids))
        else:
            try:
                present = await asyncio.to_thread(_existing, coordinator, ids)
            except Exception as exc:
                _record(result, "qdrant", exc)
                return
            wanted = [(row, pid) for row, pid in zip(live, ids) if pid not in present]
        if limit is not None:
            wanted = wanted[: max(0, limit - result.missing)]
        result.missing += len(wanted)
        if dry_run or not wanted:
            continue
        try:
            stored = await asyncio.to_thread(
                coordinator.store_memories,
                [str(row["content"]) for row, _ in wanted],
                metadatas=[
                    {
                        "postgres_id": int(row["id"]),
                        "memory_type": row["memory_type"],
                        "importance": float(row["importance"] or 0.0),
                        "created_at": _created_at(row["created_at"]),
                    }
                    for row, _ in wanted
                ],
                importances=[float(row["importance"] or 0.0) for row, _ in wanted],
                point_ids=[pid for _, pid in wanted],
            )
        except Exception as exc:
            _record(result, "embedding" if _embedding_error(exc) else "qdrant", exc)
            return
        result.embedded += len(stored)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m mtc.db.reembed",
        description="Backfill Qdrant points for PostgreSQL episodic rows.",
    )
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    parser.add_argument(
        "--all",
        dest="all_rows",
        action="store_true",
        help="re-embed every row (after an EMBEDDING_MODEL change)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="environment file to read (default: .env in the working directory)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from mtc.db.bootstrap import load_settings, settings_error

    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 0):
        print(
            "reembed: --batch-size must be at least 1 and --limit >= 0", file=sys.stderr
        )
        return EXIT_USAGE
    try:
        active = load_settings(args.env_file)
    except Exception as exc:
        print(f"reembed: {settings_error(exc)}", file=sys.stderr)
        return EXIT_USAGE
    try:
        result = asyncio.run(
            reembed(
                active,
                batch_size=args.batch_size,
                limit=args.limit,
                dry_run=args.dry_run,
                all_rows=args.all_rows,
            )
        )
    except KeyboardInterrupt:
        print("reembed: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ReembedRefused, ImportError) as exc:
        print(f"reembed: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:
        status = status_for_error(exc)
        print(f"reembed: {describe_error(exc)}", file=sys.stderr)
        return EXIT_UNREACHABLE if status is CheckStatus.UNREACHABLE else EXIT_FAILED
    code = EXIT_OK
    if result.errors:
        code = EXIT_UNREACHABLE if result.unreachable else EXIT_FAILED
    if args.json:
        payload = asdict(result)
        payload["dry_run"] = bool(args.dry_run)
        payload["exit_code"] = code
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        verb = "would embed" if args.dry_run else "embedded"
        print(
            f"reembed: scanned {result.scanned} rows, {result.missing} missing, "
            f"{verb} {result.missing if args.dry_run else result.embedded}, "
            f"skipped {result.skipped_archived} archived"
        )
        for name, error in result.errors.items():
            print(f"reembed: {name} error: {error}")
    return code


if __name__ == "__main__":
    sys.exit(main())
