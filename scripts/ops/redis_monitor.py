#!/usr/bin/env python3
"""Redis working-memory monitor.

Usage: python scripts/ops/redis_monitor.py [quick | watch | snapshot]
       [--interval SECONDS] [--iterations N]

Modes:

- ``snapshot`` (the default): one full report.
- ``quick``: connectivity, memory use, cache hit rate and error count.
- ``watch``: repeats the full report every ``--interval`` seconds (default 5)
  until interrupted, or for ``--iterations`` rounds when it is given.

The report reads the manager's counters (reads, writes, cache hits and misses,
errors), the server's memory and command statistics, and the keyspace line of
the database named by the ``REDIS_DB`` setting. A figure the server did not
report is shown as ``n/a``; the cache hit rate is ``n/a`` until the manager has
seen a cache access. Connection details come from Settings through
``mtc.memory.redis_manager``; the script holds no host, port or database.

Exit status: 0 when the report was produced, 1 when Redis could not be reached
(or the optional ``redis`` client package is missing; the message names the
extra). The output is ASCII only and holds counters, never key contents.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtc.memory.protocols import StoreUnavailable  # noqa: E402
from mtc.memory.redis_manager import RedisManager  # noqa: E402

RULE = "=" * 60
MODES = ("snapshot", "quick", "watch")
DEFAULT_INTERVAL_S = 5.0


async def _pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _mb(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f} MB"


def _hit_rate(metrics: Dict[str, Any]) -> Optional[float]:
    accesses = int(metrics.get("cache_hits", 0)) + int(metrics.get("cache_misses", 0))
    if accesses == 0:
        return None
    return int(metrics["cache_hits"]) / accesses * 100.0


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _rating(value: Optional[float], good: float, fair: float) -> str:
    if value is None:
        return "no data"
    if value > good:
        return "good"
    if value > fair:
        return "fair"
    return "low"


def _memory_rating(megabytes: Optional[float]) -> str:
    if megabytes is None:
        return "no data"
    if megabytes < 100:
        return "low use"
    if megabytes < 500:
        return "moderate use"
    return "high use"


async def _server_info(manager: Any, section: str) -> Dict[str, Any]:
    """One INFO section from the manager's client; empty when unavailable.

    ``RedisManager`` exposes no public INFO call, so the script reads its
    client attribute.
    """
    client = getattr(manager, "_client", None)
    if client is None:
        return {}
    try:
        info = await client.info(section)
    except Exception:
        return {}
    return dict(info) if info else {}


async def _connect(manager: Any) -> Optional[str]:
    """None when the server answered, otherwise the failure text."""
    try:
        await manager.connect()
    except ImportError as exc:
        return str(exc)
    except StoreUnavailable as exc:
        return str(exc)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _database_name(manager: Any) -> str:
    settings = getattr(manager, "settings", None)
    number = getattr(settings, "REDIS_DB", 0)
    return f"db{number}"


def render_quick(metrics: Dict[str, Any]) -> List[str]:
    rate = _hit_rate(metrics)
    return [
        "Redis: reachable",
        f"  Memory: {_mb(metrics.get('redis_memory_mb'))}",
        f"  Cache hit rate: {_percent(rate)} ({_rating(rate, 70.0, 50.0)})",
        f"  Errors: {metrics.get('errors', 0)}",
    ]


async def render_full(manager: Any, metrics: Dict[str, Any], number: int) -> List[str]:
    rate = _hit_rate(metrics)
    lines = [
        RULE,
        f"Redis working memory monitor (report {number})",
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        RULE,
        "",
        "Counters:",
        f"  Total reads: {metrics.get('total_reads', 0)}",
        f"  Total writes: {metrics.get('total_writes', 0)}",
        f"  Cache hits: {metrics.get('cache_hits', 0)}",
        f"  Cache misses: {metrics.get('cache_misses', 0)}",
        f"  Cache hit rate: {_percent(rate)}",
        f"  Errors: {metrics.get('errors', 0)}",
        "",
        "Server memory:",
        f"  Current: {_mb(metrics.get('redis_memory_mb'))}",
        f"  Peak: {_mb(metrics.get('redis_peak_memory_mb'))}",
        "",
    ]
    stats = await _server_info(manager, "stats")
    lines += [
        "Server stats:",
        f"  Total commands: {stats.get('total_commands_processed', 'n/a')}",
        f"  Operations per second: {stats.get('instantaneous_ops_per_sec', 'n/a')}",
        "",
    ]
    keyspace = await _server_info(manager, "keyspace")
    name = _database_name(manager)
    entry = keyspace.get(name)
    lines.append(f"Keyspace ({name}):")
    if entry:
        lines.append(f"  Keys: {entry.get('keys', 'n/a')}")
        lines.append(f"  Expires: {entry.get('expires', 'n/a')}")
    else:
        lines.append("  no keys reported")
    lines += [
        "",
        "Summary (thresholds: hit rate above 70 percent good, above 50 fair;"
        " memory under 100 MB low, under 500 MB moderate):",
        f"  Cache performance: {_rating(rate, 70.0, 50.0)}",
        f"  Memory: {_memory_rating(metrics.get('redis_memory_mb'))}",
        f"  Errors: {'none' if not metrics.get('errors') else metrics['errors']}",
        RULE,
    ]
    return lines


async def _run(
    mode: str, interval: float, iterations: Optional[int], manager: Any
) -> int:
    failure = await _connect(manager)
    if failure is not None:
        print(f"Redis could not be reached: {failure}", file=sys.stderr)
        return 1
    try:
        if mode == "quick":
            print("\n".join(render_quick(await manager.get_metrics())))
            return 0
        rounds = iterations if iterations is not None else (0 if mode == "watch" else 1)
        number = 0
        while rounds == 0 or number < rounds:
            number += 1
            metrics = await manager.get_metrics()
            print("\n".join(await render_full(manager, metrics, number)))
            if rounds == 0 or number < rounds:
                await _pause(interval)
        return 0
    finally:
        disconnect = getattr(manager, "disconnect", None)
        if disconnect is not None:
            await disconnect()


def main(argv: Optional[List[str]] = None, manager: Any = None) -> int:
    """Run the monitor; return the exit status. ``manager`` is a RedisManager
    (or a stand-in with ``connect``, ``get_metrics`` and ``disconnect``); None
    builds one from Settings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=MODES, default="snapshot")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args(argv)
    if args.interval < 0 or (args.iterations is not None and args.iterations < 0):
        parser.error("--interval and --iterations must not be negative")
    target = manager if manager is not None else RedisManager.from_settings()
    try:
        return asyncio.run(_run(args.mode, args.interval, args.iterations, target))
    except KeyboardInterrupt:
        print("Monitoring stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
