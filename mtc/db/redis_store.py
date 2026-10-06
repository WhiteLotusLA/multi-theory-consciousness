"""
Redis Check
===========

Redis has no schema to create, so it gets checks only.

- ``probe`` and the required ``redis:ping`` check: ``PING`` within the timeout.
- ``redis:maxmemory_policy`` (advisory): ``CONFIG GET maxmemory-policy``;
  ``volatile-lru`` and ``allkeys-lru`` are ``ok``, any other policy is a
  ``warning`` recommending ``volatile-lru`` (some key families carry no TTL,
  so only keys with a TTL should be evicted first). A server or an access
  level that refuses ``CONFIG GET`` gives ``insufficient_data`` with the
  reason. The policy is only read, never changed from here.
- ``redis:key_families`` (advisory): the keys under ``REDIS_KEY_PREFIX`` per
  registered family (``KEY_FAMILIES``), counted with ``SCAN`` up to a cap.

The client comes from ``mtc.memory.clients.create_redis_client``. Config keys
read at call time: ``REDIS_HOST``, ``REDIS_PORT``, ``REDIS_DB``,
``REDIS_PASSWORD``, ``REDIS_KEY_PREFIX`` and the client keys. Working memory
holds recent conversation messages and emotional state; no value is read
here, and nothing is deleted or flushed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from mtc.core.checks import CheckResult, CheckStatus, describe_error, status_for_error
from mtc.core.config import Settings, get_settings
from mtc.core.schema_registry import load_artifact
from mtc.memory.clients import create_redis_client

STORE = "redis"
EXTRA = "redis"
CHECK_NAME = "store:redis"
KEY_FAMILIES_PATH = "mtc.memory.redis_manager.KEY_FAMILIES"
RECOMMENDED_POLICY = "volatile-lru"
ACCEPTED_POLICIES = ("volatile-lru", "allkeys-lru")
SCAN_CAP = 10000


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def _client(active: Settings, timeout_s: float) -> Any:
    return create_redis_client(
        active, socket_timeout=timeout_s, socket_connect_timeout=timeout_s
    )


def _error(exc: BaseException, active: Settings, name: str = CHECK_NAME):
    if isinstance(exc, ImportError):
        return CheckResult(
            name, CheckStatus.FAILED, f"{exc}", data={"missing_extra": EXTRA}
        )
    status = status_for_error(exc)
    detail = describe_error(exc, secrets=[active.REDIS_PASSWORD])
    where = f"{active.REDIS_HOST}:{active.REDIS_PORT}"
    if status is CheckStatus.UNREACHABLE:
        hint = f"Redis at {where} did not answer"
    elif "auth" in detail.lower() or "noauth" in detail.lower():
        hint = "authentication was refused; check MTC_REDIS_PASSWORD in .env"
    else:
        hint = f"Redis at {where} refused the operation"
    return CheckResult(name, status, f"{hint} ({detail})")


async def _close(client: Any) -> None:
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if callable(closer):
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=5.0)
        except Exception:  # closing is best effort after the result is known
            pass


async def probe(
    settings: Optional[Settings] = None, *, timeout_s: float = 2.0
) -> CheckResult:
    """Quick reachability check: ``PING``."""
    active = _settings(settings)
    client = None
    try:
        client = _client(active, timeout_s)
        await asyncio.wait_for(client.ping(), timeout=timeout_s + 1.0)
    except Exception as exc:
        return _error(exc, active)
    finally:
        if client is not None:
            await _close(client)
    return CheckResult(
        CHECK_NAME,
        CheckStatus.OK,
        f"answered at {active.REDIS_HOST}:{active.REDIS_PORT} db {active.REDIS_DB}",
    )


def family_prefix(pattern: str) -> str:
    """The literal start of a key family pattern (before its first slot)."""
    return pattern.split("{", 1)[0]


async def _policy(client: Any) -> CheckResult:
    name = "redis:maxmemory_policy"
    try:
        reply = await client.config_get("maxmemory-policy")
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.INSUFFICIENT_DATA,
            f"CONFIG GET was refused, so the eviction policy is unknown "
            f"({describe_error(exc)}); {RECOMMENDED_POLICY} is recommended",
            required=False,
        )
    policy = None
    if isinstance(reply, dict):
        policy = reply.get("maxmemory-policy")
    if not policy:
        return CheckResult(
            name,
            CheckStatus.INSUFFICIENT_DATA,
            "CONFIG GET returned no maxmemory-policy",
            required=False,
        )
    policy = str(policy)
    if policy in ACCEPTED_POLICIES:
        return CheckResult(
            name,
            CheckStatus.OK,
            f"maxmemory-policy {policy}",
            required=False,
            data={"policy": policy},
        )
    return CheckResult(
        name,
        CheckStatus.WARNING,
        f"maxmemory-policy {policy}; {RECOMMENDED_POLICY} is recommended because "
        "some key families carry no TTL and should not be evicted first",
        required=False,
        data={"policy": policy, "recommended": RECOMMENDED_POLICY},
    )


async def _families(client: Any, active: Settings) -> CheckResult:
    name = "redis:key_families"
    counts: Dict[str, int] = {}
    capped: List[str] = []
    try:
        for family in load_artifact(KEY_FAMILIES_PATH):
            literal = family_prefix(family.pattern)
            match = f"{active.REDIS_KEY_PREFIX}{literal}*"
            count = 0
            async for _ in client.scan_iter(match=match, count=500):
                count += 1
                if count >= SCAN_CAP:
                    capped.append(family.pattern)
                    break
            counts[family.pattern] = count
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.INSUFFICIENT_DATA,
            f"the key families could not be counted ({describe_error(exc)})",
            required=False,
        )
    total = sum(counts.values())
    note = (
        f" (counting stopped at {SCAN_CAP} for {', '.join(capped)})" if capped else ""
    )
    return CheckResult(
        name,
        CheckStatus.OK,
        f"{total} keys under {active.REDIS_KEY_PREFIX!r} across "
        f"{len(counts)} families{note}",
        required=False,
        data={"counts": counts, "capped": capped},
    )


async def check(
    client: Any = None,
    settings: Optional[Settings] = None,
    *,
    timeout_s: float = 5.0,
) -> List[CheckResult]:
    """The ping (required), the eviction policy and the key families."""
    active = _settings(settings)
    owned = client is None
    try:
        if owned:
            client = _client(active, timeout_s)
        try:
            await asyncio.wait_for(client.ping(), timeout=timeout_s + 1.0)
        except Exception as exc:
            return [_error(exc, active, "redis:ping")]
        results = [
            CheckResult(
                "redis:ping",
                CheckStatus.OK,
                f"answered at {active.REDIS_HOST}:{active.REDIS_PORT}",
            )
        ]
        results.append(await _policy(client))
        results.append(await _families(client, active))
        return results
    except Exception as exc:
        return [_error(exc, active, "redis:ping")]
    finally:
        if owned and client is not None:
            await _close(client)
