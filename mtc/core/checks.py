"""
Check Results
=============

The one status vocabulary ``mtc doctor``, the database bootstrap and the
verifier share, with the shared exit-code rule and plain renderings.

Statuses (``CheckStatus``):

- ``ok``: the check ran and the property holds.
- ``warning``: the check ran and found something advisory to fix.
- ``failed``: the check ran and the property does not hold.
- ``unreachable``: the service refused the connection or timed out.
- ``skipped``: the check was not run on purpose (for example a disabled store).
- ``insufficient_data``: the check could not be performed (a refused
  privileged command, a missing optional extra) and says why.

A check that did not run is never ``ok``.

Exit codes (``exit_code_for``): any required ``failed`` gives 1; otherwise any
required ``unreachable`` gives 3; otherwise 0. Advisory checks never change
the code, except under ``strict``, which treats every check as required and
counts ``warning`` and ``insufficient_data`` as failures. ``skipped`` never
fails.

``status_for_error`` maps a store driver's exception to ``failed`` (the server
answered: a refused authentication, a missing permission or database) or
``unreachable`` (refused, reset or timed out) by class name, without importing
any driver; ``describe_error`` renders one line with any given secret masked.

No config key is read and nothing is persisted. Details are written by the
checks themselves and never carry a secret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNREACHABLE = 3


class CheckStatus(str, Enum):
    """The shared status vocabulary."""

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"
    UNREACHABLE = "unreachable"
    SKIPPED = "skipped"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class CheckResult:
    """One check: its name, status, a plain detail and whether it is required."""

    name: str
    status: CheckStatus
    detail: str = ""
    required: bool = True
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = CheckStatus(self.status)


def summarize(results: Iterable[CheckResult]) -> Dict[str, int]:
    """Count every status (zero included) plus ``total``."""
    counts = {status.value: 0 for status in CheckStatus}
    total = 0
    for result in results:
        counts[CheckStatus(result.status).value] += 1
        total += 1
    counts["total"] = total
    return counts


def exit_code_for(results: Iterable[CheckResult], *, strict: bool = False) -> int:
    """Return 1, 3 or 0 by the shared rule (see the module docstring)."""
    failed = False
    unreachable = False
    for result in results:
        status = CheckStatus(result.status)
        required = result.required or strict
        if not required:
            continue
        if status is CheckStatus.FAILED:
            failed = True
        elif status is CheckStatus.UNREACHABLE:
            unreachable = True
        elif strict and status in (
            CheckStatus.WARNING,
            CheckStatus.INSUFFICIENT_DATA,
        ):
            failed = True
    if failed:
        return EXIT_FAILED
    if unreachable:
        return EXIT_UNREACHABLE
    return EXIT_OK


def _ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def render_text(results: Sequence[CheckResult], *, title: str) -> str:
    """ASCII text: the title, one line per check, then the summary line."""
    lines = [_ascii(title)]
    width = max((len(r.status.value) for r in results), default=2)
    for result in results:
        tag = f"[{result.status.value}]".ljust(width + 2)
        advisory = "" if result.required else " (advisory)"
        detail = f": {result.detail}" if result.detail else ""
        line = f"  {tag} {result.name}{advisory}{detail}"
        lines.append(_ascii(" ".join(line.splitlines())))
    counts = summarize(results)
    parts = [f"{status.value}={counts[status.value]}" for status in CheckStatus]
    lines.append(_ascii(f"summary: {' '.join(parts)} total={counts['total']}"))
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def to_json(results: Sequence[CheckResult], *, strict: bool = False) -> Dict[str, Any]:
    """A JSON-ready mapping: every check, the summary and the exit code."""
    return {
        "checks": [
            {
                "name": result.name,
                "status": result.status.value,
                "detail": result.detail,
                "required": bool(result.required),
                "data": _jsonable(result.data),
            }
            for result in results
        ],
        "summary": summarize(results),
        "exit_code": exit_code_for(results, strict=strict),
    }


def _error_chain(exc: BaseException) -> List[BaseException]:
    chain: List[BaseException] = []
    current: Any = exc
    while isinstance(current, BaseException) and current not in chain:
        chain.append(current)
        if len(chain) >= 8:
            break
        current = current.__cause__ or current.__context__
    return chain


def _class_names(exc: BaseException) -> List[str]:
    return [cls.__name__ for cls in type(exc).__mro__]


def status_for_error(exc: BaseException) -> CheckStatus:
    """Map a store driver's exception to ``unreachable`` or ``failed``.

    The exception and its causes are matched by class name, so no driver is
    imported: a refused authentication, a permission error or a missing
    database is ``failed`` (the server answered); a refused connection, a
    reset or a timeout is ``unreachable``; anything else is ``failed``.
    """
    chain = _error_chain(exc)
    for item in chain:
        for name in _class_names(item):
            if any(marker in name for marker in _ANSWERED_MARKERS):
                return CheckStatus.FAILED
    for item in chain:
        if isinstance(item, (OSError, TimeoutError)):
            return CheckStatus.UNREACHABLE
        names = _class_names(item)
        if any(name in _UNREACHABLE_NAMES for name in names):
            return CheckStatus.UNREACHABLE
    return CheckStatus.FAILED


def describe_error(exc: BaseException, *, secrets: Iterable[str] = ()) -> str:
    """``Type: message`` on one line, any secret value masked, at most 300 chars."""
    text = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "***")
    text = " ".join(text.split())
    return text if len(text) <= 300 else text[:297] + "..."


# Class-name fragments of driver errors raised after the server answered.
_ANSWERED_MARKERS = (
    "Auth",
    "Password",
    "Permission",
    "Privilege",
    "NoPerm",
    "InvalidCatalogName",
    "Unauthorized",
    "Forbidden",
)
# Driver error class names that mean the server was not reached in time.
_UNREACHABLE_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectionFailure",
        "AutoReconnect",
        "ServerSelectionTimeoutError",
        "NetworkTimeout",
        "ServiceUnavailable",
        "SessionExpired",
        "ConnectionAcquisitionTimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
        "ResponseHandlingException",
        "CannotConnectNowError",
        "ConnectionDoesNotExistError",
        "TimeoutError",
    }
)


def results_from_json(payload: Mapping[str, Any]) -> List[CheckResult]:
    """Rebuild the results ``to_json`` rendered."""
    return [
        CheckResult(
            name=str(item["name"]),
            status=CheckStatus(item["status"]),
            detail=str(item.get("detail", "")),
            required=bool(item.get("required", True)),
            data=dict(item.get("data") or {}),
        )
        for item in payload.get("checks", [])
    ]
