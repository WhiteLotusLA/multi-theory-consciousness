"""
Settings Export for Shell Scripts
=================================

``python3 -m mtc.core.env_export [--secrets] FIELD...`` prints one
``export MTC_FIELD=value`` line per requested Settings field, quoted with
``shlex.quote``, so the operations scripts read every default and every
derived store name from the one Settings model instead of restating them
(``eval "$(python3 -m mtc.core.env_export POSTGRES_DB REDIS_PORT)"``).

Values are the loaded Settings values: the process environment, then
``.env`` in the working directory, then the defaults, with the derived names
(``POSTGRES_DB``, ``POSTGRES_USER``, ``MONGODB_DB``, ``QDRANT_COLLECTION``,
``REDIS_KEY_PREFIX``, ``PROJECT_ROOT`` and the other derived paths) filled
in. Path values are printed as configured; a relative one is relative to
``MTC_PROJECT_ROOT``. Computed fields (``POSTGRES_URL``, ``LLM_URL`` and the
others) may be named. Booleans print ``true`` or ``false``; mappings and lists
print as JSON.

A secret field (a text field whose name holds ``PASSWORD``, ``API_KEY``,
``SECRET`` or ``TOKEN``, and the store URLs that embed a password) is refused
unless ``--secrets`` is given; a script asks for one only to hand it to a tool
through that tool's environment variable. Exit codes: 0 printed, 2 an unknown
field, a refused secret or settings that do not load (named by field, never by
value).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Any, List, Optional, Sequence

from mtc.core.config import Settings, get_settings
from mtc.core.env_example import SECRET_MARKERS

ENV_PREFIX = "MTC_"
# Computed fields that embed a store password.
SECRET_COMPUTED_FIELDS = ("POSTGRES_URL", "MONGODB_URL", "REDIS_URL")


class UnknownField(KeyError):
    """A name that is neither a Settings field nor a computed field."""


class SecretRefused(PermissionError):
    """A secret field requested without ``secrets=True``."""


def is_secret_field(name: str) -> bool:
    """True for a text field named like a secret or a password-bearing URL."""
    if name in SECRET_COMPUTED_FIELDS:
        return True
    info = Settings.model_fields.get(name)
    if info is None:
        return False
    return info.annotation is str and any(marker in name for marker in SECRET_MARKERS)


def _known(name: str) -> bool:
    return name in Settings.model_fields or name in Settings.model_computed_fields


def _render(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def export_lines(
    fields: Sequence[str],
    *,
    settings: Optional[Settings] = None,
    secrets: bool = False,
) -> List[str]:
    """Return ``export MTC_<FIELD>=<quoted value>`` lines for ``fields``.

    Raises UnknownField for a name Settings does not define and SecretRefused
    for a secret field unless ``secrets`` is true; nothing is returned in
    either case.
    """
    for name in fields:
        if not _known(name):
            raise UnknownField(name)
        if is_secret_field(name) and not secrets:
            raise SecretRefused(name)
    active = settings if settings is not None else get_settings()
    return [
        f"export {ENV_PREFIX}{name}={shlex.quote(_render(getattr(active, name)))}"
        for name in fields
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m mtc.core.env_export",
        description="Print shell export lines for Settings fields.",
    )
    parser.add_argument(
        "--secrets",
        action="store_true",
        help="allow secret fields (passwords, keys, tokens)",
    )
    parser.add_argument("fields", nargs="+", metavar="FIELD")
    args = parser.parse_args(argv)
    try:
        for name in args.fields:
            if not _known(name):
                raise UnknownField(name)
            if is_secret_field(name) and not args.secrets:
                raise SecretRefused(name)
    except UnknownField as exc:
        print(f"env_export: unknown Settings field {exc.args[0]}", file=sys.stderr)
        return 2
    except SecretRefused as exc:
        print(
            f"env_export: {exc.args[0]} is a secret; pass --secrets to print it",
            file=sys.stderr,
        )
        return 2
    try:
        settings = Settings()
    except Exception as exc:
        names = _invalid_fields(exc)
        detail = f" (fields: {', '.join(names)})" if names else ""
        print(f"env_export: settings could not be loaded{detail}", file=sys.stderr)
        return 2
    lines = export_lines(args.fields, settings=settings, secrets=args.secrets)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _invalid_fields(exc: Exception) -> List[str]:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return []
    names = []
    try:
        for error in errors():
            location = error.get("loc") or ()
            if location:
                names.append(str(location[0]))
    except Exception:
        return []
    return sorted(set(names))


if __name__ == "__main__":
    raise SystemExit(main())
