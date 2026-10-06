"""
Environment File Helpers
========================

Reads and edits a project's ``.env`` file, which holds secrets (the API key,
speaker tokens, store passwords and an Anthropic key when that provider is
chosen) as well as the identity and store settings.

- ``read_env`` parses the file with python-dotenv's reader, without
  interpolation; a missing file is an empty mapping.
- ``render_value`` writes a value so Compose and python-dotenv read it
  identically: bare when it uses only ``[A-Za-z0-9_.:/,@+=-]``, double-quoted
  otherwise (for example when it holds spaces), and refused with
  ``UnsafeEnvValue`` when it holds a dollar sign, a double quote, a
  backslash, a backtick, a newline or another control character.
- ``upsert_env`` replaces or appends assignments while keeping every unrelated
  line and comment in place. Keys named in ``keep_existing`` keep a non-empty
  existing value. The file is replaced atomically through a temporary file in
  the same directory and always ends with mode 0600. Values are never logged;
  only key counts are.
- ``env_mode_ok`` reports whether the file exists and grants nothing to group
  or others.

No config key is read. The only file written is the ``.env`` path given.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Collection, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

ENV_FILE_MODE = 0o600

_BARE = re.compile(r"[A-Za-z0-9_.:/,@+=-]*")
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_FORBIDDEN = {
    "$": "a dollar sign",
    '"': "a double quote",
    "\\": "a backslash",
    "`": "a backtick",
    "\n": "a newline",
    "\r": "a carriage return",
}


class UnsafeEnvValue(ValueError):
    """A value that Compose and python-dotenv would not read identically."""


def read_env(path: Path) -> Dict[str, str]:
    """Parse ``path`` with python-dotenv (no interpolation); ``{}`` if missing."""
    target = Path(path)
    if not target.is_file():
        return {}
    from dotenv import dotenv_values

    values = dotenv_values(target, interpolate=False, encoding="utf-8")
    return {key: ("" if value is None else value) for key, value in values.items()}


def render_value(value: str) -> str:
    """Render one value for a ``.env`` line (see the module docstring)."""
    text = str(value)
    for character, description in _FORBIDDEN.items():
        if character in text:
            raise UnsafeEnvValue(f"value holds {description}, which .env cannot carry")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise UnsafeEnvValue("value holds a control character")
    if _BARE.fullmatch(text):
        return text
    return f'"{text}"'


def _opens_multiline(raw_value: str) -> bool:
    value = raw_value.strip()
    if not value or value[0] not in "'\"":
        return False
    quote = value[0]
    return value.count(quote) == 1


def _header_lines(header: str) -> List[str]:
    lines = []
    for line in header.splitlines():
        text = line.rstrip()
        lines.append(text if text.startswith("#") else f"# {text}".rstrip())
    return lines


def upsert_env(
    path: Path,
    updates: Mapping[str, str],
    *,
    keep_existing: Collection[str] = (),
    header: Optional[str] = None,
) -> List[str]:
    """Write ``updates`` into the ``.env`` file at ``path``; return keys written.

    An existing assignment is replaced where it stands (a later duplicate of
    the same key is dropped, since python-dotenv would let it win); a new key
    is appended at the end, under ``header`` when one is given. A key in
    ``keep_existing`` whose current value is non-empty is left alone and is
    not in the returned list. Every value is rendered before anything is
    written, so an unsafe value leaves the file untouched; a failed write
    leaves the old file in place.
    """
    target = Path(path)
    rendered: Dict[str, str] = {}
    for key, value in updates.items():
        if not _KEY.fullmatch(str(key)):
            raise ValueError(f"{key!r} is not a valid environment variable name")
        try:
            rendered[key] = render_value(value)
        except UnsafeEnvValue as exc:
            raise UnsafeEnvValue(f"{key}: {exc}") from None
    existing = read_env(target)
    keep = {
        key for key in keep_existing if key in rendered and existing.get(key, "") != ""
    }
    lines = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    output: List[str] = []
    placed = set()
    written: List[str] = []
    for line in lines:
        match = _ASSIGNMENT.match(line)
        if not match or match.group(1) not in rendered:
            output.append(line)
            continue
        key = match.group(1)
        if key in keep:
            output.append(line)
            continue
        if _opens_multiline(match.group(2)):
            raise ValueError(
                f"{target.name}: the value of {key} spans several lines; edit it "
                "by hand before running this again"
            )
        if key in placed:
            continue
        output.append(f"{key}={rendered[key]}")
        placed.add(key)
        written.append(key)
    appended = [key for key in rendered if key not in placed and key not in keep]
    if appended:
        if output and output[-1].strip():
            output.append("")
        if header:
            output.extend(_header_lines(header))
        for key in appended:
            output.append(f"{key}={rendered[key]}")
            written.append(key)
    _atomic_write(target, "\n".join(output) + "\n")
    logger.debug("env file updated: %d key(s) written", len(written))
    return written


def _atomic_write(target: Path, text: str) -> None:
    directory = target.parent
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), ENV_FILE_MODE)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    os.chmod(target, ENV_FILE_MODE)


def env_mode_ok(path: Path) -> bool:
    """True when ``path`` exists and grants no permission to group or others."""
    try:
        mode = Path(path).stat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_IMODE(mode) & 0o077 == 0
