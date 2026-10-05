"""
Identity Statement Loader
=========================

Reads the identity statement a prompt assembler places at the top of a system
prompt. The statement is operator-authored text; this module ships no identity
text of its own. The file is read at call time from ``IDENTITY_STATEMENT_PATH``
(empty derives ``<AGENT_SPACE_DIR>/identity_statement.md``) unless a path is
passed. A neutral template lives in ``config/identity_statement.example.md``;
its ``{system_name}`` and ``{creator_name}`` placeholders are rendered by the
caller, and the loader returns the raw text.

A missing file yields an empty string and one INFO log. A file longer than
``max_chars`` yields an empty string and a warning, so an oversized or wrong
file never lands in a prompt. The statement may name people at runtime, so the
file belongs in the gitignored agent space.
"""

import logging
from pathlib import Path
from typing import Optional

from mtc.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 20000


def load_identity_statement(
    path: Optional[Path] = None, max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Return the stripped identity statement, or "" when none is usable."""
    if path is None:
        path = Path(get_settings().IDENTITY_STATEMENT_PATH)
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            text = handle.read(max_chars + 1)
    except FileNotFoundError:
        logger.info("no identity statement at %s", source)
        return ""
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("identity statement unreadable: %s", type(exc).__name__)
        return ""
    if len(text) > max_chars:
        logger.warning(
            "identity statement at %s exceeds %d characters; ignored",
            source,
            max_chars,
        )
        return ""
    return text.strip()
