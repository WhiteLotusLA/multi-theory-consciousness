"""
Missing extra
=============

The one plain line a command prints when the install lacks the ``fep`` extra.

The consciousness workspace builds the active inference module
unconditionally and that module raises ``ImportError`` without pymdp, so
every command that builds a workspace (``mtc assess``, ``mtc serve``, the
demos and the examples) cannot start on an install without the extra.
``explain_missing_extra`` turns that error into one line for standard error,
so the command can print it, leave out the traceback and exit with the usage
code (2): a missing extra is a configuration error.

``explain_missing_extra(exc)`` recognises the message constant of the active
inference module (``PYMDP_MISSING_MESSAGE``) and a failed import of pymdp
itself; any other error returns None and stays the caller's to raise.
``workspace_extra_line()`` asks the same question before anything starts: it
returns the line when the active inference module cannot be built here. A
caller that starts a library which logs a failed startup and returns (the
API server) needs it, because the error never reaches the caller.
"""

from __future__ import annotations

import importlib
from typing import Optional

PREFIX = "MISSING EXTRA:"
EXTRA = "fep"
MODULE = "pymdp"
PACKAGE = "multi-theory-consciousness"
ACTIVE_INFERENCE_MODULE = "mtc.consciousness.active_inference"
MESSAGE_CONSTANT = "PYMDP_MISSING_MESSAGE"

LINE = (
    f"{PREFIX} the {EXTRA} extra is not installed. The workspace builds active "
    f"inference ({MODULE}) and cannot start without it. "
    f"Install it with: pip install '{PACKAGE}[{EXTRA}]'"
)


def _module_name(exc: ImportError) -> str:
    return str(getattr(exc, "name", None) or "")


def _active_inference_message() -> Optional[str]:
    """The active inference module's own message, or None when it cannot be
    read here (the module is imported only when a caller asks)."""
    try:
        module = importlib.import_module(ACTIVE_INFERENCE_MODULE)
    except Exception:  # noqa: BLE001 - a description must not fail
        return None
    value = getattr(module, MESSAGE_CONSTANT, None)
    return value if isinstance(value, str) and value else None


def explain_missing_extra(exc: BaseException) -> Optional[str]:
    """The one line for a missing extra, or None for any other error.

    Args:
        exc: The exception a command caught.

    Returns:
        A single line starting ``MISSING EXTRA:`` that names the ``fep``
        extra and the install command, when ``exc`` is the active inference
        module's missing-pymdp error or a failed import of pymdp; None for
        anything else, including an exception that is not an ``ImportError``.
    """
    if not isinstance(exc, ImportError):
        return None
    name = _module_name(exc)
    if name == MODULE or name.startswith(f"{MODULE}."):
        return LINE
    if name:
        # An error that names another module is not the active inference
        # module's message, and that module is heavy to import.
        return None
    message = _active_inference_message()
    if message is not None and message in str(exc):
        return LINE
    return None


def workspace_extra_line() -> Optional[str]:
    """The missing-extra line when the workspace cannot be built here.

    Returns:
        The same line ``explain_missing_extra`` gives when the active
        inference module loads without pymdp's agent (so constructing it
        would raise), None when it can be built or when the module cannot be
        inspected at all.
    """
    try:
        module = importlib.import_module(ACTIVE_INFERENCE_MODULE)
    except ImportError as exc:
        return explain_missing_extra(exc)
    except Exception:  # noqa: BLE001 - an unreadable module is not a verdict
        return None
    if (
        getattr(module, "Agent", None) is None
        or getattr(module, "pymdp_utils", None) is None
    ):
        return LINE
    return None
