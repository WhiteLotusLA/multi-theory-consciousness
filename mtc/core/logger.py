"""
Logging
=======

Explicit logger setup for the framework. Nothing here configures logging on
import; an entry point (a server, a CLI, a script) calls ``setup_logging()``
once.

- The ``mtc`` logger is configured, never the root logger. Every module logs
  through ``logging.getLogger(__name__)``, so its records propagate to ``mtc``
  and stop there. Reconfiguration of the root logger by another library (PyPhi
  does this when its configuration changes) therefore cannot take the
  framework's handlers away. ``neutralize_pyphi_logging`` additionally turns
  PyPhi's reconfiguration into a no-op once its configuration module is loaded.
- Config keys read: ``LOG_LEVEL``, ``LOG_FORMAT`` (``json`` or ``plain`` for the
  main file), ``LOG_DIRECTORY``, ``LOG_ROTATION_SIZE_MB`` and
  ``LOG_ROTATION_BACKUPS``. The console always uses the plain format.
- Files under the log directory: ``mtc.log`` plus three JSON Lines specialty
  files, ``safety_audit.jsonl``, ``consciousness_metrics.jsonl`` and
  ``emotional_history.jsonl``. The specialty files hold private emotional and
  safety records and belong in a gitignored or operator-controlled directory.
- Under pytest with no directory passed, the main file is ``mtc-test.log`` and
  no specialty file is attached, so a test run never writes the production
  log or its audit trails.

The schema of the specialty records is the keys the helpers write: a
``timestamp`` (timezone-aware UTC, ISO 8601) plus the caller's fields.
"""

import json
import logging
import math
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from mtc.core.config import Settings, get_settings
from mtc.core.paths import ensure_dir, log_dir

ROOT_LOGGER_NAME = "mtc"
SAFETY_LOGGER_NAME = "mtc.specialty.safety"
METRICS_LOGGER_NAME = "mtc.specialty.metrics"
EMOTION_LOGGER_NAME = "mtc.specialty.emotional"

MAIN_LOG_NAME = "mtc.log"
TEST_LOG_NAME = "mtc-test.log"
SPECIALTY_FILES = {
    SAFETY_LOGGER_NAME: "safety_audit.jsonl",
    METRICS_LOGGER_NAME: "consciousness_metrics.jsonl",
    EMOTION_LOGGER_NAME: "emotional_history.jsonl",
}

_OWNED = "_mtc_owned"
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _utc_iso(created: float) -> str:
    return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()


class JSONFormatter(logging.Formatter):
    """One JSON object per record, including fields passed as ``extra``."""

    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "timestamp": _utc_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in data:
                data[key] = value
        return json.dumps(data, default=str)


class PlainFormatter(logging.Formatter):
    """ASCII-only single-line text: timestamp, level, logger and message."""

    def format(self, record: logging.LogRecord) -> str:
        line = (
            f"{_utc_iso(record.created)} {record.levelname} "
            f"[{record.name}] {record.getMessage()}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line.encode("ascii", errors="backslashreplace").decode("ascii")


class _JSONLinesFormatter(logging.Formatter):
    """The message is already one JSON object; write it as is."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class _ConsoleHandler(logging.StreamHandler):
    """Stream handler that looks up ``sys.stderr`` at emit time."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:  # pragma: no cover - never assigned
        pass


def _owned(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _OWNED, False))


def _mark(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _OWNED, True)
    return handler


def _drop_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if _owned(handler):
            logger.removeHandler(handler)
            handler.close()


def _file_handler(
    path: Path, max_bytes: int, backups: int, level: int, formatter: logging.Formatter
) -> logging.Handler:
    handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return _mark(handler)


def reset_logging() -> None:
    """Remove every handler ``setup_logging`` attached and reset the loggers."""
    for name in (ROOT_LOGGER_NAME, *SPECIALTY_FILES):
        logger = logging.getLogger(name)
        _drop_owned_handlers(logger)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def setup_logging(
    settings: Optional[Settings] = None,
    *,
    log_directory: Optional[Path] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``mtc`` logger and return it. The root logger is untouched.

    A second call returns the configured logger unchanged unless ``force`` is
    true, which replaces the handlers (use it after a settings change).
    """
    active = settings if settings is not None else get_settings()
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if any(_owned(h) for h in logger.handlers) and not force:
        return logger

    level_name = str(active.LOG_LEVEL).upper()
    if level_name not in _LEVELS:
        raise ValueError(f"LOG_LEVEL {active.LOG_LEVEL!r} is not a logging level")
    log_format = str(active.LOG_FORMAT).lower()
    if log_format not in ("json", "plain"):
        raise ValueError(f"LOG_FORMAT {active.LOG_FORMAT!r} must be json or plain")
    level = _LEVELS[level_name]

    under_pytest = "pytest" in sys.modules
    test_mode = under_pytest and log_directory is None
    directory = ensure_dir(
        Path(log_directory) if log_directory is not None else log_dir(active)
    )
    max_bytes = int(active.LOG_ROTATION_SIZE_MB) * 1024 * 1024
    backups = int(active.LOG_ROTATION_BACKUPS)

    _drop_owned_handlers(logger)
    logger.setLevel(level)
    logger.propagate = False

    console = _mark(_ConsoleHandler())
    console.setLevel(level)
    console.setFormatter(PlainFormatter())
    logger.addHandler(console)

    main_name = TEST_LOG_NAME if test_mode else MAIN_LOG_NAME
    file_formatter: logging.Formatter = (
        JSONFormatter() if log_format == "json" else PlainFormatter()
    )
    logger.addHandler(
        _file_handler(directory / main_name, max_bytes, backups, level, file_formatter)
    )

    for name, filename in SPECIALTY_FILES.items():
        specialty = logging.getLogger(name)
        _drop_owned_handlers(specialty)
        specialty.setLevel(logging.INFO)
        specialty.propagate = False
        if not test_mode:
            specialty.addHandler(
                _file_handler(
                    directory / filename,
                    max_bytes,
                    backups,
                    logging.INFO,
                    _JSONLinesFormatter(),
                )
            )

    neutralize_pyphi_logging()
    logger.info("logging configured (level %s, format %s)", level_name, log_format)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``mtc`` namespace for a module name."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats (invalid JSON) by None, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_specialty(logger_name: str, payload: Dict[str, Any], level: int) -> None:
    logging.getLogger(logger_name).log(
        level, json.dumps(payload, default=str, allow_nan=False)
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_safety_event(event: Mapping[str, Any]) -> None:
    """Append a safety event to ``safety_audit.jsonl``.

    The record is the event's fields plus a ``timestamp`` when absent; an
    optional ``level`` key (``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``)
    sets the logging level, default INFO. The file holds private safety
    records.
    """
    payload: Dict[str, Any] = {"timestamp": _now(), **dict(event)}
    level = _LEVELS.get(str(payload.get("level", "INFO")).upper(), logging.INFO)
    _write_specialty(SAFETY_LOGGER_NAME, _clean(payload), level)


def log_consciousness_metric(name: str, value: Optional[float], **context: Any) -> None:
    """Append one metric to ``consciousness_metrics.jsonl``.

    ``value`` is None when the metric could not be measured (the assessment
    has 30 positive and 3 negative indicators, and an unmeasured one is not a
    zero); a non-finite value is written as None.
    """
    payload = {"timestamp": _now(), "metric": name, "value": value, **context}
    _write_specialty(METRICS_LOGGER_NAME, _clean(payload), logging.INFO)


def log_emotional_state(state: Mapping[str, Any]) -> None:
    """Append an emotional state to ``emotional_history.jsonl`` (private)."""
    payload: Dict[str, Any] = {"timestamp": _now(), **dict(state)}
    _write_specialty(EMOTION_LOGGER_NAME, _clean(payload), logging.INFO)


def _clean(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _json_safe(value) for key, value in payload.items()}


_PYPHI_LOGGING_OPTIONS = ("LOG_FILE", "LOG_FILE_LEVEL", "LOG_STDOUT_LEVEL")


def _pyphi_logging_hook_neutral(obj: Any, opt: Any) -> None:
    """Replacement ``on_change`` hook: keep PyPhi's config-wide callback only."""
    callback = getattr(obj, "_callback", None)
    if callback is not None:
        callback(obj)


def neutralize_pyphi_logging() -> bool:
    """Stop PyPhi from replacing the root logging handlers when it is loaded.

    PyPhi binds its module-level ``configure_logging`` into the ``on_change``
    hooks of the ``LOG_FILE``, ``LOG_FILE_LEVEL`` and ``LOG_STDOUT_LEVEL``
    options; setting any of them calls ``logging.config.dictConfig``, which
    swaps every root handler. This replaces those three hooks with one that
    runs only the config-wide callback (PyPhi's own validation and cache
    write), so a later change to those options leaves the root handlers alone.

    Returns True when ``pyphi.conf`` was already imported and its logging hooks
    are now neutral (calling it again is harmless). PyPhi is never imported
    here; the caller that imports it (the PyPhi bridge) calls this right after
    its own import.
    """
    module = sys.modules.get("pyphi.conf")
    if module is None:
        return False
    config_class = getattr(module, "PyphiConfig", None)
    options = getattr(config_class, "options", None)
    if config_class is None or not callable(options):
        return False
    try:
        declared = options()
    except Exception as exc:  # an unexpected PyPhi layout is reported, not fatal
        logging.getLogger(__name__).debug("PyPhi options unreadable: %s", exc)
        return False
    neutral = False
    for name in _PYPHI_LOGGING_OPTIONS:
        option = declared.get(name)
        if option is None or not hasattr(option, "on_change"):
            continue
        option.on_change = _pyphi_logging_hook_neutral
        neutral = True
    return neutral
