"""
Path Resolution
===============

One place that turns the path-like Settings keys into absolute paths.

- An absolute path stays as given; ``~`` expands to the home directory; a
  relative path joins ``PROJECT_ROOT``.
- Settings are read at call time, so a changed environment takes effect on the
  next call. Nothing is cached.
- Resolution creates no directory. A writer calls ``ensure_dir`` at its first
  write.

Config keys read: ``PROJECT_ROOT``, ``LOG_DIRECTORY``, ``DATA_DIRECTORY``,
``RESULTS_DIR``, ``AGENT_SPACE_DIR``, ``STATE_DIR`` and ``SHADOW_LOG_DIR``.
The agent-space and state directories hold private runtime state.
"""

from pathlib import Path
from typing import Optional, Union

from mtc.core.config import Settings, get_settings


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def resolve_path(
    value: Union[str, Path], *, settings: Optional[Settings] = None
) -> Path:
    """Return an absolute path for a setting value."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(_settings(settings).PROJECT_ROOT).expanduser() / path


def project_root(settings: Optional[Settings] = None) -> Path:
    return Path(_settings(settings).PROJECT_ROOT).expanduser()


def log_dir(settings: Optional[Settings] = None) -> Path:
    return resolve_path(_settings(settings).LOG_DIRECTORY, settings=settings)


def data_dir(settings: Optional[Settings] = None) -> Path:
    return resolve_path(_settings(settings).DATA_DIRECTORY, settings=settings)


def results_dir(settings: Optional[Settings] = None) -> Path:
    return resolve_path(_settings(settings).RESULTS_DIR, settings=settings)


def agent_space_dir(settings: Optional[Settings] = None) -> Path:
    return resolve_path(_settings(settings).AGENT_SPACE_DIR, settings=settings)


def state_dir(settings: Optional[Settings] = None) -> Path:
    return resolve_path(_settings(settings).STATE_DIR, settings=settings)


def shadow_log_dir(settings: Optional[Settings] = None) -> Path:
    return resolve_path(_settings(settings).SHADOW_LOG_DIR, settings=settings)


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) when missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
