"""
Path Resolution
===============

One place that turns the path-like Settings keys into absolute paths.

- An absolute path stays as given; ``~`` expands to the home directory; a
  relative path joins ``PROJECT_ROOT``.
- Each helper reads the Settings object it is given, or the process-wide
  ``get_settings()`` instance. That instance is built once per process: a
  changed environment is seen only after ``reset_settings()``. The helpers
  themselves cache nothing.
- Resolution creates no directory. A writer calls ``ensure_dir`` at its first
  write.

Config keys read: ``PROJECT_ROOT``, ``LOG_DIRECTORY``, ``DATA_DIRECTORY``,
``RESULTS_DIR``, ``AGENT_SPACE_DIR``, ``STATE_DIR`` and ``SHADOW_LOG_DIR``.
The agent-space and state directories hold private runtime state.
"""

import os
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


def display_path(path: Union[str, Path], settings: Optional[Settings] = None) -> str:
    """Return a form of ``path`` fit for a status report; never an absolute path.

    A path inside ``PROJECT_ROOT`` (a relative one is taken as inside it) is
    shown relative to the root with forward slashes; any other path is shown
    by its file name only, so a home directory or another absolute location
    never reaches a status payload. The file itself is not touched.
    """
    root = Path(os.path.normpath(os.path.abspath(project_root(settings))))
    full = Path(
        os.path.normpath(os.path.abspath(resolve_path(path, settings=settings)))
    )
    try:
        return full.relative_to(root).as_posix()
    except ValueError:
        return full.name


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) when missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
