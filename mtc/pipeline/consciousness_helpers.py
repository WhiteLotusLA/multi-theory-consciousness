"""Small reads the turn makes from the thought stream and the workspace.

- ``get_recent_ctm_thoughts`` returns the prompt's thought items through the
  continuous thought machine's surfacing read
  (``mtc.consciousness.continuous.surfacing``): only thoughts the novelty gate
  marked expressible, never a dream thought (excluded by type), each text
  once, newest first.
- ``get_thought_gists`` returns the prompt's gist items: gists whose
  retention is at least one half (``GIST_MIN_RETENTION``), most important
  first, with ``source_count`` as the gist stores it.
- ``get_last_competition_intensity`` reads the workspace bottleneck's last
  competition: one minus winners over candidates, or None before any
  competition (no intensity is assumed).
- ``ConsciousnessLevelCache`` throttles an expensive level source (for
  example the latest assessment's overall score) to one call per
  ``min_interval_s`` seconds on the injected monotonic clock. A source that
  returns None (insufficient data) gives None, never a default level; a
  source that raises is counted (``status()['failures']``, the first and
  every hundredth logged at WARNING) and the previous value is kept, unless
  ``DEBUG`` is true, which raises instead.

A missing store gives an empty list; a store failure propagates so the caller
counts it. Thought text and gist summaries are private runtime data and are
never logged. Config keys: ``DEBUG`` (the level cache, through its
``settings`` or ``get_settings()`` at call time); the caller passes the
limits (``PROMPT_MAX_THOUGHTS`` and ``PROMPT_MAX_GISTS``).
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.consciousness.continuous.store import DEFAULT_EXCLUDED_PREFIXES
from mtc.consciousness.continuous.surfacing import (
    get_gists,
    get_surfaced_thoughts,
    gist_items,
    thought_items,
)
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

GIST_MIN_RETENTION = 0.5
DEFAULT_LEVEL_INTERVAL_S = 300.0


async def get_recent_ctm_thoughts(
    store: Any,
    limit: int = 5,
    exclude_type_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
) -> List[Dict[str, Any]]:
    """The expressible thoughts the prompt may show, newest first."""
    if store is None or limit <= 0:
        return []
    surfaced = await get_surfaced_thoughts(store, limit, exclude_type_prefixes)
    return thought_items(surfaced)


async def get_thought_gists(gist_store: Any, limit: int = 5) -> List[Dict[str, Any]]:
    """The gists the prompt may show (retention at least one half)."""
    if gist_store is None or limit <= 0:
        return []
    gists = await get_gists(gist_store, limit, min_retention=GIST_MIN_RETENTION)
    return gist_items(gists)


def get_last_competition_intensity(workspace: Any) -> Optional[float]:
    """One minus winners over candidates of the last competition, or None."""
    bottleneck = getattr(workspace, "bottleneck", None)
    history = getattr(bottleneck, "competition_history", None)
    if not history:
        return None
    last = history[-1]
    candidates = int(getattr(last, "candidates_count", 0) or 0)
    if candidates <= 0:
        return None
    winners = int(getattr(last, "winners_count", 0) or 0)
    return 1.0 - (winners / candidates)


class ConsciousnessLevelCache:
    """A throttled, honest reading of a consciousness level."""

    def __init__(
        self,
        source: Callable[[], Any],
        min_interval_s: float = DEFAULT_LEVEL_INTERVAL_S,
        clock: Optional[Callable[[], float]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if source is None:
            raise ValueError("ConsciousnessLevelCache needs a level source")
        self._source = source
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._clock = clock or time.monotonic
        self._settings = settings
        self._value: Optional[float] = None
        self._read_at: Optional[float] = None
        self._calls = 0
        self._failures = 0

    def _debug(self) -> bool:
        active = self._settings if self._settings is not None else get_settings()
        return bool(getattr(active, "DEBUG", False))

    async def level(self) -> Optional[float]:
        """The cached level, refreshed when the interval has passed."""
        now = float(self._clock())
        if self._read_at is not None and now - self._read_at < self.min_interval_s:
            return self._value
        self._read_at = now
        self._calls += 1
        try:
            value = self._source()
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:  # noqa: BLE001 - counted, previous value kept
            self._failures += 1
            if self._failures == 1 or self._failures % 100 == 0:
                logger.warning(
                    "consciousness level source failed (%d so far): %s",
                    self._failures,
                    type(exc).__name__,
                )
            if self._debug():
                raise
            return self._value
        self._value = None if value is None else float(value)
        return self._value

    def status(self) -> Dict[str, Any]:
        return {
            "value": self._value,
            "calls": self._calls,
            "failures": self._failures,
            "min_interval_s": self.min_interval_s,
        }


__all__ = (
    "GIST_MIN_RETENTION",
    "ConsciousnessLevelCache",
    "get_last_competition_intensity",
    "get_recent_ctm_thoughts",
    "get_thought_gists",
)
