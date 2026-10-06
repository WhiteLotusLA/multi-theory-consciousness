"""
Request Diagnostics
===================

Observe-only evidence for duplicate replies. ``RequestObserver`` counts, per
client-supplied ``request_id`` and inside a time window, how many times a
request was received and how many responses were sent for it:

- the same ``request_id`` received more than once within the window means
  the client sent it again (a retry or a replayed frame after a reconnect);
- more than one response for one ``request_id`` means the server processed
  one request twice.

A reply copied to several open connections is a third cause; the caller
records the open-connection count next to these counters for that one.

The observer never blocks, suppresses or changes a request: it only counts,
and ``duplicates()`` reports the totals. A missing ``request_id`` cannot be
correlated and is not tracked. Entries expire ``ttl_s`` seconds after their
first sighting, measured on an injectable monotonic clock.

No config keys are read and nothing is persisted. Request ids are opaque
client values; they are kept in memory for the window only and never logged.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

DEFAULT_TTL_S = 300.0


@dataclass
class _Entry:
    first_seen: float
    receipts: int = 0
    responses: int = 0


class RequestObserver:
    """Counts receipts and responses per ``request_id`` inside a TTL window.

    Args:
        ttl_s: Seconds an id is remembered after its first sighting; positive.
        clock: Monotonic clock in seconds; None uses ``time.monotonic``.
    """

    def __init__(
        self, ttl_s: float = DEFAULT_TTL_S, clock: Optional[Callable[[], float]] = None
    ) -> None:
        ttl = float(ttl_s)
        if not ttl > 0.0:
            raise ValueError(f"ttl_s must be positive, got {ttl_s!r}")
        self.ttl_s = ttl
        self._clock = clock or time.monotonic
        self._entries: Dict[str, _Entry] = {}
        self._duplicate_receipts = 0
        self._duplicate_responses = 0
        self._untracked = 0

    def _evict(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.first_seen > self.ttl_s
        ]
        for key in expired:
            del self._entries[key]

    def _entry(self, request_id: str) -> _Entry:
        now = self._clock()
        self._evict(now)
        entry = self._entries.get(request_id)
        if entry is None:
            entry = _Entry(first_seen=now)
            self._entries[request_id] = entry
        return entry

    def observe_receipt(self, request_id: Optional[str]) -> int:
        """Record a received request; return how often this id arrived within
        the window (above 1 is a duplicate receipt). None returns 0."""
        if not isinstance(request_id, str) or not request_id:
            self._untracked += 1
            return 0
        entry = self._entry(request_id)
        entry.receipts += 1
        if entry.receipts > 1:
            self._duplicate_receipts += 1
        return entry.receipts

    def observe_response(self, request_id: Optional[str]) -> int:
        """Record a sent response; return how many responses this id has had
        within the window (above 1 is a duplicate response). None returns 0."""
        if not isinstance(request_id, str) or not request_id:
            return 0
        entry = self._entry(request_id)
        entry.responses += 1
        if entry.responses > 1:
            self._duplicate_responses += 1
        return entry.responses

    def duplicates(self) -> Dict[str, int]:
        """Totals since construction plus the ids currently tracked."""
        self._evict(self._clock())
        return {
            "duplicate_receipts": self._duplicate_receipts,
            "duplicate_responses": self._duplicate_responses,
            "tracked": len(self._entries),
            "untracked": self._untracked,
        }
