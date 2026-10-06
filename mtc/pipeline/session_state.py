"""Per-speaker conversation sessions held in process.

A ``Session`` is one speaker's live conversation: its conversation id, the
history the brain sees (a list of entry dicts with ``role``, ``content``,
``speaker`` and ``timestamp``), the turn counter, start and last-turn times,
the running summary, the verbatim quotes kept when old turns were summarized
and the speaker label the session's stored rows name the speaker by (the
bracketed claim for a name no token backs; None for a session opened without
one). ``SessionManager`` keys sessions by the string
``SpeakerTrust.session_key(identity)`` returns, never by a bare name, so a
name presented without its token can never read or extend an authenticated
speaker's session: the trust source gives the two different keys.

The manager trims a history in place to ``max_history`` entries (newest kept)
and prunes sessions idle for more than ``idle_prune_hours``, reading time
from the injected clock (default ``datetime.now(timezone.utc)``). Times are
aware UTC. Session keys, histories, summaries and quotes hold speaker names
and conversation text, private runtime data: nothing here logs them, and
``status()`` reports counts only. Nothing is persisted; the conversation
repository is the durable record and ``rehydrate_recent_history`` refills a
new session from it. No config keys are read.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


@dataclass
class Session:
    """One speaker's live conversation (private content)."""

    key: str
    conversation_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    turn_number: int = 0
    started_at: Optional[datetime] = None
    last_turn_at: Optional[datetime] = None
    summary: Optional[str] = None
    pinned_quotes: List[str] = field(default_factory=list)
    speaker_label: Optional[str] = None


class SessionManager:
    """Sessions keyed by the trust source's session key."""

    def __init__(
        self,
        max_history: int = 100,
        idle_prune_hours: float = 48.0,
        clock: Optional[Clock] = None,
    ) -> None:
        if int(max_history) < 1:
            raise ValueError("max_history must be at least 1")
        if float(idle_prune_hours) <= 0:
            raise ValueError("idle_prune_hours must be positive")
        self.max_history = int(max_history)
        self.idle_prune_hours = float(idle_prune_hours)
        self._clock: Clock = clock or _utc_now
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._pruned = 0

    def _now(self) -> datetime:
        return _aware(self._clock())

    @staticmethod
    def _check_key(key: str) -> str:
        if not isinstance(key, str) or not key:
            raise TypeError(
                "a session key is the string SpeakerTrust.session_key returns"
            )
        return key

    def get(self, key: str) -> Optional[Session]:
        """Return the live session for ``key``, or None."""
        with self._lock:
            return self._sessions.get(self._check_key(key))

    def start(
        self,
        key: str,
        conversation_id: str,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        speaker_label: Optional[str] = None,
    ) -> Session:
        """Open (or replace) the session for ``key`` with an optional history
        (for example one rehydrated from the repository), trimmed to
        ``max_history``, and the label its stored rows name the speaker by
        (a non-empty string, or None)."""
        if speaker_label is not None and (
            not isinstance(speaker_label, str) or not speaker_label.strip()
        ):
            raise TypeError("a speaker label is a non-empty string or None")
        now = self._now()
        session = Session(
            key=self._check_key(key),
            conversation_id=str(conversation_id),
            history=[dict(entry) for entry in history or ()],
            started_at=now,
            last_turn_at=None,
            speaker_label=speaker_label,
        )
        with self._lock:
            self._sessions[key] = session
            self._trim(session)
        return session

    def advance(self, key: str) -> int:
        """Count one more turn, stamp ``last_turn_at`` and return the new
        turn number. An unknown key raises KeyError."""
        with self._lock:
            session = self._sessions.get(self._check_key(key))
            if session is None:
                raise KeyError("no session for this key")
            session.turn_number += 1
            session.last_turn_at = self._now()
            return session.turn_number

    def _trim(self, session: Session) -> int:
        excess = len(session.history) - self.max_history
        if excess > 0:
            del session.history[:excess]
            return excess
        return 0

    def trim(self, key: str) -> int:
        """Trim the session's history in place to the newest ``max_history``
        entries and return how many were removed (0 for an unknown key)."""
        with self._lock:
            session = self._sessions.get(self._check_key(key))
            return 0 if session is None else self._trim(session)

    def prune(self, now: Optional[datetime] = None) -> List[Session]:
        """Remove and return every session whose last turn (or start, before
        any turn) is more than ``idle_prune_hours`` before ``now``."""
        moment = _aware(now) if now is not None else self._now()
        cutoff = moment - timedelta(hours=self.idle_prune_hours)
        removed: List[Session] = []
        with self._lock:
            for key, session in list(self._sessions.items()):
                last = session.last_turn_at or session.started_at
                if last is not None and last < cutoff:
                    removed.append(self._sessions.pop(key))
            self._pruned += len(removed)
        return removed

    def end(self, key: str) -> Optional[Session]:
        """Remove and return the session for ``key`` (None when absent)."""
        with self._lock:
            return self._sessions.pop(self._check_key(key), None)

    def sessions(self) -> List[Session]:
        """Every live session, oldest start first."""
        with self._lock:
            return sorted(
                self._sessions.values(),
                key=lambda s: s.started_at or datetime.min.replace(tzinfo=timezone.utc),
            )

    def status(self) -> Dict[str, Any]:
        """Counts only; no key, name or text."""
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "pruned": self._pruned,
                "max_history": self.max_history,
                "idle_prune_hours": self.idle_prune_hours,
            }


__all__ = ("Session", "SessionManager")
