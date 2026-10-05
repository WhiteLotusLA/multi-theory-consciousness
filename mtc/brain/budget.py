"""
Token Budget
============

The one place the conversation history is trimmed before a model call. The
brain knows the final system prompt's size, so the pass runs there and nowhere
else: ``history_budget`` bounds the history by
``min(HISTORY_TOKEN_BUDGET, CONTEXT_SIZE - max_tokens - system prompt tokens)``
and ``fit_history`` keeps the largest contiguous newest suffix that fits.
Contiguity is a hard requirement: an interior gap reads as a jump cut.

History entries are plain dicts with ``role`` (``user`` or ``assistant``),
``content``, an optional ``speaker``, an aware UTC ``timestamp`` and an
optional ``is_clock_note`` flag marking a seam note the pipeline inserted
between turns. A clock note that would lead the kept window describes a gap
before messages that are no longer there, so it is dropped with them.

``estimate_tokens`` is a cheap, conservative estimate (``int(len /
CHARS_PER_TOKEN) + 1``). It only needs to never wildly underestimate: the
budget carries the safety margin.

Config keys read at call time: ``CHARS_PER_TOKEN``, ``HISTORY_TOKEN_BUDGET``
and ``CONTEXT_SIZE``. No state is kept. The history holds conversation text,
which is private at runtime; nothing here logs it.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mtc.core.config import get_settings

HistoryEntry = Mapping[str, Any]


@dataclass(frozen=True)
class BudgetResult:
    """The outcome of one budget pass over a history."""

    kept: List[Dict[str, Any]]
    dropped: int
    estimated_tokens: int
    budget_tokens: int


def _chars_per_token(chars_per_token: Optional[float]) -> float:
    value = (
        float(chars_per_token)
        if chars_per_token is not None
        else float(get_settings().CHARS_PER_TOKEN)
    )
    if value <= 0:
        raise ValueError("CHARS_PER_TOKEN must be positive")
    return value


def estimate_tokens(
    text: Optional[str], chars_per_token: Optional[float] = None
) -> int:
    """Return a conservative token estimate for ``text`` (0 for empty)."""
    if not text:
        return 0
    return int(len(text) / _chars_per_token(chars_per_token)) + 1


def fit_history(
    history: Sequence[HistoryEntry],
    budget_tokens: int,
    *,
    min_recent: int = 1,
    chars_per_token: Optional[float] = None,
) -> BudgetResult:
    """Keep the largest contiguous newest suffix of ``history`` within budget.

    The newest ``min_recent`` entries are always kept, even over budget. A
    clock note left at the head of the kept window is dropped. One pass, from
    the newest entry backwards; the result reports how many entries were
    dropped and the estimated size of what was kept.
    """
    entries = list(history)
    budget = max(0, int(budget_tokens))
    ratio = _chars_per_token(chars_per_token)
    total = len(entries)
    costs = [estimate_tokens(entry.get("content") or "", ratio) for entry in entries]

    start = total
    used = 0
    for index in range(total - 1, -1, -1):
        forced = (total - index) <= max(0, int(min_recent))
        if not forced and used + costs[index] > budget:
            break
        used += costs[index]
        start = index

    while start < total - 1 and entries[start].get("is_clock_note"):
        used -= costs[start]
        start += 1

    kept = [dict(entry) for entry in entries[start:]]
    return BudgetResult(
        kept=kept,
        dropped=start,
        estimated_tokens=used,
        budget_tokens=budget,
    )


def history_budget(
    system_prompt_tokens: int, max_tokens: int, *, settings: Any = None
) -> int:
    """Return the token budget left for the history, never negative."""
    cfg = settings if settings is not None else get_settings()
    room = int(cfg.CONTEXT_SIZE) - int(max_tokens) - int(system_prompt_tokens)
    return max(0, min(int(cfg.HISTORY_TOKEN_BUDGET), room))
