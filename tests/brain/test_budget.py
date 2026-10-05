"""Token budget: a conservative estimate and one contiguous newest-suffix pass."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtc.brain import budget as budget_module
from mtc.brain.budget import (
    BudgetResult,
    estimate_tokens,
    fit_history,
    history_budget,
)

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


def _settings(**overrides):
    values = {
        "HISTORY_TOKEN_BUDGET": 6000,
        "CONTEXT_SIZE": 131072,
        "CHARS_PER_TOKEN": 3.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def budget_settings(monkeypatch):
    holder = _settings()
    monkeypatch.setattr(budget_module, "get_settings", lambda: holder)
    return holder


def _entry(role: str, content: str, **extra):
    entry = {"role": role, "content": content, "timestamp": T0}
    entry.update(extra)
    return entry


def test_estimate_tokens_conservative(budget_settings):
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    # int(len / chars_per_token) + 1 never rounds a non-empty text down to zero.
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcdefg", 3.5) == 3
    assert estimate_tokens("x" * 100) == int(100 / 3.5) + 1
    assert estimate_tokens("x" * 100, 2.0) == 51


def test_fit_keeps_newest_contiguous_suffix(budget_settings):
    # Each entry costs int(35 / 3.5) + 1 = 11 tokens.
    history = [
        _entry("user" if i % 2 == 0 else "assistant", str(i) * 35) for i in range(6)
    ]
    result = fit_history(history, 35)
    assert isinstance(result, BudgetResult)
    assert result.kept == history[3:]
    assert result.dropped == 3
    assert result.estimated_tokens == 33
    assert result.budget_tokens == 35
    # The kept entries are the originals, in order, with no interior gap.
    assert [e["content"] for e in result.kept] == ["333" + "3" * 32, "4" * 35, "5" * 35]


def test_min_recent_kept_even_over_budget(budget_settings):
    history = [_entry("user", "short"), _entry("assistant", "y" * 4000)]
    result = fit_history(history, 10)
    assert result.kept == [history[-1]]
    assert result.dropped == 1
    assert result.estimated_tokens > result.budget_tokens
    two = fit_history(history, 10, min_recent=2)
    assert two.kept == history
    assert two.dropped == 0


def test_leading_clock_note_dropped(budget_settings):
    history = [
        _entry("user", "a" * 35),
        _entry("user", "[clock note] a night passed", is_clock_note=True),
        _entry("user", "b" * 35),
        _entry("assistant", "c" * 35),
    ]
    # Budget for exactly the last three entries, so the kept window would
    # start with the clock note.
    budget = sum(estimate_tokens(e["content"]) for e in history[1:])
    result = fit_history(history, budget)
    assert result.kept == history[2:]
    assert result.dropped == 2
    assert all(not e.get("is_clock_note") for e in result.kept)
    # A clock note inside the window stays where it is.
    whole = fit_history(history, 10_000)
    assert whole.kept == history
    assert whole.dropped == 0


def test_budget_respects_context_size(budget_settings):
    budget_settings.CONTEXT_SIZE = 4096
    assert history_budget(1000, 512) == 4096 - 512 - 1000
    budget_settings.CONTEXT_SIZE = 131072
    assert history_budget(1000, 512) == 6000
    # Explicit settings win over the module's reader.
    small = _settings(CONTEXT_SIZE=2000, HISTORY_TOKEN_BUDGET=6000)
    assert history_budget(1500, 400, settings=small) == 100
    # A system prompt that already fills the window leaves no room, never a
    # negative budget.
    assert history_budget(5000, 500, settings=small) == 0


def test_single_pass_reports_dropped_count(budget_settings):
    history = [_entry("user", "m" * 70) for _ in range(10)]
    result = fit_history(history, 21 * 4)
    assert len(result.kept) + result.dropped == len(history)
    assert result.dropped == 6
    assert result.estimated_tokens == sum(
        estimate_tokens(e["content"]) for e in result.kept
    )
    empty = fit_history([], 100)
    assert empty.kept == [] and empty.dropped == 0 and empty.estimated_tokens == 0


def test_chars_per_token_from_settings_at_call_time(budget_settings):
    assert estimate_tokens("abcd") == 2
    budget_settings.CHARS_PER_TOKEN = 1.0
    assert estimate_tokens("abcd") == 5
    # fit_history reads the same value through estimate_tokens.
    history = [_entry("user", "abcd"), _entry("assistant", "efgh")]
    assert fit_history(history, 5).kept == history[1:]
    budget_settings.CHARS_PER_TOKEN = 4.0
    assert fit_history(history, 5).kept == history
