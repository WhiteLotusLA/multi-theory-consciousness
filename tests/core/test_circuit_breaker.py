# tests/core/test_circuit_breaker.py
"""Tests for the generic CircuitBreaker."""

import asyncio
import pytest
from unittest.mock import AsyncMock


class TestCircuitBreakerStates:
    """Tests for state transitions."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test")
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_stays_closed_on_success(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test")
        op = AsyncMock(return_value="ok")
        fb = AsyncMock(return_value="fallback")

        result = await cb.call(op, fb)
        assert result == "ok"
        assert cb.state == "closed"
        fb.assert_not_called()

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=3)
        op = AsyncMock(side_effect=Exception("db down"))
        fb = AsyncMock(return_value="fallback")

        # 3 failures -> opens
        for _ in range(3):
            await cb.call(op, fb)
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_returns_fallback_when_open(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=1)
        op = AsyncMock(side_effect=Exception("db down"))
        fb = AsyncMock(return_value="fallback")

        # Open the circuit
        await cb.call(op, fb)
        assert cb.state == "open"

        # Next call should use fallback without calling op
        op.reset_mock()
        result = await cb.call(op, fb)
        assert result == "fallback"
        op.assert_not_called()

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)
        op = AsyncMock(side_effect=Exception("db down"))
        fb = AsyncMock(return_value="fallback")

        await cb.call(op, fb)
        assert cb.state == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should probe (half_open)
        op_good = AsyncMock(return_value="recovered")
        result = await cb.call(op_good, fb)
        assert result == "recovered"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)
        op = AsyncMock(side_effect=Exception("db down"))
        fb = AsyncMock(return_value="fallback")

        await cb.call(op, fb)
        assert cb.state == "open"

        await asyncio.sleep(0.15)

        # Probe fails
        result = await cb.call(op, fb)
        assert result == "fallback"
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_resets_failure_count_on_success(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=3)
        op_fail = AsyncMock(side_effect=Exception("err"))
        op_ok = AsyncMock(return_value="ok")
        fb = AsyncMock(return_value="fallback")

        # 2 failures (not enough to open)
        await cb.call(op_fail, fb)
        await cb.call(op_fail, fb)
        assert cb.state == "closed"

        # 1 success resets counter
        await cb.call(op_ok, fb)
        assert cb.state == "closed"
        assert cb.stats["failure_count"] == 0


class TestCircuitBreakerStats:
    """Tests for stats tracking."""

    @pytest.mark.asyncio
    async def test_tracks_total_failures(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=2)
        op = AsyncMock(side_effect=Exception("err"))
        fb = AsyncMock(return_value="x")

        await cb.call(op, fb)
        await cb.call(op, fb)  # opens
        await cb.call(op, fb)  # fallback (no op call)

        assert cb.stats["total_failures"] >= 2
        assert cb.stats["total_fallbacks"] >= 1

    @pytest.mark.asyncio
    async def test_tracks_recovery(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        op_fail = AsyncMock(side_effect=Exception("err"))
        op_ok = AsyncMock(return_value="ok")
        fb = AsyncMock(return_value="x")

        await cb.call(op_fail, fb)  # open
        await asyncio.sleep(0.1)
        await cb.call(op_ok, fb)  # recover

        assert cb.stats["recovery_count"] == 1

    @pytest.mark.asyncio
    async def test_can_start_open(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", start_open=True, recovery_timeout=0.05)
        assert cb.state == "open"

        op = AsyncMock(return_value="ok")
        fb = AsyncMock(return_value="fallback")

        # Immediately returns fallback
        result = await cb.call(op, fb)
        assert result == "fallback"


class TestCircuitBreakerConcurrency:
    """Tests for asyncio.Lock probe protection."""

    @pytest.mark.asyncio
    async def test_only_one_probe_in_half_open(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        op_fail = AsyncMock(side_effect=Exception("err"))
        fb = AsyncMock(return_value="fallback")

        await cb.call(op_fail, fb)  # open
        await asyncio.sleep(0.1)  # eligible for half_open

        # Slow probe operation
        async def slow_probe(*args, **kwargs):
            await asyncio.sleep(0.2)
            return "probed"

        call_count = 0

        async def counting_fallback(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "fallback"

        # Launch 5 concurrent calls - only 1 should probe
        results = await asyncio.gather(
            cb.call(slow_probe, counting_fallback),
            cb.call(slow_probe, counting_fallback),
            cb.call(slow_probe, counting_fallback),
            cb.call(slow_probe, counting_fallback),
            cb.call(slow_probe, counting_fallback),
        )

        # One should get "probed", rest should get "fallback"
        assert results.count("probed") == 1
        assert results.count("fallback") == 4


class TestCircuitBreakerNotification:
    """Tests for state change notifications."""

    @pytest.mark.asyncio
    async def test_calls_on_state_change(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        changes = []

        def on_change(name, old_state, new_state, stats):
            changes.append((name, old_state, new_state))

        cb = CircuitBreaker(name="test", failure_threshold=1, on_state_change=on_change)
        op = AsyncMock(side_effect=Exception("err"))
        fb = AsyncMock(return_value="x")

        await cb.call(op, fb)

        assert len(changes) == 1
        assert changes[0] == ("test", "closed", "open")


class TestCircuitBreakerRecoveryListeners:
    """Listeners run once on the transition back to closed."""

    @staticmethod
    async def _open_then_recover(cb, op_ok):
        op_fail = AsyncMock(side_effect=Exception("err"))
        fb = AsyncMock(return_value="x")
        await cb.call(op_fail, fb)  # opens at threshold 1
        assert cb.state == "open"
        await asyncio.sleep(0.06)
        await cb.call(op_ok, fb)  # the probe succeeds

    @pytest.mark.asyncio
    async def test_recovery_listeners_fire_once_on_recovery(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        fired = []
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0.05,
            recovery_listeners=[lambda: fired.append("ctor")],
        )
        cb.add_recovery_listener(lambda: fired.append("added"))
        await self._open_then_recover(cb, AsyncMock(return_value="ok"))
        assert cb.state == "closed"
        assert fired == ["ctor", "added"]
        # more successful calls while closed do not fire them again
        await cb.call(AsyncMock(return_value="ok"), AsyncMock())
        assert fired == ["ctor", "added"]
        # a second outage and recovery fires them once more
        await self._open_then_recover(cb, AsyncMock(return_value="ok"))
        assert fired == ["ctor", "added", "ctor", "added"]

    @pytest.mark.asyncio
    async def test_async_listener_is_scheduled_and_awaited(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        events = []

        async def listener():
            await asyncio.sleep(0)
            events.append("done")

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        cb.add_recovery_listener(listener)
        await self._open_then_recover(cb, AsyncMock(return_value="ok"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert events == ["done"]

    @pytest.mark.asyncio
    async def test_listener_exception_logged_not_raised(self, caplog):
        import logging

        from mtc.core.circuit_breaker import CircuitBreaker

        async def bad_async():
            raise RuntimeError("async listener broke")

        def bad_sync():
            raise RuntimeError("sync listener broke")

        survivors = []
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.05)
        cb.add_recovery_listener(bad_sync)
        cb.add_recovery_listener(bad_async)
        cb.add_recovery_listener(lambda: survivors.append("ran"))
        with caplog.at_level(logging.ERROR, logger="mtc.core.circuit_breaker"):
            await self._open_then_recover(cb, AsyncMock(return_value="ok"))
            for _ in range(5):
                await asyncio.sleep(0)
        assert cb.state == "closed"
        assert survivors == ["ran"]
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "sync listener broke" in text
        assert "async listener broke" in text

    @pytest.mark.asyncio
    async def test_state_change_callback_failure_logged_at_error(self, caplog):
        import logging

        from mtc.core.circuit_breaker import CircuitBreaker

        def broken(name, old, new, stats):
            raise RuntimeError("notify broke")

        cb = CircuitBreaker(name="test", failure_threshold=1, on_state_change=broken)
        with caplog.at_level(logging.ERROR, logger="mtc.core.circuit_breaker"):
            await cb.call(AsyncMock(side_effect=Exception("err")), AsyncMock())
        assert cb.state == "open"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("notify broke" in r.getMessage() for r in errors)

    @pytest.mark.asyncio
    async def test_no_listener_fires_on_open(self):
        from mtc.core.circuit_breaker import CircuitBreaker

        fired = []
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_listeners=[lambda: fired.append("x")],
        )
        await cb.call(AsyncMock(side_effect=Exception("err")), AsyncMock())
        assert cb.state == "open"
        assert fired == []
        # a failed probe reopens without firing either
        cb.recovery_timeout = 0.0
        await cb.call(AsyncMock(side_effect=Exception("still down")), AsyncMock())
        assert cb.state == "open"
        assert fired == []
