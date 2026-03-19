# mtc/core/circuit_breaker.py
"""
Generic Circuit Breaker for database resilience.

Three states: CLOSED (normal), OPEN (fallback), HALF_OPEN (probing).
Async-native with asyncio.Lock for concurrent probe protection.

No database-specific code — this is a general-purpose pattern.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Async circuit breaker with three states.

    CLOSED: operations pass through. Failures increment counter.
    OPEN: operations use fallback immediately. After recovery_timeout,
          transition to HALF_OPEN.
    HALF_OPEN: one probe attempt (protected by lock). Success -> CLOSED.
               Failure -> OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        start_open: bool = False,
        on_state_change: Optional[Callable] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.on_state_change = on_state_change

        # State
        self._state = "open" if start_open else "closed"
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = time.time() if start_open else 0.0
        self._probe_lock = asyncio.Lock()

        # Stats
        self._total_failures = 0
        self._total_fallbacks = 0
        self._lifetime_fallbacks = 0  # Never resets (cumulative)
        self._recovery_count = 0
        self._last_recovery_time: float = 0.0

    async def call(
        self,
        operation: Callable,
        fallback: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Try operation. Use fallback if circuit is open or operation fails.
        """
        if self._state == "open":
            if self._should_attempt_recovery():
                return await self._probe(operation, fallback, *args, **kwargs)
            self._total_fallbacks += 1
            return await self._call_async(fallback, *args, **kwargs)

        if self._state == "half_open":
            # Another concurrent call while probe is running — use fallback
            self._total_fallbacks += 1
            return await self._call_async(fallback, *args, **kwargs)

        # CLOSED — try the operation
        try:
            result = await self._call_async(operation, *args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            self._total_fallbacks += 1
            return await self._call_async(fallback, *args, **kwargs)

    async def _probe(
        self,
        operation: Callable,
        fallback: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Attempt a probe in HALF_OPEN state, protected by lock."""
        # Try to acquire the lock without blocking. If another coroutine
        # is already probing, we skip straight to fallback.
        if self._probe_lock.locked():
            # Another probe is running — use fallback
            self._total_fallbacks += 1
            return await self._call_async(fallback, *args, **kwargs)

        async with self._probe_lock:
            # Re-check state inside the lock (another coroutine may have
            # already recovered or re-opened while we waited)
            if self._state == "closed":
                # Another probe succeeded while we waited — normal path
                try:
                    result = await self._call_async(operation, *args, **kwargs)
                    self._on_success()
                    return result
                except Exception as e:
                    self._on_failure(e)
                    self._total_fallbacks += 1
                    return await self._call_async(fallback, *args, **kwargs)

            old_state = self._state
            self._state = "half_open"
            if old_state != "half_open":
                logger.info(f"🔌 {self.name} circuit breaker probing...")

            try:
                result = await self._call_async(operation, *args, **kwargs)
                self._on_recovery()
                return result
            except Exception as e:
                self._on_probe_failure(e)
                self._total_fallbacks += 1
                return await self._call_async(fallback, *args, **kwargs)

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (time.time() - self._opened_at) >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful operation in CLOSED state."""
        self._failure_count = 0

    def _on_failure(self, error: Exception) -> None:
        """Handle failed operation in CLOSED state."""
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.time()
        logger.debug(
            f"🔌 {self.name} failure {self._failure_count}/{self.failure_threshold}: {error}"
        )

        if self._failure_count >= self.failure_threshold:
            self._transition_to("open")

    def _on_recovery(self) -> None:
        """Handle successful probe in HALF_OPEN state."""
        self._recovery_count += 1
        self._last_recovery_time = time.time()
        duration = time.time() - self._opened_at
        logger.info(
            f"🔌 {self.name} recovered after {duration:.0f}s. "
            f"{self._total_fallbacks} ops used fallback during outage (not synced)."
        )
        self._transition_to("closed")
        self._failure_count = 0
        self._lifetime_fallbacks += self._total_fallbacks
        self._total_fallbacks = 0  # Reset per-outage counter

    def _on_probe_failure(self, error: Exception) -> None:
        """Handle failed probe in HALF_OPEN state."""
        self._total_failures += 1
        logger.warning(f"🔌 {self.name} probe failed: {error}")
        self._transition_to("open")

    def _transition_to(self, new_state: str) -> None:
        """Change state and fire notification."""
        old_state = self._state
        self._state = new_state
        if new_state == "open":
            self._opened_at = time.time()
            logger.warning(
                f"🔌 {self.name} circuit breaker OPEN — fallback active"
            )

        if self.on_state_change and old_state != new_state:
            try:
                self.on_state_change(self.name, old_state, new_state, self.stats)
            except Exception:
                pass  # notification failure must not break the breaker

    async def _call_async(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call a function, handling both sync and async callables."""
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @property
    def state(self) -> str:
        return self._state

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "total_failures": self._total_failures,
            "total_fallbacks": self._total_fallbacks,
            "lifetime_fallbacks": self._lifetime_fallbacks,
            "recovery_count": self._recovery_count,
            "last_failure_time": self._last_failure_time,
            "last_recovery_time": self._last_recovery_time,
        }
