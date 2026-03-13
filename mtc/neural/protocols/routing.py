"""
Neural Router -- Message Routing Between Substrates
=====================================================

Routes neural messages based on configurable rules. Supports:
- DIRECT: point-to-point (SNN -> CTM)
- BROADCAST: one-to-all (GWT ignition broadcast)
- HIERARCHICAL: layered propagation (SNN -> LSM -> HTM)

Rules are matched by source/target glob patterns and priority
thresholds. The router is designed for sub-millisecond decisions
(1000 routes in <100ms).

Created: March 7, 2026
Authors: Multi-Theory Consciousness Contributors
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
from fnmatch import fnmatch
import logging

from mtc.neural.protocols.message_format import (
    NeuralMessage,
    MessagePriority,
)

logger = logging.getLogger(__name__)


class RouteType(Enum):
    """How a message is delivered."""

    DIRECT = "direct"  # One source -> one target
    BROADCAST = "broadcast"  # One source -> all registered targets
    HIERARCHICAL = "hierarchical"  # Layered propagation through a chain


@dataclass
class RoutingRule:
    """
    A rule that maps source/target patterns to a route type.

    source_pattern and target_pattern use glob matching (e.g., "SNN*"
    matches "SNN_primary", "SNN_secondary").
    """

    source_pattern: str
    target_pattern: str
    route_type: RouteType = RouteType.DIRECT
    priority_threshold: MessagePriority = MessagePriority.NORMAL
    transform_func: Optional[Callable[[NeuralMessage], NeuralMessage]] = None
    enabled: bool = True

    def matches(self, message: NeuralMessage) -> bool:
        """Check if this rule matches a given message."""
        if not self.enabled:
            return False
        source_match = fnmatch(message.source, self.source_pattern)
        target_match = fnmatch(message.target, self.target_pattern)
        priority_ok = message.priority.value <= self.priority_threshold.value
        return source_match and target_match and priority_ok


class NeuralRouter:
    """
    Routes neural messages between substrates based on rules.

    Rules are evaluated in insertion order; first match wins for
    DIRECT routes. BROADCAST rules accumulate all matching targets.
    """

    def __init__(self):
        self._rules: List[RoutingRule] = []
        self._route_cache: Dict[str, RoutingRule] = {}
        self._stats = {"routed": 0, "dropped": 0, "broadcast": 0}

    def add_rule(self, rule: RoutingRule) -> None:
        """Register a new routing rule."""
        self._rules.append(rule)
        self._route_cache.clear()  # invalidate cache on rule change

    def remove_rule(self, source_pattern: str, target_pattern: str) -> bool:
        """Remove a rule by its source/target patterns."""
        before = len(self._rules)
        self._rules = [
            r
            for r in self._rules
            if not (
                r.source_pattern == source_pattern
                and r.target_pattern == target_pattern
            )
        ]
        self._route_cache.clear()
        return len(self._rules) < before

    def get_route(self, message: NeuralMessage) -> Optional[RoutingRule]:
        """
        Find the first matching routing rule for a message.

        Uses a simple cache keyed on (source, target, priority) to
        accelerate repeated lookups.
        """
        cache_key = f"{message.source}:{message.target}:{message.priority.value}"
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        for rule in self._rules:
            if rule.matches(message):
                self._route_cache[cache_key] = rule
                self._stats["routed"] += 1
                return rule

        self._stats["dropped"] += 1
        return None

    def get_all_matching_rules(self, message: NeuralMessage) -> List[RoutingRule]:
        """Return all rules that match a message (for broadcast)."""
        return [r for r in self._rules if r.matches(message)]

    def route_message(self, message: NeuralMessage) -> List[str]:
        """
        Determine delivery targets for a message.

        Returns list of target IDs the message should be sent to.
        """
        matching = self.get_all_matching_rules(message)
        targets = []
        for rule in matching:
            if rule.route_type == RouteType.BROADCAST:
                targets.append(rule.target_pattern)
                self._stats["broadcast"] += 1
            else:
                targets.append(rule.target_pattern)
        return targets

    def get_stats(self) -> Dict[str, int]:
        """Return routing statistics."""
        return dict(self._stats)

    def clear_rules(self) -> None:
        """Remove all routing rules."""
        self._rules.clear()
        self._route_cache.clear()
