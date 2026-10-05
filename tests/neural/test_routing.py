"""Tests for rule-based routing of neural messages."""

import pytest

from mtc.neural.protocols.message_format import (
    MessagePriority,
    NeuralMessage,
    NeuralMessageType,
)
from mtc.neural.protocols.routing import NeuralRouter, RouteType, RoutingRule


def _msg(source: str, target: str, priority=MessagePriority.NORMAL, kind=None):
    return NeuralMessage(
        source=source,
        target=target,
        message_type=kind or NeuralMessageType.SPIKE_TRAIN,
        priority=priority,
    )


class TestRoutingRule:
    def test_defaults(self):
        rule = RoutingRule(source_pattern="snn*", target_pattern="htm*")

        assert rule.route_type == RouteType.DIRECT
        assert rule.priority_threshold == MessagePriority.NORMAL
        assert rule.transform_func is None
        assert rule.enabled is True

    def test_glob_matching_on_source_and_target(self):
        rule = RoutingRule(
            source_pattern="snn*",
            target_pattern="htm*",
            priority_threshold=MessagePriority.LOW,
        )

        assert rule.matches(_msg("snn_primary", "htm_main")) is True
        assert rule.matches(_msg("lsm_main", "htm_main")) is False
        assert rule.matches(_msg("snn_primary", "lsm_main")) is False

    def test_disabled_rule_never_matches(self):
        rule = RoutingRule(source_pattern="*", target_pattern="*", enabled=False)

        assert rule.matches(_msg("a", "b")) is False

    @pytest.mark.parametrize(
        "priority, expected",
        [
            (MessagePriority.CRITICAL, True),
            (MessagePriority.HIGH, True),
            (MessagePriority.NORMAL, False),
            (MessagePriority.LOW, False),
        ],
    )
    def test_priority_threshold(self, priority, expected):
        rule = RoutingRule(
            source_pattern="*",
            target_pattern="*",
            priority_threshold=MessagePriority.HIGH,
        )

        assert rule.matches(_msg("a", "b", priority)) is expected

    def test_transform_function_is_kept(self):
        def tag(message):
            message.metadata["seen"] = True
            return message

        rule = RoutingRule(source_pattern="*", target_pattern="*", transform_func=tag)

        assert rule.transform_func(_msg("a", "b")).metadata == {"seen": True}


class TestNeuralRouter:
    @pytest.fixture
    def router(self):
        router = NeuralRouter()
        router.add_rule(RoutingRule(source_pattern="snn*", target_pattern="htm*"))
        router.add_rule(RoutingRule(source_pattern="lsm*", target_pattern="ctm*"))
        router.add_rule(
            RoutingRule(
                source_pattern="hub",
                target_pattern="*",
                route_type=RouteType.BROADCAST,
            )
        )
        return router

    def test_first_matching_rule_wins(self, router):
        rule = router.get_route(_msg("snn_primary", "htm_main"))

        assert rule is not None
        assert rule.source_pattern == "snn*"
        assert rule.route_type == RouteType.DIRECT

    def test_no_match_is_dropped(self, router):
        assert router.get_route(_msg("unknown", "nowhere")) is None
        assert router.get_stats()["dropped"] == 1

    def test_broadcast_rules_are_listed(self, router):
        message = _msg("hub", "any", kind=NeuralMessageType.CONTROL)

        matching = router.get_all_matching_rules(message)

        assert [r.route_type for r in matching] == [RouteType.BROADCAST]
        assert router.route_message(message) == ["*"]
        assert router.get_stats()["broadcast"] == 1

    def test_route_message_returns_target_patterns(self, router):
        assert router.route_message(_msg("snn_primary", "htm_main")) == ["htm*"]

    def test_routed_count_and_cache(self, router):
        message = _msg("snn_primary", "htm_main")

        first = router.get_route(message)
        second = router.get_route(message)

        assert first is second
        # The second lookup is served from the cache and not counted again
        assert router.get_stats()["routed"] == 1

    def test_adding_a_rule_invalidates_the_cache(self):
        router = NeuralRouter()
        router.add_rule(
            RoutingRule(
                source_pattern="snn*",
                target_pattern="htm*",
                priority_threshold=MessagePriority.CRITICAL,
            )
        )
        message = _msg("snn_primary", "htm_main", MessagePriority.HIGH)
        assert router.get_route(message) is None

        router.add_rule(
            RoutingRule(
                source_pattern="snn*",
                target_pattern="htm*",
                route_type=RouteType.HIERARCHICAL,
                priority_threshold=MessagePriority.LOW,
            )
        )

        assert router.get_route(message).route_type == RouteType.HIERARCHICAL

    def test_remove_and_clear(self):
        router = NeuralRouter()
        router.add_rule(RoutingRule(source_pattern="src", target_pattern="dst"))
        router.add_rule(RoutingRule(source_pattern="x", target_pattern="y"))
        message = _msg("src", "dst")
        assert router.get_route(message) is not None

        assert router.remove_rule("src", "dst") is True
        assert router.remove_rule("src", "dst") is False
        assert router.get_route(message) is None

        router.clear_rules()
        assert router.get_route(_msg("x", "y")) is None

    def test_many_rules_route_correctly(self):
        router = NeuralRouter()
        for i in range(100):
            router.add_rule(
                RoutingRule(
                    source_pattern=f"system_{i}",
                    target_pattern=f"system_{(i + 1) % 100}",
                )
            )

        for i in (0, 42, 99):
            rule = router.get_route(_msg(f"system_{i}", f"system_{(i + 1) % 100}"))
            assert rule.source_pattern == f"system_{i}"
        assert router.get_route(_msg("system_3", "system_9")) is None


def test_route_type_values():
    assert [t.value for t in RouteType] == ["direct", "broadcast", "hierarchical"]
