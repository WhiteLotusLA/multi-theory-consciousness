"""
Tests for the time-limited focus release.

A released focus key is demoted in focus selection for exactly the requested
number of schema updates, then returns to baseline. It is never a hard
block: when nothing else is available the released content still surfaces.
"""

import asyncio
from types import SimpleNamespace

from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionShiftType,
    AttentionTarget,
)


def _content(summary, salience, source_module="test"):
    candidate = SimpleNamespace(
        content=summary,
        content_type="thought",
        summary=summary,
        source_module=source_module,
        metadata={},
    )
    return SimpleNamespace(candidate=candidate, salience=salience)


def _workspace_state(*contents):
    contents = list(contents)
    return SimpleNamespace(
        is_conscious=bool(contents),
        primary_content=contents[0] if contents else None,
        workspace_contents=contents,
    )


def _focus_summary(module, state):
    schema = asyncio.run(module.update_schema(state))
    return schema.current_focus.summary if schema.current_focus else None


def test_focus_key_format():
    key = AttentionSchemaModule.focus_key("vision", "x" * 100)
    assert key == "vision::" + "x" * 80
    assert AttentionSchemaModule.focus_key("", "") == "::"


def test_released_key_demoted_for_n_cycles_exactly():
    module = AttentionSchemaModule()
    top = _content("top item", 1.3)
    other = _content("other item", 1.0)
    state = _workspace_state(top, other)

    module.request_release(AttentionSchemaModule.focus_key("test", "top item"), 2)
    assert _focus_summary(module, state) == "other item"
    assert _focus_summary(module, state) == "other item"
    assert module._released_keys == {}
    assert _focus_summary(module, state) == "top item"


def test_single_cycle_release_lasts_one_cycle():
    module = AttentionSchemaModule()
    state = _workspace_state(_content("top item", 1.3), _content("other item", 1.0))
    module.request_release(AttentionSchemaModule.focus_key("test", "top item"), 1)
    assert _focus_summary(module, state) == "other item"
    assert _focus_summary(module, state) == "top item"


def test_release_falls_back_when_nothing_else():
    module = AttentionSchemaModule()
    only = _workspace_state(_content("only item", 1.3))
    module.request_release(AttentionSchemaModule.focus_key("test", "only item"), 3)
    assert _focus_summary(module, only) == "only item"


def test_release_skips_every_released_key():
    module = AttentionSchemaModule()
    state = _workspace_state(_content("a", 1.3), _content("b", 1.2), _content("c", 1.1))
    module.request_release(AttentionSchemaModule.focus_key("test", "a"), 2)
    module.request_release(AttentionSchemaModule.focus_key("test", "b"), 2)
    assert _focus_summary(module, state) == "c"


def test_release_never_negative_and_ignores_empty_key():
    module = AttentionSchemaModule()
    module.request_release("", 3)
    assert module._released_keys == {}
    module.request_release("k", 0)
    assert module._released_keys == {"k": 1}
    module._decay_released()
    module._decay_released()
    assert module._released_keys == {}


def test_statistics_include_uncertain():
    module = AttentionSchemaModule()
    stats = module.get_statistics()
    assert stats["uncertain_shifts"] == 0

    module.schema.current_focus = AttentionTarget(
        summary="automatic",
        source_module="test",
        shift_type=AttentionShiftType.HABITUAL,
    )
    asyncio.run(module.update_schema(_workspace_state(_content("next", 0.4))))
    stats = module.get_statistics()
    assert stats["uncertain_shifts"] == 2
    assert stats["voluntary_ratio"] is None
