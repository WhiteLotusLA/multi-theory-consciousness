"""
Tests for honest attention provenance.

A shift is classified by its type, never by its strength: VOLUNTARY and
GOAL_DRIVEN are chosen, CAPTURED and EMOTIONAL are pulled, HABITUAL and
unknown are uncertain. voluntary_ratio = chosen / (chosen + pulled), with
uncertain excluded and 0.5 when nothing is classifiable. The legacy
``legacy_strength`` mode keeps the v0.4.0 rule (VOLUNTARY or strength at or
above 0.6 counts voluntary) so the published numbers still reproduce.
"""

import asyncio
from types import SimpleNamespace

import pytest

from mtc.consciousness.attention_schema import (
    DEFAULT_CAPTURE_SALIENCE_THRESHOLD,
    DEFAULT_FOCUS_THRESHOLD,
    DEFAULT_HYPERFOCUS_THRESHOLD,
    SALIENCE_SCALE_MAX,
    AttentionSchemaModule,
    AttentionSchemaState,
    AttentionShiftType,
    AttentionState,
    AttentionTarget,
    provenance_of,
)


def _target(shift_type, strength=0.5, salience=0.5, summary="something"):
    return AttentionTarget(
        content="x",
        summary=summary,
        source_module="test",
        attention_strength=strength,
        salience=salience,
        shift_type=shift_type,
    )


def _content(summary, salience, source_module="test", emotional=0.0):
    candidate = SimpleNamespace(
        content=summary,
        content_type="thought",
        summary=summary,
        source_module=source_module,
        metadata={"emotional_salience": emotional},
    )
    return SimpleNamespace(candidate=candidate, salience=salience)


def _workspace_state(*contents):
    contents = list(contents)
    return SimpleNamespace(
        is_conscious=bool(contents),
        primary_content=contents[0] if contents else None,
        workspace_contents=contents,
    )


def _update(module, *contents):
    return asyncio.run(module.update_schema(_workspace_state(*contents)))


# ============================================================================
# Classification
# ============================================================================


def test_chosen_pulled_uncertain_classification():
    assert provenance_of(AttentionShiftType.VOLUNTARY) == "chosen"
    assert provenance_of(AttentionShiftType.GOAL_DRIVEN) == "chosen"
    assert provenance_of(AttentionShiftType.CAPTURED) == "pulled"
    assert provenance_of(AttentionShiftType.EMOTIONAL) == "pulled"
    assert provenance_of(AttentionShiftType.HABITUAL) == "uncertain"
    assert provenance_of(None) == "uncertain"

    state = AttentionSchemaState()
    state.current_focus = _target(AttentionShiftType.CAPTURED, strength=1.4)
    assert (state.voluntary_shifts, state.captured_shifts) == (0, 1)
    state.current_focus = _target(
        AttentionShiftType.GOAL_DRIVEN, strength=0.2, summary="b"
    )
    assert (state.voluntary_shifts, state.captured_shifts) == (1, 1)
    state.current_focus = _target(AttentionShiftType.HABITUAL, summary="c")
    assert state.uncertain_shifts == 1
    missing = _target(AttentionShiftType.CAPTURED, summary="d")
    missing.shift_type = None
    state.current_focus = missing
    assert state.uncertain_shifts == 2
    assert state.voluntary_shifts == 1


def test_ratio_excludes_uncertain():
    state = AttentionSchemaState()
    state.current_focus = _target(AttentionShiftType.VOLUNTARY, summary="a")
    state.current_focus = _target(AttentionShiftType.CAPTURED, summary="b")
    state.current_focus = _target(AttentionShiftType.HABITUAL, summary="c")
    state.current_focus = _target(AttentionShiftType.HABITUAL, summary="d")
    assert state.voluntary_shifts == 1
    assert state.captured_shifts == 1
    assert state.uncertain_shifts == 2
    assert state.voluntary_ratio == pytest.approx(0.5)

    state.current_focus = _target(AttentionShiftType.EMOTIONAL, summary="e")
    assert state.voluntary_ratio == pytest.approx(1 / 3)


def test_ratio_half_when_nothing_classifiable():
    state = AttentionSchemaState()
    for summary in ("a", "b", "c"):
        state.current_focus = _target(AttentionShiftType.HABITUAL, summary=summary)
    assert state.uncertain_shifts == 3
    assert state.voluntary_ratio == 0.5

    module = AttentionSchemaModule()
    _update(module, _content("first", 0.5))
    _update(module, _content("second", 0.6))
    stats = module.get_statistics()
    assert stats["uncertain_shifts"] == 2
    assert stats["voluntary_shifts"] == 0
    assert stats["captured_shifts"] == 0
    assert stats["voluntary_ratio"] == 0.5


def test_ratio_uses_post_increment_counts():
    module = AttentionSchemaModule()
    schema = _update(module, _content("strong pull", 1.3))
    assert schema.captured_shifts == 1
    # The ratio already reflects this cycle's shift.
    assert schema.voluntary_ratio == 0.0


def test_counts_only_on_focus_change():
    module = AttentionSchemaModule()
    _update(module, _content("same thing", 1.3))
    _update(module, _content("same thing", 1.3))
    _update(module, _content("same thing", 1.3))
    assert module.schema.captured_shifts == 1

    # Losing focus is not a shift into anything.
    _update(module)
    assert module.schema.captured_shifts == 1
    assert module.schema.uncertain_shifts == 0

    _update(module, _content("new thing", 1.3))
    assert module.schema.captured_shifts == 2

    # The setter follows the same rule: re-assigning the same target, or a
    # new object for the same content, does not count again.
    state = AttentionSchemaState()
    target = _target(AttentionShiftType.CAPTURED)
    state.current_focus = target
    state.current_focus = target
    state.current_focus = _target(AttentionShiftType.CAPTURED)
    assert state.captured_shifts == 1
    assert len(state.attention_history) == 1


def test_voluntary_request_counts_once_as_chosen():
    module = AttentionSchemaModule()
    asyncio.run(module.request_voluntary_shift("chosen topic", reason="test"))
    schema = _update(module, _content("the chosen topic", 0.5))
    assert schema.current_focus.voluntary is True
    assert schema.current_focus.shift_type == AttentionShiftType.VOLUNTARY
    # Staying on the same content is not a second shift.
    _update(module, _content("the chosen topic", 0.5))
    assert module.schema.voluntary_shifts == 1
    assert module.schema.uncertain_shifts == 0
    assert module.schema.voluntary_ratio == 1.0


# ============================================================================
# Legacy mode
# ============================================================================


def test_legacy_mode_counts_by_strength():
    state = AttentionSchemaState(provenance_mode="legacy_strength")
    state.current_focus = _target(AttentionShiftType.CAPTURED, strength=0.7)
    assert (state.voluntary_shifts, state.captured_shifts) == (1, 0)
    state.current_focus = _target(AttentionShiftType.CAPTURED, strength=0.5)
    assert (state.voluntary_shifts, state.captured_shifts) == (1, 1)
    state.current_focus = _target(AttentionShiftType.VOLUNTARY, strength=0.1)
    assert (state.voluntary_shifts, state.captured_shifts) == (2, 1)
    # Any new object counts, even for the same content.
    state.current_focus = _target(AttentionShiftType.HABITUAL, strength=0.2)
    assert state.captured_shifts == 2
    assert state.uncertain_shifts == 0
    assert state.voluntary_ratio == pytest.approx(0.5)

    module = AttentionSchemaModule(
        provenance_mode="legacy_strength",
        focus_threshold=0.6,
        hyperfocus_threshold=0.9,
        capture_salience_threshold=0.8,
    )
    assert module.schema.provenance_mode == "legacy_strength"
    schema = _update(module, _content("pull", 0.95))
    # v0.4.0 computed the ratio before this cycle's increment.
    assert schema.captured_shifts == 1
    assert schema.voluntary_ratio == 0.5
    schema = _update(module, _content("pull", 0.95))
    assert schema.voluntary_ratio == 0.0


def test_legacy_mode_counts_lost_focus_as_capture():
    module = AttentionSchemaModule(provenance_mode="legacy_strength")
    _update(module, _content("pull", 0.95))
    _update(module)
    assert module.schema.captured_shifts == 2


def test_invalid_provenance_mode_raises():
    with pytest.raises(ValueError):
        AttentionSchemaModule(provenance_mode="by_vibes")
    with pytest.raises(ValueError):
        AttentionSchemaState(provenance_mode="by_vibes")


# ============================================================================
# Shift inference and thresholds
# ============================================================================


def test_capture_threshold_scale_aware():
    assert SALIENCE_SCALE_MAX == 1.5
    assert DEFAULT_FOCUS_THRESHOLD == pytest.approx(0.6 * 1.5)
    assert DEFAULT_HYPERFOCUS_THRESHOLD == pytest.approx(0.9 * 1.5)
    assert DEFAULT_CAPTURE_SALIENCE_THRESHOLD == pytest.approx(0.8 * 1.5)

    module = AttentionSchemaModule()
    # Above the old 0-1 capture bar but below the scaled one: not a known grab.
    assert (
        module._infer_shift_type(_target(None, salience=1.0), None)
        == AttentionShiftType.HABITUAL
    )
    assert (
        module._infer_shift_type(_target(None, salience=1.2), None)
        == AttentionShiftType.CAPTURED
    )

    custom = AttentionSchemaModule(capture_salience_threshold=0.9)
    assert (
        custom._infer_shift_type(_target(None, salience=1.0), None)
        == AttentionShiftType.CAPTURED
    )

    emotional = _target(None, salience=0.3)
    emotional.metadata = {"emotional_salience": 0.8}
    assert module._infer_shift_type(emotional, None) == AttentionShiftType.EMOTIONAL


def test_state_bands_on_the_salience_scale():
    module = AttentionSchemaModule()
    focused = _update(module, _content("steady", 1.0))
    assert focused.attention_state == AttentionState.FOCUSED

    module = AttentionSchemaModule()
    hyper = _update(module, _content("intense", 1.4))
    assert hyper.attention_state == AttentionState.HYPERFOCUSED


def test_no_keyword_goal_driven():
    module = AttentionSchemaModule()
    worded = _target(None, salience=0.5, summary="goal: finish the task")
    assert module._infer_shift_type(worded, None) == AttentionShiftType.HABITUAL

    legacy = AttentionSchemaModule(provenance_mode="legacy_strength")
    assert legacy._infer_shift_type(worded, None) == AttentionShiftType.GOAL_DRIVEN
    plain = _target(None, salience=0.1, summary="plain")
    assert legacy._infer_shift_type(plain, None) == AttentionShiftType.CAPTURED


def test_threshold_validation(caplog):
    with pytest.raises(ValueError):
        AttentionSchemaModule(focus_threshold=0.0)
    with pytest.raises(ValueError):
        AttentionSchemaModule(capture_salience_threshold=-1.0)
    # An inverted band is legal (a parameter sweep may sample one) but logged.
    with caplog.at_level("WARNING"):
        module = AttentionSchemaModule(focus_threshold=1.4, hyperfocus_threshold=1.0)
    assert module.focus_threshold == 1.4
    assert any("unreachable" in r.getMessage() for r in caplog.records)
