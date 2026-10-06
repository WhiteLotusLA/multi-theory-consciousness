"""Context types: honest absence vocabulary and a payload free of identity internals."""

import secrets
from dataclasses import fields
from datetime import datetime, timezone

import pytest

from mtc.pipeline.context_types import (
    ABSENT_REASONS,
    ModulatedContext,
    RawConversationContext,
    TheorySections,
    TurnResult,
)
from mtc.pipeline.speaker_trust import SpeakerIdentity


def _identity(**overrides) -> SpeakerIdentity:
    values = dict(
        name="Alice",
        role="creator",
        trust=0.9,
        authenticated=True,
        trusted=True,
        is_default=False,
    )
    values.update(overrides)
    return SpeakerIdentity(**values)


def test_turn_payload_has_no_identity_internals():
    token = secrets.token_hex(16)
    result = TurnResult(
        success=True,
        response="hello",
        thinking="reasoning",
        conversation_id="conv-1",
        message_count=3,
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        error=None,
        error_code=None,
        cycle_ok=True,
        speaker_label="[Alice]",
        provider_id="mock",
        blocked=False,
    )
    payload = result.to_payload()
    text = repr(payload)
    for forbidden in ("trust", "authenticated", "token", "role", "is_default", token):
        assert forbidden not in text
    assert payload["response"] == "hello"
    assert payload["speaker_label"] == "[Alice]"
    assert payload["timestamp"] == "2000-01-01T00:00:00+00:00"
    assert payload["cycle_ok"] is True
    assert payload["provider_id"] == "mock"
    assert set(payload) == {
        "success",
        "response",
        "thinking",
        "conversation_id",
        "message_count",
        "timestamp",
        "error",
        "error_code",
        "cycle_ok",
        "speaker_label",
        "provider_id",
        "blocked",
    }


def test_absent_reason_vocabulary():
    assert ABSENT_REASONS == (
        "warming_up",
        "cadence_not_reached",
        "no_producer",
        "no_input",
        "disabled",
    )
    context = ModulatedContext.unavailable("cycle_failed", 0.7)
    for reason in ABSENT_REASONS:
        context.absent_signals["field"] = reason
    with pytest.raises(ValueError, match="ABSENT_REASONS"):
        context.name_absent("other_field", "made_up")
    context.name_absent("rpt_deep_recurrence", "cadence_not_reached")
    assert context.absent_signals["rpt_deep_recurrence"] == "cadence_not_reached"


def test_unavailable_context_has_empty_narrative():
    context = ModulatedContext.unavailable("cycle_failed: boom", 0.65)
    assert context.prompt_context == ""
    assert context.behavioral_directives == []
    assert context.consciousness_self_awareness == ""
    assert context.consciousness_state is None
    assert context.cycle_ok is False
    assert context.failure == "cycle_failed: boom"
    assert context.generation_params == {"temperature": 0.65}
    assert context.winner_is_message is True
    assert context.workspace_winner == ""
    assert context.pad_sources_measured == ()
    assert context.extras == {}
    for section in fields(TheorySections):
        assert getattr(context.theory_sections, section.name) is None
    assert context.theory_sections.as_dict() == {}


def test_raw_context_defaults_and_identity():
    raw = RawConversationContext(message="hi there?", identity=_identity())
    assert raw.retrieved_memories == []
    assert raw.retrieved_concepts == []
    assert raw.ctm_thoughts == []
    assert raw.thought_gists == []
    assert raw.conversation_history == []
    assert raw.core_memory == {}
    assert raw.coherence_context == {}
    assert raw.tom_context == ""
    assert raw.recent_messages == []
    assert raw.now is None
    assert raw.identity.name == "Alice"
    with pytest.raises(TypeError):
        RawConversationContext(message="hi", identity="Alice")  # type: ignore[arg-type]
