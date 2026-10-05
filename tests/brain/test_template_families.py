"""Chat-template families: which request fields each family sends."""

import pytest

from mtc.brain.families import FAMILIES, TemplateFamily, get_family
from mtc.brain.providers.base import CompletionParams
from mtc.brain.providers.local_chat import build_request_body


def _params(**overrides):
    values = dict(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.6,
        max_tokens=128,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        frequency_penalty=0.3,
        thinking_mode=True,
    )
    values.update(overrides)
    return CompletionParams(**values)


def test_three_families_registered():
    assert set(FAMILIES) == {"generic", "qwen", "gemma"}
    for name, family in FAMILIES.items():
        assert isinstance(family, TemplateFamily)
        assert family.name == name
        assert get_family(name) is family


def test_generic_sends_standard_fields_only():
    body = build_request_body(get_family("generic"), _params(), model="")
    assert set(body) <= {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "stream",
        "frequency_penalty",
    }
    assert "chat_template_kwargs" not in body
    for field in ("top_k", "min_p", "repetition_penalty", "stop", "model"):
        assert field not in body, field
    assert body["frequency_penalty"] == 0.3
    off = build_request_body(get_family("generic"), _params(thinking_mode=False))
    assert "chat_template_kwargs" not in off and "stop" not in off
    assert get_family("generic").stop_when_thinking == ()


def test_qwen_and_gemma_add_their_stop_token_when_thinking():
    for name in ("qwen", "gemma"):
        family = get_family(name)
        assert family.stop_when_thinking, name
        thinking = build_request_body(family, _params(stop=["END"]))
        assert thinking["stop"][0] == "END"
        for token in family.stop_when_thinking:
            assert token in thinking["stop"], (name, token)
        quiet = build_request_body(family, _params(thinking_mode=False))
        assert "stop" not in quiet, name
    assert "<|im_end|>" in get_family("qwen").stop_when_thinking


def test_thinking_off_strategy_per_family():
    assert get_family("qwen").thinking_off == "chat_template_kwargs"
    assert get_family("gemma").thinking_off == "none"
    assert get_family("generic").thinking_off == "none"
    off = build_request_body(get_family("qwen"), _params(thinking_mode=False))
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    on = build_request_body(get_family("qwen"), _params(thinking_mode=True))
    assert "chat_template_kwargs" not in on
    gemma_off = build_request_body(get_family("gemma"), _params(thinking_mode=False))
    assert "chat_template_kwargs" not in gemma_off


def test_penalty_field_per_family():
    assert get_family("generic").penalty_field == "frequency_penalty"
    assert get_family("qwen").penalty_field == "repetition_penalty"
    assert get_family("gemma").penalty_field == "repetition_penalty"
    qwen = build_request_body(get_family("qwen"), _params())
    assert qwen["repetition_penalty"] == 1.1
    assert "frequency_penalty" not in qwen
    assert qwen["top_k"] == 20 and qwen["min_p"] == 0.05


def test_reasoning_tags_include_canonical_pair():
    for family in FAMILIES.values():
        assert ("<think>", "</think>") in family.reasoning_tags
    assert len(get_family("gemma").reasoning_tags) > 1


def test_unknown_family_rejected():
    with pytest.raises(ValueError, match="generic"):
        get_family("other")
    with pytest.raises(ValueError):
        get_family("")
