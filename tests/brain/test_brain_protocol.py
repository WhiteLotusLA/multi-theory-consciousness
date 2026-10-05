"""Brain protocol, the deterministic fake and reasoning-block stripping."""

import pytest

from mtc.brain import (
    BrainProtocol,
    BrainReply,
    FakeBrain,
    NoAnswerError,
    strip_reasoning_blocks,
)


def test_fake_brain_satisfies_protocol():
    assert isinstance(FakeBrain(), BrainProtocol)
    assert issubclass(NoAnswerError, RuntimeError)


@pytest.mark.asyncio
async def test_fake_brain_is_deterministic():
    first = await FakeBrain().think("a prompt", system_prompt="sys")
    second = await FakeBrain().think("a prompt", system_prompt="sys")
    other = await FakeBrain().think("another prompt", system_prompt="sys")
    assert isinstance(first, BrainReply)
    assert first.text and first.text == second.text
    assert first.text != other.text


@pytest.mark.asyncio
async def test_rules_match_by_substring_before_the_digest():
    brain = FakeBrain(rules={"weather": "It is dry."})
    assert (await brain.think("tell me about the weather")).text == "It is dry."
    assert (await brain.think("something else")).text != "It is dry."


@pytest.mark.asyncio
async def test_scripted_replies_and_calls_recorded():
    brain = FakeBrain(replies=["one", BrainReply("two", finish_reason="stop")])
    first = await brain.think("p1", system_prompt="s1", max_tokens=64)
    second = await brain.think("p2", temperature=0.1, thinking_mode=True)
    assert first.text == "one"
    assert second.text == "two" and second.finish_reason == "stop"
    assert [c["prompt"] for c in brain.calls] == ["p1", "p2"]
    assert brain.calls[0]["system_prompt"] == "s1"
    assert brain.calls[0]["max_tokens"] == 64
    assert brain.calls[1]["thinking_mode"] is True
    # the script is exhausted: later prompts fall through to the digest reply
    third = await brain.think("p3")
    assert third.text and len(brain.calls) == 3


@pytest.mark.asyncio
async def test_scripted_exception_is_raised():
    brain = FakeBrain(replies=[NoAnswerError("no answer"), "after"])
    with pytest.raises(NoAnswerError):
        await brain.think("p")
    assert (await brain.think("p")).text == "after"
    assert len(brain.calls) == 2


def test_strip_reasoning_blocks_removes_closed_blocks():
    text = "<think>plan\nmore</think>Hello there.<THINK>x</THINK> End."
    assert strip_reasoning_blocks(text) == "Hello there. End."


def test_strip_reasoning_blocks_handles_unclosed_tag():
    assert strip_reasoning_blocks("Answer. <think>never finished") == "Answer."
    assert strip_reasoning_blocks("<think>only reasoning") == ""


def test_strip_reasoning_blocks_handles_orphan_closing_tag():
    assert strip_reasoning_blocks("reasoning text</think>The answer") == "The answer"


def test_strip_reasoning_blocks_custom_tags_and_plain_text():
    text = "<reasoning>x</reasoning>Visible <think>kept</think>"
    assert strip_reasoning_blocks(text, tags=("reasoning",)) == (
        "Visible <think>kept</think>"
    )
    assert strip_reasoning_blocks("plain text") == "plain text"
    assert strip_reasoning_blocks("") == ""
