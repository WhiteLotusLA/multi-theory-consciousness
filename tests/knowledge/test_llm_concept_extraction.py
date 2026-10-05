"""LLM concept extraction over the brain protocol (FakeBrain only)."""

import asyncio
import json
import re

import pytest

from mtc.brain.protocol import BrainReply, FakeBrain, NoAnswerError
from mtc.core.config import get_settings
from mtc.knowledge import llm_concept_extraction as module
from mtc.knowledge.llm_concept_extraction import (
    CATEGORIES,
    ConceptExtraction,
    ExtractedConcept,
    LLMConceptExtractor,
)

GOOD = json.dumps(
    [
        {"name": "river walk", "category": "activity", "relevance": 0.8},
        {"name": "curiosity", "category": "emotion", "relevance": 0.6},
    ]
)


def run(coro):
    return asyncio.run(coro)


def test_prompt_requests_json_with_categories():
    brain = FakeBrain(replies=[GOOD])
    extractor = LLMConceptExtractor(brain, max_concepts=4)
    run(extractor.extract("We walked by the river and wondered about it."))
    call = brain.calls[0]
    prompt = call["prompt"]
    assert "JSON" in prompt and "up to 4" in prompt
    for category in CATEGORIES:
        assert f'"{category}"' in prompt
    assert "We walked by the river and wondered about it." in prompt
    assert call["temperature"] == 0.3 and call["max_tokens"] == 800
    assert call["thinking_mode"] is False


def test_parses_fenced_json():
    """Fenced replies (and plain JSON, the unfenced baseline) parse alike."""
    for reply in (
        GOOD,
        f"Here you go:\n```json\n{GOOD}\n```\nDone.",
        f"```\n{GOOD}\n```",
        f"Sure. {GOOD} Anything else?",
    ):
        result = run(LLMConceptExtractor(FakeBrain(replies=[reply])).extract("text"))
        assert result.status == "measured", reply
        assert result.concepts == [
            ExtractedConcept("river walk", "activity", 0.8),
            ExtractedConcept("curiosity", "emotion", 0.6),
        ]


def test_reasoning_stripped_before_parse():
    reply = (
        '<think>maybe [{"name": "decoy", "category": "topic", "relevance": 1}]'
        f"</think>\n{GOOD}"
    )
    result = run(LLMConceptExtractor(FakeBrain(replies=[reply])).extract("text"))
    assert result.status == "measured"
    assert [c.name for c in result.concepts] == ["river walk", "curiosity"]


def test_no_answer_returns_status_not_concepts():
    for item in (NoAnswerError("timed out"), "", BrainReply(text="   "), "<think>x"):
        result = run(LLMConceptExtractor(FakeBrain(replies=[item])).extract("text"))
        assert result == ConceptExtraction(concepts=[], status="no_answer")


def test_unparseable_returns_status_not_concepts():
    for reply in ("no json here", "[{broken", '{"name": "x"}', "[1, 2"):
        result = run(LLMConceptExtractor(FakeBrain(replies=[reply])).extract("t"))
        assert result.status == "unparseable", reply
        assert result.concepts == []


def test_relevance_clamped():
    reply = json.dumps(
        [
            {"name": "high", "category": "topic", "relevance": 4.0},
            {"name": "low", "category": "topic", "relevance": -2},
            {"name": "unsaid", "category": "topic"},
            {"name": "words", "category": "topic", "relevance": "very"},
            {"name": "Odd", "category": "spaceship", "relevance": 0.5},
            {"name": "odd", "category": "topic", "relevance": 0.4},
            {"category": "topic", "relevance": 0.9},
            "loose string",
        ]
    )
    result = run(LLMConceptExtractor(FakeBrain(replies=[reply])).extract("text"))
    assert result.status == "measured"
    got = {c.name: (c.category, c.relevance) for c in result.concepts}
    assert got == {
        "high": ("topic", 1.0),
        "low": ("topic", 0.0),
        "unsaid": ("topic", None),  # an unstated relevance is not invented
        "words": ("topic", None),
        "Odd": ("concept", 0.5),  # an unknown category becomes "concept"
    }


def test_max_concepts_caps_the_list():
    reply = json.dumps(
        [{"name": f"c{i}", "category": "topic", "relevance": 0.5} for i in range(6)]
    )
    extractor = LLMConceptExtractor(FakeBrain(replies=[reply]), max_concepts=3)
    assert len(run(extractor.extract("text")).concepts) == 3


def test_blank_text_does_not_ask_the_brain():
    brain = FakeBrain()
    result = run(LLMConceptExtractor(brain).extract("   "))
    assert result == ConceptExtraction(concepts=[], status="measured")
    assert brain.calls == []


def test_extract_batch_keeps_order_and_statuses():
    brain = FakeBrain(replies=[GOOD, NoAnswerError("x"), "nonsense"])
    results = run(LLMConceptExtractor(brain).extract_batch(["a", "b", "c"]))
    assert [r.status for r in results] == ["measured", "no_answer", "unparseable"]


def test_prompt_example_is_synthetic():
    extractor = LLMConceptExtractor(FakeBrain())
    prompt = extractor.build_prompt("text")
    example = re.search(r"EXAMPLE:\s*(\[.*?\])", prompt, flags=re.S)
    assert example, prompt
    items = json.loads(example.group(1))
    assert {item["category"] for item in items} <= set(CATEGORIES)
    for item in items:
        # no proper noun: every example name is lower case
        assert item["name"] == item["name"].lower(), item
    settings = get_settings()
    for identity in (settings.SYSTEM_NAME, settings.CREATOR_NAME):
        assert identity not in prompt
    assert "real names" in (module.__doc__ or "")


def test_constructor_validates_numbers():
    with pytest.raises(ValueError):
        LLMConceptExtractor(FakeBrain(), max_concepts=0)
    with pytest.raises(ValueError):
        LLMConceptExtractor(FakeBrain(), max_tokens=0)
