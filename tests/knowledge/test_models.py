"""Knowledge graph models: neutral examples, written relationships only."""

import json
import re
from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import BaseModel

from mtc.knowledge import models
from mtc.knowledge.models import (
    Concept,
    Emotion,
    Experience,
    Learned,
    Memory,
    MentionedIn,
    Person,
    ReferencesIn,
    RelatesTo,
    Taught,
    Topic,
    node_to_cypher_properties,
    relationship_to_cypher_properties,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NODE_MODELS = (Person, Concept, Experience, Emotion, Memory, Topic)
RELATIONSHIP_MODELS = (Taught, Learned, RelatesTo, ReferencesIn, MentionedIn)
SYNTHETIC_NAMES = {"Alice", "Operator", "Guest"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _examples():
    for model in NODE_MODELS + RELATIONSHIP_MODELS:
        extra = model.model_config.get("json_schema_extra") or {}
        example = extra.get("example")
        if example is not None:
            yield model, example


def test_examples_are_neutral():
    seen = list(_examples())
    assert {model for model, _ in seen} >= set(NODE_MODELS)
    for model, example in seen:
        text = json.dumps(example)
        for value in example.values():
            if isinstance(value, str) and DATE.match(value):
                assert value.startswith("2000-01-01"), (model.__name__, value)
        if model is Person:
            assert example["name"] in SYNTHETIC_NAMES
        if model is MentionedIn:
            assert example["speaker"] in SYNTHETIC_NAMES
        assert "bond" not in text and "trust" not in text, model.__name__
        model.model_validate(example)  # every example is a valid instance


def test_mentioned_in_and_references_in_exist():
    mention = MentionedIn(speaker="Alice", when=T0)
    assert mention.speaker == "Alice" and mention.mentions == 1
    reference = ReferencesIn(relevance=0.4, extraction_method="keyword", when=T0)
    assert reference.relevance == 0.4
    with pytest.raises(ValueError):
        ReferencesIn(relevance=1.5)


def test_unwritten_relationships_not_declared():
    for name in ("Experienced", "Felt", "TriggeredBy", "DiscussedIn", "ConnectedTo"):
        assert not hasattr(models, name), name
    declared = {
        name
        for name, value in vars(models).items()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value is not BaseModel
    }
    assert declared == {m.__name__ for m in NODE_MODELS + RELATIONSHIP_MODELS}


def test_concept_model_matches_merge_properties():
    fields = set(Concept.model_fields)
    assert {
        "id",
        "name",
        "category",
        "description",
        "importance",
        "mention_count",
        "learned_at",
        "last_mentioned",
        "understanding_level",
    } == fields
    concept = Concept(name="river", category="nature", learned_at=T0)
    assert concept.understanding_level is None  # never assumed
    assert concept.mention_count == 0 and concept.description == ""
    assert Learned(when=T0, proficiency=0.4).proficiency == 0.4
    assert Learned(when=T0).proficiency is None


def test_unmeasured_relationship_fields_default_to_none():
    learned = Learned(when=T0)
    assert learned.confidence is None and learned.depth is None
    assert Taught(when=T0).effectiveness is None
    assert relationship_to_cypher_properties(Learned(when=T0)) == {
        "when": "2000-01-01T12:00:00+00:00"
    }
    props = relationship_to_cypher_properties(Taught(when=T0))
    assert "effectiveness" not in props


def test_person_has_no_bond_or_trust_and_no_assumed_first_meeting():
    person = Person(name="Alice", relationship="friend")
    assert person.first_met is None
    assert not any("bond" in f or "trust" in f for f in Person.model_fields)


def test_timestamps_default_to_aware_utc():
    for value in (
        Concept(name="x", category="y").learned_at,
        Taught(method="conversation").when,
        Topic(name="x").first_mentioned,
    ):
        assert value.tzinfo is not None and value.utcoffset().total_seconds() == 0


def test_cypher_property_helpers():
    person = Person(name="Alice", relationship="friend", first_met=T0)
    props = node_to_cypher_properties(person)
    assert isinstance(props["id"], str) and UUID(props["id"])
    assert props["first_met"] == "2000-01-01T12:00:00+00:00"
    assert "interaction_count" in props
    assert "first_met" not in node_to_cypher_properties(
        Person(name="Alice", relationship="friend")
    )  # a None value is not written
    rel = relationship_to_cypher_properties(
        Taught(when=T0, method="conversation", effectiveness=0.9)
    )
    assert rel == {
        "when": "2000-01-01T12:00:00+00:00",
        "method": "conversation",
        "effectiveness": 0.9,
    }
    memory = Memory(
        id=str(UUID(int=7)),
        postgres_id=7,
        memory_type="episodic",
        content_preview="x",
        created_at=T0,
    )
    assert node_to_cypher_properties(memory)["postgres_id"] == 7
    assert "importance" not in node_to_cypher_properties(memory)
