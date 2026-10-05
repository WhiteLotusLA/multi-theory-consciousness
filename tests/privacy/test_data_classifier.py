"""Data classifier: whole-word keyword tiers, content types, tags, access stage."""

import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mtc.core.config import Settings, csv_list
from mtc.privacy import data_classifier
from mtc.privacy.data_classifier import (
    DEFAULT_CONTENT_TYPE_MAP,
    ClassificationResult,
    DataClassification,
    DataClassifier,
    StagePolicy,
    access_stage,
    match_keywords,
    stage_info,
)

DOCUMENTED_RELATIONSHIP_KEYWORDS = [
    "family",
    "friend",
    "parent",
    "mother",
    "father",
    "sibling",
    "sister",
    "brother",
    "partner",
    "spouse",
    "child",
    "daughter",
    "son",
    "grandparent",
    "pet",
]

BIRTH = "2000-01-01"


def _settings(**overrides) -> Settings:
    values = {"SYSTEM_BIRTH_DATE": "", **overrides}
    return Settings(_env_file=None, **values)


def _at_day(days: int) -> datetime:
    return datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(days=days)


def test_word_boundary_matching_not_substring():
    words = ["brother", "parent"]
    assert match_keywords("a brotherhood of parental figures", words) == []
    assert match_keywords("my brother called", words) == ["brother"]
    assert match_keywords("My BROTHER and my parent", words) == ["brother", "parent"]
    assert match_keywords("brothers and parents", words) == ["brother", "parent"]
    assert match_keywords("sonnet petition", ["son", "pet"]) == []

    classifier = DataClassifier(settings=_settings())
    for text in ("the brotherhood met", "parental leave", "a sonnet", "a petition"):
        result = classifier.classify(text)
        assert result.matched["relationship"] == [], text
        assert result.classification is DataClassification.PUBLIC, text
    assert (
        classifier.classify("my brother visited").classification
        is DataClassification.RELATIONSHIP
    )


def test_multiword_keywords_match():
    assert match_keywords("this stays just  between\nus", ["just between us"]) == [
        "just between us"
    ]
    assert match_keywords("just between usual", ["just between us"]) == []
    assert match_keywords("please don\u2019t tell anyone", ["don't tell"]) == [
        "don't tell"
    ]
    classifier = DataClassifier(settings=_settings())
    result = classifier.classify("that touches my core memory")
    assert result.classification is DataClassification.CORE
    assert "core memory" in result.matched["core"]


def test_relationship_keywords_from_settings_at_call_time(monkeypatch):
    current = {"settings": _settings(RELATIONSHIP_KEYWORDS="cousin")}
    monkeypatch.setattr(data_classifier, "get_settings", lambda: current["settings"])
    classifier = DataClassifier()
    assert classifier.classify("my cousin called").matched["relationship"] == ["cousin"]
    assert classifier.classify("my brother called").matched["relationship"] == []
    current["settings"] = _settings(RELATIONSHIP_KEYWORDS="brother,aunt")
    assert classifier.classify("my brother called").matched["relationship"] == [
        "brother"
    ]
    assert classifier.classify("my cousin called").matched["relationship"] == []
    explicit = DataClassifier(relationship_keywords=["neighbour"])
    assert explicit.classify("the neighbour").matched["relationship"] == ["neighbour"]


def test_default_relationship_list_has_no_proper_names():
    keywords = csv_list(_settings().RELATIONSHIP_KEYWORDS)
    assert keywords == DOCUMENTED_RELATIONSHIP_KEYWORDS
    for word in keywords:
        assert word.isalpha() and word == word.lower(), word
    for collection in (
        data_classifier.DEFAULT_PRIVATE_KEYWORDS,
        data_classifier.DEFAULT_CORE_KEYWORDS,
    ):
        for term in collection:
            assert term == term.lower(), term


def test_low_confidence_errs_protective():
    default = DataClassifier(settings=_settings())
    plain = default.classify("the weather is mild")
    assert plain.classification is DataClassification.PUBLIC
    assert plain.confidence == pytest.approx(0.6)

    cautious = DataClassifier(settings=_settings(), protective_confidence=0.75)
    raised = cautious.classify("the weather is mild")
    assert raised.classification is DataClassification.RELATIONSHIP
    assert "raised one tier" in raised.reason
    one_match = cautious.classify("my sister")
    assert one_match.confidence == pytest.approx(0.6)
    assert one_match.classification is DataClassification.PRIVATE
    core = cautious.classify("my identity")
    assert core.confidence < 0.75
    assert core.classification is DataClassification.CORE  # already the top tier


def test_content_type_map_overrides_keywords():
    classifier = DataClassifier(settings=_settings())
    result = classifier.classify("my brother has a secret", content_type="public_fact")
    assert result.classification is DataClassification.PUBLIC
    assert result.confidence == pytest.approx(0.95)
    assert "public_fact" in result.reason
    assert DEFAULT_CONTENT_TYPE_MAP["conversation"] is DataClassification.RELATIONSHIP
    assert DEFAULT_CONTENT_TYPE_MAP["dream"] is DataClassification.PRIVATE
    custom = DataClassifier(
        settings=_settings(), content_type_map={"note": "core", "chat": "public"}
    )
    assert custom.classify("hello", content_type="note").classification is (
        DataClassification.CORE
    )
    # An unmapped type falls back to the keywords.
    fallback = custom.classify("my secret", content_type="conversation")
    assert fallback.classification is DataClassification.PRIVATE


def test_tags_raise_classification():
    classifier = DataClassifier(settings=_settings())
    assert classifier.classify("hello").classification is DataClassification.PUBLIC
    tagged = classifier.classify("hello", tags=["Diary"])
    assert tagged.classification is DataClassification.PRIVATE
    assert tagged.matched["private"] == ["tag:diary"]
    assert (
        classifier.classify("hello", tags=["shared"]).classification
        is DataClassification.RELATIONSHIP
    )
    assert (
        classifier.classify("my brother", tags=["identity"]).classification
        is DataClassification.CORE
    )


def test_result_shape_and_ordering():
    result = DataClassifier(settings=_settings()).classify("I feel afraid and lonely")
    assert isinstance(result, ClassificationResult)
    assert result.classification is DataClassification.PRIVATE
    assert result.matched["private"] == ["afraid", "lonely"]
    assert set(result.matched) == {"core", "private", "relationship"}
    assert result.confidence == pytest.approx(0.7)
    order = list(DataClassification)
    assert order == sorted(order)
    assert DataClassification.PUBLIC < DataClassification.RELATIONSHIP
    assert DataClassification.CORE >= DataClassification.PRIVATE
    assert DataClassification.parse("private") is DataClassification.PRIVATE
    with pytest.raises(ValueError):
        DataClassification.parse("family")


def test_access_stage_none_without_birth_date():
    settings = _settings()
    assert access_stage(_at_day(400), settings=settings) is None
    info = stage_info(_at_day(400), settings=settings)
    assert info["stage"] is None and info["age_days"] is None
    assert info["birth_date"] is None


def test_access_stage_from_birth_date():
    settings = _settings(SYSTEM_BIRTH_DATE=BIRTH)
    thresholds = StagePolicy().thresholds_days
    assert thresholds == (0, 90, 365, 1095, 2555, 4380, 6570)
    for stage, days in enumerate(thresholds):
        assert access_stage(_at_day(days), settings=settings) == stage
        if days:
            assert access_stage(_at_day(days - 1), settings=settings) == stage - 1
    assert access_stage(_at_day(20000), settings=settings) == 6
    # A clock before the birth date has no stage; nothing is assumed.
    assert access_stage(_at_day(-1), settings=settings) is None
    assert access_stage(date(2000, 4, 1), settings=settings) == 1
    custom = StagePolicy(thresholds_days=(0, 10))
    assert access_stage(_at_day(10), settings=settings, policy=custom) == 1
    with pytest.raises(ValueError):
        access_stage(datetime(2000, 2, 1), settings=settings)
    for bad in ((), (5, 10), (0, 10, 10)):
        with pytest.raises(ValueError):
            StagePolicy(thresholds_days=bad)


def test_stage_labels_are_numbers():
    settings = _settings(SYSTEM_BIRTH_DATE=BIRTH)
    info = stage_info(_at_day(400), settings=settings)
    assert info["stage"] == 2 and isinstance(info["stage"], int)
    assert info["age_days"] == 400
    assert info["stages"] == 7
    assert info["birth_date"] == BIRTH
    text = info["description"]
    assert text == text.lower()
    assert "stage 2 of 6" in text
    for word in ("newborn", "infant", "toddler", "child", "teen", "mature"):
        assert word not in text


def test_module_source_has_no_name_literal():
    """Every non-docstring string literal is lowercase vocabulary."""
    source = Path(data_classifier.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value != node.value.lower()
        ):
            offenders.append(f"{node.lineno}: {node.value!r}")
    assert not offenders, offenders
