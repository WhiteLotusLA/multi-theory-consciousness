"""Keyword concept matching against the graph's known concept names."""

from mtc.knowledge.concept_extraction import ConceptExtractor, create_concept_extractor
from mtc.memory.protocols import ConceptMatcher

KNOWN = ["consciousness", "neural networks", "emotions", "theory of mind"]


def test_exact_then_fuzzy_matching():
    extractor = ConceptExtractor()
    text = "Consciousness and consciousness again; the networks hum."
    matches = dict(extractor.match_concepts(text, KNOWN))
    assert matches["consciousness"] == 0.7  # two exact hits: 0.5 + 2 * 0.1
    # "networks" overlaps one of the two words of "neural networks"
    assert matches["neural networks"] == 0.35
    assert "emotions" not in matches
    assert dict(extractor.match_concepts(text, KNOWN, fuzzy=False)) == {
        "consciousness": 0.7
    }
    many = " ".join(["emotions"] * 9)
    assert dict(extractor.match_concepts(many, KNOWN))["emotions"] == 1.0  # capped


def test_stop_words_ignored():
    extractor = ConceptExtractor()
    assert extractor.extract_keywords("The river and the old bridge of stone") == [
        "river",
        "bridge",
        "stone",
    ]
    # a concept never matches through a shared stop word alone
    assert extractor.match_concepts("a story of a walk", ["theory of mind"]) == []
    hits = dict(extractor.match_concepts("my theory holds", ["theory of mind"]))
    assert hits == {"theory of mind": 0.35}


def test_relevance_ordering():
    extractor = ConceptExtractor()
    text = "emotions emotions emotions shape consciousness; a mind wanders"
    matches = extractor.match_concepts(text, KNOWN)
    scores = [score for _, score in matches]
    assert scores == sorted(scores, reverse=True)
    assert matches[0][0] == "emotions"
    picked = extractor.extract_concepts_from_memory(
        text, KNOWN, min_relevance=0.6, max_concepts=1
    )
    assert picked == [("emotions", 0.8)]


def test_satisfies_concept_matcher():
    extractor = create_concept_extractor()
    assert isinstance(extractor, ConceptMatcher)
    assert extractor.match_concepts(text="", known_concepts=[], fuzzy=True) == []


def test_exact_match_needs_word_boundaries():
    extractor = ConceptExtractor()
    assert extractor.match_concepts("the start of the party", ["art"]) == []
    assert extractor.match_concepts("art and more art", ["art"]) == [("art", 0.7)]
    assert extractor.match_concepts("Art, then (art).", ["art"]) == [("art", 0.7)]
    assert extractor.match_concepts(
        "neural networks learn", ["neural networks"], fuzzy=False
    ) == [("neural networks", 0.6)]


def test_fuzzy_match_ignores_punctuation():
    extractor = ConceptExtractor()
    matches = extractor.match_concepts(
        "I enjoy learning, slowly.", ["machine learning"]
    )
    assert matches == [("machine learning", 0.35)]
    assert (
        extractor.match_concepts(
            "I enjoy learning, slowly.", ["machine learning"], fuzzy=False
        )
        == []
    )
