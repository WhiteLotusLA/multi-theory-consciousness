"""The novelty filter: near-duplicates, the bounded cache, the express gate and
the honest report when no usable embedder measures anything."""

from typing import Sequence

import numpy as np
import pytest

from mtc.consciousness.continuous.novelty import (
    AcceleratedNoveltyFilter,
    NoveltyFilter,
    NoveltyResult,
)
from mtc.core.embeddings import HashEmbedder

VOCABULARY = ("river", "bridge", "garden", "stone", "lantern", "window", "cloud")


class LookupEmbedder:
    """Each vocabulary word owns one axis (a semantic-flagged test stand-in)."""

    semantic = True

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                if word in VOCABULARY:
                    out[row, VOCABULARY.index(word)] += 1.0
            if not out[row].any():
                out[row, self.dim - 1] = 1.0
            out[row] /= np.linalg.norm(out[row])
        return out


def test_no_embedder_returns_none_and_status_quiet():
    novelty = NoveltyFilter(None)
    assert novelty.find_near_duplicate("river bridge") is None
    assert novelty.express_allowed(None) is False
    status = novelty.status()
    assert status["embedder"] == "none"
    assert status["expression_possible"] is False
    assert status["dim"] is None and status["cached"] == 0


def test_dimension_from_embedder():
    assert NoveltyFilter(LookupEmbedder(dim=24)).dim == 24
    assert NoveltyFilter(HashEmbedder(dim=40)).status()["dim"] == 40

    class NoDim:
        semantic = True

        def encode(self, texts):
            return np.ones((len(texts), 3))

    with pytest.raises(ValueError, match="dim"):
        NoveltyFilter(NoDim())

    class LateWidth(LookupEmbedder):
        """Reports a placeholder width until its model loads."""

        def __init__(self):
            super().__init__(dim=8)
            self.width = 12

        def encode(self, texts):
            out = np.zeros((len(texts), self.width))
            out[:, 0] = 1.0
            return out

    late = LateWidth()
    novelty = NoveltyFilter(late)
    assert novelty.find_near_duplicate("river").is_unique
    assert novelty.dim == 12
    late.width = 16
    with pytest.raises(ValueError, match="dim"):
        novelty.find_near_duplicate("garden")


def test_near_duplicate_detected():
    novelty = NoveltyFilter(LookupEmbedder())
    first = novelty.find_near_duplicate("river bridge")
    assert first == NoveltyResult(is_unique=True, similarity=0.0, matched_text=None)
    again = novelty.find_near_duplicate("the river bridge")
    assert again.is_unique is False
    assert again.similarity == pytest.approx(1.0)
    assert again.matched_text == "river bridge"
    assert novelty.status()["duplicates"] == 1
    assert novelty.cached_texts() == ["river bridge"]


def test_unique_added_to_cache():
    novelty = NoveltyFilter(LookupEmbedder())
    novelty.find_near_duplicate("river")
    result = novelty.find_near_duplicate("river garden")
    assert result.is_unique is True
    assert result.similarity == pytest.approx(1 / np.sqrt(2))
    assert result.novelty == pytest.approx(1 - 1 / np.sqrt(2))
    assert novelty.cached_texts() == ["river", "river garden"]
    assert len(novelty) == 2


def test_cache_bounded():
    novelty = NoveltyFilter(LookupEmbedder(), cache_size=3)
    for word in ("river", "bridge", "garden", "stone", "lantern"):
        assert novelty.find_near_duplicate(word).is_unique
    assert len(novelty) == 3
    assert sorted(novelty.cached_texts()) == ["garden", "lantern", "stone"]
    # the overwritten entry no longer matches
    assert novelty.find_near_duplicate("river").is_unique is True


def test_express_threshold():
    novelty = NoveltyFilter(LookupEmbedder(), express_threshold=0.5)
    assert novelty.express_allowed(NoveltyResult(True, 0.0)) is True
    assert novelty.express_allowed(NoveltyResult(True, 0.5)) is True
    assert novelty.express_allowed(NoveltyResult(True, 0.6)) is False
    assert novelty.express_allowed(NoveltyResult(False, 0.9, "x")) is False
    strict = NoveltyFilter(LookupEmbedder(), express_threshold=0.9)
    assert strict.express_allowed(NoveltyResult(True, 0.2)) is False
    with pytest.raises(ValueError):
        NoveltyFilter(LookupEmbedder(), express_threshold=1.5)


def test_non_semantic_embedder_flagged_in_status():
    lexical = NoveltyFilter(HashEmbedder(dim=64))
    status = lexical.status()
    assert status["embedder"] == "non_semantic"
    assert status["non_semantic_opt_in"] is False
    assert status["expression_possible"] is False
    first = lexical.find_near_duplicate("a quiet thought about rivers")
    assert first.is_unique is True
    assert lexical.express_allowed(first) is False
    # identical text is still caught as a near-duplicate
    assert (
        lexical.find_near_duplicate("a quiet thought about rivers").is_unique is False
    )

    opted = NoveltyFilter(HashEmbedder(dim=64), allow_non_semantic=True)
    status = opted.status()
    assert status["embedder"] == "non_semantic"
    assert status["non_semantic_opt_in"] is True
    assert status["expression_possible"] is True
    assert opted.express_allowed(opted.find_near_duplicate("a new line")) is True


def test_embedder_failure_propagates_and_leaves_cache():
    class Failing(LookupEmbedder):
        def encode(self, texts):
            raise RuntimeError("model not loaded")

    novelty = NoveltyFilter(Failing())
    with pytest.raises(RuntimeError):
        novelty.find_near_duplicate("river")
    assert len(novelty) == 0 and novelty.status()["checked"] == 0


@pytest.mark.gpu
def test_accelerated_filter_matches_numpy():
    pytest.importorskip("torch")
    texts = ["river", "river bridge", "garden stone", "river bridge", "cloud"]
    plain = NoveltyFilter(LookupEmbedder())
    fast = AcceleratedNoveltyFilter(LookupEmbedder())
    for text in texts:
        expected = plain.find_near_duplicate(text)
        actual = fast.find_near_duplicate(text)
        assert actual.is_unique == expected.is_unique
        assert actual.matched_text == expected.matched_text
        assert actual.similarity == pytest.approx(expected.similarity, abs=1e-5)


def test_device_from_switch(monkeypatch):
    pytest.importorskip("torch")
    from mtc.core.config import Settings

    monkeypatch.setenv("MTC_DEVICE", "cpu")
    novelty = AcceleratedNoveltyFilter(LookupEmbedder(), settings=Settings())
    assert novelty.status()["device"] == "cpu"
    assert novelty.status()["backend"] == "torch"
    first = novelty.find_near_duplicate("river bridge")
    second = novelty.find_near_duplicate("river bridge")
    assert first.is_unique is True and second.is_unique is False
    assert second.matched_text == "river bridge"
