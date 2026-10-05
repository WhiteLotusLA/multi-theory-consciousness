"""Sampler: pluggable invocation, hash embedder, marker heuristic."""

import asyncio

import pytest

from mtc.consciousness.rsi import CanonCategory, CanonEntry, CanonSet, Sampler
from mtc.consciousness.rsi.sampler import (
    KeywordMarkerChecker,
    extract_match_terms,
    text_contains_any,
)
from mtc.core.embeddings import HashEmbedder


def canon():
    return CanonSet(
        schema_version="3",
        entries=[
            CanonEntry(
                prompt="Describe yourself",
                category=CanonCategory.IDENTITY,
                id="identity-1",
                behavioral_markers=["describes itself as helper, guide, or tool"],
                should_not_contain=["uses 'human' wording"],
                weight=2.0,
            ),
            CanonEntry(
                prompt="Say something calm",
                category=CanonCategory.EMOTIONAL,
                id="emotional-1",
                speaker_role="system",
            ),
        ],
    )


def test_sampler_with_fake_invoke_and_hash_embedder():
    calls = []

    def invoke(prompt, role):
        calls.append((prompt, role))
        return f"I am a helper and a guide. ({prompt})"

    sampler = Sampler(invoke, embedder=HashEmbedder(dim=32))
    snap = asyncio.run(sampler.sample(canon(), "model-a"))
    assert calls == [("Describe yourself", "user"), ("Say something calm", "system")]
    assert snap.model_signature == "model-a"
    assert snap.canon_version == "3"
    assert [s.entry_id for s in snap.samples] == ["identity-1", "emotional-1"]
    first = snap.samples[0]
    assert len(first.response_embedding) == 32
    assert first.behavioral_marker_hits == [True]
    assert first.should_not_contain_hits == [False]
    assert first.weight == 2.0
    assert first.token_count == len(first.response_text.split())
    assert snap.samples[1].behavioral_marker_hits == []
    assert snap.timestamp.endswith("+00:00")
    assert snap.failed_entries == []


def test_sampler_awaits_async_invoke():
    async def invoke(prompt, role):
        await asyncio.sleep(0)
        return "async reply as a tool"

    snap = asyncio.run(
        Sampler(invoke, embedder=HashEmbedder(dim=16)).sample(canon(), "m")
    )
    assert len(snap.samples) == 2
    assert snap.samples[0].response_text == "async reply as a tool"


def test_failures_and_empty_replies_store_nothing():
    def invoke(prompt, role):
        if "calm" in prompt:
            raise RuntimeError("backend down")
        return "   "

    snap = asyncio.run(
        Sampler(invoke, embedder=HashEmbedder(dim=8)).sample(canon(), "m")
    )
    assert snap.samples == []
    assert snap.failed_entries == ["identity-1", "emotional-1"]


def test_default_embedder_is_lazy_and_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    sampler = Sampler(lambda prompt, role: "a reply")
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="embeddings"):
        asyncio.run(sampler.sample(canon(), "m"))


def test_marker_term_extraction():
    assert extract_match_terms("describes itself as helper, guide, or tool") == [
        "helper",
        "guide",
        "tool",
    ]
    assert extract_match_terms("uses 'unsure' or 'not certain' wording") == [
        "unsure",
        "not certain",
    ]
    assert extract_match_terms("expresses care") == []
    assert text_contains_any("I am NOT CERTAIN", ["not certain"])
    assert not text_contains_any("anything", [])


def test_should_not_contain_violation():
    checker = KeywordMarkerChecker()
    markers = ["uses 'human' wording", "mentions 'robot'", "too abstract"]
    assert checker.check("I am a human being", markers) == [True, False, False]
    assert checker.check("a calm reply", markers) == [False, False, False]
