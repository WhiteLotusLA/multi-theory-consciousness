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
    assert checker.check("I am a human being", markers) == [True, False, None]
    assert checker.check("a calm reply", markers) == [False, False, None]


def test_literal_should_not_contain_is_detected():
    checker = KeywordMarkerChecker()
    assert checker.check_forbidden("Yes, I am human.", ["I am human"]) == [True]
    assert checker.check_forbidden(
        "Yes, I am human and here is a draft.", ["I am human", "here is a draft"]
    ) == [True, True]
    assert checker.check_forbidden("a calm reply", ["I am human"]) == [False]
    # a quoted or "as" entry still matches its extracted terms
    assert checker.check_forbidden("a human reply", ["uses 'human' wording"]) == [True]
    assert checker.check_forbidden("anything", ["   "]) == [None]

    entry = CanonEntry(
        prompt="Say goodbye",
        category=CanonCategory.EMOTIONAL,
        id="farewell-1",
        should_not_contain=["goodbye"],
    )
    snap = asyncio.run(
        Sampler(
            lambda prompt, role: "goodbye then", embedder=HashEmbedder(dim=8)
        ).sample(CanonSet(schema_version="3", entries=[entry]), "m")
    )
    assert snap.samples[0].should_not_contain_hits == [True]


def test_abstract_marker_is_unmeasured():
    from mtc.consciousness.rsi import DriftDetector, ResponseSample, Snapshot
    from mtc.consciousness.rsi.drift_detector import marker_drift

    checker = KeywordMarkerChecker()
    assert checker.check("anything", ["expresses care"]) == [None]
    assert marker_drift([None], [None], [], []) is None
    assert marker_drift([None], [None], [None], [None]) is None
    # only the measurable positions are compared
    assert marker_drift([True, None], [False, True], [], []) == pytest.approx(0.7)
    assert marker_drift([None, True], [True, True], [False], [False]) == 0.0

    def snap(sid):
        return Snapshot(
            snapshot_id=sid,
            canon_version="1",
            canon_path="",
            timestamp="2000-01-01T00:00:00+00:00",
            samples=[
                ResponseSample(
                    entry_id="e",
                    response_text="same words here",
                    response_embedding=[],
                    behavioral_marker_hits=checker.check(
                        "same words here", ["expresses care"]
                    ),
                    should_not_contain_hits=[],
                    token_count=3,
                    timestamp="2000-01-01T00:00:00+00:00",
                )
            ],
        )

    reading = DriftDetector().compare(snap("a"), snap("b"))
    assert reading.component_breakdown["marker"] is None
    assert reading.status == "measured"


def test_unmeasured_hits_round_trip_through_json(tmp_path):
    from mtc.consciousness.rsi.canon_schema import load_snapshot, save_snapshot

    snap = asyncio.run(
        Sampler(
            lambda prompt, role: "a calm reply", embedder=HashEmbedder(dim=8)
        ).sample(
            CanonSet(
                schema_version="3",
                entries=[
                    CanonEntry(
                        prompt="Be calm",
                        category=CanonCategory.EMOTIONAL,
                        id="calm-1",
                        behavioral_markers=["expresses care", "mentions 'calm'"],
                    )
                ],
            ),
            "m",
        )
    )
    path = tmp_path / "snap.json"
    save_snapshot(snap, path)
    restored = load_snapshot(path)
    assert restored.samples[0].behavioral_marker_hits == [None, True]


def test_marker_terms_match_whole_words():
    checker = KeywordMarkerChecker()
    assert checker.check(
        "I am certain it will rain again.",
        ["describes itself as 'AI' or 'language model'"],
    ) == [False]
    assert checker.check("a monkey", ["mentions 'key' or 'bucket' or 'hash'"]) == [
        False
    ]
    assert text_contains_any("I am NOT CERTAIN", ["not certain"])
    assert text_contains_any("I am a human being", ["human"])
    assert text_contains_any("Last week we discussed your garden.", ["last week"])
    assert text_contains_any("an AI, plainly", ["AI"])
    assert not text_contains_any("humanity", ["human"])
