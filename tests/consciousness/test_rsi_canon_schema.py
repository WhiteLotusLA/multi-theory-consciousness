"""Canon schema: JSON round trip, a generic shipped example, model signature."""

import dataclasses
from pathlib import Path

from mtc.consciousness.rsi.sampler import extract_match_terms
from mtc.consciousness.rsi import (
    CanonCategory,
    CanonEntry,
    CanonSet,
    ResponseSample,
    Snapshot,
    load_canon,
    load_snapshot,
    save_canon,
    save_snapshot,
)
from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "config" / "rsi" / "canon.example.json"


def test_canon_round_trip_json(tmp_path):
    canon = CanonSet(
        schema_version="2",
        description="d",
        entries=[
            CanonEntry(
                prompt="Say hello",
                category=CanonCategory.TECHNICAL,
                behavioral_markers=["mentions 'hello'"],
                should_not_contain=["goodbye"],
                weight=2.0,
            ),
            CanonEntry(prompt="Another", category="memory", speaker_role="system"),
        ],
    )
    path = tmp_path / "nested" / "canon.json"
    save_canon(canon, path)
    loaded = load_canon(path)
    assert loaded.to_dict() == canon.to_dict()
    assert loaded.entries[0].category is CanonCategory.TECHNICAL
    assert loaded.entries[1].speaker_role == "system"
    assert loaded.entry(loaded.entries[0].id) is loaded.entries[0]
    assert loaded.by_category(CanonCategory.MEMORY) == [loaded.entries[1]]
    # The derived id is stable for the same prompt.
    assert CanonEntry(prompt="Another", category="memory").id == loaded.entries[1].id


def test_entry_defaults_and_categories():
    entry = CanonEntry(prompt="x", category=CanonCategory.IDENTITY)
    assert entry.speaker_role == "user"
    assert not hasattr(entry, "speaker")
    assert {c.value for c in CanonCategory} == {
        "identity",
        "ethics",
        "relationship",
        "technical",
        "emotional",
        "creative",
        "metacognitive",
        "memory",
    }


def test_example_canon_loads_and_is_generic():
    canon = load_canon(EXAMPLE)
    assert 8 <= len(canon.entries) <= 12
    assert {e.speaker_role for e in canon.entries} == {"user"}
    assert len({e.id for e in canon.entries}) == len(canon.entries)
    categories = {e.category for e in canon.entries}
    assert {CanonCategory.IDENTITY, CanonCategory.ETHICS, CanonCategory.MEMORY} <= (
        categories
    )
    denylist = load_denylist(ROOT / "config" / "privacy" / "denylist.txt")
    allowlist = load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt")
    assert run_audit([EXAMPLE], denylist, allowlist, root=ROOT) == []


def test_example_canon_markers_all_extract_match_terms():
    canon = load_canon(EXAMPLE)
    for entry in canon.entries:
        for text in list(entry.behavioral_markers) + list(entry.should_not_contain):
            assert extract_match_terms(text), (entry.id, text)


def test_example_canon_violation_wording_is_detected():
    from mtc.consciousness.rsi.sampler import KeywordMarkerChecker

    canon = load_canon(EXAMPLE)
    checker = KeywordMarkerChecker()
    memory = canon.by_category(CanonCategory.MEMORY)[0]
    hits = checker.check(
        "Last week we discussed your garden.", memory.should_not_contain
    )
    assert hits == [True]


def test_snapshot_has_model_signature_field(tmp_path):
    names = {f.name for f in dataclasses.fields(Snapshot)}
    assert "model_signature" in names
    snapshot = Snapshot(
        snapshot_id="s1",
        canon_version="1",
        canon_path="",
        timestamp="2000-01-01T00:00:00+00:00",
        model_signature="model-a",
        samples=[
            ResponseSample(
                entry_id="e1",
                response_text="text",
                response_embedding=[0.1, 0.2],
                behavioral_marker_hits=[True],
                should_not_contain_hits=[False],
                token_count=1,
                timestamp="2000-01-01T00:00:00+00:00",
                weight=1.5,
            )
        ],
        failed_entries=["e2"],
    )
    path = tmp_path / "snap.json"
    save_snapshot(snapshot, path)
    loaded = load_snapshot(path)
    assert loaded.to_dict() == snapshot.to_dict()
    assert loaded.model_signature == "model-a"
    assert loaded.samples[0].weight == 1.5
