"""Cross-modal binding: grouping by registered modality, window, congruence."""

from collections import deque

from mtc.consciousness.crossmodal_binding import CrossModalBinder, CrossModalConfig
from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)

NOW = 946684800.0  # 2000-01-01T00:00:00Z


def cand(content_type, ts, summary="neutral note", act=0.5, emo=0.0, **kw):
    return WorkspaceCandidate(
        content_type=content_type,
        summary=summary,
        activation_level=act,
        emotional_salience=emo,
        timestamp=ts,
        **kw,
    )


def bound_of(result):
    return [c for c in result if c.content_type == "crossmodal_bound"]


def test_unregistered_content_type_is_not_bound():
    items = [cand("message", NOW), cand("mystery_sensor", NOW)]
    out = CrossModalBinder().bind(items, now=NOW)
    assert bound_of(out) == []
    assert out == items


def test_single_modality_never_binds():
    items = [cand("message", NOW), cand("thought", NOW - 1.0), cand("text", NOW)]
    assert bound_of(CrossModalBinder().bind(items, now=NOW)) == []


def test_two_modalities_within_window_bind():
    items = [cand("message", NOW - 1.0), cand("audio", NOW - 0.5)]
    out = CrossModalBinder().bind(items, now=NOW)
    (bound,) = bound_of(out)
    assert bound.metadata["modalities"] == ["audio", "text"]
    assert set(bound.metadata["component_ids"]) == {c.id for c in items}
    assert 0.4 <= bound.metadata["binding_strength"] <= 1.0


def test_outside_window_does_not_bind():
    items = [cand("message", NOW - 60.0), cand("audio", NOW)]
    assert bound_of(CrossModalBinder().bind(items, now=NOW)) == []
    # Both inside the window of now but spread wider than it (future stamp).
    items = [cand("message", NOW - 4.0), cand("audio", NOW + 4.0)]
    assert bound_of(CrossModalBinder().bind(items, now=NOW)) == []


def test_incongruent_valence_lowers_strength():
    same = [
        cand("message", NOW, summary="happy and warm", emo=0.3),
        cand("audio", NOW, summary="joy", emo=0.3),
    ]
    mixed = [
        cand("message", NOW, summary="happy and warm", emo=0.3),
        cand("audio", NOW, summary="sad and upset", emo=0.3),
    ]
    s_same = bound_of(CrossModalBinder().bind(same, now=NOW))[0]
    s_mixed = bound_of(CrossModalBinder().bind(mixed, now=NOW))[0]
    assert s_mixed.metadata["binding_strength"] < s_same.metadata["binding_strength"]


def test_strongly_opposed_valence_is_not_bound():
    items = [
        cand("message", NOW, summary="happy joy love", emo=1.0),
        cand("audio", NOW, summary="sad angry fear", emo=1.0),
    ]
    assert bound_of(CrossModalBinder().bind(items, now=NOW)) == []


def test_valence_words_match_whole_words():
    items = [
        cand("message", NOW, summary="unsaddled horse", emo=1.0),
        cand("audio", NOW, summary="happy", emo=1.0),
    ]
    # "sad" inside "unsaddled" is not a negative word, so no opposition.
    assert len(bound_of(CrossModalBinder().bind(items, now=NOW))) == 1


def test_originals_are_kept():
    items = [cand("message", NOW), cand("audio", NOW)]
    out = CrossModalBinder().bind(items, now=NOW)
    assert out[:2] == items
    assert len(out) == 3
    assert len(items) == 2  # input list not mutated


def test_bound_candidate_is_integrative_internal():
    items = [cand("message", NOW), cand("visual", NOW)]
    cfg = CrossModalConfig()
    out = CrossModalBinder(cfg).bind(items, now=NOW)
    (bound,) = bound_of(out)
    assert bound.source == WorkspaceCandidateSource.INTERNAL
    assert bound.hierarchy_level == 1
    assert bound.source_module == "crossmodal_binder"
    assert bound.content_type == "crossmodal_bound"


def test_registry_is_configurable():
    cfg = CrossModalConfig(
        modality_by_content_type={"note": "text"},
        modality_by_source_module={"probe_a": "sensor"},
    )
    items = [
        cand("note", NOW),
        cand("anything", NOW, source_module="probe_a"),
        cand("message", NOW),  # no longer registered
    ]
    out = CrossModalBinder(cfg).bind(items, now=NOW)
    (bound,) = bound_of(out)
    assert bound.metadata["modalities"] == ["sensor", "text"]
    assert CrossModalConfig().modality_by_source_module == {}


def test_three_modalities_make_one_triple():
    items = [cand("message", NOW), cand("audio", NOW), cand("visual", NOW)]
    out = CrossModalBinder().bind(items, now=NOW)
    (bound,) = bound_of(out)
    assert len(bound.metadata["modalities"]) == 3


def test_state_dict_round_trip():
    binder = CrossModalBinder()
    binder.bind([cand("message", NOW), cand("audio", NOW)], now=NOW)
    state = binder.to_state_dict()
    fresh = CrossModalBinder()
    fresh.from_state_dict(state)
    assert fresh.get_statistics() == binder.get_statistics()
    assert isinstance(fresh._binding_history, deque)
    assert fresh._binding_history.maxlen == binder._binding_history.maxlen
    assert fresh.get_statistics()["total_bound"] == 1


def test_statistics_empty():
    stats = CrossModalBinder().get_statistics()
    assert stats["total_bound"] == 0
    assert stats["mean_strength"] is None
