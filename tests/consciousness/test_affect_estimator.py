"""Affect estimator: anchor-set estimation behind an injectable embedder."""

import builtins

import numpy as np
import pytest

from mtc.consciousness import affect_estimator as ae
from mtc.consciousness.affect_estimator import (
    ANCHORS_VERSION,
    AROUSAL_HIGH,
    AROUSAL_LOW,
    VALENCE_NEGATIVE,
    VALENCE_POSITIVE,
    AffectEstimator,
)


class LookupEmbedder:
    """Maps anchor sentences and a few keywords to known directions."""

    dim = 4
    model_name = "lookup-test"

    def __init__(self):
        self.table = {}
        for text in VALENCE_POSITIVE:
            self.table[text] = (1.0, 0.0, 0.0, 0.0)
        for text in VALENCE_NEGATIVE:
            self.table[text] = (-1.0, 0.0, 0.0, 0.0)
        for text in AROUSAL_HIGH:
            self.table[text] = (0.0, 1.0, 0.0, 0.0)
        for text in AROUSAL_LOW:
            self.table[text] = (0.0, -1.0, 0.0, 0.0)

    def encode(self, texts):
        rows = []
        for text in texts:
            if text in self.table:
                rows.append(self.table[text])
            elif "gladly" in text:
                rows.append((1.0, 0.0, 0.0, 0.0))
            elif "frantic" in text:
                rows.append((0.0, 1.0, 0.0, 0.0))
            elif "orthogonal" in text:
                rows.append((0.0, 0.0, 1.0, 0.0))
            else:
                rows.append((0.0, 0.0, 0.0, 1.0))
        arr = np.asarray(rows, dtype=float)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_empty_text_returns_none():
    est = AffectEstimator(embedder=LookupEmbedder())
    assert est.estimate("") is None
    assert est.estimate("   \n") is None


def test_positive_anchor_text_scores_positive_valence():
    est = AffectEstimator(embedder=LookupEmbedder())
    out = est.estimate("I gladly accept")
    assert out.valence > 0.9
    assert abs(out.arousal) < 1e-9
    assert -1.0 <= out.valence <= 1.0
    neg = est.estimate(VALENCE_NEGATIVE[0])
    assert neg.valence < -0.9


def test_high_arousal_text_scores_high_arousal():
    est = AffectEstimator(embedder=LookupEmbedder())
    out = est.estimate("a frantic rush")
    assert out.arousal > 0.9
    assert abs(out.valence) < 1e-9
    assert est.estimate(AROUSAL_LOW[0]).arousal < -0.9


def test_unrelated_text_scores_near_zero():
    est = AffectEstimator(embedder=LookupEmbedder())
    out = est.estimate("an orthogonal remark")
    assert abs(out.valence) < 1e-9 and abs(out.arousal) < 1e-9


def test_annotation_prior_blends_at_weight():
    plain = AffectEstimator(embedder=LookupEmbedder())
    base = plain.estimate("an orthogonal remark", source_kind="received")
    seen = []

    def prior(label):
        seen.append(label)
        return (0.5, -0.5) if label == "warm" else None

    est = AffectEstimator(
        embedder=LookupEmbedder(), annotation_prior=prior, prior_weight=0.3
    )
    out = est.estimate(
        "an orthogonal remark", source_kind="received", annotation=" Warm "
    )
    assert seen == ["warm"]
    assert out.valence == pytest.approx(0.7 * base.valence + 0.3 * 0.5)
    assert out.arousal == pytest.approx(0.7 * base.arousal + 0.3 * -0.5)
    assert "annotation:warm" in out.computed_from
    # Unknown annotation, other source kinds, or no hook: no blending.
    unknown = est.estimate(
        "an orthogonal remark", source_kind="received", annotation="x"
    )
    assert unknown.valence == pytest.approx(base.valence)
    other = est.estimate(
        "an orthogonal remark", source_kind="response", annotation="warm"
    )
    assert other.valence == pytest.approx(base.valence)


def test_missing_embeddings_extra_returns_none_and_warns_once(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    est = AffectEstimator()
    with caplog.at_level("WARNING"):
        assert est.estimate("hello there") is None
        assert est.estimate("hello again") is None
    messages = [r for r in caplog.records if "embeddings" in r.getMessage()]
    assert len(messages) == 1


def test_embedding_failure_returns_none_and_warns_once(caplog):
    class Broken:
        dim = 4

        def encode(self, texts):
            raise RuntimeError("model missing")

    est = AffectEstimator(embedder=Broken())
    with caplog.at_level("WARNING"):
        assert est.estimate("hello") is None
        assert est.estimate("hello") is None
    assert len(caplog.records) == 1


def test_provenance_names_anchor_version():
    out = AffectEstimator(embedder=LookupEmbedder()).estimate("I gladly accept")
    assert f"anchors:{ANCHORS_VERSION}" in out.computed_from
    assert "model:lookup-test" in out.computed_from
    assert "text:response" in out.computed_from
    assert ae.ANCHORS_VERSION == "v1"


def test_anchor_text_has_no_dash_characters():
    for text in VALENCE_POSITIVE + VALENCE_NEGATIVE + AROUSAL_HIGH + AROUSAL_LOW:
        assert chr(0x2014) not in text and chr(0x2013) not in text


def test_confidence_grows_with_margin_and_length():
    est = AffectEstimator(embedder=LookupEmbedder())
    short = est.estimate("orthogonal")
    strong = est.estimate("gladly " * 30)
    assert strong.confidence > short.confidence
    assert 0.0 <= short.confidence <= 1.0
