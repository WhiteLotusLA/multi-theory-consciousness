"""The embeddings seam: a protocol, a lazy sentence-transformer wrapper and an
offline deterministic embedder."""

import builtins

import numpy as np
import pytest

from mtc.core.embeddings import Embedder, HashEmbedder, SentenceTransformerEmbedder


def test_hash_embedder_is_deterministic_and_normalized():
    texts = ["alpha beta", "gamma", "alpha beta"]
    a = HashEmbedder().encode(texts)
    b = HashEmbedder().encode(texts)
    assert a.shape == (3, 384)
    assert np.array_equal(a, b)
    assert np.array_equal(a[0], a[2])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_hash_embedder_dim():
    emb = HashEmbedder(dim=64)
    assert emb.dim == 64
    assert emb.encode(["x", "y"]).shape == (2, 64)
    assert emb.encode([]).shape == (0, 64)
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_hash_embedder_empty_text_is_still_unit_norm():
    row = HashEmbedder(dim=32).encode([""])[0]
    assert np.isclose(np.linalg.norm(row), 1.0)


def test_missing_extra_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = SentenceTransformerEmbedder()
    with pytest.raises(ImportError) as info:
        emb.encode(["hello"])
    assert "embeddings" in str(info.value)


def test_sentence_transformer_construction_does_not_import():
    emb = SentenceTransformerEmbedder(model_name="some/model")
    assert emb.model_name == "some/model"
    assert emb.dim == 384


def test_embedders_satisfy_protocol():
    assert isinstance(HashEmbedder(), Embedder)
    assert isinstance(SentenceTransformerEmbedder(), Embedder)
