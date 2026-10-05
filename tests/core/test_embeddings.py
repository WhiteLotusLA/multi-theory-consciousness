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


# ---------------------------------------------------------------------------
# Status, device, dimension check and failure type
# ---------------------------------------------------------------------------

import logging  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

from mtc.core import embeddings as embeddings_module  # noqa: E402
from mtc.core.config import Settings  # noqa: E402
from mtc.core.embeddings import EmbeddingUnavailableError  # noqa: E402


class _FakeModel:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = np.zeros((len(texts), self._dim))
        out[:, 0] = 1.0
        return out


def _install_fake_st(monkeypatch, dim=384, raises=None):
    """A fake sentence_transformers module that records its constructor calls."""
    calls = []

    class SentenceTransformer(_FakeModel):
        def __init__(self, name, device=None):
            if raises is not None:
                raise raises
            calls.append({"name": name, "device": device})
            super().__init__(dim)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    import os

    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    def apply(**values):
        for key, value in values.items():
            monkeypatch.setenv(f"MTC_{key}", str(value))
        fresh = Settings()
        monkeypatch.setattr(embeddings_module, "get_settings", lambda: fresh)
        from mtc.core import device as device_module

        monkeypatch.setattr(device_module, "get_settings", lambda: fresh)
        return fresh

    return apply


def test_model_name_defaults_to_setting_at_load(monkeypatch, settings_env):
    settings_env(EMBEDDING_MODEL="org/other-model")
    calls = _install_fake_st(monkeypatch)
    emb = SentenceTransformerEmbedder()
    emb.encode(["x"])
    assert calls[0]["name"] == "org/other-model"
    assert emb.status()["model_name"] == "org/other-model"
    explicit = SentenceTransformerEmbedder(model_name="explicit/name")
    explicit.encode(["x"])
    assert calls[1]["name"] == "explicit/name"


def test_device_follows_setting(monkeypatch, settings_env):
    settings_env(DEVICE="cpu")
    calls = _install_fake_st(monkeypatch)
    emb = SentenceTransformerEmbedder()
    emb.encode(["x"])
    assert calls[0]["device"] == "cpu"
    assert emb.status()["device"] == "cpu"
    other = SentenceTransformerEmbedder(device="cpu")
    other.encode(["x"])
    assert calls[1]["device"] == "cpu"


def test_unavailable_device_is_a_typed_failure(monkeypatch, settings_env):
    settings_env(DEVICE="mps")
    monkeypatch.setitem(sys.modules, "torch", None)
    _install_fake_st(monkeypatch)
    emb = SentenceTransformerEmbedder()
    with pytest.raises(EmbeddingUnavailableError) as info:
        emb.encode(["x"])
    assert "MTC_DEVICE" in str(info.value)
    assert emb.status()["available"] is False


def test_dimension_mismatch_raises_naming_both_keys(monkeypatch, settings_env):
    settings_env()
    _install_fake_st(monkeypatch, dim=768)
    emb = SentenceTransformerEmbedder()
    with pytest.raises(EmbeddingUnavailableError) as info:
        emb.encode(["x"])
    message = str(info.value)
    assert "EMBEDDING_MODEL" in message and "VECTOR_DIMENSIONS" in message
    assert "768" in message and "384" in message
    status = emb.status()
    assert status["loaded"] is False
    assert "768" in status["last_error"]
    # an explicit expected width wins over the setting
    ok = SentenceTransformerEmbedder(expected_dim=768)
    assert ok.encode(["x"]).shape == (1, 768)


def test_load_failure_raises_typed_error_and_records_status(
    monkeypatch, settings_env, caplog
):
    settings_env()
    cause = OSError("weights missing")
    _install_fake_st(monkeypatch, raises=cause)
    emb = SentenceTransformerEmbedder(model_name="some/model")
    with caplog.at_level(logging.ERROR, logger="mtc.core.embeddings"):
        with pytest.raises(EmbeddingUnavailableError) as info:
            emb.encode(["x"])
    assert info.value.__cause__ is cause
    status = emb.status()
    assert status["loaded"] is False
    assert status["available"] is False
    assert status["load_attempts"] == 1
    assert "weights missing" in status["last_error"]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "pre-download" in errors[0].getMessage()
    assert emb.semantic is True


def test_reload_not_retried_before_interval(monkeypatch, settings_env):
    settings_env()
    clock = {"now": 1000.0}
    _install_fake_st(monkeypatch, raises=OSError("down"))
    emb = SentenceTransformerEmbedder(retry_after_s=300.0)
    emb._clock = lambda: clock["now"]
    with pytest.raises(EmbeddingUnavailableError):
        emb.encode(["x"])
    clock["now"] += 100.0
    with pytest.raises(EmbeddingUnavailableError) as info:
        emb.encode(["x"])
    assert "retry" in str(info.value)
    assert emb.status()["load_attempts"] == 1
    clock["now"] += 250.0
    calls = _install_fake_st(monkeypatch)
    assert emb.encode(["x"]).shape == (1, 384)
    assert emb.status()["load_attempts"] == 2
    assert emb.status()["loaded"] is True
    assert emb.status()["last_error"] is None
    assert len(calls) == 1


def test_missing_package_is_typed_and_an_import_error(
    monkeypatch, settings_env, caplog
):
    settings_env()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = SentenceTransformerEmbedder()
    with caplog.at_level(logging.WARNING, logger="mtc.core.embeddings"):
        with pytest.raises(EmbeddingUnavailableError) as info:
            emb.encode(["x"])
    assert isinstance(info.value, ImportError)
    assert "embeddings" in str(info.value)
    assert emb.status()["available"] is False
    assert [r for r in caplog.records if r.name == "mtc.core.embeddings"] == []


def test_hash_embedder_is_marked_non_semantic():
    assert HashEmbedder.semantic is False
    assert HashEmbedder().semantic is False
    assert SentenceTransformerEmbedder.semantic is True
