"""The HTM binding oscillators come from a live, inference-only pass: learning
state is untouched, metrics the HTM does not compute stay unset, and a failing
or missing HTM never raises."""

import copy

import numpy as np
import pytest

from mtc.core.config import Settings
from mtc.neural.htm.production_htm import HTMConfig, ProductionHTM
from mtc.neural.orchestrator import NeuralOrchestrator

MEASURED = {"anomaly_score", "burst_fraction", "prediction_accuracy", "active_columns"}


def _small_htm() -> ProductionHTM:
    np.random.seed(7)
    config = HTMConfig(
        num_columns=64,
        cells_per_column=4,
        input_bits=64,
        potential_radius=16,
        activation_threshold=2,
    )
    return ProductionHTM(config)


def _bare() -> NeuralOrchestrator:
    return NeuralOrchestrator.__new__(NeuralOrchestrator)


def _pattern(seed: int, size: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(size) > 0.6).astype(float)


def test_live_htm_metrics_use_inference_only():
    htm = _small_htm()
    for seed in range(3):
        htm.process(_pattern(seed), learn=True)

    permanences = htm.spatial_pooler.permanences.copy()
    boost = htm.spatial_pooler.boost_factors.copy()
    segments = copy.deepcopy(htm.temporal_memory.segments)

    out = _bare()._live_htm_binding_metrics(htm, _pattern(11))

    assert np.array_equal(htm.spatial_pooler.permanences, permanences)
    assert np.array_equal(htm.spatial_pooler.boost_factors, boost)
    assert htm.temporal_memory.segments == segments
    assert set(out) == MEASURED
    assert 0.0 <= out["anomaly_score"] <= 1.0
    assert out["burst_fraction"] == out["anomaly_score"]
    assert out["prediction_accuracy"] == 1.0 - out["anomaly_score"]
    assert 0.0 <= out["active_columns"] <= 1.0


def test_live_htm_metrics_leave_unknowns_unset():
    out = _bare()._live_htm_binding_metrics(_small_htm(), _pattern(3))
    assert set(out) <= MEASURED
    for name in (
        "semantic_density",
        "overlap_stability",
        "consolidation",
        "sequence_strength",
    ):
        assert name not in out


class _RaisingHTM:
    input_size = 8

    def process(self, pattern, learn=True):
        raise RuntimeError("substrate failure")


def _light(**kwargs) -> NeuralOrchestrator:
    return NeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False, **kwargs
    )


def test_live_htm_metrics_never_raise():
    bare = _light(settings=Settings(_env_file=None, DEBUG=False))
    assert bare._live_htm_binding_metrics(_RaisingHTM(), np.ones(8)) == {}
    assert bare._live_htm_binding_metrics(None, np.ones(8)) == {}
    assert bare._live_htm_binding_metrics(_small_htm(), None) == {}
    assert bare._live_htm_binding_metrics(_small_htm(), np.array([])) == {}
    assert bare._live_htm_binding_metrics(object(), np.ones(8)) == {}
    assert bare._live_htm_binding_metrics(_small_htm(), "not a pattern") == {}

    # Only the failed HTM pass is a failure; unusable input is not
    assert bare.step_failures == {"live_htm_binding": 1}
    assert bare.get_system_metrics()["failures"] == {"live_htm_binding": 1}


def test_live_htm_failure_raises_under_debug():
    debug = _light(settings=Settings(_env_file=None, DEBUG=True))
    with pytest.raises(RuntimeError, match="substrate failure"):
        debug._live_htm_binding_metrics(_RaisingHTM(), np.ones(8))
    assert debug.step_failures == {"live_htm_binding": 1}
