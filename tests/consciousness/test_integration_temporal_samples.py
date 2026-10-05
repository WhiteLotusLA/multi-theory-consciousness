"""The integration keeps a bounded window of assembled states for the temporal
consistency test, in the shape the verifier reads; with fewer than two
samples that test reads insufficient_data."""

import pytest

from mtc.consciousness.consciousness_integration import ConsciousnessIntegration
from mtc.consciousness.verification_framework import ConsciousnessVerifier
from mtc.consciousness.verification_state_assembler import (
    ConsciousnessStateAssembler,
)


class _Orchestrator:
    """The integration only stores its orchestrator at construction."""


def _integration(window=100):
    return ConsciousnessIntegration(
        _Orchestrator(), enable_persistence=False, temporal_window=window
    )


def _sample(values, stamp):
    return {
        "timestamp": stamp,
        "snn_data": {"is_real": True, "synchrony": 0.5, "temporal_patterns": values},
    }


def test_temporal_samples_bounded():
    integration = _integration(window=3)
    for index in range(5):
        integration.append_temporal_sample(_sample([index, index + 1.0], str(index)))
    data = integration.get_temporal_data()
    assert [s["timestamp"] for s in data["states"]] == ["2", "3", "4"]
    assert integration.temporal_samples.maxlen == 3

    # Non-dict samples are ignored, and a nested temporal_data is dropped so
    # samples never grow recursively.
    integration.append_temporal_sample(None)
    integration.append_temporal_sample(["not", "a", "state"])
    integration.append_temporal_sample({"timestamp": "5", "temporal_data": data})
    latest = integration.get_temporal_data()["states"][-1]
    assert latest == {"timestamp": "5"}

    with pytest.raises(ValueError):
        _integration(window=0)
    assert (
        ConsciousnessIntegration(
            _Orchestrator(), enable_persistence=False
        ).temporal_samples.maxlen
        == 100
    )


def test_temporal_data_shape_matches_verifier():
    integration = _integration()
    first = _sample([0.1, 0.5, 0.3, 0.9], "2000-01-01T00:00:00+00:00")
    integration.append_temporal_sample(first)
    integration.append_temporal_sample(
        _sample([0.12, 0.49, 0.33, 0.88], "2000-01-01T00:01:00+00:00")
    )
    # Stored samples are copies: mutating the caller's dict changes nothing.
    first["snn_data"]["temporal_patterns"].append(99.0)

    data = integration.get_temporal_data()
    assert set(data) == {"states", "timestamps"}
    assert data["timestamps"] == [
        "2000-01-01T00:00:00+00:00",
        "2000-01-01T00:01:00+00:00",
    ]
    assert data["states"][0]["snn_data"]["temporal_patterns"] == [0.1, 0.5, 0.3, 0.9]

    state = ConsciousnessStateAssembler.from_neural_data({}, temporal_data=data)
    suite = ConsciousnessVerifier().run_verification_suite(state)
    temporal = suite.results["temporal_consistency"]
    assert temporal.insufficient_data is False
    assert temporal.components["snn_temporal"] > 0.9
    assert temporal.details["num_states"] == 2


def test_no_samples_means_insufficient_temporal_test():
    integration = _integration()
    assert integration.get_temporal_data() == {"states": [], "timestamps": []}
    verifier = ConsciousnessVerifier()

    state = ConsciousnessStateAssembler.from_neural_data(
        {}, temporal_data=integration.get_temporal_data()
    )
    assert "temporal_consistency" in verifier.run_verification_suite(state).insufficient

    integration.append_temporal_sample(_sample([0.1, 0.5, 0.3], "t1"))
    state = ConsciousnessStateAssembler.from_neural_data(
        {}, temporal_data=integration.get_temporal_data()
    )
    result = verifier.run_verification_suite(state).results["temporal_consistency"]
    assert result.insufficient_data is True
    assert result.score is None
