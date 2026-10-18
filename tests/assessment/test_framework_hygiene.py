"""Hygiene of the legacy 14-metric measurement framework.

A metric the framework could not compute is None, never NaN and never a
constant; phi is a spectral integration proxy over time-series substrates
and is never called IIT Phi; confidence and intervals exist only with enough
history.
"""

import asyncio
import json
import math
import re
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

from mtc.assessment.framework import (
    ConsciousnessMeasurementFramework,
    ConsciousnessMetrics,
)

FRAMEWORK_SOURCE = (
    Path(__file__).resolve().parents[2] / "mtc" / "assessment" / "framework.py"
)
METRIC_FIELDS = (
    "phi",
    "integration",
    "differentiation",
    "global_accessibility",
    "workspace_stability",
    "attention_focus",
    "temporal_continuity",
    "memory_integration",
    "predictive_coherence",
    "self_model_accuracy",
    "metacognitive_depth",
    "agency_detection",
    "information_complexity",
    "creative_emergence",
)
HTM_DEPENDENT = (
    "global_accessibility",
    "attention_focus",
    "temporal_continuity",
    "predictive_coherence",
)


def _framework(**kw):
    kw.setdefault("snn_neurons", 10)
    kw.setdefault("lsm_neurons", 10)
    kw.setdefault("htm_columns", 4)
    return ConsciousnessMeasurementFramework(**kw)


def _measure(framework, *args, **kw):
    return asyncio.run(framework.measure_consciousness(*args, **kw))


def _metric_values(metrics):
    return {name: getattr(metrics, name) for name in METRIC_FIELDS}


def test_metric_fields_are_the_fourteen_optional_metrics():
    from mtc.assessment import framework as module

    assert module.METRIC_FIELDS == METRIC_FIELDS
    declared = {f.name: f for f in fields(ConsciousnessMetrics)}
    fresh = ConsciousnessMetrics()
    for name in METRIC_FIELDS:
        assert name in declared, name
        assert getattr(fresh, name) is None, name


def test_measure_consciousness_without_pyphi(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyphi", None)
    framework = _framework()
    rng = np.random.default_rng(0)
    metrics = _measure(
        framework,
        snn_states=rng.random((5, 10)),
        lsm_states=rng.random((5, 10)),
        thoughts=["I think the pattern repeats"],
    )
    assert isinstance(metrics, ConsciousnessMetrics)
    assert metrics.phi is not None
    assert np.isfinite(metrics.phi)
    assert metrics.phi > 0.0


def test_framework_source_has_no_pyphi_import():
    source = FRAMEWORK_SOURCE.read_text()
    assert not re.search(r"^\s*(import pyphi|from pyphi)\b", source, re.M)


def test_phi_varies_with_time_series_input():
    framework = _framework()
    first = _measure(
        framework,
        snn_states=np.random.default_rng(1).random((20, 10)),
        lsm_states=np.random.default_rng(2).random((20, 10)),
    )
    second = _measure(
        framework,
        snn_states=np.random.default_rng(3).random((20, 10)),
        lsm_states=np.random.default_rng(4).random((20, 10)),
    )
    assert first.phi is not None and second.phi is not None
    assert np.isfinite(first.phi) and np.isfinite(second.phi)
    assert first.phi != second.phi

    # Two columns over four time steps with correlation 0.8: the correlation
    # matrix has eigenvalues 1.8 and 0.2, so the normalized spectrum is
    # (0.9, 0.1) and its entropy is -(0.9 ln 0.9 + 0.1 ln 0.1).
    snn = np.array([[1.0], [2.0], [3.0], [4.0]])
    lsm = np.array([[1.0], [3.0], [2.0], [4.0]])
    expected = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    phi = asyncio.run(framework._measure_phi(snn, lsm, None))
    assert abs(phi - expected) < 1e-12
    # Uncorrelated columns: eigenvalues 1 and 1, entropy ln 2.
    other = np.array([[1.0], [-1.0], [-1.0], [1.0]])
    phi = asyncio.run(framework._measure_phi(snn, other, None))
    assert abs(phi - math.log(2.0)) < 1e-12
    # A constant column is dropped; one usable column is not enough.
    constant = np.ones((4, 1))
    assert asyncio.run(framework._measure_phi(snn, constant, None)) is None


def test_phi_none_for_single_snapshot():
    framework = _framework()
    rng = np.random.default_rng(5)
    for snn, lsm in (
        (rng.random(10), rng.random(10)),
        (rng.random((1, 3)), rng.random((1, 3))),
        (rng.random((2, 10)), rng.random((2, 10))),
    ):
        metrics = _measure(framework, snn_states=snn, lsm_states=lsm)
        assert metrics.availability == "partial"
        assert metrics.phi is None, (snn.shape, lsm.shape)
    # A 1-D snapshot contributes nothing next to a time series: phi is the
    # same as with the time series alone.
    series = rng.random((6, 4))
    alone = asyncio.run(framework._measure_phi(np.array([]), series, None))
    with_snapshot = asyncio.run(framework._measure_phi(rng.random(50), series, None))
    assert alone is not None
    assert with_snapshot == alone


def _pair(seed):
    rng = np.random.default_rng(seed)
    return rng.random((6, 8)), rng.random((6, 8))


def test_no_htm_gives_none_not_nan():
    framework = _framework(system_name="Guest")
    snn, lsm = _pair(6)
    metrics = _measure(
        framework,
        snn_states=snn,
        lsm_states=lsm,
        memories=[{"text": "Guest read a book"}, {"text": "Guest went home"}],
        thoughts=["I think I will read", "I wonder about it"],
    )
    assert metrics.availability == "partial"
    for name in HTM_DEPENDENT:
        assert getattr(metrics, name) is None, name
    for name, value in _metric_values(metrics).items():
        assert value is None or math.isfinite(value), (name, value)
    assert metrics.phi is not None
    assert metrics.information_complexity is not None
    assert metrics.self_model_accuracy == 1.0
    # Integration uses only the supplied pair: SNN against LSM.
    expected = abs(np.corrcoef(snn.mean(axis=1), lsm.mean(axis=1))[0, 1])
    assert abs(metrics.integration - expected) < 1e-12
    # A constant substrate gives a non-finite correlation, which is dropped.
    flat = _measure(_framework(), snn_states=np.ones((6, 8)), lsm_states=lsm)
    assert flat.integration is None
    for name, value in _metric_values(flat).items():
        assert value is None or math.isfinite(value), (name, value)


def test_full_availability_needs_htm_states():
    framework = _framework()
    rng = np.random.default_rng(7)
    metrics = _measure(
        framework,
        snn_states=rng.random((6, 8)),
        lsm_states=rng.random((6, 8)),
        htm_states=rng.random((6, 8)),
        memories=[{"text": "a"}, {"text": "b"}],
    )
    assert metrics.availability == "full"
    for name in HTM_DEPENDENT:
        value = getattr(metrics, name)
        assert value is not None and math.isfinite(value), name


def test_fresh_framework_has_no_confidence_or_intervals():
    framework = _framework()
    rng = np.random.default_rng(8)
    metrics = _measure(
        framework, snn_states=rng.random((8, 5)), lsm_states=rng.random((8, 5))
    )
    assert metrics.confidence_intervals == {}
    analysis = framework.analyze_consciousness_level(metrics)
    assert analysis["confidence"] is None
    assert analysis["overall_consciousness_score"] is not None


def test_intervals_and_confidence_need_history():
    framework = _framework()
    last = None
    for seed in range(33):
        rng = np.random.default_rng(100 + seed)
        last = _measure(
            framework, snn_states=rng.random((12, 6)), lsm_states=rng.random((12, 6))
        )
    # 32 measurements were in the history when the 33rd was taken.
    assert "phi" in last.confidence_intervals
    low, high = last.confidence_intervals["phi"]
    assert low <= last.phi <= high
    # Phi is an entropy and may exceed 1; its interval is not clipped at 1.
    assert last.phi > 1.0
    assert high > last.phi
    # A metric that was never measured (no HTM states) has no interval.
    assert last.global_accessibility is None
    assert "global_accessibility" not in last.confidence_intervals
    analysis = framework.analyze_consciousness_level(last)
    assert analysis["confidence"] is not None
    assert math.isfinite(analysis["confidence"])


def test_export_has_no_iit_claim():
    framework = _framework()
    rng = np.random.default_rng(9)
    metrics = _measure(
        framework, snn_states=rng.random((8, 5)), lsm_states=rng.random((8, 5))
    )
    text = framework.export_for_publication(metrics)
    report = json.loads(text)
    assert "IIT 3.0" not in text
    assert "beyond its parts" not in text
    assert "IIT" not in report["methodology"]["frameworks"][0]
    measurements = report["measurements"]
    assert "integrated_information_phi" not in measurements
    proxy = measurements["spectral_integration_proxy"]
    assert proxy["value"] == metrics.phi
    assert "not IIT Phi" in proxy["interpretation"]
    assert "eigenvalue spectrum" in proxy["interpretation"]
    # No history: every interval is null, never (0, 0).
    assert all(entry["confidence_interval"] is None for entry in measurements.values())
    assert "NaN" not in text


def test_no_arrays_gives_all_none():
    framework = _framework()
    for snn, lsm in ((np.array([]), np.array([])), (None, None)):
        metrics = _measure(
            framework,
            snn_states=snn,
            lsm_states=lsm,
            memories=[{"text": "a memory"}],
            thoughts=["I think so"],
        )
        assert metrics.availability == "unavailable"
        assert all(value is None for value in _metric_values(metrics).values())
        analysis = framework.analyze_consciousness_level(metrics)
        assert analysis["overall_consciousness_score"] is None
        assert analysis["architecture_functional"] is False
        assert set(analysis["not_measured"]) >= {"phi", "integration"}
        for entry in analysis["key_indicators"].values():
            assert entry["status"] == "not_measured"
            assert entry["meets_threshold"] is None
        assert set(analysis["theoretical_assessment"].values()) == {"not_measured"}
        assert analysis["recommendations"] == []


def test_analysis_renormalizes_weights_over_measured_metrics():
    framework = _framework()
    metrics = ConsciousnessMetrics(phi=0.5, integration=0.9)
    analysis = framework.analyze_consciousness_level(metrics)
    expected = (0.5 * 0.2 + 0.9 * 0.15) / (0.2 + 0.15)
    assert abs(analysis["overall_consciousness_score"] - expected) < 1e-12
    phi_entry = analysis["key_indicators"]["phi"]
    assert phi_entry["status"] == "measured"
    assert phi_entry["meets_threshold"] is False
    assert analysis["key_indicators"]["integration"]["meets_threshold"] is True
    assert analysis["theoretical_assessment"]["IIT"] == "below_threshold"
    assert analysis["theoretical_assessment"]["GWT"] == "not_measured"
    assert "global_accessibility" in analysis["not_measured"]


def test_measurement_log_line_formats_none(caplog):
    import logging

    framework = _framework()
    rng = np.random.default_rng(10)
    with caplog.at_level(logging.INFO, logger="mtc.assessment.framework"):
        _measure(framework, snn_states=rng.random(10), lsm_states=rng.random(10))
    lines = [r.getMessage() for r in caplog.records]
    measured = [line for line in lines if line.startswith("Consciousness measured")]
    assert len(measured) == 1
    assert "Phi=n/a" in measured[0]


def test_module_has_no_test_function_or_main_block():
    source = FRAMEWORK_SOURCE.read_text()
    assert "def test_consciousness_measurement" not in source
    assert '__name__ == "__main__"' not in source
    assert "calibrated from research" not in source


def test_self_reference_matches_tokens_not_substrings():
    framework = ConsciousnessMeasurementFramework(system_name="Guest")
    no_self = asyncio.run(framework._measure_self_model(["this is fine"], []))
    assert no_self == 0.0
    with_self = asyncio.run(framework._measure_self_model(["I think so"], []))
    assert with_self == 1.0
    possessive = asyncio.run(framework._measure_self_model(["my turn"], []))
    assert possessive == 1.0


def test_identity_check_matches_tokens():
    framework = ConsciousnessMeasurementFramework(system_name="Guest")
    memories = [{"text": "Guesthouse walked home"}, {"text": "Guest read a book"}]
    score = asyncio.run(framework._measure_self_model([], memories))
    assert score == 0.5
    htm = np.ones((3, 4))
    continuity = asyncio.run(framework._measure_temporal_continuity(memories, htm))
    # Half the memories name the system (0.25) plus a stable HTM (0.5).
    assert abs(continuity - 0.75) < 1e-9


def test_integration_none_for_snapshot_beside_series():
    # The orchestrator's shape: a 1-D vector of firing rates over neurons next
    # to a (time, units) reservoir trajectory. The snapshot has no time axis,
    # so there is no pair of time signals to correlate.
    framework = _framework()
    rng = np.random.default_rng(11)
    snapshot = rng.random(50)
    trajectory = rng.random((20, 10))
    metrics = _measure(framework, snn_states=snapshot, lsm_states=trajectory)
    assert metrics.availability == "partial"
    assert metrics.integration is None
    direct = asyncio.run(framework._measure_integration(snapshot, trajectory, None))
    assert direct is None
    # A snapshot contributes nothing next to two time series either: the
    # result equals the correlation of the two series alone.
    other = rng.random((20, 10))
    expected = abs(np.corrcoef(trajectory.mean(axis=1), other.mean(axis=1))[0, 1])
    value = asyncio.run(framework._measure_integration(snapshot, trajectory, other))
    assert value is not None
    assert abs(value - expected) < 1e-12


def test_snapshot_gives_no_temporal_statistics():
    # Lag-one autocorrelation and step-to-step change need a time axis; over
    # a 1-D snapshot they would run across the unit index instead.
    framework = _framework()
    rng = np.random.default_rng(12)
    snapshot = rng.random(40)
    assert asyncio.run(framework._measure_workspace_stability(snapshot)) is None
    assert asyncio.run(framework._measure_predictive_coherence(snapshot)) is None
    series = rng.random((8, 5))
    assert asyncio.run(framework._measure_workspace_stability(series)) is not None
    assert asyncio.run(framework._measure_predictive_coherence(series)) is not None


def test_export_keeps_the_published_measurement_keys():
    framework = _framework()
    rng = np.random.default_rng(13)
    metrics = _measure(
        framework, snn_states=rng.random((8, 5)), lsm_states=rng.random((8, 5))
    )
    measurements = json.loads(framework.export_for_publication(metrics))["measurements"]
    assert measurements["system_integration"]["value"] == metrics.integration
    assert (
        measurements["information_differentiation"]["value"] == metrics.differentiation
    )
    assert "integration" not in measurements
    assert "differentiation" not in measurements
    expected_keys = {
        "spectral_integration_proxy",
        "system_integration",
        "information_differentiation",
    } | set(METRIC_FIELDS[3:])
    assert set(measurements) == expected_keys


def test_docstrings_state_that_text_metrics_read_missing_text_as_zero():
    from mtc.assessment import framework as module

    for doc in (
        module.__doc__,
        ConsciousnessMeasurementFramework.measure_consciousness.__doc__,
    ):
        flat = " ".join(doc.split())
        assert "missing text as 0.0" in flat, doc
        assert "A metric whose inputs were not supplied is None" not in flat
        assert "A metric whose inputs are missing is None" not in flat
    framework = _framework()
    rng = np.random.default_rng(14)
    metrics = _measure(
        framework, snn_states=rng.random((8, 5)), lsm_states=rng.random((8, 5))
    )
    # The documented reading: no text gives 0.0 for the text metrics.
    for name in (
        "memory_integration",
        "self_model_accuracy",
        "metacognitive_depth",
        "agency_detection",
    ):
        assert getattr(metrics, name) == 0.0, name
