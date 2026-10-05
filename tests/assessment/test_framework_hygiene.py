"""Hygiene of the legacy 14-metric measurement framework."""

import asyncio
import re
import sys
from pathlib import Path

import numpy as np

from mtc.assessment.framework import (
    ConsciousnessMeasurementFramework,
    ConsciousnessMetrics,
)

FRAMEWORK_SOURCE = (
    Path(__file__).resolve().parents[2] / "mtc" / "assessment" / "framework.py"
)


def test_measure_consciousness_without_pyphi(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyphi", None)
    framework = ConsciousnessMeasurementFramework(
        snn_neurons=10, lsm_neurons=10, htm_columns=4
    )
    rng = np.random.default_rng(0)
    metrics = asyncio.run(
        framework.measure_consciousness(
            snn_states=rng.random((5, 10)),
            lsm_states=rng.random((5, 10)),
            thoughts=["I think the pattern repeats"],
        )
    )
    assert isinstance(metrics, ConsciousnessMetrics)
    assert np.isfinite(metrics.phi)


def test_framework_source_has_no_pyphi_import():
    source = FRAMEWORK_SOURCE.read_text()
    assert not re.search(r"^\s*(import pyphi|from pyphi)\b", source, re.M)


def test_self_reference_matches_tokens_not_substrings():
    framework = ConsciousnessMeasurementFramework(system_name="Ada")
    no_self = asyncio.run(framework._measure_self_model(["this is fine"], []))
    assert no_self == 0.0
    with_self = asyncio.run(framework._measure_self_model(["I think so"], []))
    assert with_self == 1.0
    possessive = asyncio.run(framework._measure_self_model(["my turn"], []))
    assert possessive == 1.0


def test_identity_check_matches_tokens():
    framework = ConsciousnessMeasurementFramework(system_name="Ada")
    memories = [{"text": "Adam walked home"}, {"text": "Ada read a book"}]
    score = asyncio.run(framework._measure_self_model([], memories))
    assert score == 0.5
    htm = np.ones((3, 4))
    continuity = asyncio.run(framework._measure_temporal_continuity(memories, htm))
    # Half the memories name the system (0.25) plus a stable HTM (0.5).
    assert abs(continuity - 0.75) < 1e-9
