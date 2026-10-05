"""Exact IIT 4.0 Phi through the bridge, the tracker path, normalization, caps."""

import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from mtc.assessment.assessment import (
    INSUFFICIENT_DATA,
    MEASURED,
    ConsciousnessAssessment,
    PhiCalculator,
)
from mtc.consciousness import pyphi_bridge
from mtc.consciousness.phi_tracker import PhiResult, PhiTracker

ROOT = Path(__file__).resolve().parents[2]


def _known_network():
    """PyPhi's noisy self-loop example: asymmetric, Phi > 0 at (0, 0, 0)."""
    pyphi = pytest.importorskip("pyphi")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        network = pyphi.examples.basic_noisy_selfloop_network()
        tpm_little = np.asarray(pyphi.convert.to_2dimensional(network.tpm))
    return pyphi, network, tpm_little, np.asarray(network.cm)


def _direct_sia_phi(pyphi, network, state):
    from pyphi import new_big_phi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subsystem = pyphi.Subsystem(network, state)
        with pyphi.config.override(PROGRESS_BARS=False):
            return max(0.0, float(new_big_phi.sia(subsystem).phi))


@pytest.mark.asyncio
async def test_exact_phi_matches_direct_pyphi_on_known_network():
    pyphi, network, tpm_little, cm_source_by_target = _known_network()
    state = (0, 0, 0)
    expected = _direct_sia_phi(pyphi, network, state)
    assert expected > 0.0

    # MTC's conventions: big-endian rows (the converter is its own inverse)
    # and target-by-source connectivity.
    tpm_big = pyphi_bridge.to_little_endian_rows(tpm_little)
    connectivity = cm_source_by_target.T.astype(float)
    measurement = await PhiCalculator().calculate_phi(
        np.array(state), connectivity, tpm=tpm_big, labels=["A", "B", "C"]
    )
    assert measurement.method == "iit4_sia"
    assert measurement.is_approximate is False
    assert measurement.phi == pytest.approx(expected, abs=1e-9)
    assert measurement.cause_repertoire == {"iit_version": "4.0"}
    assert "partition" in measurement.effect_repertoire


@pytest.mark.asyncio
async def test_convention_error_would_change_phi():
    pyphi, network, tpm_little, cm = _known_network()
    state = (0, 0, 0)
    expected = _direct_sia_phi(pyphi, network, state)
    tpm_big = pyphi_bridge.to_little_endian_rows(tpm_little)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unconverted = pyphi.Network(tpm=tpm_big, cm=cm)
    wrong = _direct_sia_phi(pyphi, unconverted, state)
    assert abs(wrong - expected) > 1e-3


@pytest.mark.asyncio
async def test_no_tpm_means_labelled_proxy():
    state = np.array([1, 0, 1, 1], dtype=np.float64)
    connectivity = np.array(
        [
            [0.0, 0.8, 0.3, 0.1],
            [0.5, 0.0, 0.7, 0.2],
            [0.2, 0.6, 0.0, 0.9],
            [0.4, 0.1, 0.8, 0.0],
        ]
    )
    measurement = await PhiCalculator().calculate_phi(state, connectivity)
    assert measurement.is_approximate is True
    assert measurement.method == "spectral_proxy"
    assert measurement.phi > 0.0


def test_only_bridge_builds_pyphi_network():
    offenders = []
    pattern = re.compile(r"pyphi\.Network\(|pyphi\.Subsystem\(")
    import_pattern = re.compile(r"^\s*(import pyphi|from pyphi)\b", re.M)
    for path in (ROOT / "mtc").rglob("*.py"):
        if path.name == "pyphi_bridge.py":
            continue
        source = path.read_text()
        if pattern.search(source) or import_pattern.search(source):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_importing_mtc_does_not_import_pyphi():
    code = "import sys, mtc; print('pyphi' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "False"


class _FakeTracker:
    N_NODES = 5

    def __init__(self, result, normalized=None):
        self._result = result
        self._normalized = normalized
        self.calls = 0

    async def compute_architectural_phi(self, current_state=None):
        self.calls += 1
        return self._result

    def normalize_phi(self, value):
        return self._normalized


def _exact_result(phi=2.5):
    return PhiResult(
        phi=phi,
        phi_normalized=None,
        distinctions_count=None,
        relations_count=None,
        partition_info="2 parts",
        is_exact=True,
        computation_time_ms=12.0,
        tpm_coverage=0.4,
        transitions_recorded=200,
        warming_up=False,
        error=None,
    )


@pytest.mark.asyncio
async def test_phi_tracker_preferred_when_warm():
    tracker = _FakeTracker(_exact_result(2.5))
    report = await ConsciousnessAssessment().run_full_assessment(
        phi_tracker=tracker, normalize=False
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == MEASURED
    assert result.score == pytest.approx(min(1.0, 2.5 / 5.0))
    assert result.evidence["source"] == "phi_tracker_iit4"
    assert result.evidence["interpretation"] == "module-graph integration proxy"
    phi = report.phi_measurement
    assert phi.method == "iit4_sia"
    assert phi.subsystem_size == 5
    assert phi.is_approximate is False
    assert phi.cause_repertoire["iit_version"] == "4.0"
    assert phi.cause_repertoire["tpm_coverage"] == pytest.approx(0.4)
    assert phi.effect_repertoire["partition"] == "2 parts"

    # The v0.4.0 path ignores the tracker
    calculator = await ConsciousnessAssessment(
        phi_source="calculator"
    ).run_full_assessment(phi_tracker=tracker, normalize=False)
    assert calculator.indicator_results["integrated_information"].status == "absent"


def _neural_states():
    rng = np.random.default_rng(1)
    raw = rng.random((10, 10))
    return {
        "combined_state": rng.random(10),
        "connectivity": (raw + raw.T) / 2.0,
    }


@pytest.mark.asyncio
async def test_tracker_warming_up_falls_back_to_calculator():
    tracker = PhiTracker()
    assert tracker.warming_up
    report = await ConsciousnessAssessment().run_full_assessment(
        phi_tracker=tracker, neural_states=_neural_states(), normalize=False
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == MEASURED
    assert result.evidence["source"] == "calculator_spectral_proxy"
    assert result.evidence["tracker_fallback"]["tracker_warming_up"] is True
    assert report.phi_measurement.method == "spectral_proxy"


@pytest.mark.asyncio
async def test_tracker_warming_up_without_neural_states_is_insufficient():
    report = await ConsciousnessAssessment().run_full_assessment(
        phi_tracker=PhiTracker(), normalize=False
    )
    result = report.indicator_results["integrated_information"]
    assert result.status == INSUFFICIENT_DATA
    assert result.evidence["tracker_warming_up"] is True
    assert report.phi_measurement is None


@pytest.mark.asyncio
async def test_percentile_normalization_option():
    calculator = PhiCalculator(normalization="percentile")
    rng = np.random.default_rng(5)
    connectivity = rng.random((10, 10))
    for _ in range(25):
        await calculator.calculate_phi(rng.random(10), connectivity)
    history = [m.phi for m in calculator.phi_history]
    measurement = await calculator.calculate_phi(rng.random(10), connectivity)
    expected = min(1.0, measurement.phi / np.percentile(history, 95))
    assert measurement.phi_normalized == pytest.approx(expected)

    fixed = PhiCalculator()
    first = await fixed.calculate_phi(rng.random(10), connectivity)
    assert first.phi_normalized == pytest.approx(min(1.0, first.phi / 5.0))

    # For the tracker, percentile means the tracker's own normalize_phi
    tracker = _FakeTracker(_exact_result(2.5), normalized=0.9)
    report = await ConsciousnessAssessment(
        phi_normalization="percentile"
    ).run_full_assessment(phi_tracker=tracker, normalize=False)
    assert report.indicator_results["integrated_information"].score == pytest.approx(
        0.9
    )


@pytest.mark.asyncio
async def test_phi_history_capped_at_100():
    calculator = PhiCalculator()
    rng = np.random.default_rng(6)
    connectivity = rng.random((10, 10))
    for _ in range(105):
        await calculator.calculate_phi(rng.random(10), connectivity)
    assert len(calculator.phi_history) == 100


@pytest.mark.asyncio
async def test_assessment_history_capped_at_50():
    assessment = ConsciousnessAssessment()
    for _ in range(55):
        await assessment.run_full_assessment(normalize=False)
    assert len(assessment.assessment_history) == 50
