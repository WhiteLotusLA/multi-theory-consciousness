"""Tests for Causal Emergence Analyzer (Hoel EI)."""

import asyncio
import json
import logging
import re
from pathlib import Path

import numpy as np
from mtc.assessment.causal_emergence import (
    CausalEmergenceAnalyzer,
    CausalEmergenceResult,
    compute_ei,
    node_tpm_to_state_tpm,
    shannon_entropy,
    generate_bell_partitions,
    build_macro_tpm,
    partition_to_label,
)


class TestEIMath:
    """Test core Effective Information computation."""

    def test_shannon_entropy_uniform(self):
        """Uniform distribution over n states has entropy log2(n)."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(shannon_entropy(p) - 2.0) < 1e-10

    def test_shannon_entropy_deterministic(self):
        """Deterministic distribution has entropy 0."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(shannon_entropy(p) - 0.0) < 1e-10

    def test_ei_identity_tpm(self):
        """Identity TPM (each state maps to itself) has maximum determinism."""
        tpm = np.eye(4)  # 4-state identity
        ei = compute_ei(tpm)
        # Determinism = log2(4) - 0 = 2.0 (each row has H=0)
        # Specificity = log2(4) - H(uniform) = 2.0 - 2.0 = 0.0
        # EI = 2.0 + 0.0 = 2.0
        assert abs(ei - 2.0) < 1e-10

    def test_ei_uniform_tpm(self):
        """Uniform TPM (every state equally likely) has EI = 0."""
        tpm = np.ones((4, 4)) / 4.0
        ei = compute_ei(tpm)
        assert abs(ei - 0.0) < 1e-10

    def test_ei_nonnegative(self):
        """EI should always be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            tpm = rng.dirichlet(np.ones(8), size=8)
            ei = compute_ei(tpm)
            assert ei >= -1e-10  # allow tiny float errors

    def test_ei_single_state_returns_zero(self):
        """Single-state TPM has EI = 0 (no causal structure possible)."""
        tpm = np.array([[1.0]])
        ei = compute_ei(tpm)
        assert ei == 0.0


class TestTPMConversion:
    """Test node-factored TPM to state-to-state conversion."""

    def test_conversion_shape(self):
        """(32, 5) node TPM converts to (32, 32) state TPM."""
        node_tpm = np.full((32, 5), 0.5)  # all nodes 50/50
        state_tpm = node_tpm_to_state_tpm(node_tpm)
        assert state_tpm.shape == (32, 32)

    def test_rows_sum_to_one(self):
        """Each row of converted TPM sums to 1.0."""
        rng = np.random.default_rng(42)
        node_tpm = rng.random((32, 5))
        state_tpm = node_tpm_to_state_tpm(node_tpm)
        row_sums = state_tpm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_deterministic_nodes(self):
        """If all nodes deterministically go ON, state 31 (all-ON) gets prob 1."""
        node_tpm = np.ones((32, 5))  # all nodes always ON
        state_tpm = node_tpm_to_state_tpm(node_tpm)
        # State 31 = all bits ON (11111)
        np.testing.assert_allclose(state_tpm[:, 31], 1.0, atol=1e-10)
        # All other columns should be 0
        for col in range(31):
            np.testing.assert_allclose(state_tpm[:, col], 0.0, atol=1e-10)

    def test_uniform_nodes(self):
        """If all nodes are 50/50, output should be uniform over all 32 states."""
        node_tpm = np.full((32, 5), 0.5)
        state_tpm = node_tpm_to_state_tpm(node_tpm)
        expected = np.full((32, 32), 1.0 / 32)
        np.testing.assert_allclose(state_tpm, expected, atol=1e-10)


class TestBellPartitions:
    """Test Bell partition enumeration."""

    def test_bell_5_count(self):
        """Bell(5) = 52 partitions."""
        partitions = generate_bell_partitions(5)
        assert len(partitions) == 52

    def test_includes_micro_and_macro(self):
        """Partitions include identity (micro) and fully-merged (macro)."""
        partitions = generate_bell_partitions(5)
        micro = [{0}, {1}, {2}, {3}, {4}]
        macro = [{0, 1, 2, 3, 4}]
        assert any(p == micro for p in partitions)
        assert any(p == macro for p in partitions)

    def test_all_elements_covered(self):
        """Every partition covers all 5 elements exactly once."""
        for partition in generate_bell_partitions(5):
            all_elements = set()
            for group in partition:
                assert len(group & all_elements) == 0  # no overlap
                all_elements |= group
            assert all_elements == {0, 1, 2, 3, 4}

    def test_bell_3_count(self):
        """Bell(3) = 5 (simpler sanity check)."""
        assert len(generate_bell_partitions(3)) == 5


class TestMacroTPM:
    """Test macro TPM construction from micro TPM + partition."""

    def test_macro_tpm_shape(self):
        """Partition of 4 nodes into 2 groups -> 4 macro states -> (4, 4) TPM."""
        rng = np.random.default_rng(42)
        micro_tpm = rng.dirichlet(np.ones(16), size=16)  # 4 nodes, 16 states
        partition = [{0, 1}, {2, 3}]  # 2 groups of 4 nodes
        macro_tpm = build_macro_tpm(micro_tpm, partition, n_micro_nodes=4)
        assert macro_tpm.shape == (4, 4)

    def test_macro_tpm_rows_sum_to_one(self):
        """Macro TPM rows sum to 1.0."""
        rng = np.random.default_rng(42)
        micro_tpm = rng.dirichlet(np.ones(32), size=32)
        partition = [{0, 1, 2}, {3, 4}]  # 2 groups of 5 nodes
        macro_tpm = build_macro_tpm(micro_tpm, partition, n_micro_nodes=5)
        row_sums = macro_tpm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_identity_partition_preserves_ei(self):
        """Identity partition (each element alone) should give same EI as micro."""
        rng = np.random.default_rng(42)
        node_tpm = rng.random((4, 2))  # 2 nodes, 4 states
        state_tpm = node_tpm_to_state_tpm(node_tpm)
        micro_ei = compute_ei(state_tpm)
        identity = [{0}, {1}]
        macro_tpm = build_macro_tpm(state_tpm, identity, n_micro_nodes=2)
        macro_ei = compute_ei(macro_tpm)
        assert abs(micro_ei - macro_ei) < 1e-8


class TestPartitionLabels:
    """Test human-readable partition labels."""

    def test_identity_label(self):
        label = partition_to_label([{0}, {1}, {2}, {3}, {4}])
        assert "GWT" in label and "BLT" in label

    def test_merged_label(self):
        label = partition_to_label([{0, 1}, {2, 3, 4}])
        assert "GWT+AST" in label or "AST+GWT" in label


class TestCausalEmergenceAnalyzer:
    """Test the full analyzer."""

    def _make_phi_tracker_mock(self):
        """Create a mock PhiTracker with a known TPM."""

        class MockPhiTracker:
            N_NODES = 5
            N_STATES = 32
            min_transitions = 128
            warming_up = False
            transitions_recorded = 200
            latest_phi = 0.5

            def build_tpm(self):
                # Slightly structured: nodes tend to stay in same state
                rng = np.random.default_rng(42)
                tpm = rng.random((32, 5))
                # Bias toward persistence
                tpm = tpm * 0.3 + 0.35
                return tpm

            def get_tpm_coverage(self):
                return 0.8

        return MockPhiTracker()

    def test_analyze_modules_returns_result(self):
        """analyze_modules() returns a CausalEmergenceResult."""
        analyzer = CausalEmergenceAnalyzer()
        phi = self._make_phi_tracker_mock()
        result = analyzer.analyze_modules(phi)
        assert isinstance(result, CausalEmergenceResult)
        assert result.micro_ei >= 0
        assert result.optimal_macro_ei >= 0
        assert result.optimal_partition is not None
        assert len(result.all_partitions) == 52

    def test_ce_is_difference(self):
        """CE = optimal_macro_ei - micro_ei."""
        analyzer = CausalEmergenceAnalyzer()
        phi = self._make_phi_tracker_mock()
        result = analyzer.analyze_modules(phi)
        expected_ce = result.optimal_macro_ei - result.micro_ei
        assert abs(result.causal_emergence - expected_ce) < 1e-10

    def test_warming_up_returns_none(self):
        """When PhiTracker is warming up, returns None."""
        analyzer = CausalEmergenceAnalyzer()
        phi = self._make_phi_tracker_mock()
        phi.warming_up = True
        result = analyzer.analyze_modules(phi)
        assert result is None

    def test_ce2_delta_cp_computed(self):
        """CE 2.0 DeltaCP should be computed."""
        analyzer = CausalEmergenceAnalyzer()
        phi = self._make_phi_tracker_mock()
        result = analyzer.analyze_modules(phi)
        assert result.ce2_delta_cp is not None

    def test_persistence_round_trip(self):
        """State dict round-trip preserves results."""
        analyzer = CausalEmergenceAnalyzer()
        phi = self._make_phi_tracker_mock()
        analyzer.analyze_modules(phi)
        state = analyzer.to_state_dict()
        json.dumps(state, allow_nan=False)
        analyzer2 = CausalEmergenceAnalyzer()
        analyzer2.from_state_dict(json.loads(json.dumps(state)))
        assert analyzer2._analysis_count == analyzer._analysis_count
        assert analyzer.get_latest_ce() is not None
        assert analyzer2.get_latest_ce() == analyzer.get_latest_ce()
        original, restored = analyzer._last_result, analyzer2._last_result
        assert restored.optimal_partition_label == original.optimal_partition_label
        assert restored.micro_ei == original.micro_ei
        assert restored.optimal_macro_ei == original.optimal_macro_ei
        assert restored.ce2_delta_cp == original.ce2_delta_cp
        assert restored.phi_comparison == original.phi_comparison
        assert sorted(sorted(g) for g in restored.optimal_partition) == sorted(
            sorted(g) for g in original.optimal_partition
        )
        assert restored.optimal_partition
        assert restored.computation_time_ms == original.computation_time_ms
        assert restored.all_partitions == []
        report = analyzer2.get_research_report()
        assert report["restored"] is True
        assert report["module_level"]["causal_emergence"] == analyzer.get_latest_ce()
        assert report["module_level"]["all_partitions"] == []
        assert analyzer.get_research_report()["restored"] is False


# ---------------------------------------------------------------------------
# Row order: the PhiTracker indexes TPM rows big-endian, the EI math reads bit
# i of a state index as node i (little-endian).
# ---------------------------------------------------------------------------


class _FixedTracker:
    """A warm tracker stand-in that returns a fixed node-factored TPM."""

    warming_up = False
    latest_phi = None

    def __init__(self, tpm):
        self._tpm = np.asarray(tpm, dtype=np.float64)

    def build_tpm(self):
        return self._tpm.copy()


def _asymmetric_little_endian_tpm() -> np.ndarray:
    """A 5-node node-factored TPM whose dynamics depend on specific nodes.

    Row s (little-endian: bit i of s is node i) gives P(node_i ON next).
    Node 0 copies node 4, node 1 is the AND of nodes 0 and 2, node 2 negates
    node 3, node 3 keeps itself and node 4 is noisy; the asymmetry makes a
    row-order slip change the macro analysis.
    """
    rng = np.random.default_rng(11)
    tpm = np.zeros((32, 5))
    for s in range(32):
        bits = [(s >> i) & 1 for i in range(5)]
        tpm[s, 0] = 0.9 if bits[4] else 0.1
        tpm[s, 1] = 0.85 if (bits[0] and bits[2]) else 0.15
        tpm[s, 2] = 0.2 if bits[3] else 0.8
        tpm[s, 3] = 0.95 if bits[3] else 0.05
        tpm[s, 4] = 0.3 + 0.4 * rng.random()
    return tpm


def test_big_endian_rows_give_same_ce_as_little_endian():
    from mtc.consciousness.pyphi_bridge import to_little_endian_rows

    tpm_le = _asymmetric_little_endian_tpm()
    tpm_be = to_little_endian_rows(tpm_le)  # bit reversal is its own inverse
    little = CausalEmergenceAnalyzer(tpm_row_order="little_endian").analyze_modules(
        _FixedTracker(tpm_le)
    )
    big = CausalEmergenceAnalyzer(tpm_row_order="big_endian").analyze_modules(
        _FixedTracker(tpm_be)
    )
    assert abs(little.causal_emergence - big.causal_emergence) < 1e-12
    assert little.optimal_partition_label == big.optimal_partition_label
    assert abs(little.micro_ei - big.micro_ei) < 1e-12


def test_little_endian_option_uses_rows_as_given():
    tpm = _asymmetric_little_endian_tpm()
    result = CausalEmergenceAnalyzer(tpm_row_order="little_endian").analyze_modules(
        _FixedTracker(tpm)
    )
    state_tpm = node_tpm_to_state_tpm(tpm)
    assert abs(result.micro_ei - compute_ei(state_tpm)) < 1e-12
    best = max(
        compute_ei(build_macro_tpm(state_tpm, p, 5))
        for p in generate_bell_partitions(5)
    )
    assert abs(result.optimal_macro_ei - max(best, result.micro_ei)) < 1e-12

    # The same rows read as big-endian reach a different macro analysis.
    swapped = CausalEmergenceAnalyzer(tpm_row_order="big_endian").analyze_modules(
        _FixedTracker(tpm)
    )
    assert (
        abs(swapped.causal_emergence - result.causal_emergence) > 1e-9
        or swapped.optimal_partition_label != result.optimal_partition_label
    )


def test_invalid_row_order_rejected():
    import pytest

    with pytest.raises(ValueError, match="tpm_row_order"):
        CausalEmergenceAnalyzer(tpm_row_order="middle_endian")


def test_ce_with_real_phi_tracker():
    from types import SimpleNamespace

    from mtc.consciousness.phi_tracker import PhiTracker

    tracker = PhiTracker(min_transitions=20)
    rng = np.random.default_rng(5)
    level = 0.5
    for _ in range(60):
        level = float(np.clip(level + rng.normal(0.0, 0.15), 0.0, 1.0))
        state = SimpleNamespace(
            integration_level=level,
            protoself_stability=float(rng.random()),
            feeling_of_knowing=1.0 - level,
            homeostatic_urgency=("curiosity", float(rng.random())),
            binding_quality=float(rng.random()),
        )
        tracker.record_module_states(state)
    assert not tracker.warming_up
    result = CausalEmergenceAnalyzer().analyze_modules(tracker)
    assert result is not None
    assert np.isfinite(result.causal_emergence)
    assert len(result.all_partitions) == 52


# ---------------------------------------------------------------------------
# Restore of the persisted result, and the module text
# ---------------------------------------------------------------------------

SOURCE = (
    Path(__file__).resolve().parents[2] / "mtc" / "assessment" / "causal_emergence.py"
)


def _analyzed():
    analyzer = CausalEmergenceAnalyzer(tpm_row_order="little_endian")
    analyzer.analyze_modules(_FixedTracker(_asymmetric_little_endian_tpm()))
    return analyzer


def test_restore_rejects_non_finite_result(caplog):
    for bad in (float("nan"), float("inf"), None, "0.3"):
        state = _analyzed().to_state_dict()
        state["histories"]["last_result"]["causal_emergence"] = bad
        restored = CausalEmergenceAnalyzer()
        with caplog.at_level(logging.WARNING, logger="mtc.assessment.causal_emergence"):
            caplog.clear()
            restored.from_state_dict(state)
        assert restored._analysis_count == 1
        assert restored.get_latest_ce() is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
    missing = _analyzed().to_state_dict()
    del missing["histories"]["last_result"]["micro_ei"]
    restored = CausalEmergenceAnalyzer()
    restored.from_state_dict(missing)
    assert restored.get_latest_ce() is None


def test_restore_validates_the_counter(caplog):
    for bad in (-1, 2.5, "3", True):
        restored = CausalEmergenceAnalyzer()
        with caplog.at_level(logging.WARNING, logger="mtc.assessment.causal_emergence"):
            caplog.clear()
            restored.from_state_dict({"scalars": {"analysis_count": bad}})
        assert restored._analysis_count == 0, bad
        assert any(r.levelno == logging.WARNING for r in caplog.records), bad
    good = CausalEmergenceAnalyzer()
    good.from_state_dict({"scalars": {"analysis_count": 4}})
    assert good._analysis_count == 4
    assert good.get_latest_ce() is None


def test_state_written_before_the_partition_was_saved_restores():
    analyzer = _analyzed()
    state = analyzer.to_state_dict()
    for key in ("optimal_partition", "computation_time_ms"):
        del state["histories"]["last_result"][key]
    restored = CausalEmergenceAnalyzer()
    restored.from_state_dict(state)
    assert restored.get_latest_ce() == analyzer.get_latest_ce()
    assert restored._last_result.optimal_partition == []
    assert restored._last_result.computation_time_ms is None
    assert restored.get_research_report()["restored"] is True


def test_assessment_reads_the_restored_analyzer():
    from mtc.assessment.assessment import ConsciousnessAssessment

    analyzer = _analyzed()
    restored = CausalEmergenceAnalyzer()
    restored.from_state_dict(json.loads(json.dumps(analyzer.to_state_dict())))

    def score(ce_analyzer):
        report = asyncio.run(
            ConsciousnessAssessment().run_full_assessment(
                causal_emergence_analyzer=ce_analyzer, normalize=False
            )
        )
        return report.indicator_results["causal_emergence"]

    original = score(analyzer)
    again = score(restored)
    assert original.evidence["status"] == "measured"
    assert again.evidence["status"] == "measured"
    assert again.score == original.score
    assert again.evidence["causal_emergence"] == original.evidence["causal_emergence"]


def test_module_docstring_makes_no_priority_claim():
    source = SOURCE.read_text(encoding="utf-8")
    assert "first systematic" not in source.lower()


def test_module_has_no_author_line():
    source = SOURCE.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(Created|Authors?):", source, re.M)
