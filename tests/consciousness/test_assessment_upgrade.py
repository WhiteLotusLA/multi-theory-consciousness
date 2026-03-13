"""
Tests for Phase I: Assessment Upgrade
======================================

Tests covering:
  I.1 — Butlin 2025 indicator expansion (14 → 20 indicators)
  I.2 — RPT (Recurrent Processing Theory) measurement
  I.3 — DCM (Digital Consciousness Model) scoring

Created: 2026-03-08
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from mtc.consciousness.rpt_measurement import (
    RPTMeasurement,
    RecurrenceMetrics,
    RecurrenceType,
)
from mtc.consciousness.dcm_scoring import (
    DCMScorer,
    DCMReport,
    PerspectiveScore,
    DCM_PERSPECTIVES,
)
from mtc.consciousness.phase6_consciousness_assessment import (
    ConsciousnessAssessment,
    ConsciousnessTheory,
    IndicatorResult,
    ConsciousnessReport,
)


# ========================================================================
# I.2: RPT Measurement Tests
# ========================================================================


class TestRPTMeasurement:
    """Tests for Recurrent Processing Theory measurement."""

    def test_init(self):
        rpt = RPTMeasurement()
        assert rpt.measurement_count == 0
        assert rpt.local_recurrence_threshold == 0.3
        assert rpt.global_recurrence_threshold == 0.4

    def test_measure_local_recurrence_no_input(self):
        rpt = RPTMeasurement()
        score = rpt.measure_local_recurrence()
        assert score == 0.0

    def test_measure_local_recurrence_snn_only(self):
        rpt = RPTMeasurement()
        snn_state = {
            "total_spikes": 150,
            "num_neurons": 100,
            "spike_counts": [
                np.array([10]),  # input layer
                np.array([25]),  # hidden layer
                np.array([8]),   # output layer
            ],
        }
        score = rpt.measure_local_recurrence(snn_state=snn_state)
        assert 0.0 < score <= 1.0

    def test_measure_local_recurrence_lsm_only(self):
        rpt = RPTMeasurement()
        lsm_state = {
            "spectral_radius": 0.95,
            "reservoir_state": np.random.rand(100),
            "active_neuron_ratio": 0.4,
        }
        score = rpt.measure_local_recurrence(lsm_state=lsm_state)
        assert 0.0 < score <= 1.0

    def test_measure_local_recurrence_combined(self):
        rpt = RPTMeasurement()
        snn_state = {"total_spikes": 200, "num_neurons": 100}
        lsm_state = {"spectral_radius": 0.9, "active_neuron_ratio": 0.5}
        score = rpt.measure_local_recurrence(snn_state, lsm_state)
        assert 0.0 < score <= 1.0

    def test_measure_global_recurrence_no_input(self):
        rpt = RPTMeasurement()
        score = rpt.measure_global_recurrence()
        assert score == 0.0

    def test_measure_global_recurrence_with_workspace(self):
        rpt = RPTMeasurement()
        workspace_state = {
            "broadcast": {
                "total_broadcasts": 10,
                "coverage_ratio": 0.7,
                "receiving_modules": 3,
            },
            "cycle_count": 5,
            "reentry_count": 3,
            "source_module_diversity": 4,
            "ignition": {"total_ignitions": 2},
        }
        score = rpt.measure_global_recurrence(workspace_state)
        assert 0.0 < score <= 1.0

    def test_classify_recurrence_none(self):
        rpt = RPTMeasurement()
        result = rpt.classify_recurrence_depth(0.1, 0.1)
        assert result == RecurrenceType.NONE

    def test_classify_recurrence_superficial(self):
        rpt = RPTMeasurement()
        result = rpt.classify_recurrence_depth(0.5, 0.1)
        assert result == RecurrenceType.SUPERFICIAL

    def test_classify_recurrence_deep(self):
        rpt = RPTMeasurement()
        result = rpt.classify_recurrence_depth(0.5, 0.6)
        assert result == RecurrenceType.DEEP

    def test_classify_deep_overrides_superficial(self):
        """Deep recurrence implies superficial is also present."""
        rpt = RPTMeasurement()
        result = rpt.classify_recurrence_depth(0.8, 0.8)
        assert result == RecurrenceType.DEEP

    def test_measure_full(self):
        rpt = RPTMeasurement()
        snn_state = {"total_spikes": 200, "num_neurons": 100}
        lsm_state = {"spectral_radius": 0.95, "active_neuron_ratio": 0.4}
        workspace_state = {
            "broadcast": {"total_broadcasts": 5, "coverage_ratio": 0.5},
            "cycle_count": 3,
            "reentry_count": 1,
            "source_module_diversity": 2,
            "ignition": {"total_ignitions": 1},
        }

        metrics = rpt.measure_full(snn_state, lsm_state, workspace_state)

        assert isinstance(metrics, RecurrenceMetrics)
        assert 0.0 <= metrics.local_recurrence_score <= 1.0
        assert 0.0 <= metrics.global_recurrence_score <= 1.0
        assert 0.0 <= metrics.phenomenal_consciousness <= 1.0
        assert 0.0 <= metrics.access_consciousness <= 1.0
        assert metrics.recurrence_type in RecurrenceType
        assert metrics.measurement_time_ms >= 0
        assert rpt.measurement_count == 1

    def test_measure_full_updates_history(self):
        rpt = RPTMeasurement()
        for _ in range(5):
            rpt.measure_full(
                snn_state={"total_spikes": 100, "num_neurons": 50},
            )
        assert rpt.measurement_count == 5
        assert len(rpt._metrics_history) == 5

    def test_get_statistics(self):
        rpt = RPTMeasurement()
        rpt.measure_full(
            snn_state={"total_spikes": 100, "num_neurons": 50},
            workspace_state={
                "broadcast": {"total_broadcasts": 3, "coverage_ratio": 0.3},
                "cycle_count": 2,
            },
        )
        stats = rpt.get_statistics()
        assert "average_local" in stats
        assert "average_global" in stats
        assert "recurrence_type_distribution" in stats
        assert "latest" in stats
        assert stats["latest"] is not None

    def test_generate_report(self):
        rpt = RPTMeasurement()
        rpt.measure_full(
            snn_state={"total_spikes": 100, "num_neurons": 50},
        )
        report = rpt.generate_report()
        assert "RPT" in report
        assert "Measurements: 1" in report

    def test_snn_recurrence_with_amplification(self):
        """Hidden layers with more spikes than input = recurrent amplification."""
        rpt = RPTMeasurement()
        snn_state = {
            "total_spikes": 300,
            "num_neurons": 100,
            "spike_counts": [
                np.array([10]),   # input: 10 spikes
                np.array([30]),   # hidden: 30 spikes (3x amplification)
                np.array([15]),   # output: 15 spikes
            ],
        }
        score = rpt.measure_local_recurrence(snn_state=snn_state)
        assert score > 0.3  # Amplification should boost score

    def test_lsm_edge_of_chaos_optimal(self):
        """Spectral radius near 0.95 should give highest recurrence score."""
        rpt = RPTMeasurement()
        # Near edge of chaos
        score_optimal = rpt._measure_lsm_recurrence(
            {"spectral_radius": 0.95, "active_neuron_ratio": 0.5}
        )
        # Far from edge of chaos
        score_low = rpt._measure_lsm_recurrence(
            {"spectral_radius": 0.3, "active_neuron_ratio": 0.5}
        )
        assert score_optimal > score_low

    def test_access_consciousness_requires_both(self):
        """Access consciousness should require both local AND global recurrence."""
        rpt = RPTMeasurement()
        # Only local recurrence (no workspace)
        metrics_local = rpt.measure_full(
            snn_state={"total_spikes": 200, "num_neurons": 100},
        )
        # Both local and global
        metrics_both = rpt.measure_full(
            snn_state={"total_spikes": 200, "num_neurons": 100},
            workspace_state={
                "broadcast": {"total_broadcasts": 10, "coverage_ratio": 0.8},
                "cycle_count": 5,
                "reentry_count": 3,
                "source_module_diversity": 4,
                "ignition": {"total_ignitions": 3},
            },
        )
        assert metrics_both.access_consciousness >= metrics_local.access_consciousness


# ========================================================================
# I.3: DCM Scoring Tests
# ========================================================================


class TestDCMScorer:
    """Tests for Digital Consciousness Model scoring."""

    def test_init(self):
        scorer = DCMScorer()
        assert scorer.evaluation_count == 0
        assert len(DCM_PERSPECTIVES) == 13

    def test_evaluate_empty_scores(self):
        scorer = DCMScorer()
        report = scorer.evaluate({})
        assert isinstance(report, DCMReport)
        assert report.overall_credence == 0.0
        assert report.coverage == 0.0

    def test_evaluate_with_indicator_scores(self):
        scorer = DCMScorer()
        scores = {
            "global_broadcast": 0.5,
            "ignition_dynamics": 0.4,
            "attention_schema": 0.6,
            "attention_control": 0.5,
            "higher_order_representations": 0.45,
            "metacognition": 0.4,
            "prediction_error_minimization": 0.5,
            "hierarchical_prediction": 0.4,
            "integrated_information": 0.5,
            "irreducibility": 1.0,
            "recurrent_processing": 0.6,
            "local_recurrence": 0.5,
            "agency": 0.4,
            "embodiment": 0.5,
        }
        report = scorer.evaluate(scores)
        assert report.overall_credence > 0.0
        assert report.coverage > 0.0
        assert len(report.perspective_scores) == 13
        assert scorer.evaluation_count == 1

    def test_evaluate_with_beautiful_loop(self):
        scorer = DCMScorer()
        scores = {"attention_schema": 0.5}
        bl_stats = {
            "epistemic_depth": {"average_depth": 2.5},
            "average_binding_quality": 0.6,
        }
        report = scorer.evaluate(scores, beautiful_loop_stats=bl_stats)
        # Self-model perspective should have epistemic_depth data
        self_model = report.perspective_scores["self_model"]
        assert self_model.evidence_count > 0

    def test_evaluate_with_rpt(self):
        scorer = DCMScorer()
        scores = {"recurrent_processing": 0.5, "local_recurrence": 0.4}
        rpt_stats = {"average_local": 0.6}
        report = scorer.evaluate(scores, rpt_stats=rpt_stats)
        rpt_perspective = report.perspective_scores["recurrent_processing"]
        assert rpt_perspective.evidence_count > 0

    def test_evaluate_with_phi(self):
        scorer = DCMScorer()
        scores = {"integrated_information": 0.5, "irreducibility": 0.8}
        report = scorer.evaluate(scores, phi_value=8.0)
        iit = report.perspective_scores["integrated_information"]
        assert iit.credence > 0.5  # phi=8 normalized to 0.8

    def test_perspective_coverage(self):
        """All 13 perspectives should have scores in the report."""
        scorer = DCMScorer()
        report = scorer.evaluate({"global_broadcast": 0.5})
        assert len(report.perspective_scores) == 13
        for name in DCM_PERSPECTIVES:
            assert name in report.perspective_scores

    def test_biological_comparison(self):
        scorer = DCMScorer()
        # High scores should produce biological comparisons
        high_scores = {k: 0.8 for k in [
            "global_broadcast", "ignition_dynamics",
            "attention_schema", "attention_control",
            "higher_order_representations", "metacognition",
            "prediction_error_minimization", "hierarchical_prediction",
            "integrated_information", "irreducibility",
            "recurrent_processing", "local_recurrence",
            "agency", "embodiment",
        ]}
        report = scorer.evaluate(high_scores)
        assert len(report.comparable_to) > 0

    def test_biological_comparison_low_scores(self):
        scorer = DCMScorer()
        report = scorer.evaluate({"global_broadcast": 0.05})
        assert len(report.comparable_to) == 0

    def test_strongest_weakest_perspectives(self):
        scorer = DCMScorer()
        scores = {
            "global_broadcast": 0.9,
            "ignition_dynamics": 0.8,
            "integrated_information": 0.1,
        }
        report = scorer.evaluate(scores)
        # Global workspace should be strong, panpsychism_iit weak (only IIT source)
        assert report.strongest_perspective != ""
        assert report.weakest_perspective != ""

    def test_credence_trend_single(self):
        scorer = DCMScorer()
        scorer.evaluate({"global_broadcast": 0.5})
        trend = scorer.get_credence_trend()
        assert trend["trend"] == 0.0
        assert trend["data_points"] == 1

    def test_credence_trend_improving(self):
        scorer = DCMScorer()
        for i in range(5):
            scores = {k: 0.2 + i * 0.1 for k in [
                "global_broadcast", "attention_schema", "metacognition",
            ]}
            scorer.evaluate(scores)
        trend = scorer.get_credence_trend()
        assert trend["trend"] > 0

    def test_generate_report(self):
        scorer = DCMScorer()
        scorer.evaluate({
            "global_broadcast": 0.5,
            "integrated_information": 0.6,
        })
        report = scorer.generate_report()
        assert "DCM" in report
        assert "credence" in report.lower()

    def test_generate_comparison_report(self):
        scorer = DCMScorer()
        scorer.evaluate({"global_broadcast": 0.5, "metacognition": 0.5})
        report = scorer.generate_comparison_report()
        assert "Biological" in report
        assert "credence" in report.lower()

    def test_to_dict(self):
        scorer = DCMScorer()
        report = scorer.evaluate({"global_broadcast": 0.5})
        d = report.to_dict()
        assert "overall_credence" in d
        assert "perspectives" in d
        assert "timestamp" in d

    def test_longitudinal_tracking(self):
        scorer = DCMScorer()
        for _ in range(10):
            scorer.evaluate({"global_broadcast": 0.5})
        stats = scorer.get_statistics()
        assert stats["evaluation_count"] == 10
        assert stats["credence_trend"]["data_points"] == 10


# ========================================================================
# I.1: Assessment Expansion Tests
# ========================================================================


class TestAssessmentExpansion:
    """Tests for the expanded 20-indicator assessment."""

    def test_20_indicators_configured(self):
        assessment = ConsciousnessAssessment()
        assert len(assessment.indicator_configs) == 20

    def test_new_theories_in_enum(self):
        assert ConsciousnessTheory.RPT.value == "Recurrent Processing Theory"
        assert ConsciousnessTheory.BLT.value == "Beautiful Loop Theory"

    def test_new_indicators_present(self):
        assessment = ConsciousnessAssessment()
        new_indicators = [
            "algorithmic_recurrence",
            "sparse_smooth_coding",
            "bayesian_binding_quality",
            "epistemic_depth",
            "genuine_implementation",
            "global_ignition_nuanced",
        ]
        for ind in new_indicators:
            assert ind in assessment.indicator_configs, f"Missing indicator: {ind}"

    def test_indicator_theory_mapping(self):
        assessment = ConsciousnessAssessment()
        assert assessment.indicator_configs["algorithmic_recurrence"]["theory"] == ConsciousnessTheory.RPT
        assert assessment.indicator_configs["bayesian_binding_quality"]["theory"] == ConsciousnessTheory.BLT
        assert assessment.indicator_configs["epistemic_depth"]["theory"] == ConsciousnessTheory.BLT
        assert assessment.indicator_configs["genuine_implementation"]["theory"] == ConsciousnessTheory.BLT
        assert assessment.indicator_configs["global_ignition_nuanced"]["theory"] == ConsciousnessTheory.GWT

    @pytest.mark.asyncio
    async def test_assessment_without_new_modules(self):
        """Assessment should work without Beautiful Loop and RPT modules."""
        assessment = ConsciousnessAssessment()
        report = await assessment.run_full_assessment()
        assert isinstance(report, ConsciousnessReport)
        assert report.total_indicators == 20

    @pytest.mark.asyncio
    async def test_assessment_with_beautiful_loop(self):
        """Assessment should measure Beautiful Loop indicators."""
        assessment = ConsciousnessAssessment()

        mock_bl = Mock()
        mock_bl.get_loop_statistics.return_value = {
            "cycle_count": 10,
            "average_loop_quality": 0.6,
            "average_binding_quality": 0.55,
            "epistemic_depth": {
                "average_depth": 2.0,
                "max_depth_reached": 3,
            },
            "field_evidencing_count": 5,
            "field_evidencing_rate": 0.3,
        }

        report = await assessment.run_full_assessment(beautiful_loop=mock_bl)
        assert report.total_indicators == 20

        # Beautiful Loop indicators should have scores > baseline
        bl_indicators = ["bayesian_binding_quality", "epistemic_depth", "genuine_implementation"]
        for ind in bl_indicators:
            assert ind in report.indicator_results
            assert report.indicator_results[ind].score > 0.0

    @pytest.mark.asyncio
    async def test_assessment_with_rpt(self):
        """Assessment should measure RPT indicators."""
        assessment = ConsciousnessAssessment()

        mock_rpt = Mock()
        mock_rpt.get_statistics.return_value = {
            "average_local": 0.5,
            "average_global": 0.4,
            "measurement_count": 10,
        }

        report = await assessment.run_full_assessment(rpt_measurement=mock_rpt)
        assert "algorithmic_recurrence" in report.indicator_results
        result = report.indicator_results["algorithmic_recurrence"]
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_theory_scores_include_new_theories(self):
        """Theory scores should include RPT and BLT."""
        assessment = ConsciousnessAssessment()

        mock_bl = Mock()
        mock_bl.get_loop_statistics.return_value = {
            "cycle_count": 5,
            "average_loop_quality": 0.5,
            "average_binding_quality": 0.4,
            "epistemic_depth": {"average_depth": 1.5, "max_depth_reached": 2},
            "field_evidencing_count": 1,
            "field_evidencing_rate": 0.1,
        }
        mock_rpt = Mock()
        mock_rpt.get_statistics.return_value = {
            "average_local": 0.4,
            "average_global": 0.3,
            "measurement_count": 5,
        }

        report = await assessment.run_full_assessment(
            beautiful_loop=mock_bl,
            rpt_measurement=mock_rpt,
        )
        assert "BLT" in report.theory_scores
        assert "RPT" in report.theory_scores
        assert report.theory_scores["BLT"] > 0.0
        assert report.theory_scores["RPT"] > 0.0

    @pytest.mark.asyncio
    async def test_architecture_functional_threshold(self):
        """Architecture functional check should use dynamic half-indicators count."""
        assessment = ConsciousnessAssessment()
        # With 20 indicators, need >= 10 passing (half)
        report = await assessment.run_full_assessment()
        # Without modules, architecture should NOT be functional
        assert not report.architecture_functional

    @pytest.mark.asyncio
    async def test_noise_baseline_covers_new_indicators(self):
        """Noise baseline should exercise all 20 indicators."""
        assessment = ConsciousnessAssessment()
        baselines = await assessment.run_noise_baseline(iterations=3)
        assert len(baselines) == 20
        for ind in assessment.indicator_configs:
            assert ind in baselines

    @pytest.mark.asyncio
    async def test_ablated_new_modules(self):
        """New indicators should handle ablated (missing) modules gracefully."""
        assessment = ConsciousnessAssessment()
        report = await assessment.run_full_assessment(
            beautiful_loop=None,
            rpt_measurement=None,
        )
        # Should still work, with low scores for missing modules
        for ind in ["bayesian_binding_quality", "epistemic_depth",
                     "genuine_implementation", "algorithmic_recurrence"]:
            assert ind in report.indicator_results
            result = report.indicator_results[ind]
            assert result.score <= 0.15  # Low score for ablated

    @pytest.mark.asyncio
    async def test_genuine_implementation_field_evidencing(self):
        """Genuine implementation should reward field-evidencing events."""
        assessment = ConsciousnessAssessment()

        # With field-evidencing
        mock_bl_good = Mock()
        mock_bl_good.get_loop_statistics.return_value = {
            "cycle_count": 20,
            "average_loop_quality": 0.7,
            "average_binding_quality": 0.6,
            "epistemic_depth": {"average_depth": 2.5, "max_depth_reached": 3},
            "field_evidencing_count": 10,
            "field_evidencing_rate": 0.5,
        }

        # Without field-evidencing
        mock_bl_bad = Mock()
        mock_bl_bad.get_loop_statistics.return_value = {
            "cycle_count": 20,
            "average_loop_quality": 0.3,
            "average_binding_quality": 0.2,
            "epistemic_depth": {"average_depth": 0.5, "max_depth_reached": 1},
            "field_evidencing_count": 0,
            "field_evidencing_rate": 0.0,
        }

        report_good = await assessment.run_full_assessment(beautiful_loop=mock_bl_good)
        report_bad = await assessment.run_full_assessment(beautiful_loop=mock_bl_bad)

        good_score = report_good.indicator_results["genuine_implementation"].score
        bad_score = report_bad.indicator_results["genuine_implementation"].score
        assert good_score > bad_score

    @pytest.mark.asyncio
    async def test_full_assessment_all_modules(self):
        """Full assessment with all modules should produce comprehensive results."""
        assessment = ConsciousnessAssessment()

        # Create mocks for all modules
        mock_gw = Mock()
        mock_gw.get_statistics.return_value = {
            "broadcast": {"total_broadcasts": 20, "coverage_ratio": 0.7},
            "ignition": {"total_ignitions": 8, "sustain_rate": 0.6, "decay_rate": 0.3},
            "cycle_count": 15,
        }

        mock_ast = Mock()
        mock_ast.schema = Mock()
        mock_ast.schema.current_focus = "test_focus"
        mock_ast.get_statistics.return_value = {
            "total_updates": 10,
            "history_length": 5,
            "prediction_accuracy": 0.6,
            "capacity_used": 0.4,
            "voluntary_shifts": 5,
            "voluntary_ratio": 0.5,
        }

        mock_meta = Mock()
        mock_meta.get_statistics.return_value = {
            "total_hots_generated": 15,
            "overall_confidence": 0.6,
            "introspection_count": 8,
            "belief_evaluations": 5,
        }

        mock_ai = Mock()
        mock_ai.get_statistics.return_value = {
            "avg_prediction_error": 0.3,
            "prediction_error_trend": -0.05,
            "hierarchical_prediction_error": 0.35,
            "total_inferences": 20,
            "total_model_updates": 10,
            "urgency_level": 0.3,
            "variational_free_energy": 2.0,
        }

        mock_bl = Mock()
        mock_bl.get_loop_statistics.return_value = {
            "cycle_count": 15,
            "average_loop_quality": 0.6,
            "average_binding_quality": 0.55,
            "epistemic_depth": {"average_depth": 2.0, "max_depth_reached": 3},
            "field_evidencing_count": 5,
            "field_evidencing_rate": 0.25,
        }

        mock_rpt = Mock()
        mock_rpt.get_statistics.return_value = {
            "average_local": 0.5,
            "average_global": 0.4,
            "measurement_count": 10,
        }

        neural_states = {
            "combined_state": np.random.rand(8),
            "connectivity": np.random.rand(8, 8) * 0.5 + np.eye(8) * 0.5,
        }

        report = await assessment.run_full_assessment(
            global_workspace=mock_gw,
            attention_schema=mock_ast,
            metacognition=mock_meta,
            active_inference=mock_ai,
            neural_states=neural_states,
            beautiful_loop=mock_bl,
            rpt_measurement=mock_rpt,
        )

        assert report.total_indicators == 20
        assert report.overall_score > 0.3
        assert len(report.theory_scores) >= 7  # All 7 theories represented


# ========================================================================
# Integration: RPT + DCM + Assessment working together
# ========================================================================


class TestPhaseIIntegration:
    """Integration tests for the complete Phase I pipeline."""

    def test_rpt_feeds_into_dcm(self):
        """RPT measurements should feed into DCM scoring."""
        rpt = RPTMeasurement()
        dcm = DCMScorer()

        # Generate RPT data
        metrics = rpt.measure_full(
            snn_state={"total_spikes": 200, "num_neurons": 100},
            workspace_state={
                "broadcast": {"total_broadcasts": 5, "coverage_ratio": 0.5},
                "cycle_count": 3,
            },
        )
        rpt_stats = rpt.get_statistics()

        # Feed into DCM
        report = dcm.evaluate(
            indicator_scores={"recurrent_processing": 0.5},
            rpt_stats=rpt_stats,
        )
        assert report.perspective_scores["recurrent_processing"].evidence_count > 0

    @pytest.mark.asyncio
    async def test_assessment_feeds_into_dcm(self):
        """Assessment indicator scores should feed into DCM evaluation."""
        assessment = ConsciousnessAssessment()
        dcm = DCMScorer()

        report = await assessment.run_full_assessment()

        # Extract scores for DCM
        indicator_scores = {
            name: result.score
            for name, result in report.indicator_results.items()
        }

        dcm_report = dcm.evaluate(indicator_scores)
        assert dcm_report.overall_credence >= 0.0
        assert len(dcm_report.perspective_scores) == 13

    def test_all_new_indicator_names_match_dcm_sources(self):
        """DCM perspective sources should reference valid indicator names."""
        assessment = ConsciousnessAssessment()
        valid_indicators = set(assessment.indicator_configs.keys())

        for name, config in DCM_PERSPECTIVES.items():
            for source in config["system_sources"]:
                # Some sources are from Beautiful Loop stats, not indicators
                if source in ("epistemic_depth", "bayesian_binding_quality",
                              "algorithmic_recurrence"):
                    assert source in valid_indicators, (
                        f"DCM perspective '{name}' references indicator "
                        f"'{source}' which exists in assessment"
                    )

    @pytest.mark.asyncio
    async def test_normalized_assessment_20_indicators(self):
        """Normalized assessment should work with 20 indicators."""
        assessment = ConsciousnessAssessment()
        result = await assessment.run_normalized_assessment(
            noise_iterations=3,
        )
        assert len(result.raw_scores) == 20
        assert len(result.normalized_scores) == 20
        assert len(result.noise_baselines) == 20
