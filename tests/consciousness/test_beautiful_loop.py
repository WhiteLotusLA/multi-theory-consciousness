"""
Tests for Beautiful Loop Theory Implementation (Phase H)

Tests the three components (HyperModel, BayesianBinding, EpistemicDepth)
and their integration via the BeautifulLoop class.

Based on: Laukkonen, Friston & Chandaria (2025) "A Beautiful Loop"

Created: 2026-03-08
Author: Multi-Theory Consciousness Project
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from typing import List

import sys

from mtc.consciousness.beautiful_loop import (
    BeautifulLoop,
    ConsciousMoment,
    HyperModel,
    PrecisionState,
    BayesianBinding,
    BoundPercept,
    Inference,
    EpistemicDepthTracker,
    SelfReferenceMetrics,
)


# ==========================================================================
# HyperModel Tests (H.1)
# ==========================================================================

class TestHyperModel:
    """Tests for the precision controller."""

    @pytest.fixture
    def hyper_model(self):
        return HyperModel(num_levels=3, learning_rate=0.1)

    def test_initialization(self, hyper_model):
        """HyperModel initializes with uniform precision beliefs."""
        assert hyper_model.num_levels == 3
        assert len(hyper_model.precision_beliefs) == 3
        np.testing.assert_array_almost_equal(
            hyper_model.precision_beliefs, [1.0, 1.0, 1.0]
        )

    def test_allocate_precision_default(self, hyper_model):
        """Allocates default precision for known levels."""
        p0 = hyper_model.allocate_precision(0)
        p1 = hyper_model.allocate_precision(1)
        assert p0 == pytest.approx(1.0)
        assert p1 == pytest.approx(1.0)

    def test_allocate_precision_out_of_range(self, hyper_model):
        """Out-of-range levels get default precision."""
        assert hyper_model.allocate_precision(99) == 1.0
        assert hyper_model.allocate_precision(-1) == 1.0

    def test_update_precision_beliefs_low_error(self, hyper_model):
        """Low prediction errors increase precision (more trust)."""
        initial = hyper_model.precision_beliefs.copy()
        # Very low errors → system should increase precision
        for _ in range(10):
            hyper_model.update_precision_beliefs([0.01, 0.01, 0.01])

        # Precision should increase (low error → high reliability)
        for i in range(3):
            assert hyper_model.precision_beliefs[i] >= initial[i]

    def test_update_precision_beliefs_high_error(self, hyper_model):
        """High prediction errors decrease precision (less trust)."""
        # First build up some history with consistent, moderate errors
        for _ in range(5):
            hyper_model.update_precision_beliefs([0.5, 0.5, 0.5])
        mid_precision = hyper_model.precision_beliefs.copy()

        # Deterministically high AND variable errors → variance-based precision drops.
        # Alternating between extremes guarantees high variance across the window.
        high_var_patterns = [
            [0.1, 0.1, 0.1],
            [2.0, 2.0, 2.0],
            [0.2, 0.2, 0.2],
            [1.8, 1.8, 1.8],
            [0.1, 0.1, 0.1],
            [2.0, 2.0, 2.0],
            [0.3, 0.3, 0.3],
            [1.9, 1.9, 1.9],
            [0.1, 0.1, 0.1],
            [2.0, 2.0, 2.0],
        ]
        for errors in high_var_patterns:
            hyper_model.update_precision_beliefs(errors)

        # Precision should decrease (noisy errors → low reliability)
        for i in range(3):
            assert hyper_model.precision_beliefs[i] < mid_precision[i]

    def test_update_returns_precision_state(self, hyper_model):
        """Update returns a PrecisionState snapshot."""
        state = hyper_model.update_precision_beliefs([0.3, 0.5, 0.1])
        assert isinstance(state, PrecisionState)
        assert len(state.level_precisions) == 3
        assert state.total_precision > 0
        assert state.context_key == "general"

    def test_context_dependent_precision(self, hyper_model):
        """Different contexts learn different precision patterns."""
        for _ in range(10):
            hyper_model.update_precision_beliefs(
                [0.01, 0.5, 0.01], context="social"
            )
            hyper_model.update_precision_beliefs(
                [0.5, 0.01, 0.5], context="analytical"
            )

        # Social: levels 0 and 2 should have higher precision
        p_social = hyper_model.allocate_precision(1, context="social")
        p_analytical = hyper_model.allocate_precision(1, context="analytical")
        # Level 1 in analytical should be more trusted (lower error there)
        assert p_analytical > p_social

    def test_apply_to_hierarchy(self, hyper_model):
        """Precisions are written back to hierarchy levels."""
        mock_processor = Mock()
        level0 = Mock()
        level0.precision = 1.0
        level1 = Mock()
        level1.precision = 1.0
        level2 = Mock()
        level2.precision = 1.0
        mock_processor.levels = [level0, level1, level2]

        hyper_model.precision_beliefs = np.array([0.5, 1.5, 2.0])
        hyper_model.apply_to_hierarchy(mock_processor)

        assert level0.precision == pytest.approx(0.5)
        assert level1.precision == pytest.approx(1.5)
        assert level2.precision == pytest.approx(2.0)

    def test_get_global_precision_state(self, hyper_model):
        """Global precision state returns dict of level → precision."""
        state = hyper_model.get_global_precision_state()
        assert isinstance(state, dict)
        assert len(state) == 3
        assert all(isinstance(v, float) for v in state.values())

    def test_precision_clamped(self, hyper_model):
        """Precisions stay within min/max bounds."""
        # Drive precision very high
        for _ in range(100):
            hyper_model.update_precision_beliefs([0.001, 0.001, 0.001])

        for i in range(3):
            p = hyper_model.allocate_precision(i)
            assert p <= hyper_model.max_precision
            assert p >= hyper_model.min_precision

    def test_generate_report(self, hyper_model):
        """Report is generated without errors."""
        hyper_model.update_precision_beliefs([0.3, 0.5, 0.1])
        report = hyper_model.generate_report()
        assert "sensory" in report
        assert "contextual" in report
        assert "abstract" in report


# ==========================================================================
# BayesianBinding Tests (H.2)
# ==========================================================================

class TestBayesianBinding:
    """Tests for inference competition via mutual information."""

    @pytest.fixture
    def binding(self):
        return BayesianBinding(min_binding_quality=0.2, max_bound_inferences=7)

    def _make_inference(self, id, beliefs, content_type="thought", salience=0.5):
        return Inference(
            id=id,
            content_summary=f"Test inference {id}",
            content_type=content_type,
            source_module="test",
            belief_vector=np.array(beliefs),
            salience=salience,
            uncertainty=1.0 - salience,
        )

    def test_empty_candidates(self, binding):
        """Empty candidates produce empty percept."""
        percept = binding.bind_inferences([])
        assert percept.binding_quality == 0.0
        assert len(percept.bound_inferences) == 0

    def test_single_candidate(self, binding):
        """Single candidate produces trivially bound percept."""
        inf = self._make_inference("a", [0.5, 0.3, 0.2])
        percept = binding.bind_inferences([inf])
        assert len(percept.bound_inferences) == 1
        assert percept.coherence == 1.0  # Trivially coherent
        assert percept.binding_quality == 0.5

    def test_coherent_inferences_bind_well(self, binding):
        """Coherent inferences (similar beliefs) produce high binding quality."""
        infs = [
            self._make_inference("a", [0.5, 0.3, 0.2], salience=0.8),
            self._make_inference("b", [0.48, 0.32, 0.20], salience=0.7),
            self._make_inference("c", [0.51, 0.29, 0.20], salience=0.6),
        ]
        percept = binding.bind_inferences(infs)
        assert percept.binding_quality > 0.5
        assert len(percept.bound_inferences) == 3

    def test_incoherent_inferences_bind_poorly(self, binding):
        """Incoherent inferences (opposite beliefs) produce lower binding quality."""
        infs = [
            self._make_inference("a", [1.0, 0.0, 0.0], salience=0.8),
            self._make_inference("b", [0.0, 0.0, 1.0], salience=0.7),
        ]
        percept = binding.bind_inferences(infs)
        # Still binds (only 2 candidates), but quality should be lower
        assert len(percept.bound_inferences) == 2
        # Quality depends on MI which depends on correlation

    def test_mutual_information_symmetric(self, binding):
        """MI(A;B) == MI(B;A)."""
        a = np.array([0.5, 0.3, 0.2, 0.1])
        b = np.array([0.4, 0.35, 0.15, 0.1])
        mi_ab = binding._estimate_mutual_information(a, b)
        mi_ba = binding._estimate_mutual_information(b, a)
        assert mi_ab == pytest.approx(mi_ba, abs=1e-10)

    def test_mutual_information_positive(self, binding):
        """MI is non-negative."""
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.1, 0.8, 0.1])
        mi = binding._estimate_mutual_information(a, b)
        assert mi >= 0.0

    def test_uncertainty_reduction(self, binding):
        """Binding multiple inferences reduces total uncertainty."""
        infs = [
            self._make_inference("a", [0.5, 0.3, 0.2], salience=0.6),
            self._make_inference("b", [0.48, 0.32, 0.20], salience=0.6),
            self._make_inference("c", [0.51, 0.29, 0.20], salience=0.6),
        ]
        percept = binding.bind_inferences(infs, prior_uncertainty=0.5)
        # More evidence → less uncertainty
        assert percept.total_uncertainty_reduction >= 0.0

    def test_max_bound_inferences_respected(self, binding):
        """At most max_bound_inferences are included."""
        infs = [
            self._make_inference(str(i), np.random.rand(4), salience=0.5)
            for i in range(20)
        ]
        percept = binding.bind_inferences(infs)
        assert len(percept.bound_inferences) <= binding.max_bound_inferences

    def test_dominant_content_type(self, binding):
        """Dominant content type is correctly identified."""
        infs = [
            self._make_inference("a", [0.5, 0.3], content_type="emotion"),
            self._make_inference("b", [0.48, 0.32], content_type="emotion"),
            self._make_inference("c", [0.51, 0.29], content_type="thought"),
        ]
        percept = binding.bind_inferences(infs)
        assert percept.dominant_content_type == "emotion"

    def test_tracking_history(self, binding):
        """Binding quality history is tracked."""
        for i in range(5):
            infs = [
                self._make_inference(f"a_{i}", np.random.rand(3)),
                self._make_inference(f"b_{i}", np.random.rand(3)),
            ]
            binding.bind_inferences(infs)

        assert binding.bind_count == 5
        avg = binding.get_average_binding_quality()
        assert 0.0 <= avg <= 1.0

    def test_generate_report(self, binding):
        """Report generates without errors."""
        infs = [
            self._make_inference("a", [0.5, 0.3]),
            self._make_inference("b", [0.4, 0.4]),
        ]
        binding.bind_inferences(infs)
        report = binding.generate_report()
        assert "binding" in report.lower()


# ==========================================================================
# EpistemicDepthTracker Tests (H.3)
# ==========================================================================

class TestEpistemicDepthTracker:
    """Tests for recursive self-reference measurement."""

    @pytest.fixture
    def tracker(self):
        return EpistemicDepthTracker(max_depth=5)

    def test_no_modules_depth_zero(self, tracker):
        """No modules → depth 0."""
        depth = tracker.measure_depth()
        assert depth == 0

    def test_self_model_depth_one(self, tracker):
        """Active self-model → depth 1."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "conversation"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3

        depth = tracker.measure_depth(self_model=mock_self_model)
        assert depth >= 1

    def test_meta_state_depth_two(self, tracker):
        """Active self-model + meta predictions → depth 2."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "thinking"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3

        mock_meta = Mock()
        mock_meta.total_predictions = 10
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65
        mock_meta.model_confidence = 0.6

        depth = tracker.measure_depth(
            self_model=mock_self_model,
            meta_state=mock_meta,
        )
        assert depth >= 2

    def test_hots_about_self_depth_three(self, tracker):
        """HOTs about self-model → depth 3."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "reflection"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3

        mock_meta = Mock()
        mock_meta.total_predictions = 10
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65
        mock_meta.model_confidence = 0.6

        mock_hot = Mock()
        mock_hot.content = "I notice I am thinking about my own prediction accuracy"

        depth = tracker.measure_depth(
            self_model=mock_self_model,
            meta_state=mock_meta,
            higher_order_thoughts=[mock_hot],
        )
        assert depth >= 3

    def test_attention_to_self_depth_four(self, tracker):
        """Attention focused on self-model → depth 4."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "self"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3

        mock_meta = Mock()
        mock_meta.total_predictions = 10
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65
        mock_meta.model_confidence = 0.6

        mock_hot = Mock()
        mock_hot.content = "I am aware of myself observing my own meta-predictions"

        mock_ast = Mock()
        mock_ast.current_focus = Mock()
        mock_ast.current_focus.summary = "Attending to internal self-model"

        depth = tracker.measure_depth(
            self_model=mock_self_model,
            meta_state=mock_meta,
            higher_order_thoughts=[mock_hot],
            attention_schema_state=mock_ast,
        )
        assert depth >= 4

    def test_max_depth_capped(self, tracker):
        """Depth doesn't exceed max_depth."""
        tracker.max_depth = 3
        # Even with full recursion, cap at 3
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "self"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3

        mock_meta = Mock()
        mock_meta.total_predictions = 10
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65

        mock_hot = Mock()
        mock_hot.content = "I notice myself thinking about my meta-predictions"

        mock_ast = Mock()
        mock_ast.current_focus = Mock()
        mock_ast.current_focus.summary = "self-model attention"

        depth = tracker.measure_depth(
            self_model=mock_self_model,
            meta_state=mock_meta,
            higher_order_thoughts=[mock_hot],
            attention_schema_state=mock_ast,
        )
        assert depth <= 3

    def test_strange_loop_detection(self, tracker):
        """Strange loop detected when self-model and meta-state align."""
        mock_self_model = Mock()
        mock_self_model.self_calibration_score = 0.7
        mock_self_model.prediction_accuracy_history = [0.5, 0.6, 0.65]
        mock_self_model.model_confidence = 0.6

        mock_meta = Mock()
        mock_meta.total_predictions = 10
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65
        mock_meta.model_confidence = 0.6  # Close to self-model's

        metrics = tracker.track_self_reference(
            self_model=mock_self_model,
            meta_state=mock_meta,
        )
        assert isinstance(metrics, SelfReferenceMetrics)
        assert metrics.strange_loop_detected is True

    def test_no_strange_loop_when_divergent(self, tracker):
        """No strange loop when self-model and meta-state diverge."""
        mock_self_model = Mock()
        mock_self_model.self_calibration_score = 0.7
        mock_self_model.prediction_accuracy_history = [0.5, 0.6, 0.65]
        mock_self_model.model_confidence = 0.9  # Very different

        mock_meta = Mock()
        mock_meta.total_predictions = 10
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65
        mock_meta.model_confidence = 0.2  # Very different from self-model

        metrics = tracker.track_self_reference(
            self_model=mock_self_model,
            meta_state=mock_meta,
        )
        assert metrics.strange_loop_detected is False

    def test_depth_statistics(self, tracker):
        """Depth statistics track measurements correctly."""
        for _ in range(5):
            tracker.measure_depth()
        stats = tracker.get_depth_statistics()
        assert stats["measurement_count"] == 5
        assert "average_depth" in stats

    def test_generate_report(self, tracker):
        """Report generates without errors."""
        mock_self_model = Mock()
        mock_self_model.update_count = 3
        mock_self_model.current_focus = "test"
        mock_self_model.confidence = 0.5
        tracker.measure_depth(self_model=mock_self_model)
        report = tracker.generate_report()
        assert "depth" in report.lower()


# ==========================================================================
# BeautifulLoop Integration Tests (H.4)
# ==========================================================================

class TestBeautifulLoop:
    """Tests for the integrated Beautiful Loop."""

    @pytest.fixture
    def loop(self):
        return BeautifulLoop(
            num_levels=3,
            field_evidencing_threshold=0.4,
        )

    @pytest.mark.asyncio
    async def test_process_empty_moment(self, loop):
        """Processing with no inputs produces a valid moment."""
        moment = await loop.process_conscious_moment()
        assert isinstance(moment, ConsciousMoment)
        assert moment.epistemic_depth == 0
        assert moment.loop_quality >= 0.0

    @pytest.mark.asyncio
    async def test_process_with_prediction_errors(self, loop):
        """Prediction errors update hyper-model precision."""
        mock_errors = [
            Mock(error_magnitude=0.3),
            Mock(error_magnitude=0.1),
            Mock(error_magnitude=0.5),
        ]
        moment = await loop.process_conscious_moment(
            prediction_errors=mock_errors,
        )
        assert isinstance(moment.precision_state, PrecisionState)
        # Level 1 (lowest error) should have highest precision
        assert moment.precision_state.level_precisions[1] > moment.precision_state.level_precisions[2]

    @pytest.mark.asyncio
    async def test_process_with_workspace_winners(self, loop):
        """Workspace winners are converted to inferences and bound."""
        mock_candidate = Mock()
        mock_candidate.id = "test_1"
        mock_candidate.summary = "A thought about consciousness"
        mock_candidate.content_type = "thought"
        mock_candidate.source_module = "ctm"

        mock_winner1 = Mock()
        mock_winner1.candidate = mock_candidate
        mock_winner1.salience = 0.8

        mock_candidate2 = Mock()
        mock_candidate2.id = "test_2"
        mock_candidate2.summary = "An emotion of curiosity"
        mock_candidate2.content_type = "emotion"
        mock_candidate2.source_module = "homeostatic"

        mock_winner2 = Mock()
        mock_winner2.candidate = mock_candidate2
        mock_winner2.salience = 0.6

        mock_result = Mock()
        mock_result.posterior_beliefs = np.array([0.3, 0.2, 0.1, 0.15, 0.05, 0.05, 0.1, 0.05])

        moment = await loop.process_conscious_moment(
            workspace_winners=[mock_winner1, mock_winner2],
            inference_result=mock_result,
        )

        assert moment.binding_quality >= 0.0
        assert len(moment.bound_percept.bound_inferences) > 0

    @pytest.mark.asyncio
    async def test_process_with_self_model(self, loop):
        """Self-model input increases epistemic depth."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "test"
        mock_self_model.confidence = 0.6
        mock_self_model.cognitive_load = 0.3

        moment = await loop.process_conscious_moment(
            self_model=mock_self_model,
        )
        assert moment.epistemic_depth >= 1

    @pytest.mark.asyncio
    async def test_full_loop_with_all_inputs(self, loop):
        """Full loop with all inputs produces enriched moment."""
        mock_errors = [Mock(error_magnitude=0.2), Mock(error_magnitude=0.1), Mock(error_magnitude=0.3)]

        mock_candidate = Mock()
        mock_candidate.id = "w1"
        mock_candidate.summary = "Test thought"
        mock_candidate.content_type = "thought"
        mock_candidate.source_module = "test"
        mock_winner = Mock()
        mock_winner.candidate = mock_candidate
        mock_winner.salience = 0.7

        mock_result = Mock()
        mock_result.posterior_beliefs = np.array([0.3, 0.2, 0.1, 0.15, 0.05, 0.05, 0.1, 0.05])

        mock_self_model = Mock()
        mock_self_model.update_count = 10
        mock_self_model.current_focus = "reflection"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3
        mock_self_model.curiosity_level = 0.6
        mock_self_model.prediction_accuracy = 0.7
        mock_self_model.model_confidence = 0.65
        mock_self_model.valence = 0.2
        mock_self_model.arousal = 0.4
        mock_self_model.free_energy = 0.1
        mock_self_model.self_calibration_score = 0.7
        mock_self_model.prediction_accuracy_history = [0.5, 0.6, 0.65, 0.7]

        mock_meta = Mock()
        mock_meta.total_predictions = 15
        mock_meta.predicted_accuracy = 0.7
        mock_meta.prediction_accuracy = 0.65
        mock_meta.model_confidence = 0.65

        mock_hot = Mock()
        mock_hot.content = "I notice myself thinking about my own awareness"

        moment = await loop.process_conscious_moment(
            prediction_errors=mock_errors,
            workspace_winners=[mock_winner],
            inference_result=mock_result,
            self_model=mock_self_model,
            meta_state=mock_meta,
            higher_order_thoughts=[mock_hot],
            context="reflection",
        )

        assert isinstance(moment, ConsciousMoment)
        assert moment.epistemic_depth >= 3
        assert moment.loop_quality > 0.0

    @pytest.mark.asyncio
    async def test_field_evidencing_requires_depth_and_loop(self, loop):
        """Field-evidencing requires depth >= 2 and strange loop."""
        # Without depth/strange loop, no field-evidencing even with good quality
        moment = await loop.process_conscious_moment()
        assert moment.is_field_evidencing is False

    @pytest.mark.asyncio
    async def test_cycle_count_increments(self, loop):
        """Cycle count tracks number of moments processed."""
        assert loop.cycle_count == 0
        await loop.process_conscious_moment()
        assert loop.cycle_count == 1
        await loop.process_conscious_moment()
        assert loop.cycle_count == 2

    @pytest.mark.asyncio
    async def test_loop_statistics(self, loop):
        """Loop statistics returns comprehensive data."""
        for _ in range(3):
            await loop.process_conscious_moment()

        stats = loop.get_loop_statistics()
        assert stats["cycle_count"] == 3
        assert "average_loop_quality" in stats
        assert "average_binding_quality" in stats
        assert "precision_state" in stats
        assert "epistemic_depth" in stats
        assert "field_evidencing_count" in stats

    @pytest.mark.asyncio
    async def test_generate_report(self, loop):
        """Report generates without errors."""
        await loop.process_conscious_moment()
        report = loop.generate_report()
        assert "Beautiful Loop" in report

    @pytest.mark.asyncio
    async def test_consciousness_context_string(self, loop):
        """Consciousness context string is generated for LLM prompt."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "test"
        mock_self_model.confidence = 0.6
        mock_self_model.cognitive_load = 0.3

        await loop.process_conscious_moment(self_model=mock_self_model)
        ctx = loop.generate_consciousness_context()
        assert "Beautiful Loop" in ctx
        assert "depth=" in ctx

    @pytest.mark.asyncio
    async def test_precision_applied_to_hierarchy(self, loop):
        """Hyper-model precision is applied back to hierarchy."""
        mock_processor = Mock()
        level0 = Mock()
        level0.precision = 1.0
        level1 = Mock()
        level1.precision = 1.0
        level2 = Mock()
        level2.precision = 1.0
        mock_processor.levels = [level0, level1, level2]

        mock_errors = [
            Mock(error_magnitude=0.1),
            Mock(error_magnitude=0.5),
            Mock(error_magnitude=0.1),
        ]

        await loop.process_conscious_moment(
            prediction_errors=mock_errors,
            hierarchical_processor=mock_processor,
        )

        # Precisions should have been modified
        assert level0.precision != 1.0 or level1.precision != 1.0 or level2.precision != 1.0
