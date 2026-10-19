"""
Tests for Beautiful Loop Theory Implementation

Tests the three components (HyperModel, BayesianBinding, EpistemicDepth)
and their integration via the BeautifulLoop class.

Based on: Laukkonen, Friston & Chandaria (2025) "A Beautiful Loop"
"""

import json
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock


from mtc.consciousness.beautiful_loop import (
    BeautifulLoop,
    ConsciousMoment,
    HyperModel,
    PrecisionState,
    BayesianBinding,
    Inference,
    EpistemicDepthTracker,
    SelfReferenceMetrics,
)


def _salience_winner(idx, salience, ctype="thought", content=None, metadata=None):
    """A workspace winner stand-in: a candidate with a summary and, when
    given, its own numeric content or a metadata embedding."""
    candidate = SimpleNamespace(
        id=f"w{idx}",
        summary=f"synthetic content {idx}",
        content_type=ctype,
        source_module="test",
    )
    if content is not None:
        candidate.content = content
    if metadata is not None:
        candidate.metadata = metadata
    return SimpleNamespace(candidate=candidate, salience=salience)


def _hub_round_trip(state: dict) -> dict:
    """Mimic the hub: JSON for scalars and histories, arrays kept as arrays."""
    return {
        "scalars": json.loads(json.dumps(state["scalars"])),
        "arrays": {k: np.asarray(v) for k, v in state["arrays"].items()},
        "histories": json.loads(json.dumps(state["histories"])),
    }


# ==========================================================================
# HyperModel Tests
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

    def test_has_observed_update_follows_the_meta_error(self, hyper_model):
        """A precision prediction alone is not an observed update: the flag
        turns on only when observe_meta_error has compared a prediction with
        the update that followed, which is also when recursive_confidence
        becomes a measurement; a restored record that holds predictions but
        no meta-error is unobserved."""
        assert hyper_model.has_observed_update is False
        hyper_model.predict_precision_update([0.2, 0.1, 0.3])
        assert hyper_model._prediction_count == 1
        assert hyper_model.has_observed_update is False
        assert hyper_model.recursive_confidence is None
        hyper_model.observe_meta_error()
        assert hyper_model.has_observed_update is True
        assert isinstance(hyper_model.recursive_confidence, float)

        restored = HyperModel(num_levels=3)
        restored.from_state_dict({"scalars": {"prediction_count": 3}})
        assert restored._prediction_count == 3
        assert restored.has_observed_update is False

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
        # Very low errors -> system should increase precision
        for _ in range(10):
            hyper_model.update_precision_beliefs([0.01, 0.01, 0.01])

        # Precision should increase (low error -> high reliability)
        for i in range(3):
            assert hyper_model.precision_beliefs[i] >= initial[i]

    def test_update_precision_beliefs_high_error(self, hyper_model):
        """High prediction errors decrease precision (less trust)."""
        # First build up some history with consistent, moderate errors
        for _ in range(5):
            hyper_model.update_precision_beliefs([0.5, 0.5, 0.5])
        mid_precision = hyper_model.precision_beliefs.copy()

        # Deterministically high AND variable errors -> variance-based precision drops.
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

        # Precision should decrease (noisy errors -> low reliability)
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
            hyper_model.update_precision_beliefs([0.01, 0.5, 0.01], context="social")
            hyper_model.update_precision_beliefs([0.5, 0.01, 0.5], context="analytical")

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
        """Global precision state returns dict of level -> precision."""
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
# BayesianBinding Tests
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
        """Empty candidates produce an empty percept whose binding quality
        is unmeasured; the legacy evidence mode reports its 0.0."""
        percept = binding.bind_inferences([])
        assert percept.binding_quality is None
        assert percept.coherence is None
        assert len(percept.bound_inferences) == 0
        legacy = BayesianBinding(binding_evidence="v04_salience_posterior")
        legacy_percept = legacy.bind_inferences([])
        assert legacy_percept.binding_quality == 0.0
        assert legacy_percept.coherence == 0.0

    def test_single_candidate(self, binding):
        """A single candidate has nothing to bind with: its percept carries
        the inference and no measured binding quality; the legacy evidence
        mode reports the v0.4.0 constant 0.5."""
        inf = self._make_inference("a", [0.5, 0.3, 0.2])
        percept = binding.bind_inferences([inf])
        assert len(percept.bound_inferences) == 1
        assert percept.coherence is None  # no pair, so nothing measured it
        assert percept.binding_quality is None
        assert binding.get_average_binding_quality() is None
        legacy = BayesianBinding(binding_evidence="v04_salience_posterior")
        legacy_percept = legacy.bind_inferences([inf])
        assert legacy_percept.binding_quality == 0.5
        assert legacy_percept.coherence == 1.0  # the v0.4.0 constant
        assert legacy.get_average_binding_quality() == 0.0

    def test_unknown_binding_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="binding_evidence"):
            BayesianBinding(binding_evidence="other")
        with pytest.raises(ValueError, match="binding_evidence"):
            BeautifulLoop(binding_evidence="other")

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
        """Incoherent inferences (opposite beliefs) bind with a lower
        quality than coherent ones."""
        coherent = binding.bind_inferences(
            [
                self._make_inference("a", [0.5, 0.3, 0.2], salience=0.8),
                self._make_inference("b", [0.48, 0.32, 0.20], salience=0.7),
            ]
        )
        infs = [
            self._make_inference("a", [1.0, 0.0, 0.0], salience=0.8),
            self._make_inference("b", [0.0, 0.0, 1.0], salience=0.7),
        ]
        percept = binding.bind_inferences(infs)
        # Still binds (only 2 candidates), but quality is lower
        assert len(percept.bound_inferences) == 2
        assert isinstance(percept.binding_quality, float)
        assert percept.binding_quality < coherent.binding_quality
        assert percept.binding_quality < 0.6 < coherent.binding_quality

    def test_unified_summary_separates_the_overflow_note(self, binding):
        """With more than three bound inferences the summary names three and
        counts the rest, with a space before the count."""
        infs = [
            self._make_inference("a", [0.5, 0.3, 0.2], salience=0.8),
            self._make_inference("b", [0.48, 0.32, 0.20], salience=0.7),
            self._make_inference("c", [0.51, 0.29, 0.20], salience=0.6),
            self._make_inference("d", [0.49, 0.31, 0.21], salience=0.5),
        ]
        percept = binding.bind_inferences(infs)
        assert len(percept.bound_inferences) == 4
        assert percept.unified_summary.endswith(" (+ 1 more)")
        names = [inf.content_summary for inf in percept.bound_inferences[:3]]
        assert percept.unified_summary == " + ".join(names) + " (+ 1 more)"

    def test_vectors_of_different_lengths_are_not_compared(self, binding):
        """Two belief vectors of different lengths carry no mutual
        information for the correlation proxy: the pair is not measured, a
        percept whose bound inferences have no comparable pair has no
        measured quality or coherence, and a comparable pair among the
        candidates is measured on its own while the odd-length candidate
        stays unbound. The legacy evidence mode keeps the v0.4.0 rule of
        truncating both vectors to their common prefix."""
        short = np.array([0.5, 0.3, 0.2])
        long = np.array([0.5, 0.3, 0.2, 0.9, 0.1, 0.4])
        assert binding._estimate_mutual_information(short, long) is None
        assert binding._estimate_mutual_information(long, short) is None
        same = binding._estimate_mutual_information(short, short * 0.9 + 0.01)
        assert isinstance(same, float)

        a = self._make_inference("a", [0.5, 0.3, 0.2], salience=0.8)
        b = self._make_inference("b", [0.48, 0.32, 0.20], salience=0.7)
        c = self._make_inference("c", [0.5, 0.3, 0.2, 0.9, 0.1, 0.4], salience=0.9)

        odd = binding.bind_inferences([a, c])
        assert odd.binding_quality is None
        assert odd.coherence is None
        assert len(odd.binding_pairs) == 1
        assert odd.binding_pairs[0].mutual_information is None
        assert odd.binding_pairs[0].coherence_score is None
        assert odd.binding_pairs[0].combined_uncertainty_reduction is None
        assert binding.get_average_binding_quality() is None

        pair = binding.bind_inferences([a, b])
        mixed = binding.bind_inferences([a, b, c])
        assert {inf.id for inf in mixed.bound_inferences} == {"a", "b"}
        assert mixed.binding_quality == pytest.approx(pair.binding_quality)
        assert mixed.coherence == pytest.approx(pair.coherence)
        assert binding.get_average_binding_quality() == pytest.approx(
            pair.binding_quality
        )

        legacy = BayesianBinding(binding_evidence="v04_salience_posterior")
        prefix = legacy._estimate_mutual_information(short, long)
        assert isinstance(prefix, float)
        assert prefix == legacy._estimate_mutual_information(short, long[:3])
        assert isinstance(legacy.bind_inferences([a, c]).binding_quality, float)

    def test_unmeasured_pair_round_trips_and_reports(self, binding):
        """A percept whose only pair was not comparable persists its None
        fields as None, survives the hub round trip, and reports
        insufficient_data for its coherence."""
        from mtc.consciousness.beautiful_loop.bayesian_binding import (
            percept_from_dict,
            percept_to_dict,
        )

        a = self._make_inference("a", [0.5, 0.3, 0.2], salience=0.8)
        c = self._make_inference("c", [0.5, 0.3, 0.2, 0.9, 0.1, 0.4], salience=0.9)
        percept = binding.bind_inferences([a, c])
        raw = json.loads(json.dumps(percept_to_dict(percept)))
        assert raw["coherence"] is None
        assert raw["binding_quality"] is None
        assert raw["binding_pairs"][0]["mutual_information"] is None
        restored = percept_from_dict(raw)
        assert restored.coherence is None
        assert restored.binding_quality is None
        assert restored.binding_pairs[0].mutual_information is None
        assert restored.binding_pairs[0].coherence_score is None
        assert restored.binding_pairs[0].combined_uncertainty_reduction is None
        assert "coherence=insufficient_data" in binding.generate_report()

        fresh = BayesianBinding()
        fresh.from_state_dict(_hub_round_trip(binding.to_state_dict()))
        latest = fresh._percept_history[-1]
        assert latest.coherence is None
        assert latest.binding_pairs[0].mutual_information is None
        assert fresh.get_average_binding_quality() is None

    def test_empty_and_single_percepts_have_no_measured_coherence(self, binding):
        """Coherence is a measure over pairs of bound inferences. The empty
        percept and the single-inference percept have no pair, so under the
        default evidence mode nothing measured their coherence: it is None,
        is written as null and restores as None, exactly like their binding
        quality. The legacy evidence mode keeps the v0.4.0 constants (0.0
        for the empty percept, 1.0 for the single one) in the same places."""
        from mtc.consciousness.beautiful_loop.bayesian_binding import (
            percept_from_dict,
            percept_to_dict,
        )

        inf = self._make_inference("a", [0.5, 0.3, 0.2])
        for percept in (binding.bind_inferences([]), binding.bind_inferences([inf])):
            assert percept.binding_quality is None
            assert percept.coherence is None
            raw = json.loads(json.dumps(percept_to_dict(percept)))
            assert raw["coherence"] is None
            assert percept_from_dict(raw).coherence is None

        legacy = BayesianBinding(binding_evidence="v04_salience_posterior")
        empty = legacy.bind_inferences([])
        single = legacy.bind_inferences([inf])
        assert (empty.coherence, single.coherence) == (0.0, 1.0)
        for percept, constant in ((empty, 0.0), (single, 1.0)):
            raw = json.loads(json.dumps(percept_to_dict(percept)))
            assert raw["coherence"] == constant
            assert percept_from_dict(raw).coherence == constant

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
        # More evidence -> less uncertainty
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
# EpistemicDepthTracker Tests
# ==========================================================================


class TestEpistemicDepthTracker:
    """Tests for recursive self-reference measurement."""

    @pytest.fixture
    def tracker(self):
        return EpistemicDepthTracker(max_depth=5)

    def test_no_modules_depth_zero(self, tracker):
        """No modules -> depth 0."""
        depth = tracker.measure_depth()
        assert depth == 0

    def test_unmeasured_self_reference_is_none(self, tracker):
        """With nothing to measure, coherence, meta-prediction error and
        calibration are None rather than a 0.5 stand-in; a meta state with
        both accuracies measures the two meta values; coherence needs three
        self-model snapshots; a round trip keeps None."""
        from mtc.consciousness.beautiful_loop.epistemic_depth import (
            metrics_from_dict,
            metrics_to_dict,
        )

        metrics = tracker.track_self_reference()
        assert metrics.self_model_coherence is None
        assert metrics.meta_prediction_error is None
        assert metrics.calibration_quality is None
        assert metrics.prediction_about_prediction is False
        report = tracker.generate_report()
        assert "Coherence: insufficient_data" in report
        assert "Calibration: insufficient_data" in report

        restored = metrics_from_dict(metrics_to_dict(metrics))
        assert restored.self_model_coherence is None
        assert restored.meta_prediction_error is None
        assert restored.calibration_quality is None
        assert metrics_from_dict({}).calibration_quality is None

        meta = SimpleNamespace(
            total_predictions=4, predicted_accuracy=0.7, prediction_accuracy=0.6
        )
        measured = tracker.track_self_reference(meta_state=meta)
        assert measured.meta_prediction_error == pytest.approx(0.1)
        assert measured.calibration_quality == pytest.approx(0.9)
        assert measured.prediction_about_prediction is True
        assert measured.self_model_coherence is None  # fewer than 3 snapshots

        sm = Mock()
        sm.update_count = 3
        sm.current_focus = "synthetic"
        sm.confidence = 0.6
        sm.cognitive_load = 0.3
        for _ in range(2):
            tracker.measure_depth(self_model=sm)
        assert tracker.track_self_reference(self_model=sm).self_model_coherence is None
        tracker.measure_depth(self_model=sm)
        coherent = tracker.track_self_reference(self_model=sm)
        assert coherent.self_model_coherence == pytest.approx(1.0)  # identical
        assert "Coherence: 1.000" in tracker.generate_report()

    def test_self_model_depth_one(self, tracker):
        """Active self-model -> depth 1."""
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "conversation"
        mock_self_model.confidence = 0.7
        mock_self_model.cognitive_load = 0.3

        depth = tracker.measure_depth(self_model=mock_self_model)
        assert depth >= 1

    def test_meta_state_depth_two(self, tracker):
        """Active self-model + meta predictions -> depth 2."""
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
        """HOTs about self-model -> depth 3."""
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
        """Attention focused on self-model -> depth 4."""
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

    def test_strange_loop_detection(self):
        """Legacy rule: a strange loop when the self-model's confidence and
        the meta-state's confidence align."""
        tracker = EpistemicDepthTracker(
            max_depth=5, strange_loop_test="v04_confidence_match"
        )
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

    def test_no_strange_loop_when_divergent(self):
        """Legacy rule: no strange loop when the two confidences diverge."""
        tracker = EpistemicDepthTracker(
            max_depth=5, strange_loop_test="v04_confidence_match"
        )
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

    def test_unknown_strange_loop_test_is_rejected(self):
        with pytest.raises(ValueError, match="strange_loop_test"):
            EpistemicDepthTracker(strange_loop_test="other")
        with pytest.raises(ValueError, match="strange_loop_test"):
            BeautifulLoop(strange_loop_test="other")

    def test_loop_forwards_the_strange_loop_test(self):
        assert EpistemicDepthTracker().strange_loop_test == "independent_calibration"
        assert BeautifulLoop().epistemic_depth.strange_loop_test == (
            "independent_calibration"
        )
        legacy = BeautifulLoop(strange_loop_test="v04_confidence_match")
        assert legacy.epistemic_depth.strange_loop_test == "v04_confidence_match"

    @staticmethod
    def _calibration_inputs(calibration, accuracy, copied_confidence=0.42):
        sm = Mock()
        sm.self_calibration_score = calibration
        sm.prediction_accuracy_history = [0.5, 0.6, 0.65, 0.7]
        sm.model_confidence = copied_confidence
        ms = Mock()
        ms.total_predictions = 10
        ms.predicted_accuracy = 0.5
        ms.prediction_accuracy = accuracy
        ms.model_confidence = copied_confidence
        return sm, ms

    def test_copied_confidence_is_not_a_strange_loop(self):
        """The workspace copies the meta-state's confidence into the
        self-model, so agreement between the two is a value agreeing with
        its own copy. The default test never reads the copied field and
        compares the self-model's scored self-prediction accuracy with the
        inference ledger's forecast accuracy; the legacy test accepts the
        copy."""
        sm, ms = self._calibration_inputs(calibration=0.9, accuracy=0.2)
        default = EpistemicDepthTracker().track_self_reference(
            self_model=sm, meta_state=ms, recursive_observed=True
        )
        assert default.strange_loop_detected is False
        assert default.strange_loop_basis == "independent_calibration"

        sm, ms = self._calibration_inputs(calibration=0.9, accuracy=0.2)
        legacy = EpistemicDepthTracker(
            strange_loop_test="v04_confidence_match"
        ).track_self_reference(self_model=sm, meta_state=ms)
        assert legacy.strange_loop_detected is True
        assert legacy.strange_loop_basis == "v04_confidence_match"

    def test_independent_calibration_agreement_detects_loop(self):
        """Two independently measured calibrations that agree, with an
        observed recursive update, make a strange loop; without the observed
        update they do not, and too little history is insufficient data."""
        sm, ms = self._calibration_inputs(calibration=0.6, accuracy=0.62)
        observed = EpistemicDepthTracker().track_self_reference(
            self_model=sm, meta_state=ms, recursive_observed=True
        )
        assert observed.strange_loop_detected is True
        assert observed.strange_loop_basis == "independent_calibration"

        sm, ms = self._calibration_inputs(calibration=0.6, accuracy=0.62)
        unobserved = EpistemicDepthTracker().track_self_reference(
            self_model=sm, meta_state=ms, recursive_observed=False
        )
        assert unobserved.strange_loop_detected is False
        assert unobserved.strange_loop_basis == "independent_calibration"

        # Standalone use: no recursive_observed argument skips that check
        sm, ms = self._calibration_inputs(calibration=0.6, accuracy=0.62)
        standalone = EpistemicDepthTracker().track_self_reference(
            self_model=sm, meta_state=ms
        )
        assert standalone.strange_loop_detected is True

        sm, ms = self._calibration_inputs(calibration=0.6, accuracy=0.62)
        sm.prediction_accuracy_history = [0.6]
        short = EpistemicDepthTracker().track_self_reference(
            self_model=sm, meta_state=ms, recursive_observed=True
        )
        assert short.strange_loop_detected is False
        assert short.strange_loop_basis == "insufficient_data"

    def test_meta_state_without_model_confidence_does_not_raise(self):
        """A meta state without the copied confidence field raises nothing
        in either mode; the legacy rule then finds no loop."""
        for mode in ("independent_calibration", "v04_confidence_match"):
            sm = Mock()
            sm.self_calibration_score = 0.5
            sm.prediction_accuracy_history = [0.5, 0.5, 0.5]
            sm.model_confidence = 0.4
            ms = SimpleNamespace(
                total_predictions=10, predicted_accuracy=0.5, prediction_accuracy=0.5
            )
            metrics = EpistemicDepthTracker(
                strange_loop_test=mode
            ).track_self_reference(
                self_model=sm, meta_state=ms, recursive_observed=True
            )
            assert isinstance(metrics, SelfReferenceMetrics)
            if mode == "v04_confidence_match":
                assert metrics.strange_loop_detected is False
            else:
                assert metrics.strange_loop_detected is True  # 0.5 agrees with 0.5

    def test_strange_loop_threshold_looseness_is_as_documented(self):
        """The default threshold 0.5 on two quantities in [0, 1] is loose,
        as the module documents: the self-calibration score starts at 0.5,
        so at that start every ledger accuracy strictly inside (0, 1) agrees
        with it and only the history minimums hold the verdict back; the
        two ends of the range sit exactly at the threshold and do not."""
        tracker = EpistemicDepthTracker()
        assert tracker.strange_loop_threshold == 0.5
        for accuracy in (0.01, 0.5, 0.99):
            sm, ms = self._calibration_inputs(calibration=0.5, accuracy=accuracy)
            metrics = tracker.track_self_reference(
                self_model=sm, meta_state=ms, recursive_observed=True
            )
            assert metrics.strange_loop_detected is True
            assert metrics.strange_loop_basis == "independent_calibration"
        for accuracy in (0.0, 1.0):
            sm, ms = self._calibration_inputs(calibration=0.5, accuracy=accuracy)
            metrics = tracker.track_self_reference(
                self_model=sm, meta_state=ms, recursive_observed=True
            )
            assert metrics.strange_loop_detected is False
            assert metrics.strange_loop_basis == "independent_calibration"

    def test_first_check_agreement_band_covers_three_quarters(self):
        """The module docstring states how loose the default threshold is
        at the first check that can run. The check needs three entries in
        the self-prediction accuracy history; metacognition appends one per
        scored prediction once three are scored and moves the calibration
        a fifth of the way toward each, so by then the score has taken
        three steps from 0.5 and lies between 0.256 and 0.744. The band
        (calibration plus or minus 0.5) then covers at least three quarters
        of [0, 1] and no more is claimed: at the upper extreme an accuracy
        of 0.2, well inside (0, 1), does not agree."""
        from mtc.consciousness.metacognition import MetacognitionModule

        extremes = {}
        for accurate in (True, False):
            module = MetacognitionModule(narrate_self_model=False)
            ids = [module.predict_self(f"context {i}").id for i in range(5)]
            for prediction_id in ids:
                module.score_prediction(prediction_id, accurate)
            assert len(module.self_model.prediction_accuracy_history) == 3
            extremes[accurate] = module.self_model
        assert extremes[True].self_calibration_score == pytest.approx(0.744)
        assert extremes[False].self_calibration_score == pytest.approx(0.256)

        tracker = EpistemicDepthTracker()
        for self_model in extremes.values():
            calibration = self_model.self_calibration_score
            low = max(0.0, calibration - tracker.strange_loop_threshold)
            high = min(1.0, calibration + tracker.strange_loop_threshold)
            assert high - low >= 0.75
            assert high - low < 1.0  # not every accuracy agrees any more
        verdicts = {}
        for key, accuracy in ((True, 0.2), (True, 0.3), (False, 0.8), (False, 0.7)):
            ms = Mock()
            ms.total_predictions = 10
            ms.predicted_accuracy = 0.5
            ms.prediction_accuracy = accuracy
            ms.model_confidence = 0.42
            metrics = tracker.track_self_reference(
                self_model=extremes[key], meta_state=ms, recursive_observed=True
            )
            assert metrics.strange_loop_basis == "independent_calibration"
            verdicts[(key, accuracy)] = metrics.strange_loop_detected
        assert verdicts == {
            (True, 0.2): False,
            (True, 0.3): True,
            (False, 0.8): False,
            (False, 0.7): True,
        }

    def test_strange_loop_basis_round_trips(self):
        from mtc.consciousness.beautiful_loop.epistemic_depth import (
            metrics_from_dict,
            metrics_to_dict,
        )

        metrics = EpistemicDepthTracker().track_self_reference()
        assert metrics.strange_loop_detected is False
        assert metrics.strange_loop_basis == "insufficient_data"
        restored = metrics_from_dict(metrics_to_dict(metrics))
        assert restored.strange_loop_basis == "insufficient_data"
        # A record written before the field existed does not claim a basis
        assert metrics_from_dict({}).strange_loop_basis == "unknown"

    def test_depth_measurement_basis_names_the_keyword_heuristic(self, tracker):
        """A depth reached through the keyword lists (3 or 4) is labelled
        keyword_heuristic; a depth read from module state is module_state."""
        sm = Mock()
        sm.update_count = 5
        sm.current_focus = "reflection"
        sm.confidence = 0.7
        sm.cognitive_load = 0.3
        ms = Mock()
        ms.total_predictions = 10
        ms.predicted_accuracy = 0.7
        ms.prediction_accuracy = 0.65
        ms.model_confidence = 0.6
        hot = Mock()
        hot.content = "a thought about my own prediction accuracy"
        ast = Mock()
        ast.current_focus = Mock()
        ast.current_focus.summary = "attending to the internal self-model"

        assert tracker.measure_depth() == 0
        assert tracker._depth_history[-1].measurement_basis == "module_state"
        assert tracker.measure_depth(self_model=sm, meta_state=ms) == 2
        assert tracker._depth_history[-1].measurement_basis == "module_state"
        depth = tracker.measure_depth(
            self_model=sm, meta_state=ms, higher_order_thoughts=[hot]
        )
        assert depth == 3
        assert tracker._depth_history[-1].measurement_basis == "keyword_heuristic"
        depth = tracker.measure_depth(
            self_model=sm,
            meta_state=ms,
            higher_order_thoughts=[hot],
            attention_schema_state=ast,
        )
        assert depth == 4
        assert tracker._depth_history[-1].measurement_basis == "keyword_heuristic"

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
# BeautifulLoop Integration Tests
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
        assert (
            moment.precision_state.level_precisions[1]
            > moment.precision_state.level_precisions[2]
        )

    @pytest.mark.asyncio
    async def test_process_with_workspace_winners(self, loop):
        """Workspace winners that carry their own numeric content are
        converted to inferences and bound; a winner whose content is not a
        numeric vector is left out rather than given a fabricated one."""
        mock_candidate = Mock()
        mock_candidate.id = "test_1"
        mock_candidate.summary = "A thought about consciousness"
        mock_candidate.content_type = "thought"
        mock_candidate.source_module = "ctm"
        mock_candidate.content = np.array([0.3, 0.2, 0.1, 0.15])

        mock_winner1 = Mock()
        mock_winner1.candidate = mock_candidate
        mock_winner1.salience = 0.8

        mock_candidate2 = Mock()
        mock_candidate2.id = "test_2"
        mock_candidate2.summary = "An emotion of curiosity"
        mock_candidate2.content_type = "emotion"
        mock_candidate2.source_module = "homeostatic"
        mock_candidate2.content = np.array([0.28, 0.21, 0.1, 0.16])

        mock_winner2 = Mock()
        mock_winner2.candidate = mock_candidate2
        mock_winner2.salience = 0.6

        # A Mock's attributes are Mocks: neither a numeric content nor a
        # metadata mapping, so this winner carries no vector
        mock_winner3 = Mock()
        mock_winner3.candidate = Mock()
        mock_winner3.salience = 0.7

        mock_result = Mock()
        mock_result.posterior_beliefs = np.array(
            [0.3, 0.2, 0.1, 0.15, 0.05, 0.05, 0.1, 0.05]
        )

        moment = await loop.process_conscious_moment(
            workspace_winners=[mock_winner1, mock_winner2, mock_winner3],
            inference_result=mock_result,
        )

        assert isinstance(moment.binding_quality, float)
        assert 0.0 <= moment.binding_quality <= 1.0
        assert len(moment.bound_percept.bound_inferences) == 2
        assert {inf.id for inf in moment.bound_percept.bound_inferences} == {
            "test_1",
            "test_2",
        }

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
        mock_errors = [
            Mock(error_magnitude=0.2),
            Mock(error_magnitude=0.1),
            Mock(error_magnitude=0.3),
        ]

        mock_candidate = Mock()
        mock_candidate.id = "w1"
        mock_candidate.summary = "Test thought"
        mock_candidate.content_type = "thought"
        mock_candidate.source_module = "test"
        mock_winner = Mock()
        mock_winner.candidate = mock_candidate
        mock_winner.salience = 0.7

        mock_result = Mock()
        mock_result.posterior_beliefs = np.array(
            [0.3, 0.2, 0.1, 0.15, 0.05, 0.05, 0.1, 0.05]
        )

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
    async def test_loop_passes_its_recursive_observation_to_the_strange_loop_test(
        self,
    ):
        """Under the default strange-loop test the loop reports whether the
        hyper-model has observed one of its own precision updates: before
        the first update the agreeing calibrations are not a strange loop,
        after it they are."""
        loop = BeautifulLoop()
        sm = Mock()
        sm.update_count = 5
        sm.current_focus = "synthetic"
        sm.confidence = 0.6
        sm.cognitive_load = 0.3
        sm.self_calibration_score = 0.6
        sm.prediction_accuracy_history = [0.55, 0.6, 0.62]
        sm.model_confidence = 0.1
        ms = Mock()
        ms.total_predictions = 10
        ms.predicted_accuracy = 0.6
        ms.prediction_accuracy = 0.62
        ms.model_confidence = 0.9

        before = await loop.process_conscious_moment(self_model=sm, meta_state=ms)
        assert before.self_reference.strange_loop_detected is False

        after = await loop.process_conscious_moment(
            prediction_errors=[0.2, 0.1, 0.3], self_model=sm, meta_state=ms
        )
        assert after.self_reference.strange_loop_detected is True
        assert after.self_reference.strange_loop_basis == "independent_calibration"

    @pytest.mark.asyncio
    async def test_predicted_but_unobserved_update_is_not_a_recursive_observation(
        self,
    ):
        """The loop reports an observed recursive update only once the
        hyper-model has compared a precision prediction with the update that
        followed (the flag behind recursive_confidence): a prediction made
        and never observed, or a restored record with predictions and no
        meta-error, is not an observation, so agreeing calibrations are no
        strange loop yet."""

        def inputs():
            sm = Mock()
            sm.update_count = 5
            sm.current_focus = "synthetic"
            sm.confidence = 0.6
            sm.cognitive_load = 0.3
            sm.self_calibration_score = 0.6
            sm.prediction_accuracy_history = [0.55, 0.6, 0.62]
            sm.model_confidence = 0.1
            ms = Mock()
            ms.total_predictions = 10
            ms.predicted_accuracy = 0.6
            ms.prediction_accuracy = 0.62
            ms.model_confidence = 0.9
            return sm, ms

        loop = BeautifulLoop()
        loop.hyper_model.predict_precision_update([0.2, 0.1, 0.3])
        assert loop.hyper_model._prediction_count == 1
        assert loop.hyper_model.recursive_confidence is None
        sm, ms = inputs()
        predicted = await loop.process_conscious_moment(self_model=sm, meta_state=ms)
        assert predicted.self_reference.strange_loop_detected is False
        assert predicted.self_reference.strange_loop_basis == "independent_calibration"

        loop.hyper_model.observe_meta_error()
        sm, ms = inputs()
        observed = await loop.process_conscious_moment(self_model=sm, meta_state=ms)
        assert observed.self_reference.strange_loop_detected is True

        restored = BeautifulLoop()
        restored.hyper_model.from_state_dict({"scalars": {"prediction_count": 4}})
        sm, ms = inputs()
        moment = await restored.process_conscious_moment(self_model=sm, meta_state=ms)
        assert moment.self_reference.strange_loop_detected is False

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
        assert (
            level0.precision != 1.0
            or level1.precision != 1.0
            or level2.precision != 1.0
        )


class TestBindingEvidence:
    """The binding_evidence switch: candidate_vectors by default, the
    v0.4.0 salience and posterior construction under the legacy value."""

    @pytest.mark.asyncio
    async def test_salience_only_winners_have_no_measured_binding(self):
        """Winners that carry nothing but a salience have no belief vector
        of their own: nothing is bound, the moment and the average report no
        binding quality, and no random number is drawn."""
        posterior = SimpleNamespace(posterior_beliefs=np.array([0.1, 0.2, 0.3, 0.4]))
        types = ["thought", "emotion", "memory", "perception"]
        np.random.seed(0)
        untouched_draw = float(np.random.rand())
        np.random.seed(0)
        for count in (2, 3, 4, 5):
            loop = BeautifulLoop()
            winners = [
                _salience_winner(i, 0.2 + 0.15 * i, types[i % 4]) for i in range(count)
            ]
            moment = await loop.process_conscious_moment(
                workspace_winners=winners, inference_result=posterior
            )
            assert moment.binding_quality is None
            assert moment.bound_percept.binding_quality is None
            assert moment.bound_percept.bound_inferences == []
            assert loop.bayesian_binding.get_average_binding_quality() is None
            assert loop.get_loop_statistics()["average_binding_quality"] is None
            assert isinstance(moment.loop_quality, float)
        assert float(np.random.rand()) == untouched_draw

    @pytest.mark.asyncio
    async def test_v04_binding_evidence_reproduces_current_value(self):
        """Under v04_salience_posterior the pre-change arithmetic runs byte
        for byte: the salience belief vectors, the shared posterior with its
        randn noise drawn from the global stream, the 0.5 single percept and
        the 0.0 empty average. The values were captured from the tree before
        the change (the binding values move only in the sixth decimal with
        the inference timestamps); the draw that follows the moments pins
        the random stream exactly."""
        winners_a = [
            _salience_winner(1, 0.9, "thought"),
            _salience_winner(2, 0.3, "emotion"),
            _salience_winner(3, 0.6, "memory"),
        ]
        winners_b = [_salience_winner(4, 0.5), _salience_winner(5, 0.8)]
        posterior = SimpleNamespace(posterior_beliefs=np.array([0.1, 0.2, 0.3, 0.4]))

        def legacy():
            return BeautifulLoop(
                binding_evidence="v04_salience_posterior",
                loop_quality_formula="v04",
                precision_floor_enabled=False,
                require_recursive_confidence=False,
            )

        np.random.seed(0)
        plain = await legacy().process_conscious_moment(workspace_winners=winners_a)
        assert plain.binding_quality == pytest.approx(0.926190835585, abs=1e-4)
        assert plain.loop_quality == pytest.approx(0.4 * plain.binding_quality)

        np.random.seed(0)
        loop = legacy()
        first = await loop.process_conscious_moment(
            workspace_winners=winners_a, inference_result=posterior
        )
        second = await loop.process_conscious_moment(
            workspace_winners=winners_b, inference_result=posterior
        )
        assert first.binding_quality == pytest.approx(0.949138976964, abs=1e-4)
        assert second.binding_quality == pytest.approx(0.999804553317, abs=1e-4)
        average = loop.bayesian_binding.get_average_binding_quality()
        assert average == pytest.approx(0.974471765140, abs=1e-4)
        assert loop.get_loop_statistics()["average_binding_quality"] == average
        # The two moments consumed exactly the v0.4.0 draws
        assert float(np.random.rand()) == 0.1433532874090464

        np.random.seed(0)
        default_ctor = BeautifulLoop(binding_evidence="v04_salience_posterior")
        moment = await default_ctor.process_conscious_moment(
            workspace_winners=winners_a, inference_result=posterior
        )
        assert moment.binding_quality == pytest.approx(0.949139382270, abs=1e-4)
        assert float(np.random.rand()) == 0.5680445610939323

        single = BeautifulLoop(binding_evidence="v04_salience_posterior")
        moment = await single.process_conscious_moment(
            workspace_winners=[_salience_winner(9, 0.9)]
        )
        assert moment.binding_quality == 0.5
        assert single.bayesian_binding.get_average_binding_quality() == 0.0
        assert single.get_loop_statistics()["average_binding_quality"] == 0.0
        empty = BeautifulLoop(binding_evidence="v04_salience_posterior")
        assert (await empty.process_conscious_moment()).binding_quality == 0.0

    @pytest.mark.asyncio
    async def test_candidate_vectors_bind_by_content(self):
        """Winners bind on the vectors they carry: near-identical content
        vectors bind higher than orthogonal ones, a metadata embedding
        serves when the content is text, and a vector shorter than three
        entries is not evidence."""
        base = np.array([0.5, 0.3, 0.2, 0.1, 0.05])
        alike = [
            _salience_winner(1, 0.8, content=base),
            _salience_winner(2, 0.7, content=base * 1.01 + 0.001),
            _salience_winner(3, 0.6, content=list(base * 0.99)),
        ]
        orthogonal = [
            _salience_winner(1, 0.8, content=np.array([1.0, 0.0, 0.0, 0.0, 0.0])),
            _salience_winner(2, 0.7, content=np.array([0.0, 0.0, 0.0, 0.0, 1.0])),
            _salience_winner(3, 0.6, content=np.array([0.0, 0.0, 1.0, 0.0, 0.0])),
        ]
        close = await BeautifulLoop().process_conscious_moment(workspace_winners=alike)
        far = await BeautifulLoop().process_conscious_moment(
            workspace_winners=orthogonal
        )
        assert isinstance(close.binding_quality, float)
        assert isinstance(far.binding_quality, float)
        assert close.binding_quality > far.binding_quality
        assert len(close.bound_percept.bound_inferences) == 3

        embedded = _salience_winner(
            4, 0.5, content="a sentence", metadata={"embedding": list(base)}
        )
        moment = await BeautifulLoop().process_conscious_moment(
            workspace_winners=[alike[0], embedded]
        )
        assert isinstance(moment.binding_quality, float)
        assert {inf.id for inf in moment.bound_percept.bound_inferences} == {
            "w1",
            "w4",
        }
        np.testing.assert_allclose(
            moment.bound_percept.bound_inferences[-1].belief_vector, base
        )

        short = _salience_winner(5, 0.9, content=np.array([0.5, 0.5]))
        text_only = _salience_winner(6, 0.9, content="a sentence")
        moment = await BeautifulLoop().process_conscious_moment(
            workspace_winners=[alike[0], short, text_only]
        )
        assert moment.binding_quality is None
        assert [inf.id for inf in moment.bound_percept.bound_inferences] == ["w1"]

    @pytest.mark.asyncio
    async def test_content_vector_and_embedding_of_different_lengths_are_not_bound(
        self,
    ):
        """A winner with a three-entry content vector and a winner with a
        longer metadata embedding carry vectors that cannot be compared: the
        loop measures no binding between them and reports no binding
        quality, instead of correlating the content with the embedding's
        first three entries. A third winner with a comparable vector is
        bound with the first on its own and the embedding winner stays
        unbound."""
        embedding = np.random.default_rng(7).random(384).tolist()
        winners = [
            _salience_winner(1, 0.8, content=[0.5, 0.3, 0.2]),
            _salience_winner(2, 0.7, metadata={"embedding": embedding}),
        ]
        loop = BeautifulLoop()
        moment = await loop.process_conscious_moment(workspace_winners=winners)
        assert moment.binding_quality is None
        assert moment.bound_percept.binding_quality is None
        assert loop.bayesian_binding.get_average_binding_quality() is None
        assert isinstance(moment.loop_quality, float)

        winners.append(_salience_winner(3, 0.6, content=[0.48, 0.32, 0.2]))
        moment = await loop.process_conscious_moment(workspace_winners=winners)
        assert {inf.id for inf in moment.bound_percept.bound_inferences} == {
            "w1",
            "w3",
        }
        assert isinstance(moment.binding_quality, float)
        assert moment.binding_quality > 0.5
        assert loop.bayesian_binding.get_average_binding_quality() == pytest.approx(
            moment.binding_quality
        )

    def test_v04_loop_formula_requires_v04_binding(self):
        """The v04 loop formula reads a binding quality that candidate
        vectors leave unmeasured, so the pair is refused at construction,
        naming both keys and the fix."""
        with pytest.raises(ValueError) as excinfo:
            BeautifulLoop(
                loop_quality_formula="v04", binding_evidence="candidate_vectors"
            )
        message = str(excinfo.value)
        assert "loop_quality_formula" in message
        assert "binding_evidence" in message
        assert "v04_salience_posterior" in message
        with pytest.raises(ValueError, match="binding_evidence"):
            BeautifulLoop(loop_quality_formula="v04")
        paired = BeautifulLoop(
            loop_quality_formula="v04", binding_evidence="v04_salience_posterior"
        )
        assert paired.binding_evidence == "v04_salience_posterior"
        assert paired.bayesian_binding.binding_evidence == "v04_salience_posterior"

    @pytest.mark.asyncio
    async def test_moment_with_no_binding_round_trips(self):
        """Winners without a vector bind nothing: the moment's percept
        carries no measured binding quality and no measured coherence, both
        persist as null in the moment history and restore as None."""
        loop = BeautifulLoop()
        unbound = await loop.process_conscious_moment(
            workspace_winners=[_salience_winner(1, 0.9), _salience_winner(2, 0.4)]
        )
        assert unbound.binding_quality is None
        assert unbound.bound_percept.coherence is None
        state = _hub_round_trip(loop.to_state_dict())
        latest = state["histories"]["moment_history"][-1]
        assert latest["binding_quality"] is None
        assert latest["bound_percept"]["binding_quality"] is None
        assert latest["bound_percept"]["coherence"] is None

        restored = BeautifulLoop()
        restored.from_state_dict(state)
        moment = restored._moment_history[-1]
        assert isinstance(moment, ConsciousMoment)
        assert moment.binding_quality is None
        assert moment.bound_percept.binding_quality is None
        assert moment.bound_percept.coherence is None
        assert restored.get_loop_statistics()["average_binding_quality"] is None
        assert "binding=not measured" in restored.generate_consciousness_context()

    @pytest.mark.asyncio
    async def test_prior_binding_quality_keeps_last_measured_value(self):
        loop = BeautifulLoop()
        base = np.array([0.5, 0.3, 0.2, 0.1])
        measured = await loop.process_conscious_moment(
            workspace_winners=[
                _salience_winner(1, 0.8, content=base),
                _salience_winner(2, 0.7, content=base * 1.02),
            ]
        )
        assert isinstance(measured.binding_quality, float)
        assert loop._prior_binding_quality == pytest.approx(measured.binding_quality)

        unmeasured = await loop.process_conscious_moment(
            workspace_winners=[_salience_winner(3, 0.6)]
        )
        assert unmeasured.binding_quality is None
        assert loop._prior_binding_quality == pytest.approx(measured.binding_quality)


class TestBeautifulLoopNarrationLabels:
    """The prompt context names its proxies and asserts no experience."""

    @staticmethod
    def _field_evidencing_inputs():
        sm = Mock()
        sm.update_count = 5
        sm.current_focus = "reflection"
        sm.confidence = 0.7
        sm.cognitive_load = 0.3
        sm.self_calibration_score = 0.7
        sm.prediction_accuracy_history = [0.5, 0.6, 0.65]
        sm.model_confidence = 0.6
        ms = Mock()
        ms.total_predictions = 10
        ms.predicted_accuracy = 0.7
        ms.prediction_accuracy = 0.65
        ms.model_confidence = 0.6
        return sm, ms

    @pytest.mark.asyncio
    async def test_consciousness_context_is_labelled_proxy(self):
        loop = BeautifulLoop(
            field_evidencing_threshold=0.0,
            strange_loop_test="v04_confidence_match",
            require_recursive_confidence=False,
        )
        sm, ms = self._field_evidencing_inputs()
        moment = await loop.process_conscious_moment(self_model=sm, meta_state=ms)
        assert moment.is_field_evidencing is True

        ctx = loop.generate_consciousness_context()
        assert "[Beautiful Loop: depth=2" in ctx
        assert (
            "[Field-evidencing proxy met: loop quality above threshold, "
            "depth >= 2, strange-loop proxy true]"
        ) in ctx
        assert "proxy" in ctx
        assert "genuinely" not in ctx
        assert " I " not in f" {ctx} "
        assert "I'm" not in ctx and "I am" not in ctx

    @pytest.mark.asyncio
    async def test_depth_labels_describe_the_measurement(self):
        loop = BeautifulLoop()
        await loop.process_conscious_moment()
        empty = loop.generate_consciousness_context()
        assert "depth=0" in empty
        assert " I " not in f" {empty} " and "I'm" not in empty
        assert "[Field-evidencing" not in empty

        sm, ms = self._field_evidencing_inputs()
        hot = Mock()
        hot.content = "a thought about my own prediction accuracy"
        await loop.process_conscious_moment(
            self_model=sm, meta_state=ms, higher_order_thoughts=[hot]
        )
        deep = loop.generate_consciousness_context()
        assert "depth=3" in deep
        assert "keyword" in deep
        assert " I " not in f" {deep} " and "I'm" not in deep


class TestBeautifulLoopNarrationGate:
    """generate_consciousness_context is gated; computation is not."""

    @pytest.mark.asyncio
    async def test_context_empty_when_gated(self):
        loop = BeautifulLoop(narrate_self_model=False)
        mock_self_model = Mock()
        mock_self_model.update_count = 5
        mock_self_model.current_focus = "test"
        mock_self_model.confidence = 0.6
        mock_self_model.cognitive_load = 0.3

        moment = await loop.process_conscious_moment(self_model=mock_self_model)
        assert loop.generate_consciousness_context() == ""
        assert moment.epistemic_depth >= 1
        assert loop.cycle_count == 1

    def test_binding_level_reexported(self):
        from mtc.consciousness.beautiful_loop import BindingLevel

        assert BindingLevel.GLOBAL.value == "global"
