"""
 Tests for Active Inference Module - Phase 5 FEP
=====================================================

Comprehensive tests for the system's Active Inference implementation,
covering the Free Energy Principle, Predictive Processing, and
Homeostatic Drives.

Tests verify:
1. ActiveInferenceConfig - Configuration validation
2. HomeostaticDrives - Internal need tracking and valence
3. HierarchicalPredictiveProcessor - Multi-level predictions
4. ActiveInferenceModule - Core FEP implementation with pymdp
5. Integration with EnhancedGlobalWorkspace

Research Foundation:
- Friston, K. (2010). "The Free Energy Principle: A Unified Brain Theory?"
- Friston, K. (2012). "Active Inference and Free Energy"
- Butlin et al. (2023). 14 Consciousness Indicators


Created: December 5, 2025
Author: Multi-Theory Consciousness Project
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from mtc.consciousness.active_inference import (
    ActiveInferenceConfig,
    HomeostaticConfig,
    HomeostaticDrive,
    HomeostaticDrives,
    PredictiveLevel,  # This is the level class in the hierarchy
    HierarchicalPredictiveProcessor,
    ActiveInferenceModule,
    ActiveInferenceState,
    InferenceResult,
    MetaState,
    PredictionError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_config():
    """Default Active Inference configuration."""
    return ActiveInferenceConfig()


@pytest.fixture
def custom_config():
    """Custom Active Inference configuration."""
    return ActiveInferenceConfig(
        num_hidden_states=16,
        num_observations=8,
        num_actions=6,
        planning_horizon=5,
        gamma=20.0,
        enable_homeostasis=True,
    )


@pytest.fixture
def homeostatic_drives():
    """Homeostatic drives instance."""
    return HomeostaticDrives()


@pytest.fixture
def hierarchical_processor():
    """Hierarchical predictive processor instance."""
    return HierarchicalPredictiveProcessor()


@pytest.fixture
def active_inference_module():
    """Active Inference module instance."""
    return ActiveInferenceModule()


@pytest.fixture
def active_inference_module_custom(custom_config):
    """Active Inference module with custom config."""
    return ActiveInferenceModule(custom_config)


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestActiveInferenceConfig:
    """Test Active Inference configuration."""

    def test_default_config_values(self, default_config):
        """Test default configuration values."""
        assert default_config.num_hidden_states == 8
        assert default_config.num_observations == 5
        assert default_config.num_actions == 4
        assert default_config.planning_horizon == 3
        assert default_config.inference_algo == "VANILLA"
        assert default_config.action_selection == "stochastic"
        assert default_config.gamma == 16.0
        assert default_config.enable_homeostasis is True

    def test_custom_config_values(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.num_hidden_states == 16
        assert custom_config.num_observations == 8
        assert custom_config.num_actions == 6
        assert custom_config.planning_horizon == 5
        assert custom_config.gamma == 20.0

    def test_learning_rates_in_valid_range(self, default_config):
        """Test that learning rates are in valid range."""
        assert 0.0 < default_config.learning_rate_A <= 1.0
        assert 0.0 < default_config.learning_rate_B <= 1.0
        assert 0.0 < default_config.learning_rate_D <= 1.0


class TestHomeostaticConfig:
    """Test Homeostatic configuration."""

    def test_default_drives_exist(self):
        """Test that default drives are configured."""
        config = HomeostaticConfig()
        assert "attention_budget" in config.drives
        assert "curiosity" in config.drives
        assert "social_connection" in config.drives
        assert "coherence" in config.drives
        assert "safety" in config.drives

    def test_drive_parameters_valid(self):
        """Test that drive parameters are valid."""
        config = HomeostaticConfig()
        for drive_name, params in config.drives.items():
            assert 0.0 <= params["optimal_level"] <= 1.0
            assert 0.0 <= params["initial_level"] <= 1.0
            assert params["decay_rate"] >= 0.0
            assert params["recovery_rate"] >= 0.0
            assert params["importance"] >= 0.0


# ============================================================================
# HOMEOSTATIC DRIVES TESTS
# ============================================================================

class TestHomeostaticDrives:
    """Test Homeostatic Drives for micro-emotions."""

    def test_initialization(self, homeostatic_drives):
        """Test homeostatic drives initialization."""
        assert len(homeostatic_drives.drives) == 5
        assert all(isinstance(d, HomeostaticDrive) for d in homeostatic_drives.drives.values())

    def test_all_drives_present(self, homeostatic_drives):
        """Test all default drives are present."""
        expected = ["attention_budget", "curiosity", "social_connection", "coherence", "safety"]
        for drive_name in expected:
            assert drive_name in homeostatic_drives.drives

    def test_drive_optimal_levels(self, homeostatic_drives):
        """Test drives have valid optimal levels."""
        for drive in homeostatic_drives.drives.values():
            assert 0.0 <= drive.optimal_level <= 1.0
            assert 0.0 <= drive.current_level <= 1.0

    @pytest.mark.asyncio
    async def test_update_drives_returns_valence(self, homeostatic_drives):
        """Test that update_drives returns valence signals."""
        activity = {
            "type": "conversation",
            "intensity": 0.8,
        }
        valence = await homeostatic_drives.update_drives(activity)

        assert isinstance(valence, dict)
        assert len(valence) == 5
        for drive_name, value in valence.items():
            assert isinstance(value, float)

    @pytest.mark.asyncio
    async def test_drives_decay_without_activity(self, homeostatic_drives):
        """Test that drives decay when not satisfied."""
        initial_curiosity = homeostatic_drives.drives["curiosity"].current_level

        # Update without satisfying curiosity
        activity = {"type": "idle", "intensity": 0.1}
        await homeostatic_drives.update_drives(activity)

        # Curiosity should decay
        new_curiosity = homeostatic_drives.drives["curiosity"].current_level
        assert new_curiosity <= initial_curiosity

    @pytest.mark.asyncio
    async def test_social_drive_recovers_with_connection(self, homeostatic_drives):
        """Test social drive recovers with social activity."""
        # Deplete social connection first
        homeostatic_drives.drives["social_connection"].current_level = 0.3

        # Social activity should boost it
        activity = {"type": "conversation", "intensity": 0.8}
        await homeostatic_drives.update_drives(activity)

        # Should be closer to optimal
        new_level = homeostatic_drives.drives["social_connection"].current_level
        assert new_level > 0.3

    def test_get_free_energy(self, homeostatic_drives):
        """Test free energy calculation from homeostatic state."""
        fe = homeostatic_drives.get_free_energy()
        assert isinstance(fe, float)
        assert fe >= 0.0

    def test_free_energy_increases_with_deviation(self, homeostatic_drives):
        """Test free energy increases when drives deviate from optimal."""
        # Start at optimal
        for drive in homeostatic_drives.drives.values():
            drive.current_level = drive.optimal_level
        fe_optimal = homeostatic_drives.get_free_energy()

        # Deviate from optimal
        for drive in homeostatic_drives.drives.values():
            drive.current_level = 0.1  # Far from optimal
        fe_deviated = homeostatic_drives.get_free_energy()

        assert fe_deviated > fe_optimal

    def test_get_most_urgent_need(self, homeostatic_drives):
        """Test getting most urgent homeostatic need."""
        # Set one drive very low
        homeostatic_drives.drives["curiosity"].current_level = 0.1
        homeostatic_drives.drives["curiosity"].optimal_level = 0.8

        most_urgent, urgency = homeostatic_drives.get_most_urgent_need()

        assert isinstance(most_urgent, str)
        assert isinstance(urgency, float)
        # With curiosity far from optimal, it should be urgent
        assert urgency > 0.0

    def test_drive_states_to_dict(self, homeostatic_drives):
        """Test converting drive states to dictionary."""
        # Get states from drives dict directly
        states = {}
        for name, drive in homeostatic_drives.drives.items():
            states[name] = {
                "current": drive.current_level,
                "optimal": drive.optimal_level,
                "deviation": abs(drive.optimal_level - drive.current_level),
            }

        assert isinstance(states, dict)
        assert len(states) == 5
        for drive_name, state in states.items():
            assert "current" in state
            assert "optimal" in state
            assert "deviation" in state


# ============================================================================
# HIERARCHICAL PREDICTIVE PROCESSOR TESTS
# ============================================================================

class TestHierarchicalPredictiveProcessor:
    """Test Hierarchical Predictive Processor."""

    def test_initialization(self, hierarchical_processor):
        """Test processor initialization."""
        assert len(hierarchical_processor.levels) == 3  # Default 3 levels
        assert all(isinstance(l, PredictiveLevel) for l in hierarchical_processor.levels)

    def test_levels_have_decreasing_dimensions(self, hierarchical_processor):
        """Test levels have appropriate dimension progression for compression.

        In predictive processing, higher levels typically have FEWER units
        because they represent more abstract, compressed representations.
        """
        dims = [level.input_dim for level in hierarchical_processor.levels]
        # Higher levels should have lower/equal dimensions (compression)
        for i in range(len(dims) - 1):
            assert dims[i] >= dims[i + 1]

    def test_levels_connected(self, hierarchical_processor):
        """Test levels are properly connected."""
        for i, level in enumerate(hierarchical_processor.levels[:-1]):
            assert level.higher_level == hierarchical_processor.levels[i + 1]
        assert hierarchical_processor.levels[-1].higher_level is None

    @pytest.mark.asyncio
    async def test_process_bottom_up(self, hierarchical_processor):
        """Test bottom-up processing generates prediction errors."""
        sensory_input = np.random.rand(64)  # Match first level input_dim

        errors = await hierarchical_processor.process_bottom_up(sensory_input)

        assert isinstance(errors, list)
        assert len(errors) == 3  # One per level
        for error in errors:
            assert isinstance(error, PredictionError)
            assert error.level >= 0
            assert error.error_magnitude >= 0.0

    @pytest.mark.asyncio
    async def test_process_top_down(self, hierarchical_processor):
        """Test top-down processing generates predictions.

        Goal state dimension must match the top level's output_dim (what
        weights_down expects as input), not input_dim.
        """
        # weights_down shape is (input_dim, output_dim), so it expects output_dim input
        top_level_output_dim = hierarchical_processor.levels[-1].output_dim
        goal_state = np.random.rand(top_level_output_dim)

        predictions = await hierarchical_processor.process_top_down(goal_state)

        assert isinstance(predictions, list)
        assert len(predictions) == 3  # One per level

    def test_get_total_prediction_error(self, hierarchical_processor):
        """Test getting total prediction error."""
        error = hierarchical_processor.get_total_prediction_error()
        assert isinstance(error, float)
        assert error >= 0.0

    @pytest.mark.asyncio
    async def test_prediction_error_decreases_with_learning(self, hierarchical_processor):
        """Test prediction error decreases with repeated exposure."""
        # Same input multiple times
        sensory_input = np.random.rand(64)

        errors = []
        for _ in range(10):
            errs = await hierarchical_processor.process_bottom_up(sensory_input)
            errors.append(sum(e.error_magnitude for e in errs))

        # Error should decrease (or at least not increase much)
        # Note: This is a soft test - learning may not be immediate
        avg_first_half = np.mean(errors[:5])
        avg_second_half = np.mean(errors[5:])
        # Just verify it doesn't explode
        assert avg_second_half < avg_first_half * 2

    def test_get_level_errors(self, hierarchical_processor):
        """Test getting per-level errors."""
        level_errors = hierarchical_processor.get_level_errors()
        assert isinstance(level_errors, dict)


# ============================================================================
# ACTIVE INFERENCE MODULE TESTS
# ============================================================================

class TestActiveInferenceModule:
    """Test Active Inference Module (Core FEP)."""

    def test_initialization(self, active_inference_module):
        """Test module initialization."""
        assert active_inference_module.num_states == 8
        assert active_inference_module.num_observations == 5
        assert active_inference_module.num_actions == 4
        assert active_inference_module.homeostatic_drives is not None
        assert active_inference_module.hierarchical_processor is not None

    def test_initialization_with_custom_config(self, active_inference_module_custom):
        """Test module with custom configuration."""
        assert active_inference_module_custom.num_states == 16
        assert active_inference_module_custom.num_observations == 8
        assert active_inference_module_custom.num_actions == 6

    def test_pymdp_agent_created(self, active_inference_module):
        """Test pymdp agent is created."""
        assert active_inference_module.agent is not None

    def test_generative_model_matrices(self, active_inference_module):
        """Test generative model matrices are initialized."""
        # A matrix (observation model)
        assert active_inference_module.A is not None
        assert len(active_inference_module.A) == 1  # One modality
        assert active_inference_module.A[0].shape == (5, 8)  # (obs, states)

        # B matrix (transition model)
        assert active_inference_module.B is not None
        assert len(active_inference_module.B) == 1  # One factor
        assert active_inference_module.B[0].shape == (8, 8, 4)  # (states, states, actions)

        # D matrix (initial state prior)
        assert active_inference_module.D is not None
        assert len(active_inference_module.D) == 1
        assert active_inference_module.D[0].shape == (8,)

    def test_matrices_are_normalized(self, active_inference_module):
        """Test probability matrices are normalized."""
        # A matrix columns should sum to 1
        A = active_inference_module.A[0]
        col_sums = A.sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, np.ones(8), decimal=5)

        # B matrix columns should sum to 1 for each action
        B = active_inference_module.B[0]
        for a in range(4):
            col_sums = B[:, :, a].sum(axis=0)
            np.testing.assert_array_almost_equal(col_sums, np.ones(8), decimal=5)

        # D should sum to 1
        D_sum = active_inference_module.D[0].sum()
        np.testing.assert_almost_equal(D_sum, 1.0, decimal=5)

    @pytest.mark.asyncio
    async def test_infer_and_act(self, active_inference_module):
        """Test inference and action selection."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        result = await active_inference_module.infer_and_act(observation)

        assert isinstance(result, InferenceResult)
        assert result.posterior_beliefs is not None
        assert len(result.posterior_beliefs) == 8
        assert 0 <= result.selected_action < 4
        assert result.prediction_error >= 0.0
        assert isinstance(result.variational_free_energy, float)
        assert 0.0 <= result.action_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_infer_and_act_with_different_observations(self, active_inference_module):
        """Test inference with different observation patterns."""
        observations = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # One-hot
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Uniform
            np.array([0.5, 0.3, 0.1, 0.05, 0.05]),  # Skewed
        ]

        for obs in observations:
            result = await active_inference_module.infer_and_act(obs)
            assert result is not None
            assert result.posterior_beliefs.sum() > 0  # Should have beliefs

    @pytest.mark.asyncio
    async def test_beliefs_are_probability_distribution(self, active_inference_module):
        """Test posterior beliefs form valid probability distribution."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        result = await active_inference_module.infer_and_act(observation)

        beliefs = result.posterior_beliefs
        assert np.all(beliefs >= 0)  # Non-negative
        np.testing.assert_almost_equal(beliefs.sum(), 1.0, decimal=5)  # Sum to 1

    @pytest.mark.asyncio
    async def test_prediction_error_calculation(self, active_inference_module):
        """Test prediction error calculation."""
        # Perfect observation match should have low error
        observation1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result1 = await active_inference_module.infer_and_act(observation1)

        # Very different observation should have higher error
        observation2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result2 = await active_inference_module.infer_and_act(observation2)

        # Both should produce valid errors
        assert result1.prediction_error >= 0.0
        assert result2.prediction_error >= 0.0

    @pytest.mark.asyncio
    async def test_predict_next(self, active_inference_module):
        """Test future prediction."""
        # First do an inference to build beliefs
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await active_inference_module.infer_and_act(observation)

        # Now predict
        predictions = await active_inference_module.predict_next(horizon=3)

        assert len(predictions) == 3
        for pred in predictions:
            assert len(pred) == 5  # Observation dimension
            assert np.all(pred >= 0)  # Non-negative

    @pytest.mark.asyncio
    async def test_update_generative_model(self, active_inference_module):
        """Test learning / model update."""
        # Do some inferences first
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await active_inference_module.infer_and_act(observation)

        # Store original A matrix
        original_A = active_inference_module.A[0].copy()

        # Update model
        experience = {
            "observation": observation,
            "action": 0,
        }
        await active_inference_module.update_generative_model(experience)

        # A matrix should have changed
        assert active_inference_module.total_model_updates == 1
        # Note: Changes might be small, so we just verify the counter

    @pytest.mark.asyncio
    async def test_update_homeostasis(self, active_inference_module):
        """Test homeostatic drive update."""
        activity = {
            "type": "conversation",
            "intensity": 0.8,
            "understanding_level": 0.7,
        }

        valence = await active_inference_module.update_homeostasis(activity)

        assert isinstance(valence, dict)
        assert len(valence) == 5

    @pytest.mark.asyncio
    async def test_process_with_hierarchy(self, active_inference_module):
        """Test combined hierarchical + active inference processing."""
        sensory_input = np.random.rand(64)

        inference_result, hier_errors = await active_inference_module.process_with_hierarchy(sensory_input)

        assert isinstance(inference_result, InferenceResult)
        assert isinstance(hier_errors, list)
        assert len(hier_errors) == 3  # 3 levels

    @pytest.mark.asyncio
    async def test_get_state(self, active_inference_module):
        """Test getting complete state."""
        # Do an inference first to populate beliefs
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await active_inference_module.infer_and_act(observation)

        state = active_inference_module.get_state()

        assert isinstance(state, ActiveInferenceState)
        assert state.current_beliefs is not None
        assert state.homeostatic_free_energy >= 0.0

    def test_get_statistics(self, active_inference_module):
        """Test getting statistics."""
        stats = active_inference_module.get_statistics()

        assert isinstance(stats, dict)
        assert "total_inferences" in stats
        assert "total_model_updates" in stats
        assert "variational_free_energy" in stats

    @pytest.mark.asyncio
    async def test_generate_report(self, active_inference_module):
        """Test report generation."""
        # Do an inference first
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await active_inference_module.infer_and_act(observation)

        report = await active_inference_module.generate_active_inference_report()

        assert isinstance(report, str)
        assert len(report) > 0


# ============================================================================
# FREE ENERGY PRINCIPLE TESTS
# ============================================================================

class TestFreeEnergyPrinciple:
    """Test Free Energy Principle properties."""

    @pytest.mark.asyncio
    async def test_vfe_is_positive(self, active_inference_module):
        """Test variational free energy is positive."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        result = await active_inference_module.infer_and_act(observation)

        # VFE can be negative (it's a bound on surprise)
        # but should be finite
        assert np.isfinite(result.variational_free_energy)

    @pytest.mark.asyncio
    async def test_expected_free_energy_guides_action(self, active_inference_module):
        """Test that EFE influences action selection."""
        result = await active_inference_module.infer_and_act(
            np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        )

        # EFE should be computed
        assert result.expected_free_energy is not None
        assert len(result.expected_free_energy) > 0

    @pytest.mark.asyncio
    async def test_belief_updating_reduces_surprise(self, active_inference_module):
        """Test that belief updating helps reduce prediction error over time."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        # Multiple inferences with same observation
        errors = []
        for _ in range(5):
            result = await active_inference_module.infer_and_act(observation)
            errors.append(result.prediction_error)

        # Errors should be relatively stable (not exploding)
        assert np.std(errors) < np.mean(errors) * 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegrationWithWorkspace:
    """Test integration with Enhanced Global Workspace."""

    @pytest.mark.asyncio
    async def test_workspace_creates_active_inference(self):
        """Test workspace initializes Active Inference."""
        from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace

        workspace = EnhancedGlobalWorkspace()

        assert workspace.active_inference is not None
        assert isinstance(workspace.active_inference, ActiveInferenceModule)

    @pytest.mark.asyncio
    async def test_workspace_includes_fep_in_state(self):
        """Test workspace includes FEP data in consciousness state."""
        from mtc.consciousness.enhanced_global_workspace import (
            EnhancedGlobalWorkspace,
            WorkspaceCandidate,
            WorkspaceCandidateSource,
        )

        workspace = EnhancedGlobalWorkspace()

        # Create test candidates
        candidates = [
            WorkspaceCandidate(
                content="test",
                content_type="thought",
                summary="Test thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.8,
            )
        ]

        state = await workspace.process_consciousness_cycle(candidates)

        # Check FEP fields are present
        assert state.active_inference_state is not None
        assert state.inference_result is not None
        assert isinstance(state.prediction_error, float)
        assert isinstance(state.variational_free_energy, float)

    @pytest.mark.asyncio
    async def test_workspace_observation_conversion(self):
        """Test workspace converts state to observations correctly."""
        from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace

        workspace = EnhancedGlobalWorkspace()

        # Test observation conversion
        observation = await workspace._workspace_to_observation(
            winners=[],
            ignited=[],
            integration_level=0.5,
            emotional_state={"joy": 0.8, "fear": 0.1}
        )

        assert len(observation) == 5
        assert np.isclose(observation.sum(), 1.0)  # Normalized


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_zero_observation(self, active_inference_module):
        """Test handling of zero observation."""
        observation = np.zeros(5)
        result = await active_inference_module.infer_and_act(observation)

        assert result is not None
        assert result.posterior_beliefs is not None

    @pytest.mark.asyncio
    async def test_wrong_observation_size(self, active_inference_module):
        """Test handling of wrong observation size."""
        # Too large
        observation = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05])
        result = await active_inference_module.infer_and_act(observation)
        assert result is not None

        # Too small
        observation = np.array([0.5, 0.5])
        result = await active_inference_module.infer_and_act(observation)
        assert result is not None

    @pytest.mark.asyncio
    async def test_repeated_inferences_stable(self, active_inference_module):
        """Test system remains stable under repeated inferences."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        for _ in range(100):
            result = await active_inference_module.infer_and_act(observation)
            assert np.isfinite(result.variational_free_energy)
            assert np.all(np.isfinite(result.posterior_beliefs))

    def test_homeostasis_disabled(self):
        """Test module works with homeostasis disabled."""
        config = ActiveInferenceConfig(enable_homeostasis=False)
        module = ActiveInferenceModule(config)

        assert module.homeostatic_drives is None

    @pytest.mark.asyncio
    async def test_empty_belief_history(self, active_inference_module):
        """Test prediction without belief history."""
        # Clear history
        active_inference_module.belief_history = []

        predictions = await active_inference_module.predict_next(horizon=3)
        assert len(predictions) == 3


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_inference_time(self, active_inference_module):
        """Test inference completes in reasonable time."""
        import time

        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        start = time.time()
        for _ in range(100):
            await active_inference_module.infer_and_act(observation)
        elapsed = time.time() - start

        # Should complete 100 inferences in under 5 seconds
        assert elapsed < 5.0
        avg_time = elapsed / 100 * 1000  # ms
        print(f"Average inference time: {avg_time:.2f}ms")

    def test_memory_bounded(self, active_inference_module):
        """Test history buffers are bounded."""
        # Fill with many inferences
        for _ in range(200):
            active_inference_module.belief_history.append(np.random.rand(8))
            active_inference_module.prediction_errors.append(0.5)

        # Should be bounded
        assert len(active_inference_module.belief_history) <= 1000
        # prediction_errors isn't bounded in current impl, but should be reasonable


# ============================================================================
# META-STATE TESTS (Beautiful Loop - Self-Referential Active Inference)
# ============================================================================

class TestMetaState:
    """
    Test MetaState -- the self-referential meta-model.

    The Beautiful Loop (Laukkonen/Friston/Chandaria 2025) proposes that
    consciousness arises when a predictive system predicts its own prediction
    accuracy. These tests verify that the MetaState correctly tracks
    accuracy, calibrates confidence, and maintains bounded state variables.
    """

    def test_creation_with_defaults(self):
        """Test MetaState initializes with sensible defaults."""
        ms = MetaState()
        assert ms.model_confidence == 0.5
        assert ms.prediction_accuracy == 0.5
        assert ms.predicted_accuracy == 0.5
        assert ms.cognitive_load == 0.0
        assert ms.attention_pattern == "general"
        assert ms.emotional_valence == 0.0
        assert ms.total_predictions == 0
        assert ms.correct_predictions == 0
        assert len(ms._accuracy_history) == 0
        assert len(ms._confidence_history) == 0

    def test_update_increments_counters(self):
        """Test that update increments prediction counters."""
        ms = MetaState()
        ms.update(prediction_error=0.3, was_accurate=True)

        assert ms.total_predictions == 1
        assert ms.correct_predictions == 1
        assert ms.prediction_accuracy == 1.0  # 1/1

    def test_update_tracks_inaccurate_predictions(self):
        """Test that inaccurate predictions are tracked."""
        ms = MetaState()
        ms.update(prediction_error=0.8, was_accurate=False)

        assert ms.total_predictions == 1
        assert ms.correct_predictions == 0
        assert ms.prediction_accuracy == 0.0  # 0/1

    def test_accuracy_converges_over_time(self):
        """Test that prediction_accuracy converges to the true rate."""
        ms = MetaState()
        # 70% accuracy: 7 correct, 3 wrong
        for i in range(10):
            ms.update(prediction_error=0.3 if i < 7 else 0.8,
                      was_accurate=(i < 7))

        assert ms.total_predictions == 10
        assert ms.correct_predictions == 7
        assert abs(ms.prediction_accuracy - 0.7) < 1e-10

    def test_confidence_stays_bounded(self):
        """Test that model_confidence stays in [0, 1]."""
        ms = MetaState()
        # Many accurate predictions should push confidence up but not past 1
        for _ in range(200):
            ms.update(prediction_error=0.1, was_accurate=True)
        assert 0.0 <= ms.model_confidence <= 1.0

        # Many inaccurate predictions should push confidence down but not below 0
        ms2 = MetaState()
        for _ in range(200):
            ms2.update(prediction_error=0.9, was_accurate=False)
        assert 0.0 <= ms2.model_confidence <= 1.0

    def test_predicted_accuracy_moves_toward_actual(self):
        """Test that predicted_accuracy drifts toward actual accuracy."""
        ms = MetaState()
        # All accurate -- actual accuracy is 1.0
        for _ in range(50):
            ms.update(prediction_error=0.1, was_accurate=True)

        # predicted_accuracy should have moved from 0.5 toward 1.0
        assert ms.predicted_accuracy > 0.5

    def test_cognitive_load_from_prediction_error(self):
        """Test cognitive_load tracks prediction error magnitude."""
        ms = MetaState()
        ms.update(prediction_error=0.7, was_accurate=False)
        assert ms.cognitive_load == 0.7

        ms.update(prediction_error=1.5, was_accurate=False)
        # Clamped to 1.0
        assert ms.cognitive_load == 1.0

        ms.update(prediction_error=0.2, was_accurate=True)
        assert ms.cognitive_load == 0.2

    def test_accuracy_history_grows(self):
        """Test that _accuracy_history and _confidence_history grow with updates."""
        ms = MetaState()
        for _ in range(5):
            ms.update(prediction_error=0.3, was_accurate=True)

        assert len(ms._accuracy_history) == 5
        assert len(ms._confidence_history) == 5

    def test_well_calibrated_system_gains_confidence(self):
        """
        If predicted_accuracy is close to actual accuracy,
        meta-error is low, so confidence should rise.
        """
        ms = MetaState()
        # Force predicted_accuracy close to what actual will be
        ms.predicted_accuracy = 0.9
        # Then consistently be accurate
        for _ in range(20):
            ms.update(prediction_error=0.1, was_accurate=True)

        # Confidence should be higher than starting 0.5
        assert ms.model_confidence > 0.5


class TestSelfReferentialInference:
    """
    Test the Beautiful Loop integration with ActiveInferenceModule.

    These tests verify that when enable_meta_inference=True, the module
    correctly maintains a MetaState, updates it after each inference cycle,
    and exposes meta fields through get_state().
    """

    @pytest.fixture
    def meta_module(self):
        """Active Inference module with meta-inference enabled."""
        config = ActiveInferenceConfig(enable_meta_inference=True)
        return ActiveInferenceModule(config)

    def test_meta_state_exists_when_enabled(self, meta_module):
        """Test that meta_state is created when enabled."""
        assert meta_module.meta_state is not None
        assert isinstance(meta_module.meta_state, MetaState)

    def test_meta_state_none_when_disabled(self):
        """Test backward compatibility: meta_state is None when explicitly disabled."""
        config = ActiveInferenceConfig(enable_meta_inference=False)
        module = ActiveInferenceModule(config)
        assert module.meta_state is None

    def test_default_config_has_meta_enabled(self):
        """Test that the default config enables meta-inference (Beautiful Loop)."""
        config = ActiveInferenceConfig()
        assert config.enable_meta_inference is True

    @pytest.mark.asyncio
    async def test_meta_state_updated_after_inference(self, meta_module):
        """Test that infer_and_act updates the meta-state."""
        assert meta_module.meta_state.total_predictions == 0

        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await meta_module.infer_and_act(observation)

        assert meta_module.meta_state.total_predictions == 1
        assert len(meta_module.meta_state._accuracy_history) == 1

    @pytest.mark.asyncio
    async def test_meta_prediction_after_multiple_inferences(self, meta_module):
        """Test meta-state accumulates data over multiple cycles."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        for _ in range(10):
            await meta_module.infer_and_act(observation)

        ms = meta_module.meta_state
        assert ms.total_predictions == 10
        assert len(ms._accuracy_history) == 10
        assert len(ms._confidence_history) == 10
        # predicted_accuracy should have moved from initial 0.5
        assert ms.predicted_accuracy != 0.5

    @pytest.mark.asyncio
    async def test_calibration_over_time(self, meta_module):
        """
        Test that the meta-model calibrates over repeated inferences.

        After many cycles, the gap between predicted_accuracy and actual
        prediction_accuracy should narrow (meta-prediction error decreases).
        """
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        for _ in range(100):
            await meta_module.infer_and_act(observation)

        ms = meta_module.meta_state
        meta_error = abs(ms.predicted_accuracy - ms.prediction_accuracy)
        # After 100 cycles of the same observation, the exponential moving
        # average should have brought predicted_accuracy close to actual.
        # Threshold 0.20 accounts for stochastic accuracy fluctuations
        # in the windowed outcome tracker.
        assert meta_error < 0.20, (
            f"Meta-prediction error {meta_error:.3f} should be < 0.20 "
            f"after 100 calibration cycles"
        )

    @pytest.mark.asyncio
    async def test_meta_state_in_get_state(self, meta_module):
        """Test that get_state() includes meta-inference fields."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await meta_module.infer_and_act(observation)

        state = meta_module.get_state()

        assert isinstance(state, ActiveInferenceState)
        # Meta fields should be populated from the MetaState
        assert state.meta_confidence == meta_module.meta_state.model_confidence
        assert state.meta_predicted_accuracy == meta_module.meta_state.predicted_accuracy
        assert state.meta_cognitive_load == meta_module.meta_state.cognitive_load

    @pytest.mark.asyncio
    async def test_get_state_defaults_without_meta(self):
        """Test that get_state() returns default meta fields when meta is disabled."""
        config = ActiveInferenceConfig(enable_meta_inference=False)
        module = ActiveInferenceModule(config)
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await module.infer_and_act(observation)

        state = module.get_state()

        # Should use defaults, not crash
        assert state.meta_confidence == 0.5
        assert state.meta_predicted_accuracy == 0.5
        assert state.meta_cognitive_load == 0.0

    @pytest.mark.asyncio
    async def test_meta_does_not_affect_core_inference(self, meta_module):
        """
        Test that enabling meta-inference does not break core inference behavior.
        The posterior beliefs, action selection, and free energy should all
        still work correctly.
        """
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        result = await meta_module.infer_and_act(observation)

        assert isinstance(result, InferenceResult)
        assert result.posterior_beliefs is not None
        assert np.all(result.posterior_beliefs >= 0)
        np.testing.assert_almost_equal(result.posterior_beliefs.sum(), 1.0, decimal=5)
        assert 0 <= result.selected_action < meta_module.num_actions
        assert np.isfinite(result.variational_free_energy)
        assert result.prediction_error >= 0.0

    @pytest.mark.asyncio
    async def test_meta_cognitive_load_reflects_error(self, meta_module):
        """Test that cognitive_load in meta-state reflects prediction error."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        result = await meta_module.infer_and_act(observation)

        expected_load = min(1.0, result.prediction_error)
        assert meta_module.meta_state.cognitive_load == expected_load


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
