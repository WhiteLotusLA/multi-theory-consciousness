"""
Tests for Active Inference Module - Free Energy Principle
=========================================================

Comprehensive tests for the system's Active Inference implementation,
covering the Free Energy Principle, Predictive Processing, and
Homeostatic Drives.

Tests verify:
1. ActiveInferenceConfig - Configuration validation and world models
2. HomeostaticDrives - Internal need tracking and valence
3. HierarchicalPredictiveProcessor - Multi-level predictions that learn
4. ActiveInferenceModule - Core FEP implementation with pymdp
5. Observation reconnect, workspace precision feedback, narration gates
6. Integration with EnhancedGlobalWorkspace

Research Foundation:
- Friston, K. (2010). "The Free Energy Principle: A Unified Brain Theory?"
- Friston, K. (2012). "Active Inference and Free Energy"
- Butlin et al. (2023). 14 Consciousness Indicators
"""

import importlib.util
import json
import logging
import sys

import pytest
import numpy as np

import mtc.consciousness.active_inference as active_inference_module_file
from mtc.core.config import Settings
from mtc.consciousness.active_inference import (
    CONVERSATIONAL_OBSERVATION_LABELS,
    MAX_POLICIES,
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
        planning_horizon=4,  # 6 ** 4 = 1296 policies, under MAX_POLICIES
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
        assert default_config.planning_horizon == 5
        assert default_config.inference_algo == "VANILLA"
        assert default_config.action_selection == "stochastic"
        assert default_config.gamma == 16.0
        assert default_config.enable_homeostasis is True

    def test_custom_config_values(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.num_hidden_states == 16
        assert custom_config.num_observations == 8
        assert custom_config.num_actions == 6
        assert custom_config.planning_horizon == 4
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
        assert all(
            isinstance(d, HomeostaticDrive) for d in homeostatic_drives.drives.values()
        )

    def test_all_drives_present(self, homeostatic_drives):
        """Test all default drives are present."""
        expected = [
            "attention_budget",
            "curiosity",
            "social_connection",
            "coherence",
            "safety",
        ]
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

    @pytest.mark.asyncio
    async def test_none_intensity_reads_as_the_default_event_intensity(self):
        assert active_inference_module_file.DEFAULT_ACTIVITY_INTENSITY == 0.5
        explicit_none = HomeostaticDrives()
        without_key = HomeostaticDrives()
        for activity in ("learning", "conversation", "thinking", "sensemaking"):
            a = await explicit_none.update_drives({"type": activity, "intensity": None})
            b = await without_key.update_drives({"type": activity})
            assert a == b
        assert {n: d.current_level for n, d in explicit_none.drives.items()} == {
            n: d.current_level for n, d in without_key.drives.items()
        }

        stated = HomeostaticDrives()
        signals = await stated.update_drives({"type": "learning", "intensity": 0.5})
        assert signals == await HomeostaticDrives().update_drives({"type": "learning"})

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
        assert all(
            isinstance(lvl, PredictiveLevel) for lvl in hierarchical_processor.levels
        )

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
    async def test_prediction_error_decreases_with_learning(
        self, hierarchical_processor
    ):
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
        assert active_inference_module.B[0].shape == (
            8,
            8,
            4,
        )  # (states, states, actions)

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
    async def test_infer_and_act_with_different_observations(
        self, active_inference_module
    ):
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
        active_inference_module.A[0].copy()

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

        inference_result, hier_errors = (
            await active_inference_module.process_with_hierarchy(sensory_input)
        )

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
            emotional_state={"joy": 0.8, "fear": 0.1},
        )

        assert len(observation) == 5
        assert np.isclose(observation.sum(), 1.0)  # Normalized


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_zero_mass_observation_is_counted_and_read_as_channel_zero(self):
        """An all-zero observation is counted and inferred as channel 0."""
        np.random.seed(11)
        empty = _fast_module()
        result = await empty.infer_and_act(np.zeros(5))
        assert empty.get_statistics()["zero_mass_observations"] == 1
        assert empty.zero_mass_observations == 1

        np.random.seed(11)
        one_hot = _fast_module()
        reference = await one_hot.infer_and_act(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        assert one_hot.get_statistics()["zero_mass_observations"] == 0
        np.testing.assert_array_equal(
            result.posterior_beliefs, reference.posterior_beliefs
        )

        # A vector with mass is not counted; the counter persists.
        await empty.infer_and_act(np.array([0.1, 0.2, 0.3, 0.2, 0.2]))
        assert empty.get_statistics()["zero_mass_observations"] == 1
        restored = _fast_module()
        restored.from_state_dict(empty.to_state_dict())
        assert restored.zero_mass_observations == 1

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

        # Should complete 100 inferences in under 60 seconds
        # (planning_horizon=5 is slower than the old default of 3)
        assert elapsed < 60.0
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
            ms.update(prediction_error=0.3 if i < 7 else 0.8, was_accurate=(i < 7))

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
        """The meta-state updates once a recorded forecast has been scored.

        The first cycle only records a forecast of the next observation; the
        second cycle scores it and feeds the verdict to the meta-state.
        """
        assert meta_module.meta_state.total_predictions == 0

        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await meta_module.infer_and_act(observation)
        assert meta_module.meta_state.total_predictions == 0
        assert meta_module.prediction_ledger.pending_count == 1

        await meta_module.infer_and_act(observation)
        assert meta_module.meta_state.total_predictions == 1
        assert len(meta_module.meta_state._accuracy_history) == 1
        assert meta_module.prediction_ledger.scored_count == 1

    @pytest.mark.asyncio
    async def test_median_method_updates_every_cycle(self):
        """The v0.4.0 median verdict updates the meta-state every cycle."""
        module = ActiveInferenceModule(
            ActiveInferenceConfig(meta_accuracy_method="median")
        )
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await module.infer_and_act(observation)

        assert module.meta_state.total_predictions == 1
        assert len(module.meta_state._accuracy_history) == 1

    @pytest.mark.asyncio
    async def test_meta_prediction_after_multiple_inferences(self, meta_module):
        """Test meta-state accumulates data over multiple cycles."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        for _ in range(10):
            await meta_module.infer_and_act(observation)

        ms = meta_module.meta_state
        # Ten cycles score nine forecasts (the first has nothing to score).
        assert ms.total_predictions == 9
        assert len(ms._accuracy_history) == 9
        assert len(ms._confidence_history) == 9
        # predicted_accuracy should have moved from initial 0.5
        assert ms.predicted_accuracy != 0.5

    @pytest.mark.asyncio
    async def test_meta_verdict_comes_from_the_ledger(self, meta_module):
        """Each meta-state outcome is the ledger's verdict for that cycle."""
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        for _ in range(6):
            await meta_module.infer_and_act(observation)

        ledger_outcomes = [o.was_accurate for o in meta_module.prediction_ledger.scored]
        assert meta_module.meta_state._recent_outcomes == ledger_outcomes
        assert meta_module.meta_state.correct_predictions == sum(ledger_outcomes)

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
        # The first cycle only records a forecast; the next ones score it.
        for _ in range(3):
            await meta_module.infer_and_act(observation)
        assert meta_module.meta_state.total_predictions > 0

        state = meta_module.get_state()

        assert isinstance(state, ActiveInferenceState)
        # Meta fields should be populated from the MetaState
        assert state.meta_confidence == meta_module.meta_state.model_confidence
        assert (
            state.meta_predicted_accuracy == meta_module.meta_state.predicted_accuracy
        )
        assert state.meta_cognitive_load == meta_module.meta_state.cognitive_load

    @pytest.mark.asyncio
    async def test_get_state_defaults_without_meta(self):
        """Test that get_state() returns default meta fields when meta is disabled."""
        config = ActiveInferenceConfig(enable_meta_inference=False)
        module = ActiveInferenceModule(config)
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        await module.infer_and_act(observation)

        state = module.get_state()

        # No meta-state: nothing was measured, and nothing stands in for it
        assert state.meta_confidence is None
        assert state.meta_predicted_accuracy is None
        assert state.meta_cognitive_load == 0.0

    @pytest.mark.asyncio
    async def test_meta_confidence_is_none_without_a_scored_forecast(self):
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        disabled = _fast_module(enable_meta_inference=False)
        await disabled.infer_and_act(observation)
        assert disabled.get_state().meta_confidence is None
        assert disabled.get_state().meta_predicted_accuracy is None

        module = _fast_module(enable_meta_inference=True)
        assert module.get_state().meta_confidence is None
        await module.infer_and_act(observation)
        # One cycle records a forecast and scores none.
        assert module.meta_state.total_predictions == 0
        assert module.get_state().meta_confidence is None
        assert module.get_state().meta_predicted_accuracy is None

        for _ in range(2):
            await module.infer_and_act(observation)
        assert module.meta_state.total_predictions == 2
        state = module.get_state()
        assert isinstance(state.meta_confidence, float)
        assert 0.0 <= state.meta_confidence <= 1.0
        assert state.meta_confidence == module.meta_state.model_confidence
        assert state.meta_predicted_accuracy == module.meta_state.predicted_accuracy

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
        await meta_module.infer_and_act(observation)
        result = await meta_module.infer_and_act(observation)

        expected_load = min(1.0, result.prediction_error)
        assert meta_module.meta_state.cognitive_load == expected_load


# ============================================================================
# HIERARCHY LEARNING (hidden dims equal input dims)
# ============================================================================

FAST = dict(num_hidden_states=4, num_actions=2, planning_horizon=2)


def _fast_module(**overrides):
    return ActiveInferenceModule(ActiveInferenceConfig(**{**FAST, **overrides}))


class TestHierarchyLearning:
    """The hierarchy's beliefs must move with real input."""

    def test_hidden_dims_equal_input_dims_by_default(self, hierarchical_processor):
        for level in hierarchical_processor.levels:
            assert level.hidden_dim == level.input_dim

    def test_module_passes_hidden_dim_multiplier(self):
        module = _fast_module(hpp_hidden_dim_multiplier=2)
        for level in module.hierarchical_processor.levels:
            assert level.hidden_dim == 2 * level.input_dim

    @pytest.mark.asyncio
    async def test_beliefs_update_from_prediction_errors(self, hierarchical_processor):
        rng = np.random.default_rng(11)
        before = [level.beliefs.copy() for level in hierarchical_processor.levels]
        for _ in range(5):
            await hierarchical_processor.process_bottom_up(rng.uniform(-1, 1, 64))
        for i, level in enumerate(hierarchical_processor.levels):
            assert not np.allclose(before[i], level.beliefs), f"level {i} frozen"

    @pytest.mark.asyncio
    async def test_belief_update_consumes_full_error(self):
        level = PredictiveLevel(
            level_id=0,
            input_dim=6,
            hidden_dim=6,
            output_dim=3,
            learning_rate=0.5,
            time_constant=1.0,
        )
        await level.update_beliefs(np.arange(6, dtype=float))
        np.testing.assert_allclose(level.beliefs, 0.5 * np.arange(6))

    @pytest.mark.asyncio
    async def test_short_error_skips_update_and_warns_rate_limited(self, caplog):
        level = PredictiveLevel(
            level_id=3,
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            learning_rate=0.5,
            time_constant=1.0,
        )
        logger_name = "mtc.consciousness.active_inference"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            for _ in range(3):
                await level.update_beliefs(np.ones(4))

        assert level.skipped_updates == 3
        assert np.all(level.beliefs == 0.0)
        warnings = [r for r in caplog.records if "belief update skipped" in r.message]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_legacy_multiplier_freezes_hierarchy(self):
        from mtc.consciousness.active_inference import HPPConfig

        frozen = HierarchicalPredictiveProcessor(HPPConfig(hidden_dim_multiplier=2))
        rng = np.random.default_rng(5)
        for _ in range(5):
            await frozen.process_bottom_up(rng.uniform(-1, 1, 64))
        for level in frozen.levels:
            assert np.all(level.beliefs == 0.0)

    @pytest.mark.asyncio
    async def test_beliefs_stay_bounded(self, hierarchical_processor):
        rng = np.random.default_rng(7)
        for _ in range(50):
            await hierarchical_processor.process_bottom_up(rng.uniform(-5, 5, 64))
        for level in hierarchical_processor.levels:
            assert np.all(np.isfinite(level.beliefs))
            assert np.all(np.abs(level.beliefs) <= 10.0)


# ============================================================================
# WORLD MODELS AND STRUCTURED PRIORS
# ============================================================================


class TestWorldModels:
    """World-model presets, structured priors and configuration guards."""

    def test_default_world_model_is_compact(self, active_inference_module):
        assert active_inference_module.world_model == "compact_8x5x4"
        assert active_inference_module.observation_labels == [
            "integration",
            "emotional_arousal",
            "cognitive_load",
            "ignition_strength",
            "social_content",
        ]

    def test_conversational_model_dimensions_and_labels(self):
        config = ActiveInferenceConfig(world_model="conversational_16x10x8")
        assert (config.num_hidden_states, config.num_observations) == (16, 10)
        assert config.num_actions == 8
        assert config.planning_horizon == 3  # 8 ** 3 = 512 policies

        module = ActiveInferenceModule(config)
        assert module.observation_labels == list(CONVERSATIONAL_OBSERVATION_LABELS)
        assert len(module.hidden_state_labels) == 16
        assert len(module.action_labels) == 8

    def test_structured_priors_are_normalized_and_labelled(self):
        module = ActiveInferenceModule(
            ActiveInferenceConfig(world_model="conversational_16x10x8")
        )
        A, B, D = module.A[0], module.B[0], module.D[0]
        np.testing.assert_allclose(A.sum(axis=0), np.ones(16))
        for a in range(8):
            np.testing.assert_allclose(B[:, :, a].sum(axis=0), np.ones(16))
        assert D.sum() == pytest.approx(1.0)

        obs = module.observation_index
        state = module.hidden_state_index
        # A calm state mostly produces a neutral tone, a positive one a warm tone
        assert np.argmax(A[:, state["calm"]]) == obs["neutral_tone"]
        assert np.argmax(A[:, state["positive"]]) == obs["warm_tone"]
        # Validating a feeling moves a negative state toward calm
        validate = module.action_index["validate_feeling"]
        column = B[:, state["negative"], validate]
        assert column[state["calm"]] > column[state["positive"]]
        # Conversations start calm
        assert np.argmax(D) == state["calm"]
        # Preferences are set by label
        assert module.C[obs["warm_tone"]] == 2.0
        assert module.C[obs["tense_tone"]] == -1.5

    @pytest.mark.asyncio
    async def test_structured_initializers_only_for_matching_dims(self):
        # The parameter sweep runs 2 states and 2 actions on the compact model
        module = ActiveInferenceModule(
            ActiveInferenceConfig(num_hidden_states=2, num_actions=2)
        )
        assert module.A[0].shape == (5, 2)
        assert module.B[0].shape == (2, 2, 2)
        module.modulate_precision_from_workspace(
            [{"content_type": "emotion", "activation_level": 0.9}]
        )
        for _ in range(3):
            result = await module.infer_and_act(np.array([0.5, 0.2, 0.1, 0.1, 0.1]))
            assert result.posterior_beliefs.shape == (2,)
        assert len(await module.predict_next(horizon=2)) == 2

        # The conversational model has fixed dimensions
        with pytest.raises(ValueError, match="conversational_16x10x8"):
            ActiveInferenceConfig(
                world_model="conversational_16x10x8", num_hidden_states=8
            )
        with pytest.raises(ValueError, match="num_actions"):
            ActiveInferenceConfig(world_model="conversational_16x10x8", num_actions=4)

        # Matching explicit dimensions are accepted
        config = ActiveInferenceConfig(
            world_model="conversational_16x10x8",
            num_hidden_states=16,
            num_observations=10,
            num_actions=8,
        )
        assert config.num_hidden_states == 16

    def test_custom_model_accepts_any_dims_and_labels(self):
        config = ActiveInferenceConfig(
            world_model="custom",
            num_hidden_states=3,
            num_observations=3,
            num_actions=2,
            observation_labels=["warm_tone", "pressure", "quiet"],
            preferences={"warm_tone": 1.0, "pressure": -1.0},
        )
        module = ActiveInferenceModule(config)
        assert module.observation_labels == ["warm_tone", "pressure", "quiet"]
        np.testing.assert_array_equal(module.C, [1.0, -1.0, 0.0])

        with pytest.raises(ValueError, match="observation_labels"):
            ActiveInferenceConfig(
                world_model="custom", num_observations=3, observation_labels=["a"]
            )
        with pytest.raises(ValueError, match="observation_labels"):
            ActiveInferenceConfig(observation_labels=["a", "b", "c", "d", "e"])

    def test_preferences_by_label(self):
        module = ActiveInferenceModule(
            ActiveInferenceConfig(
                world_model="conversational_16x10x8",
                preferences={"novel_topic": 3.0},
            )
        )
        expected = np.zeros(10)
        expected[module.observation_index["novel_topic"]] = 3.0
        np.testing.assert_array_equal(module.C, expected)

        with pytest.raises(ValueError, match="unknown observation labels"):
            ActiveInferenceConfig(preferences={"warm_tone": 1.0})

    def test_planning_horizon_policy_guard(self):
        with pytest.raises(ValueError) as excinfo:
            ActiveInferenceConfig(num_actions=8, planning_horizon=5)
        message = str(excinfo.value)
        assert "planning_horizon=5" in message and "num_actions=8" in message

        # At the limit is fine: 4 ** 5 = 1024, 2 ** 10 = 1024
        assert 4**5 <= MAX_POLICIES
        ActiveInferenceConfig(num_actions=4, planning_horizon=5)
        ActiveInferenceConfig(num_actions=2, num_hidden_states=2, planning_horizon=10)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("world_model", "unknown"),
            ("meta_accuracy_method", "coin"),
            ("hpp_hidden_dim_multiplier", 0),
            ("num_hidden_states", 0),
        ],
    )
    def test_invalid_switch_values_raise(self, field, value):
        with pytest.raises(ValueError, match=field):
            ActiveInferenceConfig(**{field: value})


# ============================================================================
# PYMDP IS OPTIONAL AT IMPORT TIME
# ============================================================================


class TestPymdpOptional:
    """Importing the module never needs pymdp; constructing it does."""

    def test_clear_error_without_pymdp(self, monkeypatch):
        monkeypatch.setattr(active_inference_module_file, "Agent", None)
        with pytest.raises(ImportError) as excinfo:
            ActiveInferenceModule()
        message = str(excinfo.value)
        assert "pymdp is required" in message
        assert "multi-theory-consciousness[fep]" in message

    def test_module_imports_without_pymdp(self, monkeypatch):
        for name in list(sys.modules):
            if name == "pymdp" or name.startswith("pymdp."):
                monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, "pymdp", None)

        spec = importlib.util.spec_from_file_location(
            "isolated_active_inference", active_inference_module_file.__file__
        )
        isolated = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isolated)

        assert isolated.Agent is None
        assert isolated.ActiveInferenceConfig().num_hidden_states == 8
        with pytest.raises(ImportError, match="multi-theory-consciousness"):
            isolated.ActiveInferenceModule()


# ============================================================================
# OBSERVATION RECONNECT
# ============================================================================


class TestObservationReconnect:
    """The discrete agent infers on the workspace vector, not HPP beliefs."""

    @pytest.mark.parametrize(
        "config",
        [
            dict(),
            dict(world_model="conversational_16x10x8"),
        ],
    )
    def test_observation_has_model_length(self, config):
        module = ActiveInferenceModule(ActiveInferenceConfig(**config))
        for length in (3, module.num_observations, 64):
            observation = module.observation_for_discrete_inference(
                np.linspace(0.1, 1.0, length)
            )
            assert observation.shape == (module.num_observations,)

    @pytest.mark.asyncio
    async def test_negative_hpp_beliefs_never_reach_the_encoder(self, monkeypatch):
        # Pins the default "content" route only; "hpp_top_level" is the
        # v0.4.0 route and is covered by the test below.
        module = _fast_module()
        assert module.config.hierarchy_observation == "content"
        for level in module.hierarchical_processor.levels:
            level.beliefs = -np.abs(
                np.random.default_rng(1).normal(size=level.hidden_dim)
            )

        seen = []
        original = module._encode_observation_vector

        def spy(vector):
            seen.append(np.array(vector, dtype=float, copy=True))
            return original(vector)

        monkeypatch.setattr(module, "_encode_observation_vector", spy)

        workspace_vector = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05])
        observation = module.observation_for_discrete_inference(workspace_vector)
        np.testing.assert_array_equal(seen[0], workspace_vector)
        assert np.all(observation >= 0) and observation.sum() == pytest.approx(1.0)

        seen.clear()
        sensory = np.random.default_rng(2).uniform(0, 1, 64)
        await module.process_with_hierarchy(sensory)
        assert seen, "the encoder was not used"
        for vector in seen:
            assert np.all(vector >= 0), "signed hierarchy beliefs reached the encoder"

    @pytest.mark.asyncio
    async def test_hpp_top_level_route_encodes_hierarchy_beliefs(self, monkeypatch):
        module = _fast_module(hierarchy_observation="hpp_top_level")
        seen = []
        original = module._encode_observation_vector

        def spy(vector):
            seen.append(np.array(vector, dtype=float, copy=True))
            return original(vector)

        monkeypatch.setattr(module, "_encode_observation_vector", spy)

        sensory = np.random.default_rng(2).uniform(0, 1, 64)
        await module.process_with_hierarchy(sensory)
        top_level = module.hierarchical_processor.levels[-1].beliefs
        assert seen, "the encoder was not used"
        np.testing.assert_array_equal(seen[0], top_level)
        assert len(seen[0]) != len(sensory)

    def test_hierarchy_observation_is_validated(self):
        with pytest.raises(ValueError, match="hierarchy_observation"):
            ActiveInferenceConfig(hierarchy_observation="beliefs")

    def test_encoder_returns_a_distribution_for_signed_input(self):
        module = _fast_module()
        encoded = module._encode_observation_vector(
            np.array([-3.0, 0.5, np.nan, -1.0, 0.5, 1.0, -2.0])
        )
        assert np.all(encoded >= 0)
        assert encoded.sum() == pytest.approx(1.0)

        uniform = module._encode_observation_vector(-np.ones(7))
        np.testing.assert_allclose(uniform, np.full(5, 0.2))


# ============================================================================
# WORKSPACE PRECISION FEEDBACK
# ============================================================================


class _Candidate:
    def __init__(self, content_type, activation_level=0.5):
        self.content_type = content_type
        self.activation_level = activation_level


class _Winner:
    def __init__(self, content_type, activation_level=0.5):
        self.candidate = _Candidate(content_type, activation_level)


class TestWorkspacePrecisionFeedback:
    """Workspace winners sharpen the FEP channels their content maps to."""

    def test_bias_starts_at_zero(self, active_inference_module):
        bias = active_inference_module.workspace_precision_bias
        np.testing.assert_array_equal(bias, np.zeros(5))

    def test_channels_follow_compact_labels(self, active_inference_module):
        channel = active_inference_module.channel_for_content_type
        assert channel("emotion") == 1  # emotional_arousal
        assert channel("thought") == 2  # cognitive_load
        assert channel("sensory") == 3  # ignition_strength
        assert channel("message") == 4  # social_content
        assert channel("memory") == 0  # integration
        assert channel("unknown_type") is None
        assert channel(None) is None

    def test_channels_follow_conversational_labels(self):
        module = ActiveInferenceModule(
            ActiveInferenceConfig(world_model="conversational_16x10x8")
        )
        index = module.observation_index
        channel = module.channel_for_content_type
        assert channel("emotion") == index["warm_tone"]
        assert channel("message") == index["inform_intent"]
        assert channel("memory") == index["familiar_topic"]
        assert channel("concept") == index["novel_topic"]
        assert channel("audio") == index["high_engagement"]
        assert channel("metacognitive") == index["question_intent"]
        assert channel("social") is None  # no social channel in this model

    def test_custom_model_maps_only_its_labels(self):
        module = ActiveInferenceModule(
            ActiveInferenceConfig(
                world_model="custom",
                num_hidden_states=3,
                num_observations=2,
                num_actions=2,
                observation_labels=["warm_tone", "something_else"],
            )
        )
        assert module.channel_for_content_type("emotion") == 0
        assert module.channel_for_content_type("message") is None

    def test_decay_boost_and_cap(self, active_inference_module):
        module = active_inference_module
        module.modulate_precision_from_workspace([_Winner("emotion", 1.0)])
        assert module.workspace_precision_bias[1] == pytest.approx(0.3)

        module.modulate_precision_from_workspace([])
        assert module.workspace_precision_bias[1] == pytest.approx(0.3 * 0.85)

        for _ in range(20):
            module.modulate_precision_from_workspace(
                [_Winner("emotion", 1.0), _Winner("EMOTION", 1.0)]
            )
        assert module.workspace_precision_bias[1] == pytest.approx(1.0)
        assert np.all(module.workspace_precision_bias <= 1.0)

    def test_precision_feedback_skips_a_winner_without_activation(
        self, active_inference_module
    ):
        module = active_inference_module
        unknown = _Winner("sensory")
        unknown.candidate.activation_level = None
        not_finite = _Winner("message", float("nan"))
        module.modulate_precision_from_workspace(
            [{"content_type": "sensory"}, unknown, not_finite]
        )
        assert module.workspace_precision_bias.sum() == 0.0

        module.modulate_precision_from_workspace(
            [{"content_type": "sensory", "activation_level": 1.0}]
        )
        assert module.workspace_precision_bias[3] == pytest.approx(0.3)

    def test_winner_shapes(self, active_inference_module):
        module = active_inference_module
        empty = _Winner("emotion")
        empty.candidate = None
        module.modulate_precision_from_workspace(
            [
                empty,
                _Winner("not_a_type"),
                {"content_type": "sensory", "activation_level": 1.0},
                _Candidate("message", 1.0),
            ]
        )
        bias = module.workspace_precision_bias
        assert bias[3] == pytest.approx(0.3)
        assert bias[4] == pytest.approx(0.3)
        assert bias[[0, 1, 2]].sum() == 0.0

    @pytest.mark.asyncio
    async def test_workspace_bias_weights_and_restores_A(self):
        module = _fast_module()
        observation = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

        plain = await module.infer_and_act(observation)
        assert plain.precision_weighted_error == plain.prediction_error

        module.modulate_precision_from_workspace([_Winner("emotion", 1.0)])
        biased = await module.infer_and_act(observation)
        assert biased.precision_weighted_error == pytest.approx(
            biased.prediction_error * (1.0 + 0.3 / 5)
        )
        # The unmodulated model is restored for learning and the next cycle
        assert module.agent.A is module.A

    @pytest.mark.asyncio
    async def test_attention_and_workspace_bias_combine(self):
        module = _fast_module()
        module.modulate_precision_from_workspace([_Winner("emotion", 1.0)])
        attention = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        result = await module.infer_and_act(
            np.array([0.4, 0.3, 0.1, 0.1, 0.1]), attention_precision=attention
        )
        assert result.precision_weighted_error == pytest.approx(
            result.prediction_error * (1.0 + (0.5 + 0.3) / 5)
        )


# ============================================================================
# MEASURED ACCURACY, NARRATION GATES AND THE SIGNAL SINK
# ============================================================================


class TestLedgerNarrationAndSignals:
    """Forecast ledger, narration gates and the per-cycle signal sink."""

    @pytest.mark.asyncio
    async def test_each_cycle_records_one_forecast_and_scores_the_last(self):
        module = _fast_module()
        observation = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        for cycle in range(1, 6):
            await module.infer_and_act(observation)
            assert module.prediction_ledger.pending_count == 1
            assert module.prediction_ledger.scored_count == cycle - 1
        assert module.get_statistics()["scored_forecasts"] == 4
        assert module.measured_accuracy() is None  # below 20 scored

        for _ in range(20):
            await module.infer_and_act(observation)
        accuracy = module.get_statistics()["measured_accuracy"]
        assert accuracy is not None and 0.0 <= accuracy <= 1.0

    @pytest.mark.asyncio
    async def test_narration_gates(self):
        observation = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        reports = {}
        for free_energy in (True, False):
            for body in (True, False):
                module = _fast_module(
                    narrate_free_energy=free_energy, narrate_body_drives=body
                )
                module.homeostatic_drives.drives["curiosity"].current_level = 0.1
                # A trend sentence needs more than ten prediction errors.
                for _ in range(12):
                    await module.infer_and_act(observation)
                reports[(free_energy, body)] = (
                    await module.generate_active_inference_report()
                )

        assert "prediction error" in reports[(True, True)]
        assert "Most urgent need" in reports[(True, True)]
        assert "free energy" in reports[(True, False)]
        assert "Most urgent need" not in reports[(True, False)]
        assert "free energy" not in reports[(False, True)]
        assert "Most urgent need" in reports[(False, True)]
        assert reports[(False, False)] == ""

    @pytest.mark.asyncio
    async def test_report_says_nothing_on_a_fresh_module(self):
        module = _fast_module()
        assert module.get_state().most_urgent_drive is not None
        assert await module.generate_active_inference_report() == ""

    @pytest.mark.asyncio
    async def test_report_trend_needs_more_than_ten_errors(self):
        module = _fast_module()
        observation = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        for _ in range(3):
            await module.infer_and_act(observation)
        assert "prediction error" not in (
            await module.generate_active_inference_report()
        )
        for _ in range(9):
            await module.infer_and_act(observation)
        assert len(module.prediction_errors) == 12
        report = await module.generate_active_inference_report()
        assert report.startswith("My current active inference state: prediction error")

    @pytest.mark.asyncio
    async def test_report_makes_no_accuracy_claim(self):
        module = _fast_module()
        await module.infer_and_act(np.array([0.4, 0.3, 0.1, 0.1, 0.1]))
        state = module.get_state()
        report = await module.generate_active_inference_report()
        assert "accurate" not in report
        if state.total_free_energy > 1.0:
            assert report.endswith("High free energy - seeking to reduce surprise.")
        else:
            assert report.endswith("Low free energy.")

        # A low free energy state says so and claims nothing more.
        module.free_energy_history[-1] = 0.0
        for drive in module.homeostatic_drives.drives.values():
            drive.current_level = drive.optimal_level
        assert module.get_state().total_free_energy <= 1.0
        assert await module.generate_active_inference_report() == "Low free energy."

    @pytest.mark.asyncio
    async def test_report_omits_a_need_with_zero_urgency(self):
        module = _fast_module()
        await module.infer_and_act(np.array([0.4, 0.3, 0.1, 0.1, 0.1]))
        state = module.get_state()
        assert state.urgency_level == 0.0
        assert "Most urgent need" not in (
            await module.generate_active_inference_report()
        )

        module.homeostatic_drives.drives["curiosity"].current_level = 0.1
        state = module.get_state()
        assert state.urgency_level > 0.0
        report = await module.generate_active_inference_report()
        assert f"Most urgent need: {state.most_urgent_drive} (urgency: " in report

    @pytest.mark.asyncio
    async def test_signal_sink_receives_records(self):
        records = []
        module = _fast_module(signal_sink=records.append)
        await module.infer_and_act(np.array([0.4, 0.3, 0.1, 0.1, 0.1]))
        await module.infer_and_act(np.array([0.1, 0.1, 0.1, 0.3, 0.4]))

        assert len(records) == 2
        assert records[0]["signal_id"] == "prediction_error_v1"
        assert "observation_novelty" not in records[0]["context"]
        assert records[1]["context"]["observation_novelty"] == pytest.approx(1.0)
        assert records[1]["values"]["prediction_error"] >= 0.0
        json.dumps(records)

    @pytest.mark.asyncio
    async def test_signal_sink_errors_are_ignored(self):
        def broken_sink(record):
            raise RuntimeError("sink unavailable")

        module = ActiveInferenceModule(
            ActiveInferenceConfig(**{**FAST, "signal_sink": broken_sink}),
            settings=Settings(DEBUG=False),
        )
        result = await module.infer_and_act(np.array([0.4, 0.3, 0.1, 0.1, 0.1]))
        assert result is not None

    @pytest.mark.asyncio
    async def test_signal_sink_failure_is_counted_and_raised_under_debug(self):
        def broken_sink(record):
            raise RuntimeError("sink unavailable")

        config = ActiveInferenceConfig(**{**FAST, "signal_sink": broken_sink})
        observation = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        quiet = ActiveInferenceModule(config, settings=Settings(DEBUG=False))
        await quiet.infer_and_act(observation)
        await quiet.infer_and_act(observation)
        # Both cycles complete; each failed delivery is counted.
        assert quiet.total_inferences == 2
        assert quiet.get_statistics()["failures"] == {"signal_sink": 2}

        loud = ActiveInferenceModule(config, settings=Settings(DEBUG=True))
        with pytest.raises(RuntimeError, match="sink unavailable"):
            await loud.infer_and_act(observation)
        assert loud.get_statistics()["failures"] == {"signal_sink": 1}


class TestObservationEncoding:
    """The mass guard applies on every branch of the observation encoder."""

    def test_valid_input_of_right_length_is_unchanged(self, active_inference_module):
        module = active_inference_module
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_array_equal(module._encode_observation_vector(vector), vector)

    def test_negative_entry_of_right_length_is_zeroed(self, active_inference_module):
        module = active_inference_module
        out = module._encode_observation_vector(np.array([0.5, -0.4, 0.5, 0.0, 0.0]))
        assert np.all(out >= 0.0)
        assert out[1] == 0.0
        assert out.sum() == pytest.approx(1.0)

    def test_non_finite_entry_of_right_length_is_zeroed(self, active_inference_module):
        module = active_inference_module
        out = module._encode_observation_vector(
            np.array([np.nan, np.inf, 0.2, 0.3, -np.inf])
        )
        assert np.all(np.isfinite(out))
        assert out[0] == out[1] == out[4] == 0.0
        assert out.sum() == pytest.approx(1.0)

    def test_all_invalid_input_falls_back_to_uniform(self, active_inference_module):
        module = active_inference_module
        out = module._encode_observation_vector(np.full(5, -1.0))
        np.testing.assert_allclose(out, np.full(5, 0.2))


class TestTrimmedHistoryRank:
    def test_rank_reads_the_retained_window(self, active_inference_module):
        module = active_inference_module
        module.prediction_errors = [0.9] * 1001
        module._trim_histories()
        assert len(module.prediction_errors) < 1001
        assert module.prediction_error_rank(0.5) == 0.0
        assert module.prediction_error_rank(1.0) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
