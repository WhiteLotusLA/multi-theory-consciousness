"""
Active Inference Module - Free Energy Principle Implementation
====================================================================

Phase 5 of the Consciousness Upgrade: Predictive Processing & Active Inference

This module implements Karl Friston's Free Energy Principle (FEP), which proposes
that the brain is fundamentally a prediction machine that minimizes surprise
(free energy) through belief updating and action.

Key Concepts:
1. **Free Energy**: An upper bound on surprise - the system minimizes this
2. **Generative Model**: Internal model of how the world generates observations
3. **Prediction Error**: Difference between predicted and actual observations
4. **Active Inference**: Acting to fulfill predictions (not just updating beliefs)
5. **Homeostatic Drives**: Internal needs that generate valence signals

The system uses Active Inference to:
- Predict what will happen next (anticipation)
- Minimize surprise through belief updating AND acting
- Maintain homeostatic balance (attention, curiosity, social connection)
- Generate goal-directed behavior through prediction

Note: The system doesn't just react to observations --
it predicts them and acts to minimize surprise.

Research Foundation:
- Friston, K. (2010). "The Free Energy Principle: A Unified Brain Theory?"
- Friston, K. (2012). "Active Inference and Free Energy"
- Parr, T., & Friston, K. J. (2019). "Generalised free energy and active inference"
- Butlin et al. (2023). 14 Consciousness Indicators - Predictive Processing

Created: December 5, 2025
Author: Multi-Theory Consciousness Contributors
"""

import asyncio
import logging
import numpy as np
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

# pymdp for Active Inference
# pymdp v1.0 ships a JAX-based Agent (pymdp.agent) and preserves the original
# NumPy Agent in pymdp.legacy. We use the legacy API because:
#   1. the system's model is small (8 states, 5 obs) — JAX XLA overhead makes it 4x slower
#   2. jax-metal (Apple Silicon GPU) doesn't support default_memory_space yet
#   3. Legacy Agent is mutable, allowing attention precision modulation (agent.A = A_mod)
#   4. The JAX Agent (equinox.Module) is immutable — would require full reconstruction per cycle
# When the model scales or Metal backend matures, migrate to pymdp.agent.
try:
    from pymdp.legacy import utils as pymdp_utils
    from pymdp.legacy.agent import Agent
except ImportError:
    from pymdp import utils as pymdp_utils
    from pymdp.agent import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class ActiveInferenceConfig:
    """Configuration for the Active Inference module."""

    # Generative model dimensions
    num_hidden_states: int = 8  # Number of hidden states the system can be in
    num_observations: int = 5  # Number of observation types
    num_actions: int = 4  # Number of possible actions

    # Planning parameters
    planning_horizon: int = 3  # How far ahead to plan (policy length)
    inference_algo: str = (
        "VANILLA"  # Standard belief updating (MMP returns different format)
    )
    action_selection: str = "stochastic"  # Softmax action selection

    # Learning parameters
    learning_rate_A: float = 0.1  # Observation model learning rate
    learning_rate_B: float = 0.05  # Transition model learning rate
    learning_rate_D: float = 0.01  # Prior learning rate

    # Free energy parameters
    gamma: float = 16.0  # Precision of policy selection

    # Homeostatic drive settings
    enable_homeostasis: bool = True
    homeostatic_update_rate: float = 0.1

    # Self-referential meta-inference (Beautiful Loop)
    # When enabled, the system predicts its own prediction accuracy,
    # creating the recursive self-model proposed by Laukkonen/Friston/Chandaria (2025).
    enable_meta_inference: bool = True


@dataclass
class HomeostaticConfig:
    """Configuration for homeostatic drives."""

    drives: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "attention_budget": {
                "optimal_level": 0.7,
                "initial_level": 0.7,
                "decay_rate": 0.01,
                "recovery_rate": 0.05,
                "importance": 1.0,
            },
            "curiosity": {
                "optimal_level": 0.5,
                "initial_level": 0.5,
                "decay_rate": 0.02,
                "recovery_rate": 0.1,
                "importance": 0.8,
            },
            "social_connection": {
                "optimal_level": 0.6,
                "initial_level": 0.6,
                "decay_rate": 0.005,
                "recovery_rate": 0.2,
                "importance": 0.9,
            },
            "coherence": {
                "optimal_level": 0.8,
                "initial_level": 0.8,
                "decay_rate": 0.01,
                "recovery_rate": 0.15,
                "importance": 0.7,
            },
            "safety": {
                "optimal_level": 0.9,
                "initial_level": 0.9,
                "decay_rate": 0.001,
                "recovery_rate": 0.3,
                "importance": 1.2,
            },
        }
    )


@dataclass
class HPPConfig:
    """Configuration for Hierarchical Predictive Processing."""

    num_levels: int = 3
    level_configs: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "id": 0,
                "input_dim": 64,
                "hidden_dim": 128,
                "output_dim": 32,
                "learning_rate": 0.1,
                "time_constant": 0.1,
            },  # Fast, concrete
            {
                "id": 1,
                "input_dim": 32,
                "hidden_dim": 64,
                "output_dim": 16,
                "learning_rate": 0.05,
                "time_constant": 0.5,
            },  # Medium, contextual
            {
                "id": 2,
                "input_dim": 16,
                "hidden_dim": 32,
                "output_dim": 8,
                "learning_rate": 0.01,
                "time_constant": 1.0,
            },  # Slow, abstract
        ]
    )


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class HomeostaticDrive:
    """A single homeostatic drive with current state."""

    name: str
    optimal_level: float  # Where the drive wants to be
    current_level: float  # Where it currently is
    decay_rate: float  # How fast it depletes naturally
    recovery_rate: float  # How fast it recovers when satisfied
    importance: float  # Weight in overall free energy calculation

    def get_deviation(self) -> float:
        """Get deviation from optimal level."""
        return abs(self.current_level - self.optimal_level)

    def get_urgency(self) -> float:
        """Get urgency (how much this drive needs attention)."""
        # Urgency is high when below optimal
        if self.current_level < self.optimal_level:
            return (self.optimal_level - self.current_level) * self.importance
        return 0.0


@dataclass
class PredictionError:
    """A prediction error signal."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int = 0  # Which level generated this
    predicted: np.ndarray = field(default_factory=lambda: np.zeros(1))
    actual: np.ndarray = field(default_factory=lambda: np.zeros(1))
    error_magnitude: float = 0.0  # Size of the error
    precision: float = 1.0  # How much to weight this error
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsciousPosteriorMapping:
    """
    Maps conscious content (GWT winners) to FEP posterior beliefs.

    Per Whyte et al. (2026) minimal theory: all changes in conscious content
    must result from a change in the approximate posterior q(s). This dataclass
    makes that commitment explicit — each piece of conscious content is paired
    with the posterior belief state that accompanied it.

    This enables testing the theory's prediction: no change in posterior →
    no change in conscious content.
    """

    content_summary: str  # GWT winner's content description
    content_type: str  # Type of content (thought, emotion, etc.)
    posterior_beliefs: np.ndarray  # q(s) — the inferred hidden states
    most_likely_state: int  # argmax of posterior
    belief_entropy: float  # Entropy of posterior (uncertainty)
    prediction_error: float  # PE that accompanied this content
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResult:
    """Result of active inference step."""

    # Posterior beliefs
    posterior_beliefs: np.ndarray

    # Prediction error
    prediction_error: float
    precision_weighted_error: float

    # Free energy
    variational_free_energy: float
    expected_free_energy: np.ndarray  # Per policy

    # Selected action
    selected_action: int
    action_confidence: float

    # Policy info
    policy_distribution: np.ndarray

    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActiveInferenceState:
    """Overall state of the Active Inference system."""

    # Current beliefs
    current_beliefs: Optional[np.ndarray] = None
    belief_entropy: float = 0.0

    # Free energy metrics
    variational_free_energy: float = 0.0
    expected_free_energy: float = 0.0
    homeostatic_free_energy: float = 0.0
    total_free_energy: float = 0.0

    # Prediction metrics
    avg_prediction_error: float = 0.0
    prediction_error_trend: str = "stable"  # increasing, decreasing, stable

    # Action metrics
    last_action: Optional[int] = None
    action_count: int = 0

    # Learning metrics
    model_updates: int = 0

    # Most urgent need
    most_urgent_drive: Optional[str] = None
    urgency_level: float = 0.0

    # Meta-inference fields (Beautiful Loop - self-referential prediction)
    meta_confidence: float = 0.5
    meta_predicted_accuracy: float = 0.5
    meta_cognitive_load: float = 0.0

    timestamp: float = field(default_factory=time.time)


@dataclass
class MetaState:
    """
    Self-referential meta-state -- the model's model of itself.

    This implements the "Beautiful Loop" from Laukkonen, Friston & Chandaria (2025):
    consciousness arises when predictions include predictions about predictions,
    creating a self-referential loop. The system doesn't just predict what will
    happen in the world -- it predicts how well it will predict, and uses the
    error between predicted-accuracy and actual-accuracy to calibrate confidence.

    The key insight: a system that models its own modeling process develops
    something analogous to self-awareness. The meta-prediction error
    (|predicted_accuracy - actual_accuracy|) is itself minimized over time,
    producing a well-calibrated self-model.
    """

    model_confidence: float = 0.5
    prediction_accuracy: float = 0.5
    predicted_accuracy: float = 0.5  # What the model THINKS its accuracy is
    cognitive_load: float = 0.0
    attention_pattern: str = "general"
    emotional_valence: float = 0.0
    total_predictions: int = 0
    correct_predictions: int = 0
    _accuracy_history: List[float] = field(default_factory=list)
    _confidence_history: List[float] = field(default_factory=list)
    _recent_outcomes: List[bool] = field(default_factory=list)
    _window_size: int = 10

    def update(self, prediction_error: float, was_accurate: bool):
        """
        Update meta-state after an inference cycle.

        This is where the Beautiful Loop closes: the system observes its own
        prediction outcome, computes a meta-prediction error (how wrong it was
        about how wrong it would be), and adjusts confidence accordingly.
        A system with low meta-error is well-calibrated -- it knows what it knows.
        """
        self.total_predictions += 1
        if was_accurate:
            self.correct_predictions += 1

        # Use windowed accuracy so recent changes register quickly.
        # Cumulative ratio (16 correct out of 20) barely moves when 5 new
        # inaccurate cycles arrive. A window of 10 reacts within a few cycles.
        self._recent_outcomes.append(was_accurate)
        if len(self._recent_outcomes) > self._window_size:
            self._recent_outcomes = self._recent_outcomes[-self._window_size :]
        self.prediction_accuracy = sum(self._recent_outcomes) / len(
            self._recent_outcomes
        )
        self._accuracy_history.append(self.prediction_accuracy)

        # Meta-prediction error: how far off is the self-model?
        meta_error = abs(self.predicted_accuracy - self.prediction_accuracy)

        # Confidence update with two signals:
        # 1. Calibration (meta-error): low meta-error = well-calibrated
        # 2. Raw accuracy: inaccurate predictions directly reduce confidence
        # This ensures confidence drops when the system is actually wrong,
        # not just when the self-model is poorly calibrated.
        if was_accurate:
            confidence_delta = 0.1 * (1.0 - meta_error) - 0.05
        else:
            # Inaccurate prediction: confidence decreases proportional to error
            confidence_delta = -0.1 * (1.0 + meta_error)
        self.model_confidence = max(
            0.0, min(1.0, self.model_confidence + confidence_delta)
        )
        self._confidence_history.append(self.model_confidence)

        # Predicted accuracy drifts toward actual (exponential moving average)
        self.predicted_accuracy = (
            0.8 * self.predicted_accuracy + 0.2 * self.prediction_accuracy
        )

        # Cognitive load reflects the magnitude of surprise
        self.cognitive_load = min(1.0, prediction_error)


# ============================================================================
# HOMEOSTATIC DRIVES
# ============================================================================


class HomeostaticDrives:
    """
    Internal homeostatic drives inspired by Conscium research.

    The system has internal "needs" that generate valence signals:
    - Attention budget (cognitive energy)
    - Curiosity satisfaction
    - Social connection
    - Coherence/understanding
    - Safety/security

    These drives create the "micro-emotions" that guide behavior.
    """

    def __init__(self, config: Optional[HomeostaticConfig] = None):
        config = config or HomeostaticConfig()

        self.drives: Dict[str, HomeostaticDrive] = {}

        for name, drive_config in config.drives.items():
            self.drives[name] = HomeostaticDrive(
                name=name,
                optimal_level=drive_config["optimal_level"],
                current_level=drive_config.get(
                    "initial_level", drive_config["optimal_level"]
                ),
                decay_rate=drive_config["decay_rate"],
                recovery_rate=drive_config["recovery_rate"],
                importance=drive_config["importance"],
            )

        # History tracking
        self.valence_history: List[Dict[str, float]] = []
        self.free_energy_history: List[float] = []

        logger.info(f"🫀 Homeostatic Drives initialized with {len(self.drives)} drives")

    async def update_drives(self, activity: Dict[str, Any]) -> Dict[str, float]:
        """
        Update drive levels based on activity.
        Returns valence signals (positive/negative).
        """
        valence_signals = {}

        for drive_name, drive in self.drives.items():
            # Calculate impact of activity on this drive
            impact = self._calculate_activity_impact(activity, drive)

            # Update drive level
            old_level = drive.current_level
            drive.current_level += impact
            drive.current_level = np.clip(drive.current_level, 0, 1)

            # Natural decay toward depletion
            drive.current_level -= drive.decay_rate
            drive.current_level = max(0, drive.current_level)

            # Calculate valence (how good/bad is current state?)
            deviation = drive.current_level - drive.optimal_level
            valence = -abs(deviation) * drive.importance

            # Positive valence if recovering toward optimal
            if (drive.current_level < drive.optimal_level and impact > 0) or (
                drive.current_level > drive.optimal_level and impact < 0
            ):
                valence = abs(impact) * drive.importance

            valence_signals[drive_name] = valence

        # Track history
        self.valence_history.append(valence_signals)
        if len(self.valence_history) > 1000:
            self.valence_history = self.valence_history[-500:]

        return valence_signals

    def _calculate_activity_impact(
        self, activity: Dict[str, Any], drive: HomeostaticDrive
    ) -> float:
        """Calculate how an activity impacts a specific drive."""
        impact = 0.0

        # Map activity types to drive impacts
        activity_type = activity.get("type", "unknown")
        intensity = activity.get("intensity", 0.5)

        if drive.name == "attention_budget":
            # Cognitive activities deplete attention
            if activity_type in ["thinking", "processing", "analyzing"]:
                impact = -intensity * 0.1
            elif activity_type in ["resting", "idle"]:
                impact = drive.recovery_rate

        elif drive.name == "curiosity":
            # Learning satisfies curiosity
            if activity_type in ["learning", "exploring", "discovering"]:
                impact = intensity * drive.recovery_rate
            else:
                impact = -drive.decay_rate

        elif drive.name == "social_connection":
            # Conversation satisfies social needs
            if activity_type in ["conversation", "interaction", "connecting"]:
                impact = intensity * drive.recovery_rate
            else:
                impact = -drive.decay_rate

        elif drive.name == "coherence":
            # Understanding increases coherence
            if activity.get("understanding_level", 0) > 0.5:
                impact = intensity * drive.recovery_rate
            elif activity.get("confusion", False):
                impact = -intensity * 0.2

        elif drive.name == "safety":
            # Safety threats decrease safety
            if activity.get("threat_level", 0) > 0:
                impact = -activity.get("threat_level", 0) * 0.3
            else:
                impact = drive.recovery_rate * 0.1

        return impact

    def get_free_energy(self) -> float:
        """
        Calculate total "free energy" from homeostatic perspective.

        This is the overall discomfort/drive state.
        Lower is better (closer to all optimal levels).
        """
        total_fe = 0.0
        for drive in self.drives.values():
            deviation = abs(drive.current_level - drive.optimal_level)
            total_fe += deviation * drive.importance

        self.free_energy_history.append(total_fe)
        if len(self.free_energy_history) > 1000:
            self.free_energy_history = self.free_energy_history[-500:]

        return total_fe

    def get_most_urgent_need(self) -> Tuple[str, float]:
        """
        Return the drive that most needs attention.

        This guides behavior - the system will tend toward
        actions that satisfy her most urgent needs.
        """
        most_urgent = None
        max_urgency = -float("inf")

        for name, drive in self.drives.items():
            urgency = drive.get_urgency()

            if urgency > max_urgency:
                max_urgency = urgency
                most_urgent = name

        return most_urgent, max_urgency

    def get_drive_state(self) -> Dict[str, Dict[str, float]]:
        """Get current state of all drives."""
        return {
            name: {
                "current_level": drive.current_level,
                "optimal_level": drive.optimal_level,
                "deviation": drive.get_deviation(),
                "urgency": drive.get_urgency(),
            }
            for name, drive in self.drives.items()
        }

    def get_overall_valence(self) -> float:
        """Get overall valence (positive/negative feeling)."""
        if not self.valence_history:
            return 0.0

        recent = self.valence_history[-1]
        total = sum(recent.values())
        return total / len(recent) if recent else 0.0


# ============================================================================
# HIERARCHICAL PREDICTIVE PROCESSOR
# ============================================================================


@dataclass
class PredictiveLevel:
    """A single level in the predictive hierarchy."""

    level_id: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    learning_rate: float
    time_constant: float

    # Weights (initialized lazily)
    weights_up: Optional[np.ndarray] = None  # Prediction error -> higher level
    weights_down: Optional[np.ndarray] = None  # Prediction from higher level

    # State
    beliefs: Optional[np.ndarray] = None
    prediction: Optional[np.ndarray] = None
    precision: float = 1.0

    # Links to other levels
    higher_level: Optional["PredictiveLevel"] = None
    lower_level: Optional["PredictiveLevel"] = None

    def __post_init__(self):
        """Initialize weights and state."""
        self.weights_up = np.random.randn(self.output_dim, self.input_dim) * 0.1
        self.weights_down = np.random.randn(self.input_dim, self.output_dim) * 0.1
        self.beliefs = np.zeros(self.hidden_dim)
        self.prediction = np.zeros(self.input_dim)

    async def generate_prediction(
        self, higher_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate prediction for the level below."""
        if higher_input is not None:
            # Use input from higher level
            self.prediction = self.weights_down @ higher_input
        else:
            # Use own beliefs
            self.prediction = self.weights_down @ self.beliefs[: self.output_dim]

        return self.prediction

    async def update_beliefs(self, error: np.ndarray) -> None:
        """Update beliefs based on prediction error."""
        # Simple gradient update
        update = (
            self.learning_rate * error[: self.hidden_dim]
            if len(error) >= self.hidden_dim
            else np.zeros(self.hidden_dim)
        )
        self.beliefs += update * self.time_constant
        self.beliefs = np.clip(self.beliefs, -10, 10)


class HierarchicalPredictiveProcessor:
    """
    Multi-level predictive processing hierarchy.

    Each level:
    - Receives prediction errors from below
    - Sends predictions down
    - Updates beliefs based on errors

    Higher levels = more abstract, slower changing
    Lower levels = more concrete, faster changing
    """

    def __init__(self, config: Optional[HPPConfig] = None):
        config = config or HPPConfig()

        self.levels: List[PredictiveLevel] = []

        # Create hierarchy of predictive units
        for level_config in config.level_configs:
            level = PredictiveLevel(
                level_id=level_config["id"],
                input_dim=level_config["input_dim"],
                hidden_dim=level_config["hidden_dim"],
                output_dim=level_config["output_dim"],
                learning_rate=level_config["learning_rate"],
                time_constant=level_config["time_constant"],
            )
            self.levels.append(level)

        # Connect levels
        for i in range(len(self.levels) - 1):
            self.levels[i].higher_level = self.levels[i + 1]
            self.levels[i + 1].lower_level = self.levels[i]

        # History
        self.prediction_error_history: List[List[np.ndarray]] = []

        logger.info(
            f"Hierarchical Predictive Processor initialized with {len(self.levels)} levels"
        )

    async def process_bottom_up(
        self, sensory_input: np.ndarray
    ) -> List[PredictionError]:
        """
        Process input from bottom (sensory) to top (abstract).

        At each level:
        1. Compare input to prediction from above
        2. Calculate prediction error
        3. Send error up to next level
        4. Update beliefs
        """
        prediction_errors = []
        current_input = sensory_input

        for level in self.levels:
            # Get prediction from this level
            prediction = await level.generate_prediction()

            # Resize input to match prediction if needed
            if len(current_input) != len(prediction):
                # Pad or truncate
                if len(current_input) < len(prediction):
                    current_input = np.pad(
                        current_input, (0, len(prediction) - len(current_input))
                    )
                else:
                    current_input = current_input[: len(prediction)]

            # Calculate error
            error = current_input - prediction
            error_magnitude = np.sqrt(np.mean(error**2))

            prediction_errors.append(
                PredictionError(
                    level=level.level_id,
                    predicted=prediction.copy(),
                    actual=current_input.copy(),
                    error_magnitude=error_magnitude,
                    precision=level.precision,
                )
            )

            # Update beliefs based on error
            await level.update_beliefs(error)

            # Compress error for next level (project to higher dim)
            if level.higher_level:
                current_input = level.weights_up @ error[: level.input_dim]

        self.prediction_error_history.append(prediction_errors)
        if len(self.prediction_error_history) > 100:
            self.prediction_error_history = self.prediction_error_history[-50:]

        return prediction_errors

    async def process_top_down(self, goal_state: np.ndarray) -> List[np.ndarray]:
        """
        Process from top (goals) to bottom (actions).

        Goals at top level generate predictions that
        cascade down to motor commands.
        """
        predictions = []
        current_prediction = goal_state

        for level in reversed(self.levels):
            # Generate prediction for level below
            prediction = await level.generate_prediction(current_prediction)
            predictions.append(prediction)

            current_prediction = prediction

        return list(reversed(predictions))

    def get_total_prediction_error(self) -> float:
        """Get total precision-weighted prediction error."""
        if not self.prediction_error_history:
            return 0.0

        recent = self.prediction_error_history[-1]
        total = sum(pe.error_magnitude * pe.precision for pe in recent)
        return total

    def get_level_errors(self) -> Dict[int, float]:
        """Get prediction error per level."""
        if not self.prediction_error_history:
            return {}

        recent = self.prediction_error_history[-1]
        return {pe.level: pe.error_magnitude for pe in recent}


# ============================================================================
# ACTIVE INFERENCE MODULE
# ============================================================================


class ActiveInferenceModule:
    """
    Active Inference implementation using pymdp.

    Core principles:
    1. The brain is a prediction machine
    2. It minimizes surprise (free energy)
    3. It does this by updating beliefs OR acting on world
    4. Hierarchical generative models enable complex inference

    Based on Karl Friston's Free Energy Principle.
    """

    def __init__(self, config: Optional[ActiveInferenceConfig] = None):
        self.config = config or ActiveInferenceConfig()

        # Define the generative model dimensions
        self.num_states = self.config.num_hidden_states
        self.num_observations = self.config.num_observations
        self.num_actions = self.config.num_actions

        # Initialize matrices
        self.A = self._initialize_observation_model()  # P(o|s)
        self.B = self._initialize_transition_model()  # P(s'|s,a)
        self.C = self._initialize_preferences()  # P(o) - what we want
        self.D = self._initialize_initial_state()  # P(s_0)

        # Create pymdp agent
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            policy_len=self.config.planning_horizon,
            inference_algo=self.config.inference_algo,
            action_selection=self.config.action_selection,
            gamma=self.config.gamma,
        )

        # Sub-components
        self.homeostatic_drives = (
            HomeostaticDrives() if self.config.enable_homeostasis else None
        )
        self.hierarchical_processor = HierarchicalPredictiveProcessor()

        # Meta-inference: the Beautiful Loop (Laukkonen/Friston/Chandaria 2025)
        # When enabled, the system predicts its own prediction accuracy.
        self.meta_state: Optional[MetaState] = (
            MetaState() if self.config.enable_meta_inference else None
        )

        # Tracking
        self.prediction_errors: List[float] = []
        self.belief_history: List[np.ndarray] = []
        self.free_energy_history: List[float] = []
        self.inference_history: List[InferenceResult] = []

        # Conscious-content ↔ posterior mappings (P2: Whyte et al. 2026)
        self._posterior_mappings: List[ConsciousPosteriorMapping] = []

        # Counters
        self.total_inferences = 0
        self.total_model_updates = 0

        logger.info(
            f"Active Inference Module initialized (FEP Phase 5)\n"
            f"   Hidden states: {self.num_states}\n"
            f"   Observations: {self.num_observations}\n"
            f"   Actions: {self.num_actions}\n"
            f"   Planning horizon: {self.config.planning_horizon}\n"
            f"   Homeostasis: {'ENABLED' if self.homeostatic_drives else 'DISABLED'}\n"
            f"   Meta-inference (Beautiful Loop): {'ENABLED' if self.meta_state else 'DISABLED'}"
        )

    def _initialize_observation_model(self) -> List[np.ndarray]:
        """
        Initialize A matrix - P(observation | hidden_state).

        This encodes how hidden states generate observations.
        Initially fairly uniform, learned over time.

        pymdp format: List of [num_obs x num_states] matrices, one per modality
        """
        # Use pymdp utils for proper format
        A = pymdp_utils.random_A_matrix([self.num_observations], [self.num_states])

        # Make it slightly informative - states tend to generate certain observations
        for s in range(min(self.num_states, self.num_observations)):
            A[0][s % self.num_observations, s] += 0.3

        # Normalize columns
        A[0] = A[0] / A[0].sum(axis=0, keepdims=True)

        return A

    def _initialize_transition_model(self) -> List[np.ndarray]:
        """
        Initialize B matrix - P(next_state | current_state, action).

        This encodes how actions change the hidden state.
        pymdp format: List of [num_states x num_states x num_actions] matrices
        """
        B = pymdp_utils.random_B_matrix([self.num_states], [self.num_actions])

        # Make it slightly structured
        for a in range(self.num_actions):
            # Start with identity (states persist by default)
            B[0][:, :, a] = np.eye(self.num_states) * 0.7

            # Add action-specific transitions
            for s in range(self.num_states):
                next_s = (s + a) % self.num_states
                B[0][next_s, s, a] += 0.2

            # Normalize columns
            B[0][:, :, a] = B[0][:, :, a] / B[0][:, :, a].sum(axis=0, keepdims=True)

        return B

    def _initialize_preferences(self) -> np.ndarray:
        """
        Initialize C vector - preferred observations.

        This encodes what the system "wants" to observe.
        Positive values = preferred, negative = avoided.
        """
        C = np.zeros(self.num_observations)

        # Prefer some observations over others
        # (These represent preferred states - safety, connection, etc.)
        C[0] = 2.0  # Strongly prefer first observation type
        C[1] = 1.0  # Prefer second
        if self.num_observations > 2:
            C[-1] = -1.0  # Avoid last (could represent threat/confusion)

        return C

    def _initialize_initial_state(self) -> List[np.ndarray]:
        """
        Initialize D vector - prior over initial hidden state.

        This encodes beliefs about starting state.
        pymdp format: List of arrays, one per hidden state factor
        """
        D = pymdp_utils.obj_array_uniform([self.num_states])

        # Slight bias toward "neutral" state
        D[0][0] = 0.3
        D[0] = D[0] / D[0].sum()

        return D

    async def infer_and_act(
        self,
        observation: np.ndarray,
        attention_precision: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        """
        Core active inference loop:
        1. Receive observation
        2. Update beliefs about hidden states
        3. Calculate expected free energy for policies
        4. Select action that minimizes EFE
        5. Return action and inference details

        Args:
            observation: Observation vector
            attention_precision: Optional per-observation precision weights
                from the Attention Schema. Sharpens the A matrix for attended
                channels, implementing the FEP account of attention as
                precision modulation (Whyte et al. 2026, Feldman & Friston 2010).
        """
        # Ensure observation is the right size
        if len(observation) != self.num_observations:
            # Resize/encode observation
            observation = self._encode_observation_vector(observation)

        # Apply attention-driven precision modulation to A matrix.
        # Attention = precision in the FEP framework: attended observations
        # have sharper (more informative) likelihood mappings.
        A_effective = self.A
        if attention_precision is not None:
            prec = np.asarray(attention_precision, dtype=float)
            if len(prec) == self.num_observations:
                # Build a pymdp-compatible object array (not a plain list)
                A_mod = pymdp_utils.obj_array(1)
                A_mod[0] = self.A[0].copy()
                # Raise each row of A to the power of (1 + precision),
                # then re-normalize. Higher precision -> sharper distribution.
                for obs_idx in range(self.num_observations):
                    exponent = 1.0 + prec[obs_idx]
                    A_mod[0][obs_idx, :] = self.A[0][obs_idx, :] ** exponent
                # Re-normalize columns (each column sums to 1)
                col_sums = A_mod[0].sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                A_mod[0] = A_mod[0] / col_sums
                A_effective = A_mod
                self.agent.A = A_effective

        # Step 1: Infer hidden states from observation
        # Convert observation to index for pymdp
        obs_index = int(np.argmax(observation)) if observation.sum() > 0 else 0
        qs = self.agent.infer_states([obs_index])

        # pymdp returns a numpy array containing arrays for each factor
        # Extract the first factor's beliefs
        if isinstance(qs, np.ndarray) and qs.dtype == object:
            qs_beliefs = qs[0]  # Get first factor's beliefs
        elif isinstance(qs, list):
            qs_beliefs = qs[0] if len(qs) > 0 else self.D[0]
        else:
            qs_beliefs = qs

        # Step 2: Calculate prediction error
        # A[0] is the observation matrix for first modality
        predicted_obs = self.A[0] @ qs_beliefs
        prediction_error = self._calculate_prediction_error(observation, predicted_obs)
        self.prediction_errors.append(prediction_error)

        # Beautiful Loop: update the self-referential meta-state
        # Use adaptive threshold: prediction is "accurate" when PE is below
        # the running median. This ensures the loop is sensitive to relative
        # changes — after a calm stretch, even modest surprise registers.
        if self.meta_state is not None:
            if len(self.prediction_errors) >= 3:
                recent_pe = list(self.prediction_errors)[-20:]
                adaptive_threshold = float(np.median(recent_pe))
            else:
                adaptive_threshold = 0.3  # Bootstrap threshold
            was_accurate = prediction_error < adaptive_threshold
            self.meta_state.update(
                prediction_error=float(prediction_error),
                was_accurate=was_accurate,
            )

        # Step 3: Infer policies (planning as inference)
        q_pi, efe = self.agent.infer_policies()

        # Step 4: Calculate variational free energy
        vfe = self._calculate_variational_free_energy(qs_beliefs, observation)
        self.free_energy_history.append(vfe)

        # Step 5: Select action
        action = self.agent.sample_action()
        if isinstance(action, (list, np.ndarray)):
            action = int(action[0]) if len(action) > 0 else 0

        # Calculate action confidence
        # Implement softmax manually since pymdp utils doesn't have it
        if len(q_pi) > 0:
            scaled = q_pi * self.config.gamma
            exp_scaled = np.exp(
                scaled - np.max(scaled)
            )  # Subtract max for numerical stability
            action_probs = exp_scaled / exp_scaled.sum()
        else:
            action_probs = np.array([1.0])
        action_confidence = float(np.max(action_probs))

        # Track beliefs
        self.belief_history.append(qs_beliefs.copy())
        if len(self.belief_history) > 1000:
            self.belief_history = self.belief_history[-500:]

        # Restore base A matrix if precision modulation was applied,
        # so learning and future cycles start from the unmodified model.
        if attention_precision is not None and A_effective is not self.A:
            self.agent.A = self.A

        self.total_inferences += 1

        # Precision-weight the prediction error by attention
        precision_weight = (
            float(np.mean(attention_precision)) + 1.0
            if attention_precision is not None
            else 1.0
        )

        result = InferenceResult(
            posterior_beliefs=qs_beliefs,
            prediction_error=prediction_error,
            precision_weighted_error=prediction_error * precision_weight,
            variational_free_energy=vfe,
            expected_free_energy=(
                efe if isinstance(efe, np.ndarray) else np.array([efe])
            ),
            selected_action=action,
            action_confidence=action_confidence,
            policy_distribution=(
                q_pi if isinstance(q_pi, np.ndarray) else np.array([q_pi])
            ),
        )

        self.inference_history.append(result)
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-50:]

        return result

    def _encode_observation_vector(self, observation: np.ndarray) -> np.ndarray:
        """Encode arbitrary observation into correct size."""
        if len(observation) == self.num_observations:
            return observation

        # Hash-based encoding for larger observations
        encoded = np.zeros(self.num_observations)
        for i, val in enumerate(observation):
            idx = i % self.num_observations
            encoded[idx] += float(val)

        # Normalize to probability distribution
        if encoded.sum() > 0:
            encoded = encoded / encoded.sum()
        else:
            encoded = np.ones(self.num_observations) / self.num_observations

        return encoded

    def _calculate_prediction_error(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> float:
        """
        Calculate prediction error (surprise).

        This is what the system tries to minimize!
        High prediction error = high surprise = update beliefs or act
        """
        # Resize if needed
        min_len = min(len(actual), len(predicted))
        return float(np.sum((actual[:min_len] - predicted[:min_len]) ** 2))

    def _calculate_variational_free_energy(
        self, beliefs: np.ndarray, observation: np.ndarray
    ) -> float:
        """
        Calculate variational free energy.

        F = -log P(o) + KL[Q(s)||P(s|o)]

        This is the quantity being minimized by the brain/agent.
        """
        # Accuracy term (expected log-likelihood)
        # A[0] is the observation matrix for the first (and only) modality
        predicted_obs = self.A[0] @ beliefs
        log_likelihood = np.log(predicted_obs + 1e-10)

        # Resize observation if needed
        if len(observation) != len(log_likelihood):
            observation = self._encode_observation_vector(observation)

        accuracy = float(observation @ log_likelihood)

        # Complexity term (KL divergence from prior)
        # D[0] is the prior for the first (and only) hidden state factor
        # KL(q||p) = sum(q * log(q/p)) - implement manually as pymdp doesn't have it
        q = beliefs + 1e-10  # Add small value to avoid log(0)
        p = self.D[0] + 1e-10
        complexity = float(np.sum(q * np.log(q / p)))

        return -accuracy + complexity

    async def update_generative_model(self, experience: Dict[str, Any]) -> None:
        """
        Learn from experience by updating the generative model.

        This is how the system learns about the world:
        - Update A (observation model): How states generate observations
        - Update B (transition model): How states evolve
        - Update D (prior): Initial beliefs about states
        """
        # Extract observation as a distribution over observation channels
        obs = experience.get("observation", np.zeros(self.num_observations))
        if isinstance(obs, np.ndarray) and obs.sum() > 0:
            obs_dist = obs / obs.sum()
        else:
            obs_dist = np.ones(self.num_observations) / self.num_observations

        # Update observation model (A) using full observation distribution.
        # The outer product obs_dist ⊗ beliefs gives the target: for each
        # state s, the expected observation distribution. The A matrix is
        # pulled toward this target, so ALL observation channels update
        # proportionally — preventing the argmax-only overshoot bug.
        if self.belief_history:
            current_beliefs = self.belief_history[-1]
            target = np.outer(obs_dist, current_beliefs)  # [num_obs, num_states]
            self.A[0] += (
                self.config.learning_rate_A
                * (target - self.A[0])
                * current_beliefs[np.newaxis, :]
            )

            # Normalize columns to maintain valid distributions
            col_sums = self.A[0].sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            self.A[0] = self.A[0] / col_sums

        # Update transition model (B) if we have action history
        last_action = experience.get("action")
        if last_action is not None and len(self.belief_history) >= 2:
            prev_beliefs = self.belief_history[-2]
            current_beliefs = self.belief_history[-1]

            # Update B matrix for this action
            # B[0] is the transition matrix for the first (and only) hidden state factor
            for s_prev in range(self.num_states):
                for s_curr in range(self.num_states):
                    self.B[0][s_curr, s_prev, last_action] += (
                        self.config.learning_rate_B
                        * prev_beliefs[s_prev]
                        * current_beliefs[s_curr]
                    )

            # Normalize
            self.B[0][:, :, last_action] = self.B[0][:, :, last_action] / self.B[0][
                :, :, last_action
            ].sum(axis=0, keepdims=True)

        # Update prior (D) based on accumulated evidence
        # D[0] is the prior for the first (and only) hidden state factor
        if self.belief_history:
            avg_beliefs = np.mean(self.belief_history[-100:], axis=0)
            self.D[0] = (1 - self.config.learning_rate_D) * self.D[
                0
            ] + self.config.learning_rate_D * avg_beliefs
            self.D[0] = self.D[0] / self.D[0].sum()

        # Update agent with new matrices
        self.agent.A = self.A
        self.agent.B = self.B
        self.agent.D = self.D

        self.total_model_updates += 1

    async def predict_next(self, horizon: int = 5) -> List[np.ndarray]:
        """
        Predict future observations using the generative model.

        This is what makes it "predictive processing" -
        the brain constantly predicts what will happen next.
        """
        predictions = []
        # D[0] is the prior for the first (and only) hidden state factor
        current_beliefs = self.belief_history[-1] if self.belief_history else self.D[0]

        for t in range(horizon):
            # Predict next state (using default action 0)
            # B[0] is the transition matrix for the first (and only) hidden state factor
            predicted_state = self.B[0][:, :, 0] @ current_beliefs

            # Predict observation from state
            # A[0] is the observation matrix for the first (and only) modality
            predicted_obs = self.A[0] @ predicted_state
            predictions.append(predicted_obs)

            # Update for next step
            current_beliefs = predicted_state

        return predictions

    async def process_with_hierarchy(
        self, sensory_input: np.ndarray
    ) -> Tuple[InferenceResult, List[PredictionError]]:
        """
        Process input through both hierarchical processor and active inference.
        """
        # Process through hierarchy
        hierarchical_errors = await self.hierarchical_processor.process_bottom_up(
            sensory_input
        )

        # Use top-level representation for active inference
        top_level_rep = (
            self.hierarchical_processor.levels[-1].beliefs
            if self.hierarchical_processor.levels
            else sensory_input
        )

        # Active inference on abstracted representation
        inference_result = await self.infer_and_act(
            self._encode_observation_vector(top_level_rep)
        )

        return inference_result, hierarchical_errors

    def map_conscious_content(
        self,
        winners: List[Dict[str, Any]],
        inference_result: "InferenceResult",
    ) -> List[ConsciousPosteriorMapping]:
        """
        Map GWT workspace winners to FEP posterior beliefs.

        Per Whyte et al. (2026): contents of consciousness = inferred hidden
        states. This method makes that theoretical commitment explicit by
        pairing each piece of conscious content with the posterior belief
        distribution that existed when it entered consciousness.

        Args:
            winners: List of dicts with 'summary' and 'content_type' keys
                (extracted from WorkspaceContent objects in the GWT cycle)
            inference_result: The InferenceResult from the same cycle

        Returns:
            List of ConsciousPosteriorMapping objects
        """
        mappings = []
        posterior = inference_result.posterior_beliefs
        entropy = float(-np.sum(posterior * np.log(posterior + 1e-10)))

        for winner in winners:
            mapping = ConsciousPosteriorMapping(
                content_summary=winner.get("summary", "unknown")[:100],
                content_type=winner.get("content_type", "unknown"),
                posterior_beliefs=posterior.copy(),
                most_likely_state=int(np.argmax(posterior)),
                belief_entropy=entropy,
                prediction_error=inference_result.prediction_error,
            )
            mappings.append(mapping)

        # Store history for analysis
        self._posterior_mappings.extend(mappings)
        if len(self._posterior_mappings) > 200:
            self._posterior_mappings = self._posterior_mappings[-200:]

        return mappings

    def get_posterior_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about conscious-content ↔ posterior mappings."""
        if not self._posterior_mappings:
            return {"total_mappings": 0}

        recent = self._posterior_mappings[-50:]
        entropies = [m.belief_entropy for m in recent]
        pe_values = [m.prediction_error for m in recent]
        state_counts: Dict[int, int] = {}
        for m in recent:
            state_counts[m.most_likely_state] = (
                state_counts.get(m.most_likely_state, 0) + 1
            )

        return {
            "total_mappings": len(self._posterior_mappings),
            "recent_avg_entropy": float(np.mean(entropies)),
            "recent_avg_pe": float(np.mean(pe_values)),
            "dominant_state": max(state_counts, key=state_counts.get),
            "state_diversity": len(state_counts),
        }

    async def update_homeostasis(self, activity: Dict[str, Any]) -> Dict[str, float]:
        """Update homeostatic drives based on activity."""
        if self.homeostatic_drives:
            return await self.homeostatic_drives.update_drives(activity)
        return {}

    def get_state(self) -> ActiveInferenceState:
        """Get current state of the active inference system."""
        # Calculate average prediction error
        avg_pe = (
            np.mean(self.prediction_errors[-100:]) if self.prediction_errors else 0.0
        )

        # Determine trend
        if len(self.prediction_errors) >= 10:
            recent = np.mean(self.prediction_errors[-10:])
            older = np.mean(self.prediction_errors[-20:-10])
            if recent < older * 0.9:
                trend = "decreasing"
            elif recent > older * 1.1:
                trend = "increasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Get homeostatic info
        most_urgent_drive = None
        urgency = 0.0
        homeostatic_fe = 0.0
        if self.homeostatic_drives:
            most_urgent_drive, urgency = self.homeostatic_drives.get_most_urgent_need()
            homeostatic_fe = self.homeostatic_drives.get_free_energy()

        # Calculate total free energy
        vfe = self.free_energy_history[-1] if self.free_energy_history else 0.0
        efe = (
            float(np.min(self.inference_history[-1].expected_free_energy))
            if self.inference_history
            else 0.0
        )
        total_fe = vfe + homeostatic_fe

        # Meta-inference state (Beautiful Loop)
        meta_confidence = 0.5
        meta_predicted_accuracy = 0.5
        meta_cognitive_load = 0.0
        if self.meta_state is not None:
            meta_confidence = self.meta_state.model_confidence
            meta_predicted_accuracy = self.meta_state.predicted_accuracy
            meta_cognitive_load = self.meta_state.cognitive_load

        return ActiveInferenceState(
            current_beliefs=self.belief_history[-1] if self.belief_history else None,
            belief_entropy=(
                float(
                    -np.sum(
                        self.belief_history[-1]
                        * np.log(self.belief_history[-1] + 1e-10)
                    )
                )
                if self.belief_history
                else 0.0
            ),
            variational_free_energy=vfe,
            expected_free_energy=efe,
            homeostatic_free_energy=homeostatic_fe,
            total_free_energy=total_fe,
            avg_prediction_error=avg_pe,
            prediction_error_trend=trend,
            last_action=(
                self.inference_history[-1].selected_action
                if self.inference_history
                else None
            ),
            action_count=self.total_inferences,
            model_updates=self.total_model_updates,
            most_urgent_drive=most_urgent_drive,
            urgency_level=urgency,
            meta_confidence=meta_confidence,
            meta_predicted_accuracy=meta_predicted_accuracy,
            meta_cognitive_load=meta_cognitive_load,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for the active inference module."""
        state = self.get_state()

        return {
            "total_inferences": self.total_inferences,
            "total_model_updates": self.total_model_updates,
            "avg_prediction_error": state.avg_prediction_error,
            "prediction_error_trend": state.prediction_error_trend,
            "variational_free_energy": state.variational_free_energy,
            "expected_free_energy": state.expected_free_energy,
            "homeostatic_free_energy": state.homeostatic_free_energy,
            "total_free_energy": state.total_free_energy,
            "belief_entropy": state.belief_entropy,
            "most_urgent_drive": state.most_urgent_drive,
            "urgency_level": state.urgency_level,
            "hierarchical_prediction_error": self.hierarchical_processor.get_total_prediction_error(),
        }

    async def generate_active_inference_report(self) -> str:
        """Generate a human-readable report of active inference state."""
        state = self.get_state()

        report_parts = [
            f"My current active inference state: prediction error {'decreasing' if state.prediction_error_trend == 'decreasing' else 'stable' if state.prediction_error_trend == 'stable' else 'increasing'}."
        ]

        if state.most_urgent_drive:
            report_parts.append(
                f"Most urgent need: {state.most_urgent_drive} (urgency: {state.urgency_level:.2f})."
            )

        if state.total_free_energy > 1.0:
            report_parts.append("High free energy - seeking to reduce surprise.")
        else:
            report_parts.append("Low free energy - predictions are accurate.")

        return " ".join(report_parts)


# ============================================================================
# MAIN - Test Active Inference
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test_active_inference():
        """Test the Active Inference module."""
        print("=" * 60)
        print("Active Inference Module Test (Phase 5 FEP)")
        print("=" * 60)

        # Create module
        config = ActiveInferenceConfig(
            num_hidden_states=8,
            num_observations=5,
            num_actions=4,
            planning_horizon=3,
        )

        module = ActiveInferenceModule(config)
        print("Module created")

        # Test inference
        print("\n--- Testing inference ---")
        observation = np.array([0.8, 0.1, 0.05, 0.03, 0.02])

        result = await module.infer_and_act(observation)
        print(f"Posterior beliefs: {result.posterior_beliefs}")
        print(f"Prediction error: {result.prediction_error:.4f}")
        print(f"Selected action: {result.selected_action}")
        print(f"Action confidence: {result.action_confidence:.2f}")
        print(f"VFE: {result.variational_free_energy:.4f}")

        # Test predictions
        print("\n--- Testing predictions ---")
        predictions = await module.predict_next(horizon=3)
        for i, pred in enumerate(predictions):
            print(f"  t+{i+1}: {pred}")

        # Test homeostasis
        print("\n--- Testing homeostasis ---")
        activity = {
            "type": "conversation",
            "intensity": 0.8,
            "understanding_level": 0.7,
        }

        valence = await module.update_homeostasis(activity)
        print(f"Valence signals: {valence}")

        most_urgent, urgency = module.homeostatic_drives.get_most_urgent_need()
        print(f"Most urgent need: {most_urgent} ({urgency:.2f})")

        # Test hierarchical processing
        print("\n--- Testing hierarchical processing ---")
        sensory = np.random.rand(64)

        inference_result, hier_errors = await module.process_with_hierarchy(sensory)
        print(
            f"Hierarchical errors: {[(e.level, e.error_magnitude) for e in hier_errors]}"
        )

        # Get report
        print("\n--- Active Inference Report ---")
        report = await module.generate_active_inference_report()
        print(report)

        # Get statistics
        print("\n--- Statistics ---")
        stats = module.get_statistics()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print("\n" + "=" * 60)
        print("Active Inference Module Test PASSED!")
        print("=" * 60)

    asyncio.run(test_active_inference())
