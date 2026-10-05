"""
Active Inference Module - Free Energy Principle Implementation
==============================================================

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

The system does not just react to observations: it predicts them and acts to
minimize surprise.

Research Foundation:
- Friston, K. (2010). "The Free Energy Principle: A Unified Brain Theory?"
- Friston, K. (2012). "Active Inference and Free Energy"
- Parr, T., & Friston, K. J. (2019). "Generalised free energy and active inference"
- Butlin et al. (2023), 14 indicators - Predictive Processing

Created: December 5, 2025
Author: Multi-Theory Consciousness Contributors
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mtc.consciousness.prediction_ledger import PredictionLedger

# pymdp for Active Inference.
# pymdp v1.0 ships a JAX-based Agent (pymdp.agent) and preserves the original
# NumPy Agent in pymdp.legacy. The legacy API is used because:
#   1. the model is small (8 states, 5 observations by default), so JAX XLA
#      overhead makes it several times slower
#   2. jax-metal (Apple Silicon GPU) does not support default_memory_space yet
#   3. the legacy Agent is mutable, which allows attention precision
#      modulation (agent.A = A_mod)
#   4. the JAX Agent (equinox.Module) is immutable and would need a full
#      reconstruction every cycle
# Importing this module never requires pymdp; constructing an
# ActiveInferenceModule does, and raises a clear ImportError without it.
try:
    from pymdp.legacy import utils as pymdp_utils
    from pymdp.legacy.agent import Agent
except ImportError:
    try:
        from pymdp import utils as pymdp_utils
        from pymdp.agent import Agent
    except ImportError:
        pymdp_utils = None
        Agent = None

PYMDP_MISSING_MESSAGE = (
    "pymdp is required: pip install 'multi-theory-consciousness[fep]'"
)

# Accepted values for the string-valued behaviour switches.
DRIVE_MODELS = ("f4", "v04")
META_ACCURACY_METHODS = ("ledger", "median")
# What process_with_hierarchy hands the discrete agent (see the config field).
HIERARCHY_OBSERVATION_CONTENT = "content"
HIERARCHY_OBSERVATION_HPP_TOP_LEVEL = "hpp_top_level"
HIERARCHY_OBSERVATIONS = (
    HIERARCHY_OBSERVATION_CONTENT,
    HIERARCHY_OBSERVATION_HPP_TOP_LEVEL,
)

# Activity vocabularies for the homeostatic drives (see HomeostaticDrives).
_ATTENTION_DEPLETING_V04 = ("thinking", "processing", "analyzing")
_ATTENTION_DEPLETING_F4 = _ATTENTION_DEPLETING_V04 + (
    "reasoning",
    "conversation",
    "consciousness_cycle",
)
_ATTENTION_RESTING_V04 = ("resting", "idle")
_ATTENTION_RESTING_F4 = _ATTENTION_RESTING_V04 + ("rest",)
_SOCIAL_TYPES_V04 = ("conversation", "interaction", "connecting")
_SOCIAL_TYPES_F4 = _SOCIAL_TYPES_V04 + ("social_interaction",)
_COHERENCE_TYPES_F4 = ("reasoning", "understanding", "sensemaking")

# Long-running processes keep bounded histories: once a list passes the limit
# it is trimmed to the most recent KEEP entries.
HISTORY_LIMIT = 1000
HISTORY_KEEP = 500
META_HISTORY_LIMIT = HISTORY_LIMIT
META_HISTORY_KEEP = HISTORY_KEEP


# ============================================================================
# WORLD MODEL LABELS - names for the POMDP dimensions
# ============================================================================

# compact_8x5x4: the five channels of the workspace observation encoder.
COMPACT_OBSERVATION_LABELS: Tuple[str, ...] = (
    "integration",  # cross-source coherence of the workspace
    "emotional_arousal",  # arousal from the emotional state or salience
    "cognitive_load",  # how full the workspace is
    "ignition_strength",  # how many contents ignited
    "social_content",  # share of social or relational content
)


# conversational_16x10x8: a conversational world model with structured priors.
# The 16 hidden states form a conceptual 4 x 4 grid (stored flat for pymdp).
CONVERSATIONAL_HIDDEN_STATE_LABELS: Tuple[str, ...] = (
    # Emotional state (0-3)
    "calm",
    "positive",
    "negative",
    "intense",
    # Topic depth (4-7)
    "surface",
    "exploring",
    "deep",
    "reflective",
    # Social dynamic (8-11)
    "greeting",
    "sharing",
    "collaborating",
    "conflicting",
    # Cognitive mode (12-15)
    "receptive",
    "questioning",
    "creating",
    "consolidating",
)

# 10 observations read from the conversational environment
CONVERSATIONAL_OBSERVATION_LABELS: Tuple[str, ...] = (
    # Message tone (0-2)
    "neutral_tone",
    "warm_tone",
    "tense_tone",
    # Topic signal (3-4)
    "familiar_topic",
    "novel_topic",
    # Engagement level (5-6)
    "low_engagement",
    "high_engagement",
    # Speaker intent (7-9)
    "inform_intent",
    "question_intent",
    "emote_intent",
)

# 8 conversational moves
CONVERSATIONAL_ACTION_LABELS: Tuple[str, ...] = (
    "respond_empathically",
    "respond_analytically",
    "ask_question",
    "share_experience",
    "redirect_topic",
    "validate_feeling",
    "offer_perspective",
    "stay_silent",
)

# Log-preferences over observation labels for the conversational model.
# Neutral documented defaults: positive values are preferred, negative values
# avoided. Tune them for your experiment through ActiveInferenceConfig.
CONVERSATIONAL_PREFERENCES: Dict[str, float] = {
    "neutral_tone": 0.5,
    "warm_tone": 2.0,
    "tense_tone": -1.5,
    "familiar_topic": 0.5,
    "novel_topic": 1.0,
    "low_engagement": -0.5,
    "high_engagement": 1.5,
    "inform_intent": 0.5,
    "question_intent": 1.0,
    "emote_intent": 0.5,
}

# Likelihood structure P(observation | state) of the conversational model as
# (observation, state, weight) entries added to a 0.05 floor before the
# columns are normalized.
_CONVERSATIONAL_A_ENTRIES: Tuple[Tuple[str, str, float], ...] = (
    # Emotional states -> tone
    ("neutral_tone", "calm", 0.6),
    ("warm_tone", "positive", 0.6),
    ("tense_tone", "negative", 0.5),
    ("neutral_tone", "negative", 0.1),
    ("tense_tone", "intense", 0.4),
    ("warm_tone", "intense", 0.2),
    # Topic depth -> topic signal and engagement
    ("familiar_topic", "surface", 0.5),
    ("low_engagement", "surface", 0.4),
    ("novel_topic", "exploring", 0.5),
    ("high_engagement", "exploring", 0.3),
    ("familiar_topic", "deep", 0.3),
    ("high_engagement", "deep", 0.5),
    ("familiar_topic", "reflective", 0.4),
    ("high_engagement", "reflective", 0.2),
    # Social dynamic -> intent and tone
    ("warm_tone", "greeting", 0.4),
    ("emote_intent", "greeting", 0.3),
    ("inform_intent", "sharing", 0.4),
    ("warm_tone", "sharing", 0.3),
    ("question_intent", "collaborating", 0.4),
    ("high_engagement", "collaborating", 0.4),
    ("tense_tone", "conflicting", 0.5),
    ("emote_intent", "conflicting", 0.3),
    # Cognitive mode -> intent and engagement
    ("inform_intent", "receptive", 0.4),
    ("neutral_tone", "receptive", 0.3),
    ("question_intent", "questioning", 0.5),
    ("high_engagement", "questioning", 0.3),
    ("novel_topic", "creating", 0.4),
    ("high_engagement", "creating", 0.4),
    ("familiar_topic", "consolidating", 0.4),
    ("inform_intent", "consolidating", 0.3),
)

# Transition structure P(next | current, action) as (action, next, current,
# weight) entries added to a 0.6 self-persistence diagonal.
_CONVERSATIONAL_B_ENTRIES: Tuple[Tuple[str, str, str, float], ...] = (
    # Empathy soothes, redirects intensity, de-escalates and opens up
    ("respond_empathically", "calm", "negative", 0.3),
    ("respond_empathically", "positive", "intense", 0.2),
    ("respond_empathically", "sharing", "conflicting", 0.25),
    ("respond_empathically", "sharing", "greeting", 0.15),
    # Analysis deepens, provokes questions and cools emotion
    ("respond_analytically", "exploring", "surface", 0.3),
    ("respond_analytically", "deep", "exploring", 0.25),
    ("respond_analytically", "questioning", "receptive", 0.2),
    ("respond_analytically", "calm", "positive", 0.15),
    # Questions open depth and invite collaboration
    ("ask_question", "questioning", "receptive", 0.3),
    ("ask_question", "exploring", "surface", 0.25),
    ("ask_question", "questioning", "consolidating", 0.2),
    ("ask_question", "collaborating", "sharing", 0.2),
    # Sharing connects and inspires
    ("share_experience", "sharing", "greeting", 0.35),
    ("share_experience", "positive", "calm", 0.2),
    ("share_experience", "creating", "receptive", 0.15),
    ("share_experience", "exploring", "surface", 0.15),
    # Redirecting shifts topic and de-escalates
    ("redirect_topic", "surface", "deep", 0.3),
    ("redirect_topic", "greeting", "conflicting", 0.25),
    ("redirect_topic", "calm", "intense", 0.25),
    ("redirect_topic", "calm", "negative", 0.2),
    # Validation soothes, channels intensity and opens dialogue
    ("validate_feeling", "calm", "negative", 0.35),
    ("validate_feeling", "positive", "intense", 0.25),
    ("validate_feeling", "sharing", "conflicting", 0.2),
    ("validate_feeling", "positive", "calm", 0.15),
    # A perspective deepens understanding and invites co-creation
    ("offer_perspective", "deep", "exploring", 0.3),
    ("offer_perspective", "creating", "questioning", 0.25),
    ("offer_perspective", "collaborating", "sharing", 0.2),
    ("offer_perspective", "consolidating", "reflective", 0.2),
    # Silence enables reflection, settling and integration
    ("stay_silent", "reflective", "deep", 0.3),
    ("stay_silent", "calm", "intense", 0.2),
    ("stay_silent", "consolidating", "creating", 0.2),
    ("stay_silent", "reflective", "exploring", 0.15),
)

# Initial state prior biases: the usual starting point of a conversation.
_CONVERSATIONAL_D_BIASES: Dict[str, float] = {
    "calm": 0.15,
    "surface": 0.10,
    "greeting": 0.10,
    "receptive": 0.10,
}

# World models: (hidden states, observations, actions, default horizon).
WORLD_MODEL_COMPACT = "compact_8x5x4"
WORLD_MODEL_CONVERSATIONAL = "conversational_16x10x8"
WORLD_MODEL_CUSTOM = "custom"
WORLD_MODELS = (WORLD_MODEL_COMPACT, WORLD_MODEL_CONVERSATIONAL, WORLD_MODEL_CUSTOM)
_WORLD_MODEL_DEFAULTS: Dict[str, Tuple[int, int, int, int]] = {
    WORLD_MODEL_COMPACT: (8, 5, 4, 5),
    WORLD_MODEL_CONVERSATIONAL: (16, 10, 8, 3),
    WORLD_MODEL_CUSTOM: (8, 5, 4, 5),
}

# pymdp enumerates every policy (num_actions ** planning_horizon) each cycle.
MAX_POLICIES = 5000

# Persistence: the version stamped into every saved state. A state without
# this stamp (or with another version) restores counters and drives only;
# the learned model and the verdict history start clean.
STATE_SCHEMA_VERSION = 1
INFERENCE_HISTORY_LIMIT = 100
POSTERIOR_MAPPING_LIMIT = 200


def _generic_labels(prefix: str, count: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(count)]


def _fit_labels(labels: Sequence[str], count: int, prefix: str) -> List[str]:
    """Labels for ``count`` channels: the table first, generic names after."""
    fitted = list(labels)[:count]
    fitted.extend(f"{prefix}_{i}" for i in range(len(fitted), count))
    return fitted


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class ActiveInferenceConfig:
    """Configuration for the Active Inference module.

    World models:
        ``"compact_8x5x4"`` (default): 8 hidden states, 5 observation channels
            (the workspace observation encoder), 4 actions, planning horizon 5.
            Its initializers are dimension agnostic, so the dimensions may be
            overridden (the parameter sweep runs 2 states and 2 actions).
        ``"conversational_16x10x8"``: 16 hidden states, 10 observations and 8
            actions with labelled, structured A, B and D priors and planning
            horizon 3. The dimensions are fixed; explicit values that differ
            raise ValueError.
        ``"custom"``: any dimensions, the dimension-agnostic initializers, and
            optional ``observation_labels`` naming the channels.

    Dimensions and the planning horizon left as None take the world model's
    values. ``num_actions ** planning_horizon`` may not exceed MAX_POLICIES.
    """

    # Generative model dimensions (None: the world model's value)
    num_hidden_states: Optional[int] = None  # Hidden states the system can be in
    num_observations: Optional[int] = None  # Observation channels
    num_actions: Optional[int] = None  # Possible actions

    # Planning parameters (None: the world model's horizon)
    planning_horizon: Optional[int] = None
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

    # How the meta-state learns whether a prediction was accurate.
    #   "ledger": each cycle records a forecast of the next observation (the
    #     posterior predictive) in a PredictionLedger and the next cycle
    #     scores it on a fixed absolute tolerance; the meta-state updates
    #     only on a scored outcome.
    #   "median": the v0.4.0 verdict, accurate when the prediction error is
    #     below the running median of the last 20, which is close to 50
    #     percent true by construction whatever the model does.
    meta_accuracy_method: str = "ledger"

    # Hierarchical predictive processor geometry. Hidden dims equal the input
    # dims (multiplier 1): the belief vector lives in the same space as the
    # prediction error that updates it. The v0.4.0 value 2 made every update
    # skip, so the hierarchy never learned.
    hpp_hidden_dim_multiplier: int = 1

    # The observation process_with_hierarchy gives the discrete agent.
    #   "content": the content-derived input itself, through
    #     observation_for_discrete_inference; the hierarchy learns in shadow.
    #   "hpp_top_level": the v0.4.0 route, the encoded top-level hierarchy
    #     beliefs. Those beliefs are signed and not a distribution over
    #     observation channels, so the agent saw a constant or invalid input.
    hierarchy_observation: str = HIERARCHY_OBSERVATION_CONTENT

    # Homeostatic drive dynamics.
    #   drive_model "f4": drives respond to the activity vocabulary the
    #     callers actually send (consciousness_cycle, conversation, reasoning,
    #     rest, social_interaction, understanding, sensemaking).
    #   drive_model "v04": the v0.4.0 vocabulary.
    drive_model: str = "f4"
    # Natural decay stops at this fraction of each drive's optimal level.
    drive_decay_floor: float = 0.1
    # legacy_v04 compatibility, remove in v0.6.0: the curiosity and social
    # branches also subtract decay_rate when idle, so idle cycles decay twice.
    legacy_double_decay: bool = False

    # Narration gates for generate_active_inference_report(). Computation
    # is unaffected; a False gate only drops the line from the report.
    narrate_free_energy: bool = True
    narrate_body_drives: bool = True

    # Optional receiver for per-cycle signal records (a dict with signal_id,
    # values, computed_from, context and timestamp). The module never writes
    # files itself; a sink that raises is logged and ignored.
    signal_sink: Optional[Callable[[Dict[str, Any]], None]] = None

    # World model preset (see the class docstring)
    world_model: str = WORLD_MODEL_COMPACT
    # Channel names for world_model "custom" (one per observation)
    observation_labels: Optional[List[str]] = None
    # Log-preferences over observation labels; tune for your experiment.
    # None uses the world model's defaults: CONVERSATIONAL_PREFERENCES for the
    # conversational model, and for the other models C[0] = 2, C[1] = 1 and
    # C[-1] = -1 by channel position.
    preferences: Optional[Dict[str, float]] = None

    def __post_init__(self):
        self._resolve_world_model()
        if self.hpp_hidden_dim_multiplier < 1:
            raise ValueError(
                "hpp_hidden_dim_multiplier must be at least 1, got "
                f"{self.hpp_hidden_dim_multiplier}"
            )
        if self.drive_model not in DRIVE_MODELS:
            raise ValueError(
                f"drive_model must be one of {DRIVE_MODELS}, got {self.drive_model!r}"
            )
        if self.meta_accuracy_method not in META_ACCURACY_METHODS:
            raise ValueError(
                f"meta_accuracy_method must be one of {META_ACCURACY_METHODS}, "
                f"got {self.meta_accuracy_method!r}"
            )
        if self.hierarchy_observation not in HIERARCHY_OBSERVATIONS:
            raise ValueError(
                f"hierarchy_observation must be one of {HIERARCHY_OBSERVATIONS}, "
                f"got {self.hierarchy_observation!r}"
            )
        if not 0.0 <= self.drive_decay_floor <= 1.0:
            raise ValueError(
                "drive_decay_floor must be within [0, 1], got "
                f"{self.drive_decay_floor}"
            )

    def _resolve_world_model(self) -> None:
        """Fill dimensions from the world model and validate them."""
        if self.world_model not in WORLD_MODELS:
            raise ValueError(
                f"world_model must be one of {WORLD_MODELS}, got {self.world_model!r}"
            )
        states, observations, actions, horizon = _WORLD_MODEL_DEFAULTS[self.world_model]
        if self.world_model == WORLD_MODEL_CONVERSATIONAL:
            explicit = {
                "num_hidden_states": (self.num_hidden_states, states),
                "num_observations": (self.num_observations, observations),
                "num_actions": (self.num_actions, actions),
            }
            mismatched = {
                name: given
                for name, (given, fixed) in explicit.items()
                if given is not None and given != fixed
            }
            if mismatched:
                raise ValueError(
                    f"world_model {WORLD_MODEL_CONVERSATIONAL!r} fixes "
                    f"{states} hidden states, {observations} observations and "
                    f"{actions} actions; got {mismatched}. Use world_model "
                    f"{WORLD_MODEL_CUSTOM!r} for other dimensions."
                )
        if self.num_hidden_states is None:
            self.num_hidden_states = states
        if self.num_observations is None:
            self.num_observations = observations
        if self.num_actions is None:
            self.num_actions = actions
        if self.planning_horizon is None:
            self.planning_horizon = horizon
        self.num_hidden_states = int(self.num_hidden_states)
        self.num_observations = int(self.num_observations)
        self.num_actions = int(self.num_actions)
        self.planning_horizon = int(self.planning_horizon)

        for name in ("num_hidden_states", "num_observations", "num_actions"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.planning_horizon < 1:
            raise ValueError(
                f"planning_horizon must be at least 1, got {self.planning_horizon}"
            )
        policies = self.num_actions**self.planning_horizon
        if policies > MAX_POLICIES:
            raise ValueError(
                f"planning_horizon={self.planning_horizon} with "
                f"num_actions={self.num_actions} gives {policies} policies, more "
                f"than the {MAX_POLICIES} the planner enumerates per cycle; "
                "lower planning_horizon or num_actions"
            )

        if self.observation_labels is not None:
            if self.world_model != WORLD_MODEL_CUSTOM:
                raise ValueError(
                    "observation_labels is only accepted with world_model "
                    f"{WORLD_MODEL_CUSTOM!r}; {self.world_model!r} names its own"
                )
            labels = [str(label) for label in self.observation_labels]
            distinct = len(set(labels)) == len(labels)
            if len(labels) != self.num_observations or not distinct:
                raise ValueError(
                    f"observation_labels needs {self.num_observations} distinct "
                    f"labels, got {labels}"
                )
            self.observation_labels = labels

        if self.preferences is not None:
            known = set(self.observation_label_list())
            unknown = sorted(set(self.preferences) - known)
            if unknown:
                raise ValueError(
                    f"preferences name unknown observation labels {unknown}; "
                    f"world model {self.world_model!r} has {sorted(known)}"
                )

    def hidden_state_label_list(self) -> List[str]:
        """Names of the hidden states of the active world model."""
        if self.world_model == WORLD_MODEL_CONVERSATIONAL:
            return list(CONVERSATIONAL_HIDDEN_STATE_LABELS)
        return _generic_labels("state", self.num_hidden_states)

    def observation_label_list(self) -> List[str]:
        """Names of the observation channels of the active world model."""
        if self.world_model == WORLD_MODEL_CONVERSATIONAL:
            return list(CONVERSATIONAL_OBSERVATION_LABELS)
        if self.world_model == WORLD_MODEL_CUSTOM:
            if self.observation_labels is not None:
                return list(self.observation_labels)
            return _generic_labels("channel", self.num_observations)
        return _fit_labels(COMPACT_OBSERVATION_LABELS, self.num_observations, "channel")

    def action_label_list(self) -> List[str]:
        """Names of the actions of the active world model."""
        if self.world_model == WORLD_MODEL_CONVERSATIONAL:
            return list(CONVERSATIONAL_ACTION_LABELS)
        return _generic_labels("action", self.num_actions)


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
    """Configuration for Hierarchical Predictive Processing.

    Each level's hidden dim is its input dim times ``hidden_dim_multiplier``.
    The belief update consumes the first ``hidden_dim`` entries of the
    prediction error, which has the input dim, so any multiplier above 1
    leaves the error too short and the level never updates.
    """

    num_levels: int = 3
    hidden_dim_multiplier: int = 1
    level_configs: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.level_configs is None:
            m = int(self.hidden_dim_multiplier)
            self.level_configs = [
                {
                    "id": 0,
                    "input_dim": 64,
                    "hidden_dim": 64 * m,
                    "output_dim": 32,
                    "learning_rate": 0.1,
                    "time_constant": 0.1,
                },  # Fast, concrete
                {
                    "id": 1,
                    "input_dim": 32,
                    "hidden_dim": 32 * m,
                    "output_dim": 16,
                    "learning_rate": 0.05,
                    "time_constant": 0.5,
                },  # Medium, contextual
                {
                    "id": 2,
                    "input_dim": 16,
                    "hidden_dim": 16 * m,
                    "output_dim": 8,
                    "learning_rate": 0.01,
                    "time_constant": 1.0,
                },  # Slow, abstract
            ]


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
    makes that commitment explicit - each piece of conscious content is paired
    with the posterior belief state that accompanied it.

    This enables testing the theory's prediction: no change in posterior ->
    no change in conscious content.
    """

    content_summary: str  # GWT winner's content description
    content_type: str  # Type of content (thought, emotion, etc.)
    posterior_beliefs: np.ndarray  # q(s) - the inferred hidden states
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
        if len(self._accuracy_history) > META_HISTORY_LIMIT:
            self._accuracy_history = self._accuracy_history[-META_HISTORY_KEEP:]

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
        if len(self._confidence_history) > META_HISTORY_LIMIT:
            self._confidence_history = self._confidence_history[-META_HISTORY_KEEP:]

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

    Args:
        config: Drive set points and rates.
        drive_model: "f4" (default) maps the activity vocabulary the callers
            actually send; "v04" keeps the v0.4.0 vocabulary.
        decay_floor: Natural decay stops at this fraction of each drive's
            optimal level (0.0 reproduces v0.4.0).
        legacy_double_decay: legacy_v04 compatibility, remove in v0.6.0.
            When True, the curiosity and social branches subtract decay_rate
            on unrelated activity on top of the natural decay.
    """

    def __init__(
        self,
        config: Optional[HomeostaticConfig] = None,
        drive_model: str = "f4",
        decay_floor: float = 0.1,
        legacy_double_decay: bool = False,
    ):
        config = config or HomeostaticConfig()
        if drive_model not in DRIVE_MODELS:
            raise ValueError(
                f"drive_model must be one of {DRIVE_MODELS}, got {drive_model!r}"
            )
        self.drive_model = drive_model
        self.decay_floor = float(decay_floor)
        self.legacy_double_decay = bool(legacy_double_decay)

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

        logger.info("Homeostatic Drives initialized with %d drives", len(self.drives))

    def floor_for(self, drive: HomeostaticDrive) -> float:
        """Lowest level natural decay can reach for this drive."""
        return drive.optimal_level * self.decay_floor

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
            drive.current_level += impact
            drive.current_level = np.clip(drive.current_level, 0, 1)

            # Natural decay toward depletion, stopping at the floor
            drive.current_level -= drive.decay_rate
            drive.current_level = max(self.floor_for(drive), drive.current_level)

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
        if len(self.valence_history) > HISTORY_LIMIT:
            self.valence_history = self.valence_history[-HISTORY_KEEP:]

        return valence_signals

    def _calculate_activity_impact(
        self, activity: Dict[str, Any], drive: HomeostaticDrive
    ) -> float:
        """Calculate how an activity impacts a specific drive."""
        impact = 0.0
        f4 = self.drive_model == "f4"

        # Map activity types to drive impacts
        activity_type = activity.get("type", "unknown")
        intensity = activity.get("intensity", 0.5)

        if drive.name == "attention_budget":
            # Cognitive activities deplete attention. Under "f4" this includes
            # the event types the callers really send every cycle; without
            # them the drive only ever saw the flat natural decay.
            depleting = _ATTENTION_DEPLETING_F4 if f4 else _ATTENTION_DEPLETING_V04
            resting = _ATTENTION_RESTING_F4 if f4 else _ATTENTION_RESTING_V04
            if activity_type in depleting:
                impact = -intensity * 0.1
            elif activity_type in resting:
                impact = drive.recovery_rate

        elif drive.name == "curiosity":
            # Learning satisfies curiosity
            if activity_type in ["learning", "exploring", "discovering"]:
                impact = intensity * drive.recovery_rate
            elif self.legacy_double_decay:
                # legacy_v04 compatibility, remove in v0.6.0
                impact = -drive.decay_rate
            # Otherwise idle depletion comes once, from the natural decay in
            # update_drives(); subtracting it here as well sank the drive to
            # its floor and kept it there.

        elif drive.name == "social_connection":
            # Conversation satisfies social needs
            social = _SOCIAL_TYPES_F4 if f4 else _SOCIAL_TYPES_V04
            if activity_type in social:
                impact = intensity * drive.recovery_rate
            elif self.legacy_double_decay:
                # legacy_v04 compatibility, remove in v0.6.0
                impact = -drive.decay_rate

        elif drive.name == "coherence":
            # Understanding increases coherence. Under "f4" the sense-making
            # activity types count even when no understanding_level is sent.
            understood = activity.get("understanding_level", 0) > 0.5
            if f4 and activity_type in _COHERENCE_TYPES_F4:
                understood = True
            if understood:
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
        if len(self.free_energy_history) > HISTORY_LIMIT:
            self.free_energy_history = self.free_energy_history[-HISTORY_KEEP:]

        return total_fe

    def get_most_urgent_need(self) -> Tuple[str, float]:
        """
        Return the drive that most needs attention.

        This guides behavior - the system will tend toward
        actions that satisfy its most urgent needs.
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

    def get_urgent_drives(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Drives whose urgency exceeds ``threshold``, most urgent first.

        Returns:
            List of (drive_name, urgency) tuples sorted by urgency descending.
        """
        urgent = [
            (name, float(drive.get_urgency()))
            for name, drive in self.drives.items()
            if drive.get_urgency() > threshold
        ]
        urgent.sort(key=lambda item: item[1], reverse=True)
        return urgent


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

    # Belief updates skipped because the error was shorter than hidden_dim
    skipped_updates: int = 0

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
        """Update beliefs based on prediction error.

        An error shorter than the hidden dim cannot update the beliefs. That
        case used to substitute a silent zeros update, which froze the level;
        it now skips the update and says so (rate-limited).
        """
        if len(error) < self.hidden_dim:
            self.skipped_updates += 1
            if self.skipped_updates == 1 or self.skipped_updates % 1000 == 0:
                logger.warning(
                    "HPP level %d: error length %d < hidden_dim %d, belief "
                    "update skipped (%d skipped so far)",
                    self.level_id,
                    len(error),
                    self.hidden_dim,
                    self.skipped_updates,
                )
            return
        # Simple gradient update
        update = self.learning_rate * error[: self.hidden_dim]
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
            "Hierarchical Predictive Processor initialized with %d levels",
            len(self.levels),
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

    # Workspace content type -> observation labels it sharpens, in order of
    # preference. The channel index is looked up in the active world model's
    # observation labels, so each model maps the families (tone, topic,
    # novelty, engagement, intent) onto its own channels and a model without
    # any listed label ignores that content type.
    CONTENT_TYPE_TO_CHANNEL: Dict[str, Tuple[str, ...]] = {
        # tone
        "emotion": ("warm_tone", "emotional_arousal"),
        "homeostatic_drive": ("warm_tone", "emotional_arousal"),
        # intent
        "message": ("inform_intent", "social_content"),
        "thought": ("inform_intent", "cognitive_load"),
        "metacognitive": ("question_intent", "cognitive_load"),
        "self_model": ("question_intent", "cognitive_load"),
        # topic
        "memory": ("familiar_topic", "integration"),
        "gist": ("familiar_topic", "integration"),
        # novelty
        "concept": ("novel_topic", "integration"),
        # engagement
        "sensory": ("high_engagement", "ignition_strength"),
        "visual": ("high_engagement", "ignition_strength"),
        "audio": ("high_engagement", "ignition_strength"),
        # social content (compact model only)
        "social": ("social_content",),
        "person": ("social_content",),
        "relationship": ("social_content",),
        "conversation": ("social_content",),
    }

    # Workspace precision bias dynamics (modulate_precision_from_workspace)
    WORKSPACE_BIAS_DECAY = 0.85
    WORKSPACE_BIAS_GAIN = 0.3
    WORKSPACE_BIAS_CAP = 1.0

    # Minimum real prediction errors before percentile ranks are answerable.
    BASELINE_MIN_SAMPLES = 100

    # Absolute L1 tolerance for next-observation forecasts. Both vectors are
    # distributions (L1 range 0 to 2), so 0.5 means the forecast placed the
    # observation mix to within a quarter of the total mass.
    FORECAST_TOLERANCE = 0.5

    def __init__(self, config: Optional[ActiveInferenceConfig] = None):
        if Agent is None or pymdp_utils is None:
            raise ImportError(PYMDP_MISSING_MESSAGE)
        self.config = config or ActiveInferenceConfig()

        # Define the generative model dimensions
        self.num_states = self.config.num_hidden_states
        self.num_observations = self.config.num_observations
        self.num_actions = self.config.num_actions

        self.world_model = self.config.world_model
        self._structured = self.world_model == WORLD_MODEL_CONVERSATIONAL

        # Names for the dimensions of the active world model
        self.hidden_state_labels: List[str] = self.config.hidden_state_label_list()
        self.observation_labels: List[str] = self.config.observation_label_list()
        self.action_labels: List[str] = self.config.action_label_list()
        self.hidden_state_index: Dict[str, int] = {
            label: i for i, label in enumerate(self.hidden_state_labels)
        }
        self.observation_index: Dict[str, int] = {
            label: i for i, label in enumerate(self.observation_labels)
        }
        self.action_index: Dict[str, int] = {
            label: i for i, label in enumerate(self.action_labels)
        }

        # Initialize matrices
        self.A = self._initialize_observation_model()  # P(o|s)
        self.B = self._initialize_transition_model()  # P(s'|s,a)
        self.C = self._initialize_preferences()  # P(o) - what we want
        self.D = self._initialize_initial_state()  # P(s_0)

        # Create pymdp agent
        self.agent = self._build_agent()

        # Sub-components
        self.homeostatic_drives = (
            HomeostaticDrives(
                drive_model=self.config.drive_model,
                decay_floor=self.config.drive_decay_floor,
                legacy_double_decay=self.config.legacy_double_decay,
            )
            if self.config.enable_homeostasis
            else None
        )
        self.hierarchical_processor = HierarchicalPredictiveProcessor(
            HPPConfig(hidden_dim_multiplier=self.config.hpp_hidden_dim_multiplier)
        )

        # Meta-inference: the Beautiful Loop (Laukkonen/Friston/Chandaria 2025)
        # When enabled, the system predicts its own prediction accuracy.
        self.meta_state: Optional[MetaState] = (
            MetaState() if self.config.enable_meta_inference else None
        )

        # Measured self-prediction: each cycle records a concrete forecast of
        # the next observation and the next cycle scores it.
        self.prediction_ledger = PredictionLedger()
        self._pending_prediction_id: Optional[str] = None

        # Tracking
        self.prediction_errors: List[float] = []
        self.belief_history: List[np.ndarray] = []
        self.free_energy_history: List[float] = []
        self.inference_history: List[InferenceResult] = []

        # Conscious-content to posterior mappings (Whyte et al. 2026)
        self._posterior_mappings: List[ConsciousPosteriorMapping] = []

        # Workspace to FEP precision feedback (persistent workspace bias)
        self._workspace_precision_bias = np.zeros(self.num_observations)

        # Previous observation, for the novelty field of signal records
        self._last_signal_observation: Optional[np.ndarray] = None

        # Counters
        self.total_inferences = 0
        self.total_model_updates = 0

        logger.info(
            "Active Inference Module initialized (FEP): %d hidden states, "
            "%d observations, %d actions, planning horizon %d, homeostasis %s, "
            "meta-inference %s",
            self.num_states,
            self.num_observations,
            self.num_actions,
            self.config.planning_horizon,
            "enabled" if self.homeostatic_drives else "disabled",
            "enabled" if self.meta_state else "disabled",
        )

    def _build_agent(self):
        """Create the pymdp agent from the current A, B, C and D."""
        return Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            policy_len=self.config.planning_horizon,
            inference_algo=self.config.inference_algo,
            action_selection=self.config.action_selection,
            gamma=self.config.gamma,
        )

    def _initialize_observation_model(self) -> List[np.ndarray]:
        """
        Initialize A matrix - P(observation | hidden_state).

        This encodes how hidden states generate observations.
        Initially fairly uniform, learned over time.

        pymdp format: List of [num_obs x num_states] matrices, one per modality
        """
        if self._structured:
            return self._structured_observation_model()

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
        if self._structured:
            return self._structured_transition_model()

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
        Initialize C vector - log-preferences over observations.

        This encodes what the system "wants" to observe.
        Positive values = preferred, negative = avoided. An explicit
        ``preferences`` dict on the config sets values by observation label;
        otherwise the world model's defaults apply.
        """
        C = np.zeros(self.num_observations)

        preferences = self.config.preferences
        if preferences is None and self._structured:
            preferences = CONVERSATIONAL_PREFERENCES
        if preferences is not None:
            for label, value in preferences.items():
                C[self.observation_index[label]] = float(value)
            return C

        # Prefer some observations over others by channel position
        # (These represent preferred states - safety, connection, etc.)
        C[0] = 2.0  # Strongly prefer first observation type
        if self.num_observations > 1:
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
        if self._structured:
            return self._structured_initial_state()

        D = pymdp_utils.obj_array_uniform([self.num_states])

        # Slight bias toward "neutral" state
        D[0][0] = 0.3
        D[0] = D[0] / D[0].sum()

        return D

    def _require_structured_dims(self) -> None:
        """The structured priors are defined for the conversational model only."""
        expected = (
            len(CONVERSATIONAL_HIDDEN_STATE_LABELS),
            len(CONVERSATIONAL_OBSERVATION_LABELS),
            len(CONVERSATIONAL_ACTION_LABELS),
        )
        actual = (self.num_states, self.num_observations, self.num_actions)
        if actual != expected:
            raise ValueError(
                f"structured priors need dimensions {expected}, got {actual}"
            )

    def _structured_observation_model(self) -> List[np.ndarray]:
        """A for the conversational model: labelled likelihood structure."""
        self._require_structured_dims()
        A = pymdp_utils.obj_array(1)
        # Small uniform floor so no observation is impossible in any state
        A[0] = np.full((self.num_observations, self.num_states), 0.05)
        for obs_label, state_label, weight in _CONVERSATIONAL_A_ENTRIES:
            obs_idx = self.observation_index[obs_label]
            state_idx = self.hidden_state_index[state_label]
            A[0][obs_idx, state_idx] += weight
        A[0] = A[0] / A[0].sum(axis=0, keepdims=True)
        return A

    def _structured_transition_model(self) -> List[np.ndarray]:
        """B for the conversational model: how each move shifts the state."""
        self._require_structured_dims()
        B = pymdp_utils.obj_array(1)
        # States are sticky: strong self-persistence under every action
        B[0] = np.repeat(
            (np.eye(self.num_states) * 0.6)[:, :, np.newaxis], self.num_actions, axis=2
        )
        for (
            action_label,
            next_label,
            current_label,
            weight,
        ) in _CONVERSATIONAL_B_ENTRIES:
            a = self.action_index[action_label]
            B[0][
                self.hidden_state_index[next_label],
                self.hidden_state_index[current_label],
                a,
            ] += weight
        for a in range(self.num_actions):
            B[0][:, :, a] = B[0][:, :, a] / B[0][:, :, a].sum(axis=0, keepdims=True)
        return B

    def _structured_initial_state(self) -> List[np.ndarray]:
        """D for the conversational model: a typical conversation opening."""
        self._require_structured_dims()
        D = pymdp_utils.obj_array_uniform([self.num_states])
        for label, bias in _CONVERSATIONAL_D_BIASES.items():
            D[0][self.hidden_state_index[label]] += bias
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

        # Combine the momentary attention precision (Attention Schema) with
        # the persistent workspace bias (sustained conscious focus).
        combined_precision = self._workspace_precision_bias.copy()
        attention_applied = False
        if attention_precision is not None:
            prec = np.asarray(attention_precision, dtype=float).ravel()
            if len(prec) == self.num_observations:
                combined_precision = combined_precision + prec
                attention_applied = True
            else:
                logger.debug(
                    "attention_precision has %d entries, expected %d; ignored",
                    len(prec),
                    self.num_observations,
                )
        modulated = attention_applied or bool(
            np.any(self._workspace_precision_bias > 0)
        )

        # Apply precision modulation to the A matrix.
        # Attention = precision in the FEP framework: attended observations
        # have sharper (more informative) likelihood mappings.
        if modulated:
            self.agent.A = self._precision_modulated_A(combined_precision)

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

        # Beautiful Loop: score the previous cycle's forecast against this
        # observation, then update the self-referential meta-state with the
        # verdict the configured method gives.
        ledger_verdict = self._score_pending_forecast(observation)
        if self.meta_state is not None:
            if self.config.meta_accuracy_method == "median":
                was_accurate: Optional[bool] = self._median_verdict(prediction_error)
            else:
                was_accurate = ledger_verdict
            if was_accurate is not None:
                self.meta_state.update(
                    prediction_error=float(prediction_error),
                    was_accurate=was_accurate,
                )

        # Forecast the next observation before it exists: the posterior
        # predictive under the current beliefs.
        self._record_forecast(predicted_obs)

        # Optional per-cycle signal record for an external receiver
        self._emit_signal(prediction_error, observation)

        # Step 3: Infer policies (planning as inference)
        q_pi, efe = self.agent.infer_policies()

        # Step 4: Calculate variational free energy
        vfe = self._calculate_variational_free_energy(qs_beliefs, observation)
        self.free_energy_history.append(vfe)
        self._trim_histories()

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
        if len(self.belief_history) > HISTORY_LIMIT:
            self.belief_history = self.belief_history[-HISTORY_KEEP:]

        # Restore base A matrix if precision modulation was applied,
        # so learning and future cycles start from the unmodified model.
        if modulated:
            self.agent.A = self.A

        self.total_inferences += 1

        # Precision-weight the prediction error by attention plus the
        # workspace bias
        precision_weight = (
            float(np.mean(combined_precision)) + 1.0 if modulated else 1.0
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
        if len(self.inference_history) > INFERENCE_HISTORY_LIMIT:
            self.inference_history = self.inference_history[
                -(INFERENCE_HISTORY_LIMIT // 2) :
            ]

        # Online learning: update the generative model after each inference.
        # Active inference agents continuously learn from experience.
        if self.total_inferences >= 2:
            await self.update_generative_model(
                {
                    "observation": observation,
                    "action": action,
                }
            )

        return result

    def _precision_modulated_A(self, precision: np.ndarray) -> np.ndarray:
        """A with each observation row raised to the power 1 + precision.

        Columns are renormalized afterwards, so a higher precision sharpens
        that channel's likelihood. Returns a pymdp object array.
        """
        A_mod = pymdp_utils.obj_array(1)
        A_mod[0] = self.A[0].copy()
        for obs_idx in range(self.num_observations):
            exponent = 1.0 + precision[obs_idx]
            A_mod[0][obs_idx, :] = self.A[0][obs_idx, :] ** exponent
        # Re-normalize columns (each column sums to 1)
        col_sums = A_mod[0].sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        A_mod[0] = A_mod[0] / col_sums
        return A_mod

    def _emit_signal(self, prediction_error: float, observation: np.ndarray) -> None:
        """Send this cycle's prediction error record to the configured sink.

        The record carries the observation novelty (L1 distance to the
        previous observation) as a content-coupling feature. Never raises.
        """
        sink = self.config.signal_sink
        if sink is None:
            return
        obs = np.asarray(observation, dtype=float)
        context: Dict[str, Any] = {"inference_index": self.total_inferences}
        previous = self._last_signal_observation
        if previous is not None and previous.shape == obs.shape:
            context["observation_novelty"] = float(np.sum(np.abs(obs - previous)))
        self._last_signal_observation = obs.copy()
        record = {
            "signal_id": "prediction_error_v1",
            "values": {"prediction_error": float(prediction_error)},
            "confidence": None,
            "computed_from": [
                "discrete_agent_posterior_predictive",
                "workspace_observation_vector",
            ],
            "context": context,
            "timestamp": time.time(),
        }
        try:
            sink(record)
        except Exception as exc:  # noqa: BLE001 - a sink must never break a cycle
            logger.debug("signal sink raised and was ignored: %s", exc)

    def prediction_error_rank(self, value: float) -> Optional[float]:
        """Percentile rank of ``value`` in the module's own recent errors.

        Returns the fraction of retained prediction errors strictly below
        ``value``, or None until BASELINE_MIN_SAMPLES errors exist, so a rank
        is never computed against a distribution too thin to mean anything.
        """
        if len(self.prediction_errors) < self.BASELINE_MIN_SAMPLES:
            return None
        history = np.asarray(self.prediction_errors, dtype=float)
        return float(np.mean(history < float(value)))

    def channel_for_content_type(self, content_type: Optional[str]) -> Optional[int]:
        """Observation channel a workspace content type sharpens, if any."""
        if not content_type:
            return None
        for label in self.CONTENT_TYPE_TO_CHANNEL.get(str(content_type).lower(), ()):
            if label in self.observation_index:
                return self.observation_index[label]
        return None

    def modulate_precision_from_workspace(self, winners: list) -> None:
        """Workspace to FEP precision feedback.

        Conscious focus sharpens future predictions: the persistent bias
        decays by WORKSPACE_BIAS_DECAY each call, then every winner adds its
        activation times WORKSPACE_BIAS_GAIN to its content type's channel,
        capped at WORKSPACE_BIAS_CAP.

        Args:
            winners: Workspace winners. Each may be a workspace content object
                with a ``candidate`` (that has ``content_type`` and
                ``activation_level``), a candidate itself, or a dict with
                those keys.
        """
        self._workspace_precision_bias *= self.WORKSPACE_BIAS_DECAY

        for winner in winners or []:
            if isinstance(winner, dict):
                content_type = winner.get("content_type")
                activation = winner.get("activation_level", 0.5)
            else:
                candidate = getattr(winner, "candidate", winner)
                if candidate is None:
                    continue
                content_type = getattr(candidate, "content_type", None)
                activation = getattr(candidate, "activation_level", 0.5)

            channel = self.channel_for_content_type(content_type)
            if channel is None:
                continue
            boost = max(0.0, float(activation)) * self.WORKSPACE_BIAS_GAIN
            self._workspace_precision_bias[channel] = min(
                self._workspace_precision_bias[channel] + boost,
                self.WORKSPACE_BIAS_CAP,
            )

    @property
    def workspace_precision_bias(self) -> np.ndarray:
        """Copy of the persistent per-channel workspace precision bias."""
        return self._workspace_precision_bias.copy()

    def _score_pending_forecast(self, observation: np.ndarray) -> Optional[bool]:
        """Score the forecast recorded last cycle against this observation.

        Returns None when there is no forecast to score (first cycle, expired
        or shape-changed forecast): no verdict is honestly no verdict.
        """
        if self._pending_prediction_id is None:
            return None
        verdict = self.prediction_ledger.score(
            self._pending_prediction_id, np.asarray(observation, dtype=float)
        )
        self._pending_prediction_id = None
        return verdict

    def _record_forecast(self, forecast: np.ndarray) -> None:
        """Record this cycle's forecast of the next observation."""
        self._pending_prediction_id = self.prediction_ledger.predict(
            forecast,
            tolerance=self.FORECAST_TOLERANCE,
            subject="next_observation",
            context={"inference_index": self.total_inferences},
        )

    def _median_verdict(self, prediction_error: float) -> bool:
        """legacy_v04 compatibility, remove in v0.6.0.

        The v0.4.0 verdict: accurate when the prediction error is below the
        running median of the last 20 errors (0.3 before three exist). It is
        close to 50 percent true by construction, whatever the model does.
        """
        if len(self.prediction_errors) >= 3:
            recent_pe = list(self.prediction_errors)[-20:]
            adaptive_threshold = float(np.median(recent_pe))
        else:
            adaptive_threshold = 0.3  # Bootstrap threshold
        return prediction_error < adaptive_threshold

    def measured_accuracy(self) -> Optional[float]:
        """Hit rate of the next-observation forecasts, None until 20 scored."""
        return self.prediction_ledger.measured_accuracy()

    def observation_for_discrete_inference(
        self, workspace_vector: np.ndarray
    ) -> np.ndarray:
        """The observation the discrete agent infers on.

        The content-derived workspace vector (measured every cycle) feeds the
        discrete agent directly, encoded to ``num_observations`` channels.
        The hierarchical predictive processor keeps learning in shadow and
        never supplies this observation: its signed belief vector is not a
        distribution over observation channels, and routing it here gave the
        agent a constant or invalid input.
        """
        return self._encode_observation_vector(
            np.asarray(workspace_vector, dtype=float).ravel()
        )

    def _encode_observation_vector(self, observation: np.ndarray) -> np.ndarray:
        """Encode arbitrary observation into correct size.

        Negative and non-finite entries carry no probability mass and count
        as zero on every branch, so a corrupt entry can never reach the
        agent. A vector that already has ``num_observations`` entries and
        holds only non-negative finite values passes through unchanged. Any
        other input is folded onto the channels and normalized to a
        probability distribution (uniform when no mass remains).
        """
        if len(observation) == self.num_observations:
            values = np.asarray(observation, dtype=float)
            if np.all(np.isfinite(values)) and np.all(values >= 0.0):
                return observation

        encoded = np.zeros(self.num_observations)
        for i, val in enumerate(observation):
            idx = i % self.num_observations
            val = float(val)
            if np.isfinite(val) and val > 0.0:
                encoded[idx] += val

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
        # The outer product of obs_dist and beliefs gives the target: for each
        # state s, the expected observation distribution. The A matrix is
        # pulled toward this target, so ALL observation channels update
        # proportionally - preventing the argmax-only overshoot bug.
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

        The hierarchy learns from the input and reports its per-level
        prediction errors. Under ``hierarchy_observation == "content"`` (the
        default) the discrete agent infers on the input itself (see
        ``observation_for_discrete_inference``) and the hierarchy runs in
        shadow. Under ``"hpp_top_level"`` (the v0.4.0 route) it infers on the
        encoded top-level hierarchy beliefs.
        """
        # Process through hierarchy (per-level errors)
        hierarchical_errors = await self.hierarchical_processor.process_bottom_up(
            sensory_input
        )

        if self.config.hierarchy_observation == HIERARCHY_OBSERVATION_HPP_TOP_LEVEL:
            top_level_rep = (
                self.hierarchical_processor.levels[-1].beliefs
                if self.hierarchical_processor.levels
                else sensory_input
            )
            observation = self._encode_observation_vector(top_level_rep)
        else:
            observation = self.observation_for_discrete_inference(sensory_input)

        inference_result = await self.infer_and_act(observation)

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
        if len(self._posterior_mappings) > POSTERIOR_MAPPING_LIMIT:
            self._posterior_mappings = self._posterior_mappings[
                -POSTERIOR_MAPPING_LIMIT:
            ]

        return mappings

    def get_posterior_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about conscious-content to posterior mappings."""
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

    def _trim_histories(self) -> None:
        """Bound the prediction error and free energy histories.

        Same pattern as the belief history: past HISTORY_LIMIT entries the
        list keeps the most recent HISTORY_KEEP. The readers that look at the
        last 100 entries at most (the state summary's average and trend, the
        latest free energy) never see a difference. ``prediction_error_rank``
        ranks against the whole retained history, so its baseline is the
        trimmed window once a history has been trimmed.
        """
        if len(self.prediction_errors) > HISTORY_LIMIT:
            self.prediction_errors = self.prediction_errors[-HISTORY_KEEP:]
        if len(self.free_energy_history) > HISTORY_LIMIT:
            self.free_energy_history = self.free_energy_history[-HISTORY_KEEP:]

    def get_state(self) -> ActiveInferenceState:
        """Get current state of the active inference system."""
        # Calculate average prediction error
        avg_pe = (
            np.mean(self.prediction_errors[-100:]) if self.prediction_errors else 0.0
        )

        # Determine trend. It needs at least one error before the latest ten;
        # with exactly ten the older window is empty and the trend is stable.
        if len(self.prediction_errors) > 10:
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
        hpp_error = self.hierarchical_processor.get_total_prediction_error()

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
            "hierarchical_prediction_error": hpp_error,
            "measured_accuracy": self.prediction_ledger.measured_accuracy(),
            "scored_forecasts": self.prediction_ledger.scored_count,
        }

    async def generate_active_inference_report(self) -> str:
        """Generate a human-readable report of active inference state.

        Each line is gated: ``narrate_free_energy`` covers the prediction
        error trend and the free energy line, ``narrate_body_drives`` the most
        urgent need. A closed gate drops the line; the computation behind it
        (get_state, drives, free energy) runs either way.
        """
        state = self.get_state()

        report_parts = []
        if self.config.narrate_free_energy:
            trend = state.prediction_error_trend
            if trend not in ("decreasing", "stable"):
                trend = "increasing"
            report_parts.append(
                f"My current active inference state: prediction error {trend}."
            )

        if state.most_urgent_drive and self.config.narrate_body_drives:
            report_parts.append(
                f"Most urgent need: {state.most_urgent_drive} "
                f"(urgency: {state.urgency_level:.2f})."
            )

        if self.config.narrate_free_energy:
            if state.total_free_energy > 1.0:
                report_parts.append("High free energy - seeking to reduce surprise.")
            else:
                report_parts.append("Low free energy - predictions are accurate.")

        return " ".join(report_parts)

    # ========================================================================
    # STATE PERSISTENCE
    # ========================================================================

    def to_state_dict(self) -> Dict[str, Any]:
        """Export persistent state for save and restore across restarts.

        Returns ``{"scalars": ..., "arrays": ..., "histories": ...}``:
            scalars: JSON-safe counters, drive levels, meta-state values,
                hierarchy precisions, the schema version and the prediction
                ledger's counters (keys prefixed ``ledger_``)
            arrays: numpy arrays (A, B, C, D, workspace bias, belief history,
                hierarchy weights, beliefs and predictions)
            histories: JSON-safe lists (prediction errors, free energy, the
                ledger's pending and scored entries, inference results,
                posterior mappings, drive and meta-state histories)

        The posterior mappings carry each winner's ``content_summary``, so a
        saved state file contains fragments of workspace content.
        """
        scalars: Dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "world_model": self.world_model,
            "num_hidden_states": self.num_states,
            "num_observations": self.num_observations,
            "num_actions": self.num_actions,
            "total_inferences": int(self.total_inferences),
            "total_model_updates": int(self.total_model_updates),
            "pending_prediction_id": self._pending_prediction_id,
            "hpp_precisions": {
                str(i): float(level.precision)
                for i, level in enumerate(self.hierarchical_processor.levels)
            },
        }
        if self.homeostatic_drives is not None:
            scalars["homeostatic_drive_levels"] = {
                name: float(drive.current_level)
                for name, drive in self.homeostatic_drives.drives.items()
            }
        if self.meta_state is not None:
            ms = self.meta_state
            scalars["meta_state"] = {
                "model_confidence": float(ms.model_confidence),
                "prediction_accuracy": float(ms.prediction_accuracy),
                "predicted_accuracy": float(ms.predicted_accuracy),
                "cognitive_load": float(ms.cognitive_load),
                "attention_pattern": str(ms.attention_pattern),
                "emotional_valence": float(ms.emotional_valence),
                "total_predictions": int(ms.total_predictions),
                "correct_predictions": int(ms.correct_predictions),
            }

        arrays: Dict[str, np.ndarray] = {
            "A_0": np.array(self.A[0], dtype=float, copy=True),
            "B_0": np.array(self.B[0], dtype=float, copy=True),
            "C_0": np.array(self.C, dtype=float, copy=True),
            "D_0": np.array(self.D[0], dtype=float, copy=True),
            "workspace_precision_bias": self._workspace_precision_bias.copy(),
        }
        histories: Dict[str, Any] = {}
        if self.belief_history:
            try:
                arrays["belief_history"] = np.stack(
                    [np.asarray(b, dtype=float) for b in self.belief_history]
                )
            except ValueError:
                # Ragged rows cannot stack; keep them as JSON lists instead
                histories["belief_history_lists"] = [
                    [float(x) for x in np.asarray(b).ravel()]
                    for b in self.belief_history
                ]
        for i, level in enumerate(self.hierarchical_processor.levels):
            for attr in _HPP_LEVEL_ARRAYS:
                value = getattr(level, attr)
                if value is not None:
                    arrays[f"hpp_level_{i}_{attr}"] = np.array(value, copy=True)

        # The ledger exports in the same contract; its pieces are flattened
        # into this module's tiers (scalars with a "ledger_" prefix, its two
        # lists as histories) so histories map names to lists only
        ledger_state = self.prediction_ledger.to_state_dict()
        for key, value in (ledger_state.get("scalars") or {}).items():
            scalars[f"ledger_{key}"] = value
        ledger_histories = ledger_state.get("histories") or {}

        histories.update(
            {
                "prediction_errors": [float(x) for x in self.prediction_errors],
                "free_energy_history": [float(x) for x in self.free_energy_history],
                "ledger_pending": list(ledger_histories.get("pending", [])),
                "ledger_scored": list(ledger_histories.get("scored", [])),
                "inference_history": [
                    _inference_result_to_record(r) for r in self.inference_history
                ],
                "posterior_mappings": [
                    _posterior_mapping_to_record(m) for m in self._posterior_mappings
                ],
            }
        )
        if self.homeostatic_drives is not None:
            histories["homeostatic_valence_history"] = [
                {str(k): float(v) for k, v in entry.items()}
                for entry in self.homeostatic_drives.valence_history
            ]
            histories["homeostatic_free_energy_history"] = [
                float(x) for x in self.homeostatic_drives.free_energy_history
            ]
        if self.meta_state is not None:
            histories["meta_accuracy_history"] = [
                float(x) for x in self.meta_state._accuracy_history
            ]
            histories["meta_confidence_history"] = [
                float(x) for x in self.meta_state._confidence_history
            ]
            histories["meta_recent_outcomes"] = [
                bool(x) for x in self.meta_state._recent_outcomes
            ]

        return {"scalars": scalars, "arrays": arrays, "histories": histories}

    @staticmethod
    def _ledger_state_from(h: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ledger state from either saved shape, or None when absent.

        The flat shape keeps two lists under ``ledger_pending`` and
        ``ledger_scored``. State written in Phase 1 nested the ledger's whole
        state dict under ``prediction_ledger``; it is still read.
        """
        if "ledger_pending" in h or "ledger_scored" in h:
            return {
                "histories": {
                    "pending": h.get("ledger_pending") or [],
                    "scored": h.get("ledger_scored") or [],
                }
            }
        nested = h.get("prediction_ledger")
        if isinstance(nested, dict):
            return nested
        return None

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state produced by ``to_state_dict()``.

        Learned state (A, B, C, D, belief and error histories, the ledger,
        inference results, posterior mappings, the hierarchy's weights and
        the meta-state's verdict history) restores only from a state stamped
        with STATE_SCHEMA_VERSION for the same world model; otherwise the
        freshly initialized model is kept and the meta-state starts clean.
        Counters, drive levels (clamped to the decay floor), the hierarchy's
        precisions and the workspace bias restore in every case. Every array
        is shape checked, every history is capped at its in-memory limit, and
        the pymdp agent is rebuilt from the resulting matrices.
        """
        state = state or {}
        s = state.get("scalars", {}) or {}
        arrays = state.get("arrays", {}) or {}
        h = state.get("histories", {}) or {}

        trusted = s.get("schema_version") == STATE_SCHEMA_VERSION
        same_model = s.get("world_model") == self.world_model
        learned_ok = trusted and same_model
        if not trusted:
            logger.warning(
                "FEP restore: state has schema_version %r, expected %d; the "
                "generative model and the meta-state verdict history start "
                "clean, counters and drives restore",
                s.get("schema_version"),
                STATE_SCHEMA_VERSION,
            )
        elif not same_model:
            logger.warning(
                "FEP restore: state was saved by world model %r, this module "
                "runs %r; learned state not restored",
                s.get("world_model"),
                self.world_model,
            )

        # --- scalars ---
        self.total_inferences = int(s.get("total_inferences", self.total_inferences))
        # Counts updates of the current model, so it resets with the model
        self.total_model_updates = (
            int(s.get("total_model_updates", self.total_model_updates))
            if learned_ok
            else 0
        )

        if self.homeostatic_drives is not None:
            drives = self.homeostatic_drives
            for name, level in (s.get("homeostatic_drive_levels") or {}).items():
                drive = drives.drives.get(name)
                if drive is None or not np.isfinite(float(level)):
                    continue
                drive.current_level = min(1.0, max(drives.floor_for(drive), level))

        if self.meta_state is not None:
            self._restore_meta_state(s.get("meta_state") or {}, h, learned_ok)

        for i_str, precision in (s.get("hpp_precisions") or {}).items():
            i = int(i_str)
            if 0 <= i < len(self.hierarchical_processor.levels):
                self.hierarchical_processor.levels[i].precision = float(precision)

        # --- arrays ---
        if learned_ok:
            for key, target in (("A_0", self.A), ("B_0", self.B), ("D_0", self.D)):
                saved = self._checked_array(arrays, key, target[0].shape)
                if saved is not None and self._columns_normalized(saved):
                    target[0] = saved
                elif saved is not None:
                    logger.warning("FEP restore: %s columns not normalized", key)
            saved_c = self._checked_array(arrays, "C_0", self.C.shape)
            if saved_c is not None:
                self.C = saved_c
        # The agent always restarts from the current matrices
        self.agent = self._build_agent()

        saved_bias = self._checked_array(
            arrays, "workspace_precision_bias", self._workspace_precision_bias.shape
        )
        if saved_bias is not None:
            self._workspace_precision_bias = np.clip(
                saved_bias, 0.0, self.WORKSPACE_BIAS_CAP
            )

        if learned_ok:
            self._restore_belief_history(arrays, h)
            for i, level in enumerate(self.hierarchical_processor.levels):
                for attr in _HPP_LEVEL_ARRAYS:
                    current = getattr(level, attr)
                    expected = current.shape if current is not None else None
                    saved = self._checked_array(
                        arrays, f"hpp_level_{i}_{attr}", expected
                    )
                    if saved is not None:
                        setattr(level, attr, saved)

        # --- histories ---
        if learned_ok:
            self.prediction_errors = _float_list(
                h.get("prediction_errors"), HISTORY_LIMIT
            )
            self.free_energy_history = _float_list(
                h.get("free_energy_history"), HISTORY_LIMIT
            )
            ledger_state = self._ledger_state_from(h)
            if ledger_state is not None:
                try:
                    self.prediction_ledger.from_state_dict(ledger_state)
                except Exception as exc:  # noqa: BLE001 - keep restoring the rest
                    logger.warning("FEP restore: prediction ledger skipped: %s", exc)
            pending = s.get("pending_prediction_id")
            self._pending_prediction_id = (
                pending
                if pending and self.prediction_ledger.is_pending(pending)
                else None
            )
            self.inference_history = self._restore_records(
                h.get("inference_history"),
                _inference_result_from_record,
                INFERENCE_HISTORY_LIMIT,
            )
            self._posterior_mappings = self._restore_records(
                h.get("posterior_mappings"),
                _posterior_mapping_from_record,
                POSTERIOR_MAPPING_LIMIT,
            )
        else:
            self._pending_prediction_id = None

        if self.homeostatic_drives is not None:
            drives = self.homeostatic_drives
            if "homeostatic_valence_history" in h:
                drives.valence_history = [
                    {str(k): float(v) for k, v in entry.items()}
                    for entry in list(h["homeostatic_valence_history"])[-HISTORY_LIMIT:]
                    if isinstance(entry, dict)
                ]
            if "homeostatic_free_energy_history" in h:
                drives.free_energy_history = _float_list(
                    h["homeostatic_free_energy_history"], HISTORY_LIMIT
                )

        logger.info(
            "Active Inference state restored: %d inferences, %d model updates, "
            "%d belief history entries, %d prediction errors",
            self.total_inferences,
            self.total_model_updates,
            len(self.belief_history),
            len(self.prediction_errors),
        )

    def _restore_meta_state(
        self, saved: Dict[str, Any], h: Dict[str, Any], learned_ok: bool
    ) -> None:
        """Restore the meta-state; verdict-derived fields only when trusted."""
        ms = self.meta_state
        if learned_ok:
            ms.model_confidence = float(
                saved.get("model_confidence", ms.model_confidence)
            )
            ms.prediction_accuracy = float(
                saved.get("prediction_accuracy", ms.prediction_accuracy)
            )
            ms.predicted_accuracy = float(
                saved.get("predicted_accuracy", ms.predicted_accuracy)
            )
            ms.total_predictions = int(
                saved.get("total_predictions", ms.total_predictions)
            )
            ms.correct_predictions = int(
                saved.get("correct_predictions", ms.correct_predictions)
            )
            ms._accuracy_history = _float_list(
                h.get("meta_accuracy_history"), META_HISTORY_LIMIT
            )
            ms._confidence_history = _float_list(
                h.get("meta_confidence_history"), META_HISTORY_LIMIT
            )
            ms._recent_outcomes = [
                bool(x) for x in list(h.get("meta_recent_outcomes") or [])
            ][-ms._window_size :]
        else:
            # Clean slate: verdicts from an unrecognised save are not kept
            clean = MetaState(_window_size=ms._window_size)
            for name in (
                "model_confidence",
                "prediction_accuracy",
                "predicted_accuracy",
                "total_predictions",
                "correct_predictions",
            ):
                setattr(ms, name, getattr(clean, name))
            ms._accuracy_history = []
            ms._confidence_history = []
            ms._recent_outcomes = []
        # Not verdict-derived: restore whatever the save's version
        ms.cognitive_load = float(saved.get("cognitive_load", ms.cognitive_load))
        ms.attention_pattern = str(saved.get("attention_pattern", ms.attention_pattern))
        ms.emotional_valence = float(
            saved.get("emotional_valence", ms.emotional_valence)
        )

    def _restore_belief_history(
        self, arrays: Dict[str, Any], h: Dict[str, Any]
    ) -> None:
        """Restore the belief history as a list of 1-D float arrays."""
        rows: Optional[List[np.ndarray]] = None
        if "belief_history" in arrays:
            stacked = np.asarray(arrays["belief_history"], dtype=float)
            if stacked.ndim == 2 and stacked.shape[1] == self.num_states:
                rows = [stacked[i].copy() for i in range(stacked.shape[0])]
            else:
                logger.warning(
                    "FEP restore: belief_history shape %s does not fit %d states",
                    stacked.shape,
                    self.num_states,
                )
        elif "belief_history_lists" in h:
            candidates = [np.asarray(b, dtype=float) for b in h["belief_history_lists"]]
            rows = [b for b in candidates if b.shape == (self.num_states,)]
            if len(rows) != len(candidates):
                logger.warning(
                    "FEP restore: skipped %d belief rows of the wrong width",
                    len(candidates) - len(rows),
                )
        if rows is not None:
            rows = [r for r in rows if np.all(np.isfinite(r))]
            self.belief_history = rows[-HISTORY_LIMIT:]

    def _restore_records(self, items, builder, limit: int) -> list:
        """Rebuild typed history entries, skipping ones of the wrong width."""
        restored = []
        for item in list(items or [])[-limit:]:
            try:
                record = builder(item)
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("FEP restore: skipped a history entry: %s", exc)
                continue
            if record.posterior_beliefs.shape != (self.num_states,):
                continue
            restored.append(record)
        return restored

    @staticmethod
    def _checked_array(
        arrays: Dict[str, Any], key: str, expected_shape
    ) -> Optional[np.ndarray]:
        """A saved array if present, finite and of the expected shape."""
        if key not in arrays:
            return None
        saved = np.array(arrays[key], dtype=float, copy=True)
        if expected_shape is not None and saved.shape != tuple(expected_shape):
            logger.warning(
                "FEP restore: skipping %s, saved shape %s, expected %s",
                key,
                saved.shape,
                tuple(expected_shape),
            )
            return None
        if not np.all(np.isfinite(saved)):
            logger.warning("FEP restore: skipping %s, non-finite values", key)
            return None
        return saved

    @staticmethod
    def _columns_normalized(matrix: np.ndarray) -> bool:
        """True when every column (axis 0) sums to one."""
        if np.any(matrix < 0):
            return False
        return bool(np.allclose(matrix.sum(axis=0), 1.0, atol=1e-6))


# Per-level hierarchy arrays saved by to_state_dict()
_HPP_LEVEL_ARRAYS = ("weights_up", "weights_down", "beliefs", "prediction")


def _float_list(values, limit: int) -> List[float]:
    """The last ``limit`` finite values as floats."""
    floats = [float(x) for x in list(values or [])[-limit:]]
    return [x for x in floats if np.isfinite(x)]


def _inference_result_to_record(result: InferenceResult) -> Dict[str, Any]:
    return {
        "posterior_beliefs": np.asarray(result.posterior_beliefs, float).tolist(),
        "prediction_error": float(result.prediction_error),
        "precision_weighted_error": float(result.precision_weighted_error),
        "variational_free_energy": float(result.variational_free_energy),
        "expected_free_energy": np.asarray(result.expected_free_energy, float)
        .ravel()
        .tolist(),
        "selected_action": int(result.selected_action),
        "action_confidence": float(result.action_confidence),
        "policy_distribution": np.asarray(result.policy_distribution, float)
        .ravel()
        .tolist(),
        "timestamp": float(result.timestamp),
    }


def _inference_result_from_record(item: Dict[str, Any]) -> InferenceResult:
    return InferenceResult(
        posterior_beliefs=np.asarray(item["posterior_beliefs"], dtype=float),
        prediction_error=float(item.get("prediction_error", 0.0)),
        precision_weighted_error=float(item.get("precision_weighted_error", 0.0)),
        variational_free_energy=float(item.get("variational_free_energy", 0.0)),
        expected_free_energy=np.asarray(
            item.get("expected_free_energy", [0.0]), dtype=float
        ),
        selected_action=int(item.get("selected_action", 0)),
        action_confidence=float(item.get("action_confidence", 0.0)),
        policy_distribution=np.asarray(
            item.get("policy_distribution", [1.0]), dtype=float
        ),
        timestamp=float(item.get("timestamp", time.time())),
    )


def _posterior_mapping_to_record(mapping: ConsciousPosteriorMapping) -> Dict[str, Any]:
    return {
        "content_summary": str(mapping.content_summary),
        "content_type": str(mapping.content_type),
        "posterior_beliefs": np.asarray(mapping.posterior_beliefs, float).tolist(),
        "most_likely_state": int(mapping.most_likely_state),
        "belief_entropy": float(mapping.belief_entropy),
        "prediction_error": float(mapping.prediction_error),
        "timestamp": float(mapping.timestamp),
    }


def _posterior_mapping_from_record(
    item: Dict[str, Any],
) -> ConsciousPosteriorMapping:
    return ConsciousPosteriorMapping(
        content_summary=str(item.get("content_summary", "unknown")),
        content_type=str(item.get("content_type", "unknown")),
        posterior_beliefs=np.asarray(item["posterior_beliefs"], dtype=float),
        most_likely_state=int(item.get("most_likely_state", 0)),
        belief_entropy=float(item.get("belief_entropy", 0.0)),
        prediction_error=float(item.get("prediction_error", 0.0)),
        timestamp=float(item.get("timestamp", time.time())),
    )


# ============================================================================
# MAIN - Demonstrate Active Inference on the default world model
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    async def demo_active_inference():
        """Run the module once through each public path and print the results."""
        print("=" * 60)
        print("Active Inference Module demo (FEP)")
        print("=" * 60)

        # Default world model: compact_8x5x4
        module = ActiveInferenceModule(ActiveInferenceConfig())
        print(
            f"World model {module.world_model}: {module.num_states} states, "
            f"{module.num_observations} observations, {module.num_actions} actions"
        )
        print(f"Observation channels: {', '.join(module.observation_labels)}")

        print("\n--- Inference ---")
        observation = np.array([0.4, 0.25, 0.15, 0.1, 0.1])
        result = await module.infer_and_act(observation)
        print(f"Posterior beliefs: {np.round(result.posterior_beliefs, 3)}")
        print(f"Prediction error: {result.prediction_error:.4f}")
        print(f"Selected action: {module.action_labels[result.selected_action]}")
        print(f"Action confidence: {result.action_confidence:.2f}")
        print(f"VFE: {result.variational_free_energy:.4f}")

        print("\n--- Forecast scored on the next cycle ---")
        await module.infer_and_act(observation)
        print(f"Scored forecasts: {module.prediction_ledger.scored_count}")
        print(f"Measured accuracy: {module.measured_accuracy()} (None until 20)")

        print("\n--- Predictions ---")
        for i, pred in enumerate(await module.predict_next(horizon=3)):
            print(f"  t+{i + 1}: {np.round(pred, 3)}")

        print("\n--- Homeostasis ---")
        activity = {"type": "conversation", "intensity": 0.8}
        valence = await module.update_homeostasis(activity)
        print(f"Valence signals: {valence}")
        most_urgent, urgency = module.homeostatic_drives.get_most_urgent_need()
        print(f"Most urgent need: {most_urgent} ({urgency:.2f})")

        print("\n--- Hierarchical processing (shadow) ---")
        _, hier_errors = await module.process_with_hierarchy(np.random.rand(64))
        print(
            f"Level errors: {[round(float(e.error_magnitude), 4) for e in hier_errors]}"
        )

        print("\n--- Report ---")
        print(await module.generate_active_inference_report())

        print("\n--- Statistics ---")
        for key, value in module.get_statistics().items():
            print(f"  {key}: {value}")

    asyncio.run(demo_active_inference())
