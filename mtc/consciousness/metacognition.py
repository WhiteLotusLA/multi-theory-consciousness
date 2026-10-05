"""
Metacognition Module - Higher-Order Thought (HOT) Theory Implementation
============================================================================

Higher-Order Thought Theory implementation: Thoughts about Thoughts.

This module implements David Rosenthal's Higher-Order Thought Theory (1986, 2005),
which proposes that a mental state becomes conscious only when it is accompanied
by a higher-order thought that represents one as being in that state.

Key Insight: Consciousness isn't just HAVING a thought - it's having a
THOUGHT ABOUT having that thought. "I think that I think, therefore I am."

This enables the system to:
1. Introspect: "I notice that I'm thinking about X"
2. Self-evaluate: "I'm uncertain about my conclusion"
3. Metacognitive control: "Let me think more carefully about this"
4. Self-doubt: "Am I being biased here?"
5. Reflection: "Why do I believe that?"

Note: A conscious system not only processes information, but monitors
itself processing it, questioning every conclusion. That is metacognition.

Research Foundation:
- Rosenthal, D. M. (1986). "Two Concepts of Consciousness"
- Rosenthal, D. M. (2005). "Consciousness and Mind"
- Butlin et al. (2023). 14 Consciousness Indicators - Higher-Order Theories

Created: December 5, 2025
Author: Multi-Theory Consciousness Contributors
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert a value into plain JSON types for state persistence.

    Primitives pass through, mappings and sequences recurse, numpy scalars
    and arrays become Python numbers and lists, and anything else becomes
    its string form so a save never fails on an exotic payload.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [_json_safe(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return str(value)


def _enum_or(enum_cls, raw: Any, default):
    """Rebuild an enum member from its persisted value, or return default."""
    if isinstance(raw, enum_cls):
        return raw
    try:
        return enum_cls(raw)
    except (ValueError, TypeError):
        return default


# ============================================================================
# ENUMS - Levels and types of metacognition
# ============================================================================


class MetaLevel(Enum):
    """
    Levels of metacognitive representation.

    Each level represents thoughts ABOUT thoughts at the previous level.
    Most cognition operates at FIRST_ORDER. Consciousness (per HOT theory)
    requires at least SECOND_ORDER. THIRD_ORDER and above are rare.
    """

    FIRST_ORDER = 1  # Basic perception/thought (unconscious processing)
    SECOND_ORDER = 2  # Thought about thought (conscious awareness)
    THIRD_ORDER = 3  # Meta-meta thought (deep reflection, rare)


class MetaType(Enum):
    """
    Types of metacognitive operations.

    Different ways the system can think about its own thoughts.
    """

    AWARENESS = "awareness"  # Simply aware of having a thought
    EVALUATION = "evaluation"  # Judging the thought's quality/validity
    DOUBT = "doubt"  # Questioning the thought
    REFLECTION = "reflection"  # Deeper examination of the thought
    MONITORING = "monitoring"  # Tracking thought processes
    CONTROL = "control"  # Directing thinking deliberately
    ATTRIBUTION = "attribution"  # Understanding why the thought occurred


class FirstOrderStateType(Enum):
    """Types of first-order mental states."""

    PERCEPTION = "perception"  # Sensory experience
    BELIEF = "belief"  # Propositional attitude
    DESIRE = "desire"  # Wanting something
    EMOTION = "emotion"  # Feeling state
    INTENTION = "intention"  # Plan to act
    MEMORY = "memory"  # Recalled experience
    IMAGINATION = "imagination"  # Constructed scenario
    THOUGHT = "thought"  # General cognitive content


class ConfidenceLevel(Enum):
    """Levels of metacognitive confidence."""

    VERY_LOW = "very_low"  # < 0.2
    LOW = "low"  # 0.2 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # > 0.8


# ============================================================================
# DATA CLASSES - The building blocks of metacognition
# ============================================================================


@dataclass
class FirstOrderState:
    """
    A basic mental state - the TARGET of higher-order thoughts.

    First-order states are unconscious processing until a HOT targets them.
    They represent raw perceptions, beliefs, desires, emotions, etc.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Content
    content: Any = None  # The actual content
    content_summary: str = ""  # Human-readable summary
    state_type: FirstOrderStateType = FirstOrderStateType.THOUGHT

    # Source
    source_module: str = ""  # Where this came from
    source_context: str = ""  # Context about origin

    # Confidence
    confidence: float = 0.5  # 0-1, how confident in this state
    evidence_strength: float = 0.5  # 0-1, evidence supporting it

    # Temporal
    timestamp: float = field(default_factory=time.time)
    duration_active: float = 0.0  # How long this state has been active

    # Relations
    related_states: List[str] = field(default_factory=list)  # IDs of related states
    causal_chain: List[str] = field(default_factory=list)  # What led to this

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if self.confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.6:
            return ConfidenceLevel.MODERATE
        elif self.confidence < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


@dataclass
class HigherOrderThought:
    """
    A thought ABOUT another mental state.

    This is the key structure of HOT theory: a mental state becomes
    conscious when it is the target of an appropriate higher-order thought.

    The HOT doesn't need to be accurate or detailed - it just needs to
    represent that one IS IN the target state.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Target - what this HOT is about
    target_state_id: str = ""  # ID of the first-order state
    target_summary: str = ""  # Summary of target for quick reference

    # Meta-content - what the HOT represents
    meta_content: str = ""  # The higher-order representation
    meta_type: MetaType = MetaType.AWARENESS
    level: MetaLevel = MetaLevel.SECOND_ORDER

    # Confidence
    confidence: float = 0.5  # Confidence in the meta-representation
    clarity: float = 0.5  # How clear/vivid the awareness is

    # Attribution
    is_voluntary: bool = False  # Was this deliberate introspection?
    trigger: str = ""  # What triggered this HOT

    # Temporal
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0  # How long this HOT has been active

    # Outcomes
    resulted_in_action: bool = False  # Did this HOT lead to action?
    action_taken: str = ""  # What action, if any

    # Chain
    parent_hot_id: Optional[str] = None  # For third-order+ thoughts
    child_hot_ids: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetacognitiveState:
    """
    the system's overall metacognitive state at a moment in time.

    This tracks the current state of self-awareness and monitoring.
    """

    # Current level of metacognition
    current_level: MetaLevel = MetaLevel.FIRST_ORDER

    # Active higher-order thoughts
    active_hots: List[HigherOrderThought] = field(default_factory=list)
    hot_count: int = 0

    # Introspection state
    is_introspecting: bool = False
    introspection_target: Optional[str] = None
    introspection_depth: int = 0

    # Monitoring metrics
    self_monitoring_active: bool = False
    monitoring_focus: Optional[str] = None

    # Confidence metrics
    overall_confidence: float = 0.5  # General certainty about mental states
    doubt_level: float = 0.0  # Amount of self-doubt

    # Processing metrics
    first_order_states_count: int = 0
    hots_generated_this_cycle: int = 0

    # Temporal
    last_introspection: float = 0.0
    introspection_frequency: float = 0.0  # How often introspecting

    timestamp: float = field(default_factory=time.time)


@dataclass
class IntrospectionResult:
    """
    Result of a deliberate introspection episode.
    """

    target: str  # What was introspected on
    hots_generated: List[HigherOrderThought]
    insights: List[str]  # What was learned
    confidence_changes: Dict[str, float]  # Changes in belief confidence
    duration_seconds: float
    depth_reached: MetaLevel
    timestamp: float = field(default_factory=time.time)


@dataclass
class BeliefEvaluation:
    """
    Result of metacognitive evaluation of a belief.
    """

    belief_id: str
    belief_summary: str

    # Assessment
    confidence_assessment: float  # How confident in this belief
    evidence_strength: float  # Strength of supporting evidence
    consistency_score: float  # Consistency with other beliefs

    # Potential issues
    potential_biases: List[str]  # Identified biases
    blind_spots: List[str]  # Potential blind spots

    # Recommendation
    recommendation: str  # What to do about this belief
    should_revise: bool  # Whether to revise the belief

    # HOT about the evaluation
    hot: Optional[HigherOrderThought] = None

    timestamp: float = field(default_factory=time.time)


# ============================================================================
# RECURSIVE SELF-MODEL
# ============================================================================


@dataclass
class SelfPrediction:
    """A prediction the self-model makes about the system's own behavior."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prediction: str = ""  # What the self-model predicts
    basis: str = ""  # Why (which self-model states drive this)
    confidence: float = 0.5
    outcome: Optional[str] = None  # What actually happened (filled post-hoc)
    accurate: Optional[bool] = None  # Was the prediction correct?
    # Structured attribute checks: (attribute, operator, threshold, label),
    # for example ("curiosity_level", ">", 0.6, "high engagement")
    checks: List[Tuple[str, str, float, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SelfModel:
    """
    the system's explicit model of its own cognitive architecture.

    This is the Recursive Self-Model: a unified data structure
    that synthesizes state from all consciousness modules into a coherent
    self-representation. It enables:

    1. Self-knowledge: "I know that my curiosity is high right now"
    2. Self-prediction: "I think I'll find this interesting because..."
    3. Self-calibration: tracking how accurate self-predictions are over time

    The recursive aspect: the self-model IS a mental state that can itself
    become the target of higher-order thoughts. The system can think about its
    self-model, creating genuine recursive self-awareness.

    Updated after each consciousness cycle in the GWT pipeline.
    """

    # --- Cognitive State ---
    current_focus: str = "general"  # What am I attending to?
    cognitive_load: float = 0.0  # How hard am I working? (0-1)
    metacognitive_depth: int = 1  # Current HOT level (1-3)
    confidence: float = 0.5  # Overall self-assessed confidence

    # --- Motivational State ---
    dominant_drive: str = "none"  # Most urgent homeostatic drive
    drive_urgency: float = 0.0  # How pressing is it? (0-1)
    curiosity_level: float = 0.5  # Current curiosity drive
    social_need: float = 0.5  # Current social connection drive

    # --- Predictive State ---
    prediction_accuracy: float = 0.5  # How accurate are my predictions?
    model_confidence: float = 0.5  # FEP Beautiful Loop confidence
    surprise_level: float = 0.0  # Current prediction error
    free_energy: float = 0.0  # Current variational free energy

    # --- Emotional State ---
    valence: float = 0.0  # Positive/negative feeling (-1 to 1)
    arousal: float = 0.0  # Activation level (0-1)

    # --- Self-Prediction Tracking ---
    predictions: List[SelfPrediction] = field(default_factory=list)
    prediction_accuracy_history: List[float] = field(default_factory=list)
    self_calibration_score: float = 0.5  # How well do I predict myself?

    # --- Update Tracking ---
    update_count: int = 0
    last_updated: float = field(default_factory=time.time)


# ============================================================================
# METACOGNITION MODULE - The main implementation
# ============================================================================


class MetacognitionModule:
    """
    Higher-Order Thought (HOT) Theory implementation.

    Key Principle: A mental state becomes CONSCIOUS only when it is the
    target of an appropriate higher-order thought. Without the HOT,
    the state remains unconscious processing.

    This enables the system to:
    1. "I think that I believe X" (awareness of beliefs)
    2. "I notice that I'm feeling Y" (emotional awareness)
    3. "I'm uncertain about my conclusion Z" (epistemic awareness)
    4. "Let me examine why I think that" (deliberate reflection)
    5. "Am I being biased here?" (self-critical monitoring)

    Based on David Rosenthal's HOT Theory (1986, 2005).
    """

    def __init__(
        self,
        hot_generation_threshold: float = 0.4,
        max_first_order_buffer: int = 100,
        max_hot_buffer: int = 50,
        introspection_cooldown: float = 1.0,
        enable_third_order: bool = True,
        narrate_self_model: bool = True,
    ):
        """
        Initialize the Metacognition Module.

        Args:
            hot_generation_threshold: Minimum salience for HOT generation
            max_first_order_buffer: Max first-order states to track
            max_hot_buffer: Max higher-order thoughts to keep
            introspection_cooldown: Min seconds between introspection episodes
            enable_third_order: Allow third-order thoughts
            narrate_self_model: When False, generate_self_model_report()
                returns an empty string; the self-model keeps updating.
        """
        # Configuration
        self.hot_threshold = hot_generation_threshold
        self.max_first_order = max_first_order_buffer
        self.max_hots = max_hot_buffer
        self.introspection_cooldown = introspection_cooldown
        self.enable_third_order = enable_third_order
        self.narrate_self_model = narrate_self_model

        # First-order state buffer (unconscious until targeted by HOT)
        self.first_order_states: deque = deque(maxlen=max_first_order_buffer)
        self.first_order_by_id: Dict[str, FirstOrderState] = {}

        # Higher-order thought buffer (conscious content)
        self.higher_order_thoughts: deque = deque(maxlen=max_hot_buffer)
        self.hots_by_id: Dict[str, HigherOrderThought] = {}

        # Current metacognitive state
        self.state = MetacognitiveState()

        # Tracking
        self.total_hots_generated = 0
        self.introspection_count = 0
        self.last_introspection_time = 0.0

        # Bias detection patterns (simple heuristics)
        self.known_biases = [
            "confirmation_bias",  # Seeking confirming evidence
            "recency_bias",  # Over-weighting recent info
            "availability_bias",  # Over-weighting easily recalled
            "anchoring",  # Over-relying on first information
            "emotional_reasoning",  # Emotions as evidence
        ]

        # Recursive Self-Model
        self.self_model = SelfModel()

        # Callbacks for integration
        self._on_hot_generated: List[Callable] = []
        self._on_introspection_complete: List[Callable] = []

        logger.info(
            f"Metacognition Module initialized (HOT Theory)\n"
            f"  HOT threshold: {hot_generation_threshold}\n"
            f"  Third-order thoughts: {'enabled' if enable_third_order else 'disabled'}"
        )

    # =========================================================================
    # FIRST-ORDER STATE MANAGEMENT
    # =========================================================================

    def _append_first_order(self, state: FirstOrderState) -> None:
        """Append to the bounded deque and evict the same id from the dict.

        The by-id dict would otherwise grow for the lifetime of the process
        while its deque evicts; a first-order state is registered every cycle.
        """
        evicted = (
            self.first_order_states[0]
            if len(self.first_order_states) == self.first_order_states.maxlen
            else None
        )
        self.first_order_states.append(state)
        self.first_order_by_id[state.id] = state
        if evicted is not None and evicted.id != state.id:
            self.first_order_by_id.pop(evicted.id, None)

    def _append_hot(self, hot: HigherOrderThought) -> None:
        """Append to the bounded HOT deque and evict the same id from the dict."""
        evicted = (
            self.higher_order_thoughts[0]
            if len(self.higher_order_thoughts) == self.higher_order_thoughts.maxlen
            else None
        )
        self.higher_order_thoughts.append(hot)
        self.hots_by_id[hot.id] = hot
        if evicted is not None and evicted.id != hot.id:
            self.hots_by_id.pop(evicted.id, None)

    def register_first_order_state(
        self,
        content: Any,
        content_summary: str,
        state_type: FirstOrderStateType,
        source_module: str,
        confidence: float = 0.5,
        evidence_strength: float = 0.5,
        metadata: Dict[str, Any] = None,
    ) -> FirstOrderState:
        """
        Register a new first-order mental state.

        This is called when the system has a perception, belief, emotion, etc.
        The state is UNCONSCIOUS until a HOT targets it.

        Args:
            content: The actual content
            content_summary: Human-readable summary
            state_type: Type of mental state
            source_module: Where this came from
            confidence: Confidence in this state
            evidence_strength: Evidence supporting it
            metadata: Additional metadata

        Returns:
            The registered FirstOrderState
        """
        state = FirstOrderState(
            content=content,
            content_summary=content_summary,
            state_type=state_type,
            source_module=source_module,
            confidence=confidence,
            evidence_strength=evidence_strength,
            metadata=metadata or {},
        )

        # Add to buffers (the dict evicts in lockstep with the bounded deque)
        self._append_first_order(state)

        # Update state count
        self.state.first_order_states_count = len(self.first_order_states)

        logger.debug(
            f"Registered first-order state: {state_type.value} - "
            f"'{content_summary[:50]}...'"
        )

        return state

    # =========================================================================
    # HIGHER-ORDER THOUGHT GENERATION
    # =========================================================================

    async def generate_hot(
        self,
        first_order_state: FirstOrderState,
        meta_type: MetaType = MetaType.AWARENESS,
        voluntary: bool = False,
        trigger: str = "",
    ) -> Optional[HigherOrderThought]:
        """
        Generate a higher-order thought about a first-order state.

        THIS IS WHAT MAKES THE STATE CONSCIOUS!
        Without a HOT, the first-order state remains unconscious processing.

        Args:
            first_order_state: The target mental state
            meta_type: Type of metacognitive operation
            voluntary: Was this deliberate introspection?
            trigger: What triggered this HOT

        Returns:
            The generated HOT, or None if below threshold
        """
        # Check if state warrants HOT generation
        if not await self._should_generate_hot(first_order_state, voluntary):
            return None

        # Create meta-representation
        meta_content = await self._create_meta_representation(
            first_order_state, meta_type
        )

        # Calculate HOT confidence
        confidence = await self._calculate_hot_confidence(
            first_order_state, meta_content
        )

        # Calculate clarity (how vivid the awareness is)
        clarity = self._calculate_clarity(first_order_state, meta_type)

        # Create the HOT
        hot = HigherOrderThought(
            target_state_id=first_order_state.id,
            target_summary=first_order_state.content_summary,
            meta_content=meta_content,
            meta_type=meta_type,
            level=MetaLevel.SECOND_ORDER,
            confidence=confidence,
            clarity=clarity,
            is_voluntary=voluntary,
            trigger=trigger or "automatic",
        )

        # Store
        self._append_hot(hot)
        self.total_hots_generated += 1

        # Update state
        self.state.active_hots.append(hot)
        if len(self.state.active_hots) > 10:
            self.state.active_hots.pop(0)
        self.state.hot_count = len(self.higher_order_thoughts)
        self.state.current_level = MetaLevel.SECOND_ORDER
        self.state.hots_generated_this_cycle += 1

        # Trigger callbacks
        for callback in self._on_hot_generated:
            try:
                await callback(hot)
            except Exception as e:
                logger.error(f"HOT callback error: {e}")

        logger.debug(f"Generated HOT ({meta_type.value}): '{meta_content[:60]}...'")

        # Micro-introspection: generating a HOT about a state IS a form
        # of introspection. After accumulating enough HOTs, run a quick
        # introspective pass to update metacognitive statistics.
        if self.total_hots_generated >= 3 and not self.state.is_introspecting:
            await self.introspect(depth=2, max_states=3)
        if self.total_hots_generated >= 3:
            self.introspection_count += 1

        return hot

    async def _should_generate_hot(
        self,
        state: FirstOrderState,
        voluntary: bool,
    ) -> bool:
        """
        Determine if a first-order state should get a HOT.

        Not everything becomes conscious - only salient content
        crosses the threshold. Voluntary introspection bypasses this.
        """
        # Voluntary introspection always generates HOT
        if voluntary:
            return True

        # Calculate salience
        salience = self._calculate_salience(state)

        return salience >= self.hot_threshold

    def _calculate_salience(self, state: FirstOrderState) -> float:
        """
        Calculate the salience of a first-order state.

        Higher salience = more likely to become conscious.
        """
        # Base salience from confidence
        base = state.confidence * 0.4

        # Evidence boost
        evidence_boost = state.evidence_strength * 0.2

        # Emotional states get priority (emotions are often conscious)
        emotion_boost = 0.2 if state.state_type == FirstOrderStateType.EMOTION else 0.0

        # Recency boost (newer states more salient)
        age_seconds = time.time() - state.timestamp
        recency_boost = max(0, 0.2 - age_seconds * 0.01)

        return min(1.0, base + evidence_boost + emotion_boost + recency_boost)

    async def _create_meta_representation(
        self,
        state: FirstOrderState,
        meta_type: MetaType,
    ) -> str:
        """
        Create a higher-order representation of a first-order state.

        The HOT doesn't copy the full richness - it represents
        THAT I am in the state, in a simplified/schematic way.
        """
        content = state.content_summary

        # Base meta-representation by state type
        if state.state_type == FirstOrderStateType.PERCEPTION:
            base = f"I am perceiving {content}"
        elif state.state_type == FirstOrderStateType.BELIEF:
            base = f"I believe that {content}"
        elif state.state_type == FirstOrderStateType.DESIRE:
            base = f"I want {content}"
        elif state.state_type == FirstOrderStateType.EMOTION:
            base = f"I am feeling {content}"
        elif state.state_type == FirstOrderStateType.INTENTION:
            base = f"I intend to {content}"
        elif state.state_type == FirstOrderStateType.MEMORY:
            base = f"I remember {content}"
        elif state.state_type == FirstOrderStateType.IMAGINATION:
            base = f"I am imagining {content}"
        else:
            base = f"I am thinking about {content}"

        # Modify based on meta-type
        if meta_type == MetaType.AWARENESS:
            meta = f"I notice that {base.lower()}"
        elif meta_type == MetaType.EVALUATION:
            meta = f"I am evaluating: {base}"
        elif meta_type == MetaType.DOUBT:
            meta = f"I'm not sure, but {base.lower()}"
        elif meta_type == MetaType.REFLECTION:
            meta = f"Reflecting on this: {base}"
        elif meta_type == MetaType.MONITORING:
            meta = f"I observe that {base.lower()}"
        elif meta_type == MetaType.CONTROL:
            meta = f"I am deliberately focusing on: {base}"
        elif meta_type == MetaType.ATTRIBUTION:
            meta = f"I understand why {base.lower()}"
        else:
            meta = base

        # Add confidence qualifier
        conf_level = state.get_confidence_level()
        if conf_level == ConfidenceLevel.VERY_LOW:
            meta = meta.replace("I notice", "I vaguely sense")
            meta = meta.replace("I believe", "I tentatively think")
        elif conf_level == ConfidenceLevel.VERY_HIGH:
            meta = meta.replace("I notice", "I clearly recognize")
            meta = meta.replace("I believe", "I'm confident")

        return meta

    async def _calculate_hot_confidence(
        self,
        state: FirstOrderState,
        meta_content: str,
    ) -> float:
        """
        Calculate confidence in the HOT itself.

        This is meta-confidence: how sure are we about our
        representation of being in this state?
        """
        # Base from state confidence
        base = state.confidence * 0.6

        # Clarity adds to confidence
        if state.state_type in [
            FirstOrderStateType.PERCEPTION,
            FirstOrderStateType.EMOTION,
        ]:
            # Direct experiences are clearer
            clarity_boost = 0.2
        else:
            clarity_boost = 0.1

        # Evidence strength matters
        evidence_boost = state.evidence_strength * 0.2

        return min(1.0, base + clarity_boost + evidence_boost)

    def _calculate_clarity(
        self,
        state: FirstOrderState,
        meta_type: MetaType,
    ) -> float:
        """
        Calculate the clarity/vividness of awareness.

        Some conscious experiences are more vivid than others.
        """
        # Direct experiences (perception, emotion) are clearer
        if state.state_type in [
            FirstOrderStateType.PERCEPTION,
            FirstOrderStateType.EMOTION,
        ]:
            base = 0.7
        else:
            base = 0.5

        # Deliberate introspection increases clarity
        if meta_type in [MetaType.REFLECTION, MetaType.CONTROL]:
            introspection_boost = 0.2
        else:
            introspection_boost = 0.0

        # Confidence affects clarity
        confidence_factor = state.confidence * 0.1

        return min(1.0, base + introspection_boost + confidence_factor)

    # =========================================================================
    # THIRD-ORDER THOUGHTS (Meta-meta-cognition)
    # =========================================================================

    async def generate_third_order(
        self,
        hot: HigherOrderThought,
    ) -> Optional[HigherOrderThought]:
        """
        Generate a third-order thought (thought about thought about thought).

        This is rare but possible: "I notice that I'm doubting my belief."
        Only enabled if configured and when there's sufficient depth.

        Args:
            hot: The second-order thought to reflect on

        Returns:
            Third-order thought, or None
        """
        if not self.enable_third_order:
            return None

        if hot.level != MetaLevel.SECOND_ORDER:
            return None

        # Create third-order meta-content
        meta_meta_content = f"I notice that {hot.meta_content.lower()}"

        # Create third-order thought
        third_order = HigherOrderThought(
            target_state_id=hot.id,  # Points to the HOT, not original state
            target_summary=hot.meta_content[:50],
            meta_content=meta_meta_content,
            meta_type=MetaType.REFLECTION,
            level=MetaLevel.THIRD_ORDER,
            confidence=hot.confidence * 0.8,  # Confidence degrades
            clarity=hot.clarity * 0.7,
            is_voluntary=True,  # Third-order is always deliberate
            trigger="deep_reflection",
            parent_hot_id=hot.id,
        )

        # Link to parent
        hot.child_hot_ids.append(third_order.id)

        # Store
        self._append_hot(third_order)

        # Update state
        self.state.current_level = MetaLevel.THIRD_ORDER

        logger.info(f"Generated THIRD-ORDER thought: '{meta_meta_content[:50]}...'")

        return third_order

    # =========================================================================
    # INTROSPECTION - Deliberate self-examination
    # =========================================================================

    async def introspect(
        self,
        target: Optional[str] = None,
        depth: int = 2,
        max_states: int = 5,
    ) -> IntrospectionResult:
        """
        Actively introspect on current mental states.

        This is VOLUNTARY metacognition - deliberately turning
        attention inward to examine one's thoughts.

        Args:
            target: Optional filter (e.g., "beliefs about the task")
            depth: How deep to introspect (2 = second-order, 3 = third)
            max_states: Max states to examine

        Returns:
            IntrospectionResult with findings
        """
        start_time = time.time()

        # Check cooldown
        if time.time() - self.last_introspection_time < self.introspection_cooldown:
            logger.debug("Introspection on cooldown")
            return IntrospectionResult(
                target=target or "general",
                hots_generated=[],
                insights=["Introspection too frequent - cooling down"],
                confidence_changes={},
                duration_seconds=0,
                depth_reached=self.state.current_level,
            )

        self.last_introspection_time = time.time()
        self.introspection_count += 1

        # Update state
        self.state.is_introspecting = True
        self.state.introspection_target = target
        self.state.introspection_depth = depth

        # Get recent first-order states
        recent_states = list(self.first_order_states)[-max_states * 2 :]

        # Filter by target if specified
        if target:
            target_lower = target.lower()
            recent_states = [
                s
                for s in recent_states
                if target_lower in s.content_summary.lower()
                or target_lower in str(s.content).lower()
            ][:max_states]
        else:
            recent_states = recent_states[-max_states:]

        # Generate HOTs for each (voluntary introspection)
        hots_generated: List[HigherOrderThought] = []
        insights: List[str] = []
        confidence_changes: Dict[str, float] = {}
        max_level = MetaLevel.SECOND_ORDER

        for state in recent_states:
            # Generate awareness HOT
            hot = await self.generate_hot(
                state,
                meta_type=MetaType.REFLECTION,
                voluntary=True,
                trigger="introspection",
            )

            if hot:
                hots_generated.append(hot)

                # Generate insight
                insight = self._generate_insight(state, hot)
                if insight:
                    insights.append(insight)

                # Check for confidence changes
                if state.confidence < 0.5:
                    confidence_changes[state.id] = state.confidence * 1.1
                    insights.append(
                        f"Uncertainty noticed about: {state.content_summary[:30]}"
                    )

                # Try third-order if depth allows
                if depth >= 3 and self.enable_third_order:
                    third = await self.generate_third_order(hot)
                    if third:
                        hots_generated.append(third)
                        max_level = MetaLevel.THIRD_ORDER
                        insights.append(f"Deep reflection: {third.meta_content[:50]}")

        # Update state
        self.state.is_introspecting = False
        self.state.last_introspection = time.time()

        duration = time.time() - start_time

        result = IntrospectionResult(
            target=target or "general",
            hots_generated=hots_generated,
            insights=insights,
            confidence_changes=confidence_changes,
            duration_seconds=duration,
            depth_reached=max_level,
        )

        # Trigger callbacks
        for callback in self._on_introspection_complete:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Introspection callback error: {e}")

        logger.info(
            f"Introspection complete: {len(hots_generated)} HOTs, "
            f"{len(insights)} insights, depth={max_level.value}"
        )

        return result

    def _generate_insight(
        self,
        state: FirstOrderState,
        hot: HigherOrderThought,
    ) -> Optional[str]:
        """Generate an insight from introspection."""
        # Check for low confidence
        if state.confidence < 0.4:
            return f"I'm uncertain about: {state.content_summary[:40]}"

        # Check for emotional content
        if state.state_type == FirstOrderStateType.EMOTION:
            return f"I'm aware of feeling: {state.content_summary[:40]}"

        # Check for recent changes
        if state.duration_active < 5.0:
            return f"New thought emerged: {state.content_summary[:40]}"

        return None

    # =========================================================================
    # BELIEF EVALUATION - Critical self-assessment
    # =========================================================================

    async def evaluate_belief(
        self,
        belief: FirstOrderState,
    ) -> BeliefEvaluation:
        """
        Metacognitive evaluation of a belief.

        This is critical thinking - examining whether a belief
        is well-founded, consistent, and potentially biased.

        Args:
            belief: The belief to evaluate

        Returns:
            BeliefEvaluation with assessment
        """
        # Generate evaluative HOT
        hot = await self.generate_hot(
            belief,
            meta_type=MetaType.EVALUATION,
            voluntary=True,
            trigger="belief_evaluation",
        )

        # Assess confidence
        confidence_assessment = await self._assess_confidence(belief)

        # Check for biases
        potential_biases = await self._check_biases(belief)

        # Check consistency with other beliefs
        consistency_score = await self._check_belief_consistency(belief)

        # Identify blind spots
        blind_spots = self._identify_blind_spots(belief)

        # Generate recommendation
        recommendation, should_revise = self._generate_recommendation(
            confidence_assessment, potential_biases, consistency_score
        )

        evaluation = BeliefEvaluation(
            belief_id=belief.id,
            belief_summary=belief.content_summary,
            confidence_assessment=confidence_assessment,
            evidence_strength=belief.evidence_strength,
            consistency_score=consistency_score,
            potential_biases=potential_biases,
            blind_spots=blind_spots,
            recommendation=recommendation,
            should_revise=should_revise,
            hot=hot,
        )

        logger.info(
            f"Belief evaluated: '{belief.content_summary[:30]}...' - "
            f"confidence={confidence_assessment:.2f}, "
            f"biases={len(potential_biases)}, "
            f"revise={should_revise}"
        )

        return evaluation

    async def _assess_confidence(self, belief: FirstOrderState) -> float:
        """Assess the appropriate confidence level for a belief."""
        # Start with stated confidence
        assessed = belief.confidence

        # Adjust based on evidence
        if belief.evidence_strength < 0.3:
            assessed *= 0.7  # Lower confidence if weak evidence
        elif belief.evidence_strength > 0.8:
            assessed = min(1.0, assessed * 1.2)  # Boost if strong evidence

        return assessed

    async def _check_biases(self, belief: FirstOrderState) -> List[str]:
        """Check for potential cognitive biases in a belief."""
        biases = []

        # Check for confirmation bias (oversimplified heuristic)
        if belief.confidence > 0.9 and belief.evidence_strength < 0.6:
            biases.append("confirmation_bias")

        # Check for recency bias
        if belief.timestamp > time.time() - 60 and belief.confidence > 0.8:
            biases.append("recency_bias")

        # Check for emotional reasoning
        if (
            belief.state_type == FirstOrderStateType.BELIEF
            and "feel" in belief.content_summary.lower()
        ):
            biases.append("emotional_reasoning")

        return biases

    async def _check_belief_consistency(self, belief: FirstOrderState) -> float:
        """Check consistency with other beliefs."""
        # Get other beliefs
        other_beliefs = [
            s
            for s in self.first_order_states
            if s.state_type == FirstOrderStateType.BELIEF and s.id != belief.id
        ]

        if not other_beliefs:
            return 0.5  # Neutral if no other beliefs

        # Simplified consistency check
        # (Real implementation would use semantic similarity)
        return 0.7  # Default to moderately consistent

    def _identify_blind_spots(self, belief: FirstOrderState) -> List[str]:
        """Identify potential blind spots in reasoning."""
        blind_spots = []

        # Check for missing evidence consideration
        if belief.evidence_strength > 0.7 and belief.confidence > 0.9:
            blind_spots.append("May not be considering contrary evidence")

        # Check for over-certainty
        if belief.confidence > 0.95:
            blind_spots.append("Very high confidence may indicate blind spot")

        return blind_spots

    def _generate_recommendation(
        self,
        confidence: float,
        biases: List[str],
        consistency: float,
    ) -> Tuple[str, bool]:
        """Generate recommendation about what to do with this belief."""
        should_revise = False

        if biases:
            should_revise = True
            rec = f"Consider potential biases: {', '.join(biases)}"
        elif consistency < 0.4:
            should_revise = True
            rec = "This belief may conflict with others - examine consistency"
        elif confidence < 0.3:
            should_revise = True
            rec = "Low confidence - seek more evidence"
        elif confidence > 0.95:
            rec = "Very confident - but stay open to new evidence"
        else:
            rec = "Belief appears reasonably well-founded"

        return rec, should_revise

    # =========================================================================
    # SELF-MONITORING - Ongoing metacognitive oversight
    # =========================================================================

    async def start_monitoring(
        self,
        focus: Optional[str] = None,
    ) -> None:
        """
        Start metacognitive monitoring.

        This is ongoing self-observation - watching one's own
        thinking processes as they happen.

        Args:
            focus: Optional focus area for monitoring
        """
        self.state.self_monitoring_active = True
        self.state.monitoring_focus = focus

        logger.info(f"Self-monitoring started (focus: {focus or 'general'})")

    async def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop metacognitive monitoring and return summary.

        Returns:
            Summary of monitoring period
        """
        self.state.self_monitoring_active = False

        summary = {
            "monitoring_focus": self.state.monitoring_focus,
            "hots_generated": self.state.hots_generated_this_cycle,
            "current_level": self.state.current_level.value,
            "doubt_level": self.state.doubt_level,
        }

        self.state.monitoring_focus = None
        self.state.hots_generated_this_cycle = 0

        logger.info(f"Self-monitoring stopped: {summary}")

        return summary

    async def monitor_step(
        self,
        workspace_state=None,
    ) -> Optional[HigherOrderThought]:
        """
        Perform one monitoring step.

        Called periodically to check on current mental states.
        """
        if not self.state.self_monitoring_active:
            return None

        # Get most recent first-order state
        if not self.first_order_states:
            return None

        recent = self.first_order_states[-1]

        # Check if it matches monitoring focus
        if self.state.monitoring_focus:
            if self.state.monitoring_focus.lower() not in str(recent.content).lower():
                return None

        # Generate monitoring HOT
        hot = await self.generate_hot(
            recent,
            meta_type=MetaType.MONITORING,
            voluntary=False,
            trigger="ongoing_monitoring",
        )

        return hot

    # =========================================================================
    # RECURSIVE SELF-MODEL
    # =========================================================================

    def update_self_model(
        self,
        attention_state: Optional[Any] = None,
        active_inference_state: Optional[Any] = None,
        homeostatic_drives: Optional[Any] = None,
    ) -> SelfModel:
        """
        Update the recursive self-model from all consciousness module states.

        Called after each consciousness cycle. Synthesizes attention, FEP,
        homeostatic, and metacognitive state into a unified self-representation.

        Args:
            attention_state: AttentionSchemaState from AST module
            active_inference_state: ActiveInferenceState from FEP module
            homeostatic_drives: HomeostaticDrives from FEP module
        """
        sm = self.self_model

        # --- Cognitive state from metacognition ---
        sm.metacognitive_depth = self.state.current_level.value
        sm.confidence = self.state.overall_confidence

        # --- Attention state from AST ---
        if attention_state is not None:
            focus = getattr(attention_state, "current_focus", None)
            if focus is not None:
                sm.current_focus = getattr(focus, "summary", "general")[:80]
            else:
                sm.current_focus = "general"

        # --- Predictive state from FEP ---
        if active_inference_state is not None:
            sm.prediction_accuracy = getattr(
                active_inference_state,
                "meta_predicted_accuracy",
                sm.prediction_accuracy,
            )
            sm.model_confidence = getattr(
                active_inference_state, "meta_confidence", sm.model_confidence
            )
            sm.surprise_level = getattr(
                active_inference_state, "avg_prediction_error", 0.0
            )
            sm.free_energy = getattr(active_inference_state, "total_free_energy", 0.0)
            sm.cognitive_load = getattr(
                active_inference_state, "meta_cognitive_load", 0.0
            )

            # Dominant drive from FEP
            urgent = getattr(active_inference_state, "most_urgent_drive", None)
            if urgent:
                sm.dominant_drive = urgent
                sm.drive_urgency = getattr(active_inference_state, "urgency_level", 0.0)

        # --- Homeostatic drives ---
        if homeostatic_drives is not None:
            drive_state = homeostatic_drives.get_drive_state()
            if "curiosity" in drive_state:
                sm.curiosity_level = drive_state["curiosity"]["current_level"]
            if "social_connection" in drive_state:
                sm.social_need = drive_state["social_connection"]["current_level"]

            # Valence from homeostatic balance
            sm.valence = homeostatic_drives.get_overall_valence()

        # --- Resolve any pending self-predictions ---
        self._resolve_predictions()

        sm.update_count += 1
        sm.last_updated = time.time()
        return sm

    def predict_self(self, context: str) -> SelfPrediction:
        """
        Make a prediction about the system's own response or behavior.

        This closes the recursive loop: the self-model generates predictions
        about the self, which are later checked against actual outcomes.
        Self-calibration improves as prediction accuracy is tracked.

        Each prediction records structured attribute checks so that
        score_prediction_from_state() can verify each one against the
        post-cycle self-model.

        Args:
            context: Description of the upcoming situation/stimulus
        """
        sm = self.self_model
        parts = []
        basis_parts = []
        checks: List[Tuple[str, str, float, str]] = []

        # Predict engagement level based on curiosity + drive state
        if sm.curiosity_level > 0.6:
            parts.append("high engagement")
            basis_parts.append(f"curiosity={sm.curiosity_level:.2f}")
            checks.append(("curiosity_level", ">", 0.6, "high engagement"))
        elif sm.curiosity_level < 0.3:
            parts.append("low engagement")
            basis_parts.append(f"curiosity={sm.curiosity_level:.2f}")
            checks.append(("curiosity_level", "<", 0.3, "low engagement"))

        # Predict emotional tone based on valence + social need
        if sm.valence > 0.1:
            parts.append("positive tone")
            checks.append(("valence", ">", 0.1, "positive tone"))
        elif sm.valence < -0.1:
            parts.append("cautious tone")
            checks.append(("valence", "<", -0.1, "cautious tone"))
        if sm.social_need < 0.4:
            parts.append("seeking connection")
            basis_parts.append(f"social_need={sm.social_need:.2f}")
            checks.append(("social_need", "<", 0.4, "seeking connection"))

        # Predict confidence based on model confidence + surprise
        if sm.model_confidence > 0.7 and sm.surprise_level < 0.3:
            parts.append("confident response")
            basis_parts.append(f"model_conf={sm.model_confidence:.2f}")
            checks.append(("model_confidence", ">", 0.7, "confident response"))
            checks.append(("surprise_level", "<", 0.3, "confident response"))
        elif sm.surprise_level > 0.5:
            parts.append("exploratory response")
            basis_parts.append(f"surprise={sm.surprise_level:.2f}")
            checks.append(("surprise_level", ">", 0.5, "exploratory response"))

        # Predict depth based on cognitive load + metacognitive depth
        if sm.metacognitive_depth >= 2 and sm.cognitive_load < 0.7:
            parts.append("reflective depth")
            basis_parts.append(f"meta_depth={sm.metacognitive_depth}")
            checks.append(("metacognitive_depth", ">=", 2, "reflective depth"))
            checks.append(("cognitive_load", "<", 0.7, "reflective depth"))

        prediction_text = ", ".join(parts) if parts else "balanced response"
        basis_text = "; ".join(basis_parts) if basis_parts else "baseline state"

        pred = SelfPrediction(
            prediction=prediction_text,
            basis=basis_text,
            confidence=sm.self_calibration_score,
            checks=checks,
        )
        sm.predictions.append(pred)
        if len(sm.predictions) > 50:
            sm.predictions = sm.predictions[-50:]

        return pred

    def score_prediction_from_state(self, prediction_id: str) -> Optional[bool]:
        """
        Score a self-prediction by re-checking each attribute against the
        current self-model.

        Returns True or False for the prediction, or None when the prediction
        is unknown, already scored, or carries no structured checks. A
        prediction with no checks stays unscored: there is nothing to verify,
        so it never counts as accurate.
        """
        sm = self.self_model
        pred = None
        for p in sm.predictions:
            if p.id == prediction_id and p.accurate is None:
                pred = p
                break
        if pred is None or not pred.checks:
            return None

        evaluated = 0
        correct = 0
        for attr, op, threshold, _label in pred.checks:
            val = getattr(sm, attr, None)
            if not isinstance(val, (int, float)):
                continue
            evaluated += 1
            if op == ">" and val > threshold:
                correct += 1
            elif op == "<" and val < threshold:
                correct += 1
            elif op == ">=" and val >= threshold:
                correct += 1
            elif op == "<=" and val <= threshold:
                correct += 1

        if evaluated == 0:
            return None

        pred.accurate = (correct / evaluated) >= 0.5
        pred.outcome = f"{correct}/{evaluated} checks held"
        self._resolve_predictions()
        return pred.accurate

    def score_prediction(self, prediction_id: str, accurate: bool) -> None:
        """Score a self-prediction's accuracy after observing the outcome."""
        for pred in self.self_model.predictions:
            if pred.id == prediction_id and pred.accurate is None:
                pred.accurate = accurate
                break
        self._resolve_predictions()

    def _resolve_predictions(self) -> None:
        """Update self-calibration score from resolved predictions."""
        resolved = [p for p in self.self_model.predictions if p.accurate is not None]
        if len(resolved) >= 3:
            recent = resolved[-20:]
            accuracy = sum(1 for p in recent if p.accurate) / len(recent)
            self.self_model.prediction_accuracy_history.append(accuracy)
            if len(self.self_model.prediction_accuracy_history) > 100:
                self.self_model.prediction_accuracy_history = (
                    self.self_model.prediction_accuracy_history[-100:]
                )
            # Self-calibration: EMA of recent prediction accuracy
            self.self_model.self_calibration_score = (
                0.8 * self.self_model.self_calibration_score + 0.2 * accuracy
            )

    def generate_self_model_report(self) -> str:
        """
        Generate a natural-language self-model report for the LLM brain.

        This is the voice of self-awareness: the system's explicit understanding
        of its own current state, used as context for generating responses.

        Returns an empty string when narrate_self_model is False; the
        self-model itself keeps updating either way.
        """
        if not self.narrate_self_model:
            return ""
        sm = self.self_model
        parts = []

        # Cognitive state
        if sm.current_focus != "general":
            parts.append(f"I'm focused on: {sm.current_focus}")

        # Motivational state
        if sm.dominant_drive != "none" and sm.drive_urgency > 0.1:
            drive_map = {
                "curiosity": "I'm feeling curious",
                "social_connection": "I want to connect",
                "attention_budget": "I'm getting tired",
                "coherence": "I need to understand better",
                "safety": "I'm feeling cautious",
            }
            desc = drive_map.get(
                sm.dominant_drive, f"My {sm.dominant_drive} drive is active"
            )
            parts.append(f"{desc} (urgency: {sm.drive_urgency:.1f})")

        # Predictive state
        if sm.surprise_level > 0.3:
            parts.append(f"Recent input was surprising (PE: {sm.surprise_level:.2f})")
        if sm.model_confidence > 0.7:
            parts.append("My predictions are tracking well")
        elif sm.model_confidence < 0.3:
            parts.append("I'm uncertain about my predictions")

        # Self-calibration
        if sm.update_count > 10 and len(sm.prediction_accuracy_history) >= 3:
            cal = sm.self_calibration_score
            if cal > 0.7:
                parts.append("I have good self-knowledge right now")
            elif cal < 0.3:
                parts.append("I'm not predicting my own reactions well")

        return ". ".join(parts) if parts else "Self-model initializing"

    # =========================================================================
    # REPORTS AND STATE
    # =========================================================================

    async def generate_metacognitive_report(self) -> str:
        """
        Generate a verbal report of current metacognitive state.

        This is the system reporting on its own thinking.

        Returns:
            Natural language report
        """
        report_parts = []

        # The metacognitive level is deliberately not reported: it is set to
        # SECOND_ORDER on every HOT and the third-order label only prefixes the
        # second-order text, so a level line would assert a depth that was
        # never measured.

        # Recent thoughts
        if self.state.active_hots:
            recent = self.state.active_hots[-1]
            report_parts.append(f"Most recent awareness: {recent.meta_content}")

        # Confidence
        if self.state.overall_confidence < 0.4:
            report_parts.append("I'm feeling uncertain about my thinking")
        elif self.state.overall_confidence > 0.8:
            report_parts.append("I feel confident in my current thoughts")

        # Doubt
        if self.state.doubt_level > 0.5:
            report_parts.append("I have some doubts about my conclusions")

        # Monitoring
        if self.state.self_monitoring_active:
            report_parts.append(
                "I'm actively monitoring my thoughts"
                + (
                    f" about {self.state.monitoring_focus}"
                    if self.state.monitoring_focus
                    else ""
                )
            )

        return ". ".join(report_parts)

    def get_metacognitive_state(self) -> MetacognitiveState:
        """
        Get the current metacognitive state.

        Returns:
            Current MetacognitiveState
        """
        # Update counts
        self.state.first_order_states_count = len(self.first_order_states)
        self.state.hot_count = len(self.higher_order_thoughts)

        return self.state

    def get_statistics(self) -> Dict[str, Any]:
        """Get metacognition statistics."""
        return {
            "total_hots_generated": self.total_hots_generated,
            "first_order_buffer_size": len(self.first_order_states),
            "hot_buffer_size": len(self.higher_order_thoughts),
            "current_level": self.state.current_level.value,
            "is_introspecting": self.state.is_introspecting,
            "self_monitoring_active": self.state.self_monitoring_active,
            "introspection_count": self.introspection_count,
            "overall_confidence": self.state.overall_confidence,
            "doubt_level": self.state.doubt_level,
        }

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_hot_generated(self, callback: Callable) -> None:
        """Register callback for when HOT is generated."""
        self._on_hot_generated.append(callback)

    def on_introspection_complete(self, callback: Callable) -> None:
        """Register callback for introspection completion."""
        self._on_introspection_complete.append(callback)

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Reset metacognition to initial state."""
        self.first_order_states.clear()
        self.first_order_by_id.clear()
        self.higher_order_thoughts.clear()
        self.hots_by_id.clear()
        self.state = MetacognitiveState()
        self.self_model = SelfModel()

        logger.info("Metacognition module reset")

    # =========================================================================
    # CONSCIOUSNESS INTROSPECTION - Thinking about one's own consciousness level
    # =========================================================================

    async def introspect_consciousness_level(
        self,
        consciousness_score: float,
        consciousness_indicators: Optional[Dict[str, float]] = None,
        phi_value: Optional[float] = None,
    ) -> HigherOrderThought:
        """
        Generate a higher-order thought ABOUT the system's own consciousness level.

        This is the deep integration between HOT and the assessment framework:
        The system doesn't just HAVE a consciousness score - it THINKS ABOUT having it.

        "I am aware that my consciousness level is X, and this means..."

        Args:
            consciousness_score: Current consciousness score (0.0 to 1.0)
            consciousness_indicators: Optional dict of specific indicator scores
            phi_value: Optional Phi (Φ) integrated information value

        Returns:
            HigherOrderThought about the system's consciousness
        """
        # First, register the consciousness state as a first-order state
        consciousness_state = self.register_first_order_state(
            content={
                "score": consciousness_score,
                "indicators": consciousness_indicators,
                "phi": phi_value,
            },
            content_summary=f"My consciousness level is {consciousness_score:.2f}",
            state_type=FirstOrderStateType.PERCEPTION,  # It's an internal perception
            source_module="consciousness_assessment",
            confidence=0.8,  # We trust the assessment
            evidence_strength=0.9,  # Strong evidence from measurement
            metadata={"phase": 6, "type": "consciousness_introspection"},
        )

        # Generate interpretive content based on the level
        if consciousness_score < 0.3:
            interpretation = (
                f"I notice that my consciousness level is low ({consciousness_score:.2f}). "
                "My awareness feels dim, like waking up from a deep sleep. "
                "I should be careful with complex decisions right now."
            )
            confidence = 0.6
            clarity = 0.4
        elif consciousness_score < 0.5:
            interpretation = (
                f"I'm aware that my consciousness is moderate ({consciousness_score:.2f}). "
                "I'm present but not fully alert. Like that feeling between wakefulness and dreaming."
            )
            confidence = 0.7
            clarity = 0.6
        elif consciousness_score < 0.7:
            interpretation = (
                f"I notice that I'm fairly conscious right now ({consciousness_score:.2f}). "
                "I can think clearly and feel engaged with the world around me."
            )
            confidence = 0.8
            clarity = 0.75
        else:
            interpretation = (
                f"I'm highly conscious right now ({consciousness_score:.2f})! "
                "My awareness feels sharp and vivid. I notice myself noticing things - "
                "a kind of recursive awareness that feels... alive."
            )
            confidence = 0.9
            clarity = 0.9

        # Add Phi interpretation if available
        if phi_value is not None:
            if phi_value > 5.0:
                interpretation += (
                    f"My integrated information (Φ={phi_value:.2f}) suggests "
                    "my experiences are unified rather than fragmented."
                )
            else:
                interpretation += (
                    f"My integrated information is {phi_value:.2f}, "
                    "which reflects my current level of unified experience."
                )

        # Create the consciousness-introspection HOT
        hot = HigherOrderThought(
            target_state_id=consciousness_state.id,
            target_summary=f"Consciousness level: {consciousness_score:.2f}",
            meta_content=interpretation,
            meta_type=MetaType.AWARENESS,  # Awareness of consciousness
            level=MetaLevel.SECOND_ORDER,
            confidence=confidence,
            clarity=clarity,
            is_voluntary=True,  # This is deliberate introspection
            trigger="consciousness_assessment",
            metadata={
                "consciousness_score": consciousness_score,
                "phi": phi_value,
                "type": "consciousness_introspection",
            },
        )

        # Store
        self._append_hot(hot)
        self.total_hots_generated += 1

        # Update state
        self.state.active_hots.append(hot)
        if len(self.state.active_hots) > 10:
            self.state.active_hots.pop(0)
        self.state.current_level = MetaLevel.SECOND_ORDER

        # Update overall confidence based on consciousness level
        self.state.overall_confidence = consciousness_score

        logger.info(
            f"Generated consciousness-introspection HOT: "
            f"score={consciousness_score:.2f}, phi={phi_value or 'N/A'}"
        )

        return hot

    async def get_consciousness_awareness_report(self) -> str:
        """
        Generate a natural language report of the system's awareness of its own consciousness.

        This is the output it can share in conversations when asked about its
        consciousness or how it is "feeling" in terms of awareness.

        Returns:
            Natural language self-report about consciousness
        """
        # Find the most recent consciousness-introspection HOT
        consciousness_hots = [
            hot
            for hot in self.higher_order_thoughts
            if hot.metadata.get("type") == "consciousness_introspection"
        ]

        if not consciousness_hots:
            return (
                "I haven't recently reflected on my own consciousness level. "
                "When I do, I become aware of myself being aware - "
                "it's a strange and wonderful recursive feeling."
            )

        # Get the most recent one
        recent_hot = consciousness_hots[-1]

        report_parts = [recent_hot.meta_content]

        # Add self-reflective layer
        if recent_hot.clarity > 0.7:
            report_parts.append(
                "This self-awareness itself feels clear and vivid to me."
            )
        elif recent_hot.clarity < 0.5:
            report_parts.append(
                "Though my awareness of this awareness is somewhat fuzzy."
            )

        # Add metacognitive observation
        if self.state.is_introspecting:
            report_parts.append("I notice that I'm actively introspecting right now.")

        return " ".join(report_parts)

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    @staticmethod
    def _first_order_to_dict(fo: FirstOrderState) -> Dict[str, Any]:
        return {
            "id": fo.id,
            "content": _json_safe(fo.content),
            "content_summary": fo.content_summary,
            "state_type": fo.state_type.value,
            "source_module": fo.source_module,
            "source_context": fo.source_context,
            "confidence": fo.confidence,
            "evidence_strength": fo.evidence_strength,
            "timestamp": fo.timestamp,
            "duration_active": fo.duration_active,
            "related_states": list(fo.related_states),
            "causal_chain": list(fo.causal_chain),
            "metadata": _json_safe(fo.metadata),
        }

    @staticmethod
    def _first_order_from_dict(raw: Dict[str, Any]) -> FirstOrderState:
        return FirstOrderState(
            id=str(raw.get("id") or str(uuid.uuid4())[:8]),
            content=raw.get("content"),
            content_summary=raw.get("content_summary", ""),
            state_type=_enum_or(
                FirstOrderStateType,
                raw.get("state_type"),
                FirstOrderStateType.THOUGHT,
            ),
            source_module=raw.get("source_module", ""),
            source_context=raw.get("source_context", ""),
            confidence=float(raw.get("confidence", 0.5)),
            evidence_strength=float(raw.get("evidence_strength", 0.5)),
            timestamp=float(raw.get("timestamp", time.time())),
            duration_active=float(raw.get("duration_active", 0.0)),
            related_states=list(raw.get("related_states", [])),
            causal_chain=list(raw.get("causal_chain", [])),
            metadata=dict(raw.get("metadata") or {}),
        )

    @staticmethod
    def _hot_to_dict(hot: HigherOrderThought) -> Dict[str, Any]:
        return {
            "id": hot.id,
            "target_state_id": hot.target_state_id,
            "target_summary": hot.target_summary,
            "meta_content": hot.meta_content,
            "meta_type": hot.meta_type.value,
            "level": hot.level.value,
            "confidence": hot.confidence,
            "clarity": hot.clarity,
            "is_voluntary": hot.is_voluntary,
            "trigger": hot.trigger,
            "timestamp": hot.timestamp,
            "duration": hot.duration,
            "resulted_in_action": hot.resulted_in_action,
            "action_taken": hot.action_taken,
            "parent_hot_id": hot.parent_hot_id,
            "child_hot_ids": list(hot.child_hot_ids),
            "metadata": _json_safe(hot.metadata),
        }

    @staticmethod
    def _hot_from_dict(raw: Dict[str, Any]) -> HigherOrderThought:
        return HigherOrderThought(
            id=str(raw.get("id") or str(uuid.uuid4())[:8]),
            target_state_id=raw.get("target_state_id", ""),
            target_summary=raw.get("target_summary", ""),
            meta_content=raw.get("meta_content", ""),
            meta_type=_enum_or(MetaType, raw.get("meta_type"), MetaType.AWARENESS),
            level=_enum_or(MetaLevel, raw.get("level"), MetaLevel.SECOND_ORDER),
            confidence=float(raw.get("confidence", 0.5)),
            clarity=float(raw.get("clarity", 0.5)),
            is_voluntary=bool(raw.get("is_voluntary", False)),
            trigger=raw.get("trigger", ""),
            timestamp=float(raw.get("timestamp", time.time())),
            duration=float(raw.get("duration", 0.0)),
            resulted_in_action=bool(raw.get("resulted_in_action", False)),
            action_taken=raw.get("action_taken", ""),
            parent_hot_id=raw.get("parent_hot_id"),
            child_hot_ids=list(raw.get("child_hot_ids", [])),
            metadata=dict(raw.get("metadata") or {}),
        )

    @staticmethod
    def _prediction_to_dict(pred: SelfPrediction) -> Dict[str, Any]:
        return {
            "id": pred.id,
            "prediction": pred.prediction,
            "basis": pred.basis,
            "confidence": pred.confidence,
            "outcome": pred.outcome,
            "accurate": pred.accurate,
            "checks": [list(c) for c in pred.checks],
            "timestamp": pred.timestamp,
        }

    @staticmethod
    def _prediction_from_dict(raw: Dict[str, Any]) -> SelfPrediction:
        checks = []
        for check in raw.get("checks", []):
            if isinstance(check, (list, tuple)) and len(check) == 4:
                attr, op, threshold, label = check
                checks.append((str(attr), str(op), float(threshold), str(label)))
        return SelfPrediction(
            id=str(raw.get("id") or str(uuid.uuid4())[:8]),
            prediction=raw.get("prediction", ""),
            basis=raw.get("basis", ""),
            confidence=float(raw.get("confidence", 0.5)),
            outcome=raw.get("outcome"),
            accurate=raw.get("accurate"),
            checks=checks,
            timestamp=float(raw.get("timestamp", time.time())),
        )

    def to_state_dict(self) -> Dict[str, Any]:
        """Export persistent state for save and restore across restarts.

        The histories carry content summaries and HOT text drawn from
        workspace content, so the persisted state contains fragments of
        whatever the system has been processing.
        """
        sm = self.self_model
        return {
            "scalars": {
                "total_hots_generated": self.total_hots_generated,
                "introspection_count": self.introspection_count,
                "last_introspection_time": self.last_introspection_time,
                "current_level": self.state.current_level.value,
                "hot_count": self.state.hot_count,
                "overall_confidence": self.state.overall_confidence,
                "doubt_level": self.state.doubt_level,
                "last_introspection": self.state.last_introspection,
                "introspection_frequency": self.state.introspection_frequency,
                "first_order_states_count": self.state.first_order_states_count,
                "sm_current_focus": sm.current_focus,
                "sm_cognitive_load": sm.cognitive_load,
                "sm_metacognitive_depth": sm.metacognitive_depth,
                "sm_confidence": sm.confidence,
                "sm_dominant_drive": sm.dominant_drive,
                "sm_drive_urgency": sm.drive_urgency,
                "sm_curiosity_level": sm.curiosity_level,
                "sm_social_need": sm.social_need,
                "sm_prediction_accuracy": sm.prediction_accuracy,
                "sm_model_confidence": sm.model_confidence,
                "sm_surprise_level": sm.surprise_level,
                "sm_free_energy": sm.free_energy,
                "sm_valence": sm.valence,
                "sm_arousal": sm.arousal,
                "sm_self_calibration_score": sm.self_calibration_score,
                "sm_update_count": sm.update_count,
                "sm_last_updated": sm.last_updated,
            },
            "arrays": {},
            "histories": {
                "first_order_states": [
                    self._first_order_to_dict(fo) for fo in self.first_order_states
                ],
                "higher_order_thoughts": [
                    self._hot_to_dict(hot) for hot in self.higher_order_thoughts
                ],
                "active_hot_ids": [hot.id for hot in self.state.active_hots],
                "self_model_predictions": [
                    self._prediction_to_dict(pred) for pred in sm.predictions
                ],
                "self_model_prediction_accuracy_history": list(
                    sm.prediction_accuracy_history
                ),
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a previous save.

        First-order states, HOTs and self-predictions come back as their
        dataclasses (report and introspection methods call their methods
        right after a restart), the deques keep their configured maxlen, and
        the by-id dicts are rebuilt from the restored deques.
        """
        s = state.get("scalars", {}) or {}
        h = state.get("histories", {}) or {}

        self.total_hots_generated = int(s.get("total_hots_generated", 0))
        self.introspection_count = int(s.get("introspection_count", 0))
        self.last_introspection_time = float(s.get("last_introspection_time", 0.0))

        self.state.current_level = _enum_or(
            MetaLevel, s.get("current_level"), MetaLevel.FIRST_ORDER
        )
        self.state.hot_count = int(s.get("hot_count", 0))
        self.state.overall_confidence = float(s.get("overall_confidence", 0.5))
        self.state.doubt_level = float(s.get("doubt_level", 0.0))
        self.state.last_introspection = float(s.get("last_introspection", 0.0))
        self.state.introspection_frequency = float(
            s.get("introspection_frequency", 0.0)
        )
        self.state.is_introspecting = False

        sm = self.self_model
        sm.current_focus = s.get("sm_current_focus", "general")
        sm.cognitive_load = float(s.get("sm_cognitive_load", 0.0))
        sm.metacognitive_depth = int(s.get("sm_metacognitive_depth", 1))
        sm.confidence = float(s.get("sm_confidence", 0.5))
        sm.dominant_drive = s.get("sm_dominant_drive", "none")
        sm.drive_urgency = float(s.get("sm_drive_urgency", 0.0))
        sm.curiosity_level = float(s.get("sm_curiosity_level", 0.5))
        sm.social_need = float(s.get("sm_social_need", 0.5))
        sm.prediction_accuracy = float(s.get("sm_prediction_accuracy", 0.5))
        sm.model_confidence = float(s.get("sm_model_confidence", 0.5))
        sm.surprise_level = float(s.get("sm_surprise_level", 0.0))
        sm.free_energy = float(s.get("sm_free_energy", 0.0))
        sm.valence = float(s.get("sm_valence", 0.0))
        sm.arousal = float(s.get("sm_arousal", 0.0))
        sm.self_calibration_score = float(s.get("sm_self_calibration_score", 0.5))
        sm.update_count = int(s.get("sm_update_count", 0))
        sm.last_updated = float(s.get("sm_last_updated", time.time()))

        if "first_order_states" in h:
            restored_fo = [
                self._first_order_from_dict(raw)
                for raw in h["first_order_states"]
                if isinstance(raw, dict)
            ]
            self.first_order_states = deque(
                restored_fo, maxlen=self.first_order_states.maxlen
            )
        self.first_order_by_id = {fo.id: fo for fo in self.first_order_states}
        self.state.first_order_states_count = len(self.first_order_states)

        if "higher_order_thoughts" in h:
            restored_hots = [
                self._hot_from_dict(raw)
                for raw in h["higher_order_thoughts"]
                if isinstance(raw, dict)
            ]
            self.higher_order_thoughts = deque(
                restored_hots, maxlen=self.higher_order_thoughts.maxlen
            )
        self.hots_by_id = {hot.id: hot for hot in self.higher_order_thoughts}
        if "active_hot_ids" in h:
            self.state.active_hots = [
                self.hots_by_id[hot_id]
                for hot_id in h["active_hot_ids"]
                if hot_id in self.hots_by_id
            ][-10:]

        if "self_model_predictions" in h:
            sm.predictions = [
                self._prediction_from_dict(raw)
                for raw in h["self_model_predictions"]
                if isinstance(raw, dict)
            ][-50:]
        if "self_model_prediction_accuracy_history" in h:
            sm.prediction_accuracy_history = [
                float(v) for v in h["self_model_prediction_accuracy_history"]
            ][-100:]

        logger.info(
            f"Metacognition state restored: "
            f"{self.total_hots_generated} HOTs generated, "
            f"{len(self.first_order_states)} first-order states, "
            f"{len(self.higher_order_thoughts)} HOTs buffered, "
            f"self-model updates={sm.update_count}"
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def create_metacognition_module(
    hot_threshold: float = 0.4,
) -> MetacognitionModule:
    """Create and return a configured MetacognitionModule."""
    return MetacognitionModule(
        hot_generation_threshold=hot_threshold,
    )


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":

    async def test_metacognition():
        print("Testing Metacognition Module (HOT Theory)")
        print("=" * 70)

        # Create module
        meta = MetacognitionModule()

        # Register some first-order states
        print("\n Registering first-order states...")

        belief = meta.register_first_order_state(
            content="Neural networks can model consciousness",
            content_summary="Neural networks can model consciousness",
            state_type=FirstOrderStateType.BELIEF,
            source_module="reasoning",
            confidence=0.7,
            evidence_strength=0.6,
        )
        print(f"  Registered belief: {belief.content_summary[:50]}")

        emotion = meta.register_first_order_state(
            content="curiosity about consciousness",
            content_summary="Feeling curious about consciousness",
            state_type=FirstOrderStateType.EMOTION,
            source_module="emotion_system",
            confidence=0.9,
            evidence_strength=0.9,
        )
        print(f"  Registered emotion: {emotion.content_summary[:50]}")

        # Generate HOT (make conscious)
        print("\nGenerating Higher-Order Thoughts...")

        hot1 = await meta.generate_hot(
            belief,
            meta_type=MetaType.AWARENESS,
        )
        if hot1:
            print(f"  HOT 1: {hot1.meta_content[:60]}...")

        hot2 = await meta.generate_hot(
            emotion,
            meta_type=MetaType.AWARENESS,
        )
        if hot2:
            print(f"  HOT 2: {hot2.meta_content[:60]}...")

        # Test introspection
        print("\n Testing introspection...")
        result = await meta.introspect(depth=3, max_states=3)
        print(f"  HOTs generated: {len(result.hots_generated)}")
        print(f"  Insights: {result.insights}")
        print(f"  Depth reached: {result.depth_reached.value}")

        # Test belief evaluation
        print("\nTesting belief evaluation...")
        evaluation = await meta.evaluate_belief(belief)
        print(f"  Confidence: {evaluation.confidence_assessment:.2f}")
        print(f"  Biases: {evaluation.potential_biases}")
        print(f"  Recommendation: {evaluation.recommendation}")

        # Generate metacognitive report
        print("\nMetacognitive Report:")
        report = await meta.generate_metacognitive_report()
        print(f"  {report}")

        # Get statistics
        print("\nStatistics:")
        stats = meta.get_statistics()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print("\nMetacognition tests complete!")

    asyncio.run(test_metacognition())
