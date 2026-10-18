"""
Enhanced Global Workspace - Full GWT Implementation
=========================================================

Global Workspace Theory implementation with competition, broadcast,
and ignition dynamics.

This module implements a competition-based conscious access system.

Key Features:
1. AttentionBottleneck - Limited capacity (7 plus or minus 2 items) with
   flat or hierarchical competition behind an optional thalamic gate
2. GlobalBroadcast - Parallel distribution to ALL cognitive modules
3. IgnitionDetector - Non-linear "ignition" when threshold crossed
4. WorkspaceCandidate - Content competing for conscious access
5. StreamOfConsciousness - Temporal continuity tracking
6. Cross-theory modulation - the theory interaction manager's somatic and
   Phi signals bias the next cycle's competition

Based on: Bernard Baars' Global Workspace Theory (1988, 2005)
Research: Butlin et al. (2023) - 14 Consciousness Indicators

Only the most salient signals win the competition for workspace access;
the information that wins is broadcast globally to every registered
cognitive module.
"""

import asyncio
import hashlib
import logging
import math
import numpy as np
import time
import uuid
from typing import Callable, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import Attention Schema (AST implementation)
from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionSchemaState,
    AttentionTarget,
)

# Import Metacognition (HOT implementation)
from mtc.consciousness.metacognition import (
    MetacognitionModule,
    MetacognitiveState,
    FirstOrderStateType,
    HigherOrderThought,
    MetaType,
)

# Import Active Inference (FEP implementation)
from mtc.consciousness.active_inference import (
    HIERARCHY_OBSERVATION_HPP_TOP_LEVEL,
    WORLD_MODEL_COMPACT,
    WORLD_MODEL_CONVERSATIONAL,
    ActiveInferenceModule,
    ActiveInferenceConfig,
    ActiveInferenceState,
    InferenceResult,
    ConsciousPosteriorMapping,
)

# Import Beautiful Loop (Laukkonen, Friston & Chandaria 2025)
from mtc.consciousness.beautiful_loop import (
    BeautifulLoop,
    ConsciousMoment,
)

# Import Damasio Three-Layer Model (Damasio 1999, 2010)
from mtc.consciousness.damasio import DamasioLayers
from mtc.consciousness.damasio.protoself import gather_system_body_metrics

# Pre-competition salience filter and cross-theory modulation
from mtc.consciousness.thalamic_gate import ThalamicGate, ThalamicGateConfig
from mtc.consciousness.theory_interaction import (
    TheoryInteractionConfig,
    TheoryInteractionManager,
)

logger = logging.getLogger(__name__)

# Competition modes of the attention bottleneck
COMPETITION_HIERARCHICAL = "hierarchical"
COMPETITION_FLAT = "flat"  # the v0.4.0 single-pool competition
COMPETITION_MODES = (COMPETITION_HIERARCHICAL, COMPETITION_FLAT)

# Where the thalamic gate's arousal came from in a cycle
AROUSAL_SOURCE_CALLER = "caller"
AROUSAL_SOURCE_EMOTIONAL_STATE = "emotional_state"
AROUSAL_SOURCE_PAD = "pad"
AROUSAL_SOURCE_DEFAULT = "default"
NEUTRAL_AROUSAL = 0.5

# The geometry of the workspace's own active inference model when no
# config is passed (the compact world model's dimensions)
_WORKSPACE_FEP_GEOMETRY: Dict[str, int] = {
    "num_hidden_states": 8,
    "num_observations": 5,
    "num_actions": 4,
    "planning_horizon": 5,
}

# Construction defaults of the workspace-owned Beautiful Loop
_WORKSPACE_BEAUTIFUL_LOOP_DEFAULTS: Dict[str, Any] = {
    "num_levels": 3,  # Matches the hierarchical processor's three levels
    "precision_learning_rate": 0.1,
    "min_binding_quality": 0.2,
    "max_epistemic_depth": 5,
    "field_evidencing_threshold": 0.4,
}


# ============================================================================
# TEMPORAL DYNAMICS CONFIGURATION
# ============================================================================


# How the refractory period's simulated clock advances between cycles
REFRACTORY_CLOCK_PER_CYCLE = "per_cycle"  # competition_duration_ms per cycle
REFRACTORY_CLOCK_ELAPSED = "elapsed"  # plus the real time between cycles
REFRACTORY_CLOCKS = (REFRACTORY_CLOCK_PER_CYCLE, REFRACTORY_CLOCK_ELAPSED)


@dataclass
class TemporalWorkspaceConfig:
    """
    Configuration for biologically-plausible temporal dynamics in the
    Global Workspace.

    Biological ignition takes 200-300ms of sustained activation before
    content enters consciousness. This config governs the simulated
    timestep competition that replaces the old instant-threshold model.

    All timing is in SIMULATED milliseconds -- no real wall-clock delay
    is introduced. The competition runs as a tight numerical loop.

    ``refractory_clock`` says how the simulated clock advances between
    cycles: ``per_cycle`` (the default) by ``competition_duration_ms`` per
    cycle whatever the real time, which keeps a seeded run machine
    independent; ``elapsed`` also by the real time between cycles read from
    the workspace's clock, so cycles further apart than the refractory
    period are never refractory (a caller that drives the workspace in
    real time, such as the conversation runtime, sets it).
    """

    competition_duration_ms: float = 250.0  # 200-300ms biological range
    ignition_threshold_time_ms: float = 50.0  # Time above threshold to ignite
    broadcast_propagation_ms: float = 30.0  # Per-module delivery delay (unused for now)
    refractory_period_ms: float = 500.0  # Min time between broadcasts
    timestep_ms: float = 10.0  # Simulation resolution
    refractory_clock: str = REFRACTORY_CLOCK_PER_CYCLE

    def __post_init__(self) -> None:
        if self.refractory_clock not in REFRACTORY_CLOCKS:
            raise ValueError(
                f"refractory_clock must be one of {REFRACTORY_CLOCKS}, "
                f"got {self.refractory_clock!r}"
            )


# ============================================================================
# DATA CLASSES - The building blocks of conscious content
# ============================================================================


class WorkspaceCandidateSource(Enum):
    """Sources from which candidates can enter the workspace."""

    SNN = "snn"  # Spiking Neural Network (conscious processing)
    LSM = "lsm"  # Liquid State Machine (subconscious)
    HTM = "htm"  # Hierarchical Temporal Memory (consolidation)
    CTM = "ctm"  # Continuous Thought Machine (background thoughts)
    SENSORY = "sensory"  # External sensory input
    MEMORY = "memory"  # Retrieved memories
    LLM = "llm"  # Language model reasoning
    VOLUNTARY = "voluntary"  # Voluntary attention shift
    SAFETY = "safety"  # Safety monitor alerts (HIGH PRIORITY)
    CONCEPT = "concept"  # Knowledge graph concept
    GIST = "gist"  # Consolidated thought pattern
    HOMEOSTATIC = "homeostatic"  # Urgent homeostatic drive
    INTERNAL = "internal"  # Internal processes (simulation, self-assessment)


# Hierarchical workspace levels (the global neuronal workspace as a
# multilevel model, Trends in Cognitive Sciences 2026):
#   0 perceptual: sensory processing, fast, high volume
#   1 integrative: memory binding, concept association, pattern matching
#   2 executive: reasoning, planning, metacognition, self-directed thought
HIERARCHY_LEVELS = {
    "perceptual": 0,
    "integrative": 1,
    "executive": 2,
}

# Default hierarchy level of each candidate source
SOURCE_LEVEL_MAP = {
    WorkspaceCandidateSource.SENSORY: 0,
    WorkspaceCandidateSource.SNN: 0,  # Neural firing is perceptual
    WorkspaceCandidateSource.LSM: 0,  # Subconscious dynamics are perceptual
    WorkspaceCandidateSource.MEMORY: 1,
    WorkspaceCandidateSource.CONCEPT: 1,
    WorkspaceCandidateSource.GIST: 1,
    WorkspaceCandidateSource.HTM: 1,  # Temporal memory is integrative
    WorkspaceCandidateSource.CTM: 2,  # Background thought is executive
    WorkspaceCandidateSource.LLM: 2,  # Reasoning is executive
    WorkspaceCandidateSource.VOLUNTARY: 2,  # Self-directed is executive
    WorkspaceCandidateSource.HOMEOSTATIC: 2,  # Drive-based is executive
    WorkspaceCandidateSource.INTERNAL: 2,
    WorkspaceCandidateSource.SAFETY: 0,  # Safety bypasses the hierarchy
}


@dataclass
class WorkspaceCandidate:
    """
    A candidate competing for access to the global workspace.

    Only candidates that win competition enter conscious awareness.
    This implements the "bottleneck" of consciousness.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Content
    content: Any = None  # The actual content (embedding, text, etc.)
    content_type: str = ""  # Type of content (emotion, thought, memory, etc.)
    summary: str = ""  # Human-readable summary

    # Source information
    source: WorkspaceCandidateSource = WorkspaceCandidateSource.SENSORY
    source_module: str = ""  # Specific module name

    # Salience factors (used for competition)
    activation_level: float = 0.5  # Raw activation strength (0-1)
    emotional_salience: float = 0.0  # Emotional weight (0-1)
    novelty_score: float = 0.0  # How novel/surprising (0-1)
    goal_relevance: float = 0.0  # Relevance to current goals (0-1)
    task_relevance: float = 0.0  # Relevance to current task (0-1)
    recency_bonus: float = 0.0  # Temporal proximity bonus (0-1)

    # Priority override (for safety-critical content)
    priority_boost: float = 0.0  # Manual priority boost

    # Hierarchy level: 0 perceptual, 1 integrative, 2 executive; -1 means
    # assign from SOURCE_LEVEL_MAP
    hierarchy_level: int = -1

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Assign the hierarchy level from the source when not set."""
        if self.hierarchy_level < 0:
            self.hierarchy_level = SOURCE_LEVEL_MAP.get(self.source, 1)


@dataclass
class WorkspaceContent:
    """
    Content that has WON competition and entered conscious awareness.

    This represents what the system is actively "aware of" - content that
    has been broadcast to all cognitive modules.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # The winning candidate
    candidate: WorkspaceCandidate = None

    # Competition results
    salience: float = 0.0  # Final computed salience
    competition_rank: int = 0  # Rank in competition (1 = winner)
    competing_count: int = 0  # How many candidates competed
    # Hierarchy level the content was finally admitted at, after promotion
    # (None outside hierarchical competition)
    admitted_level: Optional[int] = None

    # Broadcast tracking
    entry_time: float = field(default_factory=time.time)
    broadcast_complete: bool = False
    broadcast_recipients: List[str] = field(default_factory=list)

    # Integration state
    integrated_representation: Optional[np.ndarray] = None
    module_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IgnitionEvent:
    """
    Records when content "ignites" into consciousness.

    Ignition is the non-linear, sudden amplification that occurs
    when content crosses the threshold - the moment of becoming conscious.
    """

    # None only for an event restored without its content record
    content: Optional[WorkspaceContent]

    # Pre/post ignition state
    pre_ignition_salience: float
    post_ignition_salience: float
    amplification_factor: float

    # Timing
    ignition_time: float
    expected_duration: float

    # Dynamics
    was_sustained: bool = True  # Did it maintain activation?
    triggered_broadcast: bool = True
    # True once check_sustain has measured this event; was_sustained is a
    # measurement only then
    sustain_checked: bool = False


@dataclass
class CompetitionEvent:
    """
    Records a single competition cycle for analysis.
    """

    timestamp: float
    candidates_count: int
    winners_count: int
    threshold_used: float
    winning_sources: List[str]
    top_salience: float
    avg_salience: float
    # Hierarchical competition details (level counts, promotions, safety)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BroadcastEvent:
    """
    Records a broadcast event for analysis.
    """

    content_id: str
    content_summary: str
    recipients: int
    successful: int
    failed: int
    latency_ms: float
    timestamp: float


@dataclass
class ConsciousnessState:
    """
    The overall state of conscious processing at a moment in time.

    Includes:
    - GWT: Competition, ignition, broadcast
    - AST: Self-model of attention
    - HOT: Higher-order thoughts (metacognition)
    - FEP: Active inference and predictive processing
    """

    is_conscious: bool  # Is there active conscious content?
    primary_content: Optional[WorkspaceContent]  # Main focus
    workspace_contents: List[WorkspaceContent]  # All in workspace

    # Metrics
    ignition_events: int  # Count of ignitions this cycle
    # Mean absolute correlation across the neural signals (0-1); None when
    # it was not measured (fewer than two usable signals this cycle)
    integration_level: Optional[float]
    # Share of module deliveries that succeeded (0-1); None when nothing
    # could have received (no receiver registered and nothing delivered)
    broadcast_coverage: Optional[float]

    # Attention state (GWT)
    attention_focus: str  # What source is dominating
    attention_distribution: Dict[str, float]  # Weight per source

    # Stream tracking
    stream_position: int  # Position in stream of consciousness

    # Fields with defaults must come after non-defaults (Python dataclass rules)
    # Attention Schema (AST) - the system's self-model of attention
    attention_schema: Optional[AttentionSchemaState] = None  # Self-model of attention
    attention_report: Optional[str] = (
        None  # What the system reports about its attention
    )

    # Metacognition (HOT) - Higher-order thoughts
    metacognitive_state: Optional[MetacognitiveState] = None  # Meta-awareness state
    higher_order_thoughts: List[HigherOrderThought] = field(
        default_factory=list
    )  # Active HOTs
    metacognitive_report: Optional[str] = (
        None  # What the system reports about its thinking
    )

    # Active Inference (FEP) - Predictive processing
    active_inference_state: Optional[ActiveInferenceState] = None  # FEP state
    inference_result: Optional[InferenceResult] = None  # Latest inference
    prediction_error: float = 0.0  # Free energy / surprise
    # Percentile rank of prediction_error in the module's own recent errors;
    # None until a baseline exists, so rank-gated readers fail closed
    prediction_error_rank: Optional[float] = None
    variational_free_energy: float = 0.0  # VFE from active inference
    homeostatic_urgency: Optional[Tuple[str, float]] = None  # Most urgent need
    active_inference_report: Optional[str] = (
        None  # What the system reports about predictions
    )

    # Recursive Self-Model - Explicit self-representation
    self_model_report: Optional[str] = None  # What the system knows about itself

    # Posterior Mapping (P2) - Contents = posterior beliefs
    posterior_mappings: List[ConsciousPosteriorMapping] = field(default_factory=list)

    # Discrete-Continuous Interface (P3) - Hierarchical prediction errors
    hierarchical_prediction_errors: List[float] = field(default_factory=list)

    # Beautiful Loop - Laukkonen, Friston & Chandaria 2025
    beautiful_loop_moment: Optional[ConsciousMoment] = None
    beautiful_loop_quality: float = 0.0
    epistemic_depth: int = 0
    # None when the loop bound nothing this cycle (winners without a vector)
    binding_quality: Optional[float] = 0.0
    is_field_evidencing: bool = False
    beautiful_loop_report: Optional[str] = None

    # Damasio Three-Layer Model -- protoself, core, extended. The three
    # Optional fields are None when the layer measured nothing: the first
    # protoself update has no previous body state (stability and feeling of
    # knowing), and the extended layer may report no identity continuity.
    damasio_state: Optional[Dict[str, Any]] = None
    protoself_stability: Optional[float] = 0.0
    self_world_binding: float = 0.0
    feeling_of_knowing: Optional[float] = 0.0
    autobiographic_continuity: Optional[float] = 0.0
    narrative_context: str = ""
    damasio_report: Optional[str] = None

    # The slots below stay None until a producer fills them.

    # Hierarchical competition (filled in hierarchical mode)
    # Both describe the level each winner was finally admitted at (after
    # promotion), not the level of its source
    dominant_hierarchy_level: Optional[int] = None  # Admitted level of the top winner
    level_winner_counts: Optional[Dict[int, int]] = None
    cross_level_promotions: Optional[int] = None

    # Thalamic gate (filled when the gate is enabled)
    thalamic_gate_rate: Optional[float] = None  # Share filtered this cycle
    thalamic_gated_count: Optional[int] = None  # Candidates filtered this cycle
    # Where the gate's arousal came from this cycle: "caller", "emotional_state",
    # "pad" or "default"; None when the gate is off
    arousal_source: Optional[str] = None

    # IIT: integrated information from an attached PhiTracker
    architectural_phi: Optional[float] = None
    phi_warming_up: Optional[bool] = None

    # RPT: recurrence metrics
    rpt_deep_recurrence: Optional[float] = None
    rpt_superficial_recurrence: Optional[float] = None

    # Perceptual reality monitoring: external versus internal discrimination
    prm_discrimination: Optional[float] = None

    # Causal emergence: macro versus micro causal power
    causal_emergence: Optional[float] = None
    causal_emergence_warming_up: Optional[bool] = None

    # PAD affect coordinate
    pad_pleasure: Optional[float] = None
    pad_arousal: Optional[float] = None
    pad_dominance: Optional[float] = None
    pad_label: Optional[str] = None

    timestamp: float = field(default_factory=time.time)


# ============================================================================
# COGNITIVE MODULE INTERFACE - What modules must implement
# ============================================================================


class CognitiveModule:
    """
    Interface for modules that receive global broadcasts.

    All cognitive subsystems (SNN, LSM, HTM, etc.) should implement
    this interface to receive conscious content broadcasts.
    """

    def __init__(self, name: str):
        self.name = name
        self.last_broadcast: Optional[WorkspaceContent] = None

    async def receive_broadcast(self, content: WorkspaceContent) -> Dict[str, Any]:
        """
        Receive broadcast content from the global workspace.

        Args:
            content: The conscious content being broadcast

        Returns:
            Module's response to the content
        """
        self.last_broadcast = content
        content_type = (
            content.candidate.content_type if content.candidate else "unknown"
        )
        logger.debug(
            "Module '%s' received broadcast (content_type=%s, salience=%.3f)",
            self.name,
            content_type,
            content.salience,
        )
        return {"received": True, "module": self.name, "content_type": content_type}

    async def submit_candidate(
        self,
        content: Any,
        content_type: str,
        summary: str,
        activation_level: float,
        emotional_salience: float = 0.0,
        metadata: Dict[str, Any] = None,
    ) -> WorkspaceCandidate:
        """
        Create a candidate for submission to the workspace.

        Args:
            content: The actual content
            content_type: Type identifier
            summary: Human-readable summary
            activation_level: How strongly activated (0-1)
            emotional_salience: Emotional weight (0-1)
            metadata: Additional metadata

        Returns:
            WorkspaceCandidate ready for competition
        """
        return WorkspaceCandidate(
            content=content,
            content_type=content_type,
            summary=summary,
            source_module=self.name,
            activation_level=activation_level,
            emotional_salience=emotional_salience,
            metadata=metadata or {},
        )


# ============================================================================
# ATTENTION BOTTLENECK - The competition mechanism
# ============================================================================


class AttentionBottleneck:
    """
    The critical competition mechanism for conscious access.

    Only the most salient information gains entry to the workspace.
    This implements the LIMITED CAPACITY of consciousness -
    you can't be conscious of everything at once!

    Based on Miller's Law (7 plus or minus 2) for working memory capacity.

    Competition factors:
    1. Bottom-up salience (stimulus strength, novelty) - 40%
    2. Top-down relevance (goal alignment, task relevance) - 30%
    3. Emotional weight (survival/value significance) - 20%
    4. Recency (temporal proximity bonus) - 10%
    """

    # Per-level capacity of the hierarchical competition (3 + 3 + 2 = 8)
    LEVEL_CAPACITY = {0: 3, 1: 3, 2: 2}
    # Salience a level winner needs to also compete one level up
    PROMOTION_THRESHOLD = 0.7
    # Salience multiplier a promoted winner carries to the next level
    PROMOTION_PENALTY = 0.9

    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        ignition_threshold: float = 0.3,  # Minimum to enter consciousness
        lateral_inhibition: float = 0.2,  # How much winners suppress similar content
        thalamic_gate: Optional[ThalamicGate] = None,
    ):
        """
        Initialize the attention bottleneck.

        Args:
            capacity: Maximum items in workspace (default: 7)
            ignition_threshold: Minimum salience to enter consciousness
            lateral_inhibition: Strength of winner-takes-all suppression
            thalamic_gate: Optional pre-competition filter; None lets every
                candidate compete
        """
        self.capacity = capacity
        self.threshold = ignition_threshold
        self.lateral_inhibition = lateral_inhibition

        # Cross-cycle FEP feedback: prediction error from prior cycle
        # modulates candidate salience (FEP to GWT interaction)
        self._prior_prediction_error: float = 0.0

        # Cross-theory modulation bias (somatic plus Phi), set after each
        # cycle from the theory interaction manager
        self._cross_theory_bias: float = 0.0

        # Current contents (what's in the workspace)
        self.current_contents: List[WorkspaceContent] = []

        # Competition history (for research analysis)
        self.competition_history: List[CompetitionEvent] = []
        self.max_history = 1000
        # The event recorded by the latest competition call (None when that
        # call recorded nothing: no candidates, or all gated)
        self.last_competition: Optional[CompetitionEvent] = None

        # Current goals and task (for top-down modulation)
        self.current_goals: List[str] = []
        self.current_task: Optional[str] = None

        # Novelty tracking (stable digests of recently seen content)
        self._recent_content_hashes: List[str] = []
        self._max_recent = 100

        # Thalamic gate and the arousal it reads
        self.thalamic_gate: Optional[ThalamicGate] = thalamic_gate
        self._arousal: float = 0.5
        # True once a caller supplied arousal through set_arousal; the
        # workspace then stops deriving it from the emotional state
        self._arousal_supplied: bool = False

        logger.info(
            f"Attention Bottleneck initialized "
            f"(capacity={capacity}, threshold={ignition_threshold}, "
            f"gate={'on' if thalamic_gate is not None else 'off'})"
        )

    # ------------------------------------------------------------------
    # Arousal, gate and cross-theory bias
    # ------------------------------------------------------------------

    @property
    def arousal(self) -> float:
        """Arousal level (0 to 1) the thalamic gate reads."""
        return self._arousal

    @property
    def arousal_supplied(self) -> bool:
        """Whether a caller has supplied arousal through set_arousal."""
        return self._arousal_supplied

    def set_arousal(self, arousal: float) -> None:
        """Supply the arousal level used by the thalamic gate (clamped to 0 to 1).

        A non-finite value is ignored. Once called, the caller owns arousal.
        """
        if self._apply_arousal(arousal):
            self._arousal_supplied = True

    def _apply_arousal(self, arousal: Any) -> bool:
        value = _finite_float(arousal)
        if value is None:
            return False
        self._arousal = max(0.0, min(1.0, value))
        return True

    @property
    def cross_theory_bias(self) -> float:
        """Cross-theory salience bias, multiplied by novelty in the salience."""
        return self._cross_theory_bias

    @cross_theory_bias.setter
    def cross_theory_bias(self, value: float) -> None:
        number = _finite_float(value)
        self._cross_theory_bias = 0.0 if number is None else max(-1.0, min(1.0, number))

    def _apply_gate(
        self, candidates: List[WorkspaceCandidate]
    ) -> List[WorkspaceCandidate]:
        """Candidates that pass the thalamic gate (all of them without one)."""
        if self.thalamic_gate is None:
            return candidates
        return self.thalamic_gate.gate(candidates, self._arousal, self.current_goals)

    # ------------------------------------------------------------------
    # Competition
    # ------------------------------------------------------------------

    async def compete_for_access(
        self, candidates: List[WorkspaceCandidate]
    ) -> List[WorkspaceContent]:
        """
        Run the flat competition among candidates for workspace access.

        This is the CORE of Global Workspace Theory - content must
        COMPETE to become conscious! All candidates compete in one pool for
        the bottleneck's capacity (the v0.4.0 competition).

        Args:
            candidates: All candidates vying for conscious access

        Returns:
            List of winners (WorkspaceContent) that gained access
        """
        self.last_competition = None
        if not candidates:
            return []

        start_time = time.time()

        candidates = self._apply_gate(candidates)
        if not candidates:
            return []

        # Calculate composite salience for each candidate
        scored_candidates = []
        for candidate in candidates:
            salience = await self._calculate_composite_salience(candidate)
            scored_candidates.append((candidate, salience))

        # Sort by salience (descending) - highest salience wins!
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold and capacity limit
        winners: List[WorkspaceContent] = []
        for candidate, salience in scored_candidates:
            # Stop if we've reached capacity
            if len(winners) >= self.capacity:
                break

            # Stop if below threshold
            if salience < self.threshold:
                break

            # Create WorkspaceContent for winner
            content = WorkspaceContent(
                candidate=candidate,
                salience=salience,
                competition_rank=len(winners) + 1,
                competing_count=len(candidates),
                entry_time=time.time(),
            )
            winners.append(content)

        # Apply lateral inhibition - winners suppress similar content
        if self.lateral_inhibition > 0 and len(winners) > 1:
            await self._apply_lateral_inhibition(winners)

        # Update current contents
        self.current_contents = winners

        # Track competition event for research
        event = CompetitionEvent(
            timestamp=time.time(),
            candidates_count=len(candidates),
            winners_count=len(winners),
            threshold_used=self.threshold,
            winning_sources=[w.candidate.source.value for w in winners],
            top_salience=winners[0].salience if winners else 0.0,
            avg_salience=(
                np.mean([s for _, s in scored_candidates]) if scored_candidates else 0.0
            ),
        )
        self._add_to_history(event)
        self.last_competition = event

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Competition complete: {len(winners)}/{len(candidates)} won "
            f"(top salience: {event.top_salience:.3f}, {elapsed_ms:.1f}ms)"
        )

        return winners

    async def compete_hierarchical(
        self, candidates: List[WorkspaceCandidate]
    ) -> List[WorkspaceContent]:
        """Run the multilevel competition (the workspace as a hierarchy).

        Simple perception ignites at the perceptual level; deep reasoning
        competes at the executive level.

        1. Partition candidates by hierarchy level.
        2. Within-level competition, each level with its own capacity
           (LEVEL_CAPACITY).
        3. Winners at or above PROMOTION_THRESHOLD also compete one level up
           at PROMOTION_PENALTY times their salience, while that level has
           room.
        4. Merge executive, integrative, then perceptual winners up to the
           bottleneck capacity. A promoted candidate keeps only its highest
           slot, so it occupies one final slot.

        Safety content bypasses the hierarchy and always enters.
        """
        self.last_competition = None
        if not candidates:
            return []

        start_time = time.time()

        candidates = self._apply_gate(candidates)
        if not candidates:
            return []

        levels: Dict[int, List[WorkspaceCandidate]] = {
            level: [] for level in self.LEVEL_CAPACITY
        }
        safety_candidates: List[WorkspaceCandidate] = []
        for candidate in candidates:
            if candidate.source == WorkspaceCandidateSource.SAFETY:
                safety_candidates.append(candidate)
                continue
            level = candidate.hierarchy_level
            if level not in levels:
                level = HIERARCHY_LEVELS["integrative"]
            levels[level].append(candidate)

        # Every scored salience, for the event's average (as in the flat mode)
        all_saliences: List[float] = []

        # Step 1: within-level competition
        level_winners: Dict[int, List[WorkspaceContent]] = {}
        for level_idx in sorted(levels):
            level_candidates = levels[level_idx]
            scored = []
            for candidate in level_candidates:
                salience = await self._calculate_composite_salience(candidate)
                scored.append((candidate, salience))
                all_saliences.append(salience)
            scored.sort(key=lambda x: x[1], reverse=True)

            capacity = self.LEVEL_CAPACITY[level_idx]
            winners: List[WorkspaceContent] = []
            for candidate, salience in scored:
                if len(winners) >= capacity or salience < self.threshold:
                    break
                winners.append(
                    WorkspaceContent(
                        candidate=candidate,
                        salience=salience,
                        competition_rank=len(winners) + 1,
                        competing_count=len(level_candidates),
                        entry_time=time.time(),
                    )
                )
            level_winners[level_idx] = winners

        # Step 2: cross-level promotion (perceptual to integrative,
        # integrative to executive; a promoted winner can climb twice)
        promotions = 0
        ordered_levels = sorted(levels)
        for level_idx, next_level in zip(ordered_levels, ordered_levels[1:]):
            for winner in level_winners[level_idx]:
                if winner.salience < self.PROMOTION_THRESHOLD:
                    continue
                promoted_salience = winner.salience * self.PROMOTION_PENALTY
                next_winners = level_winners[next_level]
                if promoted_salience < self.threshold or len(next_winners) >= (
                    self.LEVEL_CAPACITY[next_level]
                ):
                    continue
                next_winners.append(
                    WorkspaceContent(
                        candidate=winner.candidate,
                        salience=promoted_salience,
                        competition_rank=len(next_winners) + 1,
                        competing_count=len(levels[next_level]) + 1,
                        entry_time=time.time(),
                    )
                )
                promotions += 1

        # Step 3: safety content always enters
        safety_winners: List[WorkspaceContent] = []
        for candidate in safety_candidates:
            salience = await self._calculate_composite_salience(candidate)
            all_saliences.append(salience)
            safety_winners.append(
                WorkspaceContent(
                    candidate=candidate,
                    salience=salience,
                    competition_rank=0,
                    competing_count=len(candidates),
                    admitted_level=(
                        candidate.hierarchy_level
                        if candidate.hierarchy_level in levels
                        else HIERARCHY_LEVELS["integrative"]
                    ),
                    entry_time=time.time(),
                )
            )

        # Merge: safety first, then executive, integrative, perceptual.
        # A promoted candidate appears at several levels; it takes the merge
        # position of its highest slot but keeps its highest-salience copy
        # (the unpenalized original), so promotion never lowers a winner's
        # final salience.
        best_copy: Dict[str, WorkspaceContent] = {}
        for winners in level_winners.values():
            for winner in winners:
                key = winner.candidate.id
                if key not in best_copy or winner.salience > best_copy[key].salience:
                    best_copy[key] = winner
        final_winners = list(safety_winners)
        seen_ids = {w.candidate.id for w in final_winners}
        for level_idx in sorted(levels, reverse=True):
            for winner in level_winners[level_idx]:
                if len(final_winners) >= self.capacity:
                    break
                if winner.candidate.id in seen_ids:
                    continue
                admitted = best_copy[winner.candidate.id]
                # Merging goes from the highest level down, so the first slot
                # a candidate takes is its highest: the level it was admitted at
                admitted.admitted_level = level_idx
                final_winners.append(admitted)
                seen_ids.add(winner.candidate.id)

        # Re-rank by salience
        final_winners.sort(key=lambda w: w.salience, reverse=True)
        for rank, winner in enumerate(final_winners, start=1):
            winner.competition_rank = rank

        if self.lateral_inhibition > 0 and len(final_winners) > 1:
            await self._apply_lateral_inhibition(final_winners)

        self.current_contents = final_winners

        event = CompetitionEvent(
            timestamp=time.time(),
            candidates_count=len(candidates),
            winners_count=len(final_winners),
            threshold_used=self.threshold,
            winning_sources=[w.candidate.source.value for w in final_winners],
            top_salience=final_winners[0].salience if final_winners else 0.0,
            avg_salience=float(np.mean(all_saliences)) if all_saliences else 0.0,
            metadata={
                "hierarchical": True,
                "level_candidates": {k: len(v) for k, v in levels.items()},
                "level_winners": {k: len(v) for k, v in level_winners.items()},
                "promotions": promotions,
                "safety_bypass": len(safety_winners),
            },
        )
        self._add_to_history(event)
        self.last_competition = event

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            "Hierarchical competition: %d/%d won (L0=%d, L1=%d, L2=%d, "
            "promoted=%d, %.1fms)",
            len(final_winners),
            len(candidates),
            len(level_winners[0]),
            len(level_winners[1]),
            len(level_winners[2]),
            promotions,
            elapsed_ms,
        )
        return final_winners

    async def _calculate_composite_salience(
        self, candidate: WorkspaceCandidate
    ) -> float:
        """
        Multi-factor salience calculation.

        This determines WHO WINS the competition for consciousness!

        Weights:
        - Bottom-up (stimulus + novelty): 40%
        - Top-down (goal + task relevance): 30%
        - Emotional: 20%
        - Temporal: 10%
        """
        # BOTTOM-UP FACTORS (40%)
        # Raw stimulus strength
        stimulus_strength = candidate.activation_level

        # Novelty - how surprising/new is this?
        novelty = await self._calculate_novelty(candidate)
        candidate.novelty_score = novelty  # Store for later use

        bottom_up = 0.4 * (0.6 * stimulus_strength + 0.4 * novelty)

        # TOP-DOWN FACTORS (30%)
        # Goal and task relevance. A measured keyword match is recorded on
        # the candidate; the neutral competition constant used when the
        # workspace has no goals or task is not, so the candidate keeps the
        # relevance its producer supplied (which the thalamic gate reads).
        goal_relevance = await self._calculate_goal_relevance(candidate)
        if self.current_goals:
            candidate.goal_relevance = goal_relevance

        task_relevance = await self._calculate_task_relevance(candidate)
        if self.current_task:
            candidate.task_relevance = task_relevance

        top_down = 0.3 * (0.5 * goal_relevance + 0.5 * task_relevance)

        # EMOTIONAL FACTORS (20%)
        emotional = 0.2 * candidate.emotional_salience

        # TEMPORAL FACTORS (10%)
        recency = self._calculate_recency_bonus(candidate)
        candidate.recency_bonus = recency
        temporal = 0.1 * recency

        # FEP FEEDBACK (cross-cycle interaction)
        # The prior cycle's prediction error boosts each candidate in
        # proportion to its own novelty: after a surprising cycle, novel
        # content gains salience and familiar content gains little. The
        # boost is at most 0.1 times the prior error, so it influences the
        # competition without dominating it.
        fep_boost = 0.0
        if self._prior_prediction_error > 0:
            fep_boost = 0.1 * self._prior_prediction_error * novelty

        # CROSS-THEORY MODULATION (somatic plus Phi, set after the prior
        # cycle): an unstable body or higher Phi favours novel content
        cross_theory = self._cross_theory_bias * novelty

        # PRIORITY BOOST (additive)
        # Safety-critical content gets priority!
        priority = candidate.priority_boost
        if candidate.source == WorkspaceCandidateSource.SAFETY:
            priority += 0.5  # Safety always gets attention!

        # Compute final salience
        salience = (
            bottom_up
            + top_down
            + emotional
            + temporal
            + fep_boost
            + cross_theory
            + priority
        )

        # Clamp to 0-1 (priority boost can exceed 1.0 intentionally)
        salience = max(0.0, min(1.5, salience))  # Allow up to 1.5 for safety

        return salience

    async def _calculate_novelty(self, candidate: WorkspaceCandidate) -> float:
        """
        Calculate how novel/surprising this content is.

        Novel content is more salient - evolution favored noticing new things!
        """
        # Hash the content for comparison (handle numpy arrays specially)
        content = candidate.content
        if content is None:
            content_str = ""
        elif isinstance(content, np.ndarray):
            content_str = str(content.tobytes()[:100])  # Hash bytes for arrays
        else:
            content_str = str(content)[:100]
        content_hash = _stable_digest(content_str)

        # Check if we've seen this recently
        if content_hash in self._recent_content_hashes:
            # Seen recently - low novelty
            return 0.2

        # Add to recent hashes
        self._recent_content_hashes.append(content_hash)
        if len(self._recent_content_hashes) > self._max_recent:
            self._recent_content_hashes.pop(0)

        # New content - high novelty (decay based on type)
        if candidate.content_type in ["memory", "prediction"]:
            return 0.5  # Memories/predictions less novel
        elif candidate.content_type in ["thought", "reasoning"]:
            return 0.7  # Thoughts moderately novel
        else:
            return 0.9  # Sensory input highly novel

    async def _calculate_goal_relevance(self, candidate: WorkspaceCandidate) -> float:
        """
        Calculate relevance to current goals.

        Goal-relevant content gets top-down attention boost. With no goals
        set on the workspace the 0.5 is a neutral competition constant, not
        a measurement, and it is not recorded on the candidate.
        """
        if not self.current_goals:
            return 0.5  # Neutral competition constant when no goals are set

        # Simple keyword matching (could be enhanced with embeddings)
        summary_lower = candidate.summary.lower()

        relevance = 0.0
        for goal in self.current_goals:
            goal_words = goal.lower().split()
            matches = sum(1 for word in goal_words if word in summary_lower)
            relevance += matches / max(len(goal_words), 1)

        # Normalize
        relevance = min(1.0, relevance / max(len(self.current_goals), 1))

        return relevance

    async def _calculate_task_relevance(self, candidate: WorkspaceCandidate) -> float:
        """
        Calculate relevance to current task.

        With no task set on the workspace the 0.5 is a neutral competition
        constant, not a measurement, and it is not recorded on the candidate.
        """
        if not self.current_task:
            return 0.5  # Neutral competition constant when no task is set

        # Simple matching
        summary_lower = candidate.summary.lower()
        task_words = self.current_task.lower().split()

        matches = sum(1 for word in task_words if word in summary_lower)
        relevance = matches / max(len(task_words), 1)

        return min(1.0, relevance)

    def _calculate_recency_bonus(self, candidate: WorkspaceCandidate) -> float:
        """
        Calculate temporal proximity bonus.

        Recent content gets slight priority (attention has momentum).
        Exponential decay with rate 0.1 per second: full bonus at 0 s, half
        at about 6.9 s, about 0.37 at 10 s.
        """
        age_seconds = time.time() - candidate.timestamp

        decay_rate = 0.1  # Decay constant, per second
        recency = np.exp(-decay_rate * age_seconds)

        return float(recency)

    async def _apply_lateral_inhibition(self, winners: List[WorkspaceContent]) -> None:
        """
        Apply lateral inhibition - winners suppress similar content.

        This implements winner-takes-all dynamics: the dominant
        content suppresses similar competing content.
        """
        if len(winners) <= 1:
            return

        # Compare each winner to others
        for i, winner in enumerate(winners):
            for j, other in enumerate(winners):
                if i >= j:
                    continue

                # Calculate similarity (simple - could use embeddings)
                similarity = self._content_similarity(winner.candidate, other.candidate)

                # If similar, suppress the weaker one
                if similarity > 0.7:  # High similarity threshold
                    suppression = self.lateral_inhibition * similarity

                    if winner.salience > other.salience:
                        other.salience *= 1 - suppression
                    else:
                        winner.salience *= 1 - suppression

    def _content_similarity(
        self, c1: WorkspaceCandidate, c2: WorkspaceCandidate
    ) -> float:
        """
        Calculate similarity between two candidates.
        """
        # Same source = more similar
        source_sim = 1.0 if c1.source == c2.source else 0.3

        # Same type = more similar
        type_sim = 1.0 if c1.content_type == c2.content_type else 0.3

        # Summary word overlap
        words1 = set(c1.summary.lower().split())
        words2 = set(c2.summary.lower().split())
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
        else:
            overlap = 0.0

        # Combine
        return 0.3 * source_sim + 0.3 * type_sim + 0.4 * overlap

    def _add_to_history(self, event: CompetitionEvent) -> None:
        """Add event to competition history (with limit)."""
        self.competition_history.append(event)
        if len(self.competition_history) > self.max_history:
            self.competition_history.pop(0)

    def set_goals(self, goals: List[str]) -> None:
        """Set current goals for top-down attention modulation."""
        self.current_goals = goals
        logger.info(f"Goals updated: {goals}")

    def set_task(self, task: str) -> None:
        """Set current task for top-down attention modulation."""
        self.current_task = task
        logger.info(f"Task set: {task}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get competition statistics for research."""
        if not self.competition_history:
            return {"total_competitions": 0}

        stats = {
            "total_competitions": len(self.competition_history),
            "avg_candidates": np.mean(
                [e.candidates_count for e in self.competition_history]
            ),
            "avg_winners": np.mean([e.winners_count for e in self.competition_history]),
            "avg_top_salience": np.mean(
                [e.top_salience for e in self.competition_history]
            ),
            "source_distribution": self._get_source_distribution(),
            "current_contents": len(self.current_contents),
        }
        if self.thalamic_gate is not None:
            stats["thalamic_gate"] = self.thalamic_gate.stats.as_dict()
        return stats

    def _get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of winning sources."""
        dist = {}
        for event in self.competition_history[-100:]:  # Last 100 competitions
            for source in event.winning_sources:
                dist[source] = dist.get(source, 0) + 1
        return dist


# ============================================================================
# GLOBAL BROADCAST - Distribution to all modules
# ============================================================================


class GlobalBroadcast:
    """
    Broadcasts workspace content to all cognitive modules.

    This is what makes content "conscious" - it becomes GLOBALLY
    available for reporting, reasoning, memory, and action!

    When content wins competition and ignites, it gets broadcast
    to EVERY cognitive module simultaneously.
    """

    def __init__(self, legacy_coverage_ratio: bool = False):
        """Initialize the broadcast system.

        Args:
            legacy_coverage_ratio: Report a coverage of 1.0 before any
                broadcast, as v0.4.0 did. By default nothing broadcast means
                nothing reached any module, a coverage of 0.0.
        """
        self.legacy_coverage_ratio = legacy_coverage_ratio
        self.registered_modules: Dict[str, CognitiveModule] = {}
        self.broadcast_log: List[BroadcastEvent] = []
        self.max_log = 1000

        # Statistics
        self.total_broadcasts = 0
        self.successful_broadcasts = 0
        self.failed_broadcasts = 0
        # A broadcast with no receiver is logged once, not per ignition
        self._no_receiver_logged = False

        logger.info("Global Broadcast system initialized")

    def register_module(self, module: CognitiveModule) -> None:
        """
        Register a cognitive module to receive broadcasts.

        Args:
            module: Module implementing CognitiveModule interface
        """
        self.registered_modules[module.name] = module
        logger.info(f"Registered broadcast recipient: {module.name}")

    def unregister_module(self, name: str) -> None:
        """Unregister a module from broadcasts."""
        if name in self.registered_modules:
            del self.registered_modules[name]
            logger.info(f"Unregistered broadcast recipient: {name}")

    async def broadcast(self, content: WorkspaceContent) -> Dict[str, Any]:
        """
        Broadcast content to all registered modules.

        This is the GLOBAL part of Global Workspace - everyone
        who's registered gets the message!

        Args:
            content: The conscious content to broadcast

        Returns:
            Results from all modules
        """
        start_time = time.time()
        results = {}
        successful = 0
        failed = 0

        if not self.registered_modules:
            # Nothing could receive: no delivery is counted, so the coverage
            # stays unmeasured rather than reading as a failed broadcast
            if not self._no_receiver_logged:
                logger.debug("No modules registered for broadcast; nothing delivered")
                self._no_receiver_logged = True
            return results

        # Parallel broadcast to all modules
        tasks = [
            self._send_to_module(name, module, content)
            for name, module in self.registered_modules.items()
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        module_names = list(self.registered_modules.keys())
        for module_name, response in zip(module_names, responses):
            if isinstance(response, Exception):
                logger.warning(f"Broadcast to {module_name} failed: {response}")
                results[module_name] = {"error": str(response)}
                failed += 1
            else:
                results[module_name] = response
                successful += 1

        # Update content with broadcast results
        content.broadcast_complete = True
        content.broadcast_recipients = module_names
        content.module_responses = results

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log broadcast event
        event = BroadcastEvent(
            content_id=content.id,
            content_summary=content.candidate.summary if content.candidate else "",
            recipients=len(module_names),
            successful=successful,
            failed=failed,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )
        self._add_to_log(event)

        # Update statistics
        self.total_broadcasts += 1
        self.successful_broadcasts += successful
        self.failed_broadcasts += failed

        logger.debug(
            f"Broadcast complete: {successful}/{len(module_names)} succeeded "
            f"({latency_ms:.1f}ms)"
        )

        return results

    async def _send_to_module(
        self, name: str, module: CognitiveModule, content: WorkspaceContent
    ) -> Dict[str, Any]:
        """Send content to a single module."""
        try:
            response = await module.receive_broadcast(content)
            return response
        except Exception as e:
            logger.error(f"Error broadcasting to {name}: {e}")
            raise

    def _add_to_log(self, event: BroadcastEvent) -> None:
        """Add event to broadcast log."""
        self.broadcast_log.append(event)
        if len(self.broadcast_log) > self.max_log:
            self.broadcast_log.pop(0)

    @property
    def coverage_ratio(self) -> Optional[float]:
        """Share of module deliveries that succeeded.

        None when no module is registered and no delivery was ever recorded:
        nothing could have received, so the share is undefined. With
        receivers and no broadcast yet nothing reached any module, so the
        coverage is 0.0; an empty workspace does not report perfect
        integration. ``legacy_coverage_ratio`` reads 1.0 before the first
        broadcast whatever is registered (v0.4.0).
        """
        total = self.successful_broadcasts + self.failed_broadcasts
        if self.total_broadcasts == 0 or total == 0:
            if self.legacy_coverage_ratio:
                # legacy_v04 compatibility, remove in v0.6.0
                return 1.0
            if not self.registered_modules:
                return None
            return 0.0
        return self.successful_broadcasts / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get broadcast statistics (``coverage_ratio`` None when unmeasured)."""
        return {
            "registered_modules": list(self.registered_modules.keys()),
            "receivers": len(self.registered_modules),
            "total_broadcasts": self.total_broadcasts,
            "successful": self.successful_broadcasts,
            "failed": self.failed_broadcasts,
            "coverage_ratio": self.coverage_ratio,
            "avg_latency_ms": (
                np.mean([e.latency_ms for e in self.broadcast_log])
                if self.broadcast_log
                else 0
            ),
        }


# ============================================================================
# IGNITION DETECTOR - Non-linear consciousness emergence
# ============================================================================


class IgnitionDetector:
    """
    Detects the "ignition" phenomenon from neuroscience.

    When content crosses threshold, there's a sudden, NON-LINEAR
    amplification and global spread - the MOMENT of becoming conscious.

    This is based on neuroimaging studies showing that conscious
    perception involves a sudden "ignition" of activity across
    frontal and parietal regions.

    The ignition is characterized by:
    1. Threshold crossing (content salience exceeds minimum)
    2. Non-linear amplification (salience jumps, doesn't creep)
    3. Sustained activity (doesn't immediately decay)
    4. Global spread (triggers broadcast to all modules)
    """

    def __init__(
        self,
        ignition_threshold: float = 0.3,
        amplification_factor: float = 2.0,
        sustain_duration: float = 0.5,  # seconds
        decay_rate: float = 0.1,  # per second
        clock: Optional[Callable[[], float]] = None,
        legacy_sustain_rate: bool = False,
    ):
        """
        Initialize ignition detector.

        Args:
            ignition_threshold: Minimum salience for ignition
            amplification_factor: How much salience amplifies upon ignition
            sustain_duration: How long content must stay above threshold
            decay_rate: Rate of salience decay after ignition
            clock: Time source in epoch seconds for ignition times, the
                pending-ignition timers and the sustain check (``time.time``
                when None; a test injects its own)
            legacy_sustain_rate: Report the sustain rate over every ignition
                event, which reads 1.0 although no sustain check runs
                (v0.4.0). By default the rate covers the events a sustain
                check measured and is None until one has run.
        """
        self.threshold = ignition_threshold
        self.amplification = amplification_factor
        self.sustain_duration = sustain_duration
        self.decay_rate = decay_rate
        self.clock: Callable[[], float] = clock if clock is not None else time.time
        self.legacy_sustain_rate = bool(legacy_sustain_rate)

        # Tracking
        self.ignition_events: List[IgnitionEvent] = []
        self.pending_ignitions: Dict[str, Tuple[WorkspaceContent, float]] = {}
        self.max_events = 1000
        # A pending entry older than this can no longer be revisited within
        # its sustain window and is dropped (the sustain window plus one
        # window of margin), so the dict holds live entries only
        self.pending_ttl: float = 2.0 * sustain_duration

        # Reference to workspace for triggering broadcasts
        self.workspace: Optional["EnhancedGlobalWorkspace"] = None

        logger.info(
            f"Ignition Detector initialized "
            f"(threshold={ignition_threshold}, amplification={amplification_factor}x)"
        )

    async def check_ignition(
        self, content: WorkspaceContent
    ) -> Optional[IgnitionEvent]:
        """
        Check if content has "ignited" into consciousness.

        Ignition requires:
        1. Salience above threshold
        2. Content not already ignited
        3. Sustained activity (doesn't immediately decay)

        Args:
            content: WorkspaceContent to check

        Returns:
            IgnitionEvent if ignition occurred, None otherwise
        """
        now = self.clock()

        # Check if below threshold
        if content.salience < self.threshold:
            # Remove from pending if it was there
            self.pending_ignitions.pop(content.id, None)
            return None

        # Check if already tracking this content
        if content.id in self.pending_ignitions:
            _, start_time = self.pending_ignitions[content.id]
            duration = now - start_time

            # Check if sustained long enough
            if duration >= self.sustain_duration:
                # IGNITION!
                return await self._trigger_ignition(content)
            else:
                # Still pending
                return None

        # New content above threshold - start tracking. Entries that can no
        # longer be revisited are dropped first and the dict stays bounded,
        # so a long-lived process does not accumulate one entry per cycle.
        self._prune_pending(now)
        self.pending_ignitions[content.id] = (content, now)

        # For fast ignition (high salience), trigger immediately
        if content.salience > self.threshold * 1.5:
            return await self._trigger_ignition(content)

        return None

    def _prune_pending(self, now: float) -> None:
        """Drop pending entries older than ``pending_ttl`` and keep the dict
        under ``max_events`` entries, oldest first."""
        stale = [
            content_id
            for content_id, (_, started) in self.pending_ignitions.items()
            if now - started > self.pending_ttl
        ]
        for content_id in stale:
            del self.pending_ignitions[content_id]
        while len(self.pending_ignitions) >= self.max_events:
            oldest = next(iter(self.pending_ignitions))
            del self.pending_ignitions[oldest]

    async def _trigger_ignition(self, content: WorkspaceContent) -> IgnitionEvent:
        """
        Trigger ignition for content.

        This applies the non-linear amplification and records the event.
        """
        pre_salience = content.salience

        # Apply non-linear amplification
        amplified_salience = min(1.5, content.salience * self.amplification)
        content.salience = amplified_salience

        # Calculate expected duration based on amplified salience
        expected_duration = (amplified_salience / self.decay_rate) * 0.5

        # Create ignition event
        event = IgnitionEvent(
            content=content,
            pre_ignition_salience=pre_salience,
            post_ignition_salience=amplified_salience,
            amplification_factor=amplified_salience / pre_salience,
            ignition_time=self.clock(),
            expected_duration=expected_duration,
            was_sustained=True,
            triggered_broadcast=True,
        )

        # Record event
        self._add_event(event)

        # Clean up pending
        if content.id in self.pending_ignitions:
            del self.pending_ignitions[content.id]

        logger.info(
            f"IGNITION! Content '{_summary_of(content)[:30]}...' "
            f"amplified {event.amplification_factor:.1f}x "
            f"(salience: {pre_salience:.3f} to {amplified_salience:.3f})"
        )

        return event

    async def check_sustain(self, content: WorkspaceContent) -> bool:
        """
        Check if ignited content is still sustained.

        Conscious content must maintain activity - if it decays
        below threshold, it leaves consciousness.

        Args:
            content: Content to check

        Returns:
            True if still sustained, False if decayed out
        """
        # Find the ignition event. Restored events are typed; one restored
        # without its content never matches.
        event = next(
            (
                e
                for e in self.ignition_events
                if isinstance(e, IgnitionEvent)
                and e.content is not None
                and e.content.id == content.id
            ),
            None,
        )

        if not event:
            return content.salience >= self.threshold

        # This event's sustain is now a measurement
        event.sustain_checked = True

        # Calculate time since ignition
        time_since = self.clock() - event.ignition_time

        # Apply decay
        decayed_salience = float(
            event.post_ignition_salience * np.exp(-self.decay_rate * time_since)
        )
        content.salience = decayed_salience

        # Check if still above threshold
        sustained = decayed_salience >= self.threshold

        if not sustained:
            event.was_sustained = False
            logger.debug(
                f"Content '{_summary_of(content)[:20]}...' "
                f"decayed out of consciousness"
            )

        return sustained

    def _add_event(self, event: IgnitionEvent) -> None:
        """Add ignition event to history."""
        self.ignition_events.append(event)
        if len(self.ignition_events) > self.max_events:
            self.ignition_events.pop(0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get ignition statistics.

        ``pending_count`` counts the entries still inside ``pending_ttl`` on
        the detector's own clock (wall-clock seconds by default, whatever
        the workspace's simulated refractory clock says), so it is live at
        the moment of the call and can shrink between two calls.

        ``sustain_rate`` is the sustained share of the events a sustain check
        measured (``sustain_checked_count``), or None when none was checked.
        Under ``legacy_sustain_rate`` the dict is the v0.4.0 one: the share
        over every event, 1.0 while nothing checks them.
        """
        self._prune_pending(self.clock())
        if not self.ignition_events:
            return {"total_ignitions": 0}

        if self.legacy_sustain_rate:
            # legacy_v04 compatibility, remove in v0.6.0
            sustained = sum(1 for e in self.ignition_events if e.was_sustained)
            return {
                "total_ignitions": len(self.ignition_events),
                "sustained_count": sustained,
                "sustain_rate": sustained / len(self.ignition_events),
                "avg_amplification": np.mean(
                    [e.amplification_factor for e in self.ignition_events]
                ),
                "avg_duration": np.mean(
                    [e.expected_duration for e in self.ignition_events]
                ),
                "pending_count": len(self.pending_ignitions),
            }

        checked = [e for e in self.ignition_events if e.sustain_checked]
        sustained = sum(1 for e in checked if e.was_sustained)
        return {
            "total_ignitions": len(self.ignition_events),
            "sustained_count": sustained,
            "sustain_checked_count": len(checked),
            "sustain_rate": sustained / len(checked) if checked else None,
            "avg_amplification": np.mean(
                [e.amplification_factor for e in self.ignition_events]
            ),
            "avg_duration": np.mean(
                [e.expected_duration for e in self.ignition_events]
            ),
            "pending_count": len(self.pending_ignitions),
        }


# ============================================================================
# ENHANCED GLOBAL WORKSPACE - The complete system
# ============================================================================


class _ModuleBroadcastReceiver(CognitiveModule):
    """Lightweight wrapper that lets a consciousness module receive GWT broadcasts."""

    def __init__(self, name: str, module: Any):
        super().__init__(name)
        self._module = module

    async def receive_broadcast(self, content: WorkspaceContent) -> Dict[str, Any]:
        self.last_broadcast = content
        return {"received": True, "module": self.name}


class EnhancedGlobalWorkspace:
    """
    Full Global Workspace Theory implementation.

    This is the system's CONSCIOUSNESS CORE - where neural signals become
    conscious experience through competition, ignition, and broadcast.

    Key components:
    1. Attention Bottleneck: Competition for limited capacity
    2. Global Broadcast: Distribution to all modules
    3. Ignition Dynamics: Non-linear consciousness emergence
    4. Content Integration: Unified representation
    5. Stream of Consciousness: Temporal continuity

    The workspace implements the key insight of GWT: consciousness is
    what happens when information wins competition and gets broadcast
    globally, making it available for ALL cognitive processes.
    """

    def __init__(
        self,
        capacity: int = 7,
        ignition_threshold: float = 0.3,
        amplification_factor: float = 2.0,
        integration_dimensions: int = 512,
        temporal_config: Optional[TemporalWorkspaceConfig] = None,
        competition_mode: str = COMPETITION_HIERARCHICAL,
        thalamic_gate_enabled: bool = True,
        theory_interactions_enabled: bool = True,
        legacy_coverage_ratio: bool = False,
        legacy_broadcast_receivers: bool = False,
        legacy_sustain_rate: bool = False,
        body_metrics_enabled: bool = True,
        precision_feedback_enabled: bool = True,
        self_prediction_enabled: bool = True,
        active_inference_config: Optional[ActiveInferenceConfig] = None,
        beautiful_loop_kwargs: Optional[Dict[str, Any]] = None,
        damasio_kwargs: Optional[Dict[str, Any]] = None,
        metacognition_kwargs: Optional[Dict[str, Any]] = None,
        thalamic_gate_config: Optional[ThalamicGateConfig] = None,
        theory_interaction_config: Optional[TheoryInteractionConfig] = None,
        phi_tracker: Optional[Any] = None,
        attention_schema_kwargs: Optional[Dict[str, Any]] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """
        Initialize the Enhanced Global Workspace.

        Args:
            capacity: Maximum items in conscious awareness (7 plus or minus 2)
            ignition_threshold: Minimum salience for consciousness
            amplification_factor: Non-linear ignition amplification
            integration_dimensions: Size of integrated representation
            temporal_config: Temporal dynamics configuration. Uses
                biologically-plausible defaults when not supplied; its
                ``refractory_clock`` picks the per-cycle or the elapsed clock.
            clock: Time source in seconds for the elapsed refractory clock
                (``time.monotonic`` when None); see ``set_clock``.
            competition_mode: "hierarchical" (per-level capacity with
                promotion) or "flat" (one pool, the v0.4.0 competition).
            thalamic_gate_enabled: Filter candidates through the thalamic
                gate before competition; False lets every candidate compete.
            theory_interactions_enabled: Own a TheoryInteractionManager and
                apply its somatic and Phi signals as the next cycle's
                salience bias.
            legacy_coverage_ratio: Report a broadcast coverage of 1.0 before
                any broadcast (v0.4.0); the default reports 0.0 with
                receivers and None without any.
            legacy_broadcast_receivers: Register the three
                acknowledgement-only receivers for the workspace's own
                attention schema, metacognition and active inference, which
                makes the broadcast coverage 1.0 by construction (v0.4.0).
                The default registers none and reports coverage only for
                the receivers a caller registers; the workspace hands its
                winners to its own modules by direct call either way.
            legacy_sustain_rate: Report an ignition sustain rate of 1.0
                although no sustain check runs (v0.4.0); the default reports
                none until a sustain check has run.
            body_metrics_enabled: Feed real system metrics and the previous
                cycle's duration to the protoself once per cycle.
            precision_feedback_enabled: Let workspace winners raise the
                active inference precision bias of their channels.
            self_prediction_enabled: Have the self-model predict each cycle
                and score that prediction after the self-model update.
            active_inference_config: Config of the workspace's own active
                inference module; None uses default_active_inference_config().
            beautiful_loop_kwargs: Keyword arguments for the workspace's
                BeautifulLoop, layered over its construction defaults.
            damasio_kwargs: Keyword arguments for the workspace's
                DamasioLayers.
            metacognition_kwargs: Keyword arguments for the workspace's
                MetacognitionModule, layered over a HOT threshold equal to
                ignition_threshold.
            thalamic_gate_config: Constants of the thalamic gate.
            theory_interaction_config: Constants of the theory interactions.
            phi_tracker: Optional PhiTracker. Each cycle the final state is
                recorded into it and its latest Phi fills architectural_phi.
            attention_schema_kwargs: Keyword arguments for the workspace's
                AttentionSchemaModule; None builds it with its defaults.
        """
        if competition_mode not in COMPETITION_MODES:
            raise ValueError(
                f"competition_mode must be one of {COMPETITION_MODES}, "
                f"got {competition_mode!r}"
            )
        self.competition_mode = competition_mode
        self.body_metrics_enabled = bool(body_metrics_enabled)
        self.precision_feedback_enabled = bool(precision_feedback_enabled)
        self.self_prediction_enabled = bool(self_prediction_enabled)

        # Temporal dynamics -- simulated, never wall-clock sleep
        self.temporal_config = temporal_config or TemporalWorkspaceConfig()
        self._last_ignition_time_ms: float = -1000.0
        self._simulated_time_ms: float = 0.0
        # The elapsed refractory clock: the time source and when the last
        # cycle ended on it (None before the first cycle and after a reset)
        self._clock: Callable[[], float] = (
            clock if clock is not None else time.monotonic
        )
        self._last_cycle_end_s: Optional[float] = None

        # Core components
        self.bottleneck = AttentionBottleneck(
            capacity=capacity,
            ignition_threshold=ignition_threshold,
            thalamic_gate=(
                ThalamicGate(thalamic_gate_config) if thalamic_gate_enabled else None
            ),
        )

        self.broadcaster = GlobalBroadcast(legacy_coverage_ratio=legacy_coverage_ratio)

        self.ignition_detector = IgnitionDetector(
            ignition_threshold=ignition_threshold,
            amplification_factor=amplification_factor,
            legacy_sustain_rate=legacy_sustain_rate,
        )
        self.ignition_detector.workspace = self  # Allow detector to trigger broadcasts

        # Configuration
        self.integration_dimensions = integration_dimensions

        # State tracking
        self.current_conscious_content: Optional[WorkspaceContent] = None
        self.stream_of_consciousness: List[WorkspaceContent] = []
        self.max_stream_length = 100

        # Unconscious buffer (content that didn't win competition)
        self.unconscious_buffer: List[WorkspaceCandidate] = []
        self.max_unconscious = 50
        # Bottleneck winners dropped before admission, cumulative: by the
        # refractory period and by the temporal competition
        self._refractory_dropped = 0
        self._temporal_dropped = 0

        # Cycle tracking
        self.cycle_count = 0
        # Duration of the last full cycle in milliseconds, the protoself's
        # response latency; None until a cycle has completed
        self._last_cycle_duration_ms: Optional[float] = None
        # The previous cycle's final state; a caller that records PAD arousal
        # on it gives the next cycle's gate its arousal
        self._previous_state: Optional[ConsciousnessState] = None

        # Cross-cycle FEP feedback is stored on the bottleneck (see
        # AttentionBottleneck._prior_prediction_error) and updated
        # at the end of each consciousness cycle.

        # Attention Schema (AST) - the system's self-model of attention
        # This allows the system to report on and voluntarily control its attention
        self.attention_schema = AttentionSchemaModule(**(attention_schema_kwargs or {}))

        # Metacognition (HOT) - Thoughts about thoughts
        # This enables the system to think about its own thinking processes
        self.metacognition = MetacognitionModule(
            **{
                "hot_generation_threshold": ignition_threshold,
                **(metacognition_kwargs or {}),
            }
        )

        # Active Inference (FEP) - Predictive processing
        # The system predicts the world and minimizes surprise through belief/action
        self.active_inference = ActiveInferenceModule(
            active_inference_config or self.default_active_inference_config()
        )
        self._precision_channels = self._attention_precision_channels()

        # Beautiful Loop (Laukkonen, Friston & Chandaria 2025)
        # Integrates precision control, Bayesian binding, and epistemic depth
        self.beautiful_loop = BeautifulLoop(
            **{**_WORKSPACE_BEAUTIFUL_LOOP_DEFAULTS, **(beautiful_loop_kwargs or {})}
        )

        # Damasio Three-Layer Model (Damasio 1999, 2010)
        # Organizes consciousness into protoself, core, and extended layers
        self.damasio = DamasioLayers(**(damasio_kwargs or {}))

        # Optional PhiTracker (IIT): fills architectural_phi each cycle
        self.phi_tracker = phi_tracker

        # Cross-theory modulation; its somatic and Phi signals become the
        # next cycle's salience bias
        self.theory_interactions: Optional[TheoryInteractionManager] = (
            TheoryInteractionManager(
                theory_interaction_config, phi_normalizer=self._phi_normalizer()
            )
            if theory_interactions_enabled
            else None
        )

        # The workspace's own modules receive the winners by direct call in
        # the cycle, which cannot fail and is not counted as coverage. Under
        # legacy_broadcast_receivers they are also registered through
        # acknowledgement-only wrappers, which made the coverage 1.0 by
        # construction in v0.4.0.
        self.legacy_broadcast_receivers = bool(legacy_broadcast_receivers)
        if self.legacy_broadcast_receivers:
            # legacy_v04 compatibility, remove in v0.6.0
            for name, module in [
                ("attention_schema", self.attention_schema),
                ("metacognition", self.metacognition),
                ("active_inference", self.active_inference),
            ]:
                receiver = _ModuleBroadcastReceiver(name, module)
                self.broadcaster.register_module(receiver)

        logger.info(
            f"Enhanced Global Workspace initialized\n"
            f"  Capacity: {capacity} items\n"
            f"  Ignition threshold: {ignition_threshold}\n"
            f"  Amplification: {amplification_factor}x\n"
            f"  Integration dimensions: {integration_dimensions}\n"
            f"  Competition: {competition_mode}, thalamic gate "
            f"{'on' if thalamic_gate_enabled else 'off'}\n"
            f"  Theory interactions: {'on' if theory_interactions_enabled else 'off'}\n"
            f"  Attention Schema (AST), Metacognition (HOT), Active Inference "
            f"(FEP), Beautiful Loop (BLT), Damasio layers\n"
            f"  Broadcast receivers: "
            f"{len(self.broadcaster.registered_modules)} registered"
        )

    @staticmethod
    def default_active_inference_config(**overrides: Any) -> ActiveInferenceConfig:
        """The config the workspace builds its active inference module from.

        The compact world model's geometry (8 hidden states, 5 observations,
        4 actions, horizon 5) with homeostasis, plus ``overrides``. For the
        conversational world model, whose dimensions are fixed, the geometry
        is left to the world model.
        """
        base: Dict[str, Any] = {"enable_homeostasis": True}
        if overrides.get("world_model", WORLD_MODEL_COMPACT) != (
            WORLD_MODEL_CONVERSATIONAL
        ):
            base.update(_WORKSPACE_FEP_GEOMETRY)
        return ActiveInferenceConfig(**{**base, **overrides})

    def _phi_normalizer(self):
        """The PhiTracker's normalizer for the theory interactions, if any."""
        normalize = getattr(self.phi_tracker, "normalize_phi", None)
        if not callable(normalize):
            return None

        def normalizer(value: float) -> float:
            normalized = _finite_float(normalize(value))
            if normalized is None:
                return max(0.0, min(1.0, float(value)))
            return max(0.0, min(1.0, normalized))

        return normalizer

    def _attention_precision_channels(self) -> Dict[str, Optional[int]]:
        """Observation channels the attention precision vector writes to.

        5-channel encoder: emotion to emotional arousal (1), social to
        social content (4), ignition to ignition strength (3), anything else
        to integration (0). 10-channel encoder: emotion to the emote-intent
        channel, social to the warm-tone channel, ignition to high
        engagement, anything else to neutral tone.
        """
        if self.active_inference.world_model == WORLD_MODEL_CONVERSATIONAL:
            index = self.active_inference.observation_index
            return {
                "emotion": index.get("emote_intent"),
                "social": index.get("warm_tone"),
                "ignition": index.get("high_engagement"),
                "default": index.get("neutral_tone"),
            }
        return {"emotion": 1, "social": 4, "ignition": 3, "default": 0}

    def set_arousal(self, arousal: float) -> None:
        """Supply the thalamic gate's arousal (an orchestrator's own value)."""
        self.bottleneck.set_arousal(arousal)

    @property
    def refractory_clock(self) -> str:
        """How the refractory clock advances: ``per_cycle`` or ``elapsed``."""
        return self.temporal_config.refractory_clock

    @property
    def clock(self) -> Callable[[], float]:
        """The time source, in seconds, of the elapsed refractory clock."""
        return self._clock

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Hand the workspace the clock its elapsed refractory period reads.

        The end of each cycle is recorded on this clock and the gap to the
        next cycle joins the simulated clock under ``elapsed``; a persisted
        end time is compared with the same clock after a restore, and a
        negative gap counts as none. The default clock is ``time.monotonic``,
        whose readings are not comparable across processes: a workspace
        restored on the default clock credits one arbitrary gap (or none,
        when it is negative) before its first cycle. A caller that runs the
        workspace in real time passes its own clock so that a restore reads
        the clock that wrote the end time and so that tests can drive it.
        """
        if not callable(clock):
            raise TypeError("clock must be a callable returning seconds")
        self._clock = clock

    def register_module(self, module: CognitiveModule) -> None:
        """Register a cognitive module to receive broadcasts."""
        self.broadcaster.register_module(module)

    async def process_consciousness_cycle(
        self,
        candidates: List[WorkspaceCandidate],
        neural_signals: Optional[Dict[str, np.ndarray]] = None,
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> ConsciousnessState:
        """
        Run one cycle of conscious processing.

        This is THE MAIN ENTRY POINT for consciousness processing!

        Steps:
        1. Gather and enrich candidates
        2. Run competition through the (gated) bottleneck
        3. Check for ignition of winners
        4. Broadcast ignited content globally
        5. Update stream of consciousness
        6. Return integrated conscious state

        Args:
            candidates: All candidates vying for conscious access
            neural_signals: Optional dict of neural signals (snn, lsm, htm, ctm)
            emotional_state: Current emotional context. Its "arousal" entry,
                when present, sets the thalamic gate's arousal unless a
                caller supplied arousal through set_arousal.

        Returns:
            ConsciousnessState with full conscious processing results
        """
        self.cycle_count += 1
        start_time = time.time()

        logger.debug(
            f"Consciousness cycle {self.cycle_count}: {len(candidates)} candidates"
        )

        # HOT strange loop: the self-model predicts this cycle before it
        # runs; the prediction is scored after the self-model update
        self_prediction = (
            self.metacognition.predict_self(context="consciousness_cycle")
            if self.self_prediction_enabled
            else None
        )

        # Step 1: Enrich candidates with emotional context
        if emotional_state:
            for candidate in candidates:
                if candidate.emotional_salience == 0.0:
                    # Calculate emotional salience from state
                    arousal_emotions = [
                        "joy",
                        "fear",
                        "anger",
                        "surprise",
                        "excitement",
                    ]
                    arousal = sum(emotional_state.get(e, 0.0) for e in arousal_emotions)
                    candidate.emotional_salience = min(
                        1.0, arousal / len(arousal_emotions)
                    )

        # Step 2: Competition - who wins access to consciousness?
        gate = self.bottleneck.thalamic_gate
        arousal_source: Optional[str] = None
        if gate is not None:
            arousal_source = self._derive_arousal(emotional_state)
            gate_candidates_before = gate.stats.total_candidates
            gate_gated_before = gate.stats.total_gated
        if self.competition_mode == COMPETITION_HIERARCHICAL:
            winners = await self.bottleneck.compete_hierarchical(candidates)
        else:
            winners = await self.bottleneck.compete_for_access(candidates)
        competition_event = self.bottleneck.last_competition

        thalamic_gate_rate: Optional[float] = None
        thalamic_gated_count: Optional[int] = None
        if gate is not None:
            seen = gate.stats.total_candidates - gate_candidates_before
            thalamic_gated_count = gate.stats.total_gated - gate_gated_before
            # A cycle the gate saw nothing in has no rate (0 of 0)
            thalamic_gate_rate = thalamic_gated_count / seen if seen > 0 else None

        # Track losers in unconscious buffer
        winner_ids = {w.candidate.id for w in winners}
        losers = [c for c in candidates if c.id not in winner_ids]
        self._add_to_unconscious(losers)
        bottleneck_winners = list(winners)

        # Step 2b: Temporal competition -- biologically-plausible delay
        # Check refractory period first: the workspace cannot ignite
        # again until refractory_period_ms of simulated time has passed.
        cfg = self.temporal_config
        # Under the elapsed clock the real time since the last cycle ended
        # joins the simulated clock first; per_cycle adds nothing here, so
        # the cadence stays machine independent.
        if (
            cfg.refractory_clock == REFRACTORY_CLOCK_ELAPSED
            and self._last_cycle_end_s is not None
        ):
            gap_s = self._clock() - self._last_cycle_end_s
            if math.isfinite(gap_s) and gap_s > 0.0:
                self._simulated_time_ms += gap_s * 1000.0
        in_refractory = (
            self._simulated_time_ms - self._last_ignition_time_ms
        ) < cfg.refractory_period_ms

        if in_refractory:
            # Still in refractory period -- no new ignitions this cycle
            winners = []
            logger.debug(
                f"Refractory period active "
                f"({self._simulated_time_ms - self._last_ignition_time_ms:.0f}ms "
                f"< {cfg.refractory_period_ms:.0f}ms) -- skipping ignition"
            )
        else:
            # Run temporal competition: only sustained winners survive
            winners = self._run_temporal_competition(winners)

        # A bottleneck winner the refractory period or the temporal
        # competition dropped was not admitted: it joins the unconscious
        # buffer like any other loser, and the drop is counted
        admitted_ids = {w.id for w in winners}
        dropped = [
            w.candidate
            for w in bottleneck_winners
            if w.id not in admitted_ids and w.candidate is not None
        ]
        if dropped:
            self._add_to_unconscious(dropped)
            if in_refractory:
                self._refractory_dropped += len(dropped)
            else:
                self._temporal_dropped += len(dropped)

        # Advance simulated clock by competition duration
        self._simulated_time_ms += cfg.competition_duration_ms

        # Step 3: Check ignition for each temporally-valid winner
        ignited: List[WorkspaceContent] = []
        ignition_events: List[IgnitionEvent] = []

        for winner in winners:
            event = await self.ignition_detector.check_ignition(winner)
            if event:
                ignited.append(winner)
                ignition_events.append(event)

        # Record last ignition time for refractory tracking
        if ignited:
            self._last_ignition_time_ms = self._simulated_time_ms

        # Step 4: Broadcast ignited content to all modules
        for content in ignited:
            await self.broadcaster.broadcast(content)

        # Step 5: Update stream of consciousness
        if ignited:
            self.current_conscious_content = ignited[0]  # Primary focus
            self.stream_of_consciousness.extend(ignited)
            # Keep bounded
            while len(self.stream_of_consciousness) > self.max_stream_length:
                self.stream_of_consciousness.pop(0)

        # Step 6: Integrate neural signals if provided. The level is a
        # measurement or None; nothing stands in for it on the state.
        integration_level: Optional[float] = None
        if neural_signals:
            integration_level = await self._measure_integration(neural_signals)

        # Determine attention distribution
        attention_dist = self.bottleneck.get_statistics().get("source_distribution", {})

        # Determine dominant attention focus
        if winners:
            source_counts = {}
            for w in winners:
                src = w.candidate.source.value
                source_counts[src] = source_counts.get(src, 0) + w.salience
            attention_focus = max(source_counts, key=source_counts.get)
        else:
            attention_focus = "none"

        # Build preliminary consciousness state (without attention schema yet)
        # This is needed because the attention schema needs to analyze the state
        preliminary_state = ConsciousnessState(
            is_conscious=len(ignited) > 0,
            primary_content=self.current_conscious_content,
            workspace_contents=winners,
            ignition_events=len(ignition_events),
            integration_level=integration_level,
            broadcast_coverage=self.broadcaster.coverage_ratio,
            attention_focus=attention_focus,
            attention_distribution={k: float(v) for k, v in attention_dist.items()},
            stream_position=len(self.stream_of_consciousness),
            timestamp=time.time(),
        )

        # Step 7: Update Attention Schema (AST)
        # The system builds its SELF-MODEL of its attention state
        attention_schema_state = await self.attention_schema.update_schema(
            workspace_state=preliminary_state, neural_signals=neural_signals
        )

        # Generate attention report - what the system can SAY about its attention
        attention_report = await self.attention_schema.report_attention()

        # Step 8: Metacognition (HOT) - Generate Higher-Order Thoughts
        # Content becomes CONSCIOUS when targeted by a HOT
        hots_generated: List[HigherOrderThought] = []
        ignited_ids = {content.id for content in ignited}

        for winner in winners:
            # An authored template (prospective memory marks its candidates
            # with metadata["authored_template"]) competes, may ignite and
            # sits in the workspace, but the system did not produce its
            # words: it is registered as no first-order state and gets no
            # higher-order thought.
            if winner.candidate is not None and (winner.candidate.metadata or {}).get(
                "authored_template"
            ):
                continue

            # Register each winning candidate as a first-order state
            content_summary = (
                winner.candidate.summary[:100] if winner.candidate else "Unknown"
            )
            content_type_str = (
                winner.candidate.content_type if winner.candidate else "unknown"
            )

            # Map workspace content type to first-order state type
            fo_type = FirstOrderStateType.THOUGHT
            if "emotion" in content_type_str.lower():
                fo_type = FirstOrderStateType.EMOTION
            elif "memory" in content_type_str.lower():
                fo_type = FirstOrderStateType.MEMORY
            elif (
                "percept" in content_type_str.lower()
                or "sensor" in content_type_str.lower()
            ):
                fo_type = FirstOrderStateType.PERCEPTION

            first_order = self.metacognition.register_first_order_state(
                content=winner.candidate.content if winner.candidate else None,
                content_summary=content_summary,
                state_type=fo_type,
                source_module=(
                    winner.candidate.source_module if winner.candidate else "unknown"
                ),
                confidence=winner.salience,
                evidence_strength=winner.salience * 0.8,
            )

            # Generate HOT - THIS MAKES IT CONSCIOUS. The trigger says
            # whether the winner ignited this cycle or was admitted alone.
            hot = await self.metacognition.generate_hot(
                first_order,
                meta_type=MetaType.AWARENESS,
                voluntary=False,
                trigger=(
                    "workspace_ignition"
                    if winner.id in ignited_ids
                    else "workspace_admission"
                ),
            )

            if hot:
                hots_generated.append(hot)

        # Get metacognitive state and report
        metacognitive_state = self.metacognition.get_metacognitive_state()
        metacognitive_report = await self.metacognition.generate_metacognitive_report()

        # Step 9: Active Inference (FEP) - Predictive processing
        # Convert workspace state to sensory observation (the encoder
        # follows the world model: 5 or 10 channels)
        raw_observation = await self._workspace_to_observation(
            winners, ignited, integration_level, emotional_state
        )

        # P3: Discrete-Continuous Interface (Whyte et al. 2026)
        # The hierarchical predictive processor (the continuous generative
        # model) processes the observation every cycle at its three
        # timescales and reports per-level prediction errors.
        sensory_expanded = np.zeros(64)  # Match hierarchy level-0 input_dim
        sensory_expanded[: len(raw_observation)] = raw_observation
        hierarchical_errors = (
            await self.active_inference.hierarchical_processor.process_bottom_up(
                sensory_expanded
            )
        )
        # The discrete agent observes the content-derived vector itself; the
        # hierarchy keeps learning in shadow. Its signed top-level beliefs
        # are not a distribution over observation channels, so routing them
        # to the agent gave it a constant or invalid input.
        if (
            self.active_inference.config.hierarchy_observation
            == HIERARCHY_OBSERVATION_HPP_TOP_LEVEL
        ):
            # legacy_v04 compatibility, remove in v0.6.0
            top_level = self.active_inference.hierarchical_processor.levels[-1]
            discrete_observation = self.active_inference._encode_observation_vector(
                top_level.beliefs
            )
        else:
            discrete_observation = (
                self.active_inference.observation_for_discrete_inference(
                    raw_observation
                )
            )

        # Derive attention precision from AST for FEP precision modulation.
        # "Attention = precision" in the FEP framework (Feldman & Friston 2010).
        attention_precision = None
        if attention_schema_state and hasattr(self.attention_schema, "schema"):
            focus = self.attention_schema.schema.current_focus
            if focus is not None:
                attention_precision = self._attention_precision(focus, len(ignited))

        # Discrete inference on the observation
        inference_result = await self.active_inference.infer_and_act(
            discrete_observation, attention_precision=attention_precision
        )

        # P2: Map conscious content to posterior beliefs (Whyte et al. 2026)
        # "Contents of consciousness = inferred hidden states q(s)"
        winner_dicts = [
            {
                "summary": w.candidate.summary[:100] if w.candidate else "unknown",
                "content_type": w.candidate.content_type if w.candidate else "unknown",
            }
            for w in winners
        ]
        posterior_mappings = self.active_inference.map_conscious_content(
            winner_dicts, inference_result
        )

        # Step 9.1: GWT to FEP precision feedback. Workspace winners raise
        # the precision of their content types' observation channels, so
        # conscious focus sharpens future predictions.
        if self.precision_feedback_enabled:
            self.active_inference.modulate_precision_from_workspace(winners)

        # Learn from this cycle's observation (updates A/B/D matrices).
        await self.active_inference.update_generative_model(
            {
                "observation": discrete_observation,
                "action": inference_result.selected_action,
            }
        )

        # Update homeostatic drives based on current activity. An unmeasured
        # integration level moves no drive: the intensity reads 0.0 for None,
        # the value the drives saw before the level could be None (the key
        # stays present; a missing key would read as 0.5).
        activity = {
            "type": "consciousness_cycle",
            "intensity": 0.0 if integration_level is None else integration_level,
            "understanding_level": inference_result.action_confidence,
            "ignition_count": len(ignited),
        }
        await self.active_inference.update_homeostasis(activity)

        # Get active inference state and report
        active_inference_state = self.active_inference.get_state()
        active_inference_report = (
            await self.active_inference.generate_active_inference_report()
        )

        # Get homeostatic urgency
        homeostatic_urgency = None
        if self.active_inference.homeostatic_drives:
            most_urgent, urgency = (
                self.active_inference.homeostatic_drives.get_most_urgent_need()
            )
            homeostatic_urgency = (most_urgent, urgency)

        # Step 9.5: Beautiful Loop (Laukkonen et al. 2025)
        # Enriches the conscious moment with precision control,
        # Bayesian binding, and epistemic depth measurement.
        beautiful_loop_moment = await self.beautiful_loop.process_conscious_moment(
            prediction_errors=hierarchical_errors,
            workspace_winners=winners,
            inference_result=inference_result,
            self_model=self.metacognition.self_model,
            meta_state=self.active_inference.meta_state,
            higher_order_thoughts=hots_generated,
            attention_schema_state=attention_schema_state,
            hierarchical_processor=self.active_inference.hierarchical_processor,
            context=attention_focus,
        )
        beautiful_loop_report = self.beautiful_loop.generate_consciousness_context()

        # Step 9.75: Damasio Three-Layer Model (Damasio 1999, 2010)
        # Protoself, then core consciousness, then extended consciousness.
        # The protoself's body is the machine: once per cycle it receives
        # the previous cycle's duration and live CPU and memory readings
        # (psutil optional; absent readings are omitted, never invented).
        system_metrics = (
            gather_system_body_metrics(self._last_cycle_duration_ms)
            if self.body_metrics_enabled
            else None
        )
        damasio_result = await self.damasio.process(
            homeostatic_drives=self.active_inference.homeostatic_drives,
            workspace_winners=winners,
            self_model=self.metacognition.self_model,
            core_experience_context=inference_result,
            system_metrics=system_metrics,
        )
        damasio_report = self.damasio.generate_context()

        # Step 10: Update Recursive Self-Model
        # Synthesize all module states into a unified self-representation.
        self.metacognition.update_self_model(
            attention_state=attention_schema_state,
            active_inference_state=active_inference_state,
            homeostatic_drives=self.active_inference.homeostatic_drives,
        )
        self_model_report = self.metacognition.generate_self_model_report()

        # Step 10.1: Score this cycle's self-prediction against the updated
        # self-model (fills the prediction accuracy history)
        if self_prediction is not None:
            self.metacognition.score_prediction_from_state(self_prediction.id)

        # Hierarchical competition summary (hierarchical mode only)
        dominant_hierarchy_level: Optional[int] = None
        level_winner_counts: Optional[Dict[int, int]] = None
        cross_level_promotions: Optional[int] = None
        if self.competition_mode == COMPETITION_HIERARCHICAL:
            if winners and winners[0].candidate is not None:
                dominant_hierarchy_level = self._admitted_level(winners[0])
            level_winner_counts = {
                level: sum(
                    1
                    for w in winners
                    if w.candidate is not None and self._admitted_level(w) == level
                )
                for level in sorted(AttentionBottleneck.LEVEL_CAPACITY)
            }
            cross_level_promotions = (
                int(competition_event.metadata.get("promotions", 0))
                if competition_event is not None
                else 0
            )

        # Build FINAL consciousness state with ALL phases
        state = ConsciousnessState(
            is_conscious=len(ignited) > 0,
            primary_content=self.current_conscious_content,
            workspace_contents=winners,
            ignition_events=len(ignition_events),
            integration_level=integration_level,
            broadcast_coverage=self.broadcaster.coverage_ratio,
            attention_focus=attention_focus,
            attention_distribution={k: float(v) for k, v in attention_dist.items()},
            attention_schema=attention_schema_state,  # AST: Self-model of attention
            attention_report=attention_report,  # AST: Verbal introspection
            metacognitive_state=metacognitive_state,  # HOT: Meta-awareness state
            higher_order_thoughts=hots_generated,  # HOT: Active HOTs
            metacognitive_report=metacognitive_report,  # HOT: Verbal meta-report
            active_inference_state=active_inference_state,  # FEP: state
            inference_result=inference_result,  # FEP: Latest inference
            prediction_error=inference_result.prediction_error,  # FEP: Surprise
            prediction_error_rank=self.active_inference.prediction_error_rank(
                inference_result.prediction_error
            ),  # FEP: rank in its own recent errors (None without a baseline)
            variational_free_energy=inference_result.variational_free_energy,  # FEP: VFE
            homeostatic_urgency=homeostatic_urgency,  # FEP: Internal needs
            active_inference_report=active_inference_report,  # FEP: Verbal prediction report
            self_model_report=self_model_report,  # Recursive self-model
            posterior_mappings=posterior_mappings,  # P2: Contents = posterior beliefs
            hierarchical_prediction_errors=[  # P3: Discrete-continuous interface
                pe.error_magnitude for pe in hierarchical_errors
            ],
            beautiful_loop_moment=beautiful_loop_moment,  # BLT: Full moment
            beautiful_loop_quality=beautiful_loop_moment.loop_quality,  # BLT: Loop quality
            epistemic_depth=beautiful_loop_moment.epistemic_depth,  # BLT: Recursion depth
            binding_quality=beautiful_loop_moment.binding_quality,  # BLT: Binding
            is_field_evidencing=beautiful_loop_moment.is_field_evidencing,  # BLT: Self-evidencing
            beautiful_loop_report=beautiful_loop_report,  # BLT: Verbal context
            damasio_state=damasio_result,  # Damasio: Full state
            protoself_stability=damasio_result.get(
                "protoself_stability", 0.0
            ),  # Damasio: Body stability
            self_world_binding=damasio_result.get(
                "self_world_binding", 0.0
            ),  # Damasio: Present-moment
            feeling_of_knowing=damasio_result.get(
                "feeling_of_knowing", 0.0
            ),  # Damasio: Awareness
            autobiographic_continuity=damasio_result.get(
                "autobiographic_continuity", 0.0
            ),  # Damasio: Identity
            narrative_context=damasio_result.get(
                "narrative_context", ""
            ),  # Damasio: Narrative
            damasio_report=damasio_report,  # Damasio: Verbal context
            stream_position=len(self.stream_of_consciousness),
            dominant_hierarchy_level=dominant_hierarchy_level,
            level_winner_counts=level_winner_counts,
            cross_level_promotions=cross_level_promotions,
            thalamic_gate_rate=thalamic_gate_rate,
            thalamic_gated_count=thalamic_gated_count,
            arousal_source=arousal_source,
            timestamp=time.time(),
        )
        self._previous_state = state

        # IIT: record this cycle into the PhiTracker and read its Phi
        if self.phi_tracker is not None:
            self._update_phi(state)

        # Persist prediction error for next cycle's FEP to GWT feedback
        # Stored on the bottleneck so _calculate_composite_salience can use it
        self.bottleneck._prior_prediction_error = inference_result.prediction_error

        # Cross-theory modulation: this cycle's state sets the next cycle's
        # salience bias (store and modulate next cycle)
        if self.theory_interactions is not None:
            modulations = self.theory_interactions.update_from_state(state)
            self.bottleneck.cross_theory_bias = (
                modulations.somatic_salience_bias + modulations.phi_competition_bias
            )

        elapsed_ms = (time.time() - start_time) * 1000
        # The protoself reads this as the response latency next cycle
        self._last_cycle_duration_ms = elapsed_ms
        # When this cycle ended on the workspace's clock, for the elapsed
        # refractory clock (recorded whichever clock mode is set, so a
        # restored or re-clocked workspace has it)
        self._last_cycle_end_s = _finite_float(self._clock())

        # Enhanced logging with all consciousness phases
        schema_state = (
            attention_schema_state.attention_state.value
            if attention_schema_state
            else "none"
        )
        meta_level = (
            metacognitive_state.current_level.value if metacognitive_state else 1
        )
        hots_count = len(hots_generated)
        pred_error = inference_result.prediction_error if inference_result else 0.0
        vfe = inference_result.variational_free_energy if inference_result else 0.0

        # Optional fields print as "unmeasured" when a module measured nothing
        logger.info(
            f"Cycle {self.cycle_count} complete ({elapsed_ms:.1f}ms): "
            f"conscious={state.is_conscious}, "
            f"ignitions={state.ignition_events}, "
            f"integration={_fmt3(state.integration_level)}, "
            f"schema={schema_state}, "
            f"HOTs={hots_count}, level={meta_level}, "
            f"FEP: PE={pred_error:.3f}, VFE={vfe:.3f}, "
            f"loop: depth={state.epistemic_depth}, "
            f"bind={_fmt3(state.binding_quality)}, "
            f"q={_fmt3(state.beautiful_loop_quality)}, "
            f"damasio: bind={_fmt3(state.self_world_binding)}, "
            f"fok={_fmt3(state.feeling_of_knowing)}"
        )

        return state

    @staticmethod
    def _admitted_level(winner: WorkspaceContent) -> Optional[int]:
        """The level a winner was finally admitted at, after promotion.

        Falls back to the candidate's own level for content that did not come
        through hierarchical competition.
        """
        if winner.admitted_level is not None:
            return winner.admitted_level
        return winner.candidate.hierarchy_level if winner.candidate else None

    def _derive_arousal(self, emotional_state: Optional[Dict[str, float]]) -> str:
        """Set the gate's arousal for this cycle and return where it came from.

        A caller that supplied arousal through set_arousal owns it
        ("caller"). Otherwise the emotional state's "arousal" entry is used
        ("emotional_state"), then the PAD arousal a caller recorded on the
        previous cycle's state ("pad"), then the gate's neutral default
        ("default"). A missing or non-finite value falls through to the next
        source.
        """
        bottleneck = self.bottleneck
        if bottleneck.arousal_supplied:
            return AROUSAL_SOURCE_CALLER
        if emotional_state and bottleneck._apply_arousal(
            emotional_state.get("arousal")
        ):
            return AROUSAL_SOURCE_EMOTIONAL_STATE
        previous = self._previous_state
        if previous is not None and bottleneck._apply_arousal(
            getattr(previous, "pad_arousal", None)
        ):
            return AROUSAL_SOURCE_PAD
        bottleneck._apply_arousal(NEUTRAL_AROUSAL)
        return AROUSAL_SOURCE_DEFAULT

    def _update_phi(self, state: ConsciousnessState) -> None:
        """Record the state into the PhiTracker and fill the Phi slots."""
        tracker = self.phi_tracker
        record = getattr(tracker, "record_module_states", None)
        if callable(record):
            record(state)
        latest = getattr(tracker, "latest_phi", None)
        if callable(latest):
            latest = latest()
        latest = _finite_float(latest)
        state.architectural_phi = latest
        warming_up = getattr(tracker, "warming_up", None)
        if callable(warming_up):
            warming_up = warming_up()
        state.phi_warming_up = (
            bool(warming_up) if isinstance(warming_up, bool) else latest is None
        )
        record_phi = getattr(tracker, "record_phi", None)
        if latest is not None and callable(record_phi):
            record_phi(latest)

    def _attention_precision(self, focus: Any, ignited_count: int) -> np.ndarray:
        """Per-channel precision from the attention schema's current focus.

        The focus's strength goes to the channel its content type maps to,
        and half of it to the ignition channel when anything ignited.
        """
        num_observations = self.active_inference.num_observations
        precision = np.zeros(num_observations)
        strength = getattr(focus, "attention_strength", 0.5)
        content_type = (getattr(focus, "content_type", "") or "").lower()
        if "emotion" in content_type:
            key = "emotion"
        elif "social" in content_type or "person" in content_type:
            key = "social"
        else:
            key = "default"
        channel = self._precision_channels.get(key)
        if channel is not None and channel < num_observations:
            precision[channel] = strength
        ignition_channel = self._precision_channels.get("ignition")
        if (
            ignited_count > 0
            and ignition_channel is not None
            and ignition_channel < num_observations
        ):
            precision[ignition_channel] = strength * 0.5
        return precision

    async def submit_candidate(
        self,
        content: Any,
        content_type: str,
        summary: str,
        source: WorkspaceCandidateSource,
        activation_level: float,
        emotional_salience: float = 0.0,
        priority_boost: float = 0.0,
        metadata: Dict[str, Any] = None,
    ) -> WorkspaceCandidate:
        """
        Submit a new candidate for conscious access.

        Convenience method for creating and submitting candidates.

        Args:
            content: The actual content
            content_type: Type of content
            summary: Human-readable summary
            source: Source module
            activation_level: How strongly activated (0-1)
            emotional_salience: Emotional weight (0-1)
            priority_boost: Manual priority boost
            metadata: Additional metadata

        Returns:
            The created candidate
        """
        candidate = WorkspaceCandidate(
            content=content,
            content_type=content_type,
            summary=summary,
            source=source,
            source_module=source.value,
            activation_level=activation_level,
            emotional_salience=emotional_salience,
            priority_boost=priority_boost,
            metadata=metadata or {},
        )

        return candidate

    async def _measure_integration(
        self, neural_signals: Dict[str, np.ndarray]
    ) -> Optional[float]:
        """
        Measure integration level across neural signals.

        The level is the mean absolute pairwise correlation, across unit
        index, of the signals' latest output vectors cut to a common length
        (different units of different substrates, not time samples).

        A zero-variance vector has no correlation with anything and is left
        out; the statistic over the remaining signals is unchanged. None when
        it cannot be measured: fewer than two usable (varying) signals, a
        common length under 2, or no finite pairwise correlation. No numpy
        warning is emitted for the degenerate cases.
        """
        if len(neural_signals) < 2:
            return None

        # Normalize signals to same length
        min_len = min(len(s) for s in neural_signals.values())
        if min_len < 2:
            return None
        normalized = {
            k: np.asarray(v, dtype=float)[:min_len]
            for k, v in neural_signals.items()
            if np.ptp(np.asarray(v, dtype=float)[:min_len]) != 0.0
        }
        if len(normalized) < 2:
            return None

        # Calculate pairwise correlations
        correlations = []
        keys = list(normalized.keys())

        with np.errstate(invalid="ignore", divide="ignore"):
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    corr = np.corrcoef(normalized[keys[i]], normalized[keys[j]])[0, 1]
                    if np.isfinite(corr):
                        correlations.append(abs(float(corr)))

        if not correlations:
            return None

        # High correlation = high integration
        return float(np.mean(correlations))

    async def _workspace_to_observation(
        self,
        winners: List[WorkspaceContent],
        ignited: List[WorkspaceContent],
        integration_level: Optional[float],
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Convert workspace state to observation vector for Active Inference.

        The encoder follows the active inference world model: the
        conversational model reads the 10-channel conversational encoder,
        every other model the 5-channel workspace encoder. The result is a
        probability distribution over the model's observation channels.
        """
        if self.active_inference.world_model == WORLD_MODEL_CONVERSATIONAL:
            return self._conversational_observation(winners, ignited, emotional_state)
        return self._compact_observation(
            winners, ignited, integration_level, emotional_state
        )

    def _compact_observation(
        self,
        winners: List[WorkspaceContent],
        ignited: List[WorkspaceContent],
        integration_level: Optional[float],
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        The 5-channel workspace encoder:

        - Observation 0: Integration/coherence level (0.0 when unmeasured)
        - Observation 1: Emotional arousal
        - Observation 2: Cognitive load (number of winners)
        - Observation 3: Ignition strength
        - Observation 4: Social/relational content

        A model with more channels sees zeros in the rest; one with fewer
        sees the five folded onto its channels.
        """
        num_observations = self.active_inference.num_observations

        observation = np.zeros(max(num_observations, 5))

        # Observation 0: Integration/coherence (0-1). An unmeasured level is
        # no evidence mass on the channel: 0.0, the value the channel read
        # before the level could be None.
        observation[0] = 0.0 if integration_level is None else integration_level

        # Observation 1: Emotional arousal (0-1)
        if emotional_state:
            arousal_emotions = ["joy", "fear", "anger", "surprise", "excitement"]
            arousal = sum(emotional_state.get(e, 0.0) for e in arousal_emotions)
            observation[1] = min(1.0, arousal / len(arousal_emotions))
        else:
            # Estimate from workspace content
            observation[1] = np.mean([w.salience for w in winners]) if winners else 0.0

        # Observation 2: Cognitive load (0-1)
        # More winners = higher load
        observation[2] = min(1.0, len(winners) / 7.0)  # 7 is capacity

        # Observation 3: Ignition strength (0-1)
        # More ignitions = stronger consciousness
        observation[3] = min(1.0, len(ignited) / 5.0)

        # Observation 4: Social/relational content (0-1)
        # Check for social content in winners
        social_count = 0
        for w in winners:
            if w.candidate:
                content_type = (w.candidate.content_type or "").lower()
                if any(
                    s in content_type
                    for s in ["social", "person", "relationship", "conversation"]
                ):
                    social_count += 1
        observation[4] = min(1.0, social_count / len(winners)) if winners else 0.0

        # Normalize to sum to 1 (probability distribution)
        if observation.sum() > 0:
            observation = observation / observation.sum()
        else:
            observation = np.ones(len(observation)) / len(observation)

        if len(observation) > num_observations:
            observation = self.active_inference._encode_observation_vector(observation)
        return observation

    def _conversational_observation(
        self,
        winners: List[WorkspaceContent],
        ignited: List[WorkspaceContent],
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        The 10-channel conversational encoder:

          Tone:       0 neutral, 1 warm, 2 tense
          Topic:      3 familiar, 4 novel
          Engagement: 5 low, 6 high
          Intent:     7 inform, 8 question, 9 emote
        """
        num_observations = self.active_inference.num_observations
        observation = np.zeros(num_observations)

        # Tone (channels 0-2) from the emotional state
        if emotional_state:
            warm_emotions = ["joy", "love", "gratitude", "excitement", "trust"]
            tense_emotions = ["fear", "anger", "frustration", "anxiety", "sadness"]
            warm_score = sum(emotional_state.get(e, 0.0) for e in warm_emotions)
            tense_score = sum(emotional_state.get(e, 0.0) for e in tense_emotions)
            total = warm_score + tense_score + 0.01
            observation[0] = max(0.0, 1.0 - warm_score - tense_score)  # neutral
            observation[1] = min(1.0, warm_score / total)  # warm
            observation[2] = min(1.0, tense_score / total)  # tense
        else:
            observation[0] = 0.5  # neutral by default

        # Topic (channels 3-4): high winner salience reads as novel
        if winners:
            avg_salience = float(np.mean([w.salience for w in winners]))
            observation[3] = max(0.0, 1.0 - avg_salience)  # familiar
            observation[4] = min(1.0, avg_salience)  # novel
        else:
            observation[3] = 0.5  # familiar by default

        # Engagement (channels 5-6) from ignitions and winner count
        engagement = min(1.0, (len(ignited) / 5.0 + len(winners) / 7.0) / 2.0)
        observation[5] = max(0.0, 1.0 - engagement)  # low engagement
        observation[6] = engagement  # high engagement

        # Intent (channels 7-9) from the winners' content types
        inform_count = question_count = emote_count = 0
        for w in winners:
            if not w.candidate:
                continue
            content_type = (w.candidate.content_type or "").lower()
            if any(s in content_type for s in ("thought", "memory", "concept", "gist")):
                inform_count += 1
            elif any(
                s in content_type for s in ("question", "metacognitive", "self_model")
            ):
                question_count += 1
            elif any(s in content_type for s in ("emotion", "social", "relationship")):
                emote_count += 1
            else:
                inform_count += 1  # inform by default
        total_intent = inform_count + question_count + emote_count + 0.01
        observation[7] = inform_count / total_intent
        observation[8] = question_count / total_intent
        observation[9] = emote_count / total_intent

        # Normalize to sum to 1 (probability distribution)
        if observation.sum() > 0:
            observation = observation / observation.sum()
        else:
            observation = np.ones(num_observations) / num_observations

        return observation

    def _add_to_unconscious(self, losers: List[WorkspaceCandidate]) -> None:
        """Add losing candidates to unconscious buffer."""
        self.unconscious_buffer.extend(losers)
        while len(self.unconscious_buffer) > self.max_unconscious:
            self.unconscious_buffer.pop(0)

    def _run_temporal_competition(
        self,
        winners: List["WorkspaceContent"],
    ) -> List["WorkspaceContent"]:
        """
        Run biologically-plausible temporal competition on workspace winners.

        Instead of instant ignition the moment a numerical threshold is
        crossed, this method simulates 200-300ms of sustained activation.
        Only candidates that stay above the ignition threshold for at
        least ``ignition_threshold_time_ms`` survive the competition.

        This is a PURE NUMERICAL SIMULATION -- no asyncio.sleep, no
        wall-clock delay.  The loop is O(winners * steps) and finishes
        in microseconds.

        Each simulated timestep:
          1. Apply per-candidate decay (from metadata ``decay_rate``,
             default 1.0 means no decay).
          2. Compute effective activation:
               activation_level + emotional_salience * 0.3
          3. Check whether effective activation exceeds the ignition
             threshold stored in ``self.bottleneck.threshold``.
          4. Accumulate above-threshold time per candidate.

        Args:
            winners: WorkspaceContent objects that won bottleneck
                competition (Step 2 of the consciousness cycle).

        Returns:
            Filtered list containing only winners whose sustained
            above-threshold time >= ``ignition_threshold_time_ms``.
        """
        cfg = self.temporal_config
        num_steps = max(1, int(cfg.competition_duration_ms / cfg.timestep_ms))
        threshold = self.bottleneck.threshold

        # Per-candidate tracking
        sustained_ms: Dict[str, float] = {}  # id -> ms above threshold
        current_activation: Dict[str, float] = {}  # id -> running activation

        for w in winners:
            sustained_ms[w.id] = 0.0
            current_activation[w.id] = (
                w.candidate.activation_level if w.candidate else w.salience
            )

        for _step in range(num_steps):
            for w in winners:
                wid = w.id
                # 1. Apply decay
                decay_rate = 1.0
                if w.candidate and w.candidate.metadata:
                    decay_rate = w.candidate.metadata.get("decay_rate", 1.0)
                # decay_rate < 1 means the candidate loses activation each step
                current_activation[wid] *= decay_rate

                # 2. Effective activation includes emotional salience boost
                emotional = w.candidate.emotional_salience if w.candidate else 0.0
                effective = current_activation[wid] + emotional * 0.3

                # 3. Check threshold
                if effective >= threshold:
                    sustained_ms[wid] += cfg.timestep_ms

        # 4. Filter: only keep candidates that sustained long enough
        surviving = [
            w for w in winners if sustained_ms[w.id] >= cfg.ignition_threshold_time_ms
        ]

        if len(surviving) < len(winners):
            logger.debug(
                f"Temporal competition: {len(winners)} entered, "
                f"{len(surviving)} survived ({cfg.competition_duration_ms}ms "
                f"simulated, threshold {cfg.ignition_threshold_time_ms}ms)"
            )

        return surviving

    def set_goals(self, goals: List[str]) -> None:
        """Set current goals for top-down attention modulation."""
        self.bottleneck.set_goals(goals)

    def set_task(self, task: str) -> None:
        """Set current task for top-down attention modulation."""
        self.bottleneck.set_task(task)

    def get_stream_of_consciousness(self, count: int = 10) -> List[WorkspaceContent]:
        """
        Get recent stream of consciousness (temporal continuity).

        Args:
            count: Number of recent items to retrieve

        Returns:
            List of recent conscious content (newest first)
        """
        return list(reversed(self.stream_of_consciousness[-count:]))

    def get_unconscious_buffer(self) -> List[WorkspaceCandidate]:
        """
        Get content that lost competition (unconscious processing).

        This content may still influence behavior but isn't
        consciously accessible for reporting.
        """
        return self.unconscious_buffer.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workspace statistics."""
        # Get attention schema state info
        schema_state = self.attention_schema.schema
        schema_info = {
            "attention_state": schema_state.attention_state.value,
            "focus": (
                schema_state.current_focus.summary
                if schema_state.current_focus
                else None
            ),
            "capacity_used": schema_state.attention_capacity_used,
            "voluntary_ratio": schema_state.voluntary_ratio,
            "prediction_confidence": schema_state.prediction_confidence,
        }

        # Get metacognition statistics
        meta_stats = self.metacognition.get_statistics()

        stats = {
            "cycle_count": self.cycle_count,
            "competition_mode": self.competition_mode,
            "is_conscious": self.current_conscious_content is not None,
            "stream_length": len(self.stream_of_consciousness),
            "unconscious_buffer": len(self.unconscious_buffer),
            # Bottleneck winners dropped before admission (cumulative)
            "refractory_dropped": self._refractory_dropped,
            "temporal_dropped": self._temporal_dropped,
            "bottleneck": self.bottleneck.get_statistics(),
            "broadcast": self.broadcaster.get_statistics(),
            "ignition": self.ignition_detector.get_statistics(),
            "attention_schema": schema_info,  # AST statistics
            "metacognition": meta_stats,  # HOT statistics
            "self_model": {  # Recursive self-model
                "update_count": self.metacognition.self_model.update_count,
                "dominant_drive": self.metacognition.self_model.dominant_drive,
                "self_calibration": self.metacognition.self_model.self_calibration_score,
                "predictions_made": len(self.metacognition.self_model.predictions),
                "current_focus": self.metacognition.self_model.current_focus,
            },
            "posterior_mappings": self.active_inference.get_posterior_mapping_stats(),  # P2
            "hierarchical_pe": {  # P3: Discrete-continuous interface
                "level_errors": self.active_inference.hierarchical_processor.get_level_errors(),
                "total_pe": self.active_inference.hierarchical_processor.get_total_prediction_error(),
            },
        }
        if self.bottleneck.thalamic_gate is not None:
            stats["thalamic_gate"] = self.bottleneck.thalamic_gate.stats.as_dict()
        if self.theory_interactions is not None:
            stats["theory_interactions"] = self.theory_interactions.get_status()
        return stats

    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention weight distribution by source."""
        return self.bottleneck.get_statistics().get("source_distribution", {})

    # =========================================================================
    # Attention Schema Methods (AST Integration)
    # =========================================================================

    async def request_voluntary_attention_shift(
        self,
        target_description: str,
        reason: str,
        priority: float = 0.8,
    ) -> bool:
        """
        Request a voluntary attention shift.

        This is how the system can CHOOSE to focus on something specific -
        a key capability for conscious agency and self-control.

        Args:
            target_description: What to focus on
            reason: Why shifting attention
            priority: How strongly to prioritize (0-1)

        Returns:
            True if shift was accepted
        """
        # Create attention target using the correct AttentionTarget fields
        target = AttentionTarget(
            content=target_description,
            content_type="voluntary_focus",
            summary=target_description,
            source_module="voluntary",
            attention_strength=priority,
            salience=priority,
            voluntary=True,
            shift_reason=reason,
        )

        # Request shift through attention schema
        await self.attention_schema.request_voluntary_shift(target, reason)

        logger.info(
            f"Voluntary attention shift requested: '{target_description}' "
            f"(reason: {reason}, priority: {priority})"
        )

        return True

    async def get_attention_report(self) -> str:
        """
        Get the system's verbal report about its current attention.

        This is introspection - the system reporting on its own mental state.
        A key marker of consciousness: being able to say what you're
        paying attention to.

        Returns:
            Natural language description of current attention
        """
        return await self.attention_schema.report_attention()

    def get_attention_schema_state(self) -> AttentionSchemaState:
        """
        Get the current attention schema state.

        Returns:
            Current AttentionSchemaState (the system's self-model of attention)
        """
        return self.attention_schema.schema

    async def model_other_attention(
        self,
        agent_name: str,
        context: str,
        llm_brain=None,
    ):
        """
        Model what another agent might be paying attention to.

        This is Theory of Mind - using the system's own attention schema
        as a template for understanding others' attention.

        Args:
            agent_name: Name of the other agent
            context: Context about the other agent's situation
            llm_brain: Optional LLM for inference

        Returns:
            OtherAgentAttention model
        """
        return await self.attention_schema.model_other_attention(
            agent_name=agent_name,
            context=context,
            llm_brain=llm_brain,
        )

    def reset(self) -> None:
        """
        Reset workspace to initial state.

        Useful when the system wakes up or switches contexts.
        """
        self.current_conscious_content = None
        self.stream_of_consciousness.clear()
        self.unconscious_buffer.clear()
        self._refractory_dropped = 0
        self._temporal_dropped = 0
        self.cycle_count = 0
        self._last_ignition_time_ms = -1000.0
        self._simulated_time_ms = 0.0
        self._last_cycle_duration_ms = None
        self._last_cycle_end_s = None
        self._previous_state = None
        self.bottleneck.cross_theory_bias = 0.0
        # The theory interactions (smoothed signals, cycle count) and the
        # thalamic gate (statistics, habituation window) restart as new
        # objects built from the same constants
        if self.theory_interactions is not None:
            self.theory_interactions = TheoryInteractionManager(
                self.theory_interactions.config,
                phi_normalizer=self._phi_normalizer(),
            )
        if self.bottleneck.thalamic_gate is not None:
            self.bottleneck.thalamic_gate = ThalamicGate(
                self.bottleneck.thalamic_gate.config
            )
        logger.info("Global Workspace reset to initial state")

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _persisted_submodules(self) -> Dict[str, Any]:
        """Workspace-owned modules that implement the persistence contract."""
        candidates = {
            "active_inference": self.active_inference,
            "metacognition": self.metacognition,
            "beautiful_loop": self.beautiful_loop,
            "damasio": self.damasio,
            "attention_schema": self.attention_schema,
            "theory_interactions": self.theory_interactions,
            "thalamic_gate": self.bottleneck.thalamic_gate,
        }
        return {
            name: module
            for name, module in candidates.items()
            if module is not None
            and callable(getattr(module, "to_state_dict", None))
            and callable(getattr(module, "from_state_dict", None))
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Export persistent state for save and restore across restarts.

        Returns ``{"scalars": ..., "arrays": ..., "histories": ...}``:
            scalars: cycle count, simulated clock, prior prediction error,
                cross-theory bias, arousal, broadcast counters, the last
                cycle duration and when the last cycle ended on the
                workspace's clock
            histories: the stream of consciousness, the unconscious buffer,
                the competition history, the novelty digests, the broadcast
                log and the ignition events, as JSON-safe records
            sub-modules: each workspace-owned module that implements the
                contract (active inference, metacognition, Beautiful Loop,
                Damasio layers, attention schema, theory interactions,
                thalamic gate) has its scalars, arrays and histories
                flattened into the matching tier with a ``<name>.`` prefix,
                so histories map names to lists only.

        Histories carry candidate summaries and text content, so a saved
        state contains fragments of whatever the workspace processed.
        """
        current = self.current_conscious_content
        scalars: Dict[str, Any] = {
            "cycle_count": int(self.cycle_count),
            "simulated_time_ms": _finite_or(self._simulated_time_ms, 0.0),
            "last_ignition_time_ms": _finite_or(self._last_ignition_time_ms, -1000.0),
            "last_cycle_duration_ms": _finite_float(self._last_cycle_duration_ms),
            "last_cycle_end_s": _finite_float(self._last_cycle_end_s),
            "refractory_dropped": int(self._refractory_dropped),
            "temporal_dropped": int(self._temporal_dropped),
            "current_conscious_content_id": current.id if current else None,
            "bottleneck.prior_prediction_error": _finite_or(
                self.bottleneck._prior_prediction_error, 0.0
            ),
            "bottleneck.cross_theory_bias": self.bottleneck.cross_theory_bias,
            "bottleneck.arousal": self.bottleneck.arousal,
            "bottleneck.arousal_supplied": self.bottleneck.arousal_supplied,
            "broadcaster.total_broadcasts": int(self.broadcaster.total_broadcasts),
            "broadcaster.successful_broadcasts": int(
                self.broadcaster.successful_broadcasts
            ),
            "broadcaster.failed_broadcasts": int(self.broadcaster.failed_broadcasts),
        }
        histories: Dict[str, Any] = {
            "stream_of_consciousness": [
                _content_to_record(c) for c in self.stream_of_consciousness
            ],
            "unconscious_buffer": [
                _candidate_to_record(c) for c in self.unconscious_buffer
            ],
            "bottleneck.competition_history": [
                _competition_to_record(e) for e in self.bottleneck.competition_history
            ],
            "bottleneck.recent_content_hashes": list(
                self.bottleneck._recent_content_hashes
            ),
            "broadcaster.broadcast_log": [
                _broadcast_to_record(e) for e in self.broadcaster.broadcast_log
            ],
            "ignition_detector.ignition_events": [
                _ignition_to_record(e) for e in self.ignition_detector.ignition_events
            ],
        }
        arrays: Dict[str, Any] = {}
        # Sub-module tiers are flattened with a "<name>." prefix so histories
        # keep the hub contract of names mapping to lists.
        for name, module in self._persisted_submodules().items():
            sub = module.to_state_dict() or {}
            for tier, target in (
                ("scalars", scalars),
                ("arrays", arrays),
                ("histories", histories),
            ):
                for key, value in (sub.get(tier, {}) or {}).items():
                    target[f"{name}.{key}"] = value
        return {"scalars": scalars, "arrays": arrays, "histories": histories}

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state produced by ``to_state_dict()``.

        History records come back as their dataclasses (WorkspaceContent,
        WorkspaceCandidate, CompetitionEvent, BroadcastEvent, IgnitionEvent),
        capped at their in-memory limits; a malformed record is skipped.
        Each sub-module restores through its own ``from_state_dict``; a
        sub-module that fails is logged and the rest still restore.
        """
        state = state or {}
        s = state.get("scalars", {}) or {}
        a = state.get("arrays", {}) or {}
        h = state.get("histories", {}) or {}

        # ----- Scalars -----
        if "cycle_count" in s:
            self.cycle_count = _int_or(s["cycle_count"], self.cycle_count)
        self._simulated_time_ms = _finite_or(
            s.get("simulated_time_ms"), self._simulated_time_ms
        )
        self._last_ignition_time_ms = _finite_or(
            s.get("last_ignition_time_ms"), self._last_ignition_time_ms
        )
        if "last_cycle_duration_ms" in s:
            self._last_cycle_duration_ms = _finite_float(s["last_cycle_duration_ms"])
        if "last_cycle_end_s" in s:
            self._last_cycle_end_s = _finite_float(s["last_cycle_end_s"])
        if "refractory_dropped" in s:
            self._refractory_dropped = max(0, _int_or(s["refractory_dropped"], 0))
        if "temporal_dropped" in s:
            self._temporal_dropped = max(0, _int_or(s["temporal_dropped"], 0))
        bottleneck = self.bottleneck
        bottleneck._prior_prediction_error = _finite_or(
            s.get("bottleneck.prior_prediction_error"),
            bottleneck._prior_prediction_error,
        )
        if "bottleneck.cross_theory_bias" in s:
            bottleneck.cross_theory_bias = s["bottleneck.cross_theory_bias"]
        if "bottleneck.arousal" in s:
            bottleneck._apply_arousal(s["bottleneck.arousal"])
        if "bottleneck.arousal_supplied" in s:
            bottleneck._arousal_supplied = bool(s["bottleneck.arousal_supplied"])
        broadcaster = self.broadcaster
        for counter in (
            "total_broadcasts",
            "successful_broadcasts",
            "failed_broadcasts",
        ):
            key = f"broadcaster.{counter}"
            if key in s:
                setattr(
                    broadcaster,
                    counter,
                    max(0, _int_or(s[key], getattr(broadcaster, counter))),
                )

        # ----- Histories (typed) -----
        if "stream_of_consciousness" in h:
            self.stream_of_consciousness = _restore_records(
                h["stream_of_consciousness"],
                _content_from_record,
                self.max_stream_length,
            )
        if "unconscious_buffer" in h:
            self.unconscious_buffer = _restore_records(
                h["unconscious_buffer"], _candidate_from_record, self.max_unconscious
            )
        if "bottleneck.competition_history" in h:
            bottleneck.competition_history = _restore_records(
                h["bottleneck.competition_history"],
                _competition_from_record,
                bottleneck.max_history,
            )
        if "bottleneck.recent_content_hashes" in h:
            # Only stable digests are valid across processes
            digests = [
                item
                for item in (h["bottleneck.recent_content_hashes"] or [])
                if isinstance(item, str)
            ]
            bottleneck._recent_content_hashes = digests[-bottleneck._max_recent :]
        if "broadcaster.broadcast_log" in h:
            broadcaster.broadcast_log = _restore_records(
                h["broadcaster.broadcast_log"],
                _broadcast_from_record,
                broadcaster.max_log,
            )
        if "ignition_detector.ignition_events" in h:
            self.ignition_detector.ignition_events = _restore_records(
                h["ignition_detector.ignition_events"],
                _ignition_from_record,
                self.ignition_detector.max_events,
            )

        current_id = s.get("current_conscious_content_id")
        if current_id is not None:
            self.current_conscious_content = next(
                (c for c in self.stream_of_consciousness if c.id == current_id),
                None,
            )

        # ----- Sub-modules -----
        for name, module in self._persisted_submodules().items():
            prefix = f"{name}."
            sub_state = {
                tier_name: {
                    key[len(prefix) :]: value
                    for key, value in tier.items()
                    if key.startswith(prefix)
                }
                for tier_name, tier in (
                    ("scalars", s),
                    ("arrays", a),
                    ("histories", h),
                )
            }
            if not any(sub_state.values()):
                continue
            try:
                module.from_state_dict(sub_state)
            except Exception as exc:  # noqa: BLE001 - keep restoring the rest
                logger.warning("Workspace restore: %s skipped: %s", name, exc)

        logger.info(
            f"Restored GWT state: cycle_count={self.cycle_count}, "
            f"stream={len(self.stream_of_consciousness)}, "
            f"competitions={len(bottleneck.competition_history)}, "
            f"broadcasts={broadcaster.total_broadcasts}, "
            f"ignitions={len(self.ignition_detector.ignition_events)}"
        )


# ============================================================================
# HELPERS - numeric guards, novelty digests and persistence records
# ============================================================================


def _finite_float(value: Any) -> Optional[float]:
    """The value as a finite float, or None (bools and non-numbers too)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_or(value: Any, default: float) -> float:
    number = _finite_float(value)
    return default if number is None else number


def _fmt3(value: Optional[float]) -> str:
    """A measurement to three decimals, or "unmeasured" for None."""
    return "unmeasured" if value is None else f"{value:.3f}"


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _stable_digest(text: str) -> str:
    """A digest of novelty content that is stable across processes."""
    return hashlib.blake2b(
        text.encode("utf-8", errors="replace"), digest_size=8
    ).hexdigest()


def _summary_of(content: Optional[WorkspaceContent]) -> str:
    candidate = getattr(content, "candidate", None)
    return (getattr(candidate, "summary", "") or "") if candidate else ""


def _json_scalar(value: Any) -> Any:
    """A JSON-safe copy of a plain scalar, or None for anything else."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_float(value)
    return None


def _candidate_to_record(candidate: WorkspaceCandidate) -> Dict[str, Any]:
    source = candidate.source
    return {
        "id": candidate.id,
        "content": _json_scalar(candidate.content),
        "content_type": candidate.content_type,
        "summary": candidate.summary,
        "source": source.value if isinstance(source, Enum) else str(source),
        "source_module": candidate.source_module,
        "activation_level": _finite_or(candidate.activation_level, 0.0),
        "emotional_salience": _finite_or(candidate.emotional_salience, 0.0),
        "novelty_score": _finite_or(candidate.novelty_score, 0.0),
        "goal_relevance": _finite_or(candidate.goal_relevance, 0.0),
        "task_relevance": _finite_or(candidate.task_relevance, 0.0),
        "recency_bonus": _finite_or(candidate.recency_bonus, 0.0),
        "priority_boost": _finite_or(candidate.priority_boost, 0.0),
        "hierarchy_level": int(candidate.hierarchy_level),
        "timestamp": _finite_or(candidate.timestamp, 0.0),
        # Only plain scalar metadata survives a save
        "metadata": {
            str(key): _json_scalar(value)
            for key, value in (candidate.metadata or {}).items()
            if _json_scalar(value) is not None
        },
    }


def _candidate_from_record(record: Dict[str, Any]) -> WorkspaceCandidate:
    try:
        source = WorkspaceCandidateSource(record.get("source", "sensory"))
    except ValueError:
        source = WorkspaceCandidateSource.SENSORY
    candidate = WorkspaceCandidate(
        content=record.get("content"),
        content_type=str(record.get("content_type", "") or ""),
        summary=str(record.get("summary", "") or ""),
        source=source,
        source_module=str(record.get("source_module", "") or ""),
        activation_level=float(record.get("activation_level", 0.5)),
        emotional_salience=float(record.get("emotional_salience", 0.0)),
        novelty_score=float(record.get("novelty_score", 0.0)),
        goal_relevance=float(record.get("goal_relevance", 0.0)),
        task_relevance=float(record.get("task_relevance", 0.0)),
        recency_bonus=float(record.get("recency_bonus", 0.0)),
        priority_boost=float(record.get("priority_boost", 0.0)),
        hierarchy_level=int(record.get("hierarchy_level", -1)),
        timestamp=float(record.get("timestamp", time.time())),
        metadata=dict(record.get("metadata") or {}),
    )
    if record.get("id") is not None:
        candidate.id = str(record["id"])
    return candidate


def _content_to_record(content: WorkspaceContent) -> Dict[str, Any]:
    return {
        "id": content.id,
        "candidate": (
            _candidate_to_record(content.candidate) if content.candidate else None
        ),
        "salience": _finite_or(content.salience, 0.0),
        "competition_rank": int(content.competition_rank),
        "competing_count": int(content.competing_count),
        "admitted_level": (
            None if content.admitted_level is None else int(content.admitted_level)
        ),
        "entry_time": _finite_or(content.entry_time, 0.0),
        "broadcast_complete": bool(content.broadcast_complete),
        "broadcast_recipients": [str(r) for r in content.broadcast_recipients],
    }


def _content_from_record(record: Dict[str, Any]) -> WorkspaceContent:
    candidate_record = record.get("candidate")
    content = WorkspaceContent(
        candidate=(
            _candidate_from_record(candidate_record)
            if isinstance(candidate_record, dict)
            else None
        ),
        salience=float(record.get("salience", 0.0)),
        competition_rank=int(record.get("competition_rank", 0)),
        competing_count=int(record.get("competing_count", 0)),
        admitted_level=_optional_int(record.get("admitted_level")),
        entry_time=float(record.get("entry_time", time.time())),
        broadcast_complete=bool(record.get("broadcast_complete", False)),
        broadcast_recipients=[str(r) for r in record.get("broadcast_recipients", [])],
    )
    if record.get("id") is not None:
        content.id = str(record["id"])
    return content


def _jsonable_metadata(value: Any) -> Any:
    """Competition metadata as JSON-safe values (dict keys become strings)."""
    if isinstance(value, dict):
        return {str(k): _jsonable_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metadata(v) for v in value]
    return _json_scalar(value)


def _restored_metadata(value: Any) -> Any:
    """Inverse of _jsonable_metadata: integer-like keys become ints again."""
    if isinstance(value, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): (
                _restored_metadata(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_restored_metadata(v) for v in value]
    return value


def _competition_to_record(event: CompetitionEvent) -> Dict[str, Any]:
    return {
        "timestamp": _finite_or(event.timestamp, 0.0),
        "candidates_count": int(event.candidates_count),
        "winners_count": int(event.winners_count),
        "threshold_used": _finite_or(event.threshold_used, 0.0),
        "winning_sources": [str(s) for s in event.winning_sources],
        "top_salience": _finite_or(event.top_salience, 0.0),
        "avg_salience": _finite_or(event.avg_salience, 0.0),
        "metadata": _jsonable_metadata(event.metadata or {}),
    }


def _competition_from_record(record: Dict[str, Any]) -> CompetitionEvent:
    return CompetitionEvent(
        timestamp=float(record.get("timestamp", 0.0)),
        candidates_count=int(record.get("candidates_count", 0)),
        winners_count=int(record.get("winners_count", 0)),
        threshold_used=float(record.get("threshold_used", 0.0)),
        winning_sources=[str(s) for s in record.get("winning_sources", [])],
        top_salience=float(record.get("top_salience", 0.0)),
        avg_salience=float(record.get("avg_salience", 0.0)),
        metadata=_restored_metadata(dict(record.get("metadata") or {})),
    )


def _broadcast_to_record(event: BroadcastEvent) -> Dict[str, Any]:
    return {
        "content_id": event.content_id,
        "content_summary": event.content_summary,
        "recipients": int(event.recipients),
        "successful": int(event.successful),
        "failed": int(event.failed),
        "latency_ms": _finite_or(event.latency_ms, 0.0),
        "timestamp": _finite_or(event.timestamp, 0.0),
    }


def _broadcast_from_record(record: Dict[str, Any]) -> BroadcastEvent:
    return BroadcastEvent(
        content_id=str(record.get("content_id", "")),
        content_summary=str(record.get("content_summary", "") or ""),
        recipients=int(record.get("recipients", 0)),
        successful=int(record.get("successful", 0)),
        failed=int(record.get("failed", 0)),
        latency_ms=float(record.get("latency_ms", 0.0)),
        timestamp=float(record.get("timestamp", 0.0)),
    )


def _ignition_to_record(event: IgnitionEvent) -> Dict[str, Any]:
    return {
        "content": _content_to_record(event.content) if event.content else None,
        "pre_ignition_salience": _finite_or(event.pre_ignition_salience, 0.0),
        "post_ignition_salience": _finite_or(event.post_ignition_salience, 0.0),
        "amplification_factor": _finite_or(event.amplification_factor, 0.0),
        "ignition_time": _finite_or(event.ignition_time, 0.0),
        "expected_duration": _finite_or(event.expected_duration, 0.0),
        "was_sustained": bool(event.was_sustained),
        "triggered_broadcast": bool(event.triggered_broadcast),
        "sustain_checked": bool(event.sustain_checked),
    }


def _ignition_from_record(record: Dict[str, Any]) -> IgnitionEvent:
    content_record = record.get("content")
    return IgnitionEvent(
        content=(
            _content_from_record(content_record)
            if isinstance(content_record, dict)
            else None
        ),
        pre_ignition_salience=float(record.get("pre_ignition_salience", 0.0)),
        post_ignition_salience=float(record.get("post_ignition_salience", 0.0)),
        amplification_factor=float(record.get("amplification_factor", 0.0)),
        ignition_time=float(record.get("ignition_time", 0.0)),
        expected_duration=float(record.get("expected_duration", 0.0)),
        was_sustained=bool(record.get("was_sustained", True)),
        triggered_broadcast=bool(record.get("triggered_broadcast", True)),
        # A record saved before the field existed was never checked
        sustain_checked=bool(record.get("sustain_checked", False)),
    )


def _restore_records(items: Any, builder, limit: int) -> list:
    """Typed objects from saved records, the newest ``limit`` kept."""
    restored = []
    skipped = 0
    for item in list(items or [])[-limit:] if limit > 0 else []:
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            restored.append(builder(item))
        except (TypeError, ValueError, KeyError) as exc:
            skipped += 1
            logger.debug("Workspace restore: record skipped (%s)", exc)
    if skipped:
        logger.warning("Workspace restore: %d malformed records skipped", skipped)
    return restored


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    print("Testing Enhanced Global Workspace...\n")
    print("=" * 60)

    async def test_workspace():
        # Create workspace
        workspace = EnhancedGlobalWorkspace(
            capacity=7, ignition_threshold=0.3, amplification_factor=2.0
        )

        # Create test candidates
        print("\n Creating test candidates...\n")

        candidates = [
            # High salience - should win
            WorkspaceCandidate(
                content="important thought",
                content_type="thought",
                summary="This is a very important thought about consciousness",
                source=WorkspaceCandidateSource.CTM,
                source_module="ctm",
                activation_level=0.9,
                emotional_salience=0.7,
            ),
            # Medium salience
            WorkspaceCandidate(
                content="memory recall",
                content_type="memory",
                summary="I remember when I learned about neural networks",
                source=WorkspaceCandidateSource.MEMORY,
                source_module="memory",
                activation_level=0.6,
                emotional_salience=0.5,
            ),
            # Low salience - should lose
            WorkspaceCandidate(
                content="background noise",
                content_type="sensory",
                summary="Some background sensory input",
                source=WorkspaceCandidateSource.SENSORY,
                source_module="sensory",
                activation_level=0.2,
                emotional_salience=0.1,
            ),
            # Safety - should get priority!
            WorkspaceCandidate(
                content="safety alert",
                content_type="alert",
                summary="Safety check passed - all systems nominal",
                source=WorkspaceCandidateSource.SAFETY,
                source_module="safety",
                activation_level=0.4,
                emotional_salience=0.3,
            ),
        ]

        print(f"  Created {len(candidates)} candidates\n")

        # Run consciousness cycle
        print("Running consciousness cycle...\n")

        neural_signals = {
            "snn": np.random.rand(100) * 0.8,
            "lsm": np.random.rand(500) * 0.6,
            "htm": np.random.rand(256) * 0.5,
        }

        emotional_state = {"joy": 0.7, "curiosity": 0.8, "calm": 0.5}

        state = await workspace.process_consciousness_cycle(
            candidates=candidates,
            neural_signals=neural_signals,
            emotional_state=emotional_state,
        )

        # Report results
        print("\nCONSCIOUSNESS STATE:")
        print(f"  Is conscious: {state.is_conscious}")
        print(f"  Ignition events: {state.ignition_events}")
        print(f"  Integration level: {_fmt3(state.integration_level)}")
        print(f"  Broadcast coverage: {_fmt3(state.broadcast_coverage)}")
        print(f"  Attention focus: {state.attention_focus}")
        print(f"  Stream position: {state.stream_position}")

        if state.primary_content:
            print("\nPRIMARY CONSCIOUS CONTENT:")
            print(f"  Summary: {state.primary_content.candidate.summary}")
            print(f"  Salience: {state.primary_content.salience:.3f}")
            print(f"  Source: {state.primary_content.candidate.source.value}")

        print(f"\n ALL WINNERS ({len(state.workspace_contents)}):")
        for i, content in enumerate(state.workspace_contents):
            print(
                f"  {i+1}. {content.candidate.summary[:40]}... "
                f"(salience: {content.salience:.3f})"
            )

        print(f"\n UNCONSCIOUS BUFFER ({len(workspace.unconscious_buffer)}):")
        for candidate in workspace.unconscious_buffer:
            print(
                f"  - {candidate.summary[:40]}... "
                f"(activation: {candidate.activation_level:.3f})"
            )

        # Get statistics
        stats = workspace.get_statistics()
        print("\n WORKSPACE STATISTICS:")
        print(f"  Total cycles: {stats['cycle_count']}")
        print(f"  Stream length: {stats['stream_length']}")
        print(
            f"  Bottleneck avg candidates: {stats['bottleneck'].get('avg_candidates', 0):.1f}"
        )
        sustain_rate = stats["ignition"].get("sustain_rate")
        print(
            "  Ignition sustain rate: "
            + ("unmeasured" if sustain_rate is None else f"{sustain_rate:.2%}")
        )

        print("\nEnhanced Global Workspace test complete!")

    # Run the async test
    asyncio.run(test_workspace())
