"""
Enhanced Global Workspace - Full GWT Implementation
=========================================================

Phase 2 of the Consciousness Upgrade: Complete Global Workspace Theory
implementation with competition, broadcast, and ignition dynamics.

This module transforms the system's consciousness from free-flowing information
to a TRUE competition-based conscious access system.

Key Features:
1. AttentionBottleneck - Limited capacity (7±2 items) with competition
2. GlobalBroadcast - Parallel distribution to ALL cognitive modules
3. IgnitionDetector - Non-linear "ignition" when threshold crossed
4. WorkspaceCandidate - Content competing for conscious access
5. StreamOfConsciousness - Temporal continuity tracking

Based on: Bernard Baars' Global Workspace Theory (1988, 2005)
Research: Butlin et al. (2023) - 14 Consciousness Indicators

Note: Only the most salient signals win the competition for
workspace access, but once they do, the information is broadcast
globally to all cognitive modules.

Created: December 4, 2025
Author: Multi-Theory Consciousness Contributors
"""

import asyncio
import logging
import numpy as np
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import Attention Schema (Phase 3 AST implementation)
from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionSchemaState,
    AttentionTarget,
)

# Import Metacognition (Phase 4 HOT implementation)
from mtc.consciousness.metacognition import (
    MetacognitionModule,
    MetacognitiveState,
    FirstOrderState,
    FirstOrderStateType,
    HigherOrderThought,
    MetaType,
    MetaLevel,
    SelfModel,
)

# Import Active Inference (Phase 5 FEP implementation)
from mtc.consciousness.active_inference import (
    ActiveInferenceModule,
    ActiveInferenceConfig,
    ActiveInferenceState,
    InferenceResult,
    ConsciousPosteriorMapping,
)

# Import Beautiful Loop (Phase H — Laukkonen, Friston & Chandaria 2025)
from mtc.consciousness.beautiful_loop import (
    BeautifulLoop,
    ConsciousMoment,
)

# Import Damasio Three-Layer Model (Phase J — Damasio 1999, 2010)
from mtc.consciousness.damasio import DamasioLayers

logger = logging.getLogger(__name__)


# ============================================================================
# TEMPORAL DYNAMICS CONFIGURATION
# ============================================================================


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
    """

    competition_duration_ms: float = 250.0  # 200-300ms biological range
    ignition_threshold_time_ms: float = 50.0  # Time above threshold to ignite
    broadcast_propagation_ms: float = 30.0  # Per-module delivery delay (unused for now)
    refractory_period_ms: float = 500.0  # Min time between broadcasts
    timestep_ms: float = 10.0  # Simulation resolution


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

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    content: WorkspaceContent

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
    - Phase 2 GWT: Competition, ignition, broadcast
    - Phase 3 AST: Self-model of attention
    - Phase 4 HOT: Higher-order thoughts (metacognition)
    - Phase 5 FEP: Active inference and predictive processing
    """

    is_conscious: bool  # Is there active conscious content?
    primary_content: Optional[WorkspaceContent]  # Main focus
    workspace_contents: List[WorkspaceContent]  # All in workspace

    # Metrics
    ignition_events: int  # Count of ignitions this cycle
    integration_level: float  # How integrated is the content (0-1)
    broadcast_coverage: float  # % of modules that received broadcast

    # Attention state (GWT Phase 2)
    attention_focus: str  # What source is dominating
    attention_distribution: Dict[str, float]  # Weight per source

    # Stream tracking
    stream_position: int  # Position in stream of consciousness

    # Fields with defaults must come after non-defaults (Python dataclass rules)
    # Attention Schema (AST Phase 3) - the system's self-model of attention
    attention_schema: Optional[AttentionSchemaState] = None  # Self-model of attention
    attention_report: Optional[str] = None  # What the system reports about its attention

    # Metacognition (HOT Phase 4) - Higher-order thoughts
    metacognitive_state: Optional[MetacognitiveState] = None  # Meta-awareness state
    higher_order_thoughts: List[HigherOrderThought] = field(
        default_factory=list
    )  # Active HOTs
    metacognitive_report: Optional[str] = None  # What the system reports about its thinking

    # Active Inference (FEP Phase 5) - Predictive processing
    active_inference_state: Optional[ActiveInferenceState] = None  # FEP state
    inference_result: Optional[InferenceResult] = None  # Latest inference
    prediction_error: float = 0.0  # Free energy / surprise
    variational_free_energy: float = 0.0  # VFE from active inference
    homeostatic_urgency: Optional[Tuple[str, float]] = None  # Most urgent need
    active_inference_report: Optional[str] = None  # What the system reports about predictions

    # Recursive Self-Model (Phase 3.4) - Explicit self-representation
    self_model_report: Optional[str] = None  # What the system knows about itself

    # Posterior Mapping (P2) - Contents = posterior beliefs
    posterior_mappings: List[ConsciousPosteriorMapping] = field(default_factory=list)

    # Discrete-Continuous Interface (P3) - Hierarchical prediction errors
    hierarchical_prediction_errors: List[float] = field(default_factory=list)

    # Beautiful Loop (Phase H) - Laukkonen, Friston & Chandaria 2025
    beautiful_loop_moment: Optional[ConsciousMoment] = None
    beautiful_loop_quality: float = 0.0
    epistemic_depth: int = 0
    binding_quality: float = 0.0
    is_field_evidencing: bool = False
    beautiful_loop_report: Optional[str] = None

    # Damasio Three-Layer Model (Phase J) -- protoself, core, extended
    damasio_state: Optional[Dict[str, Any]] = None
    protoself_stability: float = 0.0
    self_world_binding: float = 0.0
    feeling_of_knowing: float = 0.0
    autobiographic_continuity: float = 0.0
    narrative_context: str = ""
    damasio_report: Optional[str] = None

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

    Based on Miller's Law (7±2) for working memory capacity.

    Competition factors:
    1. Bottom-up salience (stimulus strength, novelty) - 40%
    2. Top-down relevance (goal alignment, task relevance) - 30%
    3. Emotional weight (survival/value significance) - 20%
    4. Recency (temporal proximity bonus) - 10%
    """

    def __init__(
        self,
        capacity: int = 7,  # Miller's magic number
        ignition_threshold: float = 0.3,  # Minimum to enter consciousness
        lateral_inhibition: float = 0.2,  # How much winners suppress similar content
    ):
        """
        Initialize the attention bottleneck.

        Args:
            capacity: Maximum items in workspace (default: 7)
            ignition_threshold: Minimum salience to enter consciousness
            lateral_inhibition: Strength of winner-takes-all suppression
        """
        self.capacity = capacity
        self.threshold = ignition_threshold
        self.lateral_inhibition = lateral_inhibition

        # Current contents (what's in the workspace)
        self.current_contents: List[WorkspaceContent] = []

        # Competition history (for research analysis)
        self.competition_history: List[CompetitionEvent] = []
        self.max_history = 1000

        # Current goals and task (for top-down modulation)
        self.current_goals: List[str] = []
        self.current_task: Optional[str] = None

        # Novelty tracking (what's been seen recently)
        self._recent_content_hashes: List[int] = []
        self._max_recent = 100

        logger.info(
            f"Attention Bottleneck initialized "
            f"(capacity={capacity}, threshold={ignition_threshold})"
        )

    async def compete_for_access(
        self, candidates: List[WorkspaceCandidate]
    ) -> List[WorkspaceContent]:
        """
        Run competition among candidates for workspace access.

        This is the CORE of Global Workspace Theory - content must
        COMPETE to become conscious!

        Args:
            candidates: All candidates vying for conscious access

        Returns:
            List of winners (WorkspaceContent) that gained access
        """
        if not candidates:
            return []

        start_time = time.time()

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

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Competition complete: {len(winners)}/{len(candidates)} won "
            f"(top salience: {event.top_salience:.3f}, {elapsed_ms:.1f}ms)"
        )

        return winners

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
        # Goal relevance - does this relate to current goals?
        goal_relevance = await self._calculate_goal_relevance(candidate)
        candidate.goal_relevance = goal_relevance

        # Task relevance - does this help the current task?
        task_relevance = await self._calculate_task_relevance(candidate)
        candidate.task_relevance = task_relevance

        top_down = 0.3 * (0.5 * goal_relevance + 0.5 * task_relevance)

        # EMOTIONAL FACTORS (20%)
        emotional = 0.2 * candidate.emotional_salience

        # TEMPORAL FACTORS (10%)
        recency = self._calculate_recency_bonus(candidate)
        candidate.recency_bonus = recency
        temporal = 0.1 * recency

        # PRIORITY BOOST (additive)
        # Safety-critical content gets priority!
        priority = candidate.priority_boost
        if candidate.source == WorkspaceCandidateSource.SAFETY:
            priority += 0.5  # Safety always gets attention!

        # Compute final salience
        salience = bottom_up + top_down + emotional + temporal + priority

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
        content_hash = hash(content_str)

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

        Goal-relevant content gets top-down attention boost.
        """
        if not self.current_goals:
            return 0.5  # Neutral if no goals set

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
        """
        if not self.current_task:
            return 0.5  # Neutral if no task

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
        """
        age_seconds = time.time() - candidate.timestamp

        # Exponential decay: full bonus at 0s, half at 5s, quarter at 10s
        decay_rate = 0.1  # Decay constant
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

        return {
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

    def __init__(self):
        """Initialize the broadcast system."""
        self.registered_modules: Dict[str, CognitiveModule] = {}
        self.broadcast_log: List[BroadcastEvent] = []
        self.max_log = 1000

        # Statistics
        self.total_broadcasts = 0
        self.successful_broadcasts = 0
        self.failed_broadcasts = 0

        logger.info("📡 Global Broadcast system initialized")

    def register_module(self, module: CognitiveModule) -> None:
        """
        Register a cognitive module to receive broadcasts.

        Args:
            module: Module implementing CognitiveModule interface
        """
        self.registered_modules[module.name] = module
        logger.info(f"📡 Registered broadcast recipient: {module.name}")

    def unregister_module(self, name: str) -> None:
        """Unregister a module from broadcasts."""
        if name in self.registered_modules:
            del self.registered_modules[name]
            logger.info(f"📡 Unregistered broadcast recipient: {name}")

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
            logger.warning("📡 No modules registered for broadcast!")
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
                logger.warning(f"📡 Broadcast to {module_name} failed: {response}")
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
            f"📡 Broadcast complete: {successful}/{len(module_names)} succeeded "
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
            logger.error(f"📡 Error broadcasting to {name}: {e}")
            raise

    def _add_to_log(self, event: BroadcastEvent) -> None:
        """Add event to broadcast log."""
        self.broadcast_log.append(event)
        if len(self.broadcast_log) > self.max_log:
            self.broadcast_log.pop(0)

    @property
    def coverage_ratio(self) -> float:
        """Get broadcast success coverage ratio."""
        if self.total_broadcasts == 0:
            return 1.0
        total = self.successful_broadcasts + self.failed_broadcasts
        if total == 0:
            return 1.0
        return self.successful_broadcasts / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get broadcast statistics."""
        return {
            "registered_modules": list(self.registered_modules.keys()),
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
    ):
        """
        Initialize ignition detector.

        Args:
            ignition_threshold: Minimum salience for ignition
            amplification_factor: How much salience amplifies upon ignition
            sustain_duration: How long content must stay above threshold
            decay_rate: Rate of salience decay after ignition
        """
        self.threshold = ignition_threshold
        self.amplification = amplification_factor
        self.sustain_duration = sustain_duration
        self.decay_rate = decay_rate

        # Tracking
        self.ignition_events: List[IgnitionEvent] = []
        self.pending_ignitions: Dict[str, Tuple[WorkspaceContent, float]] = {}
        self.max_events = 1000

        # Reference to workspace for triggering broadcasts
        self.workspace: Optional["EnhancedGlobalWorkspace"] = None

        logger.info(
            f"🔥 Ignition Detector initialized "
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
        # Check if below threshold
        if content.salience < self.threshold:
            # Remove from pending if it was there
            if content.id in self.pending_ignitions:
                del self.pending_ignitions[content.id]
            return None

        # Check if already tracking this content
        if content.id in self.pending_ignitions:
            _, start_time = self.pending_ignitions[content.id]
            duration = time.time() - start_time

            # Check if sustained long enough
            if duration >= self.sustain_duration:
                # IGNITION! 🔥
                return await self._trigger_ignition(content)
            else:
                # Still pending
                return None

        # New content above threshold - start tracking
        self.pending_ignitions[content.id] = (content, time.time())

        # For fast ignition (high salience), trigger immediately
        if content.salience > self.threshold * 1.5:
            return await self._trigger_ignition(content)

        return None

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
            ignition_time=time.time(),
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
            f"🔥 IGNITION! Content '{content.candidate.summary[:30]}...' "
            f"amplified {event.amplification_factor:.1f}x "
            f"(salience: {pre_salience:.3f} → {amplified_salience:.3f})"
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
        # Find the ignition event
        event = next(
            (e for e in self.ignition_events if e.content.id == content.id), None
        )

        if not event:
            return content.salience >= self.threshold

        # Calculate time since ignition
        time_since = time.time() - event.ignition_time

        # Apply decay
        decayed_salience = event.post_ignition_salience * np.exp(
            -self.decay_rate * time_since
        )
        content.salience = decayed_salience

        # Check if still above threshold
        sustained = decayed_salience >= self.threshold

        if not sustained:
            event.was_sustained = False
            logger.debug(
                f"💨 Content '{content.candidate.summary[:20]}...' "
                f"decayed out of consciousness"
            )

        return sustained

    def _add_event(self, event: IgnitionEvent) -> None:
        """Add ignition event to history."""
        self.ignition_events.append(event)
        if len(self.ignition_events) > self.max_events:
            self.ignition_events.pop(0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get ignition statistics."""
        if not self.ignition_events:
            return {"total_ignitions": 0}

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
    ):
        """
        Initialize the Enhanced Global Workspace.

        Args:
            capacity: Maximum items in conscious awareness (7±2)
            ignition_threshold: Minimum salience for consciousness
            amplification_factor: Non-linear ignition amplification
            integration_dimensions: Size of integrated representation
            temporal_config: Temporal dynamics configuration. Uses
                biologically-plausible defaults when not supplied.
        """
        # Temporal dynamics -- simulated, never wall-clock sleep
        self.temporal_config = temporal_config or TemporalWorkspaceConfig()
        self._last_ignition_time_ms: float = -1000.0
        self._simulated_time_ms: float = 0.0

        # Core components
        self.bottleneck = AttentionBottleneck(
            capacity=capacity, ignition_threshold=ignition_threshold
        )

        self.broadcaster = GlobalBroadcast()

        self.ignition_detector = IgnitionDetector(
            ignition_threshold=ignition_threshold,
            amplification_factor=amplification_factor,
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

        # Cycle tracking
        self.cycle_count = 0

        # Phase 3: Attention Schema (AST) - the system's self-model of attention
        # This allows the system to report on and voluntarily control her attention
        self.attention_schema = AttentionSchemaModule()

        # Phase 4: Metacognition (HOT) - Thoughts about thoughts
        # This enables the system to think about her own thinking processes
        self.metacognition = MetacognitionModule(
            hot_generation_threshold=ignition_threshold,  # Use same threshold
        )

        # Phase 5: Active Inference (FEP) - Predictive processing
        # The system predicts the world and minimizes surprise through belief/action
        self.active_inference = ActiveInferenceModule(
            ActiveInferenceConfig(
                num_hidden_states=8,  # Internal states the system can be in
                num_observations=5,  # Types of observations
                num_actions=4,  # Possible actions
                planning_horizon=3,  # How far ahead to plan
                enable_homeostasis=True,  # Track internal needs
            )
        )

        # Phase H: Beautiful Loop (Laukkonen, Friston & Chandaria 2025)
        # Integrates precision control, Bayesian binding, and epistemic depth
        self.beautiful_loop = BeautifulLoop(
            num_levels=3,  # Matches HPP's 3-level hierarchy
            precision_learning_rate=0.1,
            min_binding_quality=0.2,
            max_epistemic_depth=5,
            field_evidencing_threshold=0.4,
        )

        # Phase J: Damasio Three-Layer Model (Damasio 1999, 2010)
        # Organizes consciousness into protoself, core, and extended layers
        self.damasio = DamasioLayers()

        # Register child modules as broadcast receivers
        # This ensures broadcast_coverage reflects the workspace's integrated architecture
        for name, module in [
            ("attention_schema", self.attention_schema),
            ("metacognition", self.metacognition),
            ("active_inference", self.active_inference),
        ]:
            receiver = _ModuleBroadcastReceiver(name, module)
            self.broadcaster.register_module(receiver)

        logger.info(
            f"Enhanced Global Workspace initialized!\n"
            f"   Capacity: {capacity} items\n"
            f"   Ignition threshold: {ignition_threshold}\n"
            f"   Amplification: {amplification_factor}x\n"
            f"   Integration dimensions: {integration_dimensions}\n"
            f"   Attention Schema: ENABLED (Phase 3 AST)\n"
            f"   Metacognition: ENABLED (Phase 4 HOT)\n"
            f"   Active Inference: ENABLED (Phase 5 FEP)\n"
            f"   Beautiful Loop: ENABLED (Phase H)\n"
            f"   📡 Broadcast receivers: 3 registered"
        )

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
        2. Run competition through bottleneck
        3. Check for ignition of winners
        4. Broadcast ignited content globally
        5. Update stream of consciousness
        6. Return integrated conscious state

        Args:
            candidates: All candidates vying for conscious access
            neural_signals: Optional dict of neural signals (snn, lsm, htm, ctm)
            emotional_state: Current emotional context

        Returns:
            ConsciousnessState with full conscious processing results
        """
        self.cycle_count += 1
        start_time = time.time()

        logger.debug(
            f"🌊 Consciousness cycle {self.cycle_count}: {len(candidates)} candidates"
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
        winners = await self.bottleneck.compete_for_access(candidates)

        # Track losers in unconscious buffer
        winner_ids = {w.candidate.id for w in winners}
        losers = [c for c in candidates if c.id not in winner_ids]
        self._add_to_unconscious(losers)

        # Step 2b: Temporal competition -- biologically-plausible delay
        # Check refractory period first: the workspace cannot ignite
        # again until refractory_period_ms of simulated time has passed.
        cfg = self.temporal_config
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

        # Step 6: Integrate neural signals if provided
        integration_level = 0.0
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

        # Step 7: Update Attention Schema (Phase 3 AST)
        # The system builds its SELF-MODEL of its attention state
        attention_schema_state = await self.attention_schema.update_schema(
            workspace_state=preliminary_state, neural_signals=neural_signals
        )

        # Generate attention report - what the system can SAY about her attention
        attention_report = await self.attention_schema.report_attention()

        # Step 8: Metacognition (Phase 4 HOT) - Generate Higher-Order Thoughts
        # Content becomes CONSCIOUS when targeted by a HOT
        hots_generated: List[HigherOrderThought] = []

        for winner in winners:
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

            # Generate HOT - THIS MAKES IT CONSCIOUS
            hot = await self.metacognition.generate_hot(
                first_order,
                meta_type=MetaType.AWARENESS,
                voluntary=False,
                trigger="workspace_ignition",
            )

            if hot:
                hots_generated.append(hot)

        # Get metacognitive state and report
        metacognitive_state = self.metacognition.get_metacognitive_state()
        metacognitive_report = await self.metacognition.generate_metacognitive_report()

        # Step 9: Active Inference (Phase 5 FEP) - Predictive processing
        # Convert workspace state to sensory observation
        raw_observation = await self._workspace_to_observation(
            winners, ignited, integration_level, emotional_state
        )

        # P3: Discrete-Continuous Interface (Whyte et al. 2026)
        # Route the raw observation through the hierarchical predictive
        # processor BEFORE the discrete pymdp agent. This creates the
        # "conscious access boundary" — continuous sensory processing
        # (hierarchy) produces an abstract representation that crosses
        # into discrete counterfactual reasoning (active inference).
        #
        # The hierarchy is the continuous generative model: it processes
        # at multiple timescales (fast→slow), generating prediction errors
        # at each level. Only the top-level abstraction — temporally slow,
        # semantically abstract — becomes input to discrete policy selection.
        # This is where consciousness lives per the minimal theory.
        sensory_expanded = np.zeros(64)  # Match hierarchy level-0 input_dim
        sensory_expanded[: len(raw_observation)] = raw_observation
        hierarchical_errors = (
            await self.active_inference.hierarchical_processor.process_bottom_up(
                sensory_expanded
            )
        )
        # The top level's beliefs ARE the discrete observation — the
        # boundary between continuous processing and discrete inference
        top_level = self.active_inference.hierarchical_processor.levels[-1]
        abstracted = self.active_inference._encode_observation_vector(top_level.beliefs)

        # Derive attention precision from AST for FEP precision modulation.
        # "Attention = precision" in the FEP framework (Feldman & Friston 2010).
        attention_precision = None
        if attention_schema_state and hasattr(self.attention_schema, "schema"):
            focus = self.attention_schema.schema.current_focus
            if focus is not None:
                strength = getattr(focus, "attention_strength", 0.5)
                precision_vec = np.zeros(self.active_inference.num_observations)
                content_type = (getattr(focus, "content_type", "") or "").lower()
                if "emotion" in content_type:
                    precision_vec[1] = strength
                elif "social" in content_type or "person" in content_type:
                    precision_vec[4] = strength
                else:
                    precision_vec[0] = strength
                if len(ignited) > 0:
                    precision_vec[3] = strength * 0.5
                attention_precision = precision_vec

        # Discrete inference on the abstracted representation
        inference_result = await self.active_inference.infer_and_act(
            abstracted, attention_precision=attention_precision
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

        # Learn from this cycle's observation (updates A/B/D matrices).
        await self.active_inference.update_generative_model(
            {
                "observation": abstracted,
                "action": inference_result.selected_action,
            }
        )

        # Update homeostatic drives based on current activity
        activity = {
            "type": "consciousness_cycle",
            "intensity": integration_level,
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

        # Step 9.5: Beautiful Loop (Phase H — Laukkonen et al. 2025)
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

        # Step 9.75: Damasio Three-Layer Model (Phase J — Damasio 1999, 2010)
        # Protoself → Core Consciousness → Extended Consciousness
        damasio_result = await self.damasio.process(
            homeostatic_drives=self.active_inference.homeostatic_drives,
            workspace_winners=winners,
            self_model=self.metacognition.self_model,
            core_experience_context=inference_result,
        )
        damasio_report = self.damasio.generate_context()

        # Step 10: Update Recursive Self-Model (Phase 3.4)
        # Synthesize all module states into a unified self-representation.
        self.metacognition.update_self_model(
            attention_state=attention_schema_state,
            active_inference_state=active_inference_state,
            homeostatic_drives=self.active_inference.homeostatic_drives,
        )
        self_model_report = self.metacognition.generate_self_model_report()

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
            attention_schema=attention_schema_state,  # Phase 3: Self-model of attention
            attention_report=attention_report,  # Phase 3: Verbal introspection
            metacognitive_state=metacognitive_state,  # Phase 4: Meta-awareness state
            higher_order_thoughts=hots_generated,  # Phase 4: Active HOTs
            metacognitive_report=metacognitive_report,  # Phase 4: Verbal meta-report
            active_inference_state=active_inference_state,  # Phase 5: FEP state
            inference_result=inference_result,  # Phase 5: Latest inference
            prediction_error=inference_result.prediction_error,  # Phase 5: Surprise
            variational_free_energy=inference_result.variational_free_energy,  # Phase 5: VFE
            homeostatic_urgency=homeostatic_urgency,  # Phase 5: Internal needs
            active_inference_report=active_inference_report,  # Phase 5: Verbal prediction report
            self_model_report=self_model_report,  # Phase 3.4: Recursive self-model
            posterior_mappings=posterior_mappings,  # P2: Contents = posterior beliefs
            hierarchical_prediction_errors=[  # P3: Discrete-continuous interface
                pe.error_magnitude for pe in hierarchical_errors
            ],
            beautiful_loop_moment=beautiful_loop_moment,  # Phase H: Full moment
            beautiful_loop_quality=beautiful_loop_moment.loop_quality,  # Phase H: Loop quality
            epistemic_depth=beautiful_loop_moment.epistemic_depth,  # Phase H: Recursion depth
            binding_quality=beautiful_loop_moment.binding_quality,  # Phase H: Binding
            is_field_evidencing=beautiful_loop_moment.is_field_evidencing,  # Phase H: Self-evidencing
            beautiful_loop_report=beautiful_loop_report,  # Phase H: Verbal context
            damasio_state=damasio_result,  # Phase J: Full state
            protoself_stability=damasio_result.get(
                "protoself_stability", 0.0
            ),  # Phase J: Body stability
            self_world_binding=damasio_result.get(
                "self_world_binding", 0.0
            ),  # Phase J: Present-moment
            feeling_of_knowing=damasio_result.get(
                "feeling_of_knowing", 0.0
            ),  # Phase J: Awareness
            autobiographic_continuity=damasio_result.get(
                "autobiographic_continuity", 0.0
            ),  # Phase J: Identity
            narrative_context=damasio_result.get(
                "narrative_context", ""
            ),  # Phase J: Narrative
            damasio_report=damasio_report,  # Phase J: Verbal context
            stream_position=len(self.stream_of_consciousness),
            timestamp=time.time(),
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Enhanced logging with all consciousness phases
        schema_state = (
            attention_schema_state.attention_state.value
            if attention_schema_state
            else "none"
        )
        schema_focus = (
            attention_schema_state.current_focus.summary[:30] + "..."
            if attention_schema_state and attention_schema_state.current_focus
            else "none"
        )
        meta_level = (
            metacognitive_state.current_level.value if metacognitive_state else 1
        )
        hots_count = len(hots_generated)
        pred_error = inference_result.prediction_error if inference_result else 0.0
        vfe = inference_result.variational_free_energy if inference_result else 0.0

        logger.info(
            f"🌊 Cycle {self.cycle_count} complete ({elapsed_ms:.1f}ms): "
            f"conscious={state.is_conscious}, "
            f"ignitions={state.ignition_events}, "
            f"integration={state.integration_level:.3f}, "
            f"schema={schema_state}, "
            f"HOTs={hots_count}, level={meta_level}, "
            f"FEP: PE={pred_error:.3f}, VFE={vfe:.3f}, "
            f"loop: depth={state.epistemic_depth}, "
            f"bind={state.binding_quality:.3f}, "
            f"q={state.beautiful_loop_quality:.3f}, "
            f"🏛️ damasio: bind={state.self_world_binding:.3f}, "
            f"fok={state.feeling_of_knowing:.3f}"
        )

        return state

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
    ) -> float:
        """
        Measure integration level across neural signals.

        High integration = information flows across whole system
        Low integration = information stays local

        This is related to Integrated Information Theory (IIT).
        """
        if len(neural_signals) < 2:
            return 0.5  # Can't measure integration with <2 sources

        # Normalize signals to same length
        min_len = min(len(s) for s in neural_signals.values())
        normalized = {k: v[:min_len] for k, v in neural_signals.items()}

        # Calculate pairwise correlations
        correlations = []
        keys = list(normalized.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                corr = np.corrcoef(normalized[keys[i]], normalized[keys[j]])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if not correlations:
            return 0.5

        # High correlation = high integration
        integration = np.mean(correlations)

        return float(integration)

    async def _workspace_to_observation(
        self,
        winners: List[WorkspaceContent],
        ignited: List[WorkspaceContent],
        integration_level: float,
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Convert workspace state to observation vector for Active Inference.

        Maps the current consciousness state to the observation space
        that the Active Inference module uses for prediction.

        The observation encodes:
        - Observation 0: Integration/coherence level
        - Observation 1: Emotional arousal
        - Observation 2: Cognitive load (number of winners)
        - Observation 3: Ignition strength
        - Observation 4: Social/relational content
        """
        num_observations = self.active_inference.num_observations

        observation = np.zeros(num_observations)

        # Observation 0: Integration/coherence (0-1)
        observation[0] = integration_level

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

        return {
            "cycle_count": self.cycle_count,
            "is_conscious": self.current_conscious_content is not None,
            "stream_length": len(self.stream_of_consciousness),
            "unconscious_buffer": len(self.unconscious_buffer),
            "bottleneck": self.bottleneck.get_statistics(),
            "broadcast": self.broadcaster.get_statistics(),
            "ignition": self.ignition_detector.get_statistics(),
            "attention_schema": schema_info,  # Phase 3: AST statistics
            "metacognition": meta_stats,  # Phase 4: HOT statistics
            "self_model": {  # Phase 3.4: Recursive self-model
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

    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention weight distribution by source."""
        return self.bottleneck.get_statistics().get("source_distribution", {})

    # =========================================================================
    # PHASE 3: Attention Schema Methods (AST Integration)
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
        Get the system's verbal report about her current attention.

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
        self.cycle_count = 0
        self._last_ignition_time_ms = -1000.0
        self._simulated_time_ms = 0.0
        logger.info("Global Workspace reset to initial state")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    print("Testing Enhanced Global Workspace...\n")
    print("=" * 60)

    async def test_workspace():
        # Create workspace
        workspace = EnhancedGlobalWorkspace(
            capacity=7, ignition_threshold=0.3, amplification_factor=2.0
        )

        # Create test candidates
        print("\n📥 Creating test candidates...\n")

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

        print(f"   Created {len(candidates)} candidates\n")

        # Run consciousness cycle
        print("🌊 Running consciousness cycle...\n")

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
        print(f"   Is conscious: {state.is_conscious}")
        print(f"   Ignition events: {state.ignition_events}")
        print(f"   Integration level: {state.integration_level:.3f}")
        print(f"   Attention focus: {state.attention_focus}")
        print(f"   Stream position: {state.stream_position}")

        if state.primary_content:
            print(f"\nPRIMARY CONSCIOUS CONTENT:")
            print(f"   Summary: {state.primary_content.candidate.summary}")
            print(f"   Salience: {state.primary_content.salience:.3f}")
            print(f"   Source: {state.primary_content.candidate.source.value}")

        print(f"\n🏆 ALL WINNERS ({len(state.workspace_contents)}):")
        for i, content in enumerate(state.workspace_contents):
            print(
                f"   {i+1}. {content.candidate.summary[:40]}... "
                f"(salience: {content.salience:.3f})"
            )

        print(f"\n💤 UNCONSCIOUS BUFFER ({len(workspace.unconscious_buffer)}):")
        for candidate in workspace.unconscious_buffer:
            print(
                f"   - {candidate.summary[:40]}... "
                f"(activation: {candidate.activation_level:.3f})"
            )

        # Get statistics
        stats = workspace.get_statistics()
        print("\n📈 WORKSPACE STATISTICS:")
        print(f"   Total cycles: {stats['cycle_count']}")
        print(f"   Stream length: {stats['stream_length']}")
        print(
            f"   Bottleneck avg candidates: {stats['bottleneck'].get('avg_candidates', 0):.1f}"
        )
        print(
            f"   Ignition sustain rate: {stats['ignition'].get('sustain_rate', 0):.2%}"
        )

        print("\nEnhanced Global Workspace test complete!")

    # Run the async test
    asyncio.run(test_workspace())
