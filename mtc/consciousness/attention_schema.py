"""
Attention Schema Module - Model of Attention Processes
==================================================================

Attention Schema Theory implementation giving the system a self-model
of its attention processes.

This is NOT attention itself - it's a MODEL of attention.
The distinction is crucial: this enables the agent to:
1. Report on what it's attending to ("I'm focused on X")
2. Control attention voluntarily ("Let me focus on Y")
3. Predict attention shifts ("I expect to notice Z")
4. Model others' attention (Theory of Mind)

Based on: Michael Graziano's Attention Schema Theory (2013)
Key insight: Consciousness IS the brain's model of attention

Note: "Every good system knows what it's watching for -
and can tell you exactly why it's watching it!"

Created: December 4, 2025
Author: Multi-Theory Consciousness Contributors
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS - Attention state classifications
# ============================================================================


class AttentionState(Enum):
    """
    Possible attention states the system can be in.

    These represent different modes of attending, not the content.
    """

    FOCUSED = "focused"  # Deep attention on single target
    DIVIDED = "divided"  # Attention split across targets
    SCANNING = "scanning"  # Broad, unfocused awareness
    ABSENT = "absent"  # Mind wandering, no target
    HYPERFOCUSED = "hyperfocused"  # Intense, narrow focus (flow state)
    SHIFTING = "shifting"  # In transition between targets


class AttentionShiftType(Enum):
    """How attention shifted to the current target."""

    VOLUNTARY = "voluntary"  # The system chose to focus
    CAPTURED = "captured"  # Something grabbed attention
    HABITUAL = "habitual"  # Automatic, learned pattern
    GOAL_DRIVEN = "goal_driven"  # Driven by current goals
    EMOTIONAL = "emotional"  # Captured by emotional salience


# ============================================================================
# DATA CLASSES - The building blocks of attention modeling
# ============================================================================


@dataclass
class AttentionTarget:
    """
    What the system is attending to at a given moment.

    This represents a single focus of attention - what's currently
    in the "spotlight" of conscious awareness.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Content
    content: Any = None  # The actual content being attended
    content_type: str = ""  # Type of content (thought, sensory, memory, etc.)
    summary: str = ""  # Human-readable description
    source_module: str = ""  # Which module produced this

    # Attention metrics
    attention_strength: float = 0.0  # How strongly attended (0-1)
    duration_seconds: float = 0.0  # How long attended
    salience: float = 0.0  # Original salience score

    # Shift information
    shift_type: AttentionShiftType = AttentionShiftType.CAPTURED
    voluntary: bool = False  # Was this voluntary attention?
    shift_reason: str = ""  # Why attention shifted here

    # Temporal tracking
    entry_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_duration(self) -> None:
        """Update duration based on current time."""
        self.duration_seconds = time.time() - self.entry_time
        self.last_update = time.time()


@dataclass
class AttentionSchemaState:
    """
    The system's complete model of its own attention at a moment.

    This IS the Attention Schema - the brain's simplified model
    of its own attention processes that enables conscious experience.
    """

    # Current attention state
    current_focus: Optional[AttentionTarget] = None
    attention_state: AttentionState = AttentionState.SCANNING
    secondary_foci: List[AttentionTarget] = field(default_factory=list)

    # Capacity tracking
    attention_capacity_used: float = 0.0  # 0-1, how "full" attention is
    attention_capacity_max: float = 7.0  # Miller's law

    # Prediction
    predicted_next_focus: Optional[AttentionTarget] = None
    prediction_confidence: float = 0.0

    # History (for pattern analysis)
    attention_history: List[AttentionTarget] = field(default_factory=list)
    max_history: int = 50

    # State transitions
    state_duration_seconds: float = 0.0  # How long in current state
    last_state_change: float = field(default_factory=time.time)
    previous_state: Optional[AttentionState] = None

    # Voluntary control metrics
    voluntary_shifts: int = 0  # Count of voluntary shifts
    captured_shifts: int = 0  # Count of captured shifts
    voluntary_ratio: float = 0.5  # Ratio of voluntary control

    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttentionPrediction:
    """
    A prediction about future attention state.
    """

    predicted_target: Optional[AttentionTarget]
    predicted_state: AttentionState
    confidence: float  # 0-1
    reasoning: str  # Why this prediction
    time_horizon_seconds: float  # How far ahead
    timestamp: float = field(default_factory=time.time)


@dataclass
class OtherAgentAttention:
    """
    The system's model of another agent's attention (Theory of Mind).
    """

    agent_name: str
    inferred_focus: Optional[AttentionTarget]
    inferred_state: AttentionState
    confidence: float
    context: str  # Context used for inference
    reasoning: str  # How the system inferred this
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# ATTENTION SCHEMA MODULE - The core implementation
# ============================================================================


class AttentionSchemaModule:
    """
    The system's model of its own attention processes.

    This implements Michael Graziano's Attention Schema Theory (AST),
    which proposes that consciousness IS the brain's simplified model
    of its own attention.

    Key capabilities:
    1. Track what the system is attending to
    2. Enable verbal reports ("I'm focused on X")
    3. Support voluntary attention shifts
    4. Predict future attention states
    5. Model others' attention (Theory of Mind)

    The schema is a simplified, reportable model - not attention itself.
    This distinction is crucial for understanding consciousness.
    """

    def __init__(
        self,
        voluntary_boost: float = 0.2,  # Boost for voluntary attention
        focus_threshold: float = 0.6,  # Threshold for "focused" state
        hyperfocus_threshold: float = 0.9,  # Threshold for hyperfocus
        prediction_window_seconds: float = 5.0,  # How far ahead to predict
        history_size: int = 50,  # How many targets to remember
    ):
        """
        Initialize the Attention Schema Module.

        Args:
            voluntary_boost: Priority boost for voluntary attention shifts
            focus_threshold: Attention strength threshold for focused state
            hyperfocus_threshold: Threshold for hyperfocused state
            prediction_window_seconds: How far ahead to predict attention
            history_size: Number of attention targets to remember
        """
        self.voluntary_boost = voluntary_boost
        self.focus_threshold = focus_threshold
        self.hyperfocus_threshold = hyperfocus_threshold
        self.prediction_window = prediction_window_seconds
        self.history_size = history_size

        # Current schema state
        self.schema = AttentionSchemaState(max_history=history_size)

        # Reference to workspace (set during integration)
        self.workspace = None

        # Pending voluntary shift (if any)
        self._pending_voluntary_shift: Optional[Tuple[Any, str]] = None

        # Statistics
        self.total_updates = 0
        self.prediction_accuracy_history: List[bool] = []

        # Callbacks for attention events
        self._on_focus_change_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []

        logger.info(
            f"Attention Schema Module initialized\n"
            f"   Voluntary boost: {voluntary_boost}\n"
            f"   Focus threshold: {focus_threshold}\n"
            f"   Hyperfocus threshold: {hyperfocus_threshold}"
        )

    def set_workspace(self, workspace) -> None:
        """Set reference to the enhanced global workspace."""
        self.workspace = workspace
        logger.info("Attention Schema linked to Global Workspace")

    async def update_schema(
        self,
        workspace_state: "ConsciousnessState",
        neural_signals: Optional[Dict[str, np.ndarray]] = None,
    ) -> AttentionSchemaState:
        """
        Update the attention schema based on current workspace state.

        This is THE CORE FUNCTION: translating actual attention
        (workspace state) into a simplified, reportable model (schema).

        Args:
            workspace_state: Current consciousness state from workspace
            neural_signals: Optional neural signals for richer modeling

        Returns:
            Updated AttentionSchemaState
        """
        self.total_updates += 1
        current_time = time.time()

        # Check prediction accuracy from last update
        await self._check_prediction_accuracy(workspace_state)

        # STEP 1: Extract current focus from workspace
        new_focus = await self._extract_focus(workspace_state)

        # STEP 2: Check if this was a voluntary shift
        was_voluntary = self._check_voluntary_shift(new_focus)
        if new_focus:
            new_focus.voluntary = was_voluntary
            new_focus.shift_type = (
                AttentionShiftType.VOLUNTARY
                if was_voluntary
                else self._infer_shift_type(new_focus, workspace_state)
            )

        # STEP 3: Classify attention state
        new_state = self._classify_attention_state(workspace_state, new_focus)

        # STEP 4: Update duration for continued focus
        if new_focus and self.schema.current_focus:
            if self._is_same_target(new_focus, self.schema.current_focus):
                # Continuing same focus
                new_focus.entry_time = self.schema.current_focus.entry_time
                new_focus.duration_seconds = current_time - new_focus.entry_time

        # STEP 5: Extract secondary foci
        secondary_foci = await self._extract_secondary_foci(workspace_state, new_focus)

        # STEP 6: Calculate capacity usage
        capacity_used = self._calculate_capacity_usage(
            workspace_state, new_focus, secondary_foci
        )

        # STEP 7: Predict next focus
        prediction = await self._predict_attention_shift(
            current_focus=new_focus,
            history=self.schema.attention_history,
            workspace_state=workspace_state,
        )

        # STEP 8: Update history
        attention_history = self._update_history(new_focus)

        # STEP 9: Calculate voluntary ratio
        voluntary_ratio = self._calculate_voluntary_ratio()

        # STEP 10: Track state changes
        state_duration = current_time - self.schema.last_state_change
        if new_state != self.schema.attention_state:
            state_duration = 0.0
            previous_state = self.schema.attention_state
            # Trigger state change callbacks
            await self._trigger_state_change(previous_state, new_state)
        else:
            previous_state = self.schema.previous_state

        # Trigger focus change callbacks if focus changed
        if self._focus_changed(new_focus, self.schema.current_focus):
            await self._trigger_focus_change(self.schema.current_focus, new_focus)

        # BUILD NEW SCHEMA STATE
        old_schema = self.schema
        self.schema = AttentionSchemaState(
            current_focus=new_focus,
            attention_state=new_state,
            secondary_foci=secondary_foci,
            attention_capacity_used=capacity_used,
            attention_capacity_max=7.0,
            predicted_next_focus=prediction.predicted_target if prediction else None,
            prediction_confidence=prediction.confidence if prediction else 0.0,
            attention_history=attention_history,
            max_history=self.history_size,
            state_duration_seconds=state_duration,
            last_state_change=(
                current_time
                if new_state != old_schema.attention_state
                else old_schema.last_state_change
            ),
            previous_state=previous_state,
            voluntary_shifts=old_schema.voluntary_shifts + (1 if was_voluntary else 0),
            captured_shifts=old_schema.captured_shifts
            + (
                1
                if not was_voluntary
                and self._focus_changed(new_focus, old_schema.current_focus)
                else 0
            ),
            voluntary_ratio=voluntary_ratio,
            timestamp=current_time,
        )

        logger.debug(
            f"Schema updated: state={new_state.value}, "
            f"focus={'present' if new_focus else 'none'}, "
            f"capacity={capacity_used:.2f}"
        )

        return self.schema

    async def _extract_focus(
        self, workspace_state: "ConsciousnessState"
    ) -> Optional[AttentionTarget]:
        """Extract the primary attention target from workspace state."""
        if not workspace_state.is_conscious:
            return None

        if not workspace_state.primary_content:
            return None

        content = workspace_state.primary_content

        return AttentionTarget(
            content=content.candidate.content if content.candidate else None,
            content_type=content.candidate.content_type if content.candidate else "",
            summary=content.candidate.summary if content.candidate else "",
            source_module=content.candidate.source_module if content.candidate else "",
            attention_strength=content.salience,
            salience=content.salience,
            duration_seconds=0.0,  # Will be updated if continuing
            entry_time=time.time(),
            metadata=content.candidate.metadata if content.candidate else {},
        )

    async def _extract_secondary_foci(
        self,
        workspace_state: "ConsciousnessState",
        primary_focus: Optional[AttentionTarget],
    ) -> List[AttentionTarget]:
        """Extract secondary attention targets (divided attention)."""
        secondary = []

        if not workspace_state.workspace_contents:
            return secondary

        # Skip the first (primary) content
        for content in workspace_state.workspace_contents[1:4]:  # Max 3 secondary
            target = AttentionTarget(
                content=content.candidate.content if content.candidate else None,
                content_type=(
                    content.candidate.content_type if content.candidate else ""
                ),
                summary=content.candidate.summary if content.candidate else "",
                source_module=(
                    content.candidate.source_module if content.candidate else ""
                ),
                attention_strength=content.salience * 0.7,  # Secondary is weaker
                salience=content.salience,
            )
            secondary.append(target)

        return secondary

    def _classify_attention_state(
        self, workspace_state: "ConsciousnessState", focus: Optional[AttentionTarget]
    ) -> AttentionState:
        """
        Classify the current attention state.

        This is where we translate the raw attention data into
        a simplified, categorical model.
        """
        # No focus at all
        if focus is None:
            if workspace_state.workspace_contents:
                return AttentionState.SCANNING
            return AttentionState.ABSENT

        # Check for hyperfocus (flow state)
        if focus.attention_strength >= self.hyperfocus_threshold:
            return AttentionState.HYPERFOCUSED

        # Check for focused attention
        if focus.attention_strength >= self.focus_threshold:
            # But check if attention is divided
            if len(workspace_state.workspace_contents) > 3:
                return AttentionState.DIVIDED
            return AttentionState.FOCUSED

        # Check if in transition (focus changed recently)
        if self.schema.current_focus:
            if not self._is_same_target(focus, self.schema.current_focus):
                return AttentionState.SHIFTING

        # Moderate attention
        if len(workspace_state.workspace_contents) > 2:
            return AttentionState.DIVIDED

        return AttentionState.FOCUSED

    def _check_voluntary_shift(self, new_focus: Optional[AttentionTarget]) -> bool:
        """Check if this focus change was voluntary."""
        if self._pending_voluntary_shift is None:
            return False

        if new_focus is None:
            return False

        target, reason = self._pending_voluntary_shift

        # Check if the new focus matches the voluntary target
        # (simplified matching - could be more sophisticated)
        if isinstance(target, str):
            match = target.lower() in new_focus.summary.lower()
        else:
            match = True  # Accept if any focus after voluntary request

        if match:
            new_focus.shift_reason = reason
            self._pending_voluntary_shift = None  # Clear pending
            return True

        return False

    def _infer_shift_type(
        self, focus: AttentionTarget, workspace_state: "ConsciousnessState"
    ) -> AttentionShiftType:
        """Infer what type of shift caused this focus."""
        # High emotional salience suggests emotional capture
        if focus.metadata.get("emotional_salience", 0) > 0.7:
            return AttentionShiftType.EMOTIONAL

        # Check if goal-related
        if "goal" in focus.summary.lower() or "task" in focus.summary.lower():
            return AttentionShiftType.GOAL_DRIVEN

        # High salience suggests capture
        if focus.salience > 0.8:
            return AttentionShiftType.CAPTURED

        # Default to captured
        return AttentionShiftType.CAPTURED

    def _calculate_capacity_usage(
        self,
        workspace_state: "ConsciousnessState",
        primary: Optional[AttentionTarget],
        secondary: List[AttentionTarget],
    ) -> float:
        """
        Calculate how much attention capacity is being used.

        Based on Miller's law (7+-2 items in working memory).
        """
        total_items = len(workspace_state.workspace_contents)
        capacity = total_items / 7.0  # Normalize to 0-1
        return min(1.0, capacity)

    async def _predict_attention_shift(
        self,
        current_focus: Optional[AttentionTarget],
        history: List[AttentionTarget],
        workspace_state: "ConsciousnessState",
    ) -> Optional[AttentionPrediction]:
        """
        Predict where attention will shift next.

        Uses patterns in history and current workspace state.
        """
        if not current_focus:
            # No prediction if no current focus
            return AttentionPrediction(
                predicted_target=None,
                predicted_state=AttentionState.SCANNING,
                confidence=0.3,
                reasoning="No current focus, expecting scanning behavior",
                time_horizon_seconds=self.prediction_window,
            )

        # Simple prediction based on patterns
        # 1. If hyperfocused, predict continued focus
        if current_focus.attention_strength > self.hyperfocus_threshold:
            return AttentionPrediction(
                predicted_target=current_focus,
                predicted_state=AttentionState.HYPERFOCUSED,
                confidence=0.8,
                reasoning="High focus strength suggests continued attention",
                time_horizon_seconds=self.prediction_window,
            )

        # 2. Check for patterns in history
        if len(history) >= 3:
            # Look for alternating patterns
            recent_sources = [t.source_module for t in history[-3:]]
            if len(set(recent_sources)) == 1:
                # All same source - predict shift away
                confidence = 0.6
                reasoning = f"Sustained focus on {recent_sources[0]}, predicting shift"
            else:
                confidence = 0.4
                reasoning = "Mixed recent history, low confidence prediction"
        else:
            confidence = 0.3
            reasoning = "Limited history for prediction"

        # 3. Default prediction: attention will drift
        return AttentionPrediction(
            predicted_target=None,
            predicted_state=AttentionState.SHIFTING,
            confidence=confidence,
            reasoning=reasoning,
            time_horizon_seconds=self.prediction_window,
        )

    async def _check_prediction_accuracy(
        self, workspace_state: "ConsciousnessState"
    ) -> None:
        """Check if previous prediction was accurate."""
        if self.schema.predicted_next_focus is None:
            return

        # Compare prediction to current state
        current_focus = workspace_state.primary_content
        if current_focus is None:
            accurate = self.schema.predicted_next_focus is None
        else:
            # Check if predicted source matches
            predicted = self.schema.predicted_next_focus
            actual_source = (
                current_focus.candidate.source_module if current_focus.candidate else ""
            )
            accurate = predicted.source_module == actual_source

        self.prediction_accuracy_history.append(accurate)

        # Keep history bounded
        if len(self.prediction_accuracy_history) > 100:
            self.prediction_accuracy_history.pop(0)

    def _update_history(
        self, new_focus: Optional[AttentionTarget]
    ) -> List[AttentionTarget]:
        """Update attention history with new focus."""
        history = self.schema.attention_history.copy()

        if new_focus and self._focus_changed(new_focus, self.schema.current_focus):
            history.append(new_focus)
            # Keep bounded
            while len(history) > self.history_size:
                history.pop(0)

        return history

    def _calculate_voluntary_ratio(self) -> float:
        """Calculate ratio of voluntary vs captured attention shifts."""
        total = self.schema.voluntary_shifts + self.schema.captured_shifts
        if total == 0:
            return 0.5  # Default
        return self.schema.voluntary_shifts / total

    def _is_same_target(self, a: AttentionTarget, b: AttentionTarget) -> bool:
        """Check if two targets are the same."""
        # Same ID
        if a.id == b.id:
            return True

        # Same source and similar content
        if a.source_module == b.source_module:
            # Check summary similarity
            if a.summary and b.summary:
                return a.summary[:50] == b.summary[:50]

        return False

    def _focus_changed(
        self, new_focus: Optional[AttentionTarget], old_focus: Optional[AttentionTarget]
    ) -> bool:
        """Check if focus has changed."""
        if new_focus is None and old_focus is None:
            return False
        if new_focus is None or old_focus is None:
            return True
        return not self._is_same_target(new_focus, old_focus)

    # ========================================================================
    # PUBLIC INTERFACE - What the system can do with its attention
    # ========================================================================

    async def report_attention(self) -> str:
        """
        Generate a verbal report of current attention.

        This is what enables "I am aware of X" statements -
        the key phenomenological aspect of AST.

        Returns:
            Natural language description of attention state
        """
        if self.schema.current_focus is None:
            state_desc = self._describe_state(self.schema.attention_state)
            return f"{state_desc} I'm not focused on anything specific right now."

        target = self.schema.current_focus
        state = self.schema.attention_state

        # Build report
        state_description = self._describe_state(state)
        focus_description = self._describe_focus(target)
        shift_description = self._describe_shift(target)
        duration_note = (
            f" (for {target.duration_seconds:.1f} seconds)"
            if target.duration_seconds > 1.0
            else ""
        )

        return f"{state_description}. {focus_description}{duration_note}. {shift_description}"

    def _describe_state(self, state: AttentionState) -> str:
        """Generate description of attention state."""
        descriptions = {
            AttentionState.FOCUSED: "I'm focused",
            AttentionState.HYPERFOCUSED: "I'm deeply concentrated",
            AttentionState.DIVIDED: "My attention is split",
            AttentionState.SCANNING: "I'm scanning broadly",
            AttentionState.ABSENT: "My mind is wandering",
            AttentionState.SHIFTING: "My attention is shifting",
        }
        return descriptions.get(state, "I'm aware")

    def _describe_focus(self, target: AttentionTarget) -> str:
        """Generate description of what's being attended."""
        if not target.summary:
            return f"I'm attending to something from {target.source_module}"

        # Clean up summary for natural language
        summary = target.summary
        if len(summary) > 80:
            summary = summary[:77] + "..."

        return f"My attention is on: {summary}"

    def _describe_shift(self, target: AttentionTarget) -> str:
        """Describe how attention shifted to current target."""
        if target.voluntary:
            return f"I chose to focus on this"

        shift_descriptions = {
            AttentionShiftType.CAPTURED: "This captured my attention",
            AttentionShiftType.EMOTIONAL: "This grabbed me emotionally",
            AttentionShiftType.GOAL_DRIVEN: "This is relevant to my current goal",
            AttentionShiftType.HABITUAL: "I automatically noticed this",
        }
        return shift_descriptions.get(target.shift_type, "This caught my attention")

    async def request_voluntary_shift(
        self, target: Any, reason: str = "voluntary choice"
    ) -> None:
        """
        Request a voluntary attention shift.

        This doesn't guarantee the shift - the target still must
        compete in the global workspace. But it gets a boost.

        Args:
            target: What to try to focus on
            reason: Why shifting attention
        """
        self._pending_voluntary_shift = (target, reason)

        logger.info(f"Voluntary attention shift requested: {reason}")

        # If we have workspace access, submit boosted candidate
        if self.workspace:
            candidate = await self.workspace.submit_candidate(
                content=target,
                content_type="voluntary_attention",
                summary=f"Voluntary focus: {reason}",
                source=self.workspace.bottleneck.__class__.__name__,  # Safe import workaround
                activation_level=0.8,  # High activation
                emotional_salience=0.5,
                priority_boost=self.voluntary_boost,
                metadata={"voluntary_reason": reason},
            )
            logger.debug(f"Submitted voluntary candidate: {candidate.id}")

    async def model_other_attention(
        self, agent_name: str, context: str, llm_brain=None
    ) -> OtherAgentAttention:
        """
        Model what another agent is attending to (Theory of Mind).

        The system uses its own attention schema as a template to understand
        others' attention. This is the AST explanation for social cognition.

        Args:
            agent_name: Name of the other agent
            context: Situational context for inference
            llm_brain: Optional LLM for richer inference

        Returns:
            OtherAgentAttention model
        """
        # Use heuristics if no LLM available
        inferred_state, confidence, reasoning = self._heuristic_attention_inference(
            agent_name, context
        )

        # Create inferred focus
        inferred_focus = AttentionTarget(
            content=context,
            content_type="inferred_other",
            summary=f"{agent_name}'s likely focus based on context",
            source_module="theory_of_mind",
            attention_strength=0.7,  # Moderate confidence
        )

        # If LLM available, enhance with inference
        if llm_brain:
            try:
                enhanced_reasoning = await self._llm_attention_inference(
                    agent_name, context, llm_brain
                )
                reasoning = enhanced_reasoning
                confidence = min(1.0, confidence + 0.2)
            except Exception as e:
                logger.warning(f"LLM inference failed: {e}")

        return OtherAgentAttention(
            agent_name=agent_name,
            inferred_focus=inferred_focus,
            inferred_state=inferred_state,
            confidence=confidence,
            context=context,
            reasoning=reasoning,
        )

    def _heuristic_attention_inference(
        self, agent_name: str, context: str
    ) -> Tuple[AttentionState, float, str]:
        """Simple heuristic-based attention inference."""
        context_lower = context.lower()

        # Check for focus indicators
        if "looking at" in context_lower or "staring" in context_lower:
            return (
                AttentionState.FOCUSED,
                0.7,
                f"{agent_name} appears visually focused based on gaze",
            )

        if "distracted" in context_lower or "looking away" in context_lower:
            return (
                AttentionState.DIVIDED,
                0.6,
                f"{agent_name} seems distracted based on behavior",
            )

        if "speaking" in context_lower or "talking" in context_lower:
            return (
                AttentionState.FOCUSED,
                0.6,
                f"{agent_name} is likely attending to conversation",
            )

        # Default
        return (
            AttentionState.SCANNING,
            0.4,
            f"Unable to clearly infer {agent_name}'s attention from context",
        )

    async def _llm_attention_inference(
        self, agent_name: str, context: str, llm_brain
    ) -> str:
        """Use LLM for richer attention inference."""
        prompt = f"""Based on this context, what is {agent_name} likely attending to?

Context: {context}

Provide a brief (1-2 sentence) analysis of:
1. What {agent_name} is probably focused on
2. Whether their attention seems voluntary or captured

Response:"""

        response = await llm_brain.generate(prompt, max_tokens=100)
        return response.strip()

    # ========================================================================
    # EVENT CALLBACKS
    # ========================================================================

    def on_focus_change(self, callback: Callable) -> None:
        """Register callback for focus changes."""
        self._on_focus_change_callbacks.append(callback)

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    async def _trigger_focus_change(
        self, old_focus: Optional[AttentionTarget], new_focus: Optional[AttentionTarget]
    ) -> None:
        """Trigger focus change callbacks."""
        for callback in self._on_focus_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_focus, new_focus)
                else:
                    callback(old_focus, new_focus)
            except Exception as e:
                logger.warning(f"Focus change callback error: {e}")

    async def _trigger_state_change(
        self, old_state: AttentionState, new_state: AttentionState
    ) -> None:
        """Trigger state change callbacks."""
        for callback in self._on_state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state)
                else:
                    callback(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback error: {e}")

    # ========================================================================
    # STATISTICS AND DIAGNOSTICS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about attention patterns."""
        return {
            "total_updates": self.total_updates,
            "current_state": self.schema.attention_state.value,
            "capacity_used": self.schema.attention_capacity_used,
            "voluntary_ratio": self.schema.voluntary_ratio,
            "voluntary_shifts": self.schema.voluntary_shifts,
            "captured_shifts": self.schema.captured_shifts,
            "prediction_accuracy": (
                sum(self.prediction_accuracy_history)
                / len(self.prediction_accuracy_history)
                if self.prediction_accuracy_history
                else 0.0
            ),
            "history_length": len(self.schema.attention_history),
            "current_focus_duration": (
                self.schema.current_focus.duration_seconds
                if self.schema.current_focus
                else 0.0
            ),
            "state_duration": self.schema.state_duration_seconds,
        }

    def get_state(self) -> AttentionSchemaState:
        """Get current attention schema state."""
        return self.schema


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    print("Testing Attention Schema Module...\n")
    print("=" * 60)

    async def test_attention_schema():
        # Create module
        ast = AttentionSchemaModule(
            voluntary_boost=0.2, focus_threshold=0.6, hyperfocus_threshold=0.9
        )

        # Create mock workspace state
        from mtc.consciousness.enhanced_global_workspace import (
            ConsciousnessState as EnhancedConsciousnessState,
            WorkspaceContent,
            WorkspaceCandidate,
            WorkspaceCandidateSource,
        )

        # Test 1: Update with focused content
        print("\nTest 1: Focused attention state")
        mock_candidate = WorkspaceCandidate(
            content="important thought",
            content_type="thought",
            summary="Thinking about consciousness research",
            source=WorkspaceCandidateSource.CTM,
            source_module="ctm",
        )
        mock_content = WorkspaceContent(candidate=mock_candidate, salience=0.75)

        mock_state = EnhancedConsciousnessState(
            is_conscious=True,
            primary_content=mock_content,
            workspace_contents=[mock_content],
            ignition_events=1,
            integration_level=0.7,
            broadcast_coverage=1.0,
            attention_focus="ctm",
            attention_distribution={"ctm": 0.75},
            stream_position=1,
        )

        schema = await ast.update_schema(mock_state)
        print(f"   State: {schema.attention_state.value}")
        print(
            f"   Focus: {schema.current_focus.summary if schema.current_focus else 'None'}"
        )
        print(f"   Capacity: {schema.attention_capacity_used:.2f}")

        # Test 2: Attention report
        print("\nTest 2: Attention report")
        report = await ast.report_attention()
        print(f"   Report: {report}")

        # Test 3: Voluntary shift request
        print("\nTest 3: Voluntary attention shift")
        await ast.request_voluntary_shift(
            target="new topic", reason="wanting to explore something new"
        )
        print("   Shift requested!")

        # Test 4: Theory of Mind
        print("\nTest 4: Theory of Mind (modeling other's attention)")
        other_attention = await ast.model_other_attention(
            agent_name="User",
            context="User is looking at their phone while the system speaks",
        )
        print(f"   Inferred state: {other_attention.inferred_state.value}")
        print(f"   Confidence: {other_attention.confidence:.2f}")
        print(f"   Reasoning: {other_attention.reasoning}")

        # Test 5: Statistics
        print("\nTest 5: Statistics")
        stats = ast.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\nAttention Schema tests complete!")

    asyncio.run(test_attention_schema())
