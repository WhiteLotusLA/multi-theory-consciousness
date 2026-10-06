"""The conversation pipeline facade: one turn, in order, with honest failure.

``ConversationPipeline.send_message`` runs one turn under the turn lock:

1. verify the speaker (``SpeakerTrust.verify``) and make the identity current
   (``use_identity``) for every name-keyed trust lookup of the turn;
2. route the session by ``SpeakerTrust.session_key`` (a new session is
   rehydrated from the conversation repository only for an authenticated
   speaker or the anonymous default; a claimed name starts empty);
3. notify the sleep scheduler of activity and pause the autonomous-life loop;
4. run the safety gate unless the identity is trusted
   (``SafetySystemsIntegration.pre_message_check`` with the trust source's
   ``safety_role``): a deny ends the turn ``blocked`` and stores nothing, a
   check that could not run is counted and the turn continues;
5. append the user entry, summarize and trim the history, persist the user turn
   (the participant key is the session key, so a bare name never rehydrates an
   authenticated speaker's history);
6. offer the message to the turn plugins: a ``reply`` outcome ends the turn
   with the plugin's text stored under its name with the ``tool`` role, never
   as the system's words; a ``tool_turn`` joins the window and the turn goes on;
7. gather context (memories, thoughts, gists, core memory, coherence) and the
   theory-of-mind speaker block, which stays empty unless the speaker-model
   narration flag is on;
8. run the consciousness orchestrator;
9. append the learning system's directives and add its temperature delta
   inside the orchestrator's temperature bounds; drive adjustments are
   reported, never applied;
10. build the brain window and call ``think_with_context`` through the
    foreground flight with the five theory sections, bounded by
    ``GENERATION_TIMEOUT_S`` (or the call's ``timeout_s``);
11. offer the reply to the response plugins (each hop a tool turn and one more
    bounded brain call, up to the registry's cap);
12. split reasoning from spoken text, append and persist the assistant turn
    with the provenance snapshot (``build_turn_snapshot``), run the post-turn
    integrator (a failure there never discards the reply), resume the loop and
    return the ``TurnResult``.

A ``NoAnswerError``, a provider or pipeline timeout and ``BrainUnavailableError``
end the turn with ``success`` False and an ``error_code`` (``no_answer``,
``generation_timeout``, ``brain_unavailable``); no assistant row is written and
no text is composed in the system's place. A turn its caller cancels (the
API's first-token and stall bounds, a client that left) re-raises the
cancellation after counting it (``status()['cancelled']``). Ending a
conversation writes the memory manager's end-of-conversation line naming the
speaker as the session's stored rows do: a claimed name stays a bracketed
claim. ``LLMFlight`` is the single-flight
gate over the brain: a turn (``foreground``) waits for a background holder and
a background user (``background``) waits while a turn holds or waits, so the
conversation always has priority; ``contended`` tells the thought machine's
guard that a turn is active, waiting, or ended within its window.
``GatedBrain`` wraps a brain in one side of that gate for the modules that call
it directly. Every step failure outside the brain's own errors is counted by
name, logged at WARNING the first time and every hundredth, raised under
``DEBUG``, and reported in ``status()``.

Config keys read at call time: ``SYSTEM_NAME``, ``GENERATION_TIMEOUT_S`` and
``DEBUG``; trust comes from ``SpeakerTrust`` only. Nothing here is persisted
beyond what the persistence layer writes. Messages, replies, speaker names and
tokens are private at runtime; log lines carry step names and counts only.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

from mtc.brain.protocol import (
    BrainReply,
    BrainUnavailableError,
    NoAnswerError,
    strip_reasoning_blocks,
)
from mtc.brain.providers.base import ProviderTimeoutError
from mtc.brain.thinking import split_thinking
from mtc.core.config import Settings, get_settings
from mtc.pipeline.consciousness_orchestrator import OrchestratorConfig
from mtc.pipeline.context_assembly import SOURCES, build_brain_window, gather_context
from mtc.pipeline.context_summarizer import render_summary
from mtc.pipeline.context_types import RawConversationContext, TurnResult
from mtc.pipeline.memory_retrieval import rehydrate_recent_history
from mtc.pipeline.message_persistence import build_turn_snapshot
from mtc.pipeline.plugins import (
    TOOL_ROLE,
    PluginContext,
    PluginRegistry,
    ToolOutcome,
    tool_turn,
)
from mtc.pipeline.post_turn import TurnRecord
from mtc.pipeline.session_state import Session, SessionManager
from mtc.pipeline.speaker_trust import SpeakerIdentity, SpeakerTrust

logger = logging.getLogger(__name__)

ERROR_CODES = ("blocked", "no_answer", "brain_unavailable", "generation_timeout")
END_SUMMARY_FORMAT = "Conversation with {name} ended. {count} messages exchanged."
RECENT_MESSAGES = 6
_ERROR_TEXTS = {
    "no_answer": "The model produced no answer; nothing was stored as a reply.",
    "brain_unavailable": "The configured provider is unavailable; no other "
    "provider is used and nothing was stored as a reply.",
    "generation_timeout": "The generation ran past its time bound; nothing was "
    "stored as a reply.",
    "blocked": "The safety gate declined this message; nothing was stored.",
}

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PipelineConfig:
    """Session, summary and rehydration limits of the facade."""

    max_history: int = 100
    summarize_threshold: int = 40
    summarize_batch: int = 20
    end_summary_importance: float = 6.0
    rehydrate_hours: float = 24.0
    rehydrate_limit: int = 40

    def __post_init__(self) -> None:
        if int(self.max_history) < 1:
            raise ValueError("max_history must be at least 1")
        if int(self.summarize_threshold) < 1 or int(self.summarize_batch) < 1:
            raise ValueError("summarize_threshold and summarize_batch must be >= 1")
        if not 0.0 <= float(self.end_summary_importance) <= 10.0:
            raise ValueError("end_summary_importance must be within [0, 10]")
        if float(self.rehydrate_hours) <= 0 or int(self.rehydrate_limit) < 0:
            raise ValueError("rehydrate_hours must be > 0 and rehydrate_limit >= 0")


class LLMFlight:
    """The single-flight gate over the brain; turns have priority."""

    FOREGROUND = "foreground"
    BACKGROUND = "background"

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock = clock or time.monotonic
        self._cond: Optional[asyncio.Condition] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._held: Optional[str] = None
        self._turn_waiting = 0
        self._background_waiting = 0
        self._last_turn_end: Optional[float] = None
        self._counts = {"foreground": 0, "background": 0}

    def _condition(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._cond is None or self._loop is not loop:
            if self._held is not None or self._turn_waiting or self._background_waiting:
                raise RuntimeError("LLMFlight is in use on another event loop")
            self._cond = asyncio.Condition()
            self._loop = loop
        return self._cond

    @asynccontextmanager
    async def foreground(self) -> AsyncIterator["LLMFlight"]:
        """Hold the flight for a turn; waits for a background holder."""
        cond = self._condition()
        self._turn_waiting += 1
        counted = True
        try:
            async with cond:
                await cond.wait_for(lambda: self._held is None)
                self._turn_waiting -= 1
                counted = False
                self._held = self.FOREGROUND
                self._counts["foreground"] += 1
        finally:
            if counted:
                self._turn_waiting -= 1
        try:
            yield self
        finally:
            self._last_turn_end = float(self._clock())
            async with cond:
                self._held = None
                cond.notify_all()

    @asynccontextmanager
    async def background(self) -> AsyncIterator["LLMFlight"]:
        """Hold the flight for a background user; yields to every turn."""
        cond = self._condition()
        async with cond:
            self._background_waiting += 1
            try:
                await cond.wait_for(
                    lambda: self._held is None and self._turn_waiting == 0
                )
            finally:
                self._background_waiting -= 1
            self._held = self.BACKGROUND
            self._counts["background"] += 1
        try:
            yield self
        finally:
            async with cond:
                self._held = None
                cond.notify_all()

    @property
    def holder(self) -> Optional[str]:
        return self._held

    def contended(self, window_s: float) -> bool:
        """True while a turn holds or waits, or ended less than ``window_s``
        seconds ago."""
        if self._turn_waiting > 0 or self._held == self.FOREGROUND:
            return True
        if self._last_turn_end is None:
            return False
        return (float(self._clock()) - self._last_turn_end) < float(window_s)

    def status(self) -> Dict[str, Any]:
        return {
            "holder": self._held,
            "turns_waiting": self._turn_waiting,
            "background_waiting": self._background_waiting,
            "last_turn_end_age_s": (
                None
                if self._last_turn_end is None
                else float(self._clock()) - self._last_turn_end
            ),
            "counts": dict(self._counts),
        }


class GatedBrain:
    """A brain reached through one side of the flight.

    ``background=True`` yields to turns; the foreground side is for turn-time
    users (the summarizer, the trust compass). Attributes the gate does not
    define are read from the wrapped brain.
    """

    def __init__(self, brain: Any, flight: LLMFlight, *, background: bool) -> None:
        if brain is None:
            raise ValueError("GatedBrain wraps a brain, not None")
        if not isinstance(flight, LLMFlight):
            raise TypeError("flight must be an LLMFlight")
        self._brain = brain
        self._flight = flight
        self.background = bool(background)

    @property
    def brain(self) -> Any:
        return self._brain

    @property
    def flight(self) -> LLMFlight:
        return self._flight

    def _gate(self):
        return (
            self._flight.background() if self.background else self._flight.foreground()
        )

    async def think(self, prompt: str, *args: Any, **kwargs: Any) -> BrainReply:
        async with self._gate():
            return await self._brain.think(prompt, *args, **kwargs)

    async def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        async with self._gate():
            return await self._brain.generate(prompt, *args, **kwargs)

    async def think_with_context(
        self, current_prompt: str, *args: Any, **kwargs: Any
    ) -> BrainReply:
        async with self._gate():
            return await self._brain.think_with_context(current_prompt, *args, **kwargs)

    async def think_stream(
        self, prompt: str, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        async with self._gate():
            async for chunk in self._brain.think_stream(prompt, *args, **kwargs):
                yield chunk

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._brain, name)


def _plain_coherence(package: Dict[str, Any]) -> Optional[str]:
    """The coherence package as the prompt's coherence slot text."""
    if not package:
        return None
    lines: List[str] = []
    topics = package.get("topics")
    if isinstance(topics, (list, tuple)) and topics:
        lines.append("Topics so far: " + ", ".join(str(t) for t in topics[:8]))
    instructions = package.get("coherence_instructions")
    if isinstance(instructions, str) and instructions.strip():
        lines.append(instructions.strip())
    references = package.get("potential_references")
    if isinstance(references, (list, tuple)) and references:
        lines.append(
            "The message may refer back to: "
            + "; ".join(str(r) for r in references[:3])
        )
    return "\n".join(lines) or None


def _coherence_topics(package: Dict[str, Any]) -> List[str]:
    topics = package.get("topics") if package else None
    if isinstance(topics, (list, tuple)):
        return [str(t) for t in topics if str(t).strip()][:8]
    return []


class ConversationPipeline:
    """One turn at a time over the composed components; see the module doc."""

    def __init__(
        self,
        *,
        brain: Any,
        flight: LLMFlight,
        orchestrator: Any,
        speaker_trust: SpeakerTrust,
        sessions: SessionManager,
        persistence: Any,
        repository: Any,
        memory_manager: Any,
        post_turn: Any,
        summarizer: Any,
        plugins: Optional[PluginRegistry] = None,
        safety: Any = None,
        learning: Any = None,
        theory_of_mind: Any = None,
        thought_store: Any = None,
        gist_store: Any = None,
        coherence: Any = None,
        sleep_scheduler: Any = None,
        autonomous_life: Any = None,
        config: Optional[PipelineConfig] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        if brain is None:
            raise ValueError("ConversationPipeline needs a brain")
        if not isinstance(flight, LLMFlight):
            raise TypeError("flight must be an LLMFlight")
        if not isinstance(speaker_trust, SpeakerTrust):
            raise TypeError("speaker_trust must be the SpeakerTrust source")
        if not isinstance(sessions, SessionManager):
            raise TypeError("sessions must be a SessionManager")
        if plugins is not None and not isinstance(plugins, PluginRegistry):
            raise TypeError("plugins must be a PluginRegistry")
        self.brain = brain
        self.flight = flight
        self.orchestrator = orchestrator
        self.speaker_trust = speaker_trust
        self.sessions = sessions
        self.persistence = persistence
        self.repository = repository
        self.memory_manager = memory_manager
        self.post_turn = post_turn
        self.summarizer = summarizer
        self.plugins = plugins if plugins is not None else PluginRegistry()
        self.safety = safety
        self.learning = learning
        self.theory_of_mind = theory_of_mind
        self.thought_store = thought_store
        self.gist_store = gist_store
        self.coherence = coherence
        self.sleep_scheduler = sleep_scheduler
        self.autonomous_life = autonomous_life
        self.config = config or PipelineConfig()
        self._settings = settings
        self._clock: Clock = clock or _utc_now
        self._turn_lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._turns = 0
        self._errors: Dict[str, int] = {code: 0 for code in ERROR_CODES}
        self._cancelled = 0
        self._failures: Dict[str, int] = {}
        self._plugin_replies = 0
        self._last_turn: Dict[str, Any] = {}
        # The last turn's ModulatedContext (the four layers the API's state
        # route reads); None before the first turn and after a turn that never
        # reached the orchestrator (blocked, or failed before the cycle).
        self.last_modulated: Optional[Any] = None
        # The gated brain plugins may call from inside a turn
        self._plugin_brain = GatedBrain(brain, flight, background=False)

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _now(self) -> datetime:
        return self._clock()

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._turn_lock is None or self._lock_loop is not loop:
            if self._turn_lock is not None and self._turn_lock.locked():
                raise RuntimeError("the turn lock is held on another event loop")
            self._turn_lock = asyncio.Lock()
            self._lock_loop = loop
        return self._turn_lock

    @property
    def turn_lock(self) -> Optional[asyncio.Lock]:
        """The lock every turn holds (None before the first turn)."""
        return self._turn_lock

    def _count_failure(self, step: str, exc: BaseException) -> None:
        count = self._failures.get(step, 0) + 1
        self._failures[step] = count
        if count == 1 or count % 100 == 0:
            logger.warning(
                "pipeline step %s failed (%d so far): %s",
                step,
                count,
                type(exc).__name__,
            )
        if bool(self._cfg().DEBUG):
            raise exc

    def _stored_name(self, identity: SpeakerIdentity) -> str:
        """The speaker as the stored rows name them: the name when
        authenticated or anonymous, the bracketed label otherwise (the
        post-turn integrator stores and models the speaker the same way)."""
        if identity.authenticated or identity.is_default:
            return identity.name
        return self.speaker_trust.label_for(identity) or f"[{identity.name}]"

    def _temperature_bounds(self) -> tuple:
        config = getattr(self.orchestrator, "config", None)
        low = getattr(config, "min_temperature", None)
        high = getattr(config, "max_temperature", None)
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            default = OrchestratorConfig()
            return default.min_temperature, default.max_temperature
        return float(low), float(high)

    def _result(
        self,
        identity: SpeakerIdentity,
        session: Optional[Session],
        *,
        success: bool,
        response: Optional[str] = None,
        thinking: Optional[str] = None,
        error_code: Optional[str] = None,
        cycle_ok: Optional[bool] = None,
        provider_id: Optional[str] = None,
        blocked: bool = False,
    ) -> TurnResult:
        if error_code is not None:
            self._errors[error_code] = self._errors.get(error_code, 0) + 1
        return TurnResult(
            success=success,
            response=response,
            thinking=thinking,
            conversation_id=session.conversation_id if session else None,
            message_count=len(session.history) if session else 0,
            timestamp=self._now(),
            error=_ERROR_TEXTS.get(error_code) if error_code else None,
            error_code=error_code,
            cycle_ok=cycle_ok,
            speaker_label=self._stored_name(identity),
            provider_id=provider_id,
            blocked=blocked,
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def _session_for(self, identity: SpeakerIdentity) -> Session:
        key = self.speaker_trust.session_key(identity)
        session = self.sessions.get(key)
        if session is not None:
            return session
        history: List[Dict[str, Any]] = []
        if self.repository is not None and (
            identity.authenticated or identity.is_default
        ):
            try:
                # Rows come back with their stored role; a plugin's reply is a
                # tool row and joins the window as a labelled tool turn.
                history = await rehydrate_recent_history(
                    self.repository,
                    key,
                    hours=float(self.config.rehydrate_hours),
                    limit=int(self.config.rehydrate_limit),
                    settings=self._cfg(),
                )
            except Exception as exc:  # noqa: BLE001 - counted, session starts empty
                self._count_failure("rehydrate", exc)
        session = self.sessions.start(
            key,
            str(uuid.uuid4()),
            history,
            speaker_label=self._stored_name(identity),
        )
        try:
            await self.persistence.store_conversation_start(
                session.conversation_id, key
            )
        except Exception as exc:  # noqa: BLE001 - counted
            self._count_failure("conversation_start", exc)
        return session

    # ------------------------------------------------------------------
    # The turn
    # ------------------------------------------------------------------

    async def send_message(
        self,
        message: str,
        speaker: Optional[str] = None,
        speaker_token: Optional[str] = None,
        *,
        stream_callback: Optional[Callable[[str], Any]] = None,
        request_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> TurnResult:
        """Run one turn; see the module docstring for the order of steps."""
        async with self._lock():
            self._turns += 1
            identity = self.speaker_trust.verify(speaker, speaker_token)
            with self.speaker_trust.use_identity(identity):
                try:
                    return await self._turn(
                        str(message or ""),
                        identity,
                        stream_callback=stream_callback,
                        request_id=request_id,
                        timeout_s=timeout_s,
                    )
                except asyncio.CancelledError:
                    self._count_cancelled()
                    raise

    def _count_cancelled(self) -> None:
        """Count a turn its caller cancelled (a generation bound of the API, a
        client that left); the cancellation itself propagates. Logged the
        first time and every hundredth."""
        self._cancelled += 1
        self._last_turn["error"] = "cancelled"
        if self._cancelled == 1 or self._cancelled % 100 == 0:
            logger.warning(
                "pipeline turn cancelled by its caller (%d so far)", self._cancelled
            )

    def _notify_turn_started(self) -> None:
        scheduler = self.sleep_scheduler
        if scheduler is not None:
            try:
                scheduler.notify_activity()
            except Exception as exc:  # noqa: BLE001 - counted
                self._count_failure("sleep_scheduler", exc)
        life = self.autonomous_life
        if life is not None:
            try:
                life.notify_turn_started()
            except Exception as exc:  # noqa: BLE001 - counted
                self._count_failure("autonomous_life", exc)

    def _notify_turn_finished(self) -> None:
        life = self.autonomous_life
        if life is not None:
            try:
                life.notify_turn_finished()
            except Exception as exc:  # noqa: BLE001 - counted
                self._count_failure("autonomous_life", exc)

    async def _gate(self, message: str, identity: SpeakerIdentity) -> Optional[str]:
        """Run the safety gate; the decision string when it denied, else None."""
        if self.safety is None or identity.trusted:
            return None
        try:
            result = await self.safety.pre_message_check(
                message,
                self._stored_name(identity),
                speaker_role=self.speaker_trust.safety_role(identity),
            )
        except Exception as exc:  # noqa: BLE001 - counted; the turn goes on
            self._count_failure("safety_gate", exc)
            return None
        if result.get("safe") is False:
            return str(result.get("decision") or "deny")
        if not result.get("checked"):
            self._count_failure(
                "safety_gate", RuntimeError("the safety check did not run")
            )
        return None

    async def _persist_turn(
        self,
        session: Session,
        speaker: str,
        role: str,
        content: str,
        *,
        to_speaker: Optional[str],
        snapshot: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
    ) -> None:
        try:
            result = await self.persistence.store_turn(
                session.conversation_id,
                speaker,
                role,
                content,
                session.turn_number + 1,
                snapshot=snapshot,
                topics=topics,
                to_speaker=to_speaker,
            )
        except Exception as exc:  # noqa: BLE001 - counted
            self._count_failure("persist", exc)
            return
        failed = getattr(result, "failed", None)
        if failed:
            self._count_failure(
                "persist", RuntimeError(f"{len(failed)} store write(s) failed")
            )

    async def _summarize(self, session: Session) -> None:
        if self.summarizer is None:
            return
        try:
            await self.summarizer.maybe_summarize(session)
        except Exception as exc:  # noqa: BLE001 - counted
            self._count_failure("summarize", exc)

    async def _recall(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        manager = self.memory_manager
        if manager is None:
            return []
        recalled = await manager.semantic_recall(query, top_k=top_k)
        return [dict(m) for m in recalled.get("memories") or ()]

    def _plugin_context(
        self, identity: SpeakerIdentity, message: str, session: Session
    ) -> PluginContext:
        return PluginContext(
            identity=identity,
            message=message,
            history=tuple(dict(e) for e in session.history),
            brain=self._plugin_brain,
            recall=self._recall,
        )

    def _apply_learning(self, identity: SpeakerIdentity, modulated: Any) -> Dict:
        report: Dict[str, Any] = {"applied": False}
        if self.learning is None:
            return report
        try:
            modulation = self.learning.get_modulation(
                speaker=self._stored_name(identity)
            )
        except Exception as exc:  # noqa: BLE001 - counted
            self._count_failure("learning", exc)
            return report
        if modulation is None:
            return report
        directives = [
            str(d) for d in (getattr(modulation, "directives", None) or ()) if str(d)
        ]
        for directive in directives:
            if directive not in modulated.behavioral_directives:
                modulated.behavioral_directives.append(directive)
        delta = getattr(modulation, "temperature_delta", 0.0) or 0.0
        low, high = self._temperature_bounds()
        params = modulated.generation_params
        base = params.get("temperature")
        if isinstance(base, (int, float)) and isinstance(delta, (int, float)):
            params["temperature"] = max(low, min(high, float(base) + float(delta)))
        report.update(
            {
                "applied": True,
                "directives": directives,
                "temperature_delta": float(delta),
                "temperature": params.get("temperature"),
                "drive_adjustments": dict(
                    getattr(modulation, "drive_adjustments", None) or {}
                ),
                "drive_adjustments_applied": False,
            }
        )
        return report

    async def _call_brain(
        self,
        window_history: Sequence[Dict[str, Any]],
        current_prompt: str,
        *,
        gathered: Any,
        modulated: Any,
        session: Session,
        identity: SpeakerIdentity,
        window_notes: Sequence[str],
        stream_callback: Optional[Callable[[str], Any]],
        timeout_s: float,
    ) -> BrainReply:
        sections = modulated.theory_sections.as_dict()
        async with self.flight.foreground():
            return await asyncio.wait_for(
                self.brain.think_with_context(
                    current_prompt,
                    conversation_history=list(window_history),
                    core_memory=gathered.core_memory or None,
                    ctm_thoughts=gathered.ctm_thoughts,
                    thought_gists=gathered.gists,
                    coherence_context=_plain_coherence(gathered.coherence),
                    retrieved_memories=gathered.memories,
                    retrieved_concepts=gathered.concepts,
                    consciousness_context=modulated.prompt_context or None,
                    generation_params=dict(modulated.generation_params),
                    behavioral_directives=list(modulated.behavioral_directives),
                    consciousness_self_awareness=(
                        modulated.consciousness_self_awareness or None
                    ),
                    conversation_summary=render_summary(
                        session.summary, session.pinned_quotes
                    ),
                    window_notes=list(window_notes),
                    speaker=identity.view(),
                    stream_callback=stream_callback,
                    **sections,
                ),
                timeout=timeout_s,
            )

    async def _turn(
        self,
        message: str,
        identity: SpeakerIdentity,
        *,
        stream_callback: Optional[Callable[[str], Any]],
        request_id: Optional[str],
        timeout_s: Optional[float],
    ) -> TurnResult:
        cfg = self._cfg()
        system = str(cfg.SYSTEM_NAME)
        stored_name = self._stored_name(identity)
        session = await self._session_for(identity)
        self._notify_turn_started()
        self._last_turn = {
            "request_id": request_id,
            "speaker_label": stored_name,
            # The trust outcome as verified by SpeakerTrust for this turn, so a
            # caller reads it from the runtime instead of inferring it.
            "authenticated": bool(identity.authenticated),
            "trusted": bool(identity.trusted),
            "safety_gate": (
                "absent"
                if self.safety is None
                else "bypassed" if identity.trusted else "applied"
            ),
        }
        self.last_modulated = None
        try:
            decision = await self._gate(message, identity)
            if decision is not None:
                self._last_turn["blocked"] = decision
                return self._result(
                    identity, session, success=False, error_code="blocked", blocked=True
                )

            now = self._now()
            session.history.append(
                {
                    "role": "user",
                    "content": message,
                    "speaker": stored_name,
                    "timestamp": now,
                }
            )
            await self._summarize(session)
            self.sessions.trim(session.key)
            await self._persist_turn(
                session, stored_name, "user", message, to_speaker=system
            )

            if not self.plugins.is_empty:
                outcome = await self.plugins.offer_message(
                    self._plugin_context(identity, message, session)
                )
                if outcome is not None and outcome.kind == "reply":
                    return await self._plugin_reply(
                        identity, session, outcome.tool_name, outcome.content, system
                    )
                if outcome is not None:
                    entry = tool_turn(
                        outcome.tool_name, ToolOutcome(True, outcome.content), now
                    )
                    session.history.append(entry)

            gathered = await gather_context(
                message,
                session,
                memory_manager=self.memory_manager,
                thought_store=self.thought_store,
                gist_store=self.gist_store,
                coherence=self.coherence,
                settings=cfg,
            )
            # A source that raised is a counted failure; the recall method's
            # own labels (an absent vector store or graph, a manager that was
            # not configured) describe the configuration and are reported.
            recall_notes: Dict[str, str] = {}
            for name, error in gathered.errors.items():
                if name in SOURCES:
                    self._count_failure(f"gather_{name}", RuntimeError(error))
                else:
                    recall_notes[name] = error
            self._last_turn["recall_notes"] = recall_notes
            tom_context = ""
            if self.theory_of_mind is not None:
                try:
                    tom_context = (
                        self.theory_of_mind.get_context_for_prompt(stored_name) or ""
                    )
                except Exception as exc:  # noqa: BLE001 - counted
                    self._count_failure("theory_of_mind", exc)

            raw = RawConversationContext(
                message=message,
                identity=identity,
                retrieved_memories=list(gathered.memories),
                retrieved_concepts=list(gathered.concepts),
                ctm_thoughts=list(gathered.ctm_thoughts),
                thought_gists=list(gathered.gists),
                conversation_history=[dict(e) for e in session.history],
                core_memory=dict(gathered.core_memory),
                coherence_context=dict(gathered.coherence),
                tom_context=tom_context,
                recent_messages=[dict(e) for e in session.history[-RECENT_MESSAGES:]],
                now=now,
            )
            modulated = await self.orchestrator.process(raw)
            self.last_modulated = modulated
            learning_report = self._apply_learning(identity, modulated)
            self._last_turn["learning"] = learning_report

            window = build_brain_window(
                session, identity, self.speaker_trust, settings=cfg, now=now
            )
            bound = float(timeout_s) if timeout_s else float(cfg.GENERATION_TIMEOUT_S)
            history = list(window.history)
            try:
                reply = await self._call_brain(
                    history,
                    window.current_prompt,
                    gathered=gathered,
                    modulated=modulated,
                    session=session,
                    identity=identity,
                    window_notes=window.window_notes,
                    stream_callback=stream_callback,
                    timeout_s=bound,
                )
                if not self.plugins.is_empty:
                    ctx = self._plugin_context(identity, message, session)

                    async def rerun(tool_turns: List[Dict[str, Any]]) -> str:
                        hop = await self._call_brain(
                            history + list(tool_turns),
                            window.current_prompt,
                            gathered=gathered,
                            modulated=modulated,
                            session=session,
                            identity=identity,
                            window_notes=window.window_notes,
                            stream_callback=stream_callback,
                            timeout_s=bound,
                        )
                        return hop.text

                    resolved = await self.plugins.resolve_response(
                        reply.text, ctx, rerun
                    )
                    for entry in resolved.tool_turns:
                        session.history.append(entry)
                    reply = BrainReply(
                        text=resolved.text,
                        finish_reason=reply.finish_reason,
                        output_tokens=reply.output_tokens,
                        input_tokens=reply.input_tokens,
                        provider_id=reply.provider_id,
                    )
            except NoAnswerError as exc:
                self._last_turn["error"] = f"no_answer:{exc.reason}"
                return self._result(
                    identity,
                    session,
                    success=False,
                    error_code="no_answer",
                    cycle_ok=modulated.cycle_ok,
                )
            except BrainUnavailableError:
                self._last_turn["error"] = "brain_unavailable"
                return self._result(
                    identity,
                    session,
                    success=False,
                    error_code="brain_unavailable",
                    cycle_ok=modulated.cycle_ok,
                )
            except (asyncio.TimeoutError, ProviderTimeoutError):
                self._last_turn["error"] = "generation_timeout"
                return self._result(
                    identity,
                    session,
                    success=False,
                    error_code="generation_timeout",
                    cycle_ok=modulated.cycle_ok,
                )

            reasoning, spoken = split_thinking(reply.text or "")
            spoken = strip_reasoning_blocks(spoken)
            if not spoken:
                self._last_turn["error"] = "no_answer:reasoning_only"
                return self._result(
                    identity,
                    session,
                    success=False,
                    error_code="no_answer",
                    cycle_ok=modulated.cycle_ok,
                )

            answered_at = self._now()
            session.history.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "speaker": system,
                    "timestamp": answered_at,
                }
            )
            topics = _coherence_topics(gathered.coherence)
            await self._persist_turn(
                session,
                system,
                "assistant",
                spoken,
                to_speaker=stored_name,
                snapshot=build_turn_snapshot(modulated, reply.provider_id),
                topics=topics or None,
            )
            turn_number = self.sessions.advance(session.key)

            if self.post_turn is not None:
                record = TurnRecord(
                    identity=identity,
                    message=message,
                    spoken=spoken,
                    thinking=reasoning,
                    state=modulated.consciousness_state,
                    modulated=modulated,
                    conversation_id=session.conversation_id,
                    turn_number=turn_number,
                    topics=tuple(topics),
                    history=tuple(history),
                    provider_id=reply.provider_id,
                )
                try:
                    report = await self.post_turn.run(record)
                    self._last_turn["post_turn"] = dict(
                        getattr(report, "steps", {}) or {}
                    )
                except Exception as exc:  # noqa: BLE001 - the reply stands
                    self._count_failure("post_turn", exc)
            self.sessions.trim(session.key)
            self._last_turn.update(
                {
                    "cycle_ok": modulated.cycle_ok,
                    "provider_id": reply.provider_id,
                    "absent_signals": dict(modulated.absent_signals),
                    "recall_method": gathered.recall_method,
                }
            )
            return self._result(
                identity,
                session,
                success=True,
                response=spoken,
                thinking=reasoning,
                cycle_ok=modulated.cycle_ok,
                provider_id=reply.provider_id,
            )
        finally:
            self._notify_turn_finished()

    async def _plugin_reply(
        self,
        identity: SpeakerIdentity,
        session: Session,
        tool_name: str,
        content: str,
        system: str,
    ) -> TurnResult:
        """A turn plugin answered: stored under its name, never as the system."""
        now = self._now()
        entry = {
            "role": TOOL_ROLE,
            "speaker": tool_name,
            "content": content,
            "timestamp": now,
        }
        session.history.append(entry)
        # A tool row under the plugin's own name: never an assistant row,
        # never the system's words. It is addressed like the system's replies.
        await self._persist_turn(
            session,
            tool_name,
            TOOL_ROLE,
            content,
            to_speaker=self._stored_name(identity),
        )
        self.sessions.advance(session.key)
        self._plugin_replies += 1
        self._last_turn["plugin_reply"] = tool_name
        return self._result(
            identity,
            session,
            success=True,
            response=content,
            provider_id=f"plugin:{tool_name}",
        )

    # ------------------------------------------------------------------
    # Ending conversations
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_name(session: Session) -> str:
        """The speaker as the session's stored rows name them. A session the
        pipeline did not open carries no stored label; its name is shown as a
        claim, never as a verified name."""
        if session.speaker_label:
            return session.speaker_label
        return f"[{session.key.split(':', 1)[-1]}]"

    async def _end_session(self, session: Session) -> bool:
        """Write the boilerplate summary memory and close the stores."""
        name = self._summary_name(session)
        count = len(session.history)
        ok = True
        if self.memory_manager is not None:
            try:
                await self.memory_manager.store_memory(
                    END_SUMMARY_FORMAT.format(name=name, count=count),
                    memory_type="episodic",
                    importance=float(self.config.end_summary_importance),
                )
            except Exception as exc:  # noqa: BLE001 - counted
                self._count_failure("end_summary", exc)
                ok = False
        try:
            await self.persistence.store_conversation_end(
                session.conversation_id, session.summary
            )
        except Exception as exc:  # noqa: BLE001 - counted
            self._count_failure("conversation_end", exc)
            ok = False
        return ok

    async def end_conversation(
        self, speaker: Optional[str] = None, speaker_token: Optional[str] = None
    ) -> bool:
        """End the speaker's live session; False when there is none."""
        identity = self.speaker_trust.verify(speaker, speaker_token)
        async with self._lock():
            session = self.sessions.end(self.speaker_trust.session_key(identity))
            if session is None:
                return False
            await self._end_session(session)
            return True

    async def end_all(self) -> int:
        """End every live session; returns how many were ended."""
        async with self._lock():
            ended = 0
            for session in self.sessions.sessions():
                self.sessions.end(session.key)
                await self._end_session(session)
                ended += 1
            return ended

    async def prune_sessions(self) -> int:
        """End the sessions idle past the manager's threshold."""
        async with self._lock():
            removed = self.sessions.prune()
            for session in removed:
                await self._end_session(session)
            return len(removed)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Counts only: turns, errors by code, step failures, the parts."""

        def part(obj: Any) -> Any:
            reader = getattr(obj, "status", None)
            if not callable(reader):
                return None if obj is None else "present"
            try:
                return reader()
            except Exception as exc:  # noqa: BLE001 - status never raises
                return {"error": type(exc).__name__}

        return {
            "turns": self._turns,
            "errors": dict(self._errors),
            "cancelled": self._cancelled,
            "failures": dict(self._failures),
            "plugin_replies": self._plugin_replies,
            "last_turn": {
                k: v for k, v in self._last_turn.items() if k != "speaker_label"
            },
            "sessions": self.sessions.status(),
            "flight": self.flight.status(),
            "persistence": part(self.persistence),
            "summarizer": part(self.summarizer),
            "post_turn": part(self.post_turn),
            "plugins": self.plugins.status(),
            "config": {
                "max_history": self.config.max_history,
                "summarize_threshold": self.config.summarize_threshold,
                "summarize_batch": self.config.summarize_batch,
                "end_summary_importance": self.config.end_summary_importance,
                "rehydrate_hours": self.config.rehydrate_hours,
                "rehydrate_limit": self.config.rehydrate_limit,
            },
        }


END_SUMMARY_PATTERN = re.compile(
    r"Conversation with .+ ended\.\s*\d+\s+messages? exchanged", re.IGNORECASE
)

__all__ = (
    "END_SUMMARY_FORMAT",
    "END_SUMMARY_PATTERN",
    "ERROR_CODES",
    "ConversationPipeline",
    "GatedBrain",
    "LLMFlight",
    "PipelineConfig",
)
