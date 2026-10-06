"""The composition root: everything the live pipeline runs, built from Settings.

``build_runtime`` is the one place that reads the store enable keys and wires
the components together. Nothing here is a default for the modules themselves;
each keeps its own injectable constructor, and the runtime hands them what
Settings say:

- The brain: ``LLMBrain.from_settings`` (or the injected one), initialized with
  a degraded provider allowed, and one ``LLMFlight``. Every turn-time user (the
  summarizer, the trust compass, the pipeline's main call) goes through the
  foreground side of the flight; every background user (the thought enhancer,
  dreams, the autonomous-life loop, memory consolidation, self-assessment) gets
  a ``GatedBrain`` on the background side; the thought machine's contention
  guard is ``flight.contended`` over the CTM config's window.
- The stores: each store-backed implementation is built only when its enable
  key is true (``POSTGRES_ENABLED``, ``MONGODB_ENABLED``, ``REDIS_ENABLED``,
  ``QDRANT_ENABLED``, ``NEO4J_ENABLED``), each in its own try, and reported as
  in memory otherwise; a store whose build failed is reported with its error
  and replaced by the in-memory implementation, loudly. The concept graph is
  the in-memory graph store unless ``NEO4J_ENABLED`` (a failed Neo4j build is
  reported and replaced by it), so concept capture, concept recall and the
  knowledge routes always have a graph. ``CTM_STORE=mongo``
  without ``MONGODB_ENABLED`` is a configuration error naming both keys. The
  memory stack comes from Phase 5's factory; the sentence embedder is built
  when the ``embeddings`` extra imports (it loads its model on first use), an
  injected embedder needs ``allow_non_semantic`` when it is not semantic.
- The workspace through ``runner_support.build_workspace`` with the overrides
  of ``PIPELINE_PRESET`` (validated against the schema) or the ``overrides``
  argument, the live pipeline's Damasio defaults (a wall-clock circadian model
  in ``TIMEZONE`` with ``WAKE_HOUR`` and ``SLEEP_HOUR``, ``agent_name`` from
  ``SYSTEM_NAME``) unless the overrides set them, a ``PhiTracker`` from
  ``build_phi_tracker``, one ``PADAffectModel``, one prospective simulator and
  one cross-modal binder shared by the cycle augmenter and the orchestrator;
  the augmenter's switches come from the overrides with the schema defaults,
  not pinned.
- The narration flags are read once (``NarrationFlags.from_settings``) and
  applied with ``apply_narration``; speaker trust is the only trust source and
  its ``trust_lookup`` reaches theory of mind and the trust compass.
- The neural orchestrator (its data manager only when MongoDB is on) with the
  continuous thought machine attached; the safety stack; learning, theory of
  mind, goals, the attention observer, the emotion extractor and dynamics, the
  verifier with its state assembler, the snapshot service, identity continuity,
  the developmental machinery only when ``developmental_enabled()``, the sleep
  scheduler (the memory manager's consolidation plus the gist retention refresh
  as its consolidator, the pruner, the machine as its thought stream) and the
  autonomous-life loop when enabled.
- The hub: one ``ConsciousnessStateManager`` under ``STATE_DIR`` that registers
  every component implementing the persistence contract under ``HUB_NAMES``,
  each once and only when built; ``NOT_REGISTERED`` names the stateful parts
  that persist elsewhere or ride inside another component's state. The emotion
  bridge and the memory consolidator the neural orchestrator builds on its
  first experience are registered when they appear.
- Background tasks (``BackgroundTask``): the machine, the sleep scheduler and
  the autonomous-life loop per their keys; the baseline heartbeat and the
  wellbeing sampler every 60 s; the Phi idle sampler every
  ``PHI_IDLE_SAMPLE_S`` and self-assessment every ``SELF_ASSESSMENT_INTERVAL_S``
  when positive (one round trip through the background gate, the parsed report
  compared with the latest assessment and handed to the orchestrator; no answer
  records nothing); a full state save every 300 s; session pruning hourly.

``Runtime.start`` restores the hub and starts the tasks; ``Runtime.stop`` stops
the tasks, ends the open conversations, saves the hub, shuts down the neural
orchestrator, the memory stack, the store clients and the brain, in that
order. ``build_runtime`` calls ``setup_logging`` first.

Config keys read here: the five store enable keys, ``CTM_ENABLED``,
``CTM_STORE``, ``PIPELINE_PRESET``, ``SLEEP_SCHEDULER_ENABLED``,
``AUTONOMOUS_LIFE_ENABLED``, ``PHI_IDLE_SAMPLE_S``,
``SELF_ASSESSMENT_INTERVAL_S``, ``STATE_DIR``, ``RESULTS_DIR``, ``TIMEZONE``,
``WAKE_HOUR``, ``SLEEP_HOUR``, ``SYSTEM_NAME``, ``MONGODB_DB`` and ``DEBUG``;
every other key is read by the component that owns it. Persisted state lives
in the hub's files and the stores. Conversations, thoughts, memories and
speaker names are private at runtime and never logged here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import mtc
from mtc.brain.brain import LLMBrain
from mtc.brain.protocol import ConfigurationError, NoAnswerError, strip_reasoning_blocks
from mtc.consciousness.attention_observer import AttentionGrabObserver
from mtc.consciousness.consciousness_integration import ConsciousnessIntegration
from mtc.consciousness.conversation_coherence import ConversationCoherence
from mtc.consciousness.crossmodal_binding import CrossModalBinder
from mtc.consciousness.developmental_modulator import (
    DevelopmentalModulator,
    developmental_enabled,
)
from mtc.consciousness.emotion_dynamics import EmotionDynamics
from mtc.consciousness.emotion_extractor import (
    EmotionExtractor,
    InMemoryEmotionAverages,
)
from mtc.consciousness.identity_continuity import IdentityContinuityManager
from mtc.consciousness.intrinsic_goals import IntrinsicGoalSystem
from mtc.consciousness.learning.learning_system import LearningSystem
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.perceptual_reality_monitor import PerceptualRealityMonitor
from mtc.consciousness.prospective_memory import ProspectiveSimulator
from mtc.consciousness.self_assessment import ConsciousnessSelfAssessment
from mtc.consciousness.sleep_scheduler import SleepScheduler
from mtc.consciousness.snapshot_service import BACKUP_SUBDIR, SnapshotService
from mtc.consciousness.state_persistence import ConsciousnessStateManager
from mtc.consciousness.theory_of_mind import TheoryOfMind
from mtc.consciousness.verification_framework import ConsciousnessVerifier
from mtc.consciousness.verification_state_assembler import ConsciousnessStateAssembler
from mtc.core.config import Settings, get_settings
from mtc.core.embeddings import EmbeddingUnavailableError
from mtc.core.logger import setup_logging
from mtc.core.narration import NarrationFlags
from mtc.core.param_overrides import load_overrides, preset_path
from mtc.core.paths import resolve_path
from mtc.core.profiler import StageProfiler
from mtc.core.runner_support import build_phi_tracker, build_workspace
from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager
from mtc.memory.factory import MemoryStack, build_memory_manager
from mtc.memory.inmemory import (
    InMemoryAlertStore,
    InMemoryAttentionGrabStore,
    InMemoryConversationRepository,
    InMemoryConversationTurnStore,
    InMemoryDreamStore,
    InMemoryGoalStore,
    InMemoryGraphStore,
    InMemoryLearningStore,
    InMemoryNeuralRetentionStore,
    InMemorySafetyAuditStore,
    InMemorySnapshotStore,
    InMemorySpeakerModelStore,
    InMemoryThoughtGistStore,
    InMemoryWorkingMemory,
)
from mtc.memory.memory_consolidation_pipeline import MemoryConsolidationPipeline
from mtc.memory.memory_pruner import MemoryPruner
from mtc.monitoring.alert_manager import AlertManager
from mtc.monitoring.wellbeing_monitor import ResilientSampler, WellbeingMonitor
from mtc.pipeline.consciousness_helpers import ConsciousnessLevelCache
from mtc.pipeline.consciousness_orchestrator import (
    ConsciousnessOrchestrator,
    OrchestratorConfig,
    apply_narration,
)
from mtc.pipeline.conversation_pipeline import (
    ConversationPipeline,
    GatedBrain,
    LLMFlight,
    PipelineConfig,
)
from mtc.pipeline.context_summarizer import ContextSummarizer
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig
from mtc.pipeline.message_persistence import MessagePersistence
from mtc.pipeline.plugins import PluginRegistry
from mtc.pipeline.post_turn import PostTurnConfig, PostTurnIntegrator
from mtc.pipeline.session_state import SessionManager
from mtc.pipeline.speaker_trust import SpeakerTrust
from mtc.privacy.privacy_integration import IntegratedPrivacySystem
from mtc.safety.integration import SafetySystemsIntegration
from mtc.safety.safety_monitor import SafetyMonitor
from mtc.safety.trust_compass import TrustCompass

logger = logging.getLogger(__name__)

# Hub names of the components that implement the persistence contract. A
# component is registered only when it was built, and only once.
HUB_NAMES: Tuple[str, ...] = (
    "gwt",
    "phi",
    "pad_affect",
    "orchestrator",
    "augmenter",
    "rpt",
    "causal_emergence",
    "neural",
    "ctm",
    "ctm_buffer",
    "ctm_generator",
    "prm",
    "emotion_dynamics",
    "emotion_extractor",
    "sleep_scheduler",
    "autonomous_life",
    "wellbeing",
    "emotion_bridge",
    "memory_consolidator",
)

# Stateful parts of the runtime that are deliberately not hub modules, each
# with the reason (the hub test checks every field against this list).
NOT_REGISTERED: Tuple[Tuple[str, str], ...] = (
    (
        "prospective",
        "the one simulator is shared with the augmenter and rides in the "
        "augmenter's state dict",
    ),
    ("crossmodal", "the binder rides in the augmenter's state dict"),
    (
        "emotion_averages",
        "the in-memory averages ride in the emotion extractor's state dict",
    ),
    (
        "identity_continuity",
        "writes its identity markers under AGENT_SPACE_DIR itself",
    ),
    ("theory_of_mind", "persists through its speaker model store"),
    ("learning", "persists through its learning store"),
    ("goals", "persists through its goal store"),
    ("snapshots", "persists through its snapshot store"),
    ("observer", "persists through its attention grab store"),
    ("memory", "store-backed; the memory stack persists through its stores"),
    ("brain", "the provider registry persists the active provider itself"),
)

MINUTE_S = 60.0
STATE_SAVE_S = 300.0
SESSION_PRUNE_S = 3600.0
SELF_ASSESSMENT_MAX_TOKENS = 900
SELF_ASSESSMENT_TEMPERATURE = 0.3
THOUGHT_PASS_LIMIT = 500
EXPERIMENTS_DIR = Path(mtc.__file__).resolve().parents[1] / "experiments"
STORE_NAMES: Tuple[str, ...] = ("postgres", "mongodb", "redis", "qdrant", "neo4j")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class StoreBuilders:
    """Replacements for the client-level constructors (tests, hand-built
    stores). Each takes the Settings and returns the client-level object the
    runtime would otherwise create: an asyncpg pool (async), a pymongo client,
    a ``RedisManager``, a ``VectorStore`` (with the embedder) or a
    ``GraphStore``."""

    postgres_pool: Optional[Callable[[Settings], Awaitable[Any]]] = None
    mongo_client: Optional[Callable[[Settings], Any]] = None
    redis_manager: Optional[Callable[[Settings], Any]] = None
    vector: Optional[Callable[[Settings, Any], Any]] = None
    graph: Optional[Callable[[Settings], Any]] = None


@dataclass
class BackgroundTask:
    """One background loop: a directly testable ``run_once`` plus one task.

    A component that owns its own loop (the thought machine, the sleep
    scheduler, the autonomous-life loop) passes ``start_hook`` and
    ``stop_hook``; ``run_once`` is still its single step for tests. Every
    failure of ``run_once`` is counted and logged (the first and every
    hundredth) and the loop never dies of one, except under ``DEBUG`` (read
    from ``settings``, the process Settings when none was given), where the
    failure is raised instead.
    """

    name: str
    interval_s: float
    run_once: Callable[[], Any]
    start_hook: Optional[Callable[[], Any]] = None
    stop_hook: Optional[Callable[[], Any]] = None
    settings: Optional[Settings] = None
    runs: int = 0
    failures: int = 0
    last_error: Optional[str] = None
    _task: Optional["asyncio.Task[None]"] = field(default=None, repr=False)
    _owned_loop: bool = field(default=False, repr=False)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _debug(self) -> bool:
        cfg = self.settings if self.settings is not None else get_settings()
        return bool(cfg.DEBUG)

    async def tick(self) -> Any:
        """Run one step, counting a failure instead of raising (raising under
        ``DEBUG``)."""
        self.runs += 1
        try:
            return await _maybe_await(self.run_once())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - counted, the loop goes on
            self.failures += 1
            self.last_error = _describe(exc)
            if self.failures == 1 or self.failures % 100 == 0:
                logger.warning(
                    "background task %s failed (%d so far): %s",
                    self.name,
                    self.failures,
                    type(exc).__name__,
                )
            if self._debug():
                raise
            return None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(float(self.interval_s))
            await self.tick()

    async def start(self) -> None:
        if self.start_hook is not None:
            await _maybe_await(self.start_hook())
            return
        if self.running:
            return
        self._owned_loop = True
        self._task = asyncio.get_running_loop().create_task(
            self._loop(), name=f"mtc-{self.name}"
        )

    async def stop(self) -> None:
        if self.stop_hook is not None:
            await _maybe_await(self.stop_hook())
            return
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "interval_s": self.interval_s,
            "running": self.running or self.start_hook is not None,
            "owns_loop": self.start_hook is None,
            "runs": self.runs,
            "failures": self.failures,
            "last_error": self.last_error,
        }


class _ThemeSource:
    """The sleep scheduler's theme seam over the thought buffer."""

    def __init__(self, buffer: Any) -> None:
        self._buffer = buffer

    def get_recent_themes(self, n: int) -> List[str]:
        themes = list(self._buffer.get_recent_themes() or [])
        return themes[: max(0, int(n))]


class _SleepConsolidator:
    """The scheduler's consolidator seam: the memory manager's pass, then the
    thought pass and the gist retention refresh, so expiry reads fresh
    retention (the pruner runs after this)."""

    def __init__(
        self,
        manager: Any,
        pipeline: Optional[MemoryConsolidationPipeline],
        thought_store: Any,
        gist_store: Any,
    ) -> None:
        self.manager = manager
        self.pipeline = pipeline
        self.thought_store = thought_store
        self.gist_store = gist_store
        self.last_report: Dict[str, Any] = {}

    @staticmethod
    def _thought_rows(records: Sequence[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "content": record.content,
                    "thought_id": record.thought_id,
                    "timestamp": record.timestamp,
                    "emotional_context": dict(record.emotional_context or {}),
                    "expressed": bool(record.should_express),
                }
            )
        return rows

    async def consolidate(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        counts.update(await self.manager.consolidate())
        if self.pipeline is not None and self.thought_store is not None:
            records = await self.thought_store.recent(THOUGHT_PASS_LIMIT)
            report = await self.pipeline.run_thought_consolidation(
                self._thought_rows(records)
            )
            counts["thoughts_processed"] = int(report.thoughts_processed)
            counts["gists_created"] = int(report.gists_created)
            # The thought store protocol has no delete seam; the clusters a
            # gist replaced stay in the store and are reported, not removed.
            counts["gist_replaced_thoughts_kept"] = len(report.delete_ids)
            self.last_report = report.as_dict() if hasattr(report, "as_dict") else {}
        if self.pipeline is not None and self.gist_store is not None:
            counts["gists_refreshed"] = int(
                await self.pipeline.refresh_gist_retention()
            )
        return counts


@dataclass
class Runtime:
    """Every composed component, the hub, the background tasks and the
    lifecycle. Fields are None for components that were not built."""

    settings: Settings
    brain: Any = None
    flight: Optional[LLMFlight] = None
    workspace: Any = None
    phi_tracker: Any = None
    pad_model: Any = None
    neural: Any = None
    orchestrator: Any = None
    augmenter: Any = None
    speaker_trust: Optional[SpeakerTrust] = None
    pipeline: Optional[ConversationPipeline] = None
    memory: Optional[MemoryStack] = None
    repository: Any = None
    working_memory: Any = None
    turn_store: Any = None
    thought_store: Any = None
    gist_store: Any = None
    ctm: Any = None
    sleep_scheduler: Any = None
    autonomous_life: Any = None
    safety: Any = None
    compass: Any = None
    learning: Any = None
    theory_of_mind: Any = None
    goals: Any = None
    observer: Any = None
    snapshots: Any = None
    hub: Optional[ConsciousnessStateManager] = None
    background: List[BackgroundTask] = field(default_factory=list)
    rpt: Any = None
    causal_emergence: Any = None
    prm: Any = None
    prospective: Any = None
    crossmodal: Any = None
    emotion_dynamics: Any = None
    emotion_extractor: Any = None
    emotion_averages: Any = None
    verifier: Any = None
    integration: Any = None
    identity_continuity: Any = None
    developmental: Any = None
    developmental_age: Any = None
    self_assessment: Any = None
    consolidation: Any = None
    pruner: Any = None
    level_cache: Any = None
    data_manager: Any = None
    embedder: Any = None
    embedder_kind: str = "none"
    narration: Optional[NarrationFlags] = None
    narration_applied: Dict[str, bool] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    stores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    store_clients: List[Tuple[str, Any]] = field(default_factory=list)
    registered: Dict[str, Any] = field(default_factory=dict)
    thought_listeners: List[Callable[[Dict[str, Any]], Any]] = field(
        default_factory=list
    )
    listener_failures: int = 0
    started_at: Optional[datetime] = None
    stopped: bool = False
    last_stop_order: List[str] = field(default_factory=list)
    _clock: Clock = field(default=_utc_now, repr=False)
    _background_brain: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Hub registration
    # ------------------------------------------------------------------

    def register(self, name: str, component: Any) -> bool:
        """Register a built component under one of ``HUB_NAMES`` once."""
        if component is None or self.hub is None:
            return False
        if name not in HUB_NAMES:
            raise ValueError(f"{name!r} is not one of HUB_NAMES")
        if name in self.registered:
            return False
        if any(component is other for other in self.registered.values()):
            return False
        self.hub.register(name, component)
        self.registered[name] = component
        return True

    def register_late_components(self) -> List[str]:
        """Register the neural orchestrator's bridge and consolidator once
        they exist (they are built on the first experience)."""
        added: List[str] = []
        neural = self.neural
        if neural is None:
            return added
        for name, attribute in (
            ("emotion_bridge", "lsm_bridge"),
            ("memory_consolidator", "memory_consolidator"),
        ):
            component = getattr(neural, attribute, None)
            if component is not None and self.register(name, component):
                added.append(name)
        return added

    def components(self) -> Dict[str, Any]:
        """The runtime's fields by name (for the hub test's introspection)."""
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    # ------------------------------------------------------------------
    # Thought broadcast
    # ------------------------------------------------------------------

    def add_thought_listener(self, listener: Callable[[Dict[str, Any]], Any]) -> None:
        """Receive every expressed thought the machine broadcasts."""
        self.thought_listeners.append(listener)

    async def broadcast_thought(self, payload: Dict[str, Any]) -> None:
        """Hand one expressed thought to every listener. A listener's failure
        is counted (``listener_failures``, in ``status()``), logged the first
        time and every hundredth, and raised under ``DEBUG``; otherwise the
        remaining listeners still receive the thought."""
        for listener in list(self.thought_listeners):
            try:
                await _maybe_await(listener(payload))
            except Exception as exc:  # noqa: BLE001 - counted; the rest still run
                self.listener_failures += 1
                if self.listener_failures == 1 or self.listener_failures % 100 == 0:
                    logger.warning(
                        "thought listener failed (%d so far): %s",
                        self.listener_failures,
                        type(exc).__name__,
                    )
                if bool(self.settings.DEBUG):
                    raise

    # The thought machine is wired to this name; the public method is the same.
    _broadcast_thought = broadcast_thought

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> Dict[str, Any]:
        """Restore the hub and start every background task."""
        restored: Dict[str, Any] = {}
        if self.hub is not None:
            restored = self.hub.restore_all()
        for task in self.background:
            await task.start()
        self.started_at = self._clock()
        self.stopped = False
        logger.info("runtime started with %d background task(s)", len(self.background))
        return restored

    async def _close_client(self, name: str, client: Any) -> None:
        for method in ("close", "disconnect", "shutdown"):
            closer = getattr(client, method, None)
            if callable(closer):
                try:
                    await _maybe_await(closer())
                except Exception as exc:  # noqa: BLE001 - shutdown keeps going
                    logger.warning(
                        "store client %s did not close cleanly: %s",
                        name,
                        type(exc).__name__,
                    )
                return

    async def stop(self) -> Dict[str, Any]:
        """Stop the tasks, end conversations, save, then shut the parts down."""
        order: List[str] = []
        report: Dict[str, Any] = {}
        for task in reversed(self.background):
            await task.stop()
        order.append("background_tasks")
        if self.pipeline is not None:
            report["conversations_ended"] = await self.pipeline.end_all()
            order.append("conversations")
        if self.hub is not None:
            self.register_late_components()
            report["saved"] = self.hub.save_all()
            order.append("hub_save")
        if self.neural is not None:
            await self.neural.shutdown()
            order.append("neural")
        if self.memory is not None:
            await self.memory.close()
            order.append("memory")
        for name, client in self.store_clients:
            await self._close_client(name, client)
        order.append("store_clients")
        if self.brain is not None and callable(getattr(self.brain, "shutdown", None)):
            await self.brain.shutdown()
            order.append("brain")
        self.last_stop_order = order
        self.stopped = True
        logger.info("runtime stopped")
        return report

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Every part's status; counts and labels, never conversation text."""

        def read(obj: Any, *names: str) -> Any:
            if obj is None:
                return None
            for name in names:
                reader = getattr(obj, name, None)
                if callable(reader):
                    try:
                        return reader()
                    except Exception as exc:  # noqa: BLE001 - status never raises
                        return {"error": type(exc).__name__}
            return "present"

        ctm_status = None
        if self.neural is not None:
            ctm_status = read(self.neural, "get_continuous_thought_status")
        elif self.ctm is not None:
            ctm_status = read(self.ctm, "get_status")
        return {
            "started_at": (
                self.started_at.isoformat() if self.started_at is not None else None
            ),
            "stopped": self.stopped,
            "brain": read(self.brain, "status"),
            "flight": read(self.flight, "status"),
            "pipeline": read(self.pipeline, "status"),
            "orchestrator": read(self.orchestrator, "status"),
            "first_run": read(self.orchestrator, "first_run_report"),
            "ctm": ctm_status,
            "stores": {name: dict(report) for name, report in self.stores.items()},
            "embedder": self.embedder_kind,
            "memory": read(getattr(self.memory, "manager", None), "status"),
            "safety": read(self.safety, "status"),
            "speaker_trust": read(self.speaker_trust, "status"),
            "sleep_scheduler": read(self.sleep_scheduler, "get_status"),
            "autonomous_life": read(self.autonomous_life, "get_status"),
            "narration": self.narration.as_dict() if self.narration else None,
            "narration_applied": dict(self.narration_applied),
            "hub": read(self.hub, "get_status"),
            "registered": sorted(self.registered),
            "not_registered": [name for name, _ in NOT_REGISTERED],
            "background": [task.status() for task in self.background],
            "thought_listener_failures": self.listener_failures,
            "preset": self.settings.PIPELINE_PRESET or None,
        }


# ----------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------


def _store_report(enabled: bool, backend: str, **extra: Any) -> Dict[str, Any]:
    report = {"enabled": bool(enabled), "backend": backend, "error": None}
    report.update(extra)
    return report


def _neo4j_graph(settings: Settings) -> Any:
    """The Neo4j concept graph (``NEO4J_*`` keys), once it answered connect()."""
    from mtc.knowledge.graph_manager import KnowledgeGraphManager

    graph = KnowledgeGraphManager(settings=settings)
    graph.connect()
    return graph


def _store_failed(report: Dict[str, Any], exc: BaseException) -> None:
    report["backend"] = "memory"
    report["error"] = _describe(exc)
    logger.error(
        "store %s could not be built (%s); using the in-memory implementation",
        report.get("name", "?"),
        type(exc).__name__,
    )


def load_pipeline_overrides(settings: Settings) -> Dict[str, Any]:
    """The preset's overrides, validated against the schema; {} without one."""
    name = (settings.PIPELINE_PRESET or "").strip()
    if not name:
        return {}
    schema = EXPERIMENTS_DIR / "param_schema.toml"
    try:
        path = preset_path(str(EXPERIMENTS_DIR / "presets"), name)
        return load_overrides(path, str(schema))
    except (FileNotFoundError, ValueError) as exc:
        raise ConfigurationError(
            f"PIPELINE_PRESET {name!r} could not be loaded: {exc}"
        ) from exc


def _default_embedder() -> Tuple[Any, str]:
    if importlib.util.find_spec("sentence_transformers") is None:
        return None, "none (the embeddings extra is not installed)"
    from mtc.core.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(), "sentence-transformers (loads on first use)"


def _drive_reader(workspace: Any) -> Callable[[], Dict[str, Dict[str, float]]]:
    def read() -> Dict[str, Dict[str, float]]:
        drives = getattr(
            getattr(workspace, "active_inference", None), "homeostatic_drives", None
        )
        if drives is None:
            return {}
        return dict(drives.get_drive_state())

    return read


def _recall_adapter(manager: Any) -> Callable[[str, int], Awaitable[List[str]]]:
    async def recall(query: str, count: int) -> List[str]:
        recalled = await manager.semantic_recall(query, top_k=int(count))
        return [
            str(m.get("content") or "")
            for m in recalled.get("memories") or ()
            if str(m.get("content") or "").strip()
        ]

    return recall


async def build_runtime(
    settings: Optional[Settings] = None,
    *,
    brain: Any = None,
    embedder: Any = None,
    allow_non_semantic: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    clock: Optional[Clock] = None,
    substrates: bool = True,
    builders: Optional[StoreBuilders] = None,
) -> Runtime:
    """Compose the runtime from Settings; see the module docstring.

    ``substrates`` False leaves the neural orchestrator's emotion bridge and
    memory consolidator out (a default build costs tens of seconds and
    hundreds of megabytes on the first experience); ``builders`` replaces the
    client-level constructors of the enabled stores.
    """
    cfg = settings if settings is not None else get_settings()
    setup_logging(cfg)
    now: Clock = clock or _utc_now
    picks = builders or StoreBuilders()
    runtime = Runtime(settings=cfg, _clock=now)
    runtime.flight = LLMFlight()
    flight = runtime.flight

    # -- brain ----------------------------------------------------------
    if brain is None:
        brain = LLMBrain.from_settings(cfg)
    runtime.brain = brain
    if callable(getattr(brain, "initialize", None)) and (
        getattr(brain, "available", None) is None
    ):
        health = await brain.initialize()
        if health is not None and not getattr(health, "ok", True):
            logger.warning("the brain starts degraded; turns report it honestly")
    background_brain = GatedBrain(brain, flight, background=True)
    foreground_brain = GatedBrain(brain, flight, background=False)
    runtime._background_brain = background_brain

    # -- stores ---------------------------------------------------------
    for name in STORE_NAMES:
        runtime.stores[name] = _store_report(False, "memory", name=name)
    if embedder is not None:
        if not getattr(embedder, "semantic", False) and not allow_non_semantic:
            raise EmbeddingUnavailableError(
                "the injected embedder is not semantic; pass allow_non_semantic=True "
                "to use it (thoughts and vectors then measure word overlap only)"
            )
        runtime.embedder = embedder
        runtime.embedder_kind = (
            "semantic" if getattr(embedder, "semantic", False) else "non_semantic"
        )
    else:
        runtime.embedder, runtime.embedder_kind = _default_embedder()
    if cfg.CTM_STORE == "mongo" and not cfg.MONGODB_ENABLED:
        raise ConfigurationError(
            "CTM_STORE is mongo but MONGODB_ENABLED is false; enable MongoDB or "
            "set CTM_STORE to memory"
        )

    postgres_report = runtime.stores["postgres"]
    postgres_report["enabled"] = bool(cfg.POSTGRES_ENABLED)
    qdrant_report = runtime.stores["qdrant"]
    qdrant_report["enabled"] = bool(cfg.QDRANT_ENABLED)
    neo4j_report = runtime.stores["neo4j"]
    neo4j_report["enabled"] = bool(cfg.NEO4J_ENABLED)

    def concept_graph(active: Settings) -> Any:
        # Neo4j only when NEO4J_ENABLED; otherwise, and when its build fails
        # (reported with the error, loudly), the graph stays in process memory.
        if not active.NEO4J_ENABLED:
            return InMemoryGraphStore()
        try:
            return (picks.graph or _neo4j_graph)(active)
        except Exception as exc:  # noqa: BLE001 - reported, in memory instead
            _store_failed(neo4j_report, exc)
            return InMemoryGraphStore()

    factory_kwargs: Dict[str, Any] = dict(
        vectors=bool(cfg.QDRANT_ENABLED),
        graph=True,
        embedder=runtime.embedder,
        pool_builder=picks.postgres_pool,
        vector_builder=picks.vector,
        graph_builder=concept_graph,
        clock=now,
    )
    stack: Optional[MemoryStack] = None
    if cfg.POSTGRES_ENABLED:
        try:
            stack = await build_memory_manager(cfg, postgres=True, **factory_kwargs)
            postgres_report["backend"] = "postgres"
        except Exception as exc:  # noqa: BLE001 - reported, in memory instead
            _store_failed(postgres_report, exc)
    if stack is None:
        stack = await build_memory_manager(cfg, postgres=False, **factory_kwargs)
    for key, report in (("vectors", qdrant_report), ("graph", neo4j_report)):
        component = stack.components.get(key, {})
        if report["enabled"]:
            if component.get("last_error"):
                report["backend"] = "memory"
                report["error"] = component["last_error"]
            else:
                report["backend"] = str(component.get("backend") or key)
    runtime.memory = stack
    manager: EnhancedMemoryManager = stack.manager
    await manager.initialize()
    if stack.pool is not None:
        runtime.store_clients.append(("postgres_pool", stack.pool))

    repository: Any = None
    snapshot_store: Any = None
    safety_audit_store: Any = None
    if stack.pool is not None:
        from mtc.memory.conversation_store import PostgresConversationRepository
        from mtc.memory.postgres_stores import PostgresSnapshotStore
        from mtc.safety.stores import PostgresSafetyAuditStore

        try:
            repository = PostgresConversationRepository(stack.pool, clock=now)
            snapshot_store = PostgresSnapshotStore(stack.pool)
            safety_audit_store = PostgresSafetyAuditStore(stack.pool)
        except Exception as exc:  # noqa: BLE001 - reported
            postgres_report["error"] = _describe(exc)
            repository = snapshot_store = safety_audit_store = None
    if repository is None:
        repository = InMemoryConversationRepository(clock=now)
    if snapshot_store is None:
        snapshot_store = InMemorySnapshotStore()
    if safety_audit_store is None:
        safety_audit_store = InMemorySafetyAuditStore(clock=now)
    runtime.repository = repository

    mongo_report = runtime.stores["mongodb"]
    mongo_report["enabled"] = bool(cfg.MONGODB_ENABLED)
    data_manager: Any = None
    turn_store: Any = None
    learning_store: Any = None
    speaker_store: Any = None
    goal_store: Any = None
    grab_store: Any = None
    dream_store: Any = None
    alert_store: Any = None
    if cfg.MONGODB_ENABLED:
        try:
            from mtc.memory.clients import create_mongo_client
            from mtc.memory.mongo_stores import (
                MongoAttentionGrabStore,
                MongoDreamStore,
                MongoGoalStore,
                MongoLearningStore,
                MongoSpeakerModelStore,
            )
            from mtc.memory.resilient_wrappers import ResilientMongoStore
            from mtc.monitoring.stores import MongoAlertStore
            from mtc.neural.mongodb_schemas import (
                MongoConversationStore,
                NeuralDataManager,
            )

            client = (
                picks.mongo_client(cfg)
                if picks.mongo_client is not None
                else create_mongo_client(cfg)
            )
            database = client[cfg.MONGODB_DB]
            data_manager = NeuralDataManager(client=client, settings=cfg)
            turn_store = ResilientMongoStore(
                MongoConversationStore(database=database, settings=cfg), settings=cfg
            )
            learning_store = MongoLearningStore(database=database, settings=cfg)
            speaker_store = MongoSpeakerModelStore(database=database, settings=cfg)
            goal_store = MongoGoalStore(database=database, settings=cfg)
            grab_store = MongoAttentionGrabStore(database=database, settings=cfg)
            dream_store = MongoDreamStore(database=database, settings=cfg)
            alert_store = MongoAlertStore(database=database, settings=cfg, clock=now)
            runtime.store_clients.append(("mongo_client", client))
            mongo_report["backend"] = "mongodb"
        except Exception as exc:  # noqa: BLE001 - reported, in memory instead
            _store_failed(mongo_report, exc)
            data_manager = turn_store = None
            learning_store = speaker_store = goal_store = None
            grab_store = dream_store = alert_store = None
    runtime.data_manager = data_manager
    if turn_store is None:
        turn_store = InMemoryConversationTurnStore(clock=now)
    if learning_store is None:
        learning_store = InMemoryLearningStore()
    if speaker_store is None:
        speaker_store = InMemorySpeakerModelStore()
    if goal_store is None:
        goal_store = InMemoryGoalStore()
    if grab_store is None:
        grab_store = InMemoryAttentionGrabStore()
    if dream_store is None:
        dream_store = InMemoryDreamStore()
    if alert_store is None:
        alert_store = InMemoryAlertStore(clock=now)
    runtime.turn_store = turn_store

    redis_report = runtime.stores["redis"]
    redis_report["enabled"] = bool(cfg.REDIS_ENABLED)
    working_memory: Any = None
    emotion_averages: Any = None
    if cfg.REDIS_ENABLED:
        try:
            from mtc.memory.redis_manager import RedisManager
            from mtc.memory.resilient_wrappers import ResilientRedisManager

            redis_manager = (
                picks.redis_manager(cfg)
                if picks.redis_manager is not None
                else RedisManager.from_settings(cfg)
            )
            working_memory = ResilientRedisManager(redis_manager, settings=cfg)
            emotion_averages = working_memory
            runtime.store_clients.append(("redis_manager", redis_manager))
            redis_report["backend"] = "redis"
        except Exception as exc:  # noqa: BLE001 - reported, in memory instead
            _store_failed(redis_report, exc)
            working_memory = None
    if working_memory is None:
        working_memory = InMemoryWorkingMemory(
            capacity=int(cfg.WORKING_MEMORY_CAPACITY), clock=now
        )
    if emotion_averages is None:
        emotion_averages = InMemoryEmotionAverages()
    runtime.working_memory = working_memory
    runtime.emotion_averages = emotion_averages

    thought_store: Any = None
    gist_store: Any = None
    retention_store: Any = None
    if cfg.CTM_STORE == "mongo" and data_manager is not None:
        from mtc.consciousness.continuous.config import CTMConfig
        from mtc.consciousness.continuous.store import MongoThoughtStore

        ctm_config = CTMConfig.load(None, None, settings=cfg)
        thought_store = MongoThoughtStore(
            data_manager, cfg, now, surfacing=ctm_config.surfacing
        )
        gist_store = data_manager
        retention_store = data_manager
    else:
        from mtc.consciousness.continuous.store import InMemoryThoughtStore

        if cfg.CTM_STORE == "mongo":
            mongo_report["error"] = (
                mongo_report.get("error") or "CTM_STORE mongo fell back to memory"
            )
        thought_store = InMemoryThoughtStore(clock=now)
        gist_store = InMemoryThoughtGistStore()
        retention_store = InMemoryNeuralRetentionStore(gists=gist_store)
    runtime.thought_store = thought_store
    runtime.gist_store = gist_store

    # -- overrides, workspace, phi, pad, augmenter -----------------------
    layered: Dict[str, Any] = (
        dict(overrides) if overrides is not None else load_pipeline_overrides(cfg)
    )
    runtime.overrides = layered
    damasio_defaults: Dict[str, Any] = {
        "circadian": "wall_clock",
        "timezone": str(cfg.TIMEZONE),
        "wake_hour": int(cfg.WAKE_HOUR),
        "sleep_hour": int(cfg.SLEEP_HOUR),
        "agent_name": str(cfg.SYSTEM_NAME),
        "clock": now,
    }
    workspace_overrides = dict(layered)
    workspace_overrides["damasio"] = {
        **damasio_defaults,
        **dict(layered.get("damasio", {}) or {}),
    }
    phi_tracker = build_phi_tracker(dict(layered.get("phi", {}) or {}), 128)
    workspace = build_workspace(workspace_overrides, phi_tracker)
    runtime.phi_tracker = phi_tracker
    runtime.workspace = workspace

    flags = NarrationFlags.from_settings(cfg)
    runtime.narration = flags
    pad_overrides = {
        key: value
        for key, value in dict(layered.get("pad", {}) or {}).items()
        if key
        in (
            "history_window",
            "coherence_method",
            "coherence_window",
            "coherence_min_history",
        )
    }
    pad_model = PADAffectModel(narrate_affect_line=flags.affect_line, **pad_overrides)
    prospective = ProspectiveSimulator(clock=now, settings=cfg)
    binder = CrossModalBinder()
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides(
            dict(layered.get("cycle", {}) or {}), pinned_off=False
        ),
        workspace,
        pad_model=pad_model,
        prospective=prospective,
        binder=binder,
        clock=now,
    )
    runtime.pad_model = pad_model
    runtime.prospective = prospective
    runtime.crossmodal = binder
    runtime.augmenter = augmenter

    from mtc.assessment.causal_emergence import CausalEmergenceAnalyzer
    from mtc.assessment.rpt_measurement import RPTMeasurement

    rpt_overrides = dict(layered.get("rpt", {}) or {})
    runtime.rpt = RPTMeasurement(
        stdp_plasticity_enabled=bool(rpt_overrides.get("stdp_plasticity_enabled", True))
    )
    ce_overrides = dict(layered.get("causal_emergence", {}) or {})
    runtime.causal_emergence = CausalEmergenceAnalyzer(
        tpm_row_order=str(ce_overrides.get("tpm_row_order", "big_endian"))
    )
    runtime.prm = PerceptualRealityMonitor()

    # -- speaker trust, safety ------------------------------------------
    trust = SpeakerTrust.from_settings(cfg)
    runtime.speaker_trust = trust
    privacy = IntegratedPrivacySystem(settings=cfg)
    alerts = AlertManager.from_settings(cfg, store=alert_store)
    wellbeing = WellbeingMonitor(alerts=alerts, clock=now)
    compass = TrustCompass(
        brain=None, trust_lookup=trust.trust_lookup, settings=cfg, clock=now
    )
    compass.set_brain(foreground_brain)
    runtime.compass = compass
    monitor = SafetyMonitor(
        audit_store=safety_audit_store,
        compass=compass,
        alerts=alerts,
        settings=cfg,
        clock=now,
    )
    runtime.safety = SafetySystemsIntegration(
        privacy=privacy,
        wellbeing=wellbeing,
        alerts=alerts,
        safety_monitor=monitor,
        settings=cfg,
        clock=now,
    )

    # -- cognition modules ------------------------------------------------
    runtime.learning = LearningSystem(store=learning_store, settings=cfg, clock=now)
    await runtime.learning.initialize()
    runtime.theory_of_mind = TheoryOfMind(
        store=speaker_store,
        trust_lookup=trust.trust_lookup,
        narrate_speaker_model=flags.speaker_model,
        clock=now,
    )
    await runtime.theory_of_mind.initialize()
    runtime.goals = IntrinsicGoalSystem(store=goal_store, clock=now)
    await runtime.goals.initialize()
    runtime.observer = AttentionGrabObserver(store=grab_store, settings=cfg, clock=now)
    runtime.emotion_extractor = EmotionExtractor(
        averages=emotion_averages, drives=_drive_reader(workspace)
    )
    runtime.emotion_dynamics = EmotionDynamics()
    runtime.identity_continuity = IdentityContinuityManager(
        agent_id=str(cfg.SYSTEM_NAME), settings=cfg, clock=now
    )
    runtime.self_assessment = ConsciousnessSelfAssessment()
    if developmental_enabled(cfg):
        from mtc.consciousness.developmental_age import DevelopmentalAgeAssessor

        runtime.developmental = DevelopmentalModulator(settings=cfg)
        runtime.developmental_age = DevelopmentalAgeAssessor(settings=cfg)

    # -- neural orchestrator and the thought machine ----------------------
    from mtc.neural.orchestrator import NeuralOrchestrator

    neural_kwargs: Dict[str, Any] = {}
    if not substrates:
        neural_kwargs.update(emotion_bridge=False, memory_consolidator=False)
    neural = NeuralOrchestrator(
        neural_data_manager=data_manager,
        enable_mongodb=data_manager is not None,
        brain=background_brain,
        settings=cfg,
        **neural_kwargs,
    )
    runtime.neural = neural
    runtime.integration = ConsciousnessIntegration(neural, enable_persistence=False)
    runtime.verifier = ConsciousnessVerifier(clock=now)
    assembler = ConsciousnessStateAssembler(runtime.integration, clock=now)

    machine: Any = None
    ctm_window_s = 0.0
    if cfg.CTM_ENABLED:
        from mtc.consciousness.continuous import (
            MemoryManagerSource,
            PadEmotionSource,
            build_thought_machine,
        )
        from mtc.consciousness.continuous.config import CTMConfig

        ctm_window_s = float(
            CTMConfig.load(None, None, settings=cfg).contention_window_s
        )
        runtime.level_cache = ConsciousnessLevelCache(
            lambda: getattr(runtime.orchestrator, "baseline", None)
            and runtime.orchestrator.baseline.overall_score,
            settings=cfg,
        )
        machine = build_thought_machine(
            settings=cfg,
            brain=background_brain,
            store=thought_store,
            gist_store=gist_store,
            embedder=runtime.embedder,
            memory_source=MemoryManagerSource(manager),
            emotion_source=PadEmotionSource(pad_model),
            contention_guard=lambda: flight.contended(ctm_window_s),
            level_provider=runtime.level_cache.level,
            broadcaster=runtime._broadcast_thought,
            vector_search=stack.vector_store,
            allow_non_semantic=allow_non_semantic,
            data_manager=data_manager,
            clock=now,
        )
        neural.attach_thought_machine(machine)
    runtime.ctm = machine

    # -- snapshots, orchestrator ------------------------------------------
    runtime.snapshots = SnapshotService(
        store=snapshot_store,
        assess=lambda: runtime.orchestrator.assess_now(),
        backup_dir=resolve_path(cfg.RESULTS_DIR, settings=cfg) / BACKUP_SUBDIR,
        clock=now,
    )
    await runtime.snapshots.initialize()
    runtime.hub = ConsciousnessStateManager(
        state_dir=resolve_path(cfg.STATE_DIR, settings=cfg)
    )
    runtime.orchestrator = ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=flags,
        speaker_trust=trust,
        neural_orchestrator=neural,
        rpt_measurement=runtime.rpt,
        causal_emergence=runtime.causal_emergence,
        pad_model=pad_model,
        snapshot_service=runtime.snapshots,
        state_manager=runtime.hub,
        reality_monitor=runtime.prm,
        prospective=prospective,
        goal_source=runtime.goals,
        attention_observer=runtime.observer,
        developmental=runtime.developmental,
        profiler=StageProfiler(),
        config=OrchestratorConfig(),
        clock=now,
        settings=cfg,
    )
    runtime.narration_applied = apply_narration(
        workspace, pad_model, runtime.theory_of_mind, flags
    )

    # -- consolidation, pruning, sleep, autonomous life -------------------
    runtime.consolidation = MemoryConsolidationPipeline(
        memory_manager=manager,
        brain=background_brain,
        gist_store=gist_store,
        vector_store=stack.vector_store,
        settings=cfg,
        clock=now,
    )
    runtime.pruner = MemoryPruner(
        episodic=manager.episodic,
        vectors=stack.vector_store,
        graph=stack.graph,
        neural=retention_store,
        clock=now,
    )
    if cfg.SLEEP_SCHEDULER_ENABLED:
        runtime.sleep_scheduler = SleepScheduler(
            brain=background_brain,
            dream_store=dream_store,
            thought_stream=machine,
            consolidator=_SleepConsolidator(
                manager, runtime.consolidation, thought_store, gist_store
            ),
            pruner=runtime.pruner,
            themes=_ThemeSource(machine.buffer) if machine is not None else None,
            affect=pad_model,
            goals=runtime.goals,
            clock=now,
            settings=cfg,
        )
    if cfg.AUTONOMOUS_LIFE_ENABLED:
        from mtc.consciousness.activities import default_registry
        from mtc.consciousness.autonomous_life import AutonomousLife

        scheduler = runtime.sleep_scheduler

        async def consolidate_thoughts() -> int:
            return int(await runtime.consolidation.refresh_gist_retention())

        runtime.autonomous_life = AutonomousLife(
            background_brain,
            workspace.active_inference,
            registry=default_registry(
                background_brain, recall=_recall_adapter(manager)
            ),
            goals=runtime.goals,
            is_rest_time=scheduler.is_resting if scheduler is not None else None,
            consolidate_thoughts=consolidate_thoughts,
            clock=now,
        )

    # -- persistence, post-turn, pipeline ---------------------------------
    persistence = MessagePersistence(
        repository=repository,
        turn_store=turn_store,
        working_memory=working_memory,
        settings=cfg,
        clock=now,
    )
    post_turn = PostTurnIntegrator(
        speaker_trust=trust,
        neural_orchestrator=neural,
        verifier=runtime.verifier,
        state_assembler=assembler,
        integration=runtime.integration,
        learning=runtime.learning,
        theory_of_mind=runtime.theory_of_mind,
        emotion_extractor=runtime.emotion_extractor,
        emotion_dynamics=runtime.emotion_dynamics,
        persistence=persistence,
        memory_manager=manager,
        graph=stack.graph,
        workspace=workspace,
        pad_model=pad_model,
        safety=runtime.safety,
        identity_continuity=runtime.identity_continuity,
        orchestrator=runtime.orchestrator,
        config=PostTurnConfig(
            affect_shadow=runtime.embedder_kind.startswith("sentence-transformers")
        ),
        settings=cfg,
    )
    pipeline_config = PipelineConfig()
    plugins = PluginRegistry(settings=cfg, clock=now)
    assembler_hook = getattr(brain, "assembler", None)
    if assembler_hook is not None:
        plugins.register_capabilities(assembler_hook)
    runtime.pipeline = ConversationPipeline(
        brain=brain,
        flight=flight,
        orchestrator=runtime.orchestrator,
        speaker_trust=trust,
        sessions=SessionManager(max_history=pipeline_config.max_history, clock=now),
        persistence=persistence,
        repository=repository,
        memory_manager=manager,
        post_turn=post_turn,
        summarizer=ContextSummarizer(
            foreground_brain,
            threshold=pipeline_config.summarize_threshold,
            batch_size=pipeline_config.summarize_batch,
            settings=cfg,
        ),
        plugins=plugins,
        safety=runtime.safety,
        learning=runtime.learning,
        theory_of_mind=runtime.theory_of_mind,
        thought_store=thought_store if machine is not None else None,
        gist_store=gist_store if machine is not None else None,
        coherence=ConversationCoherence(),
        sleep_scheduler=runtime.sleep_scheduler,
        autonomous_life=runtime.autonomous_life,
        config=pipeline_config,
        settings=cfg,
        clock=now,
    )

    # -- hub registration ---------------------------------------------------
    runtime.register("gwt", workspace)
    runtime.register("phi", phi_tracker)
    runtime.register("pad_affect", pad_model)
    runtime.register("orchestrator", runtime.orchestrator)
    runtime.register("augmenter", augmenter)
    runtime.register("rpt", runtime.rpt)
    runtime.register("causal_emergence", runtime.causal_emergence)
    runtime.register("neural", neural)
    if machine is not None:
        runtime.register("ctm", machine)
        runtime.register("ctm_buffer", machine.buffer)
        runtime.register("ctm_generator", machine.generator)
    runtime.register("prm", runtime.prm)
    runtime.register("emotion_dynamics", runtime.emotion_dynamics)
    runtime.register("emotion_extractor", runtime.emotion_extractor)
    runtime.register("sleep_scheduler", runtime.sleep_scheduler)
    runtime.register("autonomous_life", runtime.autonomous_life)
    runtime.register("wellbeing", wellbeing)
    runtime.register_late_components()

    # -- background tasks ---------------------------------------------------
    tasks: List[BackgroundTask] = []
    if machine is not None:
        tasks.append(
            BackgroundTask(
                "ctm",
                float(machine.next_delay()),
                machine.tick,
                start_hook=machine.start,
                stop_hook=machine.stop,
                settings=cfg,
            )
        )
    if runtime.sleep_scheduler is not None:
        scheduler = runtime.sleep_scheduler
        tasks.append(
            BackgroundTask(
                "sleep_scheduler",
                float(scheduler.config.tick_seconds),
                scheduler.tick,
                start_hook=scheduler.start,
                stop_hook=scheduler.stop,
                settings=cfg,
            )
        )
    if runtime.autonomous_life is not None:
        life = runtime.autonomous_life
        tasks.append(
            BackgroundTask(
                "autonomous_life",
                float(life.config.min_interval_s),
                life.step,
                start_hook=life.start,
                stop_hook=life.stop,
                settings=cfg,
            )
        )

    async def baseline_heartbeat() -> Any:
        orchestrator = runtime.orchestrator
        if orchestrator.baseline_due:
            return await orchestrator.refresh_baseline(
                trigger=orchestrator.baseline_due_reason or "heartbeat"
            )
        return None

    sampler = ResilientSampler()

    async def wellbeing_sampler() -> Any:
        try:
            recorded = wellbeing.sample_pad(pad_model)
            report = await runtime.safety.periodic_wellbeing_check()
        except Exception:
            if sampler.record_failure():
                logger.warning("wellbeing sampling has failed repeatedly")
            raise
        if sampler.record_success():
            logger.info("wellbeing sampling recovered")
        return {"recorded": recorded, "report": report}

    tasks.append(
        BackgroundTask("baseline_heartbeat", MINUTE_S, baseline_heartbeat, settings=cfg)
    )
    tasks.append(
        BackgroundTask("wellbeing_sampler", MINUTE_S, wellbeing_sampler, settings=cfg)
    )
    if float(cfg.PHI_IDLE_SAMPLE_S) > 0:
        tasks.append(
            BackgroundTask(
                "phi_idle_sampler",
                float(cfg.PHI_IDLE_SAMPLE_S),
                lambda: runtime.orchestrator.sample_phi_state(),
                settings=cfg,
            )
        )
    if float(cfg.SELF_ASSESSMENT_INTERVAL_S) > 0:

        async def self_assess() -> Any:
            assessment = runtime.self_assessment
            try:
                reply = await background_brain.think(
                    assessment.build_prompt(),
                    max_tokens=SELF_ASSESSMENT_MAX_TOKENS,
                    temperature=SELF_ASSESSMENT_TEMPERATURE,
                    thinking_mode=False,
                )
            except NoAnswerError:
                return {"recorded": False, "reason": "no_answer"}
            text = strip_reasoning_blocks(reply.text or "")
            if not text:
                return {"recorded": False, "reason": "empty"}
            self_report = assessment.parse_response(text)
            latest = await runtime.orchestrator.assess_now()
            if latest is None:
                return {"recorded": False, "reason": "no_assessment"}
            comparison = assessment.compare_with_indicators(latest, self_report)
            runtime.orchestrator.record_self_assessment(self_report, comparison)
            return {"recorded": True, "parsed": self_report.parsed_count}

        tasks.append(
            BackgroundTask(
                "self_assessment",
                float(cfg.SELF_ASSESSMENT_INTERVAL_S),
                self_assess,
                settings=cfg,
            )
        )

    def save_state() -> Any:
        runtime.register_late_components()
        return runtime.hub.save_all() if runtime.hub is not None else None

    tasks.append(BackgroundTask("state_save", STATE_SAVE_S, save_state, settings=cfg))
    tasks.append(
        BackgroundTask(
            "session_pruning",
            SESSION_PRUNE_S,
            lambda: runtime.pipeline.prune_sessions(),
            settings=cfg,
        )
    )
    runtime.background = tasks
    logger.info(
        "runtime built: provider %s, stores %s, embedder %s, %d hub modules",
        getattr(getattr(brain, "registry", None), "active_id", None),
        {name: report["backend"] for name, report in runtime.stores.items()},
        runtime.embedder_kind,
        len(runtime.registered),
    )
    return runtime


__all__ = (
    "EXPERIMENTS_DIR",
    "HUB_NAMES",
    "NOT_REGISTERED",
    "STORE_NAMES",
    "BackgroundTask",
    "Runtime",
    "StoreBuilders",
    "build_runtime",
    "load_pipeline_overrides",
)
