"""
API Server
==========

``create_app`` builds the FastAPI application over one runtime and ``main``
serves it with uvicorn in a single process (the runtime is in-process state).

Routes (every one except ``/health`` needs the ``X-API-Key`` header):

- ``GET /health``: ``status`` (``ok``, or ``degraded`` while the runtime is
  not started or the brain is unavailable), ``ready`` and ``version``.
- ``GET /api/meta``: the system name, the version, the eight theories, every
  indicator the assessment defines with its metadata (positive and negative;
  a threshold, weight, description or theory the assessment does not define
  is null), the providers, the capabilities (no vision, the registered tools,
  which are none unless a plugin registers one), the narration flags and
  whether auth is on.
- ``POST /api/chat`` (``{message, speaker?}`` with an optional
  ``X-Speaker-Token`` header): one turn, ``TurnResult.to_payload()``; 403 when
  the safety gate blocked it, 502 when the model gave no answer, 503 when the
  brain is unavailable, 504 when the pipeline's own bound ran out (the turn's
  payload) or when the first-token or stall bound here cancelled the turn
  (``TimeoutPayload``: the bound, its limit, the elapsed seconds, the
  first-token time and the cancelled state, nothing else).
- ``GET /api/consciousness/status``: orchestrator counters and stage timings,
  the first-run report, the theory modulations with their ``consumed`` flags,
  gate statistics, the continuous thought status and quiet reason, store,
  provider and safety component reports, background task health.
- ``GET /api/consciousness/state``: the last cycle's state fields with the
  PhiTracker value under ``integration_proxy`` (labelled; it is not IIT Phi)
  and the last turn's modulation layers.
- ``GET /api/consciousness/assessment``: the last assessment;
  ``?refresh=true`` runs one (reused for 10 seconds). Nothing measured reads
  ``overall_score`` null with ``status`` ``insufficient_data``, never zero.
- ``GET /api/consciousness/snapshots`` (``?hours``, ``?limit``): snapshot
  history and events.
- ``POST /api/memory/search`` (``{query, top_k?}``): memories, concepts and
  the recall ``method`` (``vector``, ``text`` or ``none``).
- ``GET /api/knowledge/concepts`` (``?limit``) and ``POST
  /api/knowledge/concepts`` (``{name}``): concept names, related concepts,
  with the graph's backend (``memory`` unless ``NEO4J_ENABLED``).
- ``POST /api/ctm/start`` (``{state?}``), ``POST /api/ctm/stop``,
  ``GET /api/ctm/status``.
- ``GET /api/models``: the registered providers, the active one and its
  cached health.
- ``/ws`` (only when ``WEBSOCKET_ENABLED``): see ``mtc.api.websocket``.

Search text, names and messages travel in bodies, never in query strings, and
uvicorn runs with its access log off. A request that fails validation is
answered with the location and type of each error, never its input. No
documentation route is served, so no route is reachable without the key.

The lifespan calls ``setup_logging()``, refuses an unauthenticated
non-loopback ``API_HOST``, builds the runtime from Settings unless one was
given (``build_runtime`` or the injected ``runtime_builder``), starts it,
attaches the expressed-thought broadcast and the periodic state push, and on
shutdown stops all of that and the runtime.

Config keys read: ``API_HOST``, ``API_PORT``, ``API_KEY``,
``API_AUTH_ENABLED``, ``API_CORS_ORIGINS`` (credentials are never allowed),
``WEBSOCKET_ENABLED``, ``WEBSOCKET_PING_INTERVAL``,
``WEBSOCKET_PING_TIMEOUT``, ``METRICS_UPDATE_INTERVAL``, ``SYSTEM_NAME``,
``RAG_MIN_SCORE`` and, through ``mtc.api.progress``, the generation bounds.
Nothing is persisted here. Message text, names, memories and thoughts are
private runtime data; no key or token reaches a log line or a response.
"""

import argparse
import asyncio
import logging
import math
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mtc import __version__
from mtc.api.auth import (
    API_KEY_HEADER,
    SPEAKER_TOKEN_HEADER,
    UnsafeBindError,
    assert_bind_allowed,
    require_api_key,
)
from mtc.api.diagnostics import RequestObserver
from mtc.api.progress import GenerationTimeout, run_with_heartbeat
from mtc.api.schemas import (
    MAX_FRAME_BYTES,
    AssessmentResponse,
    ChatRequest,
    ConceptRelatedRequest,
    ConceptsResponse,
    CTMStartRequest,
    CTMStatusResponse,
    HealthResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MetaResponse,
    ModelsResponse,
    RelatedConceptsResponse,
    SnapshotsResponse,
    StateResponse,
    StatusResponse,
    TurnPayload,
    jsonable,
    state_payload,
    validation_details,
)
from mtc.api.websocket import ConnectionManager, handle_ws
from mtc.core.config import Settings, get_settings
from mtc.core.logger import setup_logging
from mtc.core.narration import NarrationFlags

logger = logging.getLogger(__name__)

RuntimeBuilder = Callable[[Settings], Awaitable[Any]]

# TurnResult error codes and the HTTP status each one maps to.
ERROR_STATUS: Dict[str, int] = {
    "blocked": 403,
    "no_answer": 502,
    "brain_unavailable": 503,
    "generation_timeout": 504,
}
TURN_FAILED_STATUS = 500
ASSESSMENT_CACHE_S = 10.0
MAX_SNAPSHOT_HOURS = 24.0 * 366
MAX_LIST_LIMIT = 1000

_AUTH = [Depends(require_api_key)]


# ---------------------------------------------------------------------------
# Runtime access
# ---------------------------------------------------------------------------


def _runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not ready")
    return runtime


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _status_of(component: Any, *names: str) -> Optional[Dict[str, Any]]:
    """The first status method a component has, called; None when absent."""
    if component is None:
        return None
    for name in names or ("status", "get_status"):
        reader = getattr(component, name, None)
        if callable(reader):
            try:
                return jsonable(reader())
            except Exception as exc:  # noqa: BLE001 - status never fails a page
                return {"available": False, "error": type(exc).__name__}
    return None


def _memory_manager(runtime: Any) -> Any:
    memory = getattr(runtime, "memory", None)
    return getattr(memory, "manager", memory)


def _graph(runtime: Any) -> Any:
    manager = _memory_manager(runtime)
    graph = getattr(manager, "graph", None)
    if graph is None:
        graph = getattr(getattr(runtime, "memory", None), "graph", None)
    return graph


def _graph_backend(graph: Any) -> str:
    """The graph store's own backend label (``memory`` or ``neo4j``), else its
    type name."""
    backend = (_status_of(graph) or {}).get("backend")
    return str(backend) if backend else type(graph).__name__


def _record(item: Any) -> Dict[str, Any]:
    if is_dataclass(item) and not isinstance(item, type):
        return jsonable(asdict(item))
    if isinstance(item, dict):
        return jsonable(item)
    return {"value": jsonable(item)}


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def _finite_or_none(value: Any) -> Optional[float]:
    """A finite number as a float; None when absent, not a number or not
    finite (a missing value is never reported as 0.0)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _indicator_entry(
    name: str, config: Dict[str, Any], polarity: str
) -> Dict[str, Any]:
    """One indicator's metadata. A value its config does not define is None;
    ``polarity`` is the list the indicator belongs to, used when the config
    names none."""
    theory = config.get("theory")
    description = config.get("description")
    return {
        "name": name,
        "theory": None if theory is None else getattr(theory, "name", str(theory)),
        "polarity": str(config.get("polarity") or polarity),
        "description": None if description is None else str(description),
        "threshold": _finite_or_none(config.get("threshold")),
        "weight": _finite_or_none(config.get("weight")),
        "butlin_id": config.get("butlin_id"),
        "specificity": _finite_or_none(config.get("specificity")),
        "sensitivity": _finite_or_none(config.get("sensitivity")),
        "entails": list(config.get("entails") or []),
        "supported_by": list(config.get("supported_by") or []),
    }


def _assessment_metadata(app: FastAPI) -> Dict[str, Any]:
    cached = getattr(app.state, "assessment_metadata", None)
    if cached is not None:
        return cached
    from mtc.assessment.assessment import ConsciousnessAssessment, ConsciousnessTheory

    assessment = ConsciousnessAssessment()
    metadata = {
        "theories": [
            {"id": theory.name, "name": theory.value} for theory in ConsciousnessTheory
        ],
        "indicators": [
            _indicator_entry(name, config, "positive")
            for name, config in assessment.indicator_configs.items()
        ],
        "negative_indicators": [
            _indicator_entry(name, config, "negative")
            for name, config in assessment.negative_indicator_configs.items()
        ],
    }
    app.state.assessment_metadata = metadata
    return metadata


def _tool_names(runtime: Any) -> List[str]:
    plugins = getattr(getattr(runtime, "pipeline", None), "plugins", None)
    names = getattr(plugins, "names", None)
    if callable(names):
        return [str(name) for name in names()]
    return []


def _providers(runtime: Any) -> Dict[str, Any]:
    status = _status_of(getattr(runtime, "brain", None)) or {}
    return {
        "registered": list(status.get("providers") or []),
        "active": status.get("active_provider"),
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _gate_statistics(runtime: Any) -> Dict[str, Any]:
    workspace = getattr(runtime, "workspace", None)
    bottleneck = getattr(workspace, "bottleneck", None)
    gate = getattr(bottleneck, "thalamic_gate", None)
    if gate is None:
        return {"enabled": False}
    stats = getattr(gate, "stats", None)
    as_dict = getattr(stats, "as_dict", None)
    return {"enabled": True, **(jsonable(as_dict()) if callable(as_dict) else {})}


def _ctm_status(runtime: Any) -> Optional[Dict[str, Any]]:
    ctm = getattr(runtime, "ctm", None)
    if ctm is None:
        return {"available": False, "quiet_reason": None}
    status = _status_of(ctm, "get_status", "status") or {}
    return {"available": True, "quiet_reason": status.get("quiet_reason"), **status}


# Runtime status keys the status page already reports under its own names.
_REPORTED_SEPARATELY = frozenset(
    {
        "brain",
        "pipeline",
        "orchestrator",
        "first_run",
        "ctm",
        "stores",
        "safety",
        "speaker_trust",
        "background",
    }
)


def _status_payload(app: FastAPI, runtime: Any) -> Dict[str, Any]:
    orchestrator = getattr(runtime, "orchestrator", None)
    orch_status = _status_of(orchestrator) or {}
    first_run = None
    reader = getattr(orchestrator, "first_run_report", None)
    if callable(reader):
        try:
            first_run = jsonable(reader())
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            first_run = {"available": False, "error": type(exc).__name__}
    runtime_status = _status_of(runtime) or {}
    return {
        "orchestrator": orch_status or None,
        "first_run": first_run,
        "theory_modulations": orch_status.get("theory_modulations"),
        "gate": _gate_statistics(runtime),
        "ctm": _ctm_status(runtime),
        "stores": runtime_status.get("stores"),
        "provider": _status_of(getattr(runtime, "brain", None)),
        "safety": _status_of(getattr(runtime, "safety", None)),
        "background": runtime_status.get("background"),
        "pipeline": _status_of(getattr(runtime, "pipeline", None)),
        "speakers": _status_of(getattr(runtime, "speaker_trust", None)),
        "requests": app.state.observer.duplicates(),
        "websocket": app.state.connections.status(),
        "runtime": {
            key: value
            for key, value in runtime_status.items()
            if key not in _REPORTED_SEPARATELY
        },
    }


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


INSUFFICIENT_DATA = "insufficient_data"


def _withhold_unmeasured(record: Dict[str, Any]) -> Dict[str, Any]:
    """Null the overall score of a record that measured nothing.

    The assessment includes absent indicators at 0.0, so a report that
    measured no indicator can still carry an overall 0.0; the API reports
    such a record as ``overall_score`` null with ``overall_withheld`` true,
    never as a measured zero.
    """
    if record.get("status") == INSUFFICIENT_DATA and "overall_score" in record:
        record["overall_withheld"] = record.get("overall_score") is not None
        record["overall_score"] = None
    return record


def _assessment_payload(report: Any, *, cached: bool, age_s: float) -> Dict[str, Any]:
    to_dict = getattr(report, "to_dict", None)
    body = jsonable(to_dict()) if callable(to_dict) else jsonable(report)
    if isinstance(body, dict):
        body = _withhold_unmeasured(body)
    status = str(getattr(report, "status", "unknown"))
    overall = jsonable(getattr(report, "overall_score", None))
    return {
        "status": status,
        "overall_score": None if status == INSUFFICIENT_DATA else overall,
        "overall_withheld": status == INSUFFICIENT_DATA and overall is not None,
        "coverage": jsonable(getattr(report, "coverage", None)),
        "cached": cached,
        "age_s": round(age_s, 3),
        "report": body,
    }


def _timeout_payload(exc: GenerationTimeout) -> Dict[str, Any]:
    """The 504 body for a turn a generation bound cancelled (``TimeoutPayload``):
    what is known when the bound fires and nothing else. The turn never
    returned, so its reply, session, speaker label and counts are unknown and
    are left out rather than filled with stand-in values."""
    first = exc.first_token_s
    return {
        "success": False,
        "error": exc.describe(),
        "error_code": "generation_timeout",
        "bound": exc.bound,
        "limit_s": exc.limit_s,
        "elapsed_s": round(exc.elapsed_s, 3),
        "first_token_s": None if first is None else round(float(first), 3),
        "state": "cancelled",
    }


def _turn_lock(runtime: Any) -> Optional[asyncio.Lock]:
    """The pipeline's turn lock, so an assessment never runs inside a turn
    (None before the first turn created it)."""
    lock = getattr(getattr(runtime, "pipeline", None), "turn_lock", None)
    return lock if isinstance(lock, asyncio.Lock) else None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


async def _default_builder(settings: Settings) -> Any:
    from mtc.pipeline.runtime import build_runtime

    return await build_runtime(settings)


def _attach_thought_broadcast(app: FastAPI, runtime: Any) -> None:
    """Send the machine's expressed thoughts to the keyed websocket clients.

    A runtime with ``add_thought_listener`` gets the connection manager as one
    more listener; otherwise the machine's own ``broadcaster`` hook is chained.
    """
    manager: ConnectionManager = app.state.connections
    add = getattr(runtime, "add_thought_listener", None)
    if callable(add):
        add(manager.broadcast_thought)
        app.state.thought_hook = ("listener", runtime, manager.broadcast_thought)
        return
    ctm = getattr(runtime, "ctm", None)
    if ctm is None or not hasattr(ctm, "broadcaster"):
        return
    previous = ctm.broadcaster

    async def broadcast(message: Dict[str, Any]) -> None:
        if previous is not None:
            result = previous(message)
            if asyncio.iscoroutine(result):
                await result
        await manager.broadcast_thought(message)

    ctm.broadcaster = broadcast
    app.state.thought_hook = ("broadcaster", ctm, previous)


def _detach_thought_broadcast(app: FastAPI) -> None:
    hook = getattr(app.state, "thought_hook", None)
    app.state.thought_hook = None
    if hook is None:
        return
    kind, owner, value = hook
    if kind == "listener":
        listeners = getattr(owner, "thought_listeners", None)
        if isinstance(listeners, list) and value in listeners:
            listeners.remove(value)
    else:
        owner.broadcaster = value


async def _state_push_loop(app: FastAPI, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        await push_state_once(app)


async def push_state_once(app: FastAPI) -> int:
    """Send the current state to every subscriber; return how many it reached."""
    manager: ConnectionManager = app.state.connections
    runtime = getattr(app.state, "runtime", None)
    if runtime is None or not manager.subscribers():
        return 0
    try:
        return await manager.push_state(state_payload(runtime))
    except Exception as exc:  # noqa: BLE001 - a push never stops the loop
        logger.warning("periodic state push failed: %s", type(exc).__name__)
        return 0


def _install_handlers(app: FastAPI) -> None:
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422, content={"detail": validation_details(exc)}
        )

    app.add_exception_handler(RequestValidationError, validation_error)


def create_app(
    *,
    settings: Optional[Settings] = None,
    runtime: Any = None,
    runtime_builder: Optional[RuntimeBuilder] = None,
) -> FastAPI:
    """Build the application over ``runtime`` (or one built at startup)."""
    cfg = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(cfg)
        assert_bind_allowed(cfg)
        active = app.state.runtime
        if active is None:
            builder = runtime_builder or _default_builder
            active = await builder(cfg)
            app.state.runtime = active
        await active.start()
        app.state.started = True
        _attach_thought_broadcast(app, active)
        push_task = None
        interval = float(cfg.METRICS_UPDATE_INTERVAL)
        if cfg.WEBSOCKET_ENABLED and interval > 0:
            push_task = asyncio.get_running_loop().create_task(
                _state_push_loop(app, interval), name="api-state-push"
            )
        logger.info("API ready (auth %s)", "on" if cfg.API_AUTH_ENABLED else "off")
        try:
            yield
        finally:
            if push_task is not None:
                push_task.cancel()
                await asyncio.gather(push_task, return_exceptions=True)
            pending = list(app.state.turn_tasks)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            _detach_thought_broadcast(app)
            app.state.started = False
            await active.stop()
            logger.info("API stopped")

    app = FastAPI(
        title="Multi-Theory Consciousness API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.runtime = runtime
    app.state.started = False
    app.state.observer = RequestObserver()
    app.state.connections = ConnectionManager()
    app.state.turn_tasks = set()
    app.state.assessment = None
    app.state.thought_hook = None

    origins = [origin for origin in cfg.API_CORS_ORIGINS_LIST if origin]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[API_KEY_HEADER, SPEAKER_TOKEN_HEADER, "Content-Type"],
    )
    _install_handlers(app)
    _add_routes(app)
    if cfg.WEBSOCKET_ENABLED:

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await handle_ws(websocket, getattr(websocket.app.state, "runtime", None))

    return app


def _add_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> Dict[str, Any]:
        runtime = getattr(request.app.state, "runtime", None)
        ready = bool(runtime is not None and request.app.state.started)
        available = getattr(getattr(runtime, "brain", None), "available", None)
        healthy = ready and available is True
        return {
            "status": "ok" if healthy else "degraded",
            "ready": ready,
            "version": __version__,
        }

    @app.get("/api/meta", response_model=MetaResponse, dependencies=_AUTH)
    async def meta(request: Request) -> Dict[str, Any]:
        cfg = _settings(request)
        runtime = _runtime(request)
        metadata = _assessment_metadata(request.app)
        return {
            "system_name": cfg.SYSTEM_NAME,
            "version": __version__,
            **metadata,
            "providers": _providers(runtime),
            "capabilities": {
                "vision": False,
                "tools": _tool_names(runtime),
                "websocket": bool(cfg.WEBSOCKET_ENABLED),
                "streaming": bool(cfg.WEBSOCKET_ENABLED),
            },
            "narration": NarrationFlags.from_settings(cfg).as_dict(),
            "auth": {
                "enabled": bool(cfg.API_AUTH_ENABLED),
                "header": API_KEY_HEADER,
                "speaker_token_header": SPEAKER_TOKEN_HEADER,
            },
        }

    @app.post("/api/chat", response_model=TurnPayload, dependencies=_AUTH)
    async def chat(
        body: ChatRequest,
        request: Request,
        speaker_token: Optional[str] = Header(default=None, alias=SPEAKER_TOKEN_HEADER),
    ) -> JSONResponse:
        cfg = _settings(request)
        runtime = _runtime(request)
        pipeline = getattr(runtime, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="pipeline not built")
        try:
            result = await run_with_heartbeat(
                lambda mark: pipeline.send_message(
                    body.message,
                    body.speaker,
                    speaker_token,
                    stream_callback=mark.wrap(None),
                ),
                on_heartbeat=None,
                settings=cfg,
            )
        except GenerationTimeout as exc:
            logger.warning("chat turn cancelled: %s", exc.detail())
            return JSONResponse(
                status_code=ERROR_STATUS["generation_timeout"],
                content=_timeout_payload(exc),
            )
        payload = jsonable(result.to_payload())
        if result.success:
            return JSONResponse(status_code=200, content=payload)
        code = "blocked" if result.blocked else str(result.error_code or "")
        status_code = ERROR_STATUS.get(code, TURN_FAILED_STATUS)
        return JSONResponse(status_code=status_code, content=payload)

    @app.get(
        "/api/consciousness/status", response_model=StatusResponse, dependencies=_AUTH
    )
    async def consciousness_status(request: Request) -> Dict[str, Any]:
        return _status_payload(request.app, _runtime(request))

    @app.get(
        "/api/consciousness/state", response_model=StateResponse, dependencies=_AUTH
    )
    async def consciousness_state(request: Request) -> Dict[str, Any]:
        return state_payload(_runtime(request))

    @app.get(
        "/api/consciousness/assessment",
        response_model=AssessmentResponse,
        dependencies=_AUTH,
    )
    async def consciousness_assessment(
        request: Request, refresh: bool = False
    ) -> Dict[str, Any]:
        runtime = _runtime(request)
        state = request.app.state
        now = time.monotonic()
        cached = state.assessment
        fresh = cached is not None and now - cached[1] <= ASSESSMENT_CACHE_S
        if refresh and not fresh:
            orchestrator = getattr(runtime, "orchestrator", None)
            assess = getattr(orchestrator, "assess_now", None)
            if not callable(assess):
                raise HTTPException(status_code=503, detail="orchestrator not built")
            lock = _turn_lock(runtime)
            if lock is not None:
                async with lock:
                    report = await assess()
            else:
                report = await assess()
            if report is None:
                # No report exists, so it has no age either.
                return {
                    "status": "failed",
                    "overall_score": None,
                    "coverage": None,
                    "cached": False,
                    "age_s": None,
                    "report": None,
                }
            state.assessment = (report, time.monotonic())
            return _assessment_payload(report, cached=False, age_s=0.0)
        if cached is None:
            return {
                "status": "not_run",
                "overall_score": None,
                "coverage": None,
                "cached": False,
                "age_s": None,
                "report": None,
            }
        report, at = cached
        return _assessment_payload(report, cached=True, age_s=now - at)

    @app.get(
        "/api/consciousness/snapshots",
        response_model=SnapshotsResponse,
        dependencies=_AUTH,
    )
    async def consciousness_snapshots(
        request: Request,
        hours: float = Query(24.0, gt=0, le=MAX_SNAPSHOT_HOURS),
        limit: int = Query(100, ge=1, le=MAX_LIST_LIMIT),
    ) -> Dict[str, Any]:
        runtime = _runtime(request)
        service = getattr(runtime, "snapshots", None)
        if service is None:
            return {"available": False, "hours": hours, "limit": limit}
        history = await service.get_history(hours=hours, limit=limit)
        events = await service.get_events(hours=hours, limit=limit)
        return {
            "available": True,
            "hours": hours,
            "limit": limit,
            "history": [_withhold_unmeasured(_record(item)) for item in history],
            "events": [_record(item) for item in events],
        }

    @app.post(
        "/api/memory/search", response_model=MemorySearchResponse, dependencies=_AUTH
    )
    async def memory_search(body: MemorySearchRequest, request: Request) -> Dict:
        cfg = _settings(request)
        manager = _memory_manager(_runtime(request))
        recall = getattr(manager, "semantic_recall", None)
        if not callable(recall):
            return {"method": "none", "stats": {"error": "memory manager absent"}}
        found = await recall(
            body.query, top_k=body.top_k, min_score=float(cfg.RAG_MIN_SCORE)
        )
        stats = dict(found.get("search_stats") or {})
        stats.pop("query", None)
        return {
            "method": str(stats.get("method", "none")),
            "memories": jsonable(found.get("memories") or []),
            "concepts": jsonable(found.get("concepts") or []),
            "stats": jsonable(stats),
        }

    @app.get(
        "/api/knowledge/concepts", response_model=ConceptsResponse, dependencies=_AUTH
    )
    async def concepts_list(
        request: Request, limit: int = Query(100, ge=1, le=MAX_LIST_LIMIT)
    ) -> Dict[str, Any]:
        graph = _graph(_runtime(request))
        if graph is None:
            return {"available": False, "backend": None, "concepts": []}
        names = await asyncio.to_thread(graph.list_concept_names, limit)
        return {
            "available": True,
            "backend": _graph_backend(graph),
            "concepts": [str(name) for name in names],
        }

    @app.post(
        "/api/knowledge/concepts",
        response_model=RelatedConceptsResponse,
        dependencies=_AUTH,
    )
    async def concepts_related(
        body: ConceptRelatedRequest, request: Request
    ) -> Dict[str, Any]:
        graph = _graph(_runtime(request))
        if graph is None:
            return {"available": False, "concept": body.name, "related": []}
        related = await asyncio.to_thread(graph.get_related_concepts, body.name, 10)
        return {
            "available": True,
            "concept": body.name,
            "related": jsonable(list(related)),
        }

    def _ctm_or_none(request: Request) -> Any:
        return getattr(_runtime(request), "ctm", None)

    def _ctm_payload(ctm: Any) -> Dict[str, Any]:
        if ctm is None:
            return {"available": False, "running": False, "quiet_reason": None}
        status = _status_of(ctm, "get_status", "status") or {}
        return {
            "available": True,
            "running": bool(status.get("running", False)),
            "quiet_reason": status.get("quiet_reason"),
            "status": status,
        }

    @app.post("/api/ctm/start", response_model=CTMStatusResponse, dependencies=_AUTH)
    async def ctm_start(request: Request, body: Optional[CTMStartRequest] = None):
        ctm = _ctm_or_none(request)
        if ctm is None:
            raise HTTPException(status_code=503, detail="continuous thought is off")
        if body is not None and body.state is not None:
            ctm.set_consciousness_state(body.state)
        await ctm.start()
        return _ctm_payload(ctm)

    @app.post("/api/ctm/stop", response_model=CTMStatusResponse, dependencies=_AUTH)
    async def ctm_stop(request: Request) -> Dict[str, Any]:
        ctm = _ctm_or_none(request)
        if ctm is None:
            raise HTTPException(status_code=503, detail="continuous thought is off")
        await ctm.stop()
        return _ctm_payload(ctm)

    @app.get("/api/ctm/status", response_model=CTMStatusResponse, dependencies=_AUTH)
    async def ctm_status(request: Request) -> Dict[str, Any]:
        return _ctm_payload(_ctm_or_none(request))

    @app.get("/api/models", response_model=ModelsResponse, dependencies=_AUTH)
    async def models(request: Request) -> Dict[str, Any]:
        status = _status_of(getattr(_runtime(request), "brain", None)) or {}
        return {
            "providers": list(status.get("providers") or []),
            "active": status.get("active_provider"),
            "model_name": status.get("model_name"),
            "available": status.get("available"),
            "health": status.get("health"),
            "provider": status.get("provider"),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Serve the API with uvicorn on ``API_HOST``:``API_PORT``, one process.

    Returns 2 when the bind is refused (authentication off outside loopback).
    """
    parser = argparse.ArgumentParser(
        prog="python -m mtc.api.server",
        description="Serve the MTC API in one process; every option comes from "
        "Settings (the MTC_ environment or .env).",
    )
    parser.parse_args(list(argv) if argv is not None else None)
    cfg = get_settings()
    try:
        assert_bind_allowed(cfg)
    except UnsafeBindError as exc:
        logger.error("%s", exc)
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    from mtc.memory.clients import require_client

    uvicorn = require_client("uvicorn", "api")
    app = create_app(settings=cfg)
    config = uvicorn.Config(
        app,
        host=cfg.API_HOST,
        port=int(cfg.API_PORT),
        workers=1,
        access_log=False,
        log_config=None,
        server_header=False,
        proxy_headers=False,
        ws_max_size=MAX_FRAME_BYTES,
        ws_ping_interval=float(cfg.WEBSOCKET_PING_INTERVAL),
        ws_ping_timeout=float(cfg.WEBSOCKET_PING_TIMEOUT),
        lifespan="on",
    )
    uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
