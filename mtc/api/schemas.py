"""
API Schemas
===========

Pydantic models for every HTTP route and every websocket frame type.

Request models forbid unknown fields and cap their text: a chat message at
``MAX_MESSAGE_CHARS`` (32000) characters, a speaker name at
``MAX_SPEAKER_CHARS`` (200). A websocket frame is a JSON text frame
``{"type", "data"}`` of at most ``MAX_FRAME_BYTES`` (1 MiB). The speaker
token of a websocket ``message`` frame is a ``SecretStr``, so its value is
hidden from every repr and validation message; it is read once, to hand it
to the speaker trust source.

Validation errors are reported through ``validation_details``, which keeps
the location, the message and the type of each error and drops the input
value and the context, so a refused request never echoes its text, a name
or a token.

Search text, names and messages travel in bodies, never in query strings.
No config keys are read and nothing is persisted. Message text, speaker
names and tokens are private at runtime.
"""

import dataclasses
import enum
import math
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

MAX_MESSAGE_CHARS = 32000
MAX_SPEAKER_CHARS = 200
MAX_QUERY_CHARS = 2000
MAX_CONCEPT_CHARS = 200
MAX_REQUEST_ID_CHARS = 128
MAX_TOKEN_CHARS = 512
MAX_TOP_K = 50
MAX_FRAME_BYTES = 1024 * 1024

THOUGHT_STATES: Tuple[str, ...] = ("awake", "drowsy", "dreaming", "deep_sleep")

CLIENT_FRAME_TYPES: Tuple[str, ...] = (
    "message",
    "ping",
    "subscribe_state",
    "unsubscribe_state",
)
SERVER_FRAME_TYPES: Tuple[str, ...] = (
    "status",
    "thinking",
    "stream_chunk",
    "stream_retry",
    "response",
    "consciousness_state",
    "spontaneous_thought",
    "error",
    "pong",
)

# Error codes shared by the HTTP status mapping and the websocket error frame.
ERROR_CODES: Tuple[str, ...] = (
    "blocked",
    "no_answer",
    "brain_unavailable",
    "generation_timeout",
    "turn_failed",
    "runtime_not_ready",
    "invalid_message",
    "unknown_type",
    "frame_too_large",
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow")


_MAX_DEPTH = 12


def jsonable(value: Any, _depth: int = 0) -> Any:
    """A JSON-ready copy of ``value``.

    Mappings get string keys, sequences become lists, datetimes become ISO
    strings, enums their values, dataclasses and pydantic models their
    fields, numpy scalars and arrays plain numbers and lists, and a
    non-finite float becomes None (JSON has no NaN). A ``SecretStr`` stays
    masked. Anything else becomes its type name, so an unknown object never
    leaks its repr.
    """
    if _depth > _MAX_DEPTH:
        return None
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, SecretStr):
        return "**********"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return jsonable(value.value, _depth + 1)
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(), _depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: jsonable(getattr(value, item.name), _depth + 1)
            for item in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item, _depth + 1) for item in value]
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        try:
            return jsonable(item(), _depth + 1)
        except (TypeError, ValueError):
            return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return jsonable(tolist(), _depth + 1)
        except (TypeError, ValueError):
            return None
    return type(value).__name__


def validation_details(exc: Any) -> List[Dict[str, Any]]:
    """Location, message and type of each error; never the input value.

    Accepts a pydantic ``ValidationError`` or any error carrying ``errors()``
    (the web framework's request validation error); only the three keys are
    kept, so the input and the context never leave this function.
    """
    if isinstance(exc, ValidationError):
        errors = exc.errors(
            include_url=False, include_context=False, include_input=False
        )
    else:
        reader = getattr(exc, "errors", None)
        errors = reader() if callable(reader) else []
    details = []
    for error in errors:
        details.append(
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
        )
    return details


# ---------------------------------------------------------------------------
# HTTP requests
# ---------------------------------------------------------------------------


class ChatRequest(_Request):
    """``POST /api/chat``; the speaker token travels in ``X-Speaker-Token``."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    speaker: Optional[str] = Field(default=None, max_length=MAX_SPEAKER_CHARS)


class MemorySearchRequest(_Request):
    """``POST /api/memory/search``."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top_k: Optional[int] = Field(default=None, ge=1, le=MAX_TOP_K)


class ConceptRelatedRequest(_Request):
    """``POST /api/knowledge/concepts``: the concept whose relations to list."""

    name: str = Field(min_length=1, max_length=MAX_CONCEPT_CHARS)


class CTMStartRequest(_Request):
    """``POST /api/ctm/start``: optionally the thought mode to start in."""

    state: Optional[Literal["awake", "drowsy", "dreaming", "deep_sleep"]] = None


# ---------------------------------------------------------------------------
# HTTP responses
# ---------------------------------------------------------------------------


class HealthResponse(_Response):
    status: Literal["ok", "degraded"]
    ready: bool
    version: str


class TurnPayload(_Response):
    """``TurnResult.to_payload()``: the display label only, never identity
    internals or tokens."""

    success: bool
    response: Optional[str] = None
    thinking: Optional[str] = None
    conversation_id: Optional[str] = None
    message_count: int = 0
    timestamp: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    cycle_ok: Optional[bool] = None
    speaker_label: str = ""
    provider_id: Optional[str] = None
    blocked: bool = False


class TimeoutPayload(_Response):
    """``POST /api/chat`` when a generation bound cancelled the turn (504).

    Only what is known when the bound fires: the bound (``first_token`` or
    ``stall``), its limit, the seconds elapsed, when the first token came
    (None when none did) and the turn's state (``cancelled``). The turn's
    reply, session, speaker label and counts are not known here, so they are
    absent rather than filled with stand-in values.
    """

    success: Literal[False] = False
    error: str
    error_code: Literal["generation_timeout"] = "generation_timeout"
    bound: Literal["first_token", "stall"]
    limit_s: float
    elapsed_s: float
    first_token_s: Optional[float] = None
    state: Literal["cancelled"] = "cancelled"


class TheoryInfo(_Response):
    id: str
    name: str


class IndicatorInfo(_Response):
    """One indicator's metadata; a value the assessment does not define for
    it (a threshold, a weight, a description, a theory) is None."""

    name: str
    theory: Optional[str] = None
    polarity: str
    description: Optional[str] = None
    threshold: Optional[float] = None
    weight: Optional[float] = None
    butlin_id: Optional[str] = None
    specificity: Optional[float] = None
    sensitivity: Optional[float] = None
    entails: List[str] = Field(default_factory=list)
    supported_by: List[str] = Field(default_factory=list)


class MetaResponse(_Response):
    system_name: str
    version: str
    theories: List[TheoryInfo]
    indicators: List[IndicatorInfo]
    negative_indicators: List[IndicatorInfo]
    providers: Dict[str, Any]
    capabilities: Dict[str, Any]
    narration: Dict[str, bool]
    auth: Dict[str, Any]


class StatusResponse(_Response):
    orchestrator: Optional[Dict[str, Any]] = None
    first_run: Optional[Dict[str, Any]] = None
    theory_modulations: Optional[Dict[str, Any]] = None
    gate: Optional[Dict[str, Any]] = None
    ctm: Optional[Dict[str, Any]] = None
    stores: Optional[Dict[str, Any]] = None
    provider: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    background: Optional[Any] = None
    pipeline: Optional[Dict[str, Any]] = None
    speakers: Optional[Dict[str, Any]] = None
    requests: Optional[Dict[str, Any]] = None
    websocket: Optional[Dict[str, Any]] = None


class StateResponse(_Response):
    available: bool
    state: Optional[Dict[str, Any]] = None
    integration_proxy: Dict[str, Any]
    last_turn: Optional[Dict[str, Any]] = None


class AssessmentResponse(_Response):
    status: str
    overall_score: Optional[float] = None
    overall_withheld: bool = False
    coverage: Optional[float] = None
    cached: bool = False
    age_s: Optional[float] = None
    report: Optional[Dict[str, Any]] = None


class SnapshotsResponse(_Response):
    available: bool
    hours: float
    limit: int
    history: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class MemorySearchResponse(_Response):
    method: str
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    concepts: List[Any] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class ConceptsResponse(_Response):
    available: bool
    backend: Optional[str] = None
    concepts: List[str] = Field(default_factory=list)


class RelatedConceptsResponse(_Response):
    available: bool
    concept: str
    related: List[Dict[str, Any]] = Field(default_factory=list)


class CTMStatusResponse(_Response):
    available: bool
    running: bool = False
    quiet_reason: Optional[str] = None
    status: Dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(_Response):
    providers: List[str] = Field(default_factory=list)
    active: Optional[str] = None
    model_name: Optional[str] = None
    available: Optional[bool] = None
    health: Optional[Dict[str, Any]] = None
    provider: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Websocket frames
# ---------------------------------------------------------------------------


class ClientFrame(_Request):
    """A frame from the client: ``{"type", "data"}``."""

    type: Literal["message", "ping", "subscribe_state", "unsubscribe_state"]
    data: Dict[str, Any] = Field(default_factory=dict)


class WSMessageData(_Request):
    """The data of a ``message`` frame."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    speaker: Optional[str] = Field(default=None, max_length=MAX_SPEAKER_CHARS)
    speaker_token: Optional[SecretStr] = Field(default=None, max_length=MAX_TOKEN_CHARS)
    request_id: Optional[str] = Field(default=None, max_length=MAX_REQUEST_ID_CHARS)


class ServerFrame(BaseModel):
    """A frame to the client: ``{"type", "data"}``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "status",
        "thinking",
        "stream_chunk",
        "stream_retry",
        "response",
        "consciousness_state",
        "spontaneous_thought",
        "error",
        "pong",
    ]
    data: Dict[str, Any] = Field(default_factory=dict)


class StatusData(_Response):
    system_name: str
    version: str
    client_authenticated: bool
    brain_available: Optional[bool] = None
    max_frame_bytes: int = MAX_FRAME_BYTES
    heartbeat_s: Optional[float] = None


class ThinkingData(_Response):
    elapsed_s: float
    request_id: Optional[str] = None


class StreamChunkData(_Response):
    content: str
    request_id: Optional[str] = None


class StreamRetryData(_Response):
    request_id: Optional[str] = None


class ErrorData(_Response):
    code: str
    message: str
    request_id: Optional[str] = None


class SpontaneousThoughtData(_Response):
    thought_id: str
    content: str
    thought_type: str
    timestamp: Optional[str] = None


class PongData(_Response):
    timestamp: str


# ---------------------------------------------------------------------------
# State payload (shared by the state route and the websocket state frames)
# ---------------------------------------------------------------------------

INTEGRATION_PROXY_LABEL = "integration proxy"
INTEGRATION_PROXY_NOTE = (
    "the PhiTracker's per-cycle integration proxy over five module signals, "
    "in bits; it is not IIT Phi"
)

# State fields holding the measured Phi slot; they are reported only under
# ``integration_proxy``, with its label.
PHI_STATE_FIELDS: Tuple[str, ...] = ("architectural_phi", "phi_warming_up")

# State fields that hold workspace content objects (conversation material)
# or nested module state; the payload reports their counts instead.
_CONTENT_FIELDS: Tuple[str, ...] = (
    "primary_content",
    "workspace_contents",
    "attention_schema",
    "metacognitive_state",
    "higher_order_thoughts",
    "active_inference_state",
    "inference_result",
    "posterior_mappings",
    "beautiful_loop_moment",
)


def latest_state(runtime: Any) -> Optional[Any]:
    """The consciousness state of the last cycle, or None before one ran."""
    orchestrator = getattr(runtime, "orchestrator", None)
    if orchestrator is None:
        return None
    state = getattr(orchestrator, "last_state", None)
    if callable(state):
        state = state()
    if state is None:
        # The orchestrator keeps the last cycle's state privately; it has no
        # public accessor yet.
        state = getattr(orchestrator, "_last_state", None)
    return state


def state_fields(state: Any) -> Dict[str, Any]:
    """The state's scalar and small structured fields, JSON-ready; content
    objects are replaced by counts and the Phi slots are left out."""
    if state is None:
        return {}
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        names = [item.name for item in dataclasses.fields(state)]
    else:
        names = [name for name in vars(state) if not name.startswith("_")]
    fields_out: Dict[str, Any] = {}
    for name in names:
        if name in PHI_STATE_FIELDS:
            continue
        value = getattr(state, name, None)
        if name in _CONTENT_FIELDS:
            if isinstance(value, (list, tuple)):
                fields_out[f"{name}_count"] = len(value)
            else:
                fields_out[f"{name}_present"] = value is not None
            continue
        fields_out[name] = jsonable(value)
    return fields_out


def _tracker(runtime: Any) -> Optional[Any]:
    tracker = getattr(runtime, "phi_tracker", None)
    if tracker is None:
        tracker = getattr(getattr(runtime, "workspace", None), "phi_tracker", None)
    return tracker


def integration_proxy(runtime: Any, state: Any = None) -> Dict[str, Any]:
    """The PhiTracker value, labelled as the integration proxy it is."""
    tracker = _tracker(runtime)
    value = getattr(state, "architectural_phi", None) if state is not None else None
    warming = getattr(state, "phi_warming_up", None) if state is not None else None
    if tracker is not None:
        if value is None and not getattr(tracker, "warming_up", True):
            try:
                value = tracker.latest_phi
            except Exception:  # noqa: BLE001 - an unreadable proxy is None
                value = None
        if warming is None:
            warming = bool(getattr(tracker, "warming_up", True))
    return {
        "label": INTEGRATION_PROXY_LABEL,
        "description": INTEGRATION_PROXY_NOTE,
        "unit": "bits",
        "value": jsonable(value),
        "warming_up": warming,
        "available": tracker is not None,
        "transitions_recorded": jsonable(
            getattr(tracker, "transitions_recorded", None) if tracker else None
        ),
    }


def _layers(modulated: Any) -> Dict[str, Any]:
    sections = getattr(modulated, "theory_sections", None)
    as_dict = getattr(sections, "as_dict", None)
    return {
        "narrative": getattr(modulated, "prompt_context", None),
        "theory_sections": jsonable(as_dict() if callable(as_dict) else None),
        "generation_params": jsonable(getattr(modulated, "generation_params", None)),
        "behavioral_directives": jsonable(
            getattr(modulated, "behavioral_directives", None)
        ),
        "self_awareness": getattr(modulated, "consciousness_self_awareness", None),
    }


# Keys of the pipeline's last-turn summary the state payload carries.
_TURN_SUMMARY_KEYS: Tuple[str, ...] = (
    "request_id",
    "cycle_ok",
    "provider_id",
    "absent_signals",
    "recall_method",
    "error",
    "blocked",
    "plugin_reply",
)


def last_turn_layers(runtime: Any) -> Optional[Dict[str, Any]]:
    """The last turn's outcome and, when the pipeline keeps the modulated
    context, its four modulation layers; None before the first turn.

    ``layers`` is None (with ``layers_available`` false) while the pipeline
    keeps no public copy of the last turn's ``ModulatedContext``; the turn's
    outcome, absent signals and recall method still come from its status.
    """
    pipeline = getattr(runtime, "pipeline", None)
    if pipeline is None:
        return None
    summary: Dict[str, Any] = {}
    reader = getattr(pipeline, "status", None)
    if callable(reader):
        try:
            last = (reader() or {}).get("last_turn") or {}
        except Exception:  # noqa: BLE001 - a status page never fails
            last = {}
        summary = {
            key: jsonable(last[key]) for key in _TURN_SUMMARY_KEYS if key in last
        }
    modulated = getattr(pipeline, "last_modulated", None)
    if modulated is None and not summary:
        return None
    layers = _layers(modulated) if modulated is not None else None
    if modulated is not None:
        summary.update(
            {
                "cycle_ok": getattr(modulated, "cycle_ok", None),
                "failure": getattr(modulated, "failure", None),
                "workspace_winner": getattr(modulated, "workspace_winner", None),
                "winner_is_message": getattr(modulated, "winner_is_message", None),
                "absent_signals": jsonable(getattr(modulated, "absent_signals", None)),
                "pad_sources_measured": jsonable(
                    getattr(modulated, "pad_sources_measured", None)
                ),
            }
        )
    return {**summary, "layers_available": layers is not None, "layers": layers}


def state_payload(runtime: Any) -> Dict[str, Any]:
    """The latest state, the labelled integration proxy and the last turn."""
    state = latest_state(runtime)
    return {
        "available": state is not None,
        "state": state_fields(state) if state is not None else None,
        "integration_proxy": integration_proxy(runtime, state),
        "last_turn": last_turn_layers(runtime),
    }
