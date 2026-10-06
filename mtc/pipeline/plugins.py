"""The plugin protocol for command and tool dispatch; no plugin ships.

Two kinds of plugin may join a turn:

- A ``TurnPlugin`` sees the incoming message first (``try_handle``). It may
  answer with a ``PluginOutcome`` of kind ``reply``, which ends the turn: the
  content is stored under the plugin's name with the ``tool`` role, never as
  the system's words, and no assistant row is written. It may instead return a
  ``tool_turn``: the content joins the brain window as a tool turn and the
  turn continues to the brain.
- A ``ResponsePlugin`` scans the brain's reply (``scan``) and may ask for one
  tool hop: it returns the tool's name and a ``ToolOutcome``. The outcome
  becomes a tool turn and the brain is called again with it, up to
  ``max_hops`` hops; the cap ends the loop with the last reply and is counted.

A tool turn is a history entry with role ``tool`` and the tool's name as its
speaker (the brain window renders it as a labelled user turn). A failed
``ToolOutcome`` says only that the tool failed: its content is never quoted,
because a tool's error text is not conversation. ``PluginContext`` carries the
verified identity, the message, a read-only view of the history, the brain and
an optional recall callable; it carries no persistence handle, so a plugin can
read but never write the stores.

``PluginRegistry.register_capabilities`` registers every plugin's
``describe()`` text with the prompt assembler, so the prompt advertises a
tool only when one is installed. The registry ships empty: the reference
command handlers and the response tool scanner carried only excluded features
and nothing from them is here. A plugin that raises is counted by name
(``status()['failures']``), logged at WARNING the first time and every
hundredth, skipped, and re-raised when ``DEBUG`` is true.

Config keys: ``DEBUG`` (through ``settings`` or ``get_settings()``). No state
is persisted. Messages, replies and tool results are private at runtime and
are never logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from mtc.brain.protocol import BrainProtocol
from mtc.core.config import Settings, get_settings
from mtc.pipeline.speaker_trust import SpeakerIdentity

logger = logging.getLogger(__name__)

TOOL_ROLE = "tool"
PluginKind = Literal["reply", "tool_turn"]
PLUGIN_KINDS: Tuple[str, ...] = ("reply", "tool_turn")
FAILED_TOOL_TEXT = "The tool {name} failed and returned no result."

RecallFn = Callable[[str, int], Awaitable[List[Dict[str, Any]]]]
RerunFn = Callable[[List[Dict[str, Any]]], Awaitable[str]]


@dataclass(frozen=True)
class PluginContext:
    """What a plugin may read: no store, no session, no persistence."""

    identity: SpeakerIdentity
    message: str
    history: Tuple[Mapping[str, Any], ...]
    brain: BrainProtocol
    recall: Optional[RecallFn] = None


@dataclass(frozen=True)
class PluginOutcome:
    """What a turn plugin did with the message."""

    kind: PluginKind
    tool_name: str
    content: str

    def __post_init__(self) -> None:
        if self.kind not in PLUGIN_KINDS:
            raise ValueError(f"PluginOutcome.kind must be one of {PLUGIN_KINDS}")
        if not str(self.tool_name).strip():
            raise ValueError("PluginOutcome needs the plugin's name")


@dataclass(frozen=True)
class ToolOutcome:
    """A tool's result; a failed outcome's content is never quoted."""

    ok: bool
    content: str


@runtime_checkable
class TurnPlugin(Protocol):
    """Sees the incoming message before the brain does."""

    name: str

    def describe(self) -> str:
        """One line for the prompt's capabilities section."""
        ...

    async def try_handle(self, ctx: PluginContext) -> Optional[PluginOutcome]:
        """Return an outcome to act on, or None to leave the turn alone."""
        ...


@runtime_checkable
class ResponsePlugin(Protocol):
    """Scans the brain's reply for a tool request."""

    name: str

    def describe(self) -> str:
        """One line for the prompt's capabilities section."""
        ...

    async def scan(
        self, response: str, ctx: PluginContext
    ) -> Optional[Tuple[str, ToolOutcome]]:
        """Return ``(tool_name, outcome)`` for one hop, or None."""
        ...


@dataclass
class ResolvedResponse:
    """The reply after the response plugins had their hops."""

    text: str
    tool_turns: List[Dict[str, Any]] = field(default_factory=list)
    hops: int = 0
    capped: bool = False


def tool_turn(
    tool_name: str, outcome: ToolOutcome, at: Optional[datetime] = None
) -> Dict[str, Any]:
    """The history entry a tool result becomes.

    Role ``tool``, the tool's name as speaker, the content when the tool
    succeeded and a plain failure line otherwise (the failed content is
    never quoted).
    """
    name = str(tool_name).strip()
    if not name:
        raise ValueError("a tool turn needs the tool's name")
    content = (
        str(outcome.content or "") if outcome.ok else FAILED_TOOL_TEXT.format(name=name)
    )
    return {
        "role": TOOL_ROLE,
        "speaker": name,
        "content": content,
        "timestamp": at if at is not None else datetime.now(timezone.utc),
        "tool_ok": bool(outcome.ok),
    }


def _check(plugin: Any, protocol: type, label: str) -> str:
    if not isinstance(plugin, protocol):
        raise TypeError(f"{label} plugins must implement {protocol.__name__}")
    name = str(getattr(plugin, "name", "")).strip()
    if not name:
        raise ValueError(f"{label} plugin needs a non-empty name")
    return name


class PluginRegistry:
    """The installed plugins and the tool-hop loop."""

    def __init__(
        self,
        turn_plugins: Sequence[Any] = (),
        response_plugins: Sequence[Any] = (),
        max_hops: int = 4,
        *,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if int(max_hops) < 0:
            raise ValueError("max_hops must be at least 0")
        self.max_hops = int(max_hops)
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._turn: Dict[str, Any] = {}
        self._response: Dict[str, Any] = {}
        for plugin in turn_plugins:
            name = _check(plugin, TurnPlugin, "turn")
            if name in self._turn:
                raise ValueError(f"duplicate turn plugin name {name!r}")
            self._turn[name] = plugin
        for plugin in response_plugins:
            name = _check(plugin, ResponsePlugin, "response")
            if name in self._response:
                raise ValueError(f"duplicate response plugin name {name!r}")
            self._response[name] = plugin
        self._failures: Dict[str, int] = {}
        self._counts = {"offered": 0, "handled": 0, "scans": 0, "hops": 0}
        self._capped = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self._turn and not self._response

    def names(self) -> List[str]:
        """Every installed plugin name, turn plugins first."""
        return list(self._turn) + [n for n in self._response if n not in self._turn]

    def describe_all(self) -> Dict[str, str]:
        """Each plugin's capability text by name (empty texts left out)."""
        texts: Dict[str, str] = {}
        for name, plugin in list(self._turn.items()) + list(self._response.items()):
            text = str(plugin.describe() or "").strip()
            if text and name not in texts:
                texts[name] = text
        return texts

    def register_capabilities(self, assembler: Any) -> List[str]:
        """Register every plugin's text with the prompt assembler; the prompt
        then advertises exactly the tools that are installed."""
        registered: List[str] = []
        for name, text in self.describe_all().items():
            assembler.register_capability(name, text)
            registered.append(name)
        return registered

    # ------------------------------------------------------------------
    # Failures
    # ------------------------------------------------------------------

    def _debug(self) -> bool:
        active = self._settings if self._settings is not None else get_settings()
        return bool(getattr(active, "DEBUG", False))

    def _fail(self, name: str, exc: BaseException) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count == 1 or count % 100 == 0:
            logger.warning(
                "plugin %s failed (%d so far): %s", name, count, type(exc).__name__
            )
        if self._debug():
            raise exc

    # ------------------------------------------------------------------
    # The turn
    # ------------------------------------------------------------------

    async def offer_message(self, ctx: PluginContext) -> Optional[PluginOutcome]:
        """Offer the message to the turn plugins in order; the first outcome
        wins. A plugin that raises is counted and skipped."""
        self._counts["offered"] += 1
        for name, plugin in self._turn.items():
            try:
                outcome = await plugin.try_handle(ctx)
            except Exception as exc:  # noqa: BLE001 - counted, isolated
                self._fail(name, exc)
                continue
            if outcome is None:
                continue
            if not isinstance(outcome, PluginOutcome):
                self._fail(name, TypeError("try_handle must return a PluginOutcome"))
                continue
            self._counts["handled"] += 1
            return outcome
        return None

    async def scan_response(
        self, response: str, ctx: PluginContext
    ) -> Optional[Tuple[str, ToolOutcome]]:
        """Ask the response plugins, in order, for one tool hop."""
        self._counts["scans"] += 1
        for name, plugin in self._response.items():
            try:
                request = await plugin.scan(response, ctx)
            except Exception as exc:  # noqa: BLE001 - counted, isolated
                self._fail(name, exc)
                continue
            if request is None:
                continue
            tool_name, outcome = request
            if not isinstance(outcome, ToolOutcome):
                self._fail(name, TypeError("scan must return a ToolOutcome"))
                continue
            return str(tool_name or name), outcome
        return None

    async def resolve_response(
        self, response: str, ctx: PluginContext, rerun: RerunFn
    ) -> ResolvedResponse:
        """Run the tool-hop loop over a reply.

        Each hop turns the requested tool's outcome into a tool turn and asks
        ``rerun`` (the caller's brain call over the accumulated tool turns)
        for the next reply. After ``max_hops`` hops a further request is not
        served: the last reply stands and ``capped`` is set.
        """
        resolved = ResolvedResponse(text=str(response or ""))
        if not self._response:
            return resolved
        while True:
            request = await self.scan_response(resolved.text, ctx)
            if request is None:
                return resolved
            if resolved.hops >= self.max_hops:
                resolved.capped = True
                self._capped += 1
                logger.warning(
                    "tool hop cap of %d reached; the last reply stands", self.max_hops
                )
                return resolved
            tool_name, outcome = request
            resolved.tool_turns.append(tool_turn(tool_name, outcome, self._clock()))
            resolved.hops += 1
            self._counts["hops"] += 1
            resolved.text = str(await rerun(list(resolved.tool_turns)) or "")

    def status(self) -> Dict[str, Any]:
        """Installed names, counters and failures; no content."""
        return {
            "turn_plugins": list(self._turn),
            "response_plugins": list(self._response),
            "max_hops": self.max_hops,
            "counts": dict(self._counts),
            "capped": self._capped,
            "failures": dict(self._failures),
        }


__all__ = (
    "FAILED_TOOL_TEXT",
    "PLUGIN_KINDS",
    "TOOL_ROLE",
    "PluginContext",
    "PluginKind",
    "PluginOutcome",
    "PluginRegistry",
    "RecallFn",
    "ResolvedResponse",
    "ResponsePlugin",
    "ToolOutcome",
    "TurnPlugin",
    "tool_turn",
)
