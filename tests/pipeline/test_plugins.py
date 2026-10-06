"""The plugin protocol: empty by default, tool turns never spoken as the
system's words, a hop cap, no persistence handle, registered capability text."""

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from mtc.brain.prompt import PromptAssembler, PromptInputs
from mtc.brain.protocol import FakeBrain
from mtc.core.config import Settings
from mtc.pipeline.plugins import (
    TOOL_ROLE,
    PluginContext,
    PluginOutcome,
    PluginRegistry,
    ResolvedResponse,
    ResponsePlugin,
    ToolOutcome,
    TurnPlugin,
    tool_turn,
)
from mtc.pipeline.speaker_trust import SpeakerTrust

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
        DEBUG=False,
    )
    values.update(overrides)
    return Settings(**values)


def _ctx(message: str = "hello", history=()) -> PluginContext:
    trust = SpeakerTrust(settings=Settings(_env_file=None))
    identity = trust.verify("Alice", None)
    return PluginContext(
        identity=identity,
        message=message,
        history=tuple(history),
        brain=FakeBrain(),
        recall=None,
    )


class EchoReplyPlugin:
    name = "echo"

    def describe(self) -> str:
        return "Echoes a message that starts with echo:."

    async def try_handle(self, ctx: PluginContext) -> Optional[PluginOutcome]:
        if ctx.message.startswith("echo:"):
            return PluginOutcome("reply", self.name, ctx.message[5:].strip())
        return None


class LookupTurnPlugin:
    name = "lookup"

    def describe(self) -> str:
        return "Looks a word up before the reply."

    async def try_handle(self, ctx: PluginContext) -> Optional[PluginOutcome]:
        if "lookup" in ctx.message:
            return PluginOutcome("tool_turn", self.name, "lookup result: 42")
        return None


class AlwaysToolResponsePlugin:
    name = "always"

    def __init__(self, ok: bool = True, content: str = "tool said this"):
        self.ok = ok
        self.content = content
        self.scans = 0

    def describe(self) -> str:
        return "Always asks for one more tool hop."

    async def scan(
        self, response: str, ctx: PluginContext
    ) -> Optional[Tuple[str, ToolOutcome]]:
        self.scans += 1
        return self.name, ToolOutcome(self.ok, self.content)


def test_registry_ships_empty():
    registry = PluginRegistry()
    assert registry.is_empty
    assert registry.names() == []
    assert registry.describe_all() == {}
    assert registry.max_hops == 4
    assert registry.status()["turn_plugins"] == []
    assert registry.status()["response_plugins"] == []


def test_protocols_are_runtime_checkable():
    assert isinstance(EchoReplyPlugin(), TurnPlugin)
    assert isinstance(AlwaysToolResponsePlugin(), ResponsePlugin)
    assert not isinstance(object(), TurnPlugin)
    with pytest.raises(TypeError):
        PluginRegistry(turn_plugins=[object()])
    with pytest.raises(ValueError):
        PluginRegistry(turn_plugins=[EchoReplyPlugin(), EchoReplyPlugin()])


@pytest.mark.asyncio
async def test_reply_plugin_ends_turn_with_tool_role_message():
    registry = PluginRegistry(turn_plugins=[EchoReplyPlugin()])
    outcome = await registry.offer_message(_ctx("echo: the answer"))
    assert outcome == PluginOutcome("reply", "echo", "the answer")
    entry = tool_turn(outcome.tool_name, ToolOutcome(True, outcome.content), at=T0)
    # stored under the plugin's name with role tool, never as the system's words
    assert entry["role"] == TOOL_ROLE
    assert entry["speaker"] == "echo"
    assert entry["content"] == "the answer"
    assert entry["timestamp"] == T0
    assert entry["role"] != "assistant"
    assert await registry.offer_message(_ctx("plain message")) is None


@pytest.mark.asyncio
async def test_tool_turn_plugin_continues_to_brain():
    registry = PluginRegistry(turn_plugins=[EchoReplyPlugin(), LookupTurnPlugin()])
    outcome = await registry.offer_message(_ctx("please lookup this"))
    assert outcome is not None
    assert outcome.kind == "tool_turn"
    assert outcome.tool_name == "lookup"
    entry = tool_turn(outcome.tool_name, ToolOutcome(True, outcome.content))
    assert entry["role"] == TOOL_ROLE
    assert entry["speaker"] == "lookup"
    assert entry["content"] == "lookup result: 42"


@pytest.mark.asyncio
async def test_response_plugin_hop_cap():
    plugin = AlwaysToolResponsePlugin()
    registry = PluginRegistry(response_plugins=[plugin], max_hops=2)
    reruns: List[List[dict]] = []

    async def rerun(tool_turns):
        reruns.append(list(tool_turns))
        return f"reply after {len(tool_turns)} tool turn(s)"

    resolved = await registry.resolve_response("first reply", _ctx(), rerun)
    assert isinstance(resolved, ResolvedResponse)
    assert resolved.hops == 2
    assert resolved.capped is True
    assert len(reruns) == 2
    assert resolved.text == "reply after 2 tool turn(s)"
    assert [t["speaker"] for t in resolved.tool_turns] == ["always", "always"]
    assert registry.status()["capped"] == 1


@pytest.mark.asyncio
async def test_response_plugin_stops_when_no_tool_is_asked():
    class OnceResponsePlugin(AlwaysToolResponsePlugin):
        async def scan(self, response, ctx):
            self.scans += 1
            if self.scans == 1:
                return self.name, ToolOutcome(True, "one result")
            return None

    registry = PluginRegistry(response_plugins=[OnceResponsePlugin()], max_hops=4)

    async def rerun(tool_turns):
        return "final reply"

    resolved = await registry.resolve_response("first", _ctx(), rerun)
    assert resolved.hops == 1
    assert resolved.capped is False
    assert resolved.text == "final reply"


@pytest.mark.asyncio
async def test_failed_tool_outcome_never_quoted_as_content():
    secret = "stack trace with a private path inside"
    plugin = AlwaysToolResponsePlugin(ok=False, content=secret)
    registry = PluginRegistry(response_plugins=[plugin], max_hops=1)
    seen: List[dict] = []

    async def rerun(tool_turns):
        seen.extend(tool_turns)
        return "reply"

    resolved = await registry.resolve_response("first", _ctx(), rerun)
    assert len(seen) == 1
    assert secret not in seen[0]["content"]
    assert "failed" in seen[0]["content"]
    assert seen[0]["speaker"] == "always"
    assert seen[0]["role"] == TOOL_ROLE
    assert secret not in resolved.text
    direct = tool_turn("always", ToolOutcome(False, secret))
    assert secret not in direct["content"]


def test_context_has_no_persistence_handle():
    names = {f.name for f in dataclasses.fields(PluginContext)}
    assert names == {"identity", "message", "history", "brain", "recall"}
    forbidden = ("persist", "repository", "session", "store", "memory", "turn")
    assert not any(any(word in name for word in forbidden) for name in names)
    ctx = _ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.message = "changed"  # type: ignore[misc]


def test_registered_plugin_registers_its_capability_text(tmp_path):
    cfg = _settings(tmp_path)
    assembler = PromptAssembler.from_settings(cfg)
    registry = PluginRegistry(turn_plugins=[LookupTurnPlugin()])
    registered = registry.register_capabilities(assembler)
    assert registered == ["lookup"]
    rendered = assembler.render(PromptInputs()).text
    assert "Looks a word up before the reply." in rendered
    # an empty registry registers nothing
    bare = PromptAssembler.from_settings(cfg)
    assert PluginRegistry().register_capabilities(bare) == []
    assert "Looks a word up" not in bare.render(PromptInputs()).text


@pytest.mark.asyncio
async def test_failing_plugin_is_counted_and_skipped():
    class Broken:
        name = "broken"

        def describe(self) -> str:
            return "Raises."

        async def try_handle(self, ctx: PluginContext) -> Optional[PluginOutcome]:
            raise RuntimeError("plugin bug")

    registry = PluginRegistry(
        turn_plugins=[Broken(), EchoReplyPlugin()],
        settings=Settings(_env_file=None, DEBUG=False),
    )
    outcome = await registry.offer_message(_ctx("echo: still answered"))
    assert outcome is not None and outcome.content == "still answered"
    assert registry.status()["failures"] == {"broken": 1}
    strict = PluginRegistry(
        turn_plugins=[Broken()], settings=Settings(_env_file=None, DEBUG=True)
    )
    with pytest.raises(RuntimeError):
        await strict.offer_message(_ctx("anything"))


def test_tool_turn_requires_a_name():
    with pytest.raises(ValueError):
        tool_turn("", ToolOutcome(True, "x"))
    assert isinstance(tool_turn("t", ToolOutcome(True, "x"))["timestamp"], datetime)
