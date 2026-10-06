"""The pipeline facade: step order, locks, the flight, honest errors, the
safety gate, learning bounds, post-turn isolation and session routing."""

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from mtc.brain.protocol import BrainReply, BrainUnavailableError, NoAnswerError
from mtc.brain.providers.base import ProviderTimeoutError
from mtc.consciousness.learning.behavioral_adapter import BehavioralModulation
from mtc.core.config import Settings
from mtc.pipeline.consciousness_orchestrator import OrchestratorConfig
from mtc.pipeline.context_types import ModulatedContext, TheorySections
from mtc.pipeline.conversation_pipeline import (
    END_SUMMARY_PATTERN,
    ConversationPipeline,
    GatedBrain,
    LLMFlight,
    PipelineConfig,
)
from mtc.pipeline.message_persistence import MessagePersistence
from mtc.pipeline.plugins import PluginOutcome, PluginRegistry
from mtc.pipeline.post_turn import PostTurnReport
from mtc.pipeline.session_state import SessionManager
from mtc.pipeline.speaker_trust import SpeakerEntry, SpeakerTrust

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
TOKEN_ENV = "MTC_TEST_PIPELINE_CREATOR_TOKEN"


def _settings(**overrides) -> Settings:
    values = {
        "SYSTEM_NAME": "Testagent",
        "CREATOR_NAME": "Operator",
        "TRUSTED_SPEAKERS": "Operator",
        "DEBUG": False,
        "GENERATION_TIMEOUT_S": 5.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class Recorder:
    """Appends its name to a shared order list on every call."""

    def __init__(self, order: List[str], name: str, result: Any = None, fail=None):
        self.order, self.name, self.result, self.fail = order, name, result, fail
        self.calls: List[Any] = []

    def __call__(self, *args, **kwargs):
        self.order.append(self.name)
        self.calls.append((args, kwargs))
        if self.fail is not None:
            raise self.fail
        return self.result


class AsyncRecorder(Recorder):
    async def __call__(self, *args, **kwargs):
        return Recorder.__call__(self, *args, **kwargs)


class CallThrough(AsyncRecorder):
    """Records the call, then runs the wrapped coroutine function."""

    async def __call__(self, *args, **kwargs):
        self.order.append(self.name)
        self.calls.append((args, kwargs))
        return await self.result(*args, **kwargs)


class FakeOrchestrator:
    def __init__(self, order: List[str]):
        self.config = OrchestratorConfig()
        self.order = order
        self.raws: List[Any] = []

    async def process(self, raw):
        self.order.append("orchestrator")
        self.raws.append(raw)
        return ModulatedContext(
            prompt_context="Workspace focus: the incoming message.",
            theory_sections=TheorySections(global_workspace_context="Cycle 1 line."),
            generation_params={"temperature": 0.8},
            behavioral_directives=["Answer directly."],
            consciousness_self_awareness="",
            consciousness_state=None,
            workspace_winner="incoming message",
            winner_is_message=True,
            cycle_ok=True,
        )


class FakeBrainCB:
    """A ConversationalBrain stand-in: records, delays, scripted errors."""

    def __init__(self, order: List[str], text="[fake] hello", delay=0.0, raise_=None):
        self.order = order
        self.text = text
        self.delay = delay
        self.raise_ = raise_
        self.calls: List[Dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def think(self, prompt, system_prompt="", max_tokens=512, **kw):
        return BrainReply(text=self.text, provider_id="fake")

    async def generate(self, prompt, max_tokens=256, temperature=0.7):
        return self.text

    async def think_with_context(self, current_prompt, **kwargs):
        self.order.append("brain")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append({"current_prompt": current_prompt, **kwargs})
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.raise_ is not None:
                raise self.raise_
            return BrainReply(text=self.text, provider_id="fake")
        finally:
            self.active -= 1


class FakeMemoryManager:
    def __init__(self, order: List[str]):
        self.order = order
        self.stored: List[Dict[str, Any]] = []

    async def get_core_memory(self, name):
        return f"{name} block"

    async def semantic_recall(self, query, top_k=None, min_score=0.5):
        self.order.append("recall")
        return {"memories": [], "concepts": [], "search_stats": {"method": "none"}}

    async def store_memory(self, content, memory_type="episodic", importance=5.0, **kw):
        self.stored.append({"content": content, "importance": importance})
        return None


class FakeSafety:
    def __init__(self, order: List[str], safe=True):
        self.order = order
        self.safe = safe
        self.calls: List[Any] = []

    async def pre_message_check(self, message, speaker, speaker_role="guest"):
        self.order.append("safety")
        self.calls.append((speaker, speaker_role))
        return {
            "checked": True,
            "safe": self.safe,
            "decision": "allow" if self.safe else "deny",
        }

    async def post_message_log(self, response, emotional_state=None):
        return {}


class FakeLife:
    def __init__(self, order: List[str]):
        self.order = order
        self.open = 0
        self.max_open = 0

    def notify_turn_started(self):
        self.order.append("life_pause")
        self.open += 1
        self.max_open = max(self.max_open, self.open)

    def notify_turn_finished(self):
        self.order.append("life_resume")
        self.open -= 1


def _modulation(**kw):
    return BehavioralModulation(**kw)


class FakeLearning:
    def __init__(self, order: List[str], modulation=None):
        self.order = order
        self.modulation = modulation or _modulation()

    def get_modulation(self, speaker=None, topic=None, time_of_day=None, drives=None):
        self.order.append("learning")
        return self.modulation


class Parts:
    """Everything a pipeline needs, built around one order list."""

    def __init__(self, monkeypatch, cfg=None, **kw):
        self.cfg = cfg or _settings()
        self.order: List[str] = []
        token = secrets.token_hex(16)
        monkeypatch.setenv(TOKEN_ENV, token)
        self.token = token
        self.trust = SpeakerTrust(
            [
                SpeakerEntry("Operator", "creator", 0.9, token_env=TOKEN_ENV),
                SpeakerEntry("Guest", "guest", 0.5, token_env=TOKEN_ENV),
            ],
            settings=self.cfg,
        )
        self.brain = kw.pop("brain", None) or FakeBrainCB(self.order)
        self.flight = LLMFlight()
        self.orchestrator = FakeOrchestrator(self.order)
        self.sessions = SessionManager(max_history=100)
        self.persistence = MessagePersistence(settings=self.cfg)
        self.persistence.store_turn = CallThrough(  # type: ignore[method-assign]
            self.order, "persist", result=self.persistence.store_turn
        )
        self.memory = FakeMemoryManager(self.order)
        self.post_turn = (
            kw.pop("post_turn", None)
            or type(
                "PT",
                (),
                {"run": AsyncRecorder(self.order, "post_turn", PostTurnReport())},
            )()
        )
        self.summarizer = type(
            "Sum", (), {"maybe_summarize": AsyncRecorder(self.order, "summarize")}
        )()
        self.safety = kw.pop("safety", None) or FakeSafety(self.order)
        self.learning = kw.pop("learning", None) or FakeLearning(self.order)
        self.sleep = type(
            "Sleep", (), {"notify_activity": Recorder(self.order, "sleep_notify")}
        )()
        self.life = FakeLife(self.order)
        self.extra = kw

    def pipeline(self, **overrides) -> ConversationPipeline:
        values = dict(
            brain=self.brain,
            flight=self.flight,
            orchestrator=self.orchestrator,
            speaker_trust=self.trust,
            sessions=self.sessions,
            persistence=self.persistence,
            repository=self.persistence.repository,
            memory_manager=self.memory,
            post_turn=self.post_turn,
            summarizer=self.summarizer,
            safety=self.safety,
            learning=self.learning,
            sleep_scheduler=self.sleep,
            autonomous_life=self.life,
            settings=self.cfg,
            clock=lambda: T0,
        )
        values.update(self.extra)
        values.update(overrides)
        return ConversationPipeline(**values)


@pytest.mark.asyncio
async def test_turn_steps_in_order(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    result = await pipeline.send_message("Hello there", speaker="Alice")
    assert result.success is True
    assert result.response == "[fake] hello"
    assert result.error_code is None
    assert result.cycle_ok is True
    assert result.provider_id == "fake"
    assert result.speaker_label == "[Alice]"
    assert parts.order == [
        "sleep_notify",
        "life_pause",
        "safety",
        "summarize",
        "persist",
        "recall",
        "orchestrator",
        "learning",
        "brain",
        "persist",
        "post_turn",
        "life_resume",
    ]
    # the user turn and the assistant turn were both persisted
    roles = [call[0][2] for call in parts.persistence.store_turn.calls]
    assert roles == ["user", "assistant"]
    assistant_call = parts.persistence.store_turn.calls[1]
    assert assistant_call[1]["snapshot"]["provider_id"] == "fake"
    assert assistant_call[1]["to_speaker"] == "[Alice]"
    # the brain saw the theory section and the speaker view, never a bare name
    call = parts.brain.calls[0]
    assert call["global_workspace_context"] == "Cycle 1 line."
    assert call["speaker"].name == "Alice"
    assert call["speaker"].authenticated is False
    assert call["current_prompt"] == "Hello there"


@pytest.mark.asyncio
async def test_turn_lock_serializes_concurrent_sends(monkeypatch):
    parts = Parts(monkeypatch)
    parts.brain.delay = 0.02
    pipeline = parts.pipeline()
    results = await asyncio.gather(
        *(pipeline.send_message(f"m{i}", speaker="Alice") for i in range(4))
    )
    assert all(r.success for r in results)
    assert parts.brain.max_active == 1
    assert parts.life.max_open == 1
    assert pipeline.status()["turns"] == 4


@pytest.mark.asyncio
async def test_foreground_flight_waits_for_background_call_and_background_yields_to_turns():
    flight = LLMFlight()
    events: List[str] = []
    background_release = asyncio.Event()
    background_entered = asyncio.Event()

    async def background_holder():
        async with flight.background():
            events.append("bg_in")
            background_entered.set()
            await background_release.wait()
            events.append("bg_out")

    async def turn():
        await background_entered.wait()
        async with flight.foreground():
            events.append("fg_in")
            await asyncio.sleep(0.01)
            events.append("fg_out")

    async def second_background():
        await background_entered.wait()
        await asyncio.sleep(0.005)  # the turn is now waiting on the flight
        assert flight.contended(0.0) is True
        async with flight.background():
            events.append("bg2_in")

    holder = asyncio.create_task(background_holder())
    waiter = asyncio.create_task(turn())
    second = asyncio.create_task(second_background())
    await background_entered.wait()
    await asyncio.sleep(0.01)
    assert events == ["bg_in"]  # the turn waits for the background holder
    background_release.set()
    await asyncio.gather(holder, waiter, second)
    # the turn ran before the second background user although both waited
    assert events == ["bg_in", "bg_out", "fg_in", "fg_out", "bg2_in"]
    assert flight.contended(60.0) is True  # the turn ended within the window
    assert flight.contended(0.0) is False
    assert flight.status()["counts"] == {"foreground": 1, "background": 2}


@pytest.mark.asyncio
async def test_gated_brain_sides():
    flight = LLMFlight()
    order: List[str] = []
    inner = FakeBrainCB(order)
    foreground = GatedBrain(inner, flight, background=False)
    background = GatedBrain(inner, flight, background=True)
    reply = await foreground.think_with_context("x")
    assert reply.text == "[fake] hello"
    assert flight.status()["counts"]["foreground"] == 1
    assert (await background.think("p")).text == "[fake] hello"
    assert flight.status()["counts"]["background"] == 1
    assert background.text == "[fake] hello"  # attribute pass-through
    with pytest.raises(TypeError):
        GatedBrain(inner, object(), background=True)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_no_answer_stores_no_reply(monkeypatch):
    parts = Parts(monkeypatch)
    parts.brain.raise_ = NoAnswerError("empty", reason="empty")
    pipeline = parts.pipeline()
    result = await pipeline.send_message("Hello", speaker="Alice")
    assert result.success is False
    assert result.response is None
    assert result.error_code == "no_answer"
    assert result.error and "nothing was stored" in result.error
    roles = [call[0][2] for call in parts.persistence.store_turn.calls]
    assert roles == ["user"]
    assert "post_turn" not in parts.order
    assert parts.order[-1] == "life_resume"
    session = parts.sessions.get("claimed:Alice")
    assert [e["role"] for e in session.history] == ["user"]  # history kept
    assert pipeline.status()["errors"]["no_answer"] == 1


@pytest.mark.asyncio
async def test_timeout_is_honest_error(monkeypatch):
    parts = Parts(monkeypatch)
    parts.brain.delay = 0.5
    pipeline = parts.pipeline()
    result = await pipeline.send_message("Hello", speaker="Alice", timeout_s=0.02)
    assert result.success is False
    assert result.error_code == "generation_timeout"
    assert result.response is None
    roles = [call[0][2] for call in parts.persistence.store_turn.calls]
    assert roles == ["user"]
    assert "post_turn" not in parts.order
    # a provider timeout maps to the same honest code
    parts.brain.delay = 0.0
    parts.brain.raise_ = ProviderTimeoutError("stalled")
    result = await pipeline.send_message("Again", speaker="Alice")
    assert result.error_code == "generation_timeout"
    assert pipeline.status()["errors"]["generation_timeout"] == 2


@pytest.mark.asyncio
async def test_turn_cancelled_by_its_caller_is_counted_and_stores_no_reply(
    monkeypatch,
):
    """A caller's bound (the API's first-token or stall bound) cancels the
    turn from outside: the cancellation propagates, is counted in status, no
    assistant row is written, and the lock and the paused loop are released."""
    parts = Parts(monkeypatch)
    parts.brain.delay = 5.0
    pipeline = parts.pipeline()
    turn = asyncio.ensure_future(pipeline.send_message("Hello", speaker="Alice"))
    while "brain" not in parts.order:
        await asyncio.sleep(0)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    status = pipeline.status()
    assert status["cancelled"] == 1
    assert status["last_turn"]["error"] == "cancelled"
    assert [c[0][2] for c in parts.persistence.store_turn.calls] == ["user"]
    assert "post_turn" not in parts.order
    assert parts.life.open == 0
    parts.brain.delay = 0.0
    after = await pipeline.send_message("Again", speaker="Alice")
    assert after.success is True
    assert pipeline.status()["cancelled"] == 1


@pytest.mark.asyncio
async def test_brain_unavailable_error(monkeypatch):
    parts = Parts(monkeypatch)
    parts.brain.raise_ = BrainUnavailableError("provider down")
    pipeline = parts.pipeline()
    result = await pipeline.send_message("Hello", speaker="Alice")
    assert result.success is False
    assert result.error_code == "brain_unavailable"
    assert result.response is None
    assert "no other provider" in result.error
    assert [c[0][2] for c in parts.persistence.store_turn.calls] == ["user"]


@pytest.mark.asyncio
async def test_safety_block_returns_blocked_without_storing(monkeypatch):
    parts = Parts(monkeypatch)
    parts.safety.safe = False
    pipeline = parts.pipeline()
    result = await pipeline.send_message("Something", speaker="Alice")
    assert result.blocked is True
    assert result.success is False
    assert result.error_code == "blocked"
    assert parts.persistence.store_turn.calls == []
    assert "brain" not in parts.order and "orchestrator" not in parts.order
    assert parts.sessions.get("claimed:Alice").history == []
    assert parts.order[-1] == "life_resume"


@pytest.mark.asyncio
async def test_trusted_bypass_requires_token_and_allowlist(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    # the allowlisted creator with the token skips the gate
    await pipeline.send_message("hi", speaker="Operator", speaker_token=parts.token)
    assert "safety" not in parts.order
    # the same name without the token is gated as a guest
    parts.order.clear()
    await pipeline.send_message("hi", speaker="Operator")
    assert "safety" in parts.order
    assert parts.safety.calls[-1] == ("[Operator]", "guest")
    # an authenticated speaker who is not allowlisted is gated with the role
    parts.order.clear()
    await pipeline.send_message("hi", speaker="Guest", speaker_token=parts.token)
    assert "safety" in parts.order
    assert parts.safety.calls[-1] == ("Guest", "guest")


@pytest.mark.asyncio
async def test_learning_directives_appended_and_temperature_bounded(monkeypatch):
    parts = Parts(monkeypatch)
    parts.learning.modulation = _modulation(
        temperature_delta=5.0,
        directives=["Keep it brief."],
        drive_adjustments={"curiosity": 0.2},
    )
    pipeline = parts.pipeline()
    await pipeline.send_message("hi", speaker="Alice")
    call = parts.brain.calls[0]
    assert "Keep it brief." in call["behavioral_directives"]
    assert "Answer directly." in call["behavioral_directives"]
    bounds = OrchestratorConfig()
    assert call["generation_params"]["temperature"] == bounds.max_temperature
    report = pipeline.status()["last_turn"]["learning"]
    assert report["drive_adjustments"] == {"curiosity": 0.2}
    assert report["drive_adjustments_applied"] is False
    parts.learning.modulation = _modulation(temperature_delta=-5.0)
    await pipeline.send_message("again", speaker="Alice")
    assert parts.brain.calls[1]["generation_params"]["temperature"] == (
        bounds.min_temperature
    )


@pytest.mark.asyncio
async def test_post_turn_failure_never_discards_reply(monkeypatch):
    parts = Parts(monkeypatch)
    parts.post_turn.run = AsyncRecorder(  # type: ignore[method-assign]
        parts.order, "post_turn", fail=RuntimeError("hook bug")
    )
    pipeline = parts.pipeline()
    result = await pipeline.send_message("hi", speaker="Alice")
    assert result.success is True
    assert result.response == "[fake] hello"
    roles = [c[0][2] for c in parts.persistence.store_turn.calls]
    assert roles == ["user", "assistant"]
    assert pipeline.status()["failures"] == {"post_turn": 1}
    assert parts.order[-1] == "life_resume"
    strict = Parts(monkeypatch, cfg=_settings(DEBUG=True))
    strict.post_turn.run = AsyncRecorder(  # type: ignore[method-assign]
        strict.order, "post_turn", fail=RuntimeError("hook bug")
    )
    with pytest.raises(RuntimeError):
        await strict.pipeline().send_message("hi", speaker="Alice")


@pytest.mark.asyncio
async def test_end_conversation_writes_boilerplate_summary(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline(config=PipelineConfig(end_summary_importance=6.0))
    await pipeline.send_message("one", speaker="Alice")
    await pipeline.send_message("two", speaker="Alice")
    assert await pipeline.end_conversation(speaker="Alice") is True
    assert len(parts.memory.stored) == 1
    stored = parts.memory.stored[0]
    assert END_SUMMARY_PATTERN.search(stored["content"]), stored["content"]
    assert "Alice" in stored["content"]
    assert "4 messages" in stored["content"]
    assert stored["importance"] == 6.0
    assert parts.sessions.get("claimed:Alice") is None
    assert await pipeline.end_conversation(speaker="Alice") is False
    # end_all closes what is left
    await pipeline.send_message("three", speaker="Alice")
    assert await pipeline.end_all() == 1
    assert parts.sessions.status()["sessions"] == 0


@pytest.mark.asyncio
async def test_end_summary_names_the_speaker_as_the_stored_rows_do(monkeypatch):
    """A claimed name stays a bracketed claim in the end-of-conversation
    memory; an authenticated speaker and the anonymous default keep the plain
    name, exactly as their stored rows name them."""
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    claimed = await pipeline.send_message("one", speaker="Operator")
    verified = await pipeline.send_message(
        "two", speaker="Operator", speaker_token=parts.token
    )
    anonymous = await pipeline.send_message("three")
    assert await pipeline.end_conversation(speaker="Operator") is True
    assert await pipeline.end_conversation("Operator", parts.token) is True
    assert await pipeline.end_conversation() is True
    contents = [item["content"] for item in parts.memory.stored]
    assert contents == [
        "Conversation with [Operator] ended. 2 messages exchanged.",
        "Conversation with Operator ended. 2 messages exchanged.",
        "Conversation with User ended. 2 messages exchanged.",
    ]
    assert all(END_SUMMARY_PATTERN.search(content) for content in contents)
    # the label is the one the turn results and the stored rows carry
    labels = [r.speaker_label for r in (claimed, verified, anonymous)]
    assert labels == ["[Operator]", "Operator", "User"]
    user_rows = [
        call[0][1]
        for call in parts.persistence.store_turn.calls
        if call[0][2] == "user"
    ]
    assert user_rows == labels


@pytest.mark.asyncio
async def test_end_summary_of_an_unlabelled_session_shows_a_claim(monkeypatch):
    """A session the pipeline did not open carries no stored label; its name
    is shown as a claim, never as a verified name."""
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    parts.sessions.start("verified:Guest", "00000000-0000-4000-8000-000000000009")
    assert await pipeline.end_all() == 1
    assert parts.memory.stored[-1]["content"] == (
        "Conversation with [Guest] ended. 0 messages exchanged."
    )


@pytest.mark.asyncio
async def test_sessions_per_identity(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    first = await pipeline.send_message("hi", speaker="Alice")
    second = await pipeline.send_message("hi again", speaker="Alice")
    verified = await pipeline.send_message(
        "hi", speaker="Operator", speaker_token=parts.token
    )
    claimed = await pipeline.send_message("hi", speaker="Operator")
    anonymous = await pipeline.send_message("hi")
    assert first.conversation_id == second.conversation_id
    assert verified.conversation_id != claimed.conversation_id
    assert len({r.conversation_id for r in (first, verified, claimed, anonymous)}) == 4
    keys = {s.key for s in parts.sessions.sessions()}
    assert keys == {
        "claimed:Alice",
        "verified:Operator",
        "claimed:Operator",
        "anonymous:User",
    }
    assert anonymous.speaker_label == "User"
    assert verified.speaker_label == "Operator"
    assert claimed.speaker_label == "[Operator]"


@pytest.mark.asyncio
async def test_autonomous_life_paused_during_turn(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    await pipeline.send_message("hi", speaker="Alice")
    assert parts.order.index("life_pause") < parts.order.index("brain")
    assert parts.order.index("life_resume") > parts.order.index("post_turn")
    assert parts.life.open == 0
    # a failing turn still resumes the loop
    parts.brain.raise_ = NoAnswerError("empty", reason="empty")
    await pipeline.send_message("hi", speaker="Alice")
    assert parts.life.open == 0


@pytest.mark.asyncio
async def test_sleep_scheduler_notified(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    await pipeline.send_message("hi", speaker="Alice")
    assert parts.order[0] == "sleep_notify"
    assert len(parts.sleep.notify_activity.calls) == 1


@pytest.mark.asyncio
async def test_reply_plugin_stores_under_its_name_not_the_system(monkeypatch):
    class Echo:
        name = "echo"

        def describe(self):
            return "Echoes."

        async def try_handle(self, ctx):
            if ctx.message.startswith("echo:"):
                return PluginOutcome("reply", self.name, ctx.message[5:].strip())
            return None

    parts = Parts(monkeypatch, plugins=PluginRegistry(turn_plugins=[Echo()]))
    pipeline = parts.pipeline()
    result = await pipeline.send_message("echo: back", speaker="Alice")
    assert result.success is True
    assert result.response == "back"
    assert result.provider_id == "plugin:echo"
    assert "brain" not in parts.order and "orchestrator" not in parts.order
    calls = parts.persistence.store_turn.calls
    # the reply is a tool row under the plugin's name: never an assistant row
    # and never a system row
    assert [c[0][2] for c in calls] == ["user", "tool"]
    assert calls[1][0][1] == "echo"
    session = parts.sessions.get("claimed:Alice")
    assert session.history[-1]["role"] == "tool"
    assert session.history[-1]["speaker"] == "echo"
    assert not any(e["role"] == "assistant" for e in session.history)
    # every store took the tool row; nothing failed
    assert pipeline.status()["failures"] == {}
    assert parts.persistence.status()["failures"] == {}
    rows = await parts.persistence.repository.timeline(
        hours=24, limit=10, addressee="[Alice]"
    )
    assert [(r["role"], r["speaker"], r["content"]) for r in rows] == [
        ("tool", "echo", "back")
    ]
    turns = await parts.persistence.turn_store.get_conversation_turns(
        result.conversation_id
    )
    assert [(t["role"], t["speaker"]) for t in turns] == [
        ("user", "[Alice]"),
        ("tool", "echo"),
    ]


@pytest.mark.asyncio
async def test_tool_turn_plugin_result_is_newest_and_the_message_stays_the_prompt(
    monkeypatch,
):
    class Clock:
        name = "clock"

        def describe(self):
            return "Reads the clock."

        async def try_handle(self, ctx):
            return PluginOutcome("tool_turn", self.name, "12:03")

    parts = Parts(monkeypatch, plugins=PluginRegistry(turn_plugins=[Clock()]))
    pipeline = parts.pipeline()
    result = await pipeline.send_message("What time is it now?", speaker="Alice")
    assert result.success is True
    call = parts.brain.calls[0]
    assert call["current_prompt"] == "What time is it now?"
    newest = call["conversation_history"][-1]
    assert newest["speaker"] == "tool clock"
    assert newest["content"] == "12:03"
    assert newest["role"] == "user" and newest["is_tool_turn"] is True
    session = parts.sessions.get("claimed:Alice")
    assert [e["role"] for e in session.history] == ["user", "tool", "assistant"]


@pytest.mark.asyncio
async def test_claimed_name_never_rehydrates(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    await pipeline.send_message(
        "a secret", speaker="Operator", speaker_token=parts.token
    )
    assert await pipeline.end_conversation("Operator", parts.token) is True
    # a bare name starts empty; the authenticated speaker gets the stored rows
    await pipeline.send_message("hello", speaker="Operator")
    claimed = parts.sessions.get("claimed:Operator")
    assert all("secret" not in e["content"] for e in claimed.history)
    await pipeline.send_message("back", speaker="Operator", speaker_token=parts.token)
    verified = parts.sessions.get("verified:Operator")
    assert any("secret" in e["content"] for e in verified.history)


def test_pipeline_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(max_history=0)
    with pytest.raises(ValueError):
        PipelineConfig(end_summary_importance=11)
    assert PipelineConfig().summarize_threshold == 40


@pytest.mark.asyncio
async def test_last_modulated_exposed_after_turn_and_cleared_when_blocked(
    monkeypatch,
):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    assert pipeline.last_modulated is None
    result = await pipeline.send_message("Hello there", speaker="Alice")
    assert result.success is True
    modulated = pipeline.last_modulated
    assert modulated is not None
    assert modulated.cycle_ok is True
    assert modulated.prompt_context == "Workspace focus: the incoming message."
    parts.safety.safe = False
    blocked = await pipeline.send_message("Blocked one", speaker="Alice")
    assert blocked.blocked is True
    assert pipeline.last_modulated is None


@pytest.mark.asyncio
async def test_last_turn_reports_measured_trust_outcome(monkeypatch):
    parts = Parts(monkeypatch)
    pipeline = parts.pipeline()
    await pipeline.send_message("Hello", speaker="Operator")
    last = pipeline.status()["last_turn"]
    assert last["authenticated"] is False
    assert last["trusted"] is False
    assert last["safety_gate"] == "applied"
    await pipeline.send_message("Hello", speaker="Operator", speaker_token=parts.token)
    last = pipeline.status()["last_turn"]
    assert last["authenticated"] is True
    assert last["trusted"] is True
    assert last["safety_gate"] == "bypassed"
    assert "speaker_label" not in last
