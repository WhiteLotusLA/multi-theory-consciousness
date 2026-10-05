"""
LLM Brain
=========

The runtime's brain: a ``ConversationalBrain`` (and so a ``BrainProtocol``)
over the provider registry and the prompt assembler. Every model call in the
system goes through it, and it keeps three promises:

- Honesty. An empty reply raises ``NoAnswerError("empty")``, a reply that is
  only reasoning raises ``NoAnswerError("reasoning_only")``, and a thinking
  loop that outlives ``THINKING_LOOP_MAX_RETRIES`` raises
  ``NoAnswerError("loop")``. Nothing is composed in the model's place, and a
  reply is returned in the canonical form (``<think>R</think>`` and a blank
  line before the content when reasoning exists, the content alone otherwise)
  so callers split it with ``mtc.brain.thinking.split_thinking``.
- One provider. The active provider is ``LLM_PROVIDER``. ``initialize`` runs
  its health check; an unhealthy provider leaves the brain degraded, every
  call raises ``BrainUnavailableError``, and no other provider is tried. A
  provider failure during a call raises the same error; a timeout propagates
  as ``ProviderTimeoutError`` so the pipeline can report it as such.
- One budget pass. ``think_with_context`` renders the system prompt, computes
  the history budget from the prompt size, ``max_tokens`` and ``CONTEXT_SIZE``
  (bounded by ``HISTORY_TOKEN_BUDGET``) and trims the history once with
  ``fit_history``; ``status()`` reports what the last pass dropped. The
  earlier-conversation summary stays inside the system prompt, because some
  chat templates reject a system role after the first message.

The loop detector runs only on reasoning, and only when the call asked for
thinking; each retry raises the template family's penalty field by
``THINKING_LOOP_PENALTY_STEP`` and, on a streamed call, sends
``STREAM_RETRY_SENTINEL`` so the consumer discards the attempt it saw.

``image`` input raises ``CapabilityUnavailableError``: vision is not part of
this release. There is no knowledge-synthesis method here; consolidation
synthesizes through ``BrainProtocol.think``.

Config keys read at call time: ``LLM_PROVIDER``, ``LLM_TEMPLATE_FAMILY``,
``LLM_MAX_TOKENS``, ``TEMPERATURE``, ``TOP_P``, ``TOP_K``, ``MIN_P``,
``REPEAT_PENALTY``, ``THINKING_MODE``, ``THINKING_LOOP_MAX_RETRIES``,
``THINKING_LOOP_PENALTY_STEP``, ``PROVIDER_STATE_PATH``, ``ANTHROPIC_API_KEY``
and ``ANTHROPIC_MODEL`` (through ``from_settings``), plus the budget and prompt
keys their modules read. No state is persisted beyond the registry's active
choice. Messages carry private conversation text and are never logged.
"""

import asyncio
import inspect
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from mtc.brain.budget import BudgetResult, estimate_tokens, fit_history, history_budget
from mtc.brain.families import TemplateFamily, get_family
from mtc.brain.prompt import PromptAssembler, PromptInputs, SpeakerView
from mtc.brain.protocol import (
    NO_ANSWER_REASONS,
    BrainReply,
    BrainUnavailableError,
    CapabilityUnavailableError,
    ConfigurationError,
    NoAnswerError,
)
from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ProviderError,
    ProviderTimeoutError,
)
from mtc.brain.providers.registry import ProviderRegistry
from mtc.brain.thinking import canonical, normalize_reasoning, split_thinking
from mtc.brain.thinking_loop_detector import ThinkingLoopDetected, ThinkingLoopDetector
from mtc.core.config import Settings, get_settings
from mtc.core.paths import resolve_path

logger = logging.getLogger(__name__)

# Sent to a stream consumer when a thinking loop restarts the generation; the
# chunks before it belong to the discarded attempt.
STREAM_RETRY_SENTINEL = "<|retry|>"
HISTORY_ROLES = ("user", "assistant")
DEFAULT_THINK_MAX_TOKENS = 512


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


async def _emit(callback: Callable[[str], Any], chunk: str) -> None:
    result = callback(chunk)
    if inspect.isawaitable(result):
        await result


class LLMBrain:
    """The provider-backed brain the runtime hands to every module."""

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        assembler: Optional[PromptAssembler] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings
        self.registry = (
            registry if registry is not None else ProviderRegistry(settings=settings)
        )
        self._assembler = assembler
        self._health: Optional[HealthStatus] = None
        self._initialized = False
        self.calls = 0
        self.retries = 0
        self.provider_errors = 0
        self.timeouts = 0
        self.no_answers: Dict[str, int] = {reason: 0 for reason in NO_ANSWER_REASONS}
        self.last_budget: Optional[BudgetResult] = None
        self.last_sections: List[Tuple[str, int]] = []
        self.last_stream_reply: Optional[BrainReply] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "LLMBrain":
        """Register the configured providers and make ``LLM_PROVIDER`` active.

        The local provider is always registered, the Anthropic provider when
        both ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_MODEL`` are set, and the mock
        provider when it is selected. Selecting an unregistered provider raises
        ``ConfigurationError`` naming the keys.
        """
        from mtc.brain.providers.local_chat import LocalChatCompletionsProvider
        from mtc.brain.providers.mock_provider import MockProvider

        cfg = settings if settings is not None else get_settings()
        registry = ProviderRegistry(
            state_path=resolve_path(cfg.PROVIDER_STATE_PATH, settings=cfg),
            settings=cfg,
        )
        registry.register(LocalChatCompletionsProvider.from_settings(cfg))
        if cfg.ANTHROPIC_API_KEY and cfg.ANTHROPIC_MODEL:
            from mtc.brain.providers.anthropic_provider import AnthropicProvider

            registry.register(AnthropicProvider.from_settings(cfg))
        elif cfg.LLM_PROVIDER == "anthropic":
            raise ConfigurationError(
                "LLM_PROVIDER is anthropic but ANTHROPIC_API_KEY or ANTHROPIC_MODEL "
                "is empty"
            )
        if cfg.LLM_PROVIDER == "mock":
            registry.register(MockProvider())
        registry.set_active(cfg.LLM_PROVIDER)
        return cls(
            registry=registry,
            assembler=PromptAssembler.from_settings(cfg),
            settings=cfg,
        )

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    @property
    def assembler(self) -> PromptAssembler:
        """The prompt assembler, built from Settings on first use."""
        if self._assembler is None:
            self._assembler = PromptAssembler.from_settings(self._settings)
        return self._assembler

    # ------------------------------------------------------------------
    # Lifecycle and health
    # ------------------------------------------------------------------

    async def initialize(self) -> HealthStatus:
        """Initialize the active provider and record its health.

        An unhealthy provider leaves the brain degraded: calls raise
        ``BrainUnavailableError`` until a later ``initialize`` or
        ``refresh_health`` finds it healthy. No other provider is tried.
        """
        provider = self._active_provider()
        try:
            await provider.initialize()
            health = await provider.health_check()
        except ProviderError as exc:
            health = HealthStatus(False, f"{type(exc).__name__}: {exc}", _now(), True)
        except ImportError as exc:
            health = HealthStatus(False, str(exc), _now(), False)
        self._record_health(provider, health)
        self._initialized = True
        return health

    async def refresh_health(self) -> HealthStatus:
        """Re-run the active provider's health check and record the result."""
        provider = self._active_provider()
        try:
            health = await provider.health_check()
        except ProviderError as exc:
            health = HealthStatus(False, f"{type(exc).__name__}: {exc}", _now(), True)
        except ImportError as exc:
            health = HealthStatus(False, str(exc), _now(), False)
        self._record_health(provider, health)
        return health

    def _record_health(self, provider: Any, health: HealthStatus) -> None:
        self._health = health
        if not health.ok:
            logger.warning(
                "brain degraded: provider %s is unavailable (%s); no other "
                "provider is used",
                provider.model_info().id,
                health.detail,
            )

    @property
    def available(self) -> Optional[bool]:
        """True or False after a health check; None before one ran."""
        return None if self._health is None else bool(self._health.ok)

    @property
    def model_name(self) -> str:
        try:
            return self.registry.active().model_info().name
        except RuntimeError:
            return "unavailable"

    async def shutdown(self) -> None:
        """Shut the active provider down; the brain can be initialized again."""
        with suppress(RuntimeError):
            await self.registry.active().shutdown()
        self._initialized = False

    def _active_provider(self) -> Any:
        try:
            return self.registry.active()
        except RuntimeError as exc:
            raise BrainUnavailableError(str(exc)) from exc

    def _ready_provider(self) -> Any:
        provider = self._active_provider()
        if self._health is not None and not self._health.ok:
            raise BrainUnavailableError(
                f"provider {provider.model_info().id} is unavailable: "
                f"{self._health.detail}"
            )
        return provider

    # ------------------------------------------------------------------
    # BrainProtocol
    # ------------------------------------------------------------------

    async def think(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_THINK_MAX_TOKENS,
        temperature: float = 0.7,
        thinking_mode: bool = False,
    ) -> BrainReply:
        """One-shot call: the system prompt (when given) and one user turn."""
        messages = self._one_shot_messages(prompt, system_prompt)
        return await self._complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode,
        )

    async def think_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_THINK_MAX_TOKENS,
        temperature: float = 0.7,
        thinking_mode: bool = False,
    ) -> AsyncIterator[str]:
        """Yield the reply as it streams; reasoning arrives inside think tags.

        ``STREAM_RETRY_SENTINEL`` is yielded when a thinking loop restarts the
        attempt. The finished ``BrainReply`` is kept in ``last_stream_reply``;
        a no-answer or provider failure is raised from the iterator.
        """
        messages = self._one_shot_messages(prompt, system_prompt)
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        done = object()

        async def on_chunk(chunk: str) -> None:
            await queue.put(chunk)

        async def run() -> None:
            try:
                reply = await self._complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_mode=thinking_mode,
                    stream_callback=on_chunk,
                )
                await queue.put(reply)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the reader
                await queue.put(exc)
            finally:
                await queue.put(done)

        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, BrainReply):
                    self.last_stream_reply = item
                    continue
                yield item
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7
    ) -> str:
        """The spoken text only, reasoning removed; raises ``NoAnswerError``."""
        reply = await self.think(
            prompt, max_tokens=max_tokens, temperature=temperature, thinking_mode=False
        )
        _reasoning, spoken = split_thinking(reply.text)
        if not spoken:
            self._count_no_answer("reasoning_only")
            raise NoAnswerError(
                "the model produced reasoning but no spoken text",
                reason="reasoning_only",
            )
        return spoken

    @staticmethod
    def _one_shot_messages(prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if _text(system_prompt):
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # ConversationalBrain
    # ------------------------------------------------------------------

    async def think_with_context(
        self,
        current_prompt: str,
        *,
        conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
        core_memory: Optional[Any] = None,
        ctm_thoughts: Optional[Sequence[Mapping[str, Any]]] = None,
        thought_gists: Optional[Sequence[Mapping[str, Any]]] = None,
        coherence_context: Optional[Any] = None,
        retrieved_memories: Optional[Sequence[Mapping[str, Any]]] = None,
        retrieved_concepts: Optional[Sequence[Any]] = None,
        consciousness_context: Optional[Any] = None,
        generation_params: Optional[Mapping[str, Any]] = None,
        behavioral_directives: Optional[Any] = None,
        consciousness_self_awareness: Optional[Any] = None,
        conversation_summary: Optional[str] = None,
        window_notes: Optional[Any] = None,
        image: Optional[Any] = None,
        speaker: Optional[SpeakerView] = None,
        stream_callback: Optional[Callable[[str], Any]] = None,
        neural_state: Optional[Any] = None,
        attention_guidance: Optional[Any] = None,
        metacognitive_insights: Optional[Any] = None,
        active_inference_context: Optional[Any] = None,
        global_workspace_context: Optional[Any] = None,
        max_tokens: Optional[int] = None,
    ) -> BrainReply:
        """Render the system prompt, fit the history once and call the model."""
        if image is not None:
            raise CapabilityUnavailableError(
                "image input is not part of this release (VISION_ENABLED is false)"
            )
        if speaker is not None and not isinstance(speaker, SpeakerView):
            raise TypeError(
                "speaker must be a SpeakerView built from the verified identity, "
                "not a name"
            )
        cfg = self._cfg()
        provider = self._ready_provider()
        overrides = dict(generation_params or {})
        chosen_max = max_tokens or overrides.get("max_tokens") or cfg.LLM_MAX_TOKENS
        temperature = overrides.get("temperature")
        if temperature is None:
            temperature = cfg.TEMPERATURE
        thinking = overrides.get("thinking_mode")
        if thinking is None:
            thinking = bool(cfg.THINKING_MODE) and bool(provider.supports_thinking())

        history = list(conversation_history or [])
        last_turn_at = history[-1].get("timestamp") if history else None
        inputs = PromptInputs(
            speaker=speaker,
            core_memory=self._core_memory(core_memory),
            ctm_thoughts=list(ctm_thoughts or []),
            thought_gists=list(thought_gists or []),
            neural_state=neural_state,
            attention_guidance=attention_guidance,
            metacognitive_insights=metacognitive_insights,
            consciousness_narrative=consciousness_context,
            active_inference=active_inference_context,
            global_workspace=global_workspace_context,
            memories=list(retrieved_memories or []),
            concepts=list(retrieved_concepts or []),
            coherence=coherence_context,
            summary=conversation_summary,
            directives=self._items(behavioral_directives),
            self_awareness=consciousness_self_awareness,
            last_turn_at=last_turn_at if isinstance(last_turn_at, datetime) else None,
            window_notes=self._items(window_notes),
        )
        assembled = self.assembler.render(inputs)
        self.last_sections = list(assembled.sections)

        rendered_history = self._render_history(history)
        budget = history_budget(
            estimate_tokens(assembled.text), int(chosen_max), settings=cfg
        )
        fitted = fit_history(rendered_history, budget)
        self.last_budget = fitted

        messages: List[Dict[str, Any]] = [{"role": "system", "content": assembled.text}]
        messages.extend(
            {"role": entry["role"], "content": entry["content"]}
            for entry in fitted.kept
        )
        messages.append({"role": "user", "content": current_prompt})
        return await self._complete(
            messages,
            max_tokens=int(chosen_max),
            temperature=float(temperature),
            thinking_mode=bool(thinking),
            overrides=overrides,
            stream_callback=stream_callback,
        )

    @staticmethod
    def _core_memory(value: Any) -> Optional[Mapping[str, Any]]:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value
        text = _text(value)
        return {"core": text} if text else None

    @staticmethod
    def _items(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item) for item in value if _text(item)]

    @staticmethod
    def _render_history(history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """History entries as message dicts; a user entry's speaker prefixes it."""
        rendered: List[Dict[str, Any]] = []
        for entry in history:
            role = _text(entry.get("role")) or "user"
            if role not in HISTORY_ROLES:
                role = "user"
            content = _text(entry.get("content"))
            if not content:
                continue
            speaker = _text(entry.get("speaker"))
            if role == "user" and speaker:
                content = f"{speaker}: {content}"
            rendered.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": entry.get("timestamp"),
                    "is_clock_note": bool(entry.get("is_clock_note")),
                }
            )
        return rendered

    # ------------------------------------------------------------------
    # The call
    # ------------------------------------------------------------------

    async def _complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        thinking_mode: bool,
        overrides: Optional[Mapping[str, Any]] = None,
        stream_callback: Optional[Callable[[str], Any]] = None,
    ) -> BrainReply:
        provider = self._ready_provider()
        cfg = self._cfg()
        family = get_family(cfg.LLM_TEMPLATE_FAMILY)
        max_retries = max(0, int(cfg.THINKING_LOOP_MAX_RETRIES))
        step = float(cfg.THINKING_LOOP_PENALTY_STEP)
        attempt = 0
        while True:
            params = self._params(
                messages,
                family,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                overrides=overrides,
                penalty_extra=attempt * step,
            )
            self.calls += 1
            try:
                if stream_callback is None:
                    result = await provider.complete(params)
                    reasoning, text = self._split(result, family.name)
                    if thinking_mode and reasoning:
                        ThinkingLoopDetector.for_family(family.name).check(reasoning)
                else:
                    reasoning, text, result = await self._stream(
                        provider, params, family.name, thinking_mode, stream_callback
                    )
            except ThinkingLoopDetected as exc:
                attempt += 1
                logger.warning(
                    "thinking loop detected (%s) on attempt %d; %d retries allowed",
                    exc.strategy,
                    attempt,
                    max_retries,
                )
                if attempt > max_retries:
                    self._count_no_answer("loop")
                    raise NoAnswerError(
                        "the reasoning looped on every attempt", reason="loop"
                    ) from exc
                self.retries += 1
                if stream_callback is not None:
                    await _emit(stream_callback, STREAM_RETRY_SENTINEL)
                continue
            except ProviderTimeoutError:
                self.timeouts += 1
                self.last_error = "timeout"
                raise
            except ProviderError as exc:
                self.provider_errors += 1
                self.last_error = type(exc).__name__
                raise BrainUnavailableError(
                    f"provider {provider.model_info().id} failed: {exc}"
                ) from exc
            return self._reply(provider, result, reasoning, text)

    def _params(
        self,
        messages: List[Dict[str, Any]],
        family: TemplateFamily,
        *,
        max_tokens: int,
        temperature: float,
        thinking_mode: bool,
        overrides: Optional[Mapping[str, Any]],
        penalty_extra: float,
    ) -> CompletionParams:
        cfg = self._cfg()
        given = dict(overrides or {})

        def pick(key: str, default: Any) -> Any:
            value = given.get(key)
            return default if value is None else value

        params = CompletionParams(
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=_opt_float(pick("top_p", cfg.TOP_P)),
            top_k=_opt_int(pick("top_k", cfg.TOP_K)),
            min_p=_opt_float(pick("min_p", cfg.MIN_P)),
            stop=list(given["stop"]) if given.get("stop") else None,
            thinking_mode=bool(thinking_mode),
        )
        if family.penalty_field == "repetition_penalty":
            base = float(pick("repetition_penalty", cfg.REPEAT_PENALTY))
            params.repetition_penalty = base + penalty_extra
        else:
            base = given.get("frequency_penalty")
            if base is not None or penalty_extra > 0:
                params.frequency_penalty = float(base or 0.0) + penalty_extra
        return params

    @staticmethod
    def _split(result: CompletionResult, family: str) -> Tuple[Optional[str], str]:
        """Reasoning and spoken text, with any inline tags the provider left."""
        inline, text = normalize_reasoning(result.text or "", family)
        parts = [part for part in (result.reasoning, inline) if part and part.strip()]
        reasoning = "\n".join(part.strip() for part in parts) or None
        return reasoning, text.strip()

    async def _stream(
        self,
        provider: Any,
        params: CompletionParams,
        family: str,
        thinking_mode: bool,
        callback: Callable[[str], Any],
    ) -> Tuple[Optional[str], str, CompletionResult]:
        detector = ThinkingLoopDetector.for_family(family) if thinking_mode else None
        chunks: List[str] = []
        stream = provider.stream(params)
        try:
            async for chunk in stream:
                chunks.append(chunk)
                if detector is not None:
                    detector.feed(chunk)
                await _emit(callback, chunk)
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with suppress(Exception):
                    await aclose()
        reasoning, text = normalize_reasoning("".join(chunks), family)
        usage = getattr(provider, "last_stream_usage", None) or {}
        result = CompletionResult(
            text=text,
            reasoning=reasoning,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=usage.get("finish_reason"),
        )
        return reasoning, text.strip(), result

    def _reply(
        self,
        provider: Any,
        result: CompletionResult,
        reasoning: Optional[str],
        text: str,
    ) -> BrainReply:
        if not text:
            reason = "reasoning_only" if reasoning else "empty"
            self._count_no_answer(reason)
            raise NoAnswerError(
                (
                    "the model produced reasoning but no spoken text"
                    if reasoning
                    else "the model produced no text"
                ),
                reason=reason,
            )
        return BrainReply(
            text=canonical(reasoning, text),
            finish_reason=result.finish_reason,
            output_tokens=result.output_tokens,
            input_tokens=result.input_tokens,
            provider_id=provider.model_info().id,
        )

    def _count_no_answer(self, reason: str) -> None:
        self.no_answers[reason] = self.no_answers.get(reason, 0) + 1

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Provider, health, counters and the last budget pass, for status pages."""
        cfg = self._cfg()
        try:
            info: Optional[Dict[str, Any]] = (
                self.registry.active().model_info().as_dict()
            )
        except RuntimeError:
            info = None
        budget = self.last_budget
        return {
            "provider": info,
            "active_provider": self.registry.active_id,
            "providers": [item.id for item in self.registry.list_available()],
            "model_name": self.model_name,
            "initialized": self._initialized,
            "available": self.available,
            "health": self._health.as_dict() if self._health is not None else None,
            "family": cfg.LLM_TEMPLATE_FAMILY,
            "thinking_mode": bool(cfg.THINKING_MODE),
            "calls": self.calls,
            "retries": self.retries,
            "provider_errors": self.provider_errors,
            "timeouts": self.timeouts,
            "no_answers": dict(self.no_answers),
            "last_error": self.last_error,
            "last_budget": (
                None
                if budget is None
                else {
                    "kept": len(budget.kept),
                    "dropped": budget.dropped,
                    "estimated_tokens": budget.estimated_tokens,
                    "budget_tokens": budget.budget_tokens,
                }
            ),
            "last_sections": list(self.last_sections),
            "persona_source": (
                self._assembler.persona_source if self._assembler is not None else None
            ),
        }


# ----------------------------------------------------------------------
# Process-wide brain
# ----------------------------------------------------------------------

_brain: Optional[LLMBrain] = None
_brain_lock: Optional[asyncio.Lock] = None
_brain_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _lock_for_running_loop() -> asyncio.Lock:
    """The singleton lock for the running event loop.

    An ``asyncio.Lock`` binds to the loop that first contends on it, and a
    later use from another loop raises. The runtime and the API test client
    each call ``get_brain`` from fresh loops, so the lock is created lazily
    and replaced whenever the running loop is not the one it was made in.
    """
    global _brain_lock, _brain_lock_loop
    loop = asyncio.get_running_loop()
    if _brain_lock is None or _brain_lock_loop is not loop:
        _brain_lock = asyncio.Lock()
        _brain_lock_loop = loop
    return _brain_lock


async def get_brain(settings: Optional[Settings] = None) -> LLMBrain:
    """The process-wide brain, built from Settings and initialized once.

    Concurrent first callers in one event loop wait on one lock, so
    ``from_settings`` and ``initialize`` run once; the instance is published
    only after ``initialize`` returned (degraded or not).
    """
    global _brain
    if _brain is not None:
        return _brain
    async with _lock_for_running_loop():
        if _brain is None:
            brain = LLMBrain.from_settings(settings)
            await brain.initialize()
            _brain = brain
    return _brain


def reset_brain_for_tests() -> None:
    """Forget the process-wide brain and its lock so the next ``get_brain``
    builds anew, in whatever event loop calls it."""
    global _brain, _brain_lock, _brain_lock_loop
    _brain = None
    _brain_lock = None
    _brain_lock_loop = None
