"""
Local Chat-Completions Provider
===============================

Speaks the common chat-completions HTTP format to a local model server (the
format most local inference servers expose). The request shape follows the
configured template family (``mtc.brain.families``): the ``generic`` family
sends only the standard fields, the reasoning families add their stop tokens,
thinking switch and extended sampling fields.

Reasoning is read from a ``reasoning_content`` or ``reasoning`` field or from
inline tags in the content, normalized to canonical think tags. A streamed
reply rebuilds reasoning deltas as standalone ``<think>`` and ``</think>``
chunks, passes inline canonical tags through, maps a family's own tags to the
canonical pair, and closes a block the stream left open. Usage reported on the
last stream event is kept in ``last_stream_usage``.

Config keys read by ``from_settings`` (at call time): ``LLM_BASE_URL`` (empty
derives ``LLM_URL`` from ``LLM_HOST`` and ``LLM_PORT``), ``LLM_MODEL`` (empty
sends no model field), ``LLM_COMPLETIONS_PATH`` and ``LLM_HEALTH_PATHS`` (URL
paths under the base URL; a leading slash is optional),
``LLM_TEMPLATE_FAMILY``, ``LLM_CONNECT_TIMEOUT_S``, ``LLM_REQUEST_TIMEOUT_S``,
``LLM_EXTRA_BODY`` (merged last into every request body) and ``CONTEXT_SIZE``
(the reported context window). ``GENERATION_TIMEOUT_S`` is read at each
streamed call: the wait for the stream's next bytes is
``max(request timeout, GENERATION_TIMEOUT_S + 60)`` unless a fixed stream read
timeout was given, so a long silent prefill is not cut short inside the
generation bound that protects it.

``httpx`` is optional (the ``llm`` extra) and imported only when a client is
needed. An injected client is used as given and never closed here. No state is
persisted; request bodies carry private conversation text and are never
logged.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

from mtc.brain.families import TemplateFamily, get_family
from mtc.brain.protocol import ConfigurationError
from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ModelInfo,
    ProviderError,
    ProviderTimeoutError,
)
from mtc.brain.thinking import (
    THINK_CLOSE,
    THINK_OPEN,
    StreamTagNormalizer,
    normalize_reasoning,
)
from mtc.core.config import Settings, csv_list, get_settings
from mtc.memory.clients import require_client

logger = logging.getLogger(__name__)

_ERROR_BODY_CHARS = 200


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_request_body(
    family: TemplateFamily,
    params: CompletionParams,
    *,
    model: str = "",
    extra_body: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """The chat-completions body for ``params`` under ``family``.

    ``params.extra_body`` is merged after the family fields and ``extra_body``
    (the configured ``LLM_EXTRA_BODY``) last of all.
    """
    body: Dict[str, Any] = {}
    chosen_model = params.model or model
    if chosen_model:
        body["model"] = chosen_model
    body["messages"] = params.messages
    if params.temperature is not None:
        body["temperature"] = params.temperature
    if params.top_p is not None:
        body["top_p"] = params.top_p
    if params.max_tokens is not None:
        body["max_tokens"] = params.max_tokens
    if family.extended_sampling:
        if params.top_k is not None:
            body["top_k"] = params.top_k
        if params.min_p is not None:
            body["min_p"] = params.min_p
    penalty = getattr(params, family.penalty_field)
    if penalty is not None:
        body[family.penalty_field] = penalty
    stop: List[str] = list(params.stop or [])
    if params.thinking_mode:
        stop.extend(token for token in family.stop_when_thinking if token not in stop)
    if stop:
        body["stop"] = stop
    if not params.thinking_mode and family.thinking_off == "chat_template_kwargs":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body["stream"] = stream
    if stream and family.stream_usage:
        body["stream_options"] = {"include_usage": True}
    body.update(params.extra_body)
    body.update(extra_body or {})
    return body


def _usage_counts(usage: Any) -> Dict[str, Optional[int]]:
    usage = usage if isinstance(usage, dict) else {}

    def count(key: str) -> Optional[int]:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    prompt, completion = count("prompt_tokens"), count("completion_tokens")
    total = count("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


class LocalChatCompletionsProvider:
    """A provider for a local server that speaks chat completions over HTTP."""

    def __init__(
        self,
        base_url: str,
        model: str = "",
        completions_path: str = "/v1/chat/completions",
        health_paths: Sequence[str] = ("/health", "/v1/models"),
        family: str = "generic",
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 600.0,
        stream_read_timeout_s: Optional[float] = None,
        health_timeout_s: float = 5.0,
        extra_body: Optional[Dict[str, Any]] = None,
        provider_id: str = "local",
        context_window: Optional[int] = None,
        client: Any = None,
        *,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise ConfigurationError(
                "LLM_BASE_URL (or LLM_HOST and LLM_PORT) must name the local server"
            )
        self.base_url = base
        self.model = model or ""
        self.completions_path = completions_path
        self.health_paths = tuple(health_paths)
        self.family = get_family(family)
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.stream_read_timeout_s = stream_read_timeout_s
        self.health_timeout_s = float(health_timeout_s)
        self.extra_body: Dict[str, Any] = dict(extra_body or {})
        self.provider_id = provider_id
        self.context_window = context_window
        self.last_stream_usage: Optional[Dict[str, Any]] = None
        self._settings = settings
        self._clock = clock or _now
        self._client = client
        self._owns_client = client is None

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None, **overrides: Any
    ) -> "LocalChatCompletionsProvider":
        """Build the provider from Settings; ``overrides`` replace arguments."""
        active = settings if settings is not None else get_settings()
        arguments: Dict[str, Any] = {
            "base_url": active.LLM_BASE_URL or active.LLM_URL,
            "model": active.LLM_MODEL,
            "completions_path": active.LLM_COMPLETIONS_PATH,
            "health_paths": tuple(csv_list(active.LLM_HEALTH_PATHS)),
            "family": active.LLM_TEMPLATE_FAMILY,
            "connect_timeout_s": active.LLM_CONNECT_TIMEOUT_S,
            "request_timeout_s": active.LLM_REQUEST_TIMEOUT_S,
            "extra_body": dict(active.LLM_EXTRA_BODY),
            "context_window": active.CONTEXT_SIZE,
            "settings": active,
        }
        arguments.update(overrides)
        return cls(**arguments)

    # -- protocol surface ---------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return self.model_info()

    def model_info(self) -> ModelInfo:
        name = (
            f"{self.model} (local)" if self.model else "local chat-completions server"
        )
        return ModelInfo(
            id=self.provider_id,
            name=name,
            provider_type="local",
            context_window=self.context_window,
            cost="local",
            supports_thinking=self.supports_thinking(),
        )

    def supports_thinking(self) -> bool:
        return self.family.is_reasoning_family

    async def initialize(self) -> None:
        """Build the HTTP client (no connection is opened until a request)."""
        self._ensure_client()

    async def shutdown(self) -> None:
        """Close the client this provider built; an injected one stays open."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def complete(self, params: CompletionParams) -> CompletionResult:
        httpx = self._httpx()
        client = self._ensure_client()
        body = build_request_body(
            self.family, params, model=self.model, extra_body=self.extra_body
        )
        try:
            response = await client.post(
                self._url(self.completions_path),
                json=body,
                timeout=httpx.Timeout(
                    self.request_timeout_s, connect=self.connect_timeout_s
                ),
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"local provider timed out ({type(exc).__name__})"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"local provider request failed ({type(exc).__name__})"
            ) from exc
        if response.status_code != 200:
            raise ProviderError(
                f"local provider: server answered {response.status_code}: "
                f"{response.text[:_ERROR_BODY_CHARS]}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(
                "local provider: unexpected response format", status_code=200
            ) from exc
        content = message.get("content") or ""
        field_reasoning = message.get("reasoning_content") or message.get("reasoning")
        inline_reasoning, text = normalize_reasoning(str(content), self.family.name)
        parts = [part for part in (field_reasoning, inline_reasoning) if part]
        reasoning = "\n".join(str(part).strip() for part in parts) or None
        usage = _usage_counts(data.get("usage"))
        return CompletionResult(
            text=text.strip(),
            reasoning=reasoning,
            finish_reason=choice.get("finish_reason"),
            **usage,
        )

    async def stream(self, params: CompletionParams) -> AsyncIterator[str]:
        httpx = self._httpx()
        client = self._ensure_client()
        body = build_request_body(
            self.family,
            params,
            model=self.model,
            extra_body=self.extra_body,
            stream=True,
        )
        self.last_stream_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
        }
        normalizer = StreamTagNormalizer(self.family.name)
        emitted: List[str] = []
        think_open = False
        try:
            async with client.stream(
                "POST",
                self._url(self.completions_path),
                json=body,
                timeout=self._stream_timeout(httpx),
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread())[:_ERROR_BODY_CHARS]
                    raise ProviderError(
                        f"local provider: server answered {response.status_code}: "
                        f"{detail.decode('utf-8', 'replace')}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    event = self._parse_event(line)
                    if event is None:
                        continue
                    reasoning, content = event
                    if reasoning:
                        if not think_open:
                            think_open = True
                            yield THINK_OPEN
                        yield reasoning
                    if content:
                        if think_open:
                            think_open = False
                            yield THINK_CLOSE
                        out = normalizer.feed(content)
                        if out:
                            emitted.append(out)
                            yield out
                rest = normalizer.flush()
                if rest:
                    emitted.append(rest)
                    yield rest
                if think_open:
                    # The stream ended inside a rebuilt block (a cut or a
                    # reasoning-only reply): close it.
                    yield THINK_CLOSE
                else:
                    text = "".join(emitted)
                    if text.rfind(THINK_OPEN) > text.rfind(THINK_CLOSE):
                        yield THINK_CLOSE
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"local provider stream timed out ({type(exc).__name__})"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"local provider stream failed ({type(exc).__name__})"
            ) from exc

    async def health_check(self) -> HealthStatus:
        """Probe each health path in order; the first 200 marks it healthy."""
        try:
            httpx = self._httpx()
        except ImportError as exc:
            return HealthStatus(False, str(exc), self._clock(), False)
        client = self._ensure_client()
        failures: List[str] = []
        for path in self.health_paths:
            try:
                response = await client.get(
                    self._url(path), timeout=self.health_timeout_s
                )
            except httpx.HTTPError as exc:
                failures.append(f"{path}: {type(exc).__name__}")
                continue
            if response.status_code == 200:
                return HealthStatus(True, f"{path} answered 200", self._clock(), True)
            failures.append(f"{path}: {response.status_code}")
        detail = "no health path answered 200"
        if failures:
            detail += " (" + "; ".join(failures) + ")"
        return HealthStatus(False, detail, self._clock(), bool(failures))

    # -- helpers ------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _httpx():
        return require_client("httpx", "llm")

    def _ensure_client(self):
        if self._client is None:
            httpx = self._httpx()
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.request_timeout_s, connect=self.connect_timeout_s
                )
            )
            self._owns_client = True
        return self._client

    def _stream_timeout(self, httpx):
        read = self.stream_read_timeout_s
        if read is None:
            settings = self._settings if self._settings is not None else get_settings()
            read = max(
                self.request_timeout_s, float(settings.GENERATION_TIMEOUT_S) + 60.0
            )
        return httpx.Timeout(
            self.request_timeout_s, connect=self.connect_timeout_s, read=float(read)
        )

    def _parse_event(self, line: str):
        """Return ``(reasoning, content)`` for one stream line, or None."""
        line = (line or "").strip()
        if not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            data = json.loads(payload)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        stats = self.last_stream_usage
        if data.get("usage") and stats is not None:
            stats.update(_usage_counts(data.get("usage")))
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        choice = choices[0]
        if choice.get("finish_reason") and stats is not None:
            stats["finish_reason"] = choice["finish_reason"]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        content = delta.get("content") or ""
        return str(reasoning), str(content)
