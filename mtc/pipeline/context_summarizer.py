"""Running summary of a session's oldest turns.

When a session's history reaches ``threshold`` entries, ``ContextSummarizer``
compresses the oldest batch (at most ``batch_size`` entries) into a running
summary through the brain and removes exactly those entries from the history.
The batch shrinks to stay within ``max_batch_chars`` characters; message text
is never cut, so a long turn reaches the summarizer whole (the first entry
always enters, whatever its length). Clock notes in the batch roll off with it
and are not shown to the model as speech.

The summary prompt is neutral and third person: it asks for the facts,
decisions, commitments and open questions of the excerpt, and nothing that is
not in it. The reply passes through ``strip_reasoning_blocks``; an empty or
reasoning-only reply, a ``NoAnswerError`` or any other brain failure leaves the
history and the summary exactly as they were, returns status ``no_answer`` and
is counted (``status()['failures']``, the first and every hundredth logged at
WARNING; under ``DEBUG`` a failure other than ``NoAnswerError`` is raised).
No text is ever composed in the model's place.

The running summary is capped at ``summary_cap`` characters by dropping its
oldest sentences. Verbatim quotes from each summarized batch
(``mtc.pipeline.quote_pinning``) are kept on the session beside the summary,
not inside it, so the cap never removes a quote; the quote ledger has its own
``quote_cap`` and drops its oldest quotes first. ``render_summary`` joins both
into the text the prompt's summary section receives.

Config keys: ``DEBUG`` (raise instead of count) and, through quote pinning,
``PROTECTED_NAMES``. Summaries and quotes are private conversation content;
log lines carry counts only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks
from mtc.core.config import get_settings
from mtc.pipeline.quote_pinning import extract_pinned_quotes
from mtc.pipeline.session_state import Session

logger = logging.getLogger(__name__)

SUMMARY_STATUSES = ("summarized", "not_needed", "no_answer")
SUMMARY_MAX_TOKENS = 300
QUOTES_HEADING = "Quoted verbatim from earlier in this conversation:"

SUMMARY_SYSTEM_PROMPT = (
    "You write short, factual summaries of conversation excerpts in the third "
    "person. You report only what the excerpt contains."
)
SUMMARY_INSTRUCTIONS = (
    "Summarize the conversation excerpt below in a few plain sentences, in the "
    "third person. Keep the facts stated, the decisions made, the commitments "
    "given and the questions left open, and say who said what when it matters. "
    "Do not add opinions, feelings or details that are not in the excerpt."
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SummaryResult:
    """What one ``maybe_summarize`` call did.

    ``summarized`` is the number of history entries compressed (0 unless the
    status is ``summarized``); ``summary`` and ``quotes`` are the session's
    running summary and quote ledger after the call.
    """

    summarized: int
    summary: Optional[str]
    quotes: Tuple[str, ...]
    status: str


def render_summary(summary: Optional[str], quotes: Sequence[str]) -> Optional[str]:
    """Join the running summary and the verbatim quotes into the summary
    section's text, or None when both are empty."""
    parts: List[str] = []
    if summary and summary.strip():
        parts.append(summary.strip())
    if quotes:
        parts.append(QUOTES_HEADING + "\n" + "\n".join(f"- {q}" for q in quotes))
    return "\n\n".join(parts) if parts else None


def _cap_summary(text: str, cap: int) -> str:
    """Drop the oldest sentences until ``text`` fits ``cap`` characters; a
    single sentence over the cap keeps its newest words."""
    text = text.strip()
    if len(text) <= cap:
        return text
    sentences = _SENTENCE_END.split(text)
    while len(sentences) > 1 and len(" ".join(sentences)) > cap:
        sentences.pop(0)
    text = " ".join(sentences)
    if len(text) <= cap:
        return text
    tail = text[-cap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


class ContextSummarizer:
    """Compresses a session's oldest turns into its running summary."""

    def __init__(
        self,
        brain: BrainProtocol,
        threshold: int = 40,
        batch_size: int = 20,
        max_batch_chars: int = 12000,
        summary_cap: int = 2000,
        quote_cap: int = 2000,
        settings: Any = None,
    ) -> None:
        if brain is None:
            raise ValueError(
                "ContextSummarizer needs a brain; without one, history is "
                "trimmed, never summarized by code"
            )
        if int(batch_size) < 1 or int(threshold) < 1:
            raise ValueError("threshold and batch_size must be at least 1")
        self._brain = brain
        self.threshold = int(threshold)
        self.batch_size = int(batch_size)
        self.max_batch_chars = int(max_batch_chars)
        self.summary_cap = int(summary_cap)
        self.quote_cap = int(quote_cap)
        self._settings = settings
        self._locks: Dict[str, asyncio.Lock] = {}
        self._failures: Dict[str, int] = {}
        self._counters = {"cycles": 0, "summarized_messages": 0, "not_needed": 0}

    def _cfg(self) -> Any:
        return self._settings if self._settings is not None else get_settings()

    def set_brain(self, brain: BrainProtocol) -> None:
        """Replace the brain (for example after the runtime rebuilt it)."""
        if brain is None:
            raise ValueError("set_brain needs a brain")
        self._brain = brain

    def _count_failure(self, name: str, exc: Optional[BaseException]) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count == 1 or count % 100 == 0:
            logger.warning(
                "summary step %s produced no summary (%d so far)%s",
                name,
                count,
                f": {type(exc).__name__}" if exc is not None else "",
            )

    def _batch(self, history: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        batch: List[Mapping[str, Any]] = []
        total = 0
        for entry in history[: self.batch_size]:
            size = len(str(entry.get("content") or ""))
            if batch and total + size > self.max_batch_chars:
                break
            batch.append(entry)
            total += size
        return batch

    @staticmethod
    def build_prompt(
        batch: Sequence[Mapping[str, Any]], previous: Optional[str]
    ) -> str:
        """The summary prompt for one batch: the instructions, the previous
        summary when there is one, and every message in full."""
        lines = []
        for entry in batch:
            if entry.get("is_clock_note"):
                continue
            speaker = str(entry.get("speaker") or entry.get("role") or "unknown")
            lines.append(f"{speaker}: {entry.get('content') or ''}")
        parts = [SUMMARY_INSTRUCTIONS]
        if previous and previous.strip():
            parts.append("Summary so far:\n" + previous.strip())
        parts.append("Excerpt:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _result(self, session: Session, status: str, count: int = 0) -> SummaryResult:
        return SummaryResult(
            summarized=count,
            summary=session.summary,
            quotes=tuple(session.pinned_quotes),
            status=status,
        )

    def _merge_quotes(self, session: Session, batch: Sequence[Mapping[str, Any]]):
        found = extract_pinned_quotes(
            batch, max_chars=self.quote_cap, settings=self._settings
        )
        for quote in found:
            if quote not in session.pinned_quotes:
                session.pinned_quotes.append(quote)
        while (
            session.pinned_quotes
            and len("\n".join(session.pinned_quotes)) > self.quote_cap
        ):
            session.pinned_quotes.pop(0)

    async def maybe_summarize(self, session: Session) -> SummaryResult:
        """Summarize the oldest batch when the history has reached the
        threshold; otherwise report ``not_needed``."""
        if len(session.history) < self.threshold:
            self._counters["not_needed"] += 1
            return self._result(session, "not_needed")
        lock = self._locks.setdefault(session.key, asyncio.Lock())
        async with lock:
            if len(session.history) < self.threshold:
                self._counters["not_needed"] += 1
                return self._result(session, "not_needed")
            batch = self._batch(session.history)
            if not batch:
                self._counters["not_needed"] += 1
                return self._result(session, "not_needed")
            spoken = [entry for entry in batch if not entry.get("is_clock_note")]
            self._counters["cycles"] += 1
            if spoken:
                prompt = self.build_prompt(spoken, session.summary)
                try:
                    reply = await self._brain.think(
                        prompt,
                        system_prompt=SUMMARY_SYSTEM_PROMPT,
                        max_tokens=SUMMARY_MAX_TOKENS,
                        thinking_mode=False,
                    )
                except NoAnswerError as exc:
                    self._count_failure(f"no_answer_{exc.reason}", None)
                    return self._result(session, "no_answer")
                except Exception as exc:
                    self._count_failure(f"error_{type(exc).__name__}", exc)
                    if bool(self._cfg().DEBUG):
                        raise
                    return self._result(session, "no_answer")
                text = strip_reasoning_blocks(getattr(reply, "text", "") or "")
                if not text:
                    self._count_failure("empty_reply", None)
                    return self._result(session, "no_answer")
                combined = (
                    f"{session.summary.strip()} {text}"
                    if session.summary and session.summary.strip()
                    else text
                )
                session.summary = _cap_summary(combined, self.summary_cap)
                self._merge_quotes(session, spoken)
            del session.history[: len(batch)]
            self._counters["summarized_messages"] += len(batch)
            logger.info(
                "summarized %d history entries; %d remain",
                len(batch),
                len(session.history),
            )
            return self._result(session, "summarized", len(batch))

    def status(self) -> Dict[str, Any]:
        """Counters and failures by step name; no summary text."""
        return {
            **self._counters,
            "failures": dict(self._failures),
            "threshold": self.threshold,
            "batch_size": self.batch_size,
        }


__all__ = (
    "QUOTES_HEADING",
    "SUMMARY_STATUSES",
    "ContextSummarizer",
    "SummaryResult",
    "render_summary",
)
