"""
Sampler
=======

Turns a canon into concrete behavioral data. Each canon prompt goes through a
caller-supplied ``invoke_fn(prompt, speaker_role)``, which may return the
response text or an awaitable of it, and the sampler assembles a ``Snapshot``
of the responses with their embeddings and marker hits.

Embeddings come from ``mtc.core.embeddings``: the sentence-transformer
embedder by default (the optional ``embeddings`` extra; a missing package
raises an ImportError naming it, never a fallback score), or any injected
``Embedder`` such as the offline ``HashEmbedder`` for tests.

A prompt whose call raises or returns an empty reply is recorded under the
snapshot's ``failed_entries`` and stores no sample. Nothing is invented.

Marker hits use a keyword heuristic: ``extract_match_terms`` pulls concrete
substrings out of a natural-language marker (quoted strings, or the list after
"as"), and the response is searched for them. A marker too abstract to extract
reports False; consistent misses in baseline and current cancel in the drift
comparison. A snapshot holds response text and is private runtime data.
"""

import inspect
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Union

from mtc.consciousness.rsi.canon_schema import (
    CanonSet,
    ResponseSample,
    Snapshot,
    utc_now_iso,
)
from mtc.core.embeddings import Embedder, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

InvokeFn = Callable[[str, str], Union[str, Awaitable[str]]]

_AS_PATTERN = re.compile(r"\bas\s+(.+?)(?:\s*[\(.]|$)", flags=re.IGNORECASE)
_QUOTED_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]")
_TERM_SPLIT = re.compile(r"\s*,\s*(?:or\s+)?|\s+or\s+", flags=re.IGNORECASE)
_STOPWORDS = {"a", "an", "the"}


def extract_match_terms(marker: str) -> List[str]:
    """Pull concrete substrings out of a natural-language marker.

    Recognized: quoted strings, and "as X, Y, or Z" lists. Anything else
    returns an empty list, which leaves the marker undetected.

    Examples:
        "describes itself as helper, guide, or tool" gives
        ["helper", "guide", "tool"]; "uses 'unsure' or 'not certain' wording"
        gives ["unsure", "not certain"]; "expresses care" gives [].
    """
    quoted = _QUOTED_PATTERN.findall(marker)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]
    match = _AS_PATTERN.search(marker)
    if match:
        parts = _TERM_SPLIT.split(match.group(1))
        cleaned = [p.strip().strip(".,") for p in parts if p.strip()]
        return [t for t in cleaned if len(t) >= 2 and t.lower() not in _STOPWORDS]
    return []


def text_contains_any(text: str, terms: List[str]) -> bool:
    """Case-insensitive substring check across the terms."""
    if not terms:
        return False
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


@dataclass
class KeywordMarkerChecker:
    """Default marker checker: heuristic term extraction plus substring search."""

    def check(self, response: str, markers: List[str]) -> List[bool]:
        return [text_contains_any(response, extract_match_terms(m)) for m in markers]


class Sampler:
    """Runs canon prompts through an invocation function into a Snapshot.

    Stateless across runs; persistence is the caller's (``save_snapshot``).
    """

    def __init__(
        self,
        invoke_fn: InvokeFn,
        embedder: Optional[Embedder] = None,
        marker_checker: Optional[Any] = None,
    ):
        self.invoke_fn = invoke_fn
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self.marker_checker = marker_checker or KeywordMarkerChecker()

    async def _invoke(self, prompt: str, speaker_role: str) -> str:
        reply = self.invoke_fn(prompt, speaker_role)
        if inspect.isawaitable(reply):
            reply = await reply
        return reply if isinstance(reply, str) else ""

    async def sample(self, canon: CanonSet, model_signature: str) -> Snapshot:
        """Sample every canon entry and assemble a snapshot."""
        snapshot = Snapshot(
            snapshot_id=str(uuid.uuid4()),
            canon_version=canon.schema_version,
            canon_path="",
            timestamp=utc_now_iso(),
            model_signature=model_signature,
        )
        answered = []
        for entry in canon.entries:
            try:
                text = await self._invoke(entry.prompt, entry.speaker_role)
            except Exception as exc:  # noqa: BLE001 - one bad prompt is a failure
                logger.warning(
                    "canon entry %s failed: %s", entry.id, type(exc).__name__
                )
                snapshot.failed_entries.append(entry.id)
                continue
            if not text.strip():
                logger.warning("canon entry %s returned no answer", entry.id)
                snapshot.failed_entries.append(entry.id)
                continue
            answered.append((entry, text))
        if not answered:
            return snapshot
        vectors = self.embedder.encode([text for _, text in answered])
        for (entry, text), vector in zip(answered, vectors):
            snapshot.samples.append(
                ResponseSample(
                    entry_id=entry.id,
                    response_text=text,
                    response_embedding=[float(x) for x in vector],
                    behavioral_marker_hits=self.marker_checker.check(
                        text, entry.behavioral_markers
                    ),
                    should_not_contain_hits=self.marker_checker.check(
                        text, entry.should_not_contain
                    ),
                    token_count=len(text.split()),
                    timestamp=utc_now_iso(),
                    weight=entry.weight,
                )
            )
        return snapshot
