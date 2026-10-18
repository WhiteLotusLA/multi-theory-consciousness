"""
Affect shadow taps
==================

Small, never-raising functions that feed the affect estimator's output to the
shadow log: one for a conversation turn (the produced response and, when
given, the received message, with the PAD coordinate as context) and one for
an autonomous thought. Shadow only: nothing here touches a prompt or a
narration, and any failure is logged and swallowed.

Records go to the shadow log under signal id ``affect_estimator_v1`` with
``values`` (valence and arousal), ``confidence``, ``computed_from`` (which
names the anchor version) and a ``context`` mapping. The PAD coordinate is
read from a ``PADState`` or a mapping with ``pleasure``, ``arousal`` and
``dominance``. Annotation parsing is off unless ``annotation_pattern`` is
given: when it is, the pattern's first group on the received text is the
annotation passed to the estimator, unless an ``annotation`` is passed
explicitly.

Config: the shadow log directory (``Settings.SHADOW_LOG_DIR``). The module
holds one lazily created estimator and logger unless they are injected; the
default estimator needs the ``embeddings`` extra and otherwise records
nothing. Logged text-derived context is private data. The pipeline's
post-turn step (``mtc.pipeline.post_turn``, step ``affect_shadow``) calls the
turn tap; the runtime turns that step on when its embedder is a sentence
embedder. Nothing in the package calls the thought tap.
"""

import logging
from typing import Any, Dict, Optional, Pattern

from mtc.validation.shadow_log import ShadowSignalLogger

# Named _log because the taps take a ``logger`` parameter for the shadow log.
_log = logging.getLogger(__name__)

SIGNAL_ID = "affect_estimator_v1"

_estimator: Optional[Any] = None
_default_logger: Optional[ShadowSignalLogger] = None


def get_affect_estimator() -> Any:
    """Process-wide estimator (one embedding model instance)."""
    global _estimator
    if _estimator is None:
        from mtc.consciousness.affect_estimator import AffectEstimator

        _estimator = AffectEstimator()
    return _estimator


def _get_default_logger() -> ShadowSignalLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = ShadowSignalLogger(SIGNAL_ID)
    return _default_logger


def parse_annotation(
    text: Optional[str], pattern: Optional[Pattern[str]]
) -> Optional[str]:
    """First capture group of ``pattern`` in ``text`` (lowercased), if any."""
    if not text or pattern is None:
        return None
    match = pattern.search(text)
    if not match or not match.groups():
        return None
    value = match.group(1)
    return value.lower() if value else None


def _pad_context(pad: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if pad is None:
        return out
    for dim in ("pleasure", "arousal", "dominance"):
        if isinstance(pad, dict):
            value = pad.get(dim)
        else:
            value = getattr(pad, dim, None)
        if value is not None:
            out[f"pad_{dim}"] = value
    return out


def _write(
    estimate: Any, shadow_logger: ShadowSignalLogger, context: Dict[str, Any]
) -> None:
    shadow_logger.log(
        {
            "values": {"valence": estimate.valence, "arousal": estimate.arousal},
            "confidence": estimate.confidence,
            "computed_from": list(estimate.computed_from),
            "context": context,
        }
    )


def log_turn_affect_shadow(
    response_text: str,
    received_text: Optional[str] = None,
    pad: Optional[Any] = None,
    annotation: Optional[str] = None,
    *,
    estimator: Optional[Any] = None,
    logger: Optional[ShadowSignalLogger] = None,
    annotation_pattern: Optional[Pattern[str]] = None,
) -> None:
    """Shadow-log affect estimates for one turn. Never raises."""
    try:
        shadow_logger = logger or _get_default_logger()
        est = estimator or get_affect_estimator()
        base = _pad_context(pad)

        response = est.estimate(response_text or "", source_kind="response")
        if response is not None:
            _write(response, shadow_logger, {**base, "source_kind": "response"})

        if received_text:
            note = annotation or parse_annotation(received_text, annotation_pattern)
            kwargs: Dict[str, Any] = {"source_kind": "received"}
            if note:
                kwargs["annotation"] = note
            received = est.estimate(received_text, **kwargs)
            if received is not None:
                ctx = {**base, "source_kind": "received"}
                if note:
                    ctx["annotation"] = note
                _write(received, shadow_logger, ctx)
    except Exception as exc:  # a shadow tap never breaks the turn
        _log.warning("turn affect shadow failed: %s", exc)


def log_thought_affect_shadow(
    thought_text: str,
    thought_id: Optional[str] = None,
    thought_type: Optional[str] = None,
    *,
    estimator: Optional[Any] = None,
    logger: Optional[ShadowSignalLogger] = None,
) -> None:
    """Shadow-log the affect estimate for one autonomous thought. Never raises."""
    try:
        shadow_logger = logger or _get_default_logger()
        est = estimator or get_affect_estimator()
        result = est.estimate(thought_text or "", source_kind="thought")
        if result is None:
            return
        _write(
            result,
            shadow_logger,
            {
                "source_kind": "thought",
                "thought_id": thought_id,
                "thought_type": thought_type,
            },
        )
    except Exception as exc:
        _log.warning("thought affect shadow failed: %s", exc)
