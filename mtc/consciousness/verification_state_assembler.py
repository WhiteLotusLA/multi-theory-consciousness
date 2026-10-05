"""
Verification State Assembler
============================

Builds the consciousness-state dict that
``ConsciousnessVerifier.run_verification_suite`` reads, from the substrate
blocks of ``ConsciousnessIntegration.collect_neural_data()`` (or any dict of
blocks in that shape), the PAD affect and the conversation history.

A block reaches the state only when it carries ``is_real`` true; otherwise its
key holds an empty dict. The LSM's intuitive valence and arousal are passed
through as ``lsm_data_affect`` only when the LSM block is real. Longitudinal
inputs the caller does not supply (temporal samples, personality traits, core
beliefs) are omitted, never filled in, so the tests that need them return
``insufficient_data``. The assembler holds no state and reads no config keys;
the timestamp is timezone-aware UTC.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_PAD_AXES = ("pleasure", "arousal", "dominance")


def _real(block: Any) -> Dict[str, Any]:
    """A copy of the block when it carries ``is_real`` true, else {}."""
    if isinstance(block, dict) and block.get("is_real") is True:
        return dict(block)
    return {}


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pad_dict(pad_state: Any) -> Optional[Dict[str, Optional[float]]]:
    """PAD axes from a ``PADState`` or a mapping; None when none is present."""
    if pad_state is None:
        return None
    values: Dict[str, Optional[float]] = {}
    for axis in _PAD_AXES:
        if isinstance(pad_state, dict):
            raw = pad_state.get(axis)
        else:
            raw = getattr(pad_state, axis, None)
        values[axis] = _finite(raw)
    if all(v is None for v in values.values()):
        return None
    return values


class ConsciousnessStateAssembler:
    """Assembles the verification state from an integration or neural data.

    Args:
        integration: A ``ConsciousnessIntegration`` (or anything with
            ``collect_neural_data()`` and optionally ``get_temporal_data()``);
            None assembles a state with no real block.
        clock: Returns the current aware UTC time.
    """

    def __init__(
        self,
        integration: Optional[Any] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._integration = integration
        self._clock = clock

    def assemble(
        self,
        *,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        pad_state: Any = None,
    ) -> Dict[str, Any]:
        """The state for this assembler's integration."""
        now = self._clock() if self._clock is not None else None
        return self.from_integration(
            self._integration,
            pad_state=pad_state,
            conversation_history=conversation_history,
            now=now,
        )

    @staticmethod
    def from_neural_data(
        neural_data: Optional[Dict[str, Any]],
        pad_state: Any = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        temporal_data: Optional[Dict[str, List[Any]]] = None,
        *,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """The verification state for one set of substrate blocks.

        Args:
            neural_data: Dict with ``snn``, ``lsm`` and ``htm`` blocks.
            pad_state: A ``PADState`` or a mapping of the PAD axes.
            conversation_history: Turns with ``content`` (and any role).
            temporal_data: ``ConsciousnessIntegration.get_temporal_data()``
                output; omitted from the state when None.
            now: The assembly time; defaults to the current UTC time.
        """
        neural = neural_data if isinstance(neural_data, dict) else {}
        lsm = _real(neural.get("lsm"))
        lsm_affect = (
            {
                "intuitive_valence": lsm.get("intuitive_valence"),
                "intuitive_arousal": lsm.get("intuitive_arousal"),
            }
            if lsm
            else None
        )
        moment = now or datetime.now(timezone.utc)
        state: Dict[str, Any] = {
            "timestamp": moment.isoformat(),
            "snn_data": _real(neural.get("snn")),
            "lsm_data": lsm,
            "htm_data": _real(neural.get("htm")),
            "lsm_data_affect": lsm_affect,
            "pad_affect": _pad_dict(pad_state),
            "conversation_history": list(conversation_history or []),
        }
        if temporal_data is not None:
            state["temporal_data"] = temporal_data
        return state

    @classmethod
    def from_integration(
        cls,
        integration: Optional[Any],
        pad_state: Any = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        *,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """The verification state from an integration's collection and samples.

        A collection that fails leaves every block empty; the temporal
        samples are included when the integration offers them.
        """
        neural: Dict[str, Any] = {}
        temporal: Optional[Dict[str, List[Any]]] = None
        if integration is not None:
            try:
                collected = integration.collect_neural_data()
                neural = collected if isinstance(collected, dict) else {}
            except Exception as e:
                logger.warning(f"Neural data collection failed: {e}")
            get_temporal = getattr(integration, "get_temporal_data", None)
            if callable(get_temporal):
                try:
                    temporal = get_temporal()
                except Exception as e:
                    logger.warning(f"Temporal samples unavailable: {e}")
        return cls.from_neural_data(
            neural,
            pad_state=pad_state,
            conversation_history=conversation_history,
            temporal_data=temporal,
            now=now,
        )
