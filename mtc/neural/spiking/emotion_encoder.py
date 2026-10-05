"""
Emotion Encoder
===============

Encodes an emotional state into spike patterns for a spiking network by rate
coding: each of the ten emotion channels drives one input neuron with a current
proportional to its intensity, modulated by arousal. The decoded output is the
output layer's firing rate per channel.

Two encoders share the channel list, the state record and the decoding:

- ``EmotionEncoder`` records spike events (bounded) and, when spike storage is
  on, writes up to ``max_stored_spikes`` of them as ``NeuralSpike`` documents.
- ``OptimizedEmotionEncoder`` counts spikes per layer and, when spike storage
  is on, writes one summary document per encoding. It also offers
  ``process_emotion(emotions, context=None)``, which derives valence and arousal
  from the emotions (both encoders inherit it).

The network is any object with ``simulate_step``, ``get_firing_rates``, ``dt``
and ``current_time``; the default is the vectorized
``OptimizedSpikingNeuralNetwork``. Neuron ids stored with spikes are derived
from the layer sizes the network reports (hidden neurons start after the input
layer, output neurons after the hidden layer), so they never collide. The
membrane potential stored with a spike is read from the network after the step
in which the neuron fired (so after its reset) when the network exposes it, and
is None otherwise; it is never an approximation. Noise comes from a local
generator (``seed`` makes it reproducible); the global random state is never
touched.

Config keys: none read here; spike storage is off unless ``store_spikes`` is
passed, and asking for it without a data manager raises ValueError. Persisted
state: spike documents in the ``neural_spikes`` collection through the data
manager (see ``mtc.neural.mongodb_schemas``, which also holds the schema
artifact). ``status()`` reports the network, the channel count and the last
storage error.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mtc.neural.mongodb_schemas import NeuralDataManager, NeuralSpike, to_utc
from mtc.neural.spiking.snn_core_optimized import OptimizedSpikingNeuralNetwork

logger = logging.getLogger(__name__)

EMOTION_CHANNELS: Tuple[str, ...] = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "love",
    "trust",
    "anticipation",
    "curiosity",
)

# Documented defaults for deriving valence in ``process_emotion``; override
# them per encoder.
DEFAULT_POSITIVE_EMOTIONS: Tuple[str, ...] = ("joy", "love", "trust")
DEFAULT_NEGATIVE_EMOTIONS: Tuple[str, ...] = ("sadness", "anger", "fear")

# Drive per unit of intensity, in the network's current units, and the noise
# added to every input current at every step.
INPUT_CURRENT_SCALE = 50.0
INPUT_NOISE_STD = 2.0
# Output rate (Hz) that decodes to an intensity of 1.0.
OUTPUT_RATE_SCALE_HZ = 100.0

_LAYERS = ("input", "hidden", "output")


@dataclass
class EmotionState:
    """An emotional state: intensities per channel plus the overall dimensions.

    ``dominance`` is None when no source supplies it.
    """

    timestamp: datetime
    emotions: Dict[str, float]
    valence: float
    arousal: float
    dominance: Optional[float] = None


class _SpikeEncoder:
    """Shared construction, input mapping, decoding and status."""

    EMOTION_CHANNELS: Tuple[str, ...] = EMOTION_CHANNELS
    _encoder_name = "emotion_encoder"

    def __init__(
        self,
        snn: Optional[Any] = None,
        neural_data_manager: Optional[NeuralDataManager] = None,
        store_spikes: bool = False,
        hidden_neurons: int = 50,
        positive_emotions: Sequence[str] = DEFAULT_POSITIVE_EMOTIONS,
        negative_emotions: Sequence[str] = DEFAULT_NEGATIVE_EMOTIONS,
        seed: Optional[int] = None,
    ) -> None:
        if store_spikes and neural_data_manager is None:
            raise ValueError("store_spikes=True needs a neural_data_manager")
        channels = len(self.EMOTION_CHANNELS)
        self.snn = snn or OptimizedSpikingNeuralNetwork(
            n_input=channels,
            n_hidden=hidden_neurons,
            n_output=channels,
            dt=1.0,
            learning_enabled=False,
        )
        self.neural_data_manager = neural_data_manager
        self.store_spikes = bool(store_spikes)
        self.positive_emotions = tuple(positive_emotions)
        self.negative_emotions = tuple(negative_emotions)
        self.emotion_to_neuron = {
            emotion: i for i, emotion in enumerate(self.EMOTION_CHANNELS)
        }
        self._rng = np.random.default_rng(seed)
        self.total_spikes_encoded = 0
        self._last_error: Optional[str] = None
        logger.info("%s initialized with %d channels", type(self).__name__, channels)

    # ---- input and output mapping ------------------------------------

    def _input_intensities(self, emotion_state: EmotionState) -> np.ndarray:
        intensities = np.zeros(len(self.EMOTION_CHANNELS))
        for emotion, intensity in emotion_state.emotions.items():
            index = self.emotion_to_neuron.get(emotion)
            if index is not None:
                intensities[index] = float(intensity)
        arousal_factor = 0.5 + 0.5 * float(emotion_state.arousal)
        return intensities * arousal_factor

    def _decode_output(self, firing_rates: Dict[str, np.ndarray]) -> Dict[str, float]:
        rates = firing_rates["output"]
        return {
            emotion: float(rates[i] / OUTPUT_RATE_SCALE_HZ)
            for i, emotion in enumerate(self.EMOTION_CHANNELS)
        }

    def _begin(self) -> None:
        reset = getattr(self.snn, "reset_spike_counts", None)
        if callable(reset):
            reset()

    def _stamp(self, emotion_state: EmotionState) -> datetime:
        if emotion_state.timestamp is None:
            return datetime.now(timezone.utc)
        return to_utc(emotion_state.timestamp)

    def _membrane(self, layer: str, index: int) -> Optional[float]:
        potentials = getattr(self.snn, f"V_{layer}", None)
        if potentials is None:
            return None
        try:
            return float(potentials[index])
        except (IndexError, TypeError, ValueError):
            return None

    # ---- convenience API ----------------------------------------------

    def process_emotion(
        self, emotions: Dict[str, float], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Encode a plain emotion dictionary and return the encoding result.

        Valence is the mean of the positive emotions minus the mean of the
        negative ones (``positive_emotions`` and ``negative_emotions``);
        arousal is the mean of the intensities given, and 0.0 when none are
        given. ``context`` is accepted for API compatibility and is neither
        logged nor stored.
        """
        positive = [emotions.get(e, 0.0) for e in self.positive_emotions]
        negative = [emotions.get(e, 0.0) for e in self.negative_emotions]
        valence = (sum(positive) / len(positive) if positive else 0.0) - (
            sum(negative) / len(negative) if negative else 0.0
        )
        arousal = sum(emotions.values()) / len(emotions) if emotions else 0.0
        state = EmotionState(
            timestamp=datetime.now(timezone.utc),
            emotions=dict(emotions),
            valence=valence,
            arousal=arousal,
        )
        return self.encode_emotion(state, duration_ms=100.0)

    def encode_emotion(
        self, emotion_state: EmotionState, duration_ms: float = 100.0
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # ---- storage ------------------------------------------------------

    def _store(self, spikes: List[NeuralSpike]) -> None:
        if not (self.store_spikes and self.neural_data_manager and spikes):
            return
        try:
            self.neural_data_manager.store_spike_batch(spikes)
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("spike storage failed: %s", type(exc).__name__)

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` plus the network
        class, the channel count and the storage state."""
        storage = None
        if self.neural_data_manager is not None:
            storage = self.neural_data_manager.status()
        return {
            "backend": "numpy",
            "available": True,
            "last_error": self._last_error,
            "network": type(self.snn).__name__,
            "channels": len(self.EMOTION_CHANNELS),
            "spike_storage": self.store_spikes,
            "storage": storage,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return encoder counters and the network's own metrics."""
        get = getattr(self.snn, "get_metrics", None)
        return {
            "total_spikes_encoded": self.total_spikes_encoded,
            "snn_metrics": get() if callable(get) else None,
            "emotions_tracked": list(self.EMOTION_CHANNELS),
            "spike_storage_enabled": self.store_spikes,
        }


class EmotionEncoder(_SpikeEncoder):
    """Encoder that records individual spike events.

    ``max_stored_spikes`` bounds how many events of one encoding are written
    when storage is on (the first ones in time order); ``history_limit`` bounds
    the in-memory event history behind ``get_emotion_pattern_similarity``.
    """

    _encoder_name = "emotion_encoder"

    def __init__(
        self,
        snn: Optional[Any] = None,
        neural_data_manager: Optional[NeuralDataManager] = None,
        store_spikes: bool = False,
        max_stored_spikes: int = 100,
        history_limit: int = 10000,
        **kwargs: Any,
    ) -> None:
        super().__init__(snn, neural_data_manager, store_spikes, **kwargs)
        self.max_stored_spikes = int(max_stored_spikes)
        self.spike_history: Deque[Tuple[datetime, str, int, float]] = deque(
            maxlen=int(history_limit)
        )

    def encode_emotion(
        self, emotion_state: EmotionState, duration_ms: float = 100.0
    ) -> Dict[str, Any]:
        """Encode a state, simulate the network and return the result.

        The result holds ``input_emotions``, ``output_emotions`` (firing rates
        scaled by ``OUTPUT_RATE_SCALE_HZ``), ``total_spikes``,
        ``spikes_by_layer``, ``firing_rates`` per layer, ``processing_time_ms``
        and ``network_time_ms``.
        """
        started = datetime.now(timezone.utc)
        intensities = self._input_intensities(emotion_state)
        stamp = self._stamp(emotion_state)
        self._begin()
        steps = int(duration_ms / self.snn.dt)
        events: List[NeuralSpike] = []
        counts = {layer: 0 for layer in _LAYERS}

        for _ in range(steps):
            currents = intensities * INPUT_CURRENT_SCALE
            currents = currents + self._rng.normal(0.0, INPUT_NOISE_STD, len(currents))
            spikes = self.snn.simulate_step(currents)
            sizes = {layer: len(spikes[layer]) for layer in _LAYERS}
            offsets = {
                "input": 0,
                "hidden": sizes["input"],
                "output": sizes["input"] + sizes["hidden"],
            }
            for layer in _LAYERS:
                for index in np.flatnonzero(spikes[layer]):
                    index = int(index)
                    counts[layer] += 1
                    if layer == "hidden":
                        emotion_type, intensity = "processing", 1.0
                    else:
                        emotion_type = self.EMOTION_CHANNELS[index]
                        intensity = (
                            float(intensities[index]) if layer == "input" else 1.0
                        )
                    if layer == "input":
                        self.spike_history.append(
                            (stamp, emotion_type, index, intensity)
                        )
                    if self.store_spikes and len(events) < self.max_stored_spikes:
                        events.append(
                            NeuralSpike(
                                timestamp=stamp,
                                neuron_id=offsets[layer] + index,
                                spike_time=float(self.snn.current_time),
                                emotion_type=emotion_type,
                                intensity=intensity,
                                membrane_potential=self._membrane(layer, index),
                                metadata={
                                    "layer": layer,
                                    "encoder": self._encoder_name,
                                },
                            )
                        )

        firing_rates = self.snn.get_firing_rates(window=duration_ms)
        total = sum(counts.values())
        self.total_spikes_encoded += total
        self._store(events)
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
        return {
            "input_emotions": emotion_state.emotions,
            "output_emotions": self._decode_output(firing_rates),
            "total_spikes": total,
            "spikes_by_layer": dict(counts),
            "firing_rates": {layer: firing_rates[layer].tolist() for layer in _LAYERS},
            "processing_time_ms": elapsed_ms,
            "network_time_ms": float(self.snn.current_time),
        }

    def get_emotion_pattern_similarity(
        self, emotion1: str, emotion2: str, window_ms: float = 100.0
    ) -> float:
        """Similarity (0 to 1) of two emotions' recent input spike counts.

        The window ends at the newest recorded event, not at the wall clock,
        so the answer depends only on what was encoded. 0.0 when either
        emotion has no spikes in the window.
        """
        if not self.spike_history:
            return 0.0
        newest = max(event[0] for event in self.spike_history)
        floor = newest.timestamp() - window_ms / 1000.0
        first = sum(
            1
            for e in self.spike_history
            if e[1] == emotion1 and e[0].timestamp() >= floor
        )
        second = sum(
            1
            for e in self.spike_history
            if e[1] == emotion2 and e[0].timestamp() >= floor
        )
        if first == 0 or second == 0:
            return 0.0
        return 1.0 - abs(first - second) / max(first, second)

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["spike_events_recorded"] = len(self.spike_history)
        return metrics


class OptimizedEmotionEncoder(_SpikeEncoder):
    """Encoder that counts spikes per layer and stores one summary per encoding."""

    _encoder_name = "optimized_emotion_encoder"

    def encode_emotion(
        self, emotion_state: EmotionState, duration_ms: float = 100.0
    ) -> Dict[str, Any]:
        """Encode a state and return the same result shape as ``EmotionEncoder``.

        The per-layer counts and rates are read from the network's spike
        counters, which are cleared at the start of each encoding.
        """
        started = datetime.now(timezone.utc)
        intensities = self._input_intensities(emotion_state)
        stamp = self._stamp(emotion_state)
        self._begin()
        steps = int(duration_ms / self.snn.dt)
        noise = self._rng.normal(0.0, INPUT_NOISE_STD, (steps, len(intensities)))
        base = intensities * INPUT_CURRENT_SCALE
        for step in range(steps):
            self.snn.simulate_step(base + noise[step])

        firing_rates = self.snn.get_firing_rates(window=duration_ms)
        by_layer = {
            layer: int(np.sum(getattr(self.snn, f"spike_counts_{layer}")))
            for layer in _LAYERS
        }
        total = sum(by_layer.values())
        self.total_spikes_encoded += total
        if self.store_spikes:
            self._store([self._summary(emotion_state, stamp, total, firing_rates)])
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
        return {
            "input_emotions": emotion_state.emotions,
            "output_emotions": self._decode_output(firing_rates),
            "total_spikes": total,
            "spikes_by_layer": by_layer,
            "firing_rates": {layer: firing_rates[layer].tolist() for layer in _LAYERS},
            "processing_time_ms": elapsed_ms,
            "network_time_ms": float(self.snn.current_time),
        }

    def _summary(
        self,
        emotion_state: EmotionState,
        stamp: datetime,
        total: int,
        firing_rates: Dict[str, np.ndarray],
    ) -> NeuralSpike:
        values = [float(v) for v in emotion_state.emotions.values()]
        return NeuralSpike(
            timestamp=stamp,
            neuron_id=None,
            spike_time=float(self.snn.current_time),
            emotion_type=",".join(sorted(emotion_state.emotions)),
            intensity=float(np.mean(values)) if values else 0.0,
            membrane_potential=None,
            metadata={
                "record": "summary",
                "total_spikes": total,
                "avg_input_rate": float(np.mean(firing_rates["input"])),
                "avg_hidden_rate": float(np.mean(firing_rates["hidden"])),
                "avg_output_rate": float(np.mean(firing_rates["output"])),
                "encoder": self._encoder_name,
            },
        )
