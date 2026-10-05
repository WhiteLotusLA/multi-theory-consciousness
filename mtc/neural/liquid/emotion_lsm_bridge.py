"""
Emotion to Liquid State Bridge
==============================

Connects the conscious emotion encoding (a spiking network) to the subconscious
reservoir (a liquid state machine). An emotional state is encoded as spikes, the
output layer's activity drives the reservoir, and the reservoir's dynamics are
summarized as a ``SubconsciousState``.

Two paths, chosen at construction and reported by ``status()``:

- ``production``: ``ProductionSNN`` and ``ProductionLSM`` (the ``neural`` extra:
  torch, snntorch and reservoirpy). The SNN runs on the device from
  ``EmotionLSMBridgeConfig.device`` or, when that is None, from ``MTC_DEVICE``
  through ``mtc.core.device``.
- ``numpy``: the vectorized ``OptimizedEmotionEncoder`` and the pure NumPy
  ``LiquidStateMachine``, used when the extra is missing or the config says
  ``production=False``. ``production=True`` raises an ImportError naming the
  extra instead of falling back.

An untrained readout reports nothing. The intuitive valence, intuitive arousal
and emotional coherence are the first three outputs of a trained readout (the
readout was trained on targets whose first three columns are valence, arousal
and coherence, see ``READOUT_CHANNELS``); until a readout is trained (the
production path: the RLS readout has seen samples; the NumPy path: ``train_readout``
ran) all three are None and ``SubconsciousState.readout_trained`` is False.
Reservoir energy, entropy and sparsity, the pattern class and the trajectory are
measured from the reservoir and always reported; a quantity that cannot be
computed from what was observed (too few states for a pattern class, a silent
reservoir for an entropy) is None, never a stand-in constant. The orchestrator
therefore reads the LSM as not real until a readout is trained.

Trajectory: the production reservoir returns only the state at the end of each
call, so the input is split into ``lsm_chunks`` consecutive chunks (the
reservoir keeps its state between calls) and the trajectory is the mean
absolute activation after each chunk. The NumPy path records every step.

Units: ``ProductionSNN`` reports the fraction of timesteps each output neuron
spiked, which is used as the reservoir drive directly (0 to 1); the NumPy
encoder reports Hz, which is scaled by ``OUTPUT_RATE_SCALE_HZ``. The reservoir
input is dithered with Gaussian noise of ``input_noise_std`` (0.0 turns it off)
drawn from a local generator seeded by ``seed``; the global random state is
never touched.

Config keys: none read here (sizes come from ``EmotionLSMBridgeConfig``);
``MTC_DEVICE`` is read through the device switch. Persisted state:
``to_state_dict`` and ``from_state_dict`` carry the bounded state history (the
reservoir's own readout persists through ``ProductionLSM``); liquid states are
written to MongoDB through the data manager only when ``store_states`` is on
(the collection and its schema artifact belong to ``mtc.neural.mongodb_schemas``).
The sizes below are defaults chosen for latency on one Apple Silicon machine.
"""

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mtc.neural.mongodb_schemas import LiquidState, NeuralDataManager, to_utc
from mtc.neural.spiking.emotion_encoder import (
    EMOTION_CHANNELS,
    OUTPUT_RATE_SCALE_HZ,
    EmotionState,
    OptimizedEmotionEncoder,
)
from mtc.neural.spiking.snn_core_optimized import OptimizedSpikingNeuralNetwork

logger = logging.getLogger(__name__)

# The readout outputs read as valence, arousal and coherence, in this order.
READOUT_CHANNELS: Tuple[str, str, str] = ("valence", "arousal", "coherence")

__all__ = [
    "EMOTION_CHANNELS",
    "READOUT_CHANNELS",
    "EmotionLSMBridge",
    "EmotionLSMBridgeConfig",
    "EmotionState",
    "SubconsciousState",
]


@dataclass
class EmotionLSMBridgeConfig:
    """Sizes, thresholds and switches of the bridge.

    ``production`` is None for automatic (the production path when its
    libraries import), True to require it, False to use the NumPy path.
    """

    # Production SNN
    snn_hidden_layers: Tuple[int, ...] = (2048, 2048)
    snn_timesteps: int = 10
    device: Optional[str] = None
    # Production reservoir
    reservoir_size: int = 5000
    lsm_input_dim: int = 100
    lsm_output_dim: int = 50
    spectral_radius: float = 0.9
    connectivity: float = 0.1
    reservoir_seed: Optional[int] = 42
    lsm_chunks: int = 10
    # NumPy path
    numpy_hidden_neurons: int = 50
    numpy_reservoir_size: int = 1000
    # Pattern classification
    chaotic_energy_std: float = 10.0
    stable_energy_std: float = 2.0
    rhythmic_sign_changes: int = 10
    min_states_for_pattern: int = 10
    pattern_window: int = 20
    trajectory_length: int = 20
    # Input handling and history
    input_noise_std: float = 0.05
    history_limit: int = 100
    summary_window: int = 10
    seed: Optional[int] = None
    production: Optional[bool] = None


@dataclass
class SubconsciousState:
    """The subconscious state summarized from the reservoir.

    ``intuitive_valence``, ``intuitive_arousal`` and ``emotional_coherence`` are
    None unless ``readout_trained``. ``reservoir_entropy`` (normalized Shannon
    entropy of the activation magnitudes), ``pattern_strength`` and
    ``prediction_confidence`` are None when they cannot be computed from what
    was observed. ``path`` is ``production`` or ``numpy``.
    """

    timestamp: datetime
    reservoir_energy: float
    reservoir_entropy: Optional[float]
    reservoir_sparsity: float
    intuitive_valence: Optional[float]
    intuitive_arousal: Optional[float]
    emotional_coherence: Optional[float]
    dominant_pattern: str
    pattern_strength: Optional[float]
    state_trajectory: List[float]
    prediction_confidence: Optional[float]
    metadata: Dict[str, Any]
    readout_trained: bool = False
    path: str = "numpy"


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fit_vector(vector: np.ndarray, size: int) -> np.ndarray:
    flat = np.ravel(np.asarray(vector, dtype=float))
    if flat.size < size:
        return np.pad(flat, (0, size - flat.size))
    return flat[:size]


def _normalized_entropy(state: np.ndarray) -> Optional[float]:
    """Shannon entropy of the activation magnitudes over their maximum.

    None for a silent reservoir (nothing to distribute) or a single unit.
    """
    magnitudes = np.abs(np.ravel(np.asarray(state, dtype=float)))
    total = float(magnitudes.sum())
    if magnitudes.size < 2 or total <= 0.0 or not math.isfinite(total):
        return None
    p = magnitudes / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum() / math.log(magnitudes.size))


class EmotionLSMBridge:
    """Bridge from emotion encoding to reservoir dynamics.

    Args:
        neural_data_manager: Data manager that stores liquid states.
        store_states: Write each liquid state through the data manager;
            True without a manager raises ValueError.
        config: Sizes, thresholds and switches; defaults to
            ``EmotionLSMBridgeConfig()``.

    ``build_on_first_use`` tells the orchestrator to defer building a default
    bridge until its first experience (the production substrates are large).
    """

    build_on_first_use = True

    def __init__(
        self,
        neural_data_manager: Optional[NeuralDataManager] = None,
        store_states: bool = False,
        config: Optional[EmotionLSMBridgeConfig] = None,
    ) -> None:
        if store_states and neural_data_manager is None:
            raise ValueError("store_states=True needs a neural_data_manager")
        self.config = config or EmotionLSMBridgeConfig()
        self.neural_data_manager = neural_data_manager
        self.store_states = bool(store_states)
        self.emotion_channels = list(EMOTION_CHANNELS)
        self.emotion_channel_map = {e: i for i, e in enumerate(self.emotion_channels)}
        self._rng = np.random.default_rng(self.config.seed)
        self.state_history: List[SubconsciousState] = []
        self.reservoir_trajectory: List[np.ndarray] = []
        self.states_processed = 0
        self._last_error: Optional[str] = None
        self._last_snn_metrics: Dict[str, Any] = {}
        self._numpy_readout_trained = False
        self.emotion_encoder: Optional[OptimizedEmotionEncoder] = None
        self.snn: Any = None
        self.lsm: Any = None
        self._fallback_reason: Optional[str] = None
        self.path = self._build_substrates()

    # ---- construction -----------------------------------------------

    def _build_substrates(self) -> str:
        wanted = self.config.production
        if wanted is not False:
            try:
                self._build_production()
                return "production"
            except ImportError as exc:
                if wanted:
                    raise ImportError(
                        "the production emotion bridge needs the 'neural' extra "
                        f"(torch, snntorch, reservoirpy): {exc}"
                    ) from exc
                self._fallback_reason = f"neural extra not importable: {exc}"
                logger.warning(
                    "emotion bridge: using the NumPy path (%s)", self._fallback_reason
                )
        else:
            self._fallback_reason = "production=False in the config"
        self._build_numpy()
        return "numpy"

    def _build_production(self) -> None:
        import torch  # noqa: F401  (the import is the availability check)
        import reservoirpy  # noqa: F401
        import snntorch  # noqa: F401

        from mtc.neural.liquid.production_lsm import LSMConfig, ProductionLSM
        from mtc.neural.spiking.production_snn import ProductionSNN, SNNConfig

        cfg = self.config
        snn_config = SNNConfig(
            input_dim=len(self.emotion_channels),
            hidden_layers=list(cfg.snn_hidden_layers),
            output_dim=len(self.emotion_channels),
            timesteps=cfg.snn_timesteps,
            use_gpu=True,
            device=cfg.device,
            stdp_enabled=False,
        )
        lsm_config = LSMConfig(
            reservoir_size=cfg.reservoir_size,
            input_dim=cfg.lsm_input_dim,
            output_dim=cfg.lsm_output_dim,
            spectral_radius=cfg.spectral_radius,
            connectivity=cfg.connectivity,
            seed=cfg.reservoir_seed,
        )
        self.snn = ProductionSNN(snn_config)
        self.lsm = ProductionLSM(config=lsm_config, fast_mode=True)
        logger.info(
            "emotion bridge on the production path: %d SNN neurons on %s, "
            "%d reservoir units",
            self.snn._count_neurons(),
            self.snn.device,
            cfg.reservoir_size,
        )

    def _build_numpy(self) -> None:
        from mtc.neural.liquid.lsm_core import LiquidStateMachine

        cfg = self.config
        channels = len(self.emotion_channels)
        self.snn = OptimizedSpikingNeuralNetwork(
            n_input=channels,
            n_hidden=cfg.numpy_hidden_neurons,
            n_output=channels,
            dt=1.0,
            learning_enabled=False,
        )
        self.emotion_encoder = OptimizedEmotionEncoder(
            snn=self.snn, store_spikes=False, seed=cfg.seed
        )
        self.lsm = LiquidStateMachine(
            n_input=channels,
            n_reservoir=cfg.numpy_reservoir_size,
            n_output=cfg.lsm_output_dim,
            dt=1.0,
            spectral_radius=cfg.spectral_radius,
            connectivity=cfg.connectivity,
        )

    # ---- readout ------------------------------------------------------

    def readout_trained(self) -> bool:
        """True once the readout has been trained."""
        if self.path == "numpy":
            return self._numpy_readout_trained
        try:
            scalars = self.lsm.to_state_dict().get("scalars") or {}
            return bool(scalars.get("rls_initialized")) and (
                float(scalars.get("rls_samples", 0.0)) > 0.0
            )
        except Exception as exc:
            logger.debug("readout state unavailable: %s", exc)
            return False

    def train_readout(self, input_sequences: Any, target_sequences: Any) -> Any:
        """Train the readout; the first three target columns must be valence,
        arousal and coherence (``READOUT_CHANNELS``).

        Production path: ``ProductionLSM.train_rls`` on arrays shaped
        (samples, timesteps, dim); returns its loss. NumPy path:
        ``LiquidStateMachine.train_readout`` on lists of sequences; returns None.
        """
        if self.path == "numpy":
            self.lsm.train_readout(list(input_sequences), list(target_sequences))
            self._numpy_readout_trained = True
            return None
        return self.lsm.train_rls(input_sequences, target_sequences)

    # ---- processing -----------------------------------------------------

    def _emotion_vector(self, emotion_state: EmotionState) -> np.ndarray:
        vector = np.zeros(len(self.emotion_channels))
        for emotion, intensity in emotion_state.emotions.items():
            index = self.emotion_channel_map.get(emotion)
            if index is not None:
                vector[index] = float(intensity)
        return vector * (0.5 + 0.5 * float(emotion_state.arousal))

    def _run_snn(
        self, emotion_state: EmotionState, snn_duration_ms: float
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Return the SNN result dict and the output activity (0 to 1 scale)."""
        if self.path == "production":
            import torch

            vector = torch.tensor(
                self._emotion_vector(emotion_state), dtype=torch.float32
            )
            output, metrics = self.snn.process_emotion(vector)
            activity = np.asarray(output.detach().cpu().numpy(), dtype=float)
            activity = np.squeeze(activity)
            if activity.ndim > 1:
                activity = activity.mean(axis=tuple(range(activity.ndim - 1)))
            activity = np.atleast_1d(activity)
            total = metrics.get("total_spikes", 0)
            snn_result = {
                "firing_rates": {"output": activity.tolist()},
                "output_unit": "spike_fraction",
                "metrics": metrics,
                "spike_counts": metrics.get("spike_counts", []),
                "total_spikes": int(total) if isinstance(total, (int, float)) else 0,
                "processing_time_ms": metrics.get("processing_time_ms"),
            }
            self._last_snn_metrics = {
                "total_spikes": snn_result["total_spikes"],
                "mean_firing_rate": metrics.get("mean_firing_rate"),
            }
            return snn_result, np.clip(activity, 0.0, 1.0)

        result = self.emotion_encoder.encode_emotion(
            emotion_state, duration_ms=snn_duration_ms
        )
        rates = np.asarray(result["firing_rates"]["output"], dtype=float)
        result = dict(result)
        result["output_unit"] = "hz"
        self._last_snn_metrics = {
            "total_spikes": result["total_spikes"],
            "mean_firing_rate": None,
        }
        return result, np.clip(rates / OUTPUT_RATE_SCALE_HZ, 0.0, 1.0)

    def _drive(self, activity: np.ndarray, width: int, steps: int) -> np.ndarray:
        base = _fit_vector(activity, width)
        noise = self._rng.normal(0.0, self.config.input_noise_std, (steps, width))
        return np.clip(base + noise, 0.0, 1.0)

    def _run_lsm(
        self, activity: np.ndarray, lsm_duration_ms: float
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], int]:
        """Return the final reservoir state, the final readout, the states used
        for the trajectory and the number of steps run."""
        steps = max(int(lsm_duration_ms), 1)
        if self.path == "production":
            drive = self._drive(activity, self.config.lsm_input_dim, steps)
            chunks = [
                c for c in np.array_split(drive, self.config.lsm_chunks) if len(c)
            ]
            states: List[np.ndarray] = []
            readout = np.zeros(self.config.lsm_output_dim)
            for chunk in chunks:
                output, state = self.lsm.process_input(chunk)
                final = state.get("final_reservoir_state")
                if final is None and state.get("reservoir_states"):
                    final = state["reservoir_states"][-1]
                states.append(np.asarray(final, dtype=float))
                readout = np.asarray(output[-1], dtype=float)
            return states[-1], readout, states, steps
        dt = float(self.lsm.dt)
        steps = max(int(lsm_duration_ms / dt), 1)
        drive = self._drive(activity, len(self.emotion_channels), steps)
        states = []
        readout = np.zeros(self.config.lsm_output_dim)
        for row in drive:
            reservoir_state, readout = self.lsm.step(row)
            states.append(np.array(reservoir_state, dtype=float))
        return states[-1], np.asarray(readout, dtype=float), states, steps

    def process_emotion(
        self,
        emotion_state: EmotionState,
        snn_duration_ms: float = 50.0,
        lsm_duration_ms: float = 100.0,
    ) -> Tuple[Dict[str, Any], SubconsciousState]:
        """Process one emotional state through the SNN and the reservoir.

        Returns ``(snn_result, subconscious_state)``. ``snn_result`` carries
        ``firing_rates["output"]``, ``output_unit``, ``total_spikes`` and
        ``processing_time_ms`` (and ``metrics`` on the production path).
        """
        started = time.perf_counter()
        snn_result, activity = self._run_snn(emotion_state, snn_duration_ms)
        final_state, readout, states, steps = self._run_lsm(activity, lsm_duration_ms)
        cfg = self.config

        energies = [float(np.sum(s**2)) for s in states[-cfg.pattern_window :]]
        pattern, strength = self._classify_pattern(energies)
        trajectory = [
            float(np.mean(np.abs(s))) for s in states[-cfg.trajectory_length :]
        ]
        trained = self.readout_trained()
        valence = arousal = coherence = None
        if trained:
            valence, arousal, coherence = (
                _finite_or_none(readout[i]) if i < len(readout) else None
                for i in range(len(READOUT_CHANNELS))
            )

        stamp = emotion_state.timestamp or datetime.now()
        processing_ms = (time.perf_counter() - started) * 1000.0
        subconscious = SubconsciousState(
            timestamp=stamp,
            reservoir_energy=float(np.sum(final_state**2)),
            reservoir_entropy=_normalized_entropy(final_state),
            reservoir_sparsity=float(np.mean(np.abs(final_state) < 0.1)),
            intuitive_valence=valence,
            intuitive_arousal=arousal,
            emotional_coherence=coherence,
            dominant_pattern=pattern,
            pattern_strength=strength,
            state_trajectory=trajectory,
            prediction_confidence=self._prediction_confidence(trajectory),
            metadata={
                "conscious_emotions": list(emotion_state.emotions.keys()),
                "snn_total_spikes": snn_result.get("total_spikes"),
                "snn_processing_time_ms": snn_result.get("processing_time_ms"),
                "snn_mean_firing_rate": self._last_snn_metrics.get("mean_firing_rate"),
                "lsm_steps": steps,
                "processing_time_ms": processing_ms,
            },
            readout_trained=trained,
            path=self.path,
        )
        self.state_history.append(subconscious)
        del self.state_history[: -cfg.history_limit]
        self.reservoir_trajectory.append(final_state)
        del self.reservoir_trajectory[: -cfg.history_limit]
        self.states_processed += 1
        if self.store_states:
            self._store_liquid_state(emotion_state, subconscious, final_state, readout)
        logger.info(
            "subconscious processing complete: path=%s pattern=%s readout_trained=%s",
            self.path,
            pattern,
            trained,
        )
        return snn_result, subconscious

    def _classify_pattern(self, energies: List[float]) -> Tuple[str, Optional[float]]:
        """Classify the dominant pattern in the recent reservoir energies.

        ``("insufficient_data", None)`` with fewer than ``min_states_for_pattern``
        states. The strength is measured from the energies for the stable,
        chaotic and rhythmic classes. ``transitional`` is the residual class
        (between the stable and chaotic spreads and not rhythmic); nothing
        measures how strongly a state is transitional, so its strength is None
        and never a stand-in constant.
        """
        cfg = self.config
        if len(energies) < cfg.min_states_for_pattern:
            return "insufficient_data", None
        spread = float(np.std(energies))
        if spread > cfg.chaotic_energy_std:
            return "chaotic", min(spread / (2.0 * cfg.chaotic_energy_std), 1.0)
        if spread < cfg.stable_energy_std:
            return "stable", 1.0 - spread / cfg.stable_energy_std
        changes = int(np.sum(np.diff(np.sign(np.diff(energies))) != 0))
        if changes > cfg.rhythmic_sign_changes:
            return "rhythmic", min(changes / float(len(energies)), 1.0)
        return "transitional", None

    @staticmethod
    def _prediction_confidence(trajectory: List[float]) -> Optional[float]:
        """Smoothness of the trajectory (1 over 1 plus the variance of its
        differences); None with fewer than three points."""
        if len(trajectory) < 3:
            return None
        variance = float(np.var(np.diff(trajectory)))
        return float(np.clip(1.0 / (1.0 + variance), 0.0, 1.0))

    # ---- storage ------------------------------------------------------

    def _reservoir_size(self) -> int:
        config = getattr(self.lsm, "config", None)
        size = getattr(config, "reservoir_size", None)
        return int(size if size is not None else getattr(self.lsm, "n_reservoir", 0))

    def _store_liquid_state(
        self,
        emotion_state: EmotionState,
        subconscious: SubconsciousState,
        reservoir_state: np.ndarray,
        readout: np.ndarray,
    ) -> None:
        trained = subconscious.readout_trained
        readout_values = None
        if trained:
            readout_values = {
                name: _finite_or_none(readout[i])
                for i, name in enumerate(READOUT_CHANNELS)
                if i < len(readout)
            }
        state = LiquidState(
            timestamp=to_utc(subconscious.timestamp),
            reservoir_size=self._reservoir_size(),
            neuron_states=[float(v) for v in reservoir_state],
            input_pattern=[
                float(emotion_state.emotions.get(e, 0.0)) for e in self.emotion_channels
            ],
            readout_values=readout_values,
            entropy=subconscious.reservoir_entropy,
            coherence=subconscious.emotional_coherence,
            emotional_valence=subconscious.intuitive_valence,
            arousal_level=subconscious.intuitive_arousal,
            metadata={
                "dominant_pattern": subconscious.dominant_pattern,
                "pattern_strength": subconscious.pattern_strength,
                "prediction_confidence": subconscious.prediction_confidence,
                "readout_trained": trained,
                "path": self.path,
                **subconscious.metadata,
            },
        )
        try:
            self.neural_data_manager.store_liquid_state(state)
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("liquid state storage failed: %s", type(exc).__name__)

    # ---- reporting ------------------------------------------------------

    def get_intuitive_summary(self) -> Dict[str, Any]:
        """Summarize the most recent subconscious states.

        ``intuitive_feelings`` values are averages over the states that have a
        readout value and None when none do (an untrained readout).
        """
        if not self.state_history:
            return {
                "status": "no_data",
                "message": "No subconscious states recorded yet",
            }
        recent = self.state_history[-self.config.summary_window :]

        def mean_of(name: str) -> Optional[float]:
            values = [getattr(s, name) for s in recent if getattr(s, name) is not None]
            return float(np.mean(values)) if values else None

        valences = [
            s.intuitive_valence for s in recent if s.intuitive_valence is not None
        ]
        trend = None
        if len(valences) >= 2:
            trend = "increasing" if valences[-1] > valences[0] else "decreasing"
        patterns = [s.dominant_pattern for s in recent]
        counts = {p: patterns.count(p) for p in set(patterns)}
        return {
            "status": "active",
            "num_states": len(recent),
            "readout_trained": any(s.readout_trained for s in recent),
            "intuitive_feelings": {
                "valence": mean_of("intuitive_valence"),
                "arousal": mean_of("intuitive_arousal"),
                "coherence": mean_of("emotional_coherence"),
                "valence_trend": trend,
            },
            "patterns": {
                "dominant": max(counts.items(), key=lambda kv: kv[1])[0],
                "distribution": counts,
            },
            "reservoir_metrics": {
                "avg_energy": float(np.mean([s.reservoir_energy for s in recent])),
                "avg_entropy": mean_of("reservoir_entropy"),
                "avg_sparsity": float(np.mean([s.reservoir_sparsity for s in recent])),
            },
            "prediction_confidence": mean_of("prediction_confidence"),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return the SNN, reservoir and bridge metrics."""
        if self.path == "production":
            snn_metrics = {
                "total_spikes": self._last_snn_metrics.get("total_spikes"),
                "mean_firing_rate": self._last_snn_metrics.get("mean_firing_rate"),
                "neurons": self.snn._count_neurons(),
                "production": True,
            }
            lsm_metrics = {
                "reservoir_size": self._reservoir_size(),
                "type": "ProductionLSM",
            }
        else:
            snn_metrics = self.emotion_encoder.get_metrics()
            snn_metrics["production"] = False
            lsm_metrics = {
                "reservoir_size": self._reservoir_size(),
                "type": "LiquidStateMachine",
            }
        return {
            "snn_metrics": snn_metrics,
            "lsm_metrics": lsm_metrics,
            "bridge_metrics": {
                "total_states_processed": self.states_processed,
                "liquid_state_storage_enabled": self.store_states,
                "reservoir_size": self._reservoir_size(),
                "readout_trained": self.readout_trained(),
            },
            "intuitive_summary": self.get_intuitive_summary(),
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` plus the path,
        the reason for the NumPy path, the SNN device and the readout state."""
        device = getattr(self.snn, "device", None)
        storage = None
        if self.neural_data_manager is not None:
            storage = self.neural_data_manager.status()
        return {
            "backend": self.path,
            "available": True,
            "last_error": self._last_error,
            "path": self.path,
            "reason": self._fallback_reason,
            "device": str(device) if device is not None else "cpu",
            "readout_trained": self.readout_trained(),
            "states_processed": self.states_processed,
            "storage": storage,
        }

    # ---- persistence ----------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize the bounded state history and the counters.

        The history holds the subconscious summaries (including the names of
        the emotions that were present), so treat persisted bridge state as
        private data. The reservoir's readout persists through the LSM's own
        state dict; the per-state reservoir vectors are not persisted.
        """
        entries = []
        for state in self.state_history:
            entries.append(
                {
                    "timestamp": state.timestamp.isoformat(),
                    "reservoir_energy": state.reservoir_energy,
                    "reservoir_entropy": state.reservoir_entropy,
                    "reservoir_sparsity": state.reservoir_sparsity,
                    "intuitive_valence": state.intuitive_valence,
                    "intuitive_arousal": state.intuitive_arousal,
                    "emotional_coherence": state.emotional_coherence,
                    "dominant_pattern": state.dominant_pattern,
                    "pattern_strength": state.pattern_strength,
                    "state_trajectory": list(state.state_trajectory),
                    "prediction_confidence": state.prediction_confidence,
                    "metadata": dict(state.metadata),
                    "readout_trained": state.readout_trained,
                    "path": state.path,
                }
            )
        return {
            "scalars": {
                "states_processed": self.states_processed,
                "path": self.path,
            },
            "arrays": {},
            "histories": {"subconscious_states": entries},
        }

    @staticmethod
    def _state_from_entry(entry: Any) -> Optional[SubconsciousState]:
        if not isinstance(entry, dict):
            return None
        try:
            stamp = datetime.fromisoformat(entry["timestamp"])
            trajectory = [float(v) for v in entry.get("state_trajectory", [])]
            energy = float(entry["reservoir_energy"])
            sparsity = float(entry["reservoir_sparsity"])
            pattern = str(entry["dominant_pattern"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in trajectory + [energy, sparsity]):
            return None
        metadata = entry.get("metadata")
        return SubconsciousState(
            timestamp=stamp,
            reservoir_energy=energy,
            reservoir_entropy=_finite_or_none(entry.get("reservoir_entropy")),
            reservoir_sparsity=sparsity,
            intuitive_valence=_finite_or_none(entry.get("intuitive_valence")),
            intuitive_arousal=_finite_or_none(entry.get("intuitive_arousal")),
            emotional_coherence=_finite_or_none(entry.get("emotional_coherence")),
            dominant_pattern=pattern,
            pattern_strength=_finite_or_none(entry.get("pattern_strength")),
            state_trajectory=trajectory,
            prediction_confidence=_finite_or_none(entry.get("prediction_confidence")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            readout_trained=bool(entry.get("readout_trained", False)),
            path=str(entry.get("path", "numpy")),
        )

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the history and counters saved by ``to_state_dict``.

        Malformed entries are skipped with a logged warning; a restore never
        raises on bad data. The readout weights are not part of this state, so
        a restored bridge reports its readout as trained only when the
        substrate itself says so (the production LSM restores its own readout;
        the NumPy readout is not restored and reads as untrained).
        """
        scalars = state.get("scalars") or {}
        count = scalars.get("states_processed")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            self.states_processed = count
        entries = (state.get("histories") or {}).get("subconscious_states")
        if entries is None:
            return
        if not isinstance(entries, list):
            logger.warning("emotion bridge restore skipped a bad state history")
            return
        restored = [self._state_from_entry(e) for e in entries]
        kept = [s for s in restored if s is not None]
        if len(kept) != len(restored):
            logger.warning(
                "emotion bridge restore skipped %d malformed history entries",
                len(restored) - len(kept),
            )
        self.state_history = kept[-self.config.history_limit :]
