"""The legacy integration awaits every indicator, reports a failed one as None,
aggregates and weights confidence by coverage, and collects only what the
orchestrator really produced. Every orchestrator here is a fake except the one
whose emitted metric keys are checked."""

import asyncio
import gc
import warnings
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from mtc.consciousness.consciousness_integration import (
    ConsciousnessIntegration,
    ConsciousnessSnapshot,
    ConsciousNeuralOrchestrator,
)
from mtc.consciousness.consciousness_metrics import INDICATOR_NAMES
from mtc.core.config import Settings
from mtc.neural.orchestrator import NeuralOrchestrator

SUBSTRATES = ("snn", "lsm", "htm", "memory", "thoughts")


class _OkIndicator:
    def __init__(self, score):
        self.score = score
        self.awaited = 0

    async def measure(self, data):
        await asyncio.sleep(0)
        self.awaited += 1
        return self.score


class _FailIndicator:
    async def measure(self, data):
        raise ValueError("Missing required data: ['attention_focus']")


def _memory_metrics():
    return {
        "episodic_memories": 4,
        "semantic_patterns": 2,
        "experiences_in_buffer": 1,
        "consolidation_strength": None,
    }


class _FakeOrchestrator:
    def __init__(self, experience=None, htm_metrics=None, machine=None):
        self.state = SimpleNamespace(
            current_experience=experience,
            avg_processing_time_ms=2.5,
            experiences_processed=3,
        )
        self.total_sleep_cycles = 1
        self.continuous_thought_machine = machine
        self.enhanced_workspace = None
        self.latest_neural_signals = None
        self._last_binding_result = None
        self._last_retrieved_memories = []
        self.metrics = {"memory": _memory_metrics(), "htm_metrics": htm_metrics}

    def get_system_metrics(self):
        return self.metrics

    def get_continuous_thought_status(self):
        return {
            "total_thoughts_generated": 12,
            "consciousness_state": "awake",
            "stream_summary": {"current_intensity": 0.5, "thought_frequency": 0.5},
        }


def _integration(orchestrator=None, indicators=None, manager=None):
    integration = ConsciousnessIntegration(
        neural_orchestrator=orchestrator or _FakeOrchestrator(),
        neural_data_manager=manager,
        enable_persistence=manager is not None,
    )
    if indicators is not None:
        integration.metrics.indicators = dict(indicators)
    return integration


def _experience(snn_result=None, subconscious=None):
    return SimpleNamespace(snn_result=snn_result, subconscious_state=subconscious)


def test_assess_awaits_every_indicator():
    indicators = {name: _OkIndicator(0.5) for name in INDICATOR_NAMES}
    integration = _integration(indicators=indicators)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        snapshot = asyncio.run(integration.assess_consciousness())
        gc.collect()

    assert all(indicator.awaited == 1 for indicator in indicators.values())
    for name in INDICATOR_NAMES:
        assert isinstance(getattr(snapshot, name), float)
    assert snapshot.overall_consciousness_level == pytest.approx(0.5)
    assert snapshot.status == "measured"
    assert snapshot.coverage == pytest.approx(1.0)
    assert snapshot.consciousness_confidence == pytest.approx(1.0)
    assert not [w for w in caught if "never awaited" in str(w.message)]


def test_total_failure_is_insufficient_data_not_confident_zero():
    # The real 14 indicators against an orchestrator that produced nothing.
    snapshot = asyncio.run(_integration().assess_consciousness())

    assert snapshot.overall_consciousness_level is None
    assert snapshot.consciousness_confidence == 0.0
    assert snapshot.status == "insufficient_data"
    assert snapshot.coverage == 0.0
    assert snapshot.indicators_succeeded == 0
    assert snapshot.indicators_total == 14
    assert set(snapshot.failed_indicators) == set(INDICATOR_NAMES)
    assert all(snapshot.failed_indicators.values())
    for name in INDICATOR_NAMES:
        assert getattr(snapshot, name) is None


def test_partial_failure_scales_confidence_by_coverage():
    indicators = {
        "global_workspace_integration": _OkIndicator(0.8),
        "integrated_information_phi": _OkIndicator(0.6),
        "attention_mechanisms": _FailIndicator(),
        "working_memory_capacity": _FailIndicator(),
    }
    snapshot = asyncio.run(_integration(indicators=indicators).assess_consciousness())

    assert snapshot.overall_consciousness_level == pytest.approx(0.7)
    assert snapshot.status == "partial"
    assert snapshot.indicators_succeeded == 2
    assert snapshot.indicators_total == 4
    assert snapshot.coverage == pytest.approx(0.5)
    agreement = 1.0 - min(float(np.var([0.8, 0.6])) * 2, 1.0)
    assert snapshot.consciousness_confidence == pytest.approx(agreement * 0.5)


def test_failed_indicator_fields_stay_none():
    indicators = {
        "global_workspace_integration": _OkIndicator(0.8),
        "attention_mechanisms": _FailIndicator(),
    }
    snapshot = asyncio.run(_integration(indicators=indicators).assess_consciousness())
    assert snapshot.global_workspace_integration == pytest.approx(0.8)
    assert snapshot.attention_mechanisms is None
    assert "Missing required data" in snapshot.failed_indicators["attention_mechanisms"]
    measurement = snapshot.raw_metrics["measurements"]["attention_mechanisms"]
    assert measurement["score"] is None
    assert measurement["status"] == "insufficient_data"
    # A fresh snapshot claims nothing.
    fresh = ConsciousnessSnapshot(timestamp=datetime.now())
    assert fresh.status == "insufficient_data" and fresh.coverage == 0.0
    assert all(getattr(fresh, name) is None for name in INDICATOR_NAMES)


def test_snn_is_real_only_with_synchrony():
    real = _experience(
        snn_result={
            "metrics": {"synchrony": 0.8},
            "firing_rates": {"output": [0.1, 0.4, 0.9]},
            "total_spikes": 7,
            "spike_rate": 0.2,
        }
    )
    block = _integration(_FakeOrchestrator(real)).collect_neural_data()["snn"]
    assert block["is_real"] is True
    assert block["synchrony"] == pytest.approx(0.8)
    assert block["temporal_patterns"] == [0.1, 0.4, 0.9]
    assert block["total_spikes"] == 7

    silent = _experience(snn_result={"metrics": {}, "total_spikes": 1})
    block = _integration(_FakeOrchestrator(silent)).collect_neural_data()["snn"]
    assert block["is_real"] is False
    assert block["synchrony"] is None
    assert block["temporal_patterns"] == []


def test_lsm_reads_emotional_coherence():
    # No ``coherence`` attribute: reading it is what used to push the whole
    # collection into the fallback.
    subconscious = SimpleNamespace(
        emotional_coherence=0.77,
        dominant_pattern="rhythmic",
        intuitive_valence=0.4,
        intuitive_arousal=-0.2,
        state_trajectory=[0.1, 0.2, 0.3],
    )
    data = _integration(
        _FakeOrchestrator(_experience(subconscious=subconscious))
    ).collect_neural_data()
    assert "collection_error" not in data
    block = data["lsm"]
    assert block["is_real"] is True
    assert block["coherence"] == pytest.approx(0.77)
    assert block["dominant_pattern"] == "rhythmic"
    assert block["intuitive_valence"] == pytest.approx(0.4)
    assert block["intuitive_arousal"] == pytest.approx(-0.2)
    assert block["state_trajectory"] == [0.1, 0.2, 0.3]
    assert data["memory"]["episodic_count"] == 4

    no_coherence = SimpleNamespace(dominant_pattern="idle", state_trajectory=[])
    block = _integration(
        _FakeOrchestrator(_experience(subconscious=no_coherence))
    ).collect_neural_data()["lsm"]
    assert block["is_real"] is False
    assert block["coherence"] is None


def test_htm_is_real_only_with_anomaly():
    measured = {"anomaly_score": 0.2, "active_columns": 5, "predicted_columns": 2}
    block = _integration(_FakeOrchestrator(htm_metrics=measured)).collect_neural_data()[
        "htm"
    ]
    assert block["is_real"] is True
    assert block["anomaly_score"] == pytest.approx(0.2)
    assert block["active_columns"] == 5

    block = _integration(
        _FakeOrchestrator(htm_metrics={"active_columns": 5})
    ).collect_neural_data()["htm"]
    assert block["is_real"] is False
    assert block["anomaly_score"] is None

    block = _integration(_FakeOrchestrator(htm_metrics=None)).collect_neural_data()[
        "htm"
    ]
    assert block["is_real"] is False
    assert block["anomaly_score"] is None


def _numbers(block):
    return [
        value
        for key, value in block.items()
        if key not in ("is_real", "current_state", "error", "temporal_patterns")
        and key != "state_trajectory"
    ]


def test_no_fallback_constants_when_substrates_absent():
    data = _integration(_FakeOrchestrator()).collect_neural_data()
    for name in ("snn", "lsm", "htm", "thoughts"):
        assert data[name]["is_real"] is False, name
        assert all(value is None for value in _numbers(data[name])), name
    assert data["snn"]["temporal_patterns"] == []
    assert data["memory"]["consolidation_strength"] is None

    class _Broken(_FakeOrchestrator):
        def get_system_metrics(self):
            raise RuntimeError("metrics unavailable")

    data = _integration(_Broken()).collect_neural_data()
    assert "metrics unavailable" in data["collection_error"]
    for name in SUBSTRATES:
        assert data[name]["is_real"] is False
        assert "metrics unavailable" in data[name]["error"]
        assert all(value is None for value in _numbers(data[name])), name


def test_legacy_workspace_indicator_is_failed_without_token_attention():
    signals = {"snn": np.array([0.1, 0.4, 0.2, 0.7])}
    orchestrator = _FakeOrchestrator()
    orchestrator.latest_neural_signals = signals
    integration = _integration(orchestrator)
    data = integration.collect_neural_data()
    assert "llm_processing" not in data
    snapshot = asyncio.run(integration.assess_consciousness(neural_data=data))

    assert snapshot.global_workspace_integration is None
    assert "global_workspace_integration" in snapshot.failed_indicators
    assert "llm_processing.token_attention" in (
        snapshot.failed_indicators["global_workspace_integration"]
    )
    # Coverage counts only measured indicators, so the failed one is out.
    measured = [n for n in INDICATOR_NAMES if getattr(snapshot, n) is not None]
    assert "global_workspace_integration" not in measured
    assert snapshot.coverage == pytest.approx(len(measured) / 14)


def test_thoughts_only_when_the_machine_runs():
    running = SimpleNamespace(running=True)
    data = _integration(_FakeOrchestrator(machine=running)).collect_neural_data()
    assert data["thoughts"]["is_real"] is True
    assert data["thoughts"]["thoughts_generated"] == 12
    assert data["thoughts"]["coherence_score"] == pytest.approx(0.5)

    stopped = SimpleNamespace(running=False)
    data = _integration(_FakeOrchestrator(machine=stopped)).collect_neural_data()
    assert data["thoughts"]["is_real"] is False
    assert data["thoughts"]["coherence_score"] is None


class _RecordingDict(dict):
    """A dict that records every key path read through it."""

    def __init__(self, source, log, prefix=""):
        super().__init__(source)
        self._log = log
        self._prefix = prefix

    def _wrap(self, key, value):
        path = f"{self._prefix}{key}"
        self._log.add(path)
        if isinstance(value, dict):
            return _RecordingDict(value, self._log, f"{path}.")
        return value

    def __getitem__(self, key):
        return self._wrap(key, super().__getitem__(key))

    def get(self, key, default=None):
        if key in self:
            return self._wrap(key, super().__getitem__(key))
        self._log.add(f"{self._prefix}{key}")
        return default


def _has_path(source, path):
    node = source
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def test_integration_reads_only_keys_the_orchestrator_emits():
    orchestrator = NeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    real = orchestrator.get_system_metrics()
    reads = set()
    orchestrator.get_system_metrics = lambda: _RecordingDict(real, reads)

    data = ConsciousnessIntegration(
        neural_orchestrator=orchestrator, enable_persistence=False
    ).collect_neural_data()

    assert "collection_error" not in data
    assert reads
    for path in sorted(reads):
        assert _has_path(real, path), f"read {path!r} that the orchestrator never emits"


def _snapshot(level, **scores):
    snapshot = ConsciousnessSnapshot(timestamp=datetime.now())
    snapshot.overall_consciousness_level = level
    for name, value in scores.items():
        setattr(snapshot, name, value)
    return snapshot


def test_summary_excludes_unmeasured():
    integration = _integration()
    integration.snapshot_history = [
        _snapshot(0.4, global_workspace_integration=0.5),
        _snapshot(None),
        _snapshot(0.6, global_workspace_integration=0.7),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = integration.get_statistical_summary()

    assert summary["overall"]["measured_samples"] == 2
    assert summary["overall"]["mean"] == pytest.approx(0.5)
    assert summary["overall"]["trend"] == "increasing"
    gwi = summary["global_workspace_integration"]
    assert gwi["measured_samples"] == 2
    assert gwi["mean"] == pytest.approx(0.6)
    assert gwi["current"] == pytest.approx(0.7)
    empty = summary["embodiment"]
    assert empty["measured_samples"] == 0
    assert empty["mean"] is None and empty["current"] is None

    integration.snapshot_history = [_snapshot(None), _snapshot(None)]
    summary = integration.get_statistical_summary()
    assert summary["overall"]["measured_samples"] == 0
    assert summary["overall"]["trend"] == "insufficient_data"


def test_hypothesis_needs_two_measured_and_uses_linregress():
    integration = _integration()
    integration.snapshot_history = [_snapshot(None) for _ in range(31)]
    integration.snapshot_history[5] = _snapshot(0.3)
    result = integration.validate_consciousness_hypothesis("consciousness_increasing")
    assert result["valid"] is False
    assert "insufficient_data" in result["reason"]
    assert result["samples"] == 1

    # Two measured points leave the fit zero degrees of freedom, where
    # linregress reports p = 0.0 for any increase; that is not evidence.
    integration.snapshot_history[20] = _snapshot(0.4)
    result = integration.validate_consciousness_hypothesis("consciousness_increasing")
    assert result["valid"] is False
    assert "insufficient_data" in result["reason"]
    assert result["samples"] == 2
    assert "p_value" not in result

    rng = np.random.default_rng(3)
    levels = [0.2 + 0.01 * i + float(rng.normal(0, 0.005)) for i in range(32)]
    history = [_snapshot(level) for level in levels]
    history[10] = _snapshot(None)
    integration.snapshot_history = history
    result = integration.validate_consciousness_hypothesis("consciousness_increasing")

    positions = [i for i in range(32) if i != 10]
    expected = stats.linregress(positions, [levels[i] for i in positions])
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["correlation"] == pytest.approx(expected.rvalue)
    assert result["samples"] == 31
    assert result["valid"] is True
    assert result["trend"] == "positive"


class _FakeAssessor:
    def __init__(self, level):
        self.level = level

    def collect_neural_data(self):
        return {}

    async def assess_consciousness(self, neural_data=None):
        await asyncio.sleep(0)
        snapshot = ConsciousnessSnapshot(timestamp=datetime.now())
        snapshot.overall_consciousness_level = self.level
        snapshot.status = "measured" if self.level is not None else "insufficient_data"
        return snapshot


def test_process_experience_inside_running_loop():
    orchestrator = ConsciousNeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    orchestrator.consciousness = _FakeAssessor(0.42)

    async def inside_loop():
        experience = orchestrator.process_experience("observation", {"joy": 0.5})
        # The level arrives after the call returns.
        assert experience.consciousness_level is None
        await asyncio.gather(*list(orchestrator._pending_assessments))
        return experience

    experience = asyncio.run(inside_loop())
    assert experience.consciousness_level == pytest.approx(0.42)
    assert not orchestrator._pending_assessments

    # Outside a running loop the assessment completes before the call returns.
    experience = orchestrator.process_experience("observation", {"joy": 0.5})
    assert experience.consciousness_level == pytest.approx(0.42)

    orchestrator.consciousness = _FakeAssessor(None)
    experience = orchestrator.process_experience("observation", {"joy": 0.5})
    assert experience.consciousness_level is None


class _ContextIndicator:
    """Scores the experience whose data it measures: A as 0.2, B as 0.7."""

    async def measure(self, data):
        await asyncio.sleep(0)
        return {"A": 0.2, "B": 0.7}[data["_context"]]


def test_back_to_back_experiences_in_loop_measure_their_own_data():
    orchestrator = ConsciousNeuralOrchestrator(
        enable_mongodb=False, emotion_bridge=False, memory_consolidator=False
    )
    integration = orchestrator.consciousness
    integration.metrics.indicators = {
        "global_workspace_integration": _ContextIndicator()
    }
    collect = integration.collect_neural_data
    collected = []

    def tagged_collect():
        data = collect()
        data["_context"] = orchestrator.state.current_experience.context
        collected.append(data["_context"])
        return data

    integration.collect_neural_data = tagged_collect

    async def inside_loop():
        first = orchestrator.process_experience(
            "observation", {"joy": 0.5}, context="A"
        )
        second = orchestrator.process_experience(
            "observation", {"fear": 0.5}, context="B"
        )
        await asyncio.gather(*list(orchestrator._pending_assessments))
        return first, second

    first, second = asyncio.run(inside_loop())
    assert collected == ["A", "B"]
    assert first.consciousness_level == pytest.approx(0.2)
    assert second.consciousness_level == pytest.approx(0.7)


class _SyncManager:
    def __init__(self):
        self.stored = []

    def store_consciousness_snapshot(self, doc):
        self.stored.append(doc)
        return "id"


class _AsyncManager(_SyncManager):
    async def store_consciousness_snapshot(self, doc):
        await asyncio.sleep(0)
        self.stored.append(doc)
        return "id"


@pytest.mark.parametrize("manager_cls", [_SyncManager, _AsyncManager])
def test_persist_snapshot_sync_or_async(manager_cls):
    manager = manager_cls()
    indicators = {
        "global_workspace_integration": _OkIndicator(0.8),
        "attention_mechanisms": _FailIndicator(),
    }
    integration = _integration(indicators=indicators, manager=manager)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(integration.assess_consciousness())
        gc.collect()

    assert len(manager.stored) == 1
    doc = manager.stored[0]
    assert doc["_type"] == "consciousness_snapshot"
    assert doc["status"] == "partial"
    assert doc["coverage"] == pytest.approx(0.5)
    assert doc["attention_mechanisms"] is None
    assert "attention_mechanisms" in doc["failed_indicators"]
    assert isinstance(doc["timestamp"], str)
    assert not [w for w in caught if "never awaited" in str(w.message)]


def test_snapshot_persist_failure_is_counted_and_raised_under_debug():
    class _Failing:
        def store_consciousness_snapshot(self, doc):
            raise ConnectionError("snapshot store down")

    indicators = {"global_workspace_integration": _OkIndicator(0.8)}
    quiet = ConsciousnessIntegration(
        neural_orchestrator=_FakeOrchestrator(),
        neural_data_manager=_Failing(),
        enable_persistence=True,
        settings=Settings(DEBUG=False),
    )
    quiet.metrics.indicators = dict(indicators)
    snapshot = asyncio.run(quiet.assess_consciousness())
    # The assessment completes and is kept as before; the write is counted.
    assert snapshot.overall_consciousness_level == pytest.approx(0.8)
    assert quiet.snapshot_history == [snapshot]
    assert quiet.total_assessments == 1
    assert quiet.get_status()["failures"] == {"persist_snapshot": 1}

    loud = ConsciousnessIntegration(
        neural_orchestrator=_FakeOrchestrator(),
        neural_data_manager=_Failing(),
        enable_persistence=True,
        settings=Settings(DEBUG=True),
    )
    loud.metrics.indicators = dict(indicators)
    with pytest.raises(ConnectionError, match="snapshot store down"):
        asyncio.run(loud.assess_consciousness())
    assert loud.get_status()["failures"] == {"persist_snapshot": 1}


def test_monitoring_loop_failure_is_counted_and_raised_under_debug():
    async def broken_broadcaster(message):
        raise RuntimeError("broadcaster down")

    async def run(settings):
        integration = ConsciousnessIntegration(
            neural_orchestrator=_FakeOrchestrator(),
            enable_persistence=False,
            assessment_interval_ms=1,
            settings=settings,
        )
        integration.metrics.indicators = {
            "global_workspace_integration": _OkIndicator(0.8)
        }
        integration.monitoring_active = True
        task = asyncio.ensure_future(integration._monitoring_loop(broken_broadcaster))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if integration.failures or task.done():
                break
        integration.monitoring_active = False
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return integration, task

    quiet, task = asyncio.run(run(Settings(DEBUG=False)))
    assert quiet.get_status()["failures"] == {"monitoring_loop": 1}
    assert task.cancelled()

    loud, task = asyncio.run(run(Settings(DEBUG=True)))
    assert loud.get_status()["failures"] == {"monitoring_loop": 1}
    assert isinstance(task.exception(), RuntimeError)


def test_trajectory_window_uses_snapshot_times():
    integration = _integration()
    old = _snapshot(0.1)
    old.timestamp = datetime.now() - timedelta(hours=3)
    integration.snapshot_history = [old, _snapshot(0.2)]
    window = integration.get_consciousness_trajectory(duration_minutes=60)
    assert [s.overall_consciousness_level for s in window] == [0.2]
