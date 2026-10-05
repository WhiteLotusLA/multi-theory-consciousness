"""The persistence hub: tiers, atomic writes, manifest version, NaN gate."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from mtc.consciousness import state_persistence
from mtc.consciousness.metacognition import (
    FirstOrderStateType,
    HigherOrderThought,
    MetacognitionModule,
    MetaType,
)
from mtc.consciousness.pad_affect import PADAffectModel, PADSources, PADState
from mtc.consciousness.state_persistence import (
    STATE_VERSION,
    ConsciousnessStateManager,
)
from mtc.core import config


class _ArrayModule:
    """A minimal module with all three tiers."""

    def __init__(self):
        self.counter = 0
        self.weights = np.zeros(3)
        self.events = []
        self.restored = None

    def to_state_dict(self):
        return {
            "scalars": {"counter": self.counter},
            "arrays": {"weights": self.weights.copy()},
            "histories": {"events": list(self.events)},
        }

    def from_state_dict(self, state):
        self.restored = state
        self.counter = state.get("scalars", {}).get("counter", 0)
        if "weights" in state.get("arrays", {}):
            self.weights = np.array(state["arrays"]["weights"], dtype=float)
        self.events = list(state.get("histories", {}).get("events", []))


class _NaNModule(_ArrayModule):
    def to_state_dict(self):
        state = super().to_state_dict()
        state["scalars"]["level"] = float("nan")
        return state


async def _active_metacognition() -> MetacognitionModule:
    meta = MetacognitionModule(hot_generation_threshold=0.0)
    for i in range(4):
        state = meta.register_first_order_state(
            content=f"synthetic {i}",
            content_summary=f"synthetic state {i}",
            state_type=FirstOrderStateType.BELIEF,
            source_module="test",
            confidence=0.8,
            evidence_strength=0.7,
        )
        await meta.generate_hot(state, meta_type=MetaType.AWARENESS, voluntary=True)
    return meta


@pytest.mark.asyncio
async def test_save_and_restore_round_trip_with_two_modules(tmp_path):
    pad = PADAffectModel()
    for i in range(6):
        pad.compute(PADSources(somatic_valence=0.1 * i, protoself_arousal=0.5))
    meta = await _active_metacognition()

    hub = ConsciousnessStateManager(tmp_path, {"pad": pad, "metacognition": meta})
    report = hub.save_all()
    assert set(report["modules"]) == {"pad", "metacognition"}
    assert (tmp_path / "manifest.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == STATE_VERSION

    pad2 = PADAffectModel()
    meta2 = MetacognitionModule()
    hub2 = ConsciousnessStateManager(tmp_path, {"pad": pad2, "metacognition": meta2})
    assert hub2.restore_all() == {"pad": True, "metacognition": True}

    assert [s.pleasure for s in pad2._history] == pytest.approx(
        [s.pleasure for s in pad._history]
    )
    assert all(isinstance(s, PADState) for s in pad2._history)
    assert pad2.generate_context() == pad.generate_context()
    assert [h.id for h in meta2.higher_order_thoughts] == [
        h.id for h in meta.higher_order_thoughts
    ]
    assert all(isinstance(h, HigherOrderThought) for h in meta2.higher_order_thoughts)
    assert set(meta2.hots_by_id) == {h.id for h in meta2.higher_order_thoughts}
    report_text = await meta2.generate_metacognitive_report()
    assert "Most recent awareness" in report_text


def test_restore_without_saved_state_starts_fresh(tmp_path):
    module = _ArrayModule()
    hub = ConsciousnessStateManager(tmp_path / "missing", {"m": module})
    assert hub.restore_all() == {"m": False}
    assert module.restored is None


def test_version_mismatch_starts_fresh(tmp_path):
    module = _ArrayModule()
    module.counter = 7
    hub = ConsciousnessStateManager(tmp_path, {"m": module})
    hub.save_all()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = STATE_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    fresh = _ArrayModule()
    hub2 = ConsciousnessStateManager(tmp_path, {"m": fresh})
    assert hub2.restore_all() == {"m": False}
    assert fresh.restored is None
    assert fresh.counter == 0
    # The stale files are set aside, so a later partial save cannot mix versions
    assert not (tmp_path / "m_scalars.json").exists()
    assert not manifest_path.exists()
    archived = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(archived) == 1
    assert (archived[0] / "m_scalars.json").exists()


def test_nan_scalars_are_skipped_and_logged(tmp_path, caplog):
    good = _ArrayModule()
    bad = _NaNModule()
    hub = ConsciousnessStateManager(tmp_path, {"good": good, "bad": bad})
    with caplog.at_level(logging.WARNING, logger=state_persistence.__name__):
        report = hub.save_all()
    assert "bad" in caplog.text and "NaN" in caplog.text
    assert report["modules"]["good"]
    assert report["modules"]["bad"] == "skipped: non-finite scalars"
    assert not list(tmp_path.glob("bad_*"))
    assert (tmp_path / "good_scalars.json").exists()

    # A NaN that reaches disk anyway is refused on restore
    (tmp_path / "good_scalars.json").write_text('{"counter": NaN}')
    target = _ArrayModule()
    hub2 = ConsciousnessStateManager(tmp_path, {"good": target})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=state_persistence.__name__):
        assert hub2.restore_all() == {"good": False}
    assert "good" in caplog.text
    assert target.restored is None


def test_arrays_saved_every_n_cycles_only(tmp_path):
    module = _ArrayModule()
    hub = ConsciousnessStateManager(tmp_path, {"m": module}, arrays_every=3)
    npz = tmp_path / "m_arrays.npz"

    for cycle in range(1, 7):
        module.counter = cycle
        module.weights = np.full(3, float(cycle))
        module.events.append(cycle)
        hub.on_cycle_complete()
        scalars = json.loads((tmp_path / "m_scalars.json").read_text())
        assert scalars["counter"] == cycle
        if cycle < 3:
            assert not npz.exists()
        else:
            with np.load(npz, allow_pickle=False) as data:
                expected = 3.0 if cycle < 6 else 6.0
                np.testing.assert_allclose(data["weights"], np.full(3, expected))
        # Histories are written on demand only
        assert not (tmp_path / "m_histories.json").exists()

    hub.save_all()
    assert json.loads((tmp_path / "m_histories.json").read_text()) == {
        "events": [1, 2, 3, 4, 5, 6]
    }


def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    module = _ArrayModule()
    module.counter = 1
    module.weights = np.ones(3)
    hub = ConsciousnessStateManager(tmp_path, {"m": module})
    hub.save_all()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    module.counter = 2
    module.weights = np.full(3, 2.0)

    def failing_fsync(fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(state_persistence, "_fsync", failing_fsync)
    report = hub.save_all()
    assert str(report["modules"]["m"]).startswith("error")

    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert after == before
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]

    restored = _ArrayModule()
    monkeypatch.setattr(state_persistence, "_fsync", os.fsync)
    assert ConsciousnessStateManager(tmp_path, {"m": restored}).restore_all() == {
        "m": True
    }
    assert restored.counter == 1
    np.testing.assert_allclose(restored.weights, np.ones(3))


def test_object_arrays_are_refused(tmp_path):
    class _ObjectArrays(_ArrayModule):
        def to_state_dict(self):
            state = super().to_state_dict()
            state["arrays"]["bad"] = np.array([{"a": 1}], dtype=object)
            return state

    hub = ConsciousnessStateManager(tmp_path, {"m": _ObjectArrays()})
    report = hub.save_all()
    assert str(report["modules"]["m"]).startswith("error")
    assert not (tmp_path / "m_arrays.npz").exists()


def test_register_rejects_modules_without_contract(tmp_path):
    hub = ConsciousnessStateManager(tmp_path)
    with pytest.raises(TypeError):
        hub.register("plain", object())
    with pytest.raises(ValueError):
        hub.register("../escape", _ArrayModule())


def test_default_dir_from_settings(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    assert config.Settings().STATE_DIR == "agent_space/consciousness_state"

    target = tmp_path / "custom_state"
    monkeypatch.setenv("MTC_STATE_DIR", str(target))
    monkeypatch.setattr(config, "_settings", None)
    hub = ConsciousnessStateManager()
    assert hub.state_dir == Path(target)
    assert not target.exists()  # nothing is created until the first save
    hub.register("m", _ArrayModule())
    hub.save_all()
    assert (target / "manifest.json").exists()


@pytest.mark.asyncio
async def test_hub_round_trip_with_array_modules(tmp_path):
    """String and float arrays survive the npz path with pickle disabled."""
    from mtc.consciousness.beautiful_loop import BeautifulLoop, ConsciousMoment
    from mtc.consciousness.damasio import DamasioLayers

    class _Drives:
        def get_drive_state(self):
            return {"attention_budget": {"current_level": 0.6, "urgency": 0.1}}

        def get_overall_valence(self):
            return 0.2

        def get_free_energy(self):
            return 0.1

    loop = BeautifulLoop()
    layers = DamasioLayers()
    fresh_loop = BeautifulLoop()  # empty context map: zero-length string array
    for step in range(4):
        await loop.process_conscious_moment(
            prediction_errors=[0.1 * step, 0.2, 0.3],
            context="general" if step % 2 else "focus",
        )
        await layers.process(homeostatic_drives=_Drives(), workspace_winners=[])

    modules = {"blt": loop, "damasio": layers, "blt_fresh": fresh_loop}
    ConsciousnessStateManager(tmp_path, modules).save_all()

    loop2, layers2, fresh2 = BeautifulLoop(), DamasioLayers(), BeautifulLoop()
    hub = ConsciousnessStateManager(
        tmp_path, {"blt": loop2, "damasio": layers2, "blt_fresh": fresh2}
    )
    assert hub.restore_all() == {"blt": True, "damasio": True, "blt_fresh": True}
    assert set(loop2.hyper_model.context_precision_map) == {"general", "focus"}
    assert fresh2.hyper_model.context_precision_map == {}
    assert all(isinstance(m, ConsciousMoment) for m in loop2._moment_history)
    assert isinstance(loop2.generate_report(), str)
    np.testing.assert_allclose(
        layers2.protoself._previous_body_vector, layers.protoself._previous_body_vector
    )
    assert layers2.generate_context() == layers.generate_context()
