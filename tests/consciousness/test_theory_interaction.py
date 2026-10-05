"""
Theory interaction manager: cross-theory modulation signals.

Each of the six interactions reads one field of a consciousness state, the
signals are smoothed with an exponential moving average, a missing field
contributes nothing, and the manager persists through the module contract.
"""

import json
import math
from types import SimpleNamespace

import pytest

from mtc.consciousness.theory_interaction import (
    TheoryInteractionConfig,
    TheoryInteractionManager,
    TheoryModulations,
)

SIGNALS = (
    "somatic_salience_bias",
    "phi_competition_bias",
    "metacognitive_precision_bias",
    "free_energy_arousal",
    "recurrence_binding_boost",
    "epistemic_attention_expansion",
)


def _raw_manager(**config) -> TheoryInteractionManager:
    """A manager without smoothing, so each update reports the raw signal."""
    return TheoryInteractionManager(TheoryInteractionConfig(smoothing=0.0, **config))


def _only(modulations: TheoryModulations, name: str) -> float:
    values = modulations.to_dict()
    others = {k: v for k, v in values.items() if k != name}
    assert all(v == 0.0 for v in others.values()), others
    return values[name]


def test_config_defaults_carry_the_scale_constants():
    config = TheoryInteractionConfig()
    assert config.smoothing == 0.3
    assert config.somatic_scale == 0.15
    assert config.phi_scale == 0.10
    assert config.hot_precision_scale == 0.12
    assert config.vfe_arousal_scale == 0.20
    assert config.recurrence_scale == 0.10
    assert config.epistemic_scale == 0.08


def test_config_rejects_smoothing_outside_unit_interval():
    with pytest.raises(ValueError):
        TheoryInteractionConfig(smoothing=1.0)
    with pytest.raises(ValueError):
        TheoryInteractionConfig(smoothing=-0.1)


def test_each_interaction_reads_its_state_field():
    # Damasio to GWT: an unstable protoself heightens salience
    value = _only(
        _raw_manager().update_from_state(SimpleNamespace(protoself_stability=0.0)),
        "somatic_salience_bias",
    )
    assert value == pytest.approx(0.15)
    calm = _raw_manager().update_from_state(SimpleNamespace(protoself_stability=1.0))
    assert calm.somatic_salience_bias == pytest.approx(-0.15)

    # IIT to GWT: phi enriches competition (identity normalizer by default)
    value = _only(
        _raw_manager().update_from_state(SimpleNamespace(architectural_phi=0.5)),
        "phi_competition_bias",
    )
    assert value == pytest.approx(0.05)

    # HOT to FEP: metacognitive depth sets precision confidence
    value = _only(
        _raw_manager().update_from_state(
            SimpleNamespace(higher_order_thoughts=["a", "b", "c"])
        ),
        "metacognitive_precision_bias",
    )
    assert value == pytest.approx(0.12)
    none_seen = _raw_manager().update_from_state(
        SimpleNamespace(higher_order_thoughts=[])
    )
    assert none_seen.metacognitive_precision_bias == pytest.approx(-0.06)

    # FEP to Damasio: surprise becomes somatic arousal
    value = _only(
        _raw_manager().update_from_state(
            SimpleNamespace(variational_free_energy=0.25, prediction_error=0.1)
        ),
        "free_energy_arousal",
    )
    assert value == pytest.approx(0.10)

    # RPT to BLT: deep recurrence strengthens binding
    value = _only(
        _raw_manager().update_from_state(SimpleNamespace(rpt_deep_recurrence=0.5)),
        "recurrence_binding_boost",
    )
    assert value == pytest.approx(0.05)

    # BLT to AST: epistemic depth widens attention, saturating at depth 5
    value = _only(
        _raw_manager().update_from_state(SimpleNamespace(epistemic_depth=5)),
        "epistemic_attention_expansion",
    )
    assert value == pytest.approx(0.08)
    shallow = _raw_manager().update_from_state(SimpleNamespace(epistemic_depth=1))
    assert shallow.epistemic_attention_expansion == pytest.approx(
        0.08 * math.log1p(1) / math.log1p(5)
    )


def test_phi_normalizer_is_injected():
    manager = TheoryInteractionManager(
        TheoryInteractionConfig(smoothing=0.0), phi_normalizer=lambda value: 1.0
    )
    result = manager.update_from_state(SimpleNamespace(architectural_phi=0.01))
    assert result.phi_competition_bias == pytest.approx(0.10)


def test_smoothing_is_ema():
    manager = TheoryInteractionManager(TheoryInteractionConfig())
    unstable = SimpleNamespace(protoself_stability=0.0)  # raw signal 0.15
    neutral = SimpleNamespace(protoself_stability=0.5)  # raw signal 0.0

    first = manager.update_from_state(unstable).somatic_salience_bias
    assert first == pytest.approx(0.7 * 0.15)
    second = manager.update_from_state(unstable).somatic_salience_bias
    assert second == pytest.approx(0.3 * first + 0.7 * 0.15)
    third = manager.update_from_state(neutral).somatic_salience_bias
    assert third == pytest.approx(0.3 * second)
    assert manager.modulations.somatic_salience_bias == third


def test_missing_fields_yield_zero_modulation():
    empty = _raw_manager().update_from_state(SimpleNamespace())
    assert all(value == 0.0 for value in empty.to_dict().values())
    assert empty.active_count() == 0

    unfilled = _raw_manager().update_from_state(
        SimpleNamespace(
            architectural_phi=None,
            rpt_deep_recurrence=None,
            protoself_stability=None,
            variational_free_energy=float("nan"),
            epistemic_depth=0,
        )
    )
    assert all(value == 0.0 for value in unfilled.to_dict().values())


def test_signals_are_clamped():
    manager = _raw_manager(somatic_scale=10.0, vfe_arousal_scale=10.0)
    result = manager.update_from_state(
        SimpleNamespace(protoself_stability=0.0, variational_free_energy=5.0)
    )
    assert result.somatic_salience_bias == 1.0
    assert result.free_energy_arousal == 1.0


def test_state_dict_round_trip():
    manager = TheoryInteractionManager(TheoryInteractionConfig())
    state = SimpleNamespace(
        protoself_stability=0.2,
        architectural_phi=0.4,
        higher_order_thoughts=["hot"],
        variational_free_energy=0.3,
        prediction_error=0.2,
        rpt_deep_recurrence=0.6,
        epistemic_depth=3,
    )
    for _ in range(4):
        manager.update_from_state(state)

    saved = manager.to_state_dict()
    assert set(saved) == {"scalars", "arrays", "histories"}
    saved = json.loads(json.dumps(saved))

    restored = TheoryInteractionManager(TheoryInteractionConfig())
    restored.from_state_dict(saved)
    assert isinstance(restored.modulations, TheoryModulations)
    assert restored.modulations.to_dict() == manager.modulations.to_dict()
    assert restored.get_status()["cycle_count"] == 4

    # Both continue identically from the restored point
    assert (
        restored.update_from_state(state).to_dict()
        == manager.update_from_state(state).to_dict()
    )


def test_restore_ignores_unknown_and_non_finite_values():
    manager = TheoryInteractionManager(TheoryInteractionConfig())
    manager.from_state_dict(
        {
            "scalars": {
                "cycle_count": 7,
                "somatic_salience_bias": 0.05,
                "phi_competition_bias": float("nan"),
                "not_a_signal": 3.0,
            }
        }
    )
    assert manager.modulations.somatic_salience_bias == 0.05
    assert manager.modulations.phi_competition_bias == 0.0
    assert manager.get_status()["cycle_count"] == 7
    assert set(manager.modulations.to_dict()) == set(SIGNALS)
