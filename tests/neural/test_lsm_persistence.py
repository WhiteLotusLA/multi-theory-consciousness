"""Persistence of the LSM's learned RLS readout.

The reservoir weights are not saved: they regenerate from ``LSMConfig.seed``.
The state dict carries the readout's learned output weights, bias and inverse
correlation matrix P with the reservoir size and seed; a restore accepts them
only when both match and every array is finite.
"""

import logging

import numpy as np
import pytest

pytest.importorskip("reservoirpy")

from mtc.neural.liquid.production_lsm import LSMConfig, ProductionLSM  # noqa: E402

INPUT_DIM = 6
OUTPUT_DIM = 3


def _lsm(**overrides) -> ProductionLSM:
    defaults = dict(reservoir_size=50, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    defaults.update(overrides)
    return ProductionLSM(LSMConfig(**defaults))


def _trained(seed: int = 0) -> ProductionLSM:
    rng = np.random.default_rng(seed)
    lsm = _lsm()
    inputs = rng.standard_normal((2, 30, INPUT_DIM))
    targets = rng.standard_normal((2, 30, OUTPUT_DIM))
    lsm.train_rls(inputs, targets)
    return lsm


def test_reservoir_regenerates_from_seed():
    a, b = _lsm(), _lsm()
    x = np.random.default_rng(1).standard_normal((5, INPUT_DIM))

    np.testing.assert_array_equal(a.reservoir.run(x), b.reservoir.run(x))


def test_readout_round_trip():
    source = _trained()
    state = source.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    assert state["scalars"]["reservoir_size"] == 50
    assert state["scalars"]["rls_initialized"] is True
    assert state["arrays"]["rls_Wout"].shape == (50, OUTPUT_DIM)
    assert state["arrays"]["rls_P"].shape == (50, 50)

    target = _lsm()
    target.from_state_dict(state)
    readout = target.rls_readout
    np.testing.assert_array_equal(readout.Wout, source.rls_readout.Wout)
    np.testing.assert_array_equal(readout.bias, source.rls_readout.bias)
    np.testing.assert_array_equal(readout.P, source.rls_readout.P)
    assert readout.initialized

    # The restored readout acts on a reservoir rebuilt from the same seed
    x = np.random.default_rng(2).standard_normal((8, INPUT_DIM))
    reference_states = _lsm().reservoir.run(x)
    expected = reference_states @ source.rls_readout.Wout + source.rls_readout.bias
    output, _ = target.process_input(x)
    np.testing.assert_allclose(output, expected, rtol=1e-10, atol=1e-12)


def test_untrained_readout_exports_no_arrays():
    state = _lsm().to_state_dict()

    assert state["arrays"] == {}
    assert state["scalars"]["rls_initialized"] is False


def test_restore_rejects_nonfinite_readout(caplog):
    state = _trained().to_state_dict()
    wout = np.array(state["arrays"]["rls_Wout"])
    wout[0, 0] = np.nan
    state["arrays"]["rls_Wout"] = wout

    target = _lsm()
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert not getattr(target.rls_readout, "initialized", False)
    assert any("non-finite" in r.getMessage() for r in caplog.records)


def test_nonfinite_p_is_reinitialized_and_weights_kept(caplog):
    source = _trained()
    state = source.to_state_dict()
    p_matrix = np.array(state["arrays"]["rls_P"])
    p_matrix[1, 1] = np.inf
    state["arrays"]["rls_P"] = p_matrix

    target = _lsm()
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    readout = target.rls_readout
    np.testing.assert_array_equal(readout.Wout, source.rls_readout.Wout)
    np.testing.assert_allclose(readout.P, np.eye(50) / readout.alpha)
    assert any("P" in r.getMessage() for r in caplog.records)


def test_restore_rejects_reservoir_size_mismatch(caplog):
    state = _trained().to_state_dict()

    target = _lsm(reservoir_size=60)
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert not getattr(target.rls_readout, "initialized", False)
    assert any("reservoir_size" in r.getMessage() for r in caplog.records)


def test_restore_rejects_seed_mismatch(caplog):
    state = _trained().to_state_dict()

    target = _lsm(seed=7)
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert not getattr(target.rls_readout, "initialized", False)
    assert any("seed" in r.getMessage() for r in caplog.records)


def test_restore_marks_the_readout_trained():
    source = _trained()
    assert source.readout_trained is True

    target = _lsm()
    x = np.random.default_rng(3).standard_normal((12, INPUT_DIM))
    assert target.readout_trained is False
    assert target.process_subconscious(x)["emotional_valence"] is None

    target.from_state_dict(source.to_state_dict())
    assert target.readout_trained is True
    result = target.process_subconscious(x)
    assert result["emotional_valence"] == pytest.approx(result["readout_mean"])
    assert result["emotional_arousal"] == pytest.approx(result["readout_std"])


def test_rejected_restore_leaves_the_readout_untrained():
    state = _trained().to_state_dict()
    state["scalars"]["seed"] = 999

    target = _lsm()
    target.from_state_dict(state)
    assert target.readout_trained is False


def _ridge_trained(seed: int = 0) -> ProductionLSM:
    rng = np.random.default_rng(seed)
    lsm = _lsm(washout=5)
    inputs = rng.standard_normal((2, 30, INPUT_DIM))
    targets = rng.standard_normal((2, 30, OUTPUT_DIM))
    lsm.train_ridge(inputs, targets)
    return lsm


def test_restore_of_a_never_trained_readout_is_not_trained():
    """A readout that processing initialized but no training ever fitted is
    exported with zero samples; restoring it claims no trained readout and
    reports no affect."""
    source = _lsm()
    x = np.random.default_rng(5).standard_normal((6, INPUT_DIM))
    source.process_input(x)
    state = source.to_state_dict()
    assert state["scalars"]["rls_samples"] == 0.0

    target = _lsm()
    target.from_state_dict(state)

    assert target.readout_trained is False
    result = target.process_subconscious(x)
    assert result["emotional_valence"] is None
    assert result["emotional_arousal"] is None


def test_ridge_readout_round_trip():
    """The ridge readout and the choice of the live readout survive a save and
    a restore; the restored output is the ridge readout's."""
    source = _ridge_trained()
    state = source.to_state_dict()
    assert state["scalars"]["readout"] == "ridge"
    assert state["scalars"]["ridge_fitted"] is True
    assert state["arrays"]["ridge_Wout"].shape == (50, OUTPUT_DIM)
    assert state["arrays"]["ridge_bias"].shape == (OUTPUT_DIM,)

    target = _lsm(washout=5)
    target.from_state_dict(state)

    assert target.readout_trained is True
    assert target.active_readout == "ridge"
    np.testing.assert_array_equal(target.ridge_readout.Wout, source.ridge_readout.Wout)
    np.testing.assert_array_equal(target.ridge_readout.bias, source.ridge_readout.bias)
    x = np.random.default_rng(6).standard_normal((8, INPUT_DIM))
    reference = _lsm().reservoir.run(x)
    expected = reference @ source.ridge_readout.Wout + source.ridge_readout.bias
    result = target.process_subconscious(x)
    np.testing.assert_allclose(result["raw_output"], expected, rtol=1e-10, atol=1e-12)
    assert result["emotional_valence"] == pytest.approx(result["readout_mean"])


def test_rejected_ridge_restore_leaves_the_readout_untrained(caplog):
    """A non-finite ridge readout is refused, and the RLS readout is not put
    on the live path in its place."""
    source = _ridge_trained()
    source.train_rls(
        np.random.default_rng(7).standard_normal((1, 20, INPUT_DIM)),
        np.random.default_rng(8).standard_normal((1, 20, OUTPUT_DIM)),
    )
    source.train_ridge(
        np.random.default_rng(9).standard_normal((2, 30, INPUT_DIM)),
        np.random.default_rng(10).standard_normal((2, 30, OUTPUT_DIM)),
    )
    state = source.to_state_dict()
    assert state["scalars"]["readout"] == "ridge"
    wout = np.array(state["arrays"]["ridge_Wout"])
    wout[0, 0] = np.inf
    state["arrays"]["ridge_Wout"] = wout

    target = _lsm(washout=5)
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert target.readout_trained is False
    assert target.active_readout == "rls"
    assert any("ridge" in r.getMessage() for r in caplog.records)
    x = np.random.default_rng(11).standard_normal((6, INPUT_DIM))
    assert target.process_subconscious(x)["emotional_valence"] is None


def test_unknown_live_readout_is_refused(caplog):
    state = _trained().to_state_dict()
    state["scalars"]["readout"] = "lasso"

    target = _lsm()
    with caplog.at_level(logging.WARNING):
        target.from_state_dict(state)

    assert target.readout_trained is False
    assert any("lasso" in r.getMessage() for r in caplog.records)


def test_state_without_the_readout_key_restores_the_rls_readout():
    """A state saved before the live readout was recorded restores the RLS
    readout as the live one."""
    source = _trained()
    state = source.to_state_dict()
    state["scalars"].pop("readout", None)
    state["scalars"].pop("ridge_fitted", None)

    target = _lsm()
    target.from_state_dict(state)

    assert target.readout_trained is True
    assert target.active_readout == "rls"
    np.testing.assert_array_equal(target.rls_readout.Wout, source.rls_readout.Wout)
