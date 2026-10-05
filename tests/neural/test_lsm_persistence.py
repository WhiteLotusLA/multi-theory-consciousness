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
