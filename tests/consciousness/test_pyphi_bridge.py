"""
Tests for the PyPhi bridge.

MTC stores state-by-node TPMs with big-endian row order and connectivity as
target by source. PyPhi wants little-endian rows and a source-by-target
matrix. These tests check the conversions against PyPhi's own converters, not
against a hand-derived expectation, and show that the conversion changes the
answer on a network whose SIA is known.
"""

import importlib

import numpy as np
import pytest

from mtc.consciousness import pyphi_bridge
from mtc.consciousness.pyphi_bridge import (
    build_network,
    pyphi_available,
    reverse_bits,
    to_little_endian_rows,
    to_pyphi_cm,
)


def _random_big_endian_tpm(n_nodes: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((2**n_nodes, n_nodes))


# ============================================================================
# Row order
# ============================================================================


def test_bridge_rows_match_pyphi_little_endian():
    """Row for state s lands at PyPhi's own little-endian index of s."""
    pytest.importorskip("pyphi")
    from pyphi import convert

    n = 3
    tpm_be = _random_big_endian_tpm(n)
    expected = np.empty_like(tpm_be)
    for be_index in range(2**n):
        state = convert.be_index2state(be_index, n)
        # MTC's row index is PyPhi's big-endian index of the same state.
        assert convert.state2be_index(state) == be_index
        expected[convert.state2le_index(state)] = tpm_be[be_index]

    np.testing.assert_array_equal(to_little_endian_rows(tpm_be), expected)


def test_bridge_rows_match_pyphi_little_endian_five_nodes():
    """The same holds at the tracker's real size (5 nodes, 32 rows)."""
    pytest.importorskip("pyphi")
    from pyphi import convert

    n = 5
    tpm_be = _random_big_endian_tpm(n, seed=11)
    expected = np.empty_like(tpm_be)
    for be_index in range(2**n):
        state = convert.be_index2state(be_index, n)
        expected[convert.state2le_index(state)] = tpm_be[be_index]

    np.testing.assert_array_equal(to_little_endian_rows(tpm_be), expected)


def test_bit_reversal_is_an_involution():
    """Reversing twice is the identity, for indices and for whole TPMs."""
    for n_bits in range(1, 7):
        for index in range(2**n_bits):
            assert reverse_bits(reverse_bits(index, n_bits), n_bits) == index

    tpm = _random_big_endian_tpm(4, seed=3)
    np.testing.assert_array_equal(
        to_little_endian_rows(to_little_endian_rows(tpm)), tpm
    )


def test_reverse_bits_examples():
    assert reverse_bits(0b001, 3) == 0b100
    assert reverse_bits(0b110, 3) == 0b011
    assert reverse_bits(0b10000, 5) == 0b00001
    assert reverse_bits(0, 1) == 0


def test_rows_reject_wrong_shape():
    with pytest.raises(ValueError):
        to_little_endian_rows(np.zeros((6, 3)))
    with pytest.raises(ValueError):
        to_little_endian_rows(np.zeros(8))


# ============================================================================
# Connectivity
# ============================================================================


def test_bridge_transposes_connectivity():
    """Target-by-source in, source-by-target out."""
    cm_target_by_source = np.array(
        [
            [1, 1, 1],  # node 0 receives from 0, 1, 2
            [1, 1, 0],  # node 1 receives from 0, 1
            [0, 0, 1],  # node 2 receives from itself only
        ]
    )
    cm = to_pyphi_cm(cm_target_by_source)
    np.testing.assert_array_equal(cm, cm_target_by_source.T)
    # PyPhi reads cm[i][j] == 1 as "i drives j": node 2 drives node 0 ...
    assert cm[2][0] == 1
    # ... but node 2 does not drive node 1.
    assert cm[2][1] == 0


def test_connectivity_rejects_non_square():
    with pytest.raises(ValueError):
        to_pyphi_cm(np.ones((2, 3)))


# ============================================================================
# Network construction
# ============================================================================


def test_build_network_smoke():
    """A 3-node random TPM builds a pyphi.Network with the given labels."""
    pyphi = pytest.importorskip("pyphi")

    tpm_be = _random_big_endian_tpm(3)
    cm = np.ones((3, 3), dtype=int)
    network = build_network(tpm_be, cm, ["A", "B", "C"])

    assert isinstance(network, pyphi.Network)
    assert network.size == 3
    assert tuple(network.node_labels) == ("A", "B", "C")


def test_network_conditionals_match_big_endian_rows():
    """PyPhi's own indexing of the built network recovers MTC's rows."""
    pytest.importorskip("pyphi")
    from pyphi import convert

    n = 3
    tpm_be = _random_big_endian_tpm(n, seed=5)
    cm_target_by_source = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    network = build_network(tpm_be, cm_target_by_source, ["A", "B", "C"])

    multidimensional = np.asarray(network.tpm)
    for be_index in range(2**n):
        state = convert.be_index2state(be_index, n)
        np.testing.assert_allclose(multidimensional[state], tpm_be[be_index])
    np.testing.assert_array_equal(network.cm, cm_target_by_source.T)


def test_known_sia_survives_the_round_trip():
    """A network with a known nonzero SIA gives the same phi through the bridge.

    The example network ships in PyPhi's conventions. Converting it into MTC's
    conventions and back through the bridge must reproduce PyPhi's own phi,
    and skipping the row conversion must not.
    """
    pytest.importorskip("pyphi")
    from pyphi import convert, examples

    reference = examples.basic_noisy_selfloop_network()
    state = (0, 0, 0)
    expected_phi = float(
        pyphi_bridge.system_irreducibility(
            pyphi_bridge.build_subsystem(reference, state)
        ).phi
    )
    assert expected_phi > 0.0

    tpm_le = np.asarray(convert.to_2dimensional(np.asarray(reference.tpm)))
    # The involution takes little-endian rows to big-endian rows as well.
    tpm_be = to_little_endian_rows(tpm_le)
    cm_target_by_source = np.asarray(reference.cm).T

    rebuilt = build_network(tpm_be, cm_target_by_source, ["A", "B", "C"])
    phi = float(
        pyphi_bridge.system_irreducibility(
            pyphi_bridge.build_subsystem(rebuilt, state)
        ).phi
    )
    assert phi == pytest.approx(expected_phi, abs=1e-9)

    # Handing PyPhi the big-endian rows unconverted (the v0.4.0 defect) gives a
    # different answer. The bridge always converts, so pre-converting once makes
    # it pass the big-endian rows through as they are.
    unconverted = build_network(
        to_little_endian_rows(tpm_be), cm_target_by_source, ["A", "B", "C"]
    )
    wrong_phi = float(
        pyphi_bridge.system_irreducibility(
            pyphi_bridge.build_subsystem(unconverted, state)
        ).phi
    )
    assert wrong_phi != pytest.approx(expected_phi, abs=1e-6)


def test_build_network_rejects_mismatched_sizes():
    pytest.importorskip("pyphi")
    with pytest.raises(ValueError):
        build_network(_random_big_endian_tpm(3), np.ones((2, 2)), ["A", "B", "C"])
    with pytest.raises(ValueError):
        build_network(_random_big_endian_tpm(3), np.ones((3, 3)), ["A", "B"])


# ============================================================================
# Availability
# ============================================================================


def test_available_flag_matches_import():
    try:
        importlib.import_module("pyphi")
        importable = True
    except ImportError:
        importable = False
    assert pyphi_available() is importable


def test_require_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(pyphi_bridge, "pyphi_available", lambda: False)
    monkeypatch.setattr(pyphi_bridge, "_load_pyphi", lambda: None)
    with pytest.raises(ImportError):
        pyphi_bridge.require_pyphi()
