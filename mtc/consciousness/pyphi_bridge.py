"""
PyPhi bridge: the single boundary between MTC's conventions and PyPhi's.

MTC builds state-by-node transition probability matrices with big-endian row
order (node 0 is the most significant bit of the row index) and stores
connectivity as target by source (``cm[i][j] == 1`` means node j drives node
i). PyPhi expects little-endian rows (node 0 is the least significant bit)
and a source-by-target matrix (``cm[i][j] == 1`` means node i drives node j).

Handing PyPhi MTC's arrays unconverted silently scrambles which state each
row conditions on and reverses every edge, so every PyPhi construction in MTC
goes through this module. No other module builds a ``pyphi.Network``.

A state tuple needs no conversion: element i is node i's state in both
conventions. Only the row index derived from it differs.

PyPhi is optional (the ``iit`` extra). Importing this module never imports
PyPhi; the library loads lazily on the first call that needs it.
"""

import os

os.environ.setdefault("PYPHI_WELCOME_OFF", "1")

import importlib  # noqa: E402
import logging  # noqa: E402
import warnings  # noqa: E402
from typing import Any, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_pyphi_module: Optional[Any] = None
_pyphi_checked = False


def _load_pyphi() -> Optional[Any]:
    """Import PyPhi once and cache the module (None when unavailable)."""
    global _pyphi_module, _pyphi_checked
    if not _pyphi_checked:
        _pyphi_checked = True
        try:
            with warnings.catch_warnings():
                # PyPhi's import pulls cached arrays through pickle and trips
                # numpy deprecation warnings; they are noise at this boundary.
                warnings.simplefilter("ignore")
                module = importlib.import_module("pyphi")
                importlib.import_module("pyphi.new_big_phi")
            _pyphi_module = module
        except Exception as exc:  # ImportError or a broken install
            logger.debug("PyPhi unavailable: %s", exc)
            _pyphi_module = None
    return _pyphi_module


def pyphi_available() -> bool:
    """True when PyPhi imports cleanly in this interpreter."""
    return _load_pyphi() is not None


def require_pyphi() -> Any:
    """Return the PyPhi module or raise ImportError with the install hint."""
    module = _load_pyphi()
    if module is None:
        raise ImportError(
            "PyPhi is not installed; install the 'iit' extra to compute exact Phi"
        )
    return module


# ============================================================================
# Convention converters (pure numpy, no PyPhi needed)
# ============================================================================


def reverse_bits(index: int, n_bits: int) -> int:
    """Reverse the low ``n_bits`` bits of ``index``.

    Big-endian and little-endian indices of the same state are bit reversals
    of each other, and the reversal is its own inverse.
    """
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (index & 1)
        index >>= 1
    return result


def _n_nodes_for(tpm: np.ndarray) -> int:
    """Validate a 2-D state-by-node TPM and return its node count."""
    if tpm.ndim != 2:
        raise ValueError(f"expected a 2-D state-by-node TPM, got shape {tpm.shape}")
    n_rows, n_nodes = tpm.shape
    if n_nodes < 1 or n_rows != 2**n_nodes:
        raise ValueError(
            f"state-by-node TPM needs 2**n rows for n nodes, got shape {tpm.shape}"
        )
    return n_nodes


def to_little_endian_rows(tpm_big_endian: np.ndarray) -> np.ndarray:
    """Reorder a state-by-node TPM's rows from big-endian to little-endian.

    Row ``reverse_bits(b)`` of the result is row ``b`` of the input. Because
    bit reversal is an involution, applying this twice returns the input,
    and the same call converts little-endian rows back to big-endian.
    """
    tpm = np.asarray(tpm_big_endian, dtype=np.float64)
    n_nodes = _n_nodes_for(tpm)
    permutation = np.array(
        [reverse_bits(i, n_nodes) for i in range(tpm.shape[0])], dtype=np.intp
    )
    return tpm[permutation]


def to_pyphi_cm(cm_target_by_source: np.ndarray) -> np.ndarray:
    """Transpose a target-by-source connectivity matrix to PyPhi's layout."""
    cm = np.asarray(cm_target_by_source)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"connectivity must be square, got shape {cm.shape}")
    return np.ascontiguousarray(cm.T).astype(np.int64)


# ============================================================================
# PyPhi constructions
# ============================================================================


def build_network(
    tpm_big_endian: np.ndarray,
    cm_target_by_source: Optional[np.ndarray],
    labels: Sequence[str],
) -> Any:
    """Build a ``pyphi.Network`` from arrays in MTC's conventions.

    Args:
        tpm_big_endian: state-by-node TPM, shape ``(2**n, n)``, big-endian rows.
        cm_target_by_source: ``(n, n)`` connectivity, row = target, column =
            source; None means fully connected (PyPhi's own default).
        labels: one label per node, in node order.

    Raises:
        ImportError: PyPhi is not installed.
        ValueError: the shapes or label count disagree.
    """
    pyphi = require_pyphi()
    tpm_le = to_little_endian_rows(tpm_big_endian)
    n_nodes = tpm_le.shape[1]
    cm = None
    if cm_target_by_source is not None:
        cm = to_pyphi_cm(cm_target_by_source)
        if cm.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"connectivity shape {cm.shape} does not match {n_nodes} nodes"
            )
    node_labels = tuple(str(label) for label in labels)
    if len(node_labels) != n_nodes:
        raise ValueError(f"{len(node_labels)} labels for {n_nodes} nodes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pyphi.Network(tpm=tpm_le, cm=cm, node_labels=node_labels)


def build_subsystem(network: Any, state: Sequence[int]) -> Any:
    """Build a ``pyphi.Subsystem`` over the whole network at ``state``."""
    pyphi = require_pyphi()
    state_tuple: Tuple[int, ...] = tuple(int(bit) for bit in state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pyphi.Subsystem(network, state_tuple)


def system_irreducibility(subsystem: Any) -> Any:
    """Run the IIT 4.0 system irreducibility analysis (``new_big_phi.sia``).

    Progress bars are switched off for the call only; PyPhi's global
    configuration is left as it was.
    """
    pyphi = require_pyphi()
    new_big_phi = importlib.import_module("pyphi.new_big_phi")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pyphi.config.override(PROGRESS_BARS=False):
            return new_big_phi.sia(subsystem)
