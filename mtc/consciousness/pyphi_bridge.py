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

The PyPhi home. At import PyPhi reads ``pyphi_config.yml`` from the working
directory and creates its joblib cache (``__pyphi_cache__``, which also holds
its managed config copy) and its log file there. ``prepare_pyphi_home``
creates ``<DATA_DIRECTORY>/pyphi`` and copies the shipped
``config/pyphi_config.yml`` into it once (presentation and logging keys only:
no welcome banner, no progress bars, console logging at WARNING, no file log
records, no worker pool), after which the copy is the operator's to edit.
``_load_pyphi`` changes into that directory for the first import only, under a
module lock, and restores the working directory in a ``finally``; the import
is the only moment the working directory changes. It runs on the first call
that needs PyPhi (an exact Phi past the tracker's warmup, an assessment, the
doctor's check), which can come after a runtime has started serving. When the
home cannot be prepared the import runs in place and says so in the log.
Zipped installs are not supported (the shipped config is read from the
package directory).

Whose data directory. ``configure_pyphi_home(settings)`` records the Settings
object the first import resolves the home and the shipped config from; it
imports nothing and creates nothing, and a later call before the import
replaces the record (None records the process Settings again).
``build_runtime`` records the runtime's own Settings and the doctor's check
records the Settings it checks, so PyPhi's files stay under that object's
data directory. Without a record the import resolves both from the process
Settings (``get_settings()``) when it runs. The first import decides: once it
has run, ``configure_pyphi_home`` changes nothing and returns the home in use.

Config keys read: ``DATA_DIRECTORY`` (through ``mtc.core.paths``) and
``PROJECT_ROOT``, of the recorded Settings or else the process Settings. Files
written: ``<DATA_DIRECTORY>/pyphi/pyphi_config.yml`` and, by PyPhi, its cache
and log under the same directory; they hold no private data.
"""

import os

os.environ.setdefault("PYPHI_WELCOME_OFF", "1")

import importlib  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

PYPHI_HOME_NAME = "pyphi"
PYPHI_CONFIG_FILE = "pyphi_config.yml"
SHIPPED_PYPHI_CONFIG = "config/pyphi_config.yml"

_pyphi_module: Optional[Any] = None
_pyphi_checked = False
_import_lock = threading.Lock()
# Guarded by _import_lock: the Settings the first import resolves its home
# from (None: the process Settings), and the home that import ran in (None
# before it, or when it ran outside a prepared home).
_home_settings: Optional[Any] = None
_home_in_use: Optional[Path] = None


def pyphi_home(settings: Optional[Any] = None) -> Path:
    """``<DATA_DIRECTORY>/pyphi``, resolved now (nothing is created)."""
    from mtc.core.paths import data_dir

    return data_dir(settings) / PYPHI_HOME_NAME


def prepare_pyphi_home(settings: Optional[Any] = None) -> Path:
    """Create the PyPhi home and copy the shipped ``pyphi_config.yml`` once.

    An existing copy is never overwritten: after the first copy it belongs to
    the operator.
    """
    from mtc.core.paths import ensure_dir
    from mtc.core.resources import read_shipped_text

    home = ensure_dir(pyphi_home(settings))
    target = home / PYPHI_CONFIG_FILE
    if not target.exists():
        text, _ = read_shipped_text(SHIPPED_PYPHI_CONFIG, settings=settings)
        target.write_text(text, encoding="utf-8")
    return home


def configure_pyphi_home(settings: Optional[Any] = None) -> Optional[Path]:
    """Record whose ``DATA_DIRECTORY`` holds PyPhi's home at the first import.

    Before PyPhi's first import this records ``settings`` (None: the process
    Settings, read when the import runs) and returns ``<DATA_DIRECTORY>/pyphi``
    resolved from them; nothing is imported or created, and a later call
    replaces the record. The first import decides: once it has run, or when
    PyPhi was imported before the bridge loaded it, this changes nothing and
    returns the home in use, None when the import ran outside a prepared home.
    """
    global _home_settings
    with _import_lock:
        if _pyphi_checked:
            return _home_in_use
        if "pyphi" in sys.modules:
            return None
        home = pyphi_home(settings)
        _home_settings = settings
        return home


def _import_pyphi() -> Any:
    with warnings.catch_warnings():
        # PyPhi's import pulls cached arrays through pickle and trips numpy
        # deprecation warnings; they are noise at this boundary.
        warnings.simplefilter("ignore")
        module = importlib.import_module("pyphi")
        importlib.import_module("pyphi.new_big_phi")
    return module


def _load_pyphi() -> Optional[Any]:
    """Import PyPhi once and cache the module (None when unavailable).

    The import runs in the home of the recorded Settings, or of the process
    Settings when none was recorded (``configure_pyphi_home``).
    """
    global _pyphi_module, _pyphi_checked, _home_in_use
    if _pyphi_checked:
        return _pyphi_module
    with _import_lock:
        if _pyphi_checked:
            return _pyphi_module
        home: Optional[Path] = None
        if "pyphi" not in sys.modules:
            try:
                home = prepare_pyphi_home(_home_settings)
            except Exception as exc:  # a read-only data directory, for example
                logger.warning(
                    "PyPhi home could not be prepared (%s); PyPhi writes its "
                    "cache in the working directory",
                    type(exc).__name__,
                )
        _home_in_use = home
        previous = os.getcwd()
        try:
            if home is not None:
                os.chdir(home)
            module = _import_pyphi()
            # Keep a later PyPhi reconfiguration from replacing root handlers.
            from mtc.core.logger import neutralize_pyphi_logging

            neutralize_pyphi_logging()
            _pyphi_module = module
        except Exception as exc:  # ImportError or a broken install
            logger.debug("PyPhi unavailable: %s", exc)
            _pyphi_module = None
        finally:
            os.chdir(previous)
            _pyphi_checked = True
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
