"""
Consciousness State Persistence Hub
===================================

Saves and restores the state of every registered consciousness module
across restarts. The modules behind the eight theories (GWT, IIT, AST, HOT,
FEP, RPT, BLT, and agency and embodiment), the Damasio layers and the
neural substrates accumulate counters, learned weights and histories that
would otherwise be lost on every restart.

Module contract:
  to_state_dict() -> {"scalars": {...}, "arrays": {name: np.ndarray},
                      "histories": {name: list}}
  from_state_dict(state) -> None

Each module rehydrates its own types (dataclasses, enums, deques with their
configured maxlen). The hub hands back plain JSON values and numpy arrays
and never wraps them in attribute-access shims.

Tiered save strategy:
  - Scalars (counters, levels, scores): every cycle
  - Arrays (numpy matrices, learned weights): every ``arrays_every`` cycles
  - Histories (event logs, measurement records): on demand via save_all(),
    which a caller runs at shutdown and at whatever interval it chooses

Storage layout (JSON plus compressed numpy, no pickle anywhere):
  <state_dir>/
    manifest.json             version, timestamp, module list
    <module>_scalars.json     per-module scalars
    <module>_arrays.npz       per-module numpy arrays
    <module>_histories.json   per-module histories

Every file is written to a temporary name, fsynced, then renamed over the
live file, so a crash or power loss leaves either the old or the new file,
never a partial one. A manifest whose version differs from STATE_VERSION
means start fresh: the stale files are moved into a ``stale-v<N>-<time>``
folder so a later partial save cannot mix two layouts.

NaN visibility: a module whose scalars contain NaN or infinity is logged
and skipped (its last good files stay in place), JSON is written with
non-finite numbers refused, non-finite arrays are named in the log on
write and on read, and a scalars file that holds NaN is refused on restore.

Privacy note: histories hold workspace content summaries, HOT text and
message fragments. Treat the state directory as private data. The status
shows the directory relative to ``PROJECT_ROOT`` (or by its name alone when it
lies elsewhere), never as an absolute path. A manager built with a Settings
object reads that object alone, never the process Settings.
"""

import io
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from mtc.core.config import Settings
from mtc.core.paths import display_path, state_dir as default_state_dir

logger = logging.getLogger(__name__)

STATE_VERSION = 1
MANIFEST_NAME = "manifest.json"
TIERS = ("scalars", "arrays", "histories")

_MODULE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _fsync(fd: int) -> None:
    """Flush a file descriptor to disk (a seam the tests replace)."""
    os.fsync(fd)


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory so a rename is durable."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _has_nonfinite(value: Any) -> bool:
    """True when a nested scalar payload holds NaN or infinity."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_has_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(v) for v in value)
    return False


def _json_default(obj: Any) -> Any:
    """Serialize numpy scalars, arrays and sets; refuse anything else."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token} in persisted state")


class ConsciousnessStateManager:
    """Persist registered modules across restarts.

    Args:
        state_dir: Directory for the state files; defaults to the
            ``STATE_DIR`` of ``settings``, resolved under its ``PROJECT_ROOT``
            like every other path key. Created on the first save.
        modules: Optional mapping of name to module to register now.
        arrays_every: Save the arrays tier every N completed cycles.
        settings: The Settings object that supplies the default state
            directory and the project root the status shows the directory
            against; None reads the process Settings when one is needed.
    """

    def __init__(
        self,
        state_dir: Optional[Union[str, Path]] = None,
        modules: Optional[Dict[str, Any]] = None,
        arrays_every: int = 10,
        settings: Optional[Settings] = None,
    ):
        if arrays_every < 1:
            raise ValueError(f"arrays_every must be at least 1, got {arrays_every}")
        self._settings = settings
        if state_dir is None:
            state_dir = default_state_dir(settings)
        self.state_dir = Path(state_dir)
        self.arrays_every = int(arrays_every)

        self._modules: Dict[str, Any] = {}
        self._cycle_count = 0
        self._cycles_since_arrays = 0
        self._total_saves = 0
        self._last_full_save: Optional[str] = None

        for name, module in (modules or {}).items():
            self.register(name, module)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, module: Any) -> None:
        """Register a module that implements the persistence contract."""
        if not _MODULE_NAME.fullmatch(name):
            raise ValueError(
                f"module name {name!r} must be letters, digits, '_', '.' or '-' "
                "and start with a letter or digit"
            )
        if not (
            callable(getattr(module, "to_state_dict", None))
            and callable(getattr(module, "from_state_dict", None))
        ):
            raise TypeError(
                f"module {name!r} must implement to_state_dict() and "
                "from_state_dict(state)"
            )
        self._modules[name] = module
        logger.debug(f"State persistence: registered {name!r}")

    @property
    def modules(self) -> Dict[str, Any]:
        """A copy of the registered name to module mapping."""
        return dict(self._modules)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def on_cycle_complete(self) -> Dict[str, Any]:
        """Run after each consciousness cycle.

        Writes scalars for every module, and arrays every ``arrays_every``
        cycles. Histories are left to save_all().

        Returns:
            Per-module list of tiers written, or the reason nothing was.
        """
        self._cycle_count += 1
        self._cycles_since_arrays += 1
        save_arrays = self._cycles_since_arrays >= self.arrays_every
        tiers = ("scalars", "arrays") if save_arrays else ("scalars",)

        results = {
            name: self._save_module(name, module, tiers)
            for name, module in self._modules.items()
        }
        if save_arrays:
            self._cycles_since_arrays = 0
        if save_arrays or not (self.state_dir / MANIFEST_NAME).exists():
            self._write_manifest()
        return results

    def save_all(self) -> Dict[str, Any]:
        """Write every tier for every module. Call on shutdown or on demand.

        Returns:
            {"modules": {name: tiers written or reason}, "elapsed_ms": float}
        """
        t0 = time.monotonic()
        results = {
            name: self._save_module(name, module, TIERS)
            for name, module in self._modules.items()
        }
        self._cycles_since_arrays = 0
        self._total_saves += 1
        self._last_full_save = datetime.now(timezone.utc).isoformat()
        self._write_manifest()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            f"Consciousness state saved: {len(results)} modules in "
            f"{elapsed_ms:.1f}ms"
        )
        return {"modules": results, "elapsed_ms": elapsed_ms}

    def _save_module(self, name: str, module: Any, tiers: tuple) -> Any:
        try:
            state = module.to_state_dict() or {}
        except Exception as exc:
            logger.error(f"State export failed for {name!r}: {exc}", exc_info=True)
            return f"error: {exc}"

        scalars = state.get("scalars") or {}
        if _has_nonfinite(scalars):
            logger.warning(
                f"State for {name!r} skipped: its scalars contain NaN or "
                "infinity; the last good files are kept"
            )
            return "skipped: non-finite scalars"

        written: List[str] = []
        try:
            for tier in tiers:
                data = state.get(tier) or {}
                if not data:
                    # A tier the module no longer exports must not leave a
                    # stale file behind for restore to rehydrate
                    self._tier_path(name, tier).unlink(missing_ok=True)
                    continue
                if tier == "arrays":
                    self._write_npz(name, data)
                else:
                    self._write_json(name, tier, data)
                written.append(tier)
        except Exception as exc:
            logger.error(f"State save failed for {name!r}: {exc}", exc_info=True)
            return f"error: {exc}"
        return written

    # ------------------------------------------------------------------
    # Restoring
    # ------------------------------------------------------------------

    def restore_all(self) -> Dict[str, bool]:
        """Restore every registered module from disk. Call on startup.

        Returns:
            Per-module success. False means the module starts fresh: no
            saved state, a version mismatch, an unreadable or non-finite
            file, or a from_state_dict() that raised.
        """
        results = {name: False for name in self._modules}
        manifest = self._read_manifest()
        if manifest is None:
            logger.info("No saved consciousness state; starting fresh")
            return results

        saved_version = manifest.get("version")
        if saved_version != STATE_VERSION:
            archive = self._archive_stale(saved_version)
            logger.warning(
                f"State version mismatch (saved={saved_version}, "
                f"current={STATE_VERSION}); starting fresh, stale files moved "
                f"to {archive}"
            )
            return results

        self._cycle_count = int(manifest.get("cycle_count", 0))
        for name, module in self._modules.items():
            state = self._load_module_state(name)
            if state is None:
                continue
            try:
                module.from_state_dict(state)
            except Exception as exc:
                logger.error(f"State restore failed for {name!r}: {exc}", exc_info=True)
                continue
            results[name] = True

        restored = sum(results.values())
        logger.info(
            f"Consciousness state restored: {restored}/{len(results)} modules "
            f"(saved {manifest.get('timestamp', 'unknown')})"
        )
        return results

    def _load_module_state(self, name: str) -> Optional[Dict[str, Any]]:
        state: Dict[str, Any] = {}
        for tier in ("scalars", "histories"):
            path = self._tier_path(name, tier)
            if not path.exists():
                continue
            try:
                state[tier] = json.loads(
                    path.read_text(), parse_constant=_reject_constant
                )
            except ValueError as exc:
                logger.warning(f"State for {name!r} not restored: {path.name}: {exc}")
                return None
            except OSError as exc:
                logger.error(f"State for {name!r} unreadable: {path.name}: {exc}")
                return None

        arrays_path = self._tier_path(name, "arrays")
        if arrays_path.exists():
            try:
                with np.load(arrays_path, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
            except Exception as exc:
                logger.error(
                    f"State for {name!r} unreadable: {arrays_path.name}: {exc}"
                )
                return None
            self._warn_nonfinite(name, arrays, "loaded from")
            state["arrays"] = arrays

        return state or None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _tier_path(self, name: str, tier: str) -> Path:
        suffix = "npz" if tier == "arrays" else "json"
        return self.state_dir / f"{name}_{tier}.{suffix}"

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                _fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        _fsync_dir(self.state_dir)

    def _write_json(self, name: str, tier: str, data: Dict[str, Any]) -> None:
        # Serialize first: a failure here writes nothing at all
        payload = json.dumps(
            data, default=_json_default, allow_nan=False, separators=(",", ":")
        )
        self._atomic_write(self._tier_path(name, tier), payload.encode("utf-8"))

    def _write_npz(self, name: str, arrays: Dict[str, Any]) -> None:
        clean: Dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            arr = np.asarray(value)
            if arr.dtype.kind == "O":
                raise TypeError(
                    f"{name}.{key}: object arrays need pickle and are refused"
                )
            clean[str(key)] = arr
        self._warn_nonfinite(name, clean, "entering")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **clean)
        self._atomic_write(self._tier_path(name, "arrays"), buffer.getvalue())

    @staticmethod
    def _warn_nonfinite(name: str, arrays: Dict[str, np.ndarray], direction: str):
        for key, arr in arrays.items():
            if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
                bad = int(arr.size - np.count_nonzero(np.isfinite(arr)))
                logger.warning(
                    f"{name}.{key}: {bad} non-finite values {direction} persisted "
                    "state"
                )

    def _manifest(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_count": self._cycle_count,
            "total_saves": self._total_saves,
            "last_full_save": self._last_full_save,
            "modules": sorted(self._modules),
            "arrays_every": self.arrays_every,
        }

    def _write_manifest(self) -> None:
        try:
            payload = json.dumps(self._manifest(), indent=2, allow_nan=False)
            self._atomic_write(self.state_dir / MANIFEST_NAME, payload.encode())
        except Exception as exc:
            logger.error(f"State manifest write failed: {exc}", exc_info=True)

    def _read_manifest(self) -> Optional[Dict[str, Any]]:
        path = self.state_dir / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            logger.error(f"State manifest unreadable: {exc}")
            return None
        return data if isinstance(data, dict) else None

    def _archive_stale(self, saved_version: Any) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive = self.state_dir / f"stale-v{saved_version}-{stamp}"
        archive.mkdir(parents=True, exist_ok=True)
        for path in self.state_dir.iterdir():
            if path.is_file() and (
                path.name == MANIFEST_NAME
                or path.name.endswith(
                    ("_scalars.json", "_histories.json", "_arrays.npz")
                )
            ):
                os.replace(path, archive / path.name)
        return archive

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Current persistence status for monitoring; the state directory is
        shown relative to the project root, never as an absolute path."""
        manifest = self._read_manifest()
        return {
            "registered_modules": sorted(self._modules),
            "cycle_count": self._cycle_count,
            "total_saves": self._total_saves,
            "last_full_save": self._last_full_save,
            "arrays_every": self.arrays_every,
            "state_dir": display_path(self.state_dir, settings=self._settings),
            "has_saved_state": manifest is not None,
            "saved_at": manifest.get("timestamp") if manifest else None,
        }
