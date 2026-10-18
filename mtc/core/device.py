"""
Device Switch
=============

``MTC_DEVICE`` is the only switch that picks a compute device. Every module
that builds a torch device or passes a device to an encoder asks this module
(or the device manager in ``mtc.neural.mps_utils``, which asks it).

Choices (``Settings.DEVICE``): ``auto``, ``cpu``, ``mps`` and ``cuda``.

- ``auto`` prefers MPS, then CUDA, then CPU. Without torch it is ``cpu``.
- ``cpu`` is always available by name; ``resolve_device`` additionally needs
  torch to build the ``torch.device``.
- ``mps`` and ``cuda`` raise ``DeviceUnavailableError`` when the backend is
  not available; the error names the setting. No explicit device falls back
  silently.

An argument overrides the setting. Torch is imported lazily, inside functions,
so importing this module loads no model library. Each distinct resolution
(device and source) is logged once at INFO.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

from mtc.core.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = logging.getLogger(__name__)

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")

_LOGGED: Set[Tuple[str, str]] = set()


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested device is not available on this machine."""


def _import_torch() -> Optional[Any]:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _mps_available(torch_module: Any) -> bool:
    backend = getattr(torch_module.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available() and backend.is_built())


def _cuda_available(torch_module: Any) -> bool:
    return bool(torch_module.cuda.is_available())


def _choose(preference: Optional[str], settings: Optional[Settings]) -> Tuple[str, str]:
    """Return ``(device name, source)``; raise when an explicit device is absent."""
    if preference is not None:
        requested, source = preference, "argument"
    else:
        active = settings if settings is not None else get_settings()
        requested = active.DEVICE
        source = "setting" if requested != "auto" else "auto"
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"device {requested!r} is not one of {DEVICE_CHOICES}")

    # The cpu choice needs no backend probe, so it never loads torch.
    if requested == "cpu":
        return "cpu", source
    torch_module = _import_torch()
    if requested == "auto":
        if torch_module is not None:
            if _mps_available(torch_module):
                return "mps", "auto"
            if _cuda_available(torch_module):
                return "cuda", "auto"
        return "cpu", "auto"

    present = False
    if torch_module is not None:
        present = (
            _mps_available(torch_module)
            if requested == "mps"
            else _cuda_available(torch_module)
        )
    if not present:
        reason = "torch is not installed" if torch_module is None else "not available"
        raise DeviceUnavailableError(
            f"device {requested!r} was requested ({source}) but is {reason} "
            "on this machine; set MTC_DEVICE=auto or MTC_DEVICE=cpu"
        )
    return requested, source


def _log_once(name: str, source: str) -> None:
    key = (name, source)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info("compute device %s (source: %s)", name, source)


def resolve_device_name(
    preference: Optional[str] = None, *, settings: Optional[Settings] = None
) -> str:
    """Return ``cpu``, ``mps`` or ``cuda`` for the argument or ``MTC_DEVICE``."""
    name, source = _choose(preference, settings)
    _log_once(name, source)
    return name


def resolve_device(
    preference: Optional[str] = None, *, settings: Optional[Settings] = None
) -> "torch.device":
    """Return the ``torch.device`` for the argument or ``MTC_DEVICE``."""
    name = resolve_device_name(preference, settings=settings)
    torch_module = _import_torch()
    if torch_module is None:
        raise DeviceUnavailableError(
            "torch is not installed; install the 'neural' extra to build a "
            "torch device"
        )
    return torch_module.device(name)


def describe_device(settings: Optional[Settings] = None) -> Dict[str, str]:
    """Return ``{"name": ..., "source": ...}`` for the configured device.

    ``source`` is ``setting`` for an explicit ``MTC_DEVICE`` and ``auto`` when
    the order MPS, CUDA, CPU chose it.
    """
    name, source = _choose(None, settings)
    return {"name": name, "source": source}
