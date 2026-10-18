"""
Provider Registry
=================

Holds the registered providers and which one is active. There is no module
singleton: the brain owns its registry instance.

``set_active`` persists the choice to the provider state file so status
pages and the CLI can report it; ``load`` reads that record back and returns
the id, but never changes the active provider (``LLM_PROVIDER`` decides which
provider the brain uses, and a failing provider is reported, never swapped
for another). The file is written only when saved, and its directory is
created then.

Config keys read (at call time, when no explicit path is given):
``PROVIDER_STATE_PATH`` (empty derives ``<DATA_DIRECTORY>/provider_state.json``),
resolved against ``PROJECT_ROOT``. Persisted state: the active provider id and
the time it was saved. No secret is written.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mtc.brain.providers.base import ModelInfo
from mtc.core.config import Settings, get_settings
from mtc.core.paths import resolve_path

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderRegistry:
    """Registered providers by id, the active choice and its saved record."""

    def __init__(
        self,
        state_path: Optional[Path] = None,
        *,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._explicit_path = Path(state_path) if state_path is not None else None
        self._settings = settings
        self._clock = clock or _now
        self._providers: Dict[str, Any] = {}
        self._active_id: Optional[str] = None

    @property
    def state_path(self) -> Path:
        """Where the active choice is saved, resolved now."""
        if self._explicit_path is not None:
            return self._explicit_path
        active = self._settings if self._settings is not None else get_settings()
        return resolve_path(active.PROVIDER_STATE_PATH, settings=active)

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def register(self, provider: Any) -> None:
        """Add a provider under its ``model_info().id``; a same id replaces it."""
        provider_id = provider.model_info().id
        if provider_id in self._providers:
            logger.warning("Provider %s registered again; replacing it", provider_id)
        self._providers[provider_id] = provider

    def unregister(self, provider_id: str) -> None:
        """Remove a provider; removing the active one leaves none active."""
        self._providers.pop(provider_id, None)
        if self._active_id == provider_id:
            self._active_id = None

    def get(self, provider_id: str) -> Any:
        if provider_id not in self._providers:
            raise KeyError(f"Unknown provider: {provider_id}")
        return self._providers[provider_id]

    def set_active(self, provider_id: str) -> None:
        """Make a registered provider active and save the choice."""
        if provider_id not in self._providers:
            raise KeyError(f"Unknown provider: {provider_id}")
        self._active_id = provider_id
        self.save()

    def active(self) -> Any:
        """The active provider; raises ``RuntimeError`` when none is set."""
        if self._active_id is None:
            raise RuntimeError("No active provider is set")
        return self._providers[self._active_id]

    def list_available(self) -> List[ModelInfo]:
        """Each registered provider's ``ModelInfo``, in registration order."""
        return [provider.model_info() for provider in self._providers.values()]

    def save(self) -> Path:
        """Write the active choice; creates the directory when needed."""
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "active_provider": self._active_id,
            "saved_at": self._clock().isoformat(),
        }
        handle, temporary = tempfile.mkstemp(
            prefix=".provider_state.", dir=str(path.parent)
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        return path

    def load(self) -> Optional[str]:
        """The saved active id when it is registered; None otherwise."""
        path = self.state_path
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # The file name and the key only: the directory can hold a home
            # path or a user name.
            logger.warning(
                "provider state file %s (PROVIDER_STATE_PATH) is unreadable (%s)",
                path.name,
                type(exc).__name__,
            )
            return None
        provider_id = (
            record.get("active_provider") if isinstance(record, dict) else None
        )
        if provider_id is None:
            return None
        if provider_id not in self._providers:
            logger.warning(
                "Saved provider %s is not registered; ignoring the saved choice",
                provider_id,
            )
            return None
        return str(provider_id)
