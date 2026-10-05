"""
Knowledge Graph Connection Settings
===================================

``Neo4jConfig`` holds the four values the graph manager needs to reach Neo4j:
the Bolt URI, the user, the password and the database name. ``from_settings``
reads ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD`` and ``NEO4J_DATABASE``
at call time; nothing is read at import and no module-level default instance
exists, so a changed setting reaches the next config built.

An empty password is allowed (an unauthenticated development server) and is
logged once at WARNING per URI and user. ``masked()`` is the form to log: the
password is replaced by ``***``. ``create_driver()`` imports the optional
``neo4j`` package through ``require_client``, so a missing package raises an
ImportError naming the ``neo4j`` extra; building a config never opens a
connection. No persisted state.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from mtc.core.config import Settings, get_settings
from mtc.memory.clients import require_client

logger = logging.getLogger(__name__)

_WARNED: Set[Tuple[str, str]] = set()


@dataclass(frozen=True)
class Neo4jConfig:
    """Bolt URI, user, password and database of the knowledge graph."""

    uri: str
    user: str
    password: str = field(default="", repr=False)
    database: str = "neo4j"

    def __post_init__(self) -> None:
        if not self.password:
            key = (self.uri, self.user)
            if key not in _WARNED:
                _WARNED.add(key)
                logger.warning(
                    "NEO4J_PASSWORD is empty for %s as %s; a server with "
                    "authentication enabled will refuse the connection",
                    self.uri,
                    self.user,
                )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "Neo4jConfig":
        """Build the config from the ``NEO4J_*`` keys, read now."""
        active = settings if settings is not None else get_settings()
        return cls(
            uri=active.NEO4J_URI,
            user=active.NEO4J_USER,
            password=active.NEO4J_PASSWORD,
            database=active.NEO4J_DATABASE,
        )

    def masked(self) -> Dict[str, str]:
        """Return the connection details with the password masked."""
        return {
            "uri": self.uri,
            "user": self.user,
            "database": self.database,
            "password": "***" if self.password else "",
        }

    def create_driver(self, **overrides: Any) -> Any:
        """Build a neo4j driver (it connects lazily); ``overrides`` pass through."""
        neo4j = require_client("neo4j", "neo4j")
        return neo4j.GraphDatabase.driver(
            self.uri, auth=(self.user, self.password), **overrides
        )
