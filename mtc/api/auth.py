"""
API Key Authentication
======================

The API key authenticates the client application, never a speaker. It has
one carrier, the ``X-API-Key`` header, on every HTTP route except
``/health`` and on the websocket handshake. There is no Bearer form and no
query parameter: a key in a URL ends up in access logs and browser history.

Fail closed:

- ``API_AUTH_ENABLED`` true and ``API_KEY`` empty refuses every authenticated
  route with 503 and every websocket handshake with close code 1008: a server
  whose key was never configured authenticates nobody, and an empty presented
  key is never compared with an empty configured one. A refused handshake is
  accepted and then closed with 1008 and one neutral reason
  (``WS_REFUSAL_REASON``), so a client sees the close code under a real
  server too (a close before the accept reaches it as an HTTP 403).
- A missing or wrong key is 401 with ``WWW-Authenticate: ApiKey``. Keys are
  compared as UTF-8 bytes with ``secrets.compare_digest``.
- ``API_AUTH_ENABLED`` false is allowed only on a loopback bind:
  ``assert_bind_allowed`` raises ``UnsafeBindError`` naming ``API_HOST`` and
  ``API_AUTH_ENABLED`` when ``API_HOST`` is outside ``LOOPBACK_HOSTS``, and
  logs one WARNING per process when auth is off on loopback.

The speaker token is a separate secret with its own carriers
(``SPEAKER_TOKEN_HEADER`` here, the ``speaker_token`` field of a websocket
message frame); it is handed to the speaker trust source unchanged and is
never compared, logged or interpreted here.

Config keys read at call time from the application's Settings:
``API_AUTH_ENABLED``, ``API_KEY`` and ``API_HOST``. Nothing is persisted. No
key, token or any part of one is ever written to a log line or a response.
"""

import logging
import secrets
from typing import Any, Optional, Tuple

from fastapi import HTTPException, Request, status

from mtc.core import config as core_config
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
SPEAKER_TOKEN_HEADER = "X-Speaker-Token"
AUTH_SCHEME = "ApiKey"

# Websocket close codes (RFC 6455): policy violation and message too big.
WS_POLICY_VIOLATION = 1008
WS_MESSAGE_TOO_BIG = 1009
WS_NORMAL = 1000
# The one close reason of every refused handshake: it does not say whether
# the key was missing, wrong or the server has none configured.
WS_REFUSAL_REASON = "not authenticated"

_DISABLED_WARNING = (
    "API authentication is disabled (API_AUTH_ENABLED=false); the server "
    "accepts unauthenticated requests on its loopback bind"
)
_EMPTY_KEY_ERROR = (
    "API authentication is enabled but API_KEY is empty; every authenticated "
    "request is refused until a key is configured"
)

_warned_disabled = False
_reported_empty = False


class UnsafeBindError(RuntimeError):
    """Authentication is off and the bind address is not a loopback host."""


def reset_auth_warning_for_tests() -> None:
    """Forget that the once-per-process warnings were logged."""
    global _warned_disabled, _reported_empty
    _warned_disabled = False
    _reported_empty = False


def _warn_disabled_once() -> None:
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        logger.warning(_DISABLED_WARNING)


def _report_empty_once() -> None:
    global _reported_empty
    if not _reported_empty:
        _reported_empty = True
        logger.error(_EMPTY_KEY_ERROR)


def settings_for(app: Any) -> Settings:
    """The Settings an application was created with, else the process ones."""
    state = getattr(app, "state", None)
    cfg = getattr(state, "settings", None)
    return cfg if isinstance(cfg, Settings) else get_settings()


def _normalize_host(host: str) -> str:
    value = str(host or "").strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def is_loopback_host(host: str) -> bool:
    """True when ``host`` is one of ``LOOPBACK_HOSTS`` (case and brackets aside)."""
    allowed = {_normalize_host(item) for item in core_config.LOOPBACK_HOSTS}
    return _normalize_host(host) in allowed


def assert_bind_allowed(settings: Optional[Settings] = None) -> None:
    """Refuse an unauthenticated bind outside loopback; warn once on loopback.

    Raises:
        UnsafeBindError: ``API_AUTH_ENABLED`` is false and ``API_HOST`` is not
            in ``LOOPBACK_HOSTS``.
    """
    cfg = settings if settings is not None else get_settings()
    if cfg.API_AUTH_ENABLED:
        if not cfg.API_KEY:
            _report_empty_once()
        return
    if not is_loopback_host(cfg.API_HOST):
        raise UnsafeBindError(
            "refusing to serve without authentication on a non-loopback "
            "address: set API_AUTH_ENABLED=true with an API_KEY, or bind "
            "API_HOST to one of the loopback hosts "
            f"({', '.join(core_config.LOOPBACK_HOSTS)})"
        )
    _warn_disabled_once()


def _matches(presented: str, expected: str) -> bool:
    return secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def check_api_key(cfg: Settings, presented: Optional[str]) -> Tuple[bool, int, str]:
    """``(allowed, http_status, reason)`` for a presented key under ``cfg``.

    The reason is a fixed phrase; it never contains the presented value.
    """
    if not cfg.API_AUTH_ENABLED:
        _warn_disabled_once()
        return True, status.HTTP_200_OK, ""
    expected = cfg.API_KEY
    if not expected:
        _report_empty_once()
        return (
            False,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "server authentication is not configured",
        )
    if not presented:
        return False, status.HTTP_401_UNAUTHORIZED, f"missing {API_KEY_HEADER} header"
    if not _matches(presented, expected):
        logger.warning("request refused: invalid API key")
        return False, status.HTTP_401_UNAUTHORIZED, "invalid API key"
    return True, status.HTTP_200_OK, ""


async def require_api_key(request: Request) -> None:
    """FastAPI dependency for every authenticated route.

    503 when auth is on and ``API_KEY`` is empty; 401 with
    ``WWW-Authenticate: ApiKey`` on a missing or wrong key.
    """
    cfg = settings_for(request.app)
    allowed, code, reason = check_api_key(cfg, request.headers.get(API_KEY_HEADER))
    if allowed:
        return None
    headers = {"WWW-Authenticate": AUTH_SCHEME} if code == 401 else None
    raise HTTPException(status_code=code, detail=reason, headers=headers)


def ws_api_key_valid(websocket: Any) -> Tuple[bool, int, str]:
    """``(allowed, close_code, reason)`` for a websocket handshake.

    The key is read from the ``X-API-Key`` header only; a key in the query
    string is ignored, so a handshake that carries it there is refused as
    missing. A refusal closes with 1008 (policy violation) and the neutral
    ``WS_REFUSAL_REASON`` whatever the cause.
    """
    cfg = settings_for(getattr(websocket, "app", None))
    allowed, _code, _reason = check_api_key(cfg, websocket.headers.get(API_KEY_HEADER))
    if allowed:
        return True, WS_NORMAL, ""
    return False, WS_POLICY_VIOLATION, WS_REFUSAL_REASON


def client_authenticated(cfg: Settings, presented: Optional[str]) -> bool:
    """True only when auth is on and the presented key is the configured one.

    A connection on a server with auth disabled is allowed but is not an
    authenticated client, so nothing restricted to authenticated clients
    reaches it.
    """
    if not cfg.API_AUTH_ENABLED or not cfg.API_KEY or not presented:
        return False
    return _matches(presented, cfg.API_KEY)
