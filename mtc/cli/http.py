"""
Command-Line API Client
=======================

The small HTTP client behind ``mtc chat`` and ``mtc ctm``: standard-library
``urllib`` only, JSON bodies, one request per call.

- The API key travels in the ``X-API-Key`` header and a speaker's token in
  ``X-Speaker-Token``; neither is ever logged, printed or placed in a URL.
- Message text and names travel in the request body, never in a query string.
- A response is returned as ``(status, payload)`` whatever the status; only a
  server that cannot be reached raises ``ApiUnreachable`` (by reason type,
  never with a header or a body).
- Proxies from the environment are ignored: the server is the operator's own.

Config keys read by ``ApiClient.from_settings``: ``API_HOST``, ``API_PORT``,
``API_KEY``, ``GENERATION_TIMEOUT_S`` and ``GENERATION_STALL_TIMEOUT_S`` (a
chat turn may wait as long as the server lets a generation run). Nothing is
persisted. The urllib modules are imported inside the request, so importing
this module loads no HTTP machinery.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from mtc.core.config import LOOPBACK_HOSTS, Settings, get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
SPEAKER_TOKEN_HEADER = "X-Speaker-Token"

HEALTH_PATH = "/health"
CHAT_PATH = "/api/chat"
CTM_START_PATH = "/api/ctm/start"
CTM_STOP_PATH = "/api/ctm/stop"
CTM_STATUS_PATH = "/api/ctm/status"

# Seconds added to the server's own generation bounds for a chat turn.
CHAT_TIMEOUT_MARGIN_S = 30.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ApiUnreachable(Exception):
    """The API server did not accept the connection or did not answer."""


def _client_host(host: str) -> str:
    """The address a local client connects to for a bind address."""
    import ipaddress

    text = str(host).strip()
    bare = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    try:
        address = ipaddress.ip_address(bare)
    except ValueError:
        return text
    if address.is_unspecified:
        # A server bound to every interface is reached over loopback.
        return LOOPBACK_HOSTS[0]
    return f"[{bare}]" if address.version == 6 else bare


class ApiClient:
    """One API server, one key.

    Args:
        base_url: Scheme, host and port of the server, without a path.
        api_key: The key sent as ``X-API-Key``; empty sends no key header.
        timeout_s: Seconds a request may take (a chat turn uses
            ``chat_timeout_s``, which defaults to the same value).
    """

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 30.0) -> None:
        parts = urlsplit(str(base_url).strip().rstrip("/"))
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("base_url must be an http or https URL")
        if parts.query or parts.fragment:
            raise ValueError("base_url carries no query string or fragment")
        self.base_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        self._api_key = str(api_key or "")
        self.timeout_s = float(timeout_s)
        self.chat_timeout_s = float(timeout_s)

    def __repr__(self) -> str:
        return f"ApiClient(base_url={self.base_url!r})"

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "ApiClient":
        """The client for the server this project's settings describe."""
        cfg = settings if settings is not None else get_settings()
        netloc = f"{_client_host(cfg.API_HOST)}:{int(cfg.API_PORT)}"
        client = cls(urlunsplit(("http", netloc, "", "", "")), cfg.API_KEY)
        client.chat_timeout_s = (
            float(cfg.GENERATION_TIMEOUT_S)
            + float(cfg.GENERATION_STALL_TIMEOUT_S)
            + CHAT_TIMEOUT_MARGIN_S
        )
        return client

    @property
    def where(self) -> str:
        """Host and port, for messages."""
        return urlsplit(self.base_url).netloc

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        import socket
        import urllib.error
        import urllib.request

        if "?" in path or "#" in path:
            raise ValueError("request paths carry no query string")
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers[API_KEY_HEADER] = self._api_key
        headers.update(extra_headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        wait = self.timeout_s if timeout_s is None else float(timeout_s)
        try:
            with opener.open(request, timeout=wait) as response:
                status = int(response.status)
                raw = response.read(MAX_RESPONSE_BYTES)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read(MAX_RESPONSE_BYTES)
        except urllib.error.URLError as exc:
            reason = exc.reason
            kind = (
                type(reason).__name__ if isinstance(reason, BaseException) else "error"
            )
            logger.debug("API request to %s failed: %s", path, kind)
            raise ApiUnreachable(kind) from None
        except (TimeoutError, socket.timeout):
            logger.debug("API request to %s timed out", path)
            raise ApiUnreachable("timeout") from None
        except OSError as exc:
            logger.debug("API request to %s failed: %s", path, type(exc).__name__)
            raise ApiUnreachable(type(exc).__name__) from None
        logger.debug("API %s %s answered %d", method, path, status)
        return status, _payload(raw)

    def health(self) -> Tuple[int, Dict[str, Any]]:
        """``GET /health`` (needs no key)."""
        return self._request("GET", HEALTH_PATH)

    def chat(
        self,
        message: str,
        speaker: Optional[str] = None,
        speaker_token: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """One turn. The token header is sent only with a speaker name."""
        body: Dict[str, Any] = {"message": message}
        headers: Dict[str, str] = {}
        if speaker:
            body["speaker"] = speaker
            if speaker_token:
                headers[SPEAKER_TOKEN_HEADER] = speaker_token
        return self._request(
            "POST",
            CHAT_PATH,
            body,
            extra_headers=headers,
            timeout_s=self.chat_timeout_s,
        )

    def ctm_start(self, state: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        """Start continuous thought, optionally in a named state."""
        body: Dict[str, Any] = {"state": state} if state else {}
        return self._request("POST", CTM_START_PATH, body)

    def ctm_stop(self) -> Tuple[int, Dict[str, Any]]:
        """Stop continuous thought."""
        return self._request("POST", CTM_STOP_PATH, {})

    def ctm_status(self) -> Tuple[int, Dict[str, Any]]:
        """The continuous thought status."""
        return self._request("GET", CTM_STATUS_PATH)


def _payload(raw: bytes) -> Dict[str, Any]:
    """The JSON object of a response body; anything else is kept as text."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except ValueError:
        return {"detail": text[:500]}
    return loaded if isinstance(loaded, dict) else {"detail": loaded}
