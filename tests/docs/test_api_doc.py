"""docs/API.md lists exactly the routes, frames and error codes the API serves.

The route list comes from the application ``create_app`` builds (never
started, so no runtime is built and nothing is served); the frame types and
error codes come from ``mtc.api.schemas``.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from mtc.api import schemas  # noqa: E402
from mtc.api.server import create_app  # noqa: E402
from mtc.core.config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "API.md"
HEADING = re.compile(r"^### `(GET|POST) (/[^`]*)`$", re.M)
WS_HEADING = re.compile(r"^### `(/[^`]*)` \(websocket\)$", re.M)


def _served():
    app = create_app(settings=Settings(_env_file=None, WEBSOCKET_ENABLED=True))
    http, ws = set(), set()
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if methods:
            for method in methods:
                if method in ("GET", "POST"):
                    http.add((method, route.path))
        else:
            ws.add(route.path)
    return http, ws


def test_api_doc_lists_every_route():
    text = DOC.read_text(encoding="utf-8")
    http, ws = _served()
    documented = set(HEADING.findall(text))
    assert http, "the application served no route"
    assert documented == http, (
        f"missing from API.md: {sorted(http - documented)}; "
        f"not served: {sorted(documented - http)}"
    )
    assert set(WS_HEADING.findall(text)) == ws


def test_api_doc_lists_every_frame_and_error_code():
    text = DOC.read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| `([a-z_]+)` \|", text, re.M))
    for frame in schemas.CLIENT_FRAME_TYPES + schemas.SERVER_FRAME_TYPES:
        assert frame in rows, f"frame {frame!r} has no row in API.md"
    for code in schemas.ERROR_CODES:
        assert f"`{code}`" in text, f"error code {code!r} missing from API.md"
