"""HTTP and websocket API over the runtime pipeline.

The package serves the runtime built by ``mtc.pipeline.runtime.build_runtime``
through FastAPI: ``mtc.api.server.create_app`` builds the application and
``mtc.api.server.main`` runs it under uvicorn on ``API_HOST`` and
``API_PORT``. Two carriers exist and nothing else is accepted: the
``X-API-Key`` header authenticates the client application (every route except
``/health`` and the websocket handshake), and the ``X-Speaker-Token`` header
on ``/api/chat`` or the ``speaker_token`` field of a websocket ``message``
frame is handed to the speaker trust source, which alone decides who is
speaking. Neither value is logged, stored or echoed.

This package is the only part of ``mtc`` that imports the web framework at
module import (install the ``api`` extra); nothing else in ``mtc`` imports
it. The package marker imports no submodule, so importing ``mtc.api`` alone
loads nothing heavy.
"""
