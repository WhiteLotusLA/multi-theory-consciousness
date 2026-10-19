# API

`mtc serve` (or `python -m mtc.api.server`) serves one runtime in one process
with uvicorn: an HTTP API and a websocket. It needs the `api` extra (fastapi,
uvicorn, httpx). Every route, frame and status code below is read from
`mtc/api/server.py`, `mtc/api/websocket.py`, `mtc/api/auth.py` and
`mtc/api/schemas.py`, and a test (`tests/docs/test_api_doc.py`) fails when the
application serves a route this page does not list.

## Binding and authentication

- **Loopback by default.** `API_HOST` defaults to `127.0.0.1` and `API_PORT`
  to 8200. Every option comes from Settings; the server takes no flags.
- **Fail closed.** `API_AUTH_ENABLED` defaults true. With auth on and an empty
  `API_KEY`, every authenticated route answers 503 and every websocket
  handshake is closed with 1008: a server whose key was never configured
  authenticates nobody. `mtc init` generates the key into `.env`.
- **Auth off only on loopback.** With `API_AUTH_ENABLED` false and an
  `API_HOST` outside the loopback hosts (`127.0.0.1`, `::1`, `localhost`), the
  server refuses to start: `mtc serve` prints the refusal and exits 2. On
  loopback it starts and logs one warning.
- **No documentation routes.** No OpenAPI, Swagger or ReDoc route is served,
  so nothing is reachable without the key except `/health`.

## The two header carriers

| Header | Carries | Where |
|---|---|---|
| `X-API-Key` | the API key; authenticates the client application, never a speaker | every route except `/health`, and the websocket handshake |
| `X-Speaker-Token` | a speaker's token; decides who is speaking | `POST /api/chat` (the websocket carries it in the `speaker_token` field of a `message` frame) |

There is no Bearer form and no query parameter: a key in a URL ends up in
logs and browser history. A missing or wrong key is 401 with
`WWW-Authenticate: ApiKey`; keys are compared with `secrets.compare_digest`.
The speaker token is handed to the speaker trust source unchanged and is
never logged, stored or echoed. A connection's API key never makes its
speaker trusted, and a speaker name without a valid token is a bracketed
claim with the default trust. See [SECURITY.md](SECURITY.md).

## Routes

### `GET /health`

No key needed. Response: `status`, `ready`, `version` and `brain_verified`.
Always 200. `status` reads:

- `degraded` while the runtime is not started, while the brain is not
  available, or after the brain's last real provider call failed;
- `ok` when the brain is available and verified;
- `unverified` when it is available and no real exchange has confirmed the
  provider yet, and when the brain's status cannot be read.

`brain_verified` is true when the brain's recorded health came from a real
exchange that succeeded or its last real provider call returned a
completion; false when it keeps a health record and neither holds, or when
its status cannot be read; null for a brain that keeps no health record,
which keeps the availability rule alone (`ok` when ready and available,
else `degraded`).

### `GET /api/meta`

The system name, the version, the eight theories (`id`, `name`), every
indicator with its metadata (`indicators` and `negative_indicators`: name,
theory, polarity, description, threshold, weight, `butlin_id`, specificity,
sensitivity, `entails`, `supported_by`; a value the assessment does not define
is null), the providers (`registered`, `active`), the capabilities (`vision`
false, the registered tools, which are none unless a plugin registers one,
`websocket`, `streaming`), the narration flags and `auth` (`enabled`, the two
header names).

### `POST /api/chat`

Request: `{"message": str (1 to 32000 characters), "speaker": str (optional,
up to 200)}` with an optional `X-Speaker-Token` header. One conversation turn.

| Status | When | Body |
|---|---|---|
| 200 | the turn succeeded | `TurnPayload` |
| 403 | the safety gate blocked the message | `TurnPayload` with `blocked` true and `safety_decision` |
| 502 | the model gave no answer (`no_answer`) | `TurnPayload` |
| 503 | the brain is unavailable (`brain_unavailable`), the safety check could not run for an untrusted speaker (`safety_unavailable`), or the runtime or pipeline is not built | `TurnPayload`, or `{"detail": ...}` |
| 504 | the pipeline's generation bound ran out (`TurnPayload`), or the API's first-token or stall bound cancelled the turn (`TimeoutPayload`) | see below |
| 500 | any other turn failure | `TurnPayload` |

Two error codes share 503, so a client picks its message by the body's
`error_code`, not by the status.

`TurnPayload`: `success`, `response` (the spoken text), `thinking`,
`conversation_id`, `message_count`, `timestamp`, `error`, `error_code`,
`cycle_ok`, `speaker_label` (the display label only), `provider_id`,
`blocked`, `safety_gate` and `safety_decision`. No identity internals or
tokens. `safety_gate` is the state of the safety gate for the turn:
`absent` (no safety stack), `bypassed` (a trusted speaker), `applied` (the
check ran) or `unchecked` (the check did not run); it is null only for a
result built outside the pipeline. `safety_decision` is the safety monitor's
decision for a blocked turn (`emergency_shutdown` when the untrusted
conversations are suspended) and null otherwise. A turn whose gate reads
`unchecked` for an untrusted speaker ends with `success` false, `blocked`
false and `error_code` `safety_unavailable`, and nothing of it is stored,
unless `SAFETY_GATE_FAIL_CLOSED` is false (see
[SECURITY.md](SECURITY.md)).

`TimeoutPayload` (504 from the API's own bounds): `success` false, `error`,
`error_code` `generation_timeout`, `bound` (`first_token` or `stall`),
`limit_s`, `elapsed_s`, `first_token_s` (null when no token came) and `state`
`cancelled`. The turn's reply, session and counts are unknown at that moment
and are left out rather than filled in.

The bounds: the first-token bound (`GENERATION_TIMEOUT_S`, 900 seconds by
default) counts until the first streamed chunk, queueing behind another turn
included; the stall bound (`GENERATION_STALL_TIMEOUT_S`, 300 seconds) cuts a
generation that stops streaming for that long. Both stop once the pipeline
reports the reply stored, so a slow post-turn step never turns a stored
reply into a 504; the post-turn work is bounded by the pipeline itself (see
[RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md)). A client that disconnects does
not cancel its turn: the turn completes and its reply is stored.

### `POST /api/safety/resume`

Request: `{"speaker": str (1 to 200 characters, no control character or line
break)}` with the speaker's token in the `X-Speaker-Token` header; no other
field is accepted. Lifts an emergency suspension of the untrusted
conversations (see [SECURITY.md](SECURITY.md)) by calling the safety
integration with the configured `SAFETY_ESCALATION_ROLE`.

| Status | When | Body |
|---|---|---|
| 200 | the suspension was lifted, or there was none | `{"resumed": bool, "suspended": bool}` |
| 401 | the API key is missing or wrong | `{"detail": ...}` |
| 403 | the speaker trust source does not verify the speaker as trusted (the token authenticates the name and the name is allowlisted) | `{"detail": ...}` |
| 422 | the body fails validation | the validation shape below |
| 503 | the runtime has no speaker trust source or no safety stack | `{"detail": ...}` |
| 500 | the safety stack returned an outcome without a boolean `resumed` or `suspended` | `{"detail": ...}` |

`resumed` says whether a suspension was lifted by this call and `suspended`
whether the untrusted conversations are suspended now. `mtc safety resume`
is the command-line client of this route.

### `GET /api/consciousness/status`

`orchestrator` (counters, failures, stage timings, the baseline, theory
modulations, absent signals, measured PAD sources, narration, the augmenter,
the config), `first_run` (the warmup report described in
[FIRST_RUN.md](FIRST_RUN.md)), `theory_modulations` (six, each with
`consumed`), `gate` (the thalamic gate's statistics), `ctm` (the thought
machine's status and `quiet_reason`), `stores` (per store: enabled, backend,
error), `provider`, `safety` (the safety stack's report; its `backend` is the
audit store's class name, and its privacy paths are project-relative or a
bare file name), `background` (task health), `pipeline`,
`speakers` (counts by role from the trust source, never names), `requests`
(duplicate request ids), `websocket` and `runtime`.

### `GET /api/consciousness/state`

`available`, `state` (the last cycle's scalar fields; content objects are
replaced by counts), `integration_proxy` (the PhiTracker value with its label
`integration proxy` and a note that it is not IIT Phi; null while warming up)
and `last_turn` (the last turn's modulation layers).

### `GET /api/consciousness/assessment`

Query: `refresh` (bool, default false). Without `refresh` it returns the last
assessment, or `status` `not_run`. With `refresh=true` it runs one under the
turn lock and reuses it for 10 seconds. Response: `status`, `overall_score`,
`overall_withheld`, `coverage`, `cached`, `age_s` and `report`. A record that
measured nothing reads `overall_score` null with `status` `insufficient_data`
and `overall_withheld` true, never a measured zero. A refresh whose
orchestrator returns nothing reads `status` `failed`; a runtime with no
orchestrator answers 503.

### `GET /api/consciousness/snapshots`

Query: `hours` (above 0, up to 8784, default 24) and `limit` (1 to 1000,
default 100). Response: `available`, `hours`, `limit`, `history` and `events`
from the snapshot service; `available` false when there is none.

### `POST /api/memory/search`

Request: `{"query": str (1 to 2000 characters), "top_k": int (optional, 1 to
50)}`. Response: `method` (`vector`, `text` or `none`), `memories`,
`concepts` and `stats`. Search text travels in the body, never in a query
string.

### `GET /api/knowledge/concepts`

Query: `limit` (1 to 1000, default 100). Response: `available`, `backend`
(`memory` unless `NEO4J_ENABLED`, then `neo4j`) and `concepts` (names).

### `POST /api/knowledge/concepts`

Request: `{"name": str (1 to 200 characters)}`. Response: `available`,
`concept` and `related` (up to 10 related concepts with relationship and
strength).

### `POST /api/ctm/start`

Request (optional): `{"state": "awake" | "drowsy" | "dreaming" |
"deep_sleep"}`. Starts the continuous thought machine. Response: `available`,
`running`, `quiet_reason`, `status`. 503 when continuous thought is off.

### `POST /api/ctm/stop`

Stops the machine. Same response; 503 when continuous thought is off.

### `GET /api/ctm/status`

Same response; `available` false when continuous thought is off.

### `GET /api/models`

`providers`, `active`, `model_name`, `available`, `health` (cached; the
Anthropic provider reads unverified until a real call) and `provider`.

### `/ws` (websocket)

Served only when `WEBSOCKET_ENABLED` (true by default). See below.

## Error shapes

- A request that fails validation is 422 with
  `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}`: the location
  and type of each error, never the input.
- 401 and 503 from authentication, and 503 from a runtime that is not ready,
  are `{"detail": "..."}`.
- Turn failures carry a `TurnPayload` with `error_code`, as above.

Error codes shared by the HTTP mapping and the websocket `error` frame:
`blocked`, `no_answer`, `brain_unavailable`, `safety_unavailable`,
`generation_timeout`, `turn_failed`, `runtime_not_ready`, `invalid_message`,
`unknown_type`, `frame_too_large`. `brain_unavailable` and
`safety_unavailable` are both 503.

A speaker name that holds a control character, a line separator or a
paragraph separator is refused (422 on a route, `invalid_message` on the
websocket) with a fixed message that does not echo it.

## CORS

`API_CORS_ORIGINS` (comma-separated) lists the allowed origins; the default
names two local development origins on ports 3000 and 3001, and an empty value
allows none. Credentials are never allowed, the methods are GET and POST, and the
allowed headers are `X-API-Key`, `X-Speaker-Token` and `Content-Type`. A
browser page must not hold the API key or a speaker token; see
[SECURITY.md](SECURITY.md).

## The websocket protocol

`/ws` speaks JSON text frames `{"type": ..., "data": {...}}` of at most
1 MiB. The handshake needs the `X-API-Key` header (a key in the query string
is ignored). With `API_AUTH_ENABLED` false no key is needed and the
handshake's `Origin` header is checked instead (see
[SECURITY.md](SECURITY.md)).

Client to server:

| Type | Data | Effect |
|---|---|---|
| `message` | `message` (required), `speaker`, `speaker_token`, `request_id` (all optional) | one conversation turn through the pipeline |
| `ping` | none | answered with `pong` |
| `subscribe_state` | none | the connection receives `consciousness_state` frames |
| `unsubscribe_state` | none | stops them |

Server to client:

| Type | When | Data |
|---|---|---|
| `status` | on connect | system name, version, `client_authenticated`, `brain_available`, `brain_verified` (as on `/health`), `max_frame_bytes`, `heartbeat_s` |
| `thinking` | when a turn starts and after every `THINKING_HEARTBEAT_S` (15 seconds) of silence | `elapsed_s`, `request_id` |
| `stream_chunk` | as the reply streams | `content` (spoken text only), `request_id` |
| `stream_retry` | the brain restarted a looping generation | `request_id`; chunks already sent belong to the discarded attempt |
| `response` | the turn finished | `TurnPayload` and `request_id`; this text is authoritative |
| `consciousness_state` | after each turn, and every `METRICS_UPDATE_INTERVAL` (60) seconds, to every subscriber, keyed or not | the state payload |
| `spontaneous_thought` | the thought machine expressed a thought | `thought_id`, `content`, `thought_type`, `timestamp`, `intensity`, `salience`, `salience_basis` (what the salience rests on) and `form` (`template`, `model` or `injected`); the last two are null when the record lacks them; sent only to clients that presented the API key |
| `error` | a frame or turn failed | `code`, `message`, `request_id` when known; a failed turn's frame also carries `blocked`, `safety_gate` and `safety_decision` from its `TurnPayload` (the HTTP body's fields), so a client can tell an emergency suspension (`emergency_shutdown`) from an ordinary deny and see that a `safety_unavailable` refusal was an unchecked gate; an error that is not a turn result carries none of the three (`invalid_message`, `unknown_type`, `runtime_not_ready`, `generation_timeout`, and `turn_failed` when the turn raised) |
| `pong` | after a `ping` | `timestamp` |

Close codes: 1008 for a refused handshake (missing, wrong or unconfigured
key, or with authentication off an `Origin` that is not listed; one neutral
reason, `not authenticated`), 1013 when the runtime is not
built yet, 1009 for a frame over the limit. A refused handshake is accepted
and then closed, so a client sees the close code under a real server as well.

Reasoning blocks are filtered out of `stream_chunk` with a one-chunk
lookahead (`ThinkFilter`). Reasoning that ran longer than one chunk before a
stray closing tag cannot be recalled from chunks already sent; the final
`response` frame carries the authoritative split.

Turns from every connection go through the pipeline's turn lock; a connection
keeps answering pings while its turn runs. The first-token and stall bounds
apply to websocket turns too, until the pipeline reports the reply stored. A
client that disconnects does not cancel its turn: the turn completes and its
reply is stored.

## Startup and shutdown order

The application's lifespan:

1. `setup_logging()`;
2. the bind check (`assert_bind_allowed`): an unauthenticated non-loopback
   host is refused;
3. the runtime is built from Settings (`build_runtime`) unless one was
   injected, then started (`Runtime.start` restores the persistence hub and
   starts the background tasks);
4. the expressed-thought broadcast is attached and, with the websocket on and
   `METRICS_UPDATE_INTERVAL` above 0, the periodic state push starts;
5. the log line `API ready (auth on)` or `API ready (auth off)`.

On shutdown the state push and any running turns are cancelled, the broadcast
is detached and `Runtime.stop` runs (see
[RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md)), then `API stopped` is logged.
uvicorn runs with its access log off, one worker and no server header.

Before the server is configured, both entry points check that the workspace
can be built: without the `fep` extra they print one `MISSING EXTRA` line on
stderr and exit 2. `python -m mtc.api.server` exits 0 after a server that
started stops, 1 when the server did not start (uvicorn logs a failed
lifespan startup and returns; one stderr line says so), and 2 for a refused
bind or a missing `fep` extra. `mtc serve` returns the same codes and 3 when
the port is already in use.

## Using the API from the command line

`mtc chat`, `mtc ctm` and `mtc safety` are the local clients: they read
`API_HOST`, `API_PORT` and `API_KEY` from Settings, send the key in
`X-API-Key`, and `mtc chat` sends the configured speaker's token in
`X-Speaker-Token` unless `--anonymous` is given. `mtc safety status` reads
`GET /api/consciousness/status` and `mtc safety resume` calls
`POST /api/safety/resume` with the identity `mtc chat` uses. See the CLI
table in the [README](../README.md).
