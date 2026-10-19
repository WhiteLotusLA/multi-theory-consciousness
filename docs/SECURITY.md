# Security

MTC runs on one machine for one operator. Its defaults are chosen so that a
fresh install exposes nothing, trusts no one by name and keeps every secret in
one file. This page says what each default protects and what an operator must
do before exposing anything.

## The API key: one carrier, fail closed

- The API key authenticates the client application, never a speaker. Its one
  carrier is the `X-API-Key` header, on every HTTP route except `/health` and
  on the websocket handshake. There is no Bearer form and no query parameter,
  because a key in a URL ends up in access logs and browser history.
- `API_AUTH_ENABLED` defaults true. With an empty `API_KEY` every
  authenticated route answers 503 and every websocket handshake is closed:
  a server whose key was never configured authenticates nobody, and an empty
  presented key is never compared with an empty configured one.
- Keys are compared as bytes with `secrets.compare_digest`.
- `mtc init` generates the key with `secrets.token_hex(32)` into `.env` and
  keeps an existing one.

See `mtc/api/auth.py` and [API.md](API.md).

## Speakers: a token, never a name

- Who is speaking is decided by `mtc/pipeline/speaker_trust.py`, the one trust
  source. A speaker is authenticated only when the token they present matches
  the token configured for their entry. A name on its own grants nothing: an
  unknown or unauthenticated speaker gets the default trust, no safety bypass
  and is labelled as a bracketed claim, whatever they are called.
- A claimed name is reduced to one safe line before it is used: every
  character that is not a letter or a digit and not a space, period,
  apostrophe, hyphen or underscore becomes a space, runs of whitespace
  collapse to one space, and the result is cut to 64 characters
  (`MAX_CLAIMED_NAME_CHARS`). A claim with nothing usable left is the
  anonymous default speaker. The prompt renders the speaker name on one line
  and quotes it under the unverified template, so a claimed name cannot add
  a section to the prompt. The entry lookup and the token comparison use the
  presented name as given (stripped), and an authenticated speaker keeps the
  configured entry name. The API refuses a speaker name that holds a control
  character, a line separator or a paragraph separator (422 on a route,
  `invalid_message` on the websocket) with a fixed message that does not
  echo it.
- The speaker token has its own carriers, separate from the API key: the
  `X-Speaker-Token` header on `POST /api/chat` and the `speaker_token` field of
  a websocket `message` frame. A connection's API key never makes its speaker
  trusted.
- The wizard writes the creator's token (`secrets.token_hex(32)`) into `.env`
  and the creator's entry into `config/speakers.yaml` (mode 0600). `mtc chat`
  sends that token only for the configured speaker, and sends none with
  `--anonymous`.
- A tree-wide test (`tests/pipeline/test_trust_single_source.py`) fails when
  any other module reads the trust settings, and another
  (`tests/consciousness/test_no_name_based_trust.py`) fails when a module
  outside its allow list reads `CREATOR_NAME`.

## Loopback by default

The API binds `127.0.0.1`, the model server settings point at `localhost`,
and every store setting defaults to `localhost`. Turning authentication off is
allowed only on a loopback address: the server refuses to start with
`API_AUTH_ENABLED=false` and a non-loopback `API_HOST`. The Compose file binds
every store port to `127.0.0.1` on the host (the file was reviewed against
official documentation and not run on the development machine; the path run
on the development machine is the in-memory install with the mock provider).

With API_AUTH_ENABLED=false every process on the machine that can reach the
port can use every route (chat, state, memory search). Only the
spontaneous-thought websocket frames are withheld, because no connection
counts as a keyed client. A state frame can quote the text of the workspace
winner, which may be a thought or a recalled memory.

With API_AUTH_ENABLED=false the HTTP routes do not validate the Host header,
so a web page that rebinds its domain to the loopback address can call them
from a browser on the same machine; keep authentication on whenever a
browser is used on that machine.

With authentication off the websocket checks the handshake's `Origin`
header instead of a key, because a browser page from any origin can open a
websocket (CORS does not apply to it). A handshake without an `Origin`
header (a client that is not a browser) is served, and so is one whose
`Origin` equals an entry of `API_CORS_ORIGINS` after lower-casing and
stripping a trailing slash; any other origin, the literal `null` included,
is closed with 1008 and the neutral reason. With `API_CORS_ORIGINS` set to
`*` the check refuses every browser origin, because `*` is not read as
allow-all for the websocket; a handshake without an `Origin` header is
still served. With authentication on, the key decides and the `Origin` is
not read.

## The safety gate and emergency suspension

- Every message of an untrusted speaker passes the safety gate before the
  cycle runs; a trusted speaker (authenticated by token and allowlisted)
  bypasses it. The gate's state is on every turn record and payload as
  `safety_gate`: `absent` (no safety stack), `bypassed`, `applied` or
  `unchecked` (the check did not run). The safety scoring is a keyword
  heuristic ([HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md), section 12).
- The gate fails closed. With `SAFETY_GATE_FAIL_CLOSED` true (the default),
  an untrusted speaker's turn whose check could not run ends with the error
  code `safety_unavailable` (HTTP 503) and nothing of it is stored; with it
  false the message is processed. The record says `unchecked` either way.
- An emergency-level score (manipulation or deception at or above 0.8) is
  handled by `EMERGENCY_SHUTDOWN_ENABLED`. With it false (the default) the
  message is denied, an audit row records the proposed emergency shutdown as
  not enacted, and one `URGENT` alert titled "Emergency shutdown not
  enacted" goes to the escalation role (`SAFETY_ESCALATION_ROLE`); nothing
  is suspended.
- With `EMERGENCY_SHUTDOWN_ENABLED` true the same score suspends every
  untrusted conversation: from then on every untrusted speaker's message is
  refused with the decision `emergency_shutdown` (the blocked payload's
  `safety_decision`, and `mtc chat` names the resume command) until the
  suspension is lifted. Trusted speakers keep bypassing the gate.
- Only a trusted speaker may lift a suspension: `POST /api/safety/resume`
  (or `mtc safety resume`, as the creator by default or `--speaker NAME`)
  answers 403 unless the speaker trust source verifies the speaker as
  trusted, and then resumes with the configured escalation role; the safety
  monitor's role check refuses, and audits, any other role. `mtc safety
  status` shows whether the untrusted conversations are suspended. See
  [API.md](API.md) and [OPERATIONS.md](OPERATIONS.md).

## No secrets in a client bundle

A browser page or any other client that runs on someone else's machine must
not carry the API key or a speaker token. For the web page planned for a later
release: never put either in build-time public variables (anything a bundler
inlines into the shipped script); put a small server-side proxy in front of
the API that holds the key and adds the headers. CORS never allows
credentials.

## Secrets live in `.env` only

- Every secret (the API key, the creator's speaker token, store passwords, the
  Anthropic key) lives in `.env` at the project root. The wizard writes it
  with mode 0600 and an atomic replace, keeps existing values, and refuses a
  value with `$`, a double quote, a backslash, a backtick or a newline, so
  Compose and the settings loader read every value identically.
- Secrets are never printed, logged, placed in JSON output, written to the
  persona, speakers, identity, CTM or answers files, or passed on a command
  line: the backup and restore scripts hand passwords to the dump tools
  through their environment variables (for example `PGPASSWORD`).
- `mtc doctor` reports each secret as `set` or `empty`, never its value, and
  warns when `.env` is not mode 0600.
- `.env` is gitignored, and the backup state archive excludes it: keep a copy
  of your secrets somewhere else you control.

## Data that leaves the machine

On a bare install with the defaults nothing does. With the `embeddings`
extra (part of `all`) the defaults do reach the network: the first load
of the sentence embedder in each process that uses it asks the model hub
about `EMBEDDING_MODEL`, even with the model cached, until
`HF_HUB_OFFLINE=1` is set. These are the paths that can send data out:

| Path | What leaves | When |
|---|---|---|
| The Anthropic provider (`LLM_PROVIDER=anthropic`) | the system prompt (persona, recalled memories, theory sections) and the conversation | every turn and every background brain call |
| The DCM evaluator's Anthropic mode (`DCM_EVALUATOR_MODE`) | the indicator evaluation prompts | when the DCM benchmark is run in that mode |
| The embedding model (the `embeddings` extra) | the model name in `EMBEDDING_MODEL`, sent to the model hub, and the model download when it is not cached | on the first load of the sentence embedder in each process, even with the model cached, and with `mtc doctor --fetch-models`; set `HF_HUB_OFFLINE=1` once the model is cached |
| The DCM scheme fetch | a request for the scheme at `DCM_API_URL` | when the benchmark runs with no cached scheme; the bundled copy under `data/dcm` is the fallback |

A local chat-completions server on this machine keeps the conversation local.

## Stores

- Every store binds to `localhost` in its default configuration, and the
  stores are optional. Keep them on loopback unless another machine must
  reach them.
- `scripts/ops/harden_services.sh` lists the TCP listeners on the store, model
  and API ports, reports any that is not on loopback with the configuration
  line to change for Homebrew and apt installs, and with `--fix-permissions`
  sets `.env` and `config/speakers.yaml` to 0600 and the agent space and
  backup directory to 0700. It never edits a store's configuration.
- For any store reachable from another machine, enable its authentication:
  PostgreSQL with a password for the application role (`init_postgres.sh`
  creates it), MongoDB with a user and `MTC_MONGODB_AUTH_SOURCE` naming the
  database the user is defined in, Redis with a password and protected mode,
  Neo4j with authentication on (the shipped `config/neo4j.conf.example`
  keeps it on), Qdrant with the access control its own configuration
  reference describes.
- The Compose file runs MongoDB without authentication on loopback only;
  enable it before any non-local deployment.

## Backups hold private data

Backups (`scripts/ops/backup.sh`, written under `BACKUP_LOCATION`) contain
conversation text, names, thoughts, beliefs and the persona. The backup
directory is created with mode 0700. Store backups where only the operator
can read them, never in a cloud-synced or shared folder, and delete them by
the retention rules rather than by hand. See [OPERATIONS.md](OPERATIONS.md).

## What the logs hold

Log lines carry step names, error types and counts, never message text, names,
keys or tokens. The specialty JSON-lines logs (`safety_audit.jsonl`,
`consciousness_metrics.jsonl`, `emotional_history.jsonl`) hold private safety
and emotional records and live under the gitignored log directory.

## The repository

The privacy audit (`scripts/privacy_audit.py`, see
[PRIVACY_AUDIT.md](PRIVACY_AUDIT.md)) keeps personal data and build artifacts
out of the tree, every commit and CI. Everything the wizard writes is
gitignored.

## Reporting a vulnerability

Report a vulnerability privately through a security advisory on the
repository (the Security tab of the GitHub repository, "Report a
vulnerability"), not in a public issue. Include the version, the steps to
reproduce and the impact you expect.
