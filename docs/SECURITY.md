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

With the defaults nothing does. These are the paths that can send data out:

| Path | What leaves | When |
|---|---|---|
| The Anthropic provider (`LLM_PROVIDER=anthropic`) | the system prompt (persona, recalled memories, theory sections) and the conversation | every turn and every background brain call |
| The DCM evaluator's Anthropic mode (`DCM_EVALUATOR_MODE`) | the indicator evaluation prompts | when the DCM benchmark is run in that mode |
| The embedding model download | a request for `EMBEDDING_MODEL` from the model hub | once, on first use or with `mtc doctor --fetch-models`; set the hub's offline variable afterwards |
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
