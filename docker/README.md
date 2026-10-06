# Docker Compose for the MTC stores

`docker/docker-compose.yml` runs the five stores MTC can use (PostgreSQL,
MongoDB, Redis, Qdrant and Neo4j) on one machine, each published on
`127.0.0.1` only, each with a named volume and a healthcheck.

**Reviewed, not run.** The file was written and reviewed against the
official image documentation listed below. It was not run on the development
machine, where Docker is not installed. The path tested there is the
in-memory install with the mock provider; the host-service path (Homebrew or
apt services with `scripts/db/init_postgres.sh`) is described in the setup
guides. Treat the Compose path as reviewed but untested until you have run it.

## Use

Run from the project directory, with the `.env` that `mtc init` wrote:

```bash
make db-up DB_SERVICES="postgres mongo qdrant"
# which runs:
docker compose --env-file .env -f docker/docker-compose.yml up -d --wait postgres mongo qdrant
mtc db init
mtc db check
```

Start only the services the setup enabled. `make db-down` stops them and
keeps the volumes; nothing in the Makefile removes a volume.

What the file reads from `.env` (each with the Settings default for the
default `DB_PREFIX` of `mtc`): `MTC_POSTGRES_USER`, `MTC_POSTGRES_PASSWORD`,
`MTC_POSTGRES_DB`, `MTC_POSTGRES_PORT`, `MTC_MONGODB_PORT`, `MTC_REDIS_PORT`,
`MTC_QDRANT_PORT` and `MTC_NEO4J_PASSWORD`.

- `MTC_POSTGRES_PASSWORD` is required (`${MTC_POSTGRES_PASSWORD:?...}`):
  PostgreSQL is the base store. Because Compose interpolates the whole file,
  the variable must be set even to start another service on its own.
- Every other variable uses `${VAR:-default}`, so a missing optional value
  never blocks starting the other services.
- MongoDB runs without authentication, reachable on loopback only. For any
  non-local deployment enable authentication, create a user and set
  `MTC_MONGODB_USER`, `MTC_MONGODB_PASSWORD` and `MTC_MONGODB_AUTH_SOURCE`.
- Redis runs with `--appendonly yes --maxmemory-policy volatile-lru` and no
  password (loopback only); the policy matters once a memory limit is set.
- Neo4j needs `MTC_NEO4J_PASSWORD` (eight characters or more) before its
  service starts. Only the Bolt port 7687 is published, matching the default
  `NEO4J_URI`. Heap and page cache are small (512 MB and 256 MB); raise them
  for a large graph.
- The model server is not part of this file. Run it on the host
  (`scripts/ops/start_llm.sh`) or keep the mock provider.

## Verified facts

Checked on 2026-10-05 against the official sources named in each row.

| Service | Image and tag | Healthcheck command | Tool it relies on | Data path in the image | Source |
|---|---|---|---|---|---|
| postgres | `postgres:17` | `pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"` | `pg_isready` | `/var/lib/postgresql/data` | Docker Hub official image page `hub.docker.com/_/postgres` (tags `17`, data path for 17 and earlier; 18 moved to `/var/lib/postgresql/18/docker`, so the tag is pinned to 17) |
| mongo | `mongo:8` | `mongosh --quiet --eval "db.adminCommand('ping').ok"` | `mongosh` | `/data/db` | `hub.docker.com/_/mongo` and its tags page (tags `8`, `8-noble`; the page connects with `mongosh` for 5.x and later and states that no authentication is required by default) |
| redis | `redis:7` | `redis-cli ping` | `redis-cli` | `/data` | `hub.docker.com/_/redis` (tags `7`, `7-bookworm`; the page runs `redis-cli` from the image and passes server flags after `redis-server`) |
| qdrant | `qdrant/qdrant:v1.19.2` | bash TCP probe: `GET /readyz` on 6333, expect status 200 | `bash` (`/dev/tcp`) and `grep` | `/qdrant/storage` | `github.com/qdrant/qdrant/releases` (newest release v1.19.2); the repository's `Dockerfile` (a Debian slim final stage that installs `ca-certificates`, `tzdata` and `libunwind8` only, so no curl or wget; storage and snapshots under `/qdrant`); `qdrant.tech` installation and configuration pages (`/qdrant/storage`, `QDRANT__` variables, `telemetry_disabled`) |
| neo4j | `neo4j:5-community` | bash TCP probe: `GET /` on 7474, expect status 200 | `bash` (`/dev/tcp`) and `grep` | `/data` | `hub.docker.com/_/neo4j` (tags `5-community`, `5.26-community`; `/data` volume; `NEO4J_AUTH`); Neo4j 5 operations manual, Docker configuration (the `NEO4J_` variable naming: `_` becomes `__`, `.` becomes `_`; the image defaults of 512M heap and 512M page cache) |

Compose behavior, from `docs.docker.com`:

- Interpolation: `${VAR:-default}` uses the default when the variable is
  unset or empty; `${VAR:?error}` exits with the error. Precedence: the shell
  environment, then `--env-file`, then the project's `.env`. The page does
  not say whether an unresolved required variable blocks commands for
  services that are not started; this file assumes it does (the whole file is
  interpolated) and gives every optional variable a default. **Unverified.**
- `docker compose up --wait` exists: "Wait for services to be running|healthy.
  Implies detached mode", with `--wait-timeout` for a bound
  (`docs.docker.com/reference/cli/docker/compose/up/`).
- The top-level `version` key is obsolete and only produces a warning; the
  top-level `name` sets the project name (`docs.docker.com/reference/compose-file/version-and-name/`).
  The file has `name: mtc` and no `version`.

Facts that could not be confirmed on the official pages and are recorded as
**unverified**:

- That `pg_isready` sits on the `PATH` inside `postgres:17`: the image page
  does not list its tools. `pg_isready` is one of PostgreSQL's client
  applications, which the image's server packages install.
- That `bash` and `grep` are present in `qdrant/qdrant` and `neo4j`: both are
  essential packages of the Debian base the Qdrant `Dockerfile` names; the
  Neo4j image page does not list its base or tools.
- That `cypher-shell` is inside `neo4j:5-community`. The healthcheck does not
  use it: `cypher-shell` would need the password on its command line, and a
  `NEO4J_`-prefixed password variable would be read by the image as a
  configuration setting.
- That the `v1.19.2` release has a Docker Hub tag of the same name (the
  Docker Hub tag page for `qdrant/qdrant` was not fetched). Check
  `hub.docker.com/r/qdrant/qdrant/tags` before the first run.

Re-check the tags at each release: the release gate confirms they are still
current.

## Host-service configs

For stores installed on the host rather than in containers:

- `config/qdrant.yaml`: loopback service host, HTTP 6333 and gRPC 6334,
  storage and snapshots under `./data/qdrant/` (ignored by git), telemetry
  disabled. `scripts/ops/start_stack.sh` runs the Qdrant release binary with
  `--config-path config/qdrant.yaml` from the project directory.
- `config/neo4j.conf.example`: loopback listen address, the default database
  `neo4j` (the `NEO4J_DATABASE` default), a small heap and page cache, and
  authentication on. Copy it as `neo4j.conf` into a directory and point
  `NEO4J_CONF` at that directory.
