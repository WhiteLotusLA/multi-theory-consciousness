# Setup with Docker Compose

**Reviewed, not run.** `docker/docker-compose.yml` was written and reviewed
against official documentation and was not run on the development machine,
where Docker is not installed. The path run on the development machine is
the in-memory install with the mock provider ([GETTING_STARTED.md](GETTING_STARTED.md),
first path). The image tags, healthcheck tools, data paths and the Compose
behaviours the file relies on, with the source and date each was checked and
the facts that could not be confirmed, are recorded in
[docker/README.md](../docker/README.md). Treat this path as reviewed but
untested until you have run it.

The Compose file runs the five stores. The model server is not part of it:
run one on the host ([LLM_SETUP.md](LLM_SETUP.md)) or keep the mock provider.

## Prerequisites

- Docker with Compose v2 (`docker compose`, not the old `docker-compose`).
- Python 3.11 or later and MTC installed with the store clients and the
  `fep` extra, the minimum install that can run the assessment:
  `pip install -e ".[fep,databases]"`, or `.[all]`, which adds the API
  server, the model clients and the embedder. Without the `fep` extra
  `mtc assess`, `mtc serve`, the demo and the examples that build a
  workspace stop with one `MISSING EXTRA` line on stderr and exit code 2.
- A project configured by `mtc init` with the `docker` service manager and
  the stores you want enabled. The wizard writes explicit store names, ports
  and passwords into `.env`, so Compose and MTC read the same values.

## What the file defines

| Service | Image | Host port (loopback only) | Volume | Healthcheck |
|---|---|---|---|---|
| `postgres` | `postgres:17` | `MTC_POSTGRES_PORT`, default 5432 | `mtc_pgdata` | `pg_isready` |
| `mongo` | `mongo:8` | `MTC_MONGODB_PORT`, default 27017 | `mtc_mongodata` | `mongosh` ping |
| `redis` | `redis:7` | `MTC_REDIS_PORT`, default 6379 | `mtc_redisdata` | `redis-cli ping` |
| `qdrant` | `qdrant/qdrant:v1.19.2` | `MTC_QDRANT_PORT`, default 6333 | `mtc_qdrantdata` | a bash TCP probe of the readiness endpoint |
| `neo4j` | `neo4j:5-community` | 7687 (bolt) | `mtc_neo4jdata` | a bash TCP probe of the HTTP endpoint inside the container |

- The project name is `mtc` (top-level `name:`); there is no obsolete
  `version` key.
- Every port is published on `127.0.0.1` only.
- PostgreSQL's user, password and database come from `.env`
  (`MTC_POSTGRES_USER`, `MTC_POSTGRES_PASSWORD`, `MTC_POSTGRES_DB`); the
  image creates the role and the database from them, so `init_postgres.sh` is
  not needed. `MTC_POSTGRES_PASSWORD` is required (`${...:?...}`): because
  Compose interpolates the whole file, it must be set even to start another
  service alone.
- MongoDB runs without authentication on loopback. Enable authentication,
  with a user and `MTC_MONGODB_AUTH_SOURCE`, for any non-local deployment.
- Redis runs with `--appendonly yes --maxmemory-policy volatile-lru`.
- Qdrant runs with telemetry disabled.
- Neo4j needs `MTC_NEO4J_PASSWORD` (eight characters or more) before its
  service starts, and runs with a small heap and page cache (512 MB and
  256 MB); raise them for a large graph.
- Every optional variable has a default (`${VAR:-default}`) equal to the
  Settings default for `DB_PREFIX=mtc`.

## Start, initialize, check

From the project directory:

```bash
make db-up DB_SERVICES="postgres mongo qdrant"
mtc db init
mtc db check
```

`make db-up` runs
`docker compose --env-file .env -f docker/docker-compose.yml up -d --wait <services>`;
start only the services the setup enabled (the wizard's next steps print the
list). `mtc db init` applies the schema, records the install date and seeds
the stores; `mtc db check` verifies them. See [DATABASES.md](DATABASES.md).

`make db-down` runs `docker compose ... down` and keeps the volumes.

## Ports

| What | Setting | Default |
|---|---|---|
| PostgreSQL | `MTC_POSTGRES_PORT` | 5432 |
| MongoDB | `MTC_MONGODB_PORT` | 27017 |
| Redis | `MTC_REDIS_PORT` | 6379 |
| Qdrant (HTTP) | `MTC_QDRANT_PORT` | 6333 |
| Neo4j (bolt) | part of `MTC_NEO4J_URI` | 7687 |
| The model server (on the host) | `MTC_LLM_PORT` | 8080 |
| The MTC API (`mtc serve`, on the host) | `MTC_API_PORT` | 8200 |

## Troubleshooting

- **A port is already in use.** Another service, perhaps a store installed
  on the host, already listens on that port. Stop it, or set the port in
  `.env` (for example `MTC_POSTGRES_PORT=5433`); both Compose and MTC read
  the same key. `scripts/ops/harden_services.sh` lists the listeners.
- **A healthcheck fails.** `docker compose --env-file .env -f
  docker/docker-compose.yml ps` shows the state and
  `docker compose ... logs <service>` the reason. Neo4j refuses to start
  without a valid password; PostgreSQL needs `MTC_POSTGRES_PASSWORD`.
- **`docker compose down -v` deletes data.** The `-v` flag removes the named
  volumes, and with them every memory, conversation and thought. The Makefile
  never passes it. Back up first ([OPERATIONS.md](OPERATIONS.md)).
- **`mtc db check` reports `unreachable`.** The service is not up yet or
  listens on another port; run `mtc doctor` for a summary of every store.
