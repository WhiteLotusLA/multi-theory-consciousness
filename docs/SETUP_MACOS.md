# Setup with host services (Homebrew, apt)

This guide installs the stores as services on the machine itself, managed by
Homebrew on macOS or by systemd on Linux, and lets `scripts/ops/start_stack.sh`
start what is down. Every store is optional: without any of them MTC runs in
memory ([GETTING_STARTED.md](GETTING_STARTED.md), first path).

The Docker Compose alternative is in [SETUP_DOCKER.md](SETUP_DOCKER.md); that
file was reviewed against official documentation and not run on the
development machine, and the path run on the development machine is the
in-memory install with the mock provider.

## What was tested

Each step below carries one status: `tested` (run on a fresh machine by the
release's verification), `reviewed-not-run` (written from official
documentation and the code, not run on the development machine), or
`pending` (the fresh-machine verification has not recorded it yet). A test
(`tests/docs/test_setup_docs_honesty.py`) accepts only these three.

The fresh-machine verification of this release is pending; the lines below
are updated from its report when the phase closes.

| Step | Status |
|---|---|
| Python virtual environment and `pip install` | pending |
| `mtc init` from an answers file with the mock provider | pending |
| `mtc doctor` | pending |
| `mtc doctor --fetch-models` | pending |
| the full pipeline demo | pending |
| `mtc assess` (quick mode) | pending |
| `mtc serve`, `mtc chat` and `mtc ctm status` | pending |
| Homebrew store installs and `brew services` | reviewed-not-run |
| the Qdrant release binary | reviewed-not-run |
| `scripts/db/init_postgres.sh` | reviewed-not-run |
| `mtc db init` and `mtc db check` against host services | reviewed-not-run |
| `scripts/ops/start_stack.sh` and `stop_stack.sh` | reviewed-not-run |
| the Linux apt and systemd notes | reviewed-not-run |

## 1. Python and MTC

```bash
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

`all` installs every extra except `iit`. For exact IIT 4.0 Phi on Python 3.12
or later, add `pip install -e ".[iit]"` (PyPhi is GPLv3 or later; see
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)).

## 2. Configure with the wizard

```bash
mtc init
```

Choose the stores you want and the `brew` service manager (`systemd` on
Linux). The wizard writes the persona, the speakers file and `.env` with
explicit store names, ports and generated passwords. See
[ONBOARDING.md](ONBOARDING.md).

## 3. Install the stores (Homebrew)

These formulas and commands come from the vendors' and Homebrew's own pages;
they were not run on the development machine.

| Store | Install | Start as a service |
|---|---|---|
| PostgreSQL 17 | `brew install postgresql@17` | `brew services start postgresql@17` |
| MongoDB | `brew tap mongodb/brew`, `brew trust mongodb/brew`, `brew update`, `brew install mongodb-community` | `brew services start mongodb-community` |
| Redis | `brew install redis` | `brew services start redis` |
| Neo4j | `brew install neo4j` | `brew services start neo4j` |
| Qdrant | no formula; the release binary (below) or a container | `scripts/ops/start_stack.sh` runs the binary |

Notes on versions: `start_stack.sh` starts the formula names in this table.
MongoDB's installation page shows the versioned `mongodb-community@8.0`; the
unversioned `mongodb-community` in the same tap follows the current release.
If you install a versioned formula, start it with `brew services` yourself.
Homebrew's `redis` and `neo4j` formulas install the
vendors' current releases, which can be newer than the `redis:7` and
`neo4j:5-community` images the Compose file pins. MTC's Neo4j constraints use
the Neo4j 5 `CREATE CONSTRAINT ... IF NOT EXISTS ... REQUIRE` syntax; whether
a newer Neo4j release accepts it unchanged was not verified here, so run
`mtc db check` after `mtc db init`.

**Qdrant.** Download the release archive for your platform from the Qdrant
project's GitHub releases page, check it against a checksum when the release
publishes one (the asset names and checksums were not verified for this
guide), unpack the `qdrant` binary onto your `PATH`, and let
`scripts/ops/start_stack.sh` start it with
`qdrant --config-path config/qdrant.yaml` from the project directory (its
storage and snapshots then live under `./data/qdrant/`, ignored by git;
telemetry is off). Alternatively run Qdrant alone from the Compose file.

**Neo4j.** `config/neo4j.conf.example` sets a loopback listen address, the
default database `neo4j`, a small heap and page cache and authentication on.
Copy it as `neo4j.conf` into a directory and point `NEO4J_CONF` at that
directory, then set the initial password with Neo4j's own admin tool to the
value the wizard generated as `MTC_NEO4J_PASSWORD` in `.env` (copy it from
the file; do not choose a new one, or the two disagree and the client is
refused). The community edition holds one graph: give MTC its own Neo4j
server.

**MongoDB.** Without a MongoDB user, MTC connects without credentials, as
the Homebrew service accepts on loopback. If you gave the wizard a MongoDB
user, nothing creates the MongoDB user: no script provisions it. Create it
once in `mongosh`, connected as an administrator to the database
`MTC_MONGODB_AUTH_SOURCE` names (`MTC_MONGODB_DB` when it is empty), with
`db.createUser({user: "<MTC_MONGODB_USER>", pwd: passwordPrompt(), roles:
[{role: "readWrite", db: "<MTC_MONGODB_DB>"}]})`, and paste the
`MTC_MONGODB_PASSWORD` value from `.env` at the prompt. Turning on
authentication in `mongod.conf` is the operator's choice; MTC only needs the
user to exist.

## 4. Create the PostgreSQL role and database

```bash
scripts/db/init_postgres.sh --dry-run   # prints the statements, never the password
scripts/db/init_postgres.sh
```

The script reads `MTC_DB_PREFIX` and the `MTC_POSTGRES_*` lines of `.env`,
connects as the administrator through the standard `PG*` variables (on a
Homebrew install your own user is the administrator), creates the role and the
database when they do not exist, never changes an existing role's password,
never drops anything, and applies `migrations/postgres/*.sql` as the new role.
With an empty password it creates the role for trust authentication and
warns.

## 5. Initialize and check the stores

```bash
mtc db init
mtc db check
```

See [DATABASES.md](DATABASES.md) for what each step creates and the statuses
it reports.

## 6. Start and stop the stack

```bash
scripts/ops/start_stack.sh --dry-run    # what it would check and start
scripts/ops/start_stack.sh              # stores, the model server, the API
scripts/ops/stop_stack.sh               # the API and the model server
scripts/ops/stop_stack.sh --all         # and the stores, in reverse order
```

`start_stack.sh` reads every setting through `python3 -m mtc.core.env_export`
(activate the virtual environment first), probes each enabled store and
starts the ones that are down through `SERVICE_MANAGER` (`brew services` or
`systemctl`; with `none` it prints the command and counts the store as
down), starts the model server with `scripts/ops/start_llm.sh` when
`LLM_PROVIDER` is `local` and `LLM_BACKEND` is set, starts `mtc serve`, and
exits 0 only when no enabled component is down. A store it can check only by
its port (its client tool is not installed, and always Neo4j) reads "port
open" with the reason, not healthy. See [OPERATIONS.md](OPERATIONS.md).

## 7. The embedding model

```bash
mtc doctor                  # reports whether the model is in the local cache
mtc doctor --fetch-models   # downloads EMBEDDING_MODEL once
export HF_HUB_OFFLINE=1     # then nothing is fetched again
```

## Linux (apt and systemd)

The same steps apply with the distribution's packages and the `systemd`
service manager: PostgreSQL 14 or later from the distribution or the
PostgreSQL project's apt repository (the administrator is the `postgres`
system user, so run `init_postgres.sh` as that user or with `PGUSER` and
`PGPASSWORD` set), Redis from the distribution, MongoDB and Neo4j from their
vendors' apt repositories, and Qdrant from its release binary or a container.
`start_stack.sh` then calls `systemctl start <unit>`. These notes were not run
on the development machine.

After the services are up: `mtc doctor`, `mtc serve`, `mtc chat`.
