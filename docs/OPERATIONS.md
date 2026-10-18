# Operations

Running an MTC stack day to day: checks before a start, starting and
stopping, listeners and file modes, backups, restores and the weekly drill,
watchdogs, and the lessons the scripts are built around. Every script under
`scripts/ops/` reads its settings through `python3 -m mtc.core.env_export`
(run it from the project directory with the virtual environment active), has
`--help`, and has a `--dry-run` that prints what it would run. No script
forces a process to stop or removes a path that has not passed a safety
check.

These scripts were tested against stub commands and dummy processes, not run
against live services on the development machine.

## Pre-flight

```bash
mtc doctor             # settings, secrets set or empty, stores, brain, extras, CTM
mtc db check           # the enabled stores against the schema registry
```

`mtc doctor --strict` and `mtc db check --strict` count warnings and
`insufficient_data` as failures, for a gate before a release or a long run.

## Start and stop

```bash
scripts/ops/start_stack.sh --dry-run
scripts/ops/start_stack.sh
scripts/ops/stop_stack.sh
scripts/ops/stop_stack.sh --all
```

**Start.** For each enabled store in the order PostgreSQL, Qdrant, MongoDB,
Redis, Neo4j, `start_stack.sh` checks its health and starts it only when it
does not answer, through `SERVICE_MANAGER` (`brew`, `docker` or `systemd`;
with `none` it prints the command and counts the store as down), then waits
up to `--store-wait` (60 seconds). With `LLM_PROVIDER=local` it checks the
model server and, when it is down and `LLM_BACKEND` is set, starts it with
`start_llm.sh --background` and waits up to `--brain-wait` (420 seconds: a
cold load of a large model takes minutes) for `/v1/models`. Then it starts
`mtc serve` in the background unless the API already answers, and polls
`/health` up to `--api-wait` (150 seconds). PID files go under
`<DATA_DIRECTORY>/run` (`api.pid`, `llm.pid`, `qdrant.pid`), logs under
`LOG_DIRECTORY` (`api.log`, `llm.log`, `qdrant.log`). It exits 0 only when
every enabled component is healthy.

What each manager runs: `brew` uses `brew services start` with
`postgresql@17`, `mongodb-community`, `redis` and `neo4j`, and runs Qdrant
from its release binary (`qdrant --config-path config/qdrant.yaml`);
`docker` uses `docker compose --env-file .env -f docker/docker-compose.yml
up -d <service>` (and `stop <service>` to stop, never `down`); `systemd` uses
`systemctl start` with the units `postgresql`, `qdrant`, `mongod`,
`redis-server` and `neo4j`. The Compose path was reviewed against official
documentation and not run on the development machine; the path run on the
development machine is the in-memory install with the mock provider.

**Stop.** `stop_stack.sh` stops the API server, then the model server it
started, each with the terminate signal and a grace period (`--grace`, 30
seconds). With `--all` it also stops the stores in reverse order through the
service manager. A process still running after the grace period is reported
and the exit code is 1; it is never forced, because the API server saves the
persistence hub and ends open conversations while it shuts down (see
[RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md)). Wait, then run the stop again,
or find out why it is slow. A PID file whose process now runs another program is
not signalled.

A restart is a stop followed by a start.

When the runtime stops, each shutdown step (the background tasks, the open
conversations, the hub save, the neural orchestrator, the memory stack, each
store client, the brain) runs on its own: a step that fails is logged at
WARNING with its name and the error type and recorded in the stop report's
`errors`, the later steps still run, and the runtime always ends stopped.
With PostgreSQL enabled, the PostgreSQL store report's `in_memory_parts` is
empty when the conversation repository, the snapshot store and the safety
audit store were built on the pool; when building them failed, all three run
in memory, the list names them (`conversations`, `snapshots`,
`safety_audit`) and the failure is logged at ERROR by type.

## Safety suspension and resume

An emergency-level safety score (manipulation or deception at or above 0.8)
in an untrusted speaker's message is handled by
`EMERGENCY_SHUTDOWN_ENABLED`:

- With it false (the default) the message is denied, an audit row records
  `action_proposed` `emergency_shutdown` with `enacted` false, and one
  `URGENT` alert titled "Emergency shutdown not enacted" goes to the
  escalation role (`SAFETY_ESCALATION_ROLE`). Nothing is suspended.
- With it true the same score suspends every untrusted conversation: each
  later message of an untrusted speaker is refused with the decision
  `emergency_shutdown`, and `mtc chat` prints a second line naming
  `mtc safety resume`. Trusted speakers are not affected.

```bash
mtc safety status            # safety stack, safety mode, conversations suspended
mtc safety resume            # as the creator; --speaker NAME for another trusted speaker
```

`mtc safety status` reads `GET /api/consciousness/status` from the running
server and prints three lines (`safety stack: available`, `not available`
or `unknown`; `safety mode: ...`; `conversations suspended: yes`, `no` or
`unknown`). `mtc safety resume` calls `POST /api/safety/resume` with the
speaker's token, as `mtc chat` does; the server answers 403 unless the
speaker is verified as trusted. See [SECURITY.md](SECURITY.md).

## Listeners and file modes

```bash
scripts/ops/harden_services.sh
scripts/ops/harden_services.sh --fix-permissions
```

The script lists the TCP listeners on the five store ports, the model server
port and the API port, and reports every one that is not bound to loopback
with the configuration line that binds it there (PostgreSQL
`listen_addresses`, MongoDB `net.bindIp`, Redis `bind` with `protected-mode
yes`, Neo4j's default listen address, Qdrant's service host). `lsof` without
root lists only the current user's sockets, so an expected service (an
enabled store, the model server under the local provider, the API) with no
visible listener is named as not measured rather than counted as loopback;
run it as root to see every listener. It never edits
a store's configuration: that is an operator decision. It reports `.env` and
`config/speakers.yaml` looser than 0600 and the agent space and backup
location looser than 0700, and corrects them only with `--fix-permissions`.
Any finding exits 1.

## Health checks and watchdogs

```bash
scripts/ops/health_check.sh --quiet --log-file logs/health.log
python3 scripts/ops/backup_watchdog.py --alert
```

Both are alert-only. `health_check.sh` prints one line per enabled component
(healthy, DOWN, or "port open" when the only check that could run was a TCP
connect: the store's client tool is not installed, and always for Neo4j, so
health was not measured) and exits 1 when any is down; it never restarts
anything. Its log line reads `ok`, `unverified` (nothing down, some port open
only) or `UNHEALTHY`. `start_stack.sh` labels a store the same way.
`backup_watchdog.py` checks that the newest backup is fresh (26 hours by
default) and holds every expected file and a manifest without a failed dump;
with `--alert` it raises a backup alert through the configured notifiers
(console and file). The runtime's safety status reads the same backup status
(`mtc.monitoring.backup_status`).

Run them from a timer (cron, a launch agent or a systemd timer) every few
minutes and every few hours respectively. Deciding what to do about a down
component is the operator's call; an automatic restart loop hides the
problem it restarts.

`scripts/ops/redis_monitor.py` (`quick`, `watch` or `snapshot`) reports Redis
memory, cache hit rate and the keyspace when Redis is enabled.

## Backups

```bash
scripts/ops/backup.sh --dry-run
scripts/ops/backup.sh
```

Each run writes `<BACKUP_LOCATION>/<UTC timestamp>/` (mode 0700):

| File | Made with | When |
|---|---|---|
| `postgres.dump` | `pg_dump -Fc` | PostgreSQL enabled |
| `mongo.archive.gz` | `mongodump --archive --gzip` of `MONGODB_DB` | MongoDB enabled |
| `qdrant-<collection>.snapshot` | the snapshot API, then the download | Qdrant enabled |
| `redis.rdb` | `redis-cli --rdb` | Redis enabled |
| `neo4j.dump` | `neo4j-admin database dump` | only with `--include-neo4j-offline` and Neo4j stopped (the community edition has no online backup) |
| `state.tar.gz` | the agent space, the persona, the speakers file, the CTM persona and seed packs | unless `--no-state`; never `.env` |
| `MANIFEST.txt` | sizes, SHA-256 sums, row, document and point counts | always |

Passwords reach the dump tools through their environment variables
(`PGPASSWORD`, `REDISCLI_AUTH`) or, for the MongoDB tools, a private options
file removed afterwards (also when the run is interrupted), never a command
line. With MongoDB credentials the
manifest records the document count as `unknown`, and the drill then does not
cover MongoDB. `redis-cli --rdb` copies the whole Redis instance, not only
MTC's keys. One failed dump does not stop
the others; the run then exits 1 and skips retention, so a run of failing
backups never ages out a good one. After a clean run, backups older than
`BACKUP_RETENTION_DAYS` (30) are removed, always keeping the newest
`BACKUP_KEEP_MINIMUM` (5) complete backups (one whose manifest records a
failed dump does not count toward the minimum), and only inside a backup
location that passed the safety check (not empty, not the file system root,
not the project directory or a directory that contains it). A state path
configured outside the project (an absolute `AGENT_SPACE_DIR` elsewhere, for
example) is left out of `state.tar.gz`; the output and the manifest
(`excluded state AGENT_SPACE_DIR outside_project`) say so, and that path needs
a backup of its own.

**Cadence.** Run `backup.sh` daily from a timer and `backup_watchdog.py`
after it. A backup holds conversation text, names, thoughts and beliefs: keep
`BACKUP_LOCATION` private and never on a cloud-synced or shared folder. The
`.env` file is not backed up; keep your secrets somewhere else you control.

## Restore

```bash
scripts/ops/restore.sh --from backups/<timestamp> --store postgres --dry-run
scripts/ops/restore.sh --from backups/<timestamp> --store postgres --target-suffix _check
scripts/ops/restore.sh --from backups/<timestamp> --store postgres
```

One store per run. The script checks the file against the manifest's
checksum first, then asks you to type the target's name (or `--yes`).
PostgreSQL restores with `pg_restore --clean --if-exists`, MongoDB with
`mongorestore --archive --gzip --drop`, Qdrant through the snapshot upload
API, Neo4j with `neo4j-admin database load` and only while Neo4j is stopped.
Redis is not restored (working memory expires by design); the script prints
the manual steps. `--target-suffix` restores beside the live data instead of
over it.

## The weekly drill

```bash
scripts/ops/restore_drill.sh
```

The drill restores the newest backup into `<POSTGRES_DB>_restore_test`,
`<MONGODB_DB>_restore_test` and `<QDRANT_COLLECTION>_restore_test`, compares
the counts with the manifest and drops the scratch targets again. It only
ever creates and drops names that end with `_restore_test`, and asserts that
before every create and drop. Exit 0 means every store in the backup was
restored and its count matched. Run it weekly: a backup that was never
restored is not yet a backup.

## Rollback

Releases are tagged. To roll back the code, check out the previous tag, run
`pip install -e ".[all]"` again, and restart. A release that changed a
measurement says so in [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md). The
migrations only create (every statement is `CREATE ... IF NOT EXISTS`), so
rolling the code back needs no schema step; restore a backup when the data
itself must go back. To roll back a model change, see "Changing the model" in [LLM_SETUP.md](LLM_SETUP.md).

## Lessons for service managers (launchd, systemd)

- **Logs never on external or cloud-synced volumes.** A disk that is not
  mounted or a sync client that holds a file lock stops a service from
  writing its log, and some service managers then refuse to start it.
- **Protected folders.** On macOS, a launch agent cannot read the protected
  user folders (Documents, Desktop, Downloads) without a permission grant.
  Keep the project, its scripts and its virtual environment outside them, or
  grant the permission deliberately.
- **Credentials outside the repository.** A service reads `.env` from the
  project directory; keep any other credential file out of the repository
  and out of the backups' state archive.
- **Never keep a model server alive with an automatic restart.** A server
  that runs out of memory and is restarted at once can loop and take the
  machine with it. Start it from `start_stack.sh` and let `health_check.sh`
  report it down.
- **Interval supervision.** Prefer a timer that runs a check every few
  minutes over a keep-alive flag.
- **Identical launch flags on every path.** The script, the service unit and
  any manual start should run the same command with the same flags, so a
  restart never brings up a differently configured server. `start_llm.sh`
  builds the command from Settings for this reason.
- **Long cold-start waits.** A large model can take minutes to load; set the
  service manager's start timeout and `--brain-wait` accordingly instead of
  letting it kill a slow start.

Sample units use placeholders; replace `/path/to/project` with the project
directory:

```ini
# systemd: /etc/systemd/system/mtc-health.service (a oneshot run by a timer)
[Service]
Type=oneshot
WorkingDirectory=/path/to/project
ExecStart=/path/to/project/.venv/bin/python3 -m mtc.cli.main doctor --strict
```

```xml
<!-- launchd: $HOME/Library/LaunchAgents/org.example.mtc-health.plist -->
<key>ProgramArguments</key>
<array>
  <string>/path/to/project/scripts/ops/health_check.sh</string>
  <string>--quiet</string>
</array>
<key>WorkingDirectory</key><string>/path/to/project</string>
<key>StartInterval</key><integer>300</integer>
```

These samples are illustrations and were not installed on the development
machine.

## Cache and disk growth

- **Model server caches.** A server that keeps prompt caches can grow with
  the number of conversations; see [LLM_SETUP.md](LLM_SETUP.md).
- **Disk.** The stores, `STATE_DIR`, `RESULTS_DIR`, the logs (rotated by
  `LOG_ROTATION_SIZE_MB` and `LOG_ROTATION_BACKUPS`) and the backups grow
  over time. The memory pruner removes weak memories, old telemetry and
  expired gists during sleep, and the backup retention removes old backups;
  watch the rest with the disk's own tools.
- **Night pruning deletes.** With the sleep scheduler on, each night sleep
  deletes memories the pruner scores as weak; set SLEEP_PRUNE_DRY_RUN=true to
  log what would be removed without deleting it. On a real pass the pruner
  reports the archive and delete counts the store confirmed; a pass whose
  delete the store confirmed for no row keeps the vector points and graph
  nodes of all its rows; once at least one row is confirmed, the index
  entries of every row of the pass are deleted.
