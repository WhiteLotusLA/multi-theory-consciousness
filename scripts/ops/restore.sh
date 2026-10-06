#!/usr/bin/env bash
# Restore one store from a backup that backup.sh wrote.
#
# A restore replaces data, so it asks for the target's name to be typed (or
# --yes), checks the file against the manifest's checksum first, and restores
# one store per run. Passwords reach the tools through their environment
# variables or a private options file, never on a command line. Settings come
# from the project's Settings through `python3 -m mtc.core.env_export`. Run it
# from the project directory with the project's virtual environment active.
set -euo pipefail

SCRIPT_NAME="restore.sh"
umask 077

usage() {
  cat <<'EOF'
Usage: scripts/ops/restore.sh --from DIR --store STORE
                              [--target-suffix SUFFIX] [--yes] [--dry-run]

Restores STORE (postgres, mongodb, qdrant, redis or neo4j) from the backup
directory DIR (one timestamped directory under BACKUP_LOCATION).

  postgres  pg_restore --clean --if-exists into POSTGRES_DB
  mongodb   mongorestore --archive --gzip --drop into MONGODB_DB
  qdrant    the snapshot upload API, into QDRANT_COLLECTION
  redis     prints the manual steps (working memory expires by design)
  neo4j     neo4j-admin database load; Neo4j must be stopped first

Options:
  --target-suffix SUFFIX  restore into <name>SUFFIX instead of the configured
                          database or collection (letters, digits and
                          underscores), leaving the live data untouched
  --yes                   do not ask for the target's name to be typed
  --dry-run               print the command; restore nothing
  --help                  show this text

Exit codes: 0 restored, 1 the restore failed or the file does not match the
manifest, 2 usage, settings that cannot be read or no confirmation, 130
interrupted.
EOF
}

have() {
  command -v "$1" >/dev/null 2>&1
}

tcp_open() {
  # True when something accepts TCP connections at host $1, port $2.
  (exec 3<>"/dev/tcp/$1/$2") >/dev/null 2>&1
}

sha256_of() {
  if have shasum; then
    shasum -a 256 "$1" | cut -d ' ' -f 1
  else
    python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}

manifest_sha() {
  # manifest_sha DIR NAME: the checksum the manifest records for a file.
  local kind name bytes sha
  while read -r kind name bytes sha; do
    if [ "${kind}" = "file" ] && [ "${name}" = "$2" ]; then
      printf '%s\n' "${sha}"
      return 0
    fi
    : "${bytes}"
  done <"$1/MANIFEST.txt"
  return 1
}

pg_query() {
  # pg_query DATABASE SQL: one unaligned answer; the password is in PGPASSWORD.
  PGPASSWORD="${MTC_POSTGRES_PASSWORD}" psql -At -h "${MTC_POSTGRES_HOST}" \
    -p "${MTC_POSTGRES_PORT}" -U "${MTC_POSTGRES_USER}" -d "$1" -c "$2" 2>/dev/null
}

run_or_plan() {
  # Run the command, or print it under --dry-run.
  if [ "${DRY_RUN}" -eq 1 ]; then
    printf 'would run:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

restore_postgres() {
  # restore_postgres FILE TARGET FRESH: FRESH is 1 for a suffixed target.
  local file="$1" target="$2" fresh="$3" found
  if [ "${fresh}" -eq 1 ]; then
    found=""
    if [ "${DRY_RUN}" -eq 0 ]; then
      found="$(pg_query postgres "SELECT 1 FROM pg_database WHERE datname = '${target}'" || true)"
    fi
    if [ "${found}" != "1" ]; then
      PGPASSWORD="${MTC_POSTGRES_PASSWORD}" run_or_plan createdb \
        -h "${MTC_POSTGRES_HOST}" -p "${MTC_POSTGRES_PORT}" \
        -U "${MTC_POSTGRES_USER}" "${target}" || return 1
    fi
  fi
  PGPASSWORD="${MTC_POSTGRES_PASSWORD}" run_or_plan pg_restore --clean --if-exists \
    --no-owner -h "${MTC_POSTGRES_HOST}" -p "${MTC_POSTGRES_PORT}" \
    -U "${MTC_POSTGRES_USER}" -d "${target}" "${file}"
}

MONGO_OPTIONS=""

remove_mongo_options() {
  # The private options file holds a password: remove it on every exit path.
  if [ -n "${MONGO_OPTIONS}" ]; then
    rm -f "${MONGO_OPTIONS}"
    MONGO_OPTIONS=""
  fi
}

on_interrupt() {
  remove_mongo_options
  echo "${SCRIPT_NAME}: interrupted; the restore may be incomplete" >&2
  exit 130
}

restore_mongodb() {
  local file="$1" target="$2" arguments code=0
  arguments=(--host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}"
    "--archive=${file}" --gzip --drop
    --nsFrom "${MTC_MONGODB_DB}.*" --nsTo "${target}.*")
  if [ -n "${MTC_MONGODB_USER}" ] && [ -n "${MTC_MONGODB_PASSWORD}" ]; then
    arguments+=(--username "${MTC_MONGODB_USER}"
      --authenticationDatabase "${MTC_MONGODB_AUTH_SOURCE:-${MTC_MONGODB_DB}}")
    if [ "${DRY_RUN}" -eq 0 ]; then
      # The tools read the password from a private options file, removed
      # here and by the exit and interrupt traps.
      MONGO_OPTIONS="$(mktemp "${TMPDIR:-/tmp}/mtc-mongo-tools.XXXXXX")"
      printf "password: '%s'\n" "$(printf '%s' "${MTC_MONGODB_PASSWORD}" | sed "s/'/''/g")" >"${MONGO_OPTIONS}"
      arguments+=(--config "${MONGO_OPTIONS}")
    fi
  fi
  run_or_plan mongorestore "${arguments[@]}" || code=$?
  remove_mongo_options
  return "${code}"
}

restore_qdrant() {
  local file="$1" target="$2"
  run_or_plan curl --silent --fail --max-time 3600 -X POST \
    "http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/collections/${target}/snapshots/upload?priority=snapshot" \
    -H "Content-Type: multipart/form-data" -F "snapshot=@${file}" >/dev/null
}

restore_neo4j() {
  local file="$1"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "would run: neo4j-admin database load --from-stdin --overwrite-destination=true ${MTC_NEO4J_DATABASE} < ${file}"
    return 0
  fi
  neo4j-admin database load --from-stdin --overwrite-destination=true \
    "${MTC_NEO4J_DATABASE}" <"${file}"
}

redis_steps() {
  cat <<EOF
Redis is not restored by this script: working memory and caches expire by
design, and a running Redis would overwrite the file on its next save.
To put $1 back by hand:
  1. stop Redis (scripts/ops/stop_stack.sh --all, or your service manager);
  2. copy the file over dump.rdb in the Redis data directory (CONFIG GET dir
     names it) and, when append-only persistence is on, remove the
     append-only files so the snapshot is what loads;
  3. start Redis again and check: mtc db check --only redis
EOF
}

main() {
  local from="" store="" suffix="" assume_yes=0
  DRY_RUN=0
  trap remove_mongo_options EXIT
  trap on_interrupt INT TERM HUP
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --help | -h)
        usage
        exit 0
        ;;
      --from | --store | --target-suffix)
        if [ "$#" -lt 2 ] || [ -z "$2" ]; then
          echo "${SCRIPT_NAME}: $1 needs a value" >&2
          exit 2
        fi
        case "$1" in
          --from) from="$2" ;;
          --store) store="$2" ;;
          --target-suffix) suffix="$2" ;;
        esac
        shift
        ;;
      --yes) assume_yes=1 ;;
      --dry-run) DRY_RUN=1 ;;
      *)
        echo "${SCRIPT_NAME}: unknown option $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done
  if [ -z "${from}" ] || [ -z "${store}" ]; then
    echo "${SCRIPT_NAME}: --from DIR and --store STORE are both needed" >&2
    usage >&2
    exit 2
  fi
  case "${store}" in
    postgres | mongodb | qdrant | redis | neo4j) ;;
    *)
      echo "${SCRIPT_NAME}: --store is one of postgres, mongodb, qdrant, redis, neo4j" >&2
      exit 2
      ;;
  esac
  case "${suffix}" in
    *[!A-Za-z0-9_]*)
      echo "${SCRIPT_NAME}: --target-suffix takes letters, digits and underscores" >&2
      exit 2
      ;;
  esac

  local settings
  if ! settings="$(python3 -m mtc.core.env_export --secrets PROJECT_ROOT \
    POSTGRES_HOST POSTGRES_PORT POSTGRES_USER POSTGRES_DB POSTGRES_PASSWORD \
    MONGODB_HOST MONGODB_PORT MONGODB_DB MONGODB_USER MONGODB_PASSWORD \
    MONGODB_AUTH_SOURCE QDRANT_HOST QDRANT_PORT QDRANT_COLLECTION NEO4J_URI \
    NEO4J_DATABASE)"; then
    echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
    exit 2
  fi
  eval "${settings}"

  if [ ! -d "${from}" ] || [ ! -f "${from}/MANIFEST.txt" ]; then
    echo "${SCRIPT_NAME}: ${from} is not a backup directory (no MANIFEST.txt)" >&2
    exit 2
  fi
  local file target live recorded
  case "${store}" in
    postgres)
      file="postgres.dump"
      live="${MTC_POSTGRES_DB}"
      ;;
    mongodb)
      file="mongo.archive.gz"
      live="${MTC_MONGODB_DB}"
      ;;
    qdrant)
      file="qdrant-${MTC_QDRANT_COLLECTION}.snapshot"
      live="${MTC_QDRANT_COLLECTION}"
      ;;
    redis)
      file="redis.rdb"
      live="redis"
      ;;
    neo4j)
      file="neo4j.dump"
      live="${MTC_NEO4J_DATABASE}"
      ;;
  esac
  if [ ! -f "${from}/${file}" ]; then
    echo "${SCRIPT_NAME}: this backup holds no ${file}" >&2
    exit 2
  fi
  if ! recorded="$(manifest_sha "${from}" "${file}")"; then
    echo "${SCRIPT_NAME}: the manifest does not list ${file}; not restoring a file it cannot vouch for" >&2
    exit 1
  fi
  if [ "$(sha256_of "${from}/${file}")" != "${recorded}" ]; then
    echo "${SCRIPT_NAME}: ${file} does not match the checksum in the manifest; nothing was restored" >&2
    exit 1
  fi

  if [ "${store}" = "redis" ]; then
    redis_steps "${from}/${file}"
    exit 0
  fi
  if [ "${store}" = "neo4j" ] && [ -n "${suffix}" ]; then
    echo "${SCRIPT_NAME}: --target-suffix is not available for neo4j (the community edition holds one database)" >&2
    exit 2
  fi
  target="${live}${suffix}"

  if [ "${store}" = "neo4j" ] && [ "${DRY_RUN}" -eq 0 ]; then
    local hostport="${MTC_NEO4J_URI#*://}"
    hostport="${hostport%%/*}"
    if tcp_open "${hostport%%:*}" "${hostport##*:}"; then
      echo "${SCRIPT_NAME}: Neo4j is running. Stop it first: a database is loaded only while the server is stopped. Nothing was restored." >&2
      exit 1
    fi
  fi

  if [ "${assume_yes}" -eq 0 ] && [ "${DRY_RUN}" -eq 0 ]; then
    local answer=""
    echo "This replaces the ${store} data in ${target} with ${from}/${file}." >&2
    printf 'Type the target name (%s) to continue: ' "${target}" >&2
    IFS= read -r answer || answer=""
    if [ "${answer}" != "${target}" ]; then
      echo "${SCRIPT_NAME}: not confirmed; nothing was restored" >&2
      exit 2
    fi
  fi

  local code=0 fresh=0
  if [ -n "${suffix}" ]; then
    fresh=1
  fi
  case "${store}" in
    postgres) restore_postgres "${from}/${file}" "${target}" "${fresh}" || code=$? ;;
    mongodb) restore_mongodb "${from}/${file}" "${target}" || code=$? ;;
    qdrant) restore_qdrant "${from}/${file}" "${target}" || code=$? ;;
    neo4j) restore_neo4j "${from}/${file}" || code=$? ;;
  esac
  if [ "${code}" -ne 0 ]; then
    echo "${SCRIPT_NAME}: the ${store} restore into ${target} failed" >&2
    exit 1
  fi
  if [ "${DRY_RUN}" -eq 0 ]; then
    echo "${store}: restored into ${target}"
  fi
  exit 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
