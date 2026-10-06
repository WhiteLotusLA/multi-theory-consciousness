#!/usr/bin/env bash
# Back up the enabled stores and the private state into BACKUP_LOCATION.
#
# One timestamped directory per run, private to the user (mode 0700), with a
# manifest of sizes, checksums and counts. Passwords reach the dump tools
# through their environment variables or a private options file, never on a
# command line. Settings come from the project's Settings through
# `python3 -m mtc.core.env_export`. Run it from the project directory with the
# project's virtual environment active.
#
# A backup holds conversation text, names, thoughts and beliefs: keep it as
# private as the data itself. The environment file is never part of it; keep
# the secrets somewhere else.
set -euo pipefail

SCRIPT_NAME="backup.sh"
umask 077

usage() {
  cat <<'EOF'
Usage: scripts/ops/backup.sh [--dest DIR] [--include-neo4j-offline]
                             [--no-state] [--dry-run]

Writes <BACKUP_LOCATION>/<UTC timestamp>/ (mode 0700) with, for each enabled
store:

  postgres.dump                 pg_dump -Fc
  mongo.archive.gz              mongodump --archive --gzip of MONGODB_DB
  qdrant-<collection>.snapshot  the snapshot API, then the download
  redis.rdb                     redis-cli --rdb
  neo4j.dump                    only with --include-neo4j-offline and a
                                stopped Neo4j (the community edition has no
                                online backup)
  state.tar.gz                  the agent space, the persona, the speakers
                                file and the continuous-thought persona and
                                seed packs (never the environment file)
  MANIFEST.txt                  sizes, SHA-256 sums and row, document and
                                point counts

Afterwards, backups older than BACKUP_RETENTION_DAYS are removed, always
keeping the newest BACKUP_KEEP_MINIMUM complete backups (a backup whose
manifest records a failed dump does not count toward the minimum), and only
inside a backup location that passed the safety check (not empty, not the
file system root, not the project directory or a directory that contains
it). When a dump of this run failed, retention is skipped and nothing is
removed.

A state path configured outside the project (for example an absolute
AGENT_SPACE_DIR elsewhere) is left out of the state archive; the output and
the manifest ("excluded state KEY outside_project") say so.

Options:
  --dest DIR                write under DIR instead of BACKUP_LOCATION
  --include-neo4j-offline   dump Neo4j too; it must be stopped first
  --no-state                leave the state archive out
  --dry-run                 print what would run; write and remove nothing
  --help                    show this text

Exit codes: 0 every dump succeeded, 1 a dump or the retention failed (the
other dumps still ran), 2 usage, settings that cannot be read or an unsafe
backup location, 130 interrupted.
EOF
}

have() {
  command -v "$1" >/dev/null 2>&1
}

tcp_open() {
  # True when something accepts TCP connections at host $1, port $2.
  (exec 3<>"/dev/tcp/$1/$2") >/dev/null 2>&1
}

physical_path() {
  # The path with every symbolic link and ".." resolved; it need not exist.
  python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

assert_safe_location() {
  # assert_safe_location CONFIGURED RESOLVED PROJECT: succeed only for a
  # backup location retention may work in.
  local configured="$1" resolved="$2" project="$3"
  if [ -z "${configured}" ] || [ -z "${resolved}" ]; then
    echo "${SCRIPT_NAME}: the backup location is empty" >&2
    return 1
  fi
  if [ "${resolved}" = "/" ]; then
    echo "${SCRIPT_NAME}: the backup location is the file system root" >&2
    return 1
  fi
  if [ "${resolved}" = "${project}" ]; then
    echo "${SCRIPT_NAME}: the backup location is the project directory itself" >&2
    return 1
  fi
  case "${project}/" in
    "${resolved}/"*)
      echo "${SCRIPT_NAME}: the backup location contains the project directory" >&2
      return 1
      ;;
  esac
  case "${configured}" in
    /*) ;;
    *)
      case "${resolved}/" in
        "${project}/"*) ;;
        *)
          echo "${SCRIPT_NAME}: a relative backup location must stay inside the project, and this one resolves outside it" >&2
          return 1
          ;;
      esac
      ;;
  esac
  return 0
}

is_backup_name() {
  case "$1" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) return 0 ;;
    *) return 1 ;;
  esac
}

assert_inside() {
  # assert_inside CANDIDATE PARENT: succeed only when the candidate is a real
  # directory, named like a backup, directly inside the resolved parent.
  local candidate="$1" parent="$2" name
  name="$(basename "${candidate}")"
  if ! is_backup_name "${name}"; then
    return 1
  fi
  if [ ! -d "${candidate}" ] || [ -L "${candidate}" ]; then
    return 1
  fi
  [ "$(physical_path "${candidate}")" = "${parent}/${name}" ]
}

list_backups() {
  # The backup directory names under $1, one per line.
  local entry
  for entry in "$1"/*/; do
    [ -d "${entry}" ] || continue
    entry="${entry%/}"
    entry="${entry##*/}"
    if is_backup_name "${entry}"; then
      printf '%s\n' "${entry}"
    fi
  done
}

is_complete_backup() {
  # True when backup directory $1 has a manifest with no failed dump in it.
  [ -f "$1/MANIFEST.txt" ] || return 1
  ! grep -Eq '^status [^ ]+ failed$' "$1/MANIFEST.txt"
}

apply_retention() {
  # apply_retention LOCATION DAYS KEEP: remove old backups; the newest KEEP
  # complete backups are kept whatever their age, and an incomplete one (a
  # failed dump, or no manifest) never counts toward that minimum.
  local location="$1" days="$2" keep="$3" cutoff seen name
  cutoff="$(python3 -c 'import datetime, sys; print((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(sys.argv[1]))).strftime("%Y%m%dT%H%M%SZ"))' "${days}")"
  seen=0
  while IFS= read -r name; do
    [ -n "${name}" ] || continue
    if is_complete_backup "${location}/${name}"; then
      seen=$((seen + 1))
      if [ "${seen}" -le "${keep}" ]; then
        continue
      fi
    fi
    if [ "${name}" \< "${cutoff}" ]; then
      if [ "${DRY_RUN}" -eq 1 ]; then
        echo "retention: would remove ${name} (older than ${days} days)"
        continue
      fi
      if assert_inside "${location}/${name}" "${location}"; then
        rm -rf "${location:?}/${name:?}"
        echo "retention: removed ${name} (older than ${days} days)"
      else
        echo "retention: ${name} failed the containment check and was left alone" >&2
        FAILED=1
      fi
    fi
  done <<EOF
$(list_backups "${location}" | sort -r)
EOF
}

sha256_of() {
  if have shasum; then
    shasum -a 256 "$1" | cut -d ' ' -f 1
  else
    python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}

record_file() {
  # record_file DIR NAME: add a "file NAME BYTES SHA256" line to the manifest.
  local bytes
  bytes="$(wc -c <"$1/$2" | tr -d '[:space:]')"
  printf 'file %s %s %s\n' "$2" "${bytes}" "$(sha256_of "$1/$2")" >>"$1/MANIFEST.txt"
}

json_field() {
  # json_field KEY...: print one value of the JSON document on stdin.
  python3 -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1:]:
    value = value[key]
print(value)
' "$@" 2>/dev/null
}

pg_query() {
  # pg_query DATABASE SQL: one unaligned answer; the password is in PGPASSWORD.
  PGPASSWORD="${MTC_POSTGRES_PASSWORD}" psql -At -h "${MTC_POSTGRES_HOST}" \
    -p "${MTC_POSTGRES_PORT}" -U "${MTC_POSTGRES_USER}" -d "$1" -c "$2" 2>/dev/null
}

count_postgres_rows() {
  # The rows in every table of database $1, or "unknown".
  local database="$1" tables table rows total
  if ! have psql; then
    echo "unknown"
    return 0
  fi
  if ! tables="$(pg_query "${database}" "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")"; then
    echo "unknown"
    return 0
  fi
  total=0
  while IFS= read -r table; do
    [ -n "${table}" ] || continue
    case "${table}" in
      *'"'*)
        echo "unknown"
        return 0
        ;;
    esac
    if ! rows="$(pg_query "${database}" "SELECT count(*) FROM \"${table}\"")"; then
      echo "unknown"
      return 0
    fi
    case "${rows}" in
      '' | *[!0-9]*)
        echo "unknown"
        return 0
        ;;
    esac
    total=$((total + rows))
  done <<EOF
${tables}
EOF
  echo "${total}"
}

count_mongo_documents() {
  # The documents in every collection of database $1, or "unknown".
  local database="$1" reply
  if ! have mongosh || [ -n "${MTC_MONGODB_USER}" ]; then
    # mongosh takes a password only on its command line, so with credentials
    # the count is left unknown instead.
    echo "unknown"
    return 0
  fi
  if ! reply="$(mongosh --quiet --host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}" \
    "${database}" --eval 'db.getCollectionNames().reduce((n, c) => n + db.getCollection(c).countDocuments({}), 0)' 2>/dev/null)"; then
    echo "unknown"
    return 0
  fi
  case "${reply}" in
    '' | *[!0-9]*) echo "unknown" ;;
    *) echo "${reply}" ;;
  esac
}

count_qdrant_points() {
  # The points in collection $1, or "unknown".
  local reply count
  if ! have curl; then
    echo "unknown"
    return 0
  fi
  if ! reply="$(curl --silent --fail --max-time 30 \
    "http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/collections/$1")"; then
    echo "unknown"
    return 0
  fi
  if ! count="$(printf '%s' "${reply}" | json_field result points_count)"; then
    echo "unknown"
    return 0
  fi
  case "${count}" in
    '' | *[!0-9]*) echo "unknown" ;;
    *) echo "${count}" ;;
  esac
}

dump_postgres() {
  PGPASSWORD="${MTC_POSTGRES_PASSWORD}" pg_dump -Fc -h "${MTC_POSTGRES_HOST}" \
    -p "${MTC_POSTGRES_PORT}" -U "${MTC_POSTGRES_USER}" -d "${MTC_POSTGRES_DB}" \
    -f "$1/postgres.dump"
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
  echo "${SCRIPT_NAME}: interrupted; the backup in progress is incomplete" >&2
  exit 130
}

dump_mongodb() {
  local arguments code=0
  arguments=(--host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}"
    --db "${MTC_MONGODB_DB}" "--archive=$1/mongo.archive.gz" --gzip)
  if [ -n "${MTC_MONGODB_USER}" ] && [ -n "${MTC_MONGODB_PASSWORD}" ]; then
    # The tools read the password from a private options file outside the
    # backup, removed here and by the exit and interrupt traps.
    MONGO_OPTIONS="$(mktemp "${TMPDIR:-/tmp}/mtc-mongo-tools.XXXXXX")"
    printf "password: '%s'\n" "$(printf '%s' "${MTC_MONGODB_PASSWORD}" | sed "s/'/''/g")" >"${MONGO_OPTIONS}"
    arguments+=(--username "${MTC_MONGODB_USER}"
      --authenticationDatabase "${MTC_MONGODB_AUTH_SOURCE:-${MTC_MONGODB_DB}}"
      --config "${MONGO_OPTIONS}")
  fi
  mongodump "${arguments[@]}" || code=$?
  remove_mongo_options
  return "${code}"
}

dump_qdrant() {
  local base="http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/collections/${MTC_QDRANT_COLLECTION}/snapshots"
  local reply name
  if ! reply="$(curl --silent --fail --max-time 600 -X POST "${base}")"; then
    return 1
  fi
  if ! name="$(printf '%s' "${reply}" | json_field result name)"; then
    return 1
  fi
  case "${name}" in
    '' | *[!A-Za-z0-9._-]*) return 1 ;;
  esac
  if ! curl --silent --fail --max-time 3600 \
    --output "$1/qdrant-${MTC_QDRANT_COLLECTION}.snapshot" "${base}/${name}"; then
    return 1
  fi
  # The server keeps its own copy of the snapshot it just made; remove it.
  if ! curl --silent --fail --max-time 60 -X DELETE "${base}/${name}" >/dev/null; then
    echo "qdrant: the server-side copy ${name} could not be removed" >&2
  fi
}

dump_redis() {
  if [ -n "${MTC_REDIS_PASSWORD}" ]; then
    REDISCLI_AUTH="${MTC_REDIS_PASSWORD}" redis-cli -h "${MTC_REDIS_HOST}" \
      -p "${MTC_REDIS_PORT}" -n "${MTC_REDIS_DB}" --rdb "$1/redis.rdb" >/dev/null
  else
    redis-cli -h "${MTC_REDIS_HOST}" -p "${MTC_REDIS_PORT}" -n "${MTC_REDIS_DB}" \
      --rdb "$1/redis.rdb" >/dev/null
  fi
}

dump_neo4j() {
  neo4j-admin database dump --to-stdout "${MTC_NEO4J_DATABASE}" >"$1/neo4j.dump"
}

state_items() {
  # state_items TARGET: the state paths that exist, relative to the project,
  # one per line. A configured path outside the project is left out, and the
  # output and the manifest in TARGET name the setting it came from (never
  # the path, which may hold a user name).
  local target="$1" entry key candidate
  for entry in "AGENT_SPACE_DIR:${MTC_AGENT_SPACE_DIR}" \
    "PERSONA_PATH:${MTC_PERSONA_PATH}" "SPEAKERS_PATH:${MTC_SPEAKERS_PATH}" \
    "CTM_CONFIG_DIR:${MTC_CTM_CONFIG_DIR}/persona.yaml" \
    "CTM_CONFIG_DIR:${MTC_CTM_CONFIG_DIR}/seeds.d"; do
    key="${entry%%:*}"
    candidate="${entry#*:}"
    candidate="${candidate#./}"
    case "${candidate}" in
      "${PROJECT}/"*) candidate="${candidate#"${PROJECT}"/}" ;;
      /*)
        if [ -e "${candidate}" ]; then
          echo "state: ${key} is outside the project and was left out of the archive; back it up separately" >&2
          printf 'excluded state %s outside_project\n' "${key}" >>"${target}/MANIFEST.txt"
        fi
        continue
        ;;
    esac
    if [ -e "${PROJECT}/${candidate}" ]; then
      printf '%s\n' "${candidate}"
    fi
  done
}

archive_state() {
  local items=() item
  while IFS= read -r item; do
    [ -n "${item}" ] || continue
    items+=("${item}")
  done <<EOF
$(state_items "$1")
EOF
  if [ "${#items[@]}" -eq 0 ]; then
    return 3
  fi
  # COPYFILE_DISABLE keeps macOS tar from adding extended-attribute entries.
  COPYFILE_DISABLE=1 tar -czf "$1/state.tar.gz" --exclude '.env' \
    --exclude '.env.*' -C "${PROJECT}" "${items[@]}"
}

store_enabled() {
  case "$1" in
    postgres) [ "${MTC_POSTGRES_ENABLED}" = "true" ] ;;
    mongodb) [ "${MTC_MONGODB_ENABLED}" = "true" ] ;;
    qdrant) [ "${MTC_QDRANT_ENABLED}" = "true" ] ;;
    redis) [ "${MTC_REDIS_ENABLED}" = "true" ] ;;
    neo4j) [ "${MTC_NEO4J_ENABLED}" = "true" ] ;;
    *) return 1 ;;
  esac
}

store_file() {
  case "$1" in
    postgres) echo "postgres.dump" ;;
    mongodb) echo "mongo.archive.gz" ;;
    qdrant) echo "qdrant-${MTC_QDRANT_COLLECTION}.snapshot" ;;
    redis) echo "redis.rdb" ;;
    neo4j) echo "neo4j.dump" ;;
  esac
}

store_plan() {
  case "$1" in
    postgres) echo "pg_dump -Fc -h ${MTC_POSTGRES_HOST} -p ${MTC_POSTGRES_PORT} -U ${MTC_POSTGRES_USER} -d ${MTC_POSTGRES_DB} -f postgres.dump (password in PGPASSWORD)" ;;
    mongodb) echo "mongodump --host ${MTC_MONGODB_HOST} --port ${MTC_MONGODB_PORT} --db ${MTC_MONGODB_DB} --archive=mongo.archive.gz --gzip (a password, when set, in a private options file)" ;;
    qdrant) echo "POST and GET http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/collections/${MTC_QDRANT_COLLECTION}/snapshots, saved as qdrant-${MTC_QDRANT_COLLECTION}.snapshot" ;;
    redis) echo "redis-cli -h ${MTC_REDIS_HOST} -p ${MTC_REDIS_PORT} -n ${MTC_REDIS_DB} --rdb redis.rdb (a password, when set, in REDISCLI_AUTH)" ;;
    neo4j) echo "neo4j-admin database dump --to-stdout ${MTC_NEO4J_DATABASE} > neo4j.dump" ;;
  esac
}

backup_store() {
  # backup_store STORE TARGET: dump one store and record it in the manifest.
  local store="$1" target="$2" file count
  file="$(store_file "${store}")"
  echo "${store}: dumping to ${file}"
  case "${store}" in
    postgres) dump_postgres "${target}" || return 1 ;;
    mongodb) dump_mongodb "${target}" || return 1 ;;
    qdrant) dump_qdrant "${target}" || return 1 ;;
    redis) dump_redis "${target}" || return 1 ;;
    neo4j) dump_neo4j "${target}" || return 1 ;;
  esac
  if [ ! -s "${target}/${file}" ]; then
    echo "${store}: the dump tool wrote no data" >&2
    return 1
  fi
  record_file "${target}" "${file}"
  case "${store}" in
    postgres)
      count="$(count_postgres_rows "${MTC_POSTGRES_DB}")"
      printf 'count postgres rows %s\n' "${count}" >>"${target}/MANIFEST.txt"
      ;;
    mongodb)
      count="$(count_mongo_documents "${MTC_MONGODB_DB}")"
      printf 'count mongodb documents %s\n' "${count}" >>"${target}/MANIFEST.txt"
      ;;
    qdrant)
      count="$(count_qdrant_points "${MTC_QDRANT_COLLECTION}")"
      printf 'count qdrant points %s\n' "${count}" >>"${target}/MANIFEST.txt"
      ;;
  esac
}

main() {
  local dest="" with_neo4j=0 with_state=1
  DRY_RUN=0
  FAILED=0
  trap remove_mongo_options EXIT
  trap on_interrupt INT TERM HUP
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --help | -h)
        usage
        exit 0
        ;;
      --dest)
        if [ "$#" -lt 2 ] || [ -z "$2" ]; then
          echo "${SCRIPT_NAME}: --dest needs a directory" >&2
          exit 2
        fi
        dest="$2"
        shift
        ;;
      --include-neo4j-offline) with_neo4j=1 ;;
      --no-state) with_state=0 ;;
      --dry-run) DRY_RUN=1 ;;
      *)
        echo "${SCRIPT_NAME}: unknown option $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done

  local settings
  if ! settings="$(python3 -m mtc.core.env_export --secrets PROJECT_ROOT \
    BACKUP_LOCATION BACKUP_RETENTION_DAYS BACKUP_KEEP_MINIMUM POSTGRES_ENABLED \
    MONGODB_ENABLED QDRANT_ENABLED REDIS_ENABLED NEO4J_ENABLED POSTGRES_HOST \
    POSTGRES_PORT POSTGRES_USER POSTGRES_DB POSTGRES_PASSWORD MONGODB_HOST \
    MONGODB_PORT MONGODB_DB MONGODB_USER MONGODB_PASSWORD MONGODB_AUTH_SOURCE \
    QDRANT_HOST QDRANT_PORT QDRANT_COLLECTION REDIS_HOST REDIS_PORT REDIS_DB \
    REDIS_PASSWORD NEO4J_URI NEO4J_DATABASE AGENT_SPACE_DIR PERSONA_PATH \
    SPEAKERS_PATH CTM_CONFIG_DIR)"; then
    echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
    exit 2
  fi
  eval "${settings}"
  cd "${MTC_PROJECT_ROOT}"
  PROJECT="$(pwd -P)"

  local configured="${dest:-${MTC_BACKUP_LOCATION}}" location
  case "${configured}" in
    "~"*)
      echo "${SCRIPT_NAME}: write the backup location as an absolute path or a path inside the project" >&2
      exit 2
      ;;
    /* | "") location="${configured}" ;;
    *) location="${PROJECT}/${configured#./}" ;;
  esac
  if [ -n "${location}" ]; then
    location="$(physical_path "${location}")"
  fi
  if ! assert_safe_location "${configured}" "${location}" "${PROJECT}"; then
    echo "${SCRIPT_NAME}: refusing to back up or to apply retention there; set BACKUP_LOCATION to a directory of its own" >&2
    exit 2
  fi

  local stamp target store
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  target="${location}/${stamp}"
  local neo4j_hostport="${MTC_NEO4J_URI#*://}"
  neo4j_hostport="${neo4j_hostport%%/*}"

  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "would create ${target} (mode 0700)"
    for store in postgres mongodb qdrant redis; do
      if store_enabled "${store}"; then
        echo "${store}: would run: $(store_plan "${store}")"
      else
        echo "${store}: disabled, nothing to dump"
      fi
    done
    if [ "${with_neo4j}" -eq 1 ] && store_enabled neo4j; then
      echo "neo4j: would run (only while Neo4j is stopped): $(store_plan neo4j)"
    else
      echo "neo4j: not dumped (it needs --include-neo4j-offline and a stopped server)"
    fi
    if [ "${with_state}" -eq 1 ]; then
      echo "state: would run: tar -czf state.tar.gz with the agent space, the persona, the speakers file and the continuous-thought persona and seed packs"
    fi
    echo "would write MANIFEST.txt with sizes, SHA-256 sums and counts"
    if [ -d "${location}" ]; then
      apply_retention "${location}" "${MTC_BACKUP_RETENTION_DAYS}" "${MTC_BACKUP_KEEP_MINIMUM}"
    fi
    exit 0
  fi

  mkdir -p "${location}"
  chmod 700 "${location}"
  # Two backups in the same second would share a name: wait for the next one.
  local attempt=0
  while [ -e "${target}" ]; do
    attempt=$((attempt + 1))
    if [ "${attempt}" -gt 5 ]; then
      echo "${SCRIPT_NAME}: ${stamp} already exists; another backup seems to be running. Nothing was written." >&2
      exit 1
    fi
    sleep 1
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    target="${location}/${stamp}"
  done
  if ! mkdir "${target}" 2>/dev/null; then
    echo "${SCRIPT_NAME}: the backup directory ${stamp} could not be created" >&2
    exit 1
  fi
  chmod 700 "${target}"
  {
    printf 'mtc-backup-manifest 1\n'
    printf 'created_utc %s\n' "${stamp}"
  } >"${target}/MANIFEST.txt"

  for store in postgres mongodb qdrant redis; do
    if ! store_enabled "${store}"; then
      printf 'status %s skipped\n' "${store}" >>"${target}/MANIFEST.txt"
      continue
    fi
    if backup_store "${store}" "${target}"; then
      printf 'status %s ok\n' "${store}" >>"${target}/MANIFEST.txt"
    else
      echo "${store}: the dump failed; the other stores are still backed up" >&2
      rm -f "${target}/$(store_file "${store}")"
      printf 'status %s failed\n' "${store}" >>"${target}/MANIFEST.txt"
      FAILED=1
    fi
  done

  if [ "${with_neo4j}" -eq 0 ] || ! store_enabled neo4j; then
    printf 'status neo4j skipped\n' >>"${target}/MANIFEST.txt"
  elif tcp_open "${neo4j_hostport%%:*}" "${neo4j_hostport##*:}"; then
    echo "neo4j: it is running, and the community edition can only be dumped while stopped. Stop it and run this again; nothing was dumped." >&2
    printf 'status neo4j skipped\n' >>"${target}/MANIFEST.txt"
    FAILED=1
  elif backup_store neo4j "${target}"; then
    printf 'status neo4j ok\n' >>"${target}/MANIFEST.txt"
  else
    echo "neo4j: the dump failed" >&2
    rm -f "${target}/neo4j.dump"
    printf 'status neo4j failed\n' >>"${target}/MANIFEST.txt"
    FAILED=1
  fi

  local state_code=0
  if [ "${with_state}" -eq 1 ]; then
    archive_state "${target}" || state_code=$?
    if [ "${state_code}" -eq 0 ]; then
      record_file "${target}" "state.tar.gz"
      printf 'status state ok\n' >>"${target}/MANIFEST.txt"
    elif [ "${state_code}" -eq 3 ]; then
      echo "state: nothing to archive yet"
      printf 'status state skipped\n' >>"${target}/MANIFEST.txt"
    else
      echo "state: the archive failed" >&2
      rm -f "${target}/state.tar.gz"
      printf 'status state failed\n' >>"${target}/MANIFEST.txt"
      FAILED=1
    fi
  else
    printf 'status state skipped\n' >>"${target}/MANIFEST.txt"
  fi

  if [ "${FAILED}" -eq 0 ]; then
    apply_retention "${location}" "${MTC_BACKUP_RETENTION_DAYS}" "${MTC_BACKUP_KEEP_MINIMUM}"
  else
    echo "retention: skipped because this run had failures; no earlier backup was removed" >&2
  fi

  if [ "${FAILED}" -eq 0 ]; then
    echo "backup written: ${stamp}"
  else
    echo "backup written with failures: ${stamp} (see the lines above and MANIFEST.txt)" >&2
  fi
  exit "${FAILED}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
