#!/usr/bin/env bash
# Prove a backup can be restored: load it into scratch targets, compare the
# counts with the manifest, and remove the scratch targets again.
#
# The drill only ever creates and drops names that end with _restore_test,
# and it asserts that before every create and every drop; the live database
# and collection are read by nothing here. Passwords reach the tools through
# their environment variables, never on a command line. Settings come from the
# project's Settings through `python3 -m mtc.core.env_export`. Run it from the
# project directory with the project's virtual environment active.
set -euo pipefail

SCRIPT_NAME="restore_drill.sh"
SCRATCH_SUFFIX="_restore_test"
umask 077

usage() {
  cat <<'EOF'
Usage: scripts/ops/restore_drill.sh [--from DIR] [--dry-run]

Restores the newest backup under BACKUP_LOCATION (or the backup directory
DIR) into scratch targets and checks it:

  PostgreSQL  <POSTGRES_DB>_restore_test        rows against the manifest
  MongoDB     <MONGODB_DB>_restore_test         documents against the manifest
  Qdrant      <QDRANT_COLLECTION>_restore_test  points against the manifest

Each scratch target is created, counted and dropped again. Only names that
end with _restore_test are ever created or dropped; the live data is not
touched. A store whose file is not in the backup is left out. A count the
manifest records as unknown cannot be compared and counts as not verified.

Options:
  --from DIR   drill this backup directory instead of the newest one
  --dry-run    print what would run; create and drop nothing
  --help       show this text

Exit codes: 0 every store in the backup was restored and its count matched,
1 a restore failed, a count differed or could not be compared, 2 usage,
settings that cannot be read or no backup to drill.
EOF
}

have() {
  command -v "$1" >/dev/null 2>&1
}

assert_scratch() {
  # assert_scratch NAME LIVE: succeed only when NAME is the scratch name of
  # the live name LIVE. Called before every create and every drop.
  local name="$1" live="$2"
  if [ -z "${name}" ] || [ -z "${live}" ]; then
    echo "${SCRIPT_NAME}: refusing an empty target name" >&2
    return 1
  fi
  case "${name}" in
    *"${SCRATCH_SUFFIX}") ;;
    *)
      echo "${SCRIPT_NAME}: refusing ${name}: it does not end with ${SCRATCH_SUFFIX}" >&2
      return 1
      ;;
  esac
  if [ "${name}" = "${live}" ] || [ "${name}" != "${live}${SCRATCH_SUFFIX}" ]; then
    echo "${SCRIPT_NAME}: refusing ${name}: it is not the scratch name of ${live}" >&2
    return 1
  fi
  return 0
}

is_backup_name() {
  case "$1" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) return 0 ;;
    *) return 1 ;;
  esac
}

newest_backup() {
  # The newest backup directory under $1, or nothing.
  local entry newest=""
  for entry in "$1"/*/; do
    [ -d "${entry}" ] || continue
    entry="${entry%/}"
    entry="${entry##*/}"
    if is_backup_name "${entry}" && [ "${entry}" \> "${newest}" ]; then
      newest="${entry}"
    fi
  done
  if [ -n "${newest}" ]; then
    printf '%s/%s\n' "$1" "${newest}"
  fi
}

manifest_count() {
  # manifest_count DIR STORE UNIT: the count the manifest records.
  local kind store unit value
  while read -r kind store unit value; do
    if [ "${kind}" = "count" ] && [ "${store}" = "$2" ] && [ "${unit}" = "$3" ]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done <"$1/MANIFEST.txt"
  echo "unknown"
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

pg_tool() {
  # pg_tool COMMAND ARGS...: a PostgreSQL tool with the connection options;
  # the password is in PGPASSWORD.
  local tool="$1"
  shift
  PGPASSWORD="${MTC_POSTGRES_PASSWORD}" "${tool}" -h "${MTC_POSTGRES_HOST}" \
    -p "${MTC_POSTGRES_PORT}" -U "${MTC_POSTGRES_USER}" "$@"
}

count_postgres_rows() {
  # The rows in every table of database $1, or "unknown".
  local database="$1" tables table rows total
  if ! tables="$(pg_tool psql -At -d "${database}" -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename" 2>/dev/null)"; then
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
    if ! rows="$(pg_tool psql -At -d "${database}" -c "SELECT count(*) FROM \"${table}\"" 2>/dev/null)"; then
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

drop_postgres_scratch() {
  # drop_postgres_scratch NAME LIVE: drop a scratch database, nothing else.
  assert_scratch "$1" "$2" || return 1
  pg_tool dropdb --if-exists "$1"
}

create_postgres_scratch() {
  assert_scratch "$1" "$2" || return 1
  pg_tool createdb "$1"
}

drop_mongo_scratch() {
  # drop_mongo_scratch NAME LIVE: drop a scratch database, nothing else.
  assert_scratch "$1" "$2" || return 1
  mongosh --quiet --host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}" "$1" \
    --eval 'db.dropDatabase().ok' >/dev/null
}

drop_qdrant_scratch() {
  # drop_qdrant_scratch NAME LIVE: delete a scratch collection, nothing else.
  assert_scratch "$1" "$2" || return 1
  curl --silent --fail --max-time 60 -X DELETE \
    "http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/collections/$1" >/dev/null
}

verdict() {
  # verdict STORE UNIT EXPECTED ACTUAL: print the comparison; fail on a gap.
  local store="$1" unit="$2" expected="$3" actual="$4"
  if [ "${expected}" = "unknown" ] || [ "${actual}" = "unknown" ]; then
    echo "${store}: restored, but not verified: the ${unit} count is unknown (manifest ${expected}, restored ${actual})" >&2
    return 1
  fi
  if [ "${expected}" != "${actual}" ]; then
    echo "${store}: MISMATCH: the manifest records ${expected} ${unit}, the restored copy holds ${actual}" >&2
    return 1
  fi
  echo "${store}: verified, ${actual} ${unit} match the manifest"
}

drill_postgres() {
  local backup="$1" live="${MTC_POSTGRES_DB}" target expected actual
  target="${live}${SCRATCH_SUFFIX}"
  assert_scratch "${target}" "${live}" || return 1
  expected="$(manifest_count "${backup}" postgres rows)"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "postgres: would run: dropdb --if-exists ${target}; createdb ${target}; pg_restore --no-owner -d ${target} postgres.dump; count the rows (the manifest records ${expected}); dropdb ${target}"
    return 0
  fi
  # A scratch database left by an interrupted drill goes first.
  drop_postgres_scratch "${target}" "${live}" || return 1
  create_postgres_scratch "${target}" "${live}" || return 1
  if ! pg_tool pg_restore --no-owner -d "${target}" "${backup}/postgres.dump"; then
    echo "postgres: the restore into ${target} failed" >&2
    drop_postgres_scratch "${target}" "${live}" || true
    return 1
  fi
  actual="$(count_postgres_rows "${target}")"
  if ! drop_postgres_scratch "${target}" "${live}"; then
    echo "postgres: the scratch database ${target} could not be dropped" >&2
    return 1
  fi
  verdict postgres rows "${expected}" "${actual}"
}

drill_mongodb() {
  local backup="$1" live="${MTC_MONGODB_DB}" target expected actual reply
  target="${live}${SCRATCH_SUFFIX}"
  assert_scratch "${target}" "${live}" || return 1
  expected="$(manifest_count "${backup}" mongodb documents)"
  if [ -n "${MTC_MONGODB_USER}" ]; then
    echo "mongodb: not drilled: credentials are configured, and mongosh takes a password only on its command line. Drill it by hand with restore.sh --store mongodb --target-suffix ${SCRATCH_SUFFIX}." >&2
    return 1
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "mongodb: would run: mongorestore --archive=mongo.archive.gz --gzip --drop --nsFrom ${live}.* --nsTo ${target}.*; count the documents (the manifest records ${expected}); drop the database ${target}"
    return 0
  fi
  assert_scratch "${target}" "${live}" || return 1
  if ! mongorestore --host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}" \
    "--archive=${backup}/mongo.archive.gz" --gzip --drop \
    --nsFrom "${live}.*" --nsTo "${target}.*"; then
    echo "mongodb: the restore into ${target} failed" >&2
    drop_mongo_scratch "${target}" "${live}" || true
    return 1
  fi
  actual="unknown"
  if reply="$(mongosh --quiet --host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}" \
    "${target}" --eval 'db.getCollectionNames().reduce((n, c) => n + db.getCollection(c).countDocuments({}), 0)' 2>/dev/null)"; then
    case "${reply}" in
      '' | *[!0-9]*) ;;
      *) actual="${reply}" ;;
    esac
  fi
  if ! drop_mongo_scratch "${target}" "${live}"; then
    echo "mongodb: the scratch database ${target} could not be dropped" >&2
    return 1
  fi
  verdict mongodb documents "${expected}" "${actual}"
}

drill_qdrant() {
  local backup="$1" live="${MTC_QDRANT_COLLECTION}" target expected actual reply
  local base="http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/collections"
  target="${live}${SCRATCH_SUFFIX}"
  assert_scratch "${target}" "${live}" || return 1
  expected="$(manifest_count "${backup}" qdrant points)"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "qdrant: would run: upload qdrant-${live}.snapshot into the collection ${target}; count the points (the manifest records ${expected}); delete the collection ${target}"
    return 0
  fi
  assert_scratch "${target}" "${live}" || return 1
  if ! curl --silent --fail --max-time 3600 -X POST \
    "${base}/${target}/snapshots/upload?priority=snapshot" \
    -H "Content-Type: multipart/form-data" \
    -F "snapshot=@${backup}/qdrant-${live}.snapshot" >/dev/null; then
    echo "qdrant: the restore into ${target} failed" >&2
    drop_qdrant_scratch "${target}" "${live}" || true
    return 1
  fi
  actual="unknown"
  if reply="$(curl --silent --fail --max-time 30 "${base}/${target}")"; then
    if reply="$(printf '%s' "${reply}" | json_field result points_count)"; then
      case "${reply}" in
        '' | *[!0-9]*) ;;
        *) actual="${reply}" ;;
      esac
    fi
  fi
  if ! drop_qdrant_scratch "${target}" "${live}"; then
    echo "qdrant: the scratch collection ${target} could not be deleted" >&2
    return 1
  fi
  verdict qdrant points "${expected}" "${actual}"
}

main() {
  local from=""
  DRY_RUN=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --help | -h)
        usage
        exit 0
        ;;
      --from)
        if [ "$#" -lt 2 ] || [ -z "$2" ]; then
          echo "${SCRIPT_NAME}: --from needs a backup directory" >&2
          exit 2
        fi
        from="$2"
        shift
        ;;
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
    BACKUP_LOCATION POSTGRES_HOST POSTGRES_PORT POSTGRES_USER POSTGRES_DB \
    POSTGRES_PASSWORD MONGODB_HOST MONGODB_PORT MONGODB_DB MONGODB_USER \
    QDRANT_HOST QDRANT_PORT QDRANT_COLLECTION)"; then
    echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
    exit 2
  fi
  eval "${settings}"
  cd "${MTC_PROJECT_ROOT}"

  local backup="${from}" location
  if [ -z "${backup}" ]; then
    case "${MTC_BACKUP_LOCATION}" in
      /*) location="${MTC_BACKUP_LOCATION}" ;;
      "~"* | "")
        echo "${SCRIPT_NAME}: write BACKUP_LOCATION as an absolute path or a path inside the project" >&2
        exit 2
        ;;
      *) location="${MTC_PROJECT_ROOT}/${MTC_BACKUP_LOCATION#./}" ;;
    esac
    if [ -d "${location}" ]; then
      backup="$(newest_backup "${location}")"
    fi
  fi
  if [ -z "${backup}" ] || [ ! -f "${backup}/MANIFEST.txt" ]; then
    echo "${SCRIPT_NAME}: no backup to drill (run scripts/ops/backup.sh first, or pass --from DIR)" >&2
    exit 2
  fi
  echo "drilling ${backup##*/}"

  local failed=0 drilled=0
  if [ -f "${backup}/postgres.dump" ]; then
    drilled=$((drilled + 1))
    drill_postgres "${backup}" || failed=1
  fi
  if [ -f "${backup}/mongo.archive.gz" ]; then
    drilled=$((drilled + 1))
    drill_mongodb "${backup}" || failed=1
  fi
  if [ -f "${backup}/qdrant-${MTC_QDRANT_COLLECTION}.snapshot" ]; then
    drilled=$((drilled + 1))
    drill_qdrant "${backup}" || failed=1
  fi
  if [ "${drilled}" -eq 0 ]; then
    echo "${SCRIPT_NAME}: this backup holds no PostgreSQL, MongoDB or Qdrant file, so nothing was verified" >&2
    exit 1
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "nothing was created or dropped (--dry-run)"
    exit 0
  fi
  if [ "${failed}" -eq 0 ]; then
    echo "drill passed: every store in the backup restored and matched"
  else
    echo "drill failed: see the lines above" >&2
  fi
  exit "${failed}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
