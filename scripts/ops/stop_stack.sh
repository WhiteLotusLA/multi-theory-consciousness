#!/usr/bin/env bash
# Stop the API server and the model server start_stack.sh started and, with
# --all, the enabled stores.
#
# A process is asked to stop with the terminate signal and given a grace
# period. One that is still running afterwards is reported and left alone:
# the API server saves state while it shuts down, so nothing is ever forced.
# Settings come from the project's Settings through
# `python3 -m mtc.core.env_export`. Run it from the project directory with
# the project's virtual environment active.
set -euo pipefail

SCRIPT_NAME="stop_stack.sh"

usage() {
  cat <<'EOF'
Usage: scripts/ops/stop_stack.sh [--all] [--grace SECONDS] [--dry-run]

Stops, in this order, the API server (<DATA_DIRECTORY>/run/api.pid) and the
model server start_llm.sh started (<DATA_DIRECTORY>/run/llm.pid). Each gets
the terminate signal and up to --grace seconds to exit. A process that
outlives the grace period is reported and the exit code is 1; it is never
forced, because the API server saves state while it shuts down.

Options:
  --all            also stop the enabled stores, in the reverse of the start
                   order (Neo4j, Redis, MongoDB, Qdrant, PostgreSQL), through
                   SERVICE_MANAGER; with "none" the command to run is printed
  --grace SECONDS  how long to wait for each process (default 30)
  --dry-run        print what would be signalled and run; stop nothing
  --help           show this text

Exit codes: 0 everything asked for has stopped, 1 something is still
running or could not be stopped, 2 usage or settings that cannot be read.
EOF
}

STOP_ALL=0
GRACE=30
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help | -h)
      usage
      exit 0
      ;;
    --all) STOP_ALL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --grace)
      case "${2:-}" in
        '' | *[!0-9]*)
          echo "${SCRIPT_NAME}: --grace needs a whole number of seconds" >&2
          exit 2
          ;;
      esac
      GRACE="$2"
      shift
      ;;
    *)
      echo "${SCRIPT_NAME}: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! SETTINGS="$(python3 -m mtc.core.env_export PROJECT_ROOT DATA_DIRECTORY \
  SERVICE_MANAGER POSTGRES_ENABLED MONGODB_ENABLED QDRANT_ENABLED \
  REDIS_ENABLED NEO4J_ENABLED)"; then
  echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
  exit 2
fi
eval "${SETTINGS}"
cd "${MTC_PROJECT_ROOT}"

project_path() {
  # An absolute path stays as it is; a relative one joins the project root.
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    "~"*)
      echo "${SCRIPT_NAME}: write $1 as an absolute path or a path inside the project" >&2
      return 2
      ;;
    ./*) printf '%s/%s\n' "${MTC_PROJECT_ROOT}" "${1#./}" ;;
    *) printf '%s/%s\n' "${MTC_PROJECT_ROOT}" "$1" ;;
  esac
}

RUN_DIR="$(project_path "${MTC_DATA_DIRECTORY}")/run"
FAILED=0

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

stop_pid_file() {
  # stop_pid_file LABEL FILE WORD...: stop the process a PID file names, when
  # its command line still holds one of the words.
  local label="$1" file="$2" pid command_line word matched waited
  shift 2
  if [ ! -f "${file}" ]; then
    echo "${label}: not running (no PID file)"
    return 0
  fi
  pid="$(tr -d '[:space:]' <"${file}")"
  case "${pid}" in
    '' | *[!0-9]*)
      echo "${label}: the PID file does not hold a process number; nothing was signalled" >&2
      FAILED=1
      return 0
      ;;
  esac
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: not running (the PID file was stale and is removed)"
    if [ "${DRY_RUN}" -eq 0 ]; then
      rm -f "${file}"
    fi
    return 0
  fi
  command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  matched=0
  for word in "$@"; do
    case "${command_line}" in
      *"${word}"*) matched=1 ;;
    esac
  done
  if [ "${matched}" -eq 0 ]; then
    echo "${label}: process ${pid} is another program now; nothing was signalled" >&2
    FAILED=1
    return 0
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${label}: would send the terminate signal to process ${pid} and wait up to ${GRACE} seconds"
    return 0
  fi
  echo "${label}: asking process ${pid} to stop (up to ${GRACE} seconds)"
  kill -TERM "${pid}" 2>/dev/null || true
  waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "${waited}" -ge "${GRACE}" ]; then
      echo "${label}: process ${pid} is still running after ${GRACE} seconds. It was not forced; it may still be saving state. Check it, then run this again." >&2
      FAILED=1
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "${file}"
  echo "${label}: stopped"
}

store_enabled() {
  case "$1" in
    postgres) [ "${MTC_POSTGRES_ENABLED}" = "true" ] ;;
    qdrant) [ "${MTC_QDRANT_ENABLED}" = "true" ] ;;
    mongodb) [ "${MTC_MONGODB_ENABLED}" = "true" ] ;;
    redis) [ "${MTC_REDIS_ENABLED}" = "true" ] ;;
    neo4j) [ "${MTC_NEO4J_ENABLED}" = "true" ] ;;
    *) return 1 ;;
  esac
}

compose_service() {
  case "$1" in
    mongodb) echo "mongo" ;;
    *) echo "$1" ;;
  esac
}

brew_formula() {
  case "$1" in
    postgres) echo "postgresql@17" ;;
    mongodb) echo "mongodb-community" ;;
    redis) echo "redis" ;;
    neo4j) echo "neo4j" ;;
  esac
}

systemd_unit() {
  case "$1" in
    postgres) echo "postgresql" ;;
    qdrant) echo "qdrant" ;;
    mongodb) echo "mongod" ;;
    redis) echo "redis-server" ;;
    neo4j) echo "neo4j" ;;
  esac
}

stop_store() {
  local store="$1"
  if ! store_enabled "${store}"; then
    return 0
  fi
  case "${MTC_SERVICE_MANAGER}" in
    brew)
      if [ "${store}" = "qdrant" ]; then
        stop_pid_file "qdrant" "${RUN_DIR}/qdrant.pid" "qdrant"
      elif ! run_or_plan brew services stop "$(brew_formula "${store}")"; then
        echo "${store}: brew could not stop it" >&2
        FAILED=1
      fi
      ;;
    docker)
      # Stop only: the container and its volume stay.
      if ! run_or_plan docker compose --env-file .env \
        -f docker/docker-compose.yml stop "$(compose_service "${store}")"; then
        echo "${store}: docker compose could not stop it" >&2
        FAILED=1
      fi
      ;;
    systemd)
      if ! run_or_plan systemctl stop "$(systemd_unit "${store}")"; then
        echo "${store}: systemctl could not stop it" >&2
        FAILED=1
      fi
      ;;
    *)
      echo "${store}: SERVICE_MANAGER is none, so stop it yourself"
      ;;
  esac
}

# The API is "mtc serve" (the console script, run by its full path) or the
# server module; a bare "mtc" would match any command line that merely holds
# those letters, such as a recycled PID now running an editor.
stop_pid_file "api" "${RUN_DIR}/api.pid" "/mtc serve" "mtc.api.server"
stop_pid_file "brain" "${RUN_DIR}/llm.pid" "mlx_lm" "llama-server"

if [ "${STOP_ALL}" -eq 1 ]; then
  for STORE in neo4j redis mongodb qdrant postgres; do
    stop_store "${STORE}"
  done
fi

exit "${FAILED}"
