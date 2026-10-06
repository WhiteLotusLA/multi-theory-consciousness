#!/usr/bin/env bash
# Bring the stack up: each enabled store, then the model server, then the API.
#
# Every component is checked first and started only when it does not answer.
# Settings come from the project's Settings through
# `python3 -m mtc.core.env_export`; nothing is restated here. Run it from the
# project directory with the project's virtual environment active.
set -euo pipefail

SCRIPT_NAME="start_stack.sh"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/ops/start_stack.sh [--dry-run] [--no-brain] [--no-api]
                                  [--brain-wait SECONDS] [--api-wait SECONDS]
                                  [--store-wait SECONDS]

For each enabled store, in the order PostgreSQL, Qdrant, MongoDB, Redis,
Neo4j: check its health and, when it does not answer, start it through
SERVICE_MANAGER (brew, docker or systemd; with "none" the command to run is
printed and the store counts as down), then wait for it.

With LLM_PROVIDER=local the model server's health paths are checked; when it
is down and LLM_BACKEND is set, start_llm.sh --background starts it and this
script waits for /v1/models. Then `mtc serve` is started in the background
unless the API already answers, and /health is polled until it is ready.

Options:
  --dry-run             print what would be checked and run; start nothing
  --no-brain            leave the model server alone
  --no-api              leave the API server alone
  --brain-wait SECONDS  how long to wait for the model server (default 420:
                        a cold load of a large model takes minutes)
  --api-wait SECONDS    how long to wait for the API (default 150)
  --store-wait SECONDS  how long to wait for a store it started (default 60)
  --help                show this text

PID files are written under <DATA_DIRECTORY>/run and logs under
<LOG_DIRECTORY>. A store whose only possible check is a TCP connect (its
client tool is not installed, and always Neo4j) reads "port open" with the
reason, never "healthy": an open port is not a health answer. Exit codes: 0
no enabled component is down, 1 one is, 2 usage or settings that cannot be
read.
EOF
}

DRY_RUN=0
WITH_BRAIN=1
WITH_API=1
BRAIN_WAIT=420
API_WAIT=150
STORE_WAIT=60

need_seconds() {
  # need_seconds OPTION VALUE: exit 2 unless VALUE is a whole number.
  case "$2" in
    '' | *[!0-9]*)
      echo "${SCRIPT_NAME}: $1 needs a whole number of seconds" >&2
      exit 2
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help | -h)
      usage
      exit 0
      ;;
    --dry-run) DRY_RUN=1 ;;
    --no-brain) WITH_BRAIN=0 ;;
    --no-api) WITH_API=0 ;;
    --brain-wait | --api-wait | --store-wait)
      need_seconds "$1" "${2:-}"
      case "$1" in
        --brain-wait) BRAIN_WAIT="$2" ;;
        --api-wait) API_WAIT="$2" ;;
        --store-wait) STORE_WAIT="$2" ;;
      esac
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
  LOG_DIRECTORY SERVICE_MANAGER POSTGRES_ENABLED MONGODB_ENABLED \
  QDRANT_ENABLED REDIS_ENABLED NEO4J_ENABLED POSTGRES_HOST POSTGRES_PORT \
  MONGODB_HOST MONGODB_PORT QDRANT_HOST QDRANT_PORT REDIS_HOST REDIS_PORT \
  NEO4J_URI LLM_PROVIDER LLM_BACKEND LLM_BASE_URL LLM_HEALTH_PATHS API_HOST \
  API_PORT)"; then
  echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
  exit 2
fi
eval "${SETTINGS}"
cd "${MTC_PROJECT_ROOT}"

have() {
  command -v "$1" >/dev/null 2>&1
}

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

tcp_open() {
  # True when something accepts TCP connections at host $1, port $2.
  (exec 3<>"/dev/tcp/$1/$2") >/dev/null 2>&1
}

http_ok() {
  # True when the URL answers with success within three seconds.
  have curl && curl --silent --fail --max-time 3 --output /dev/null "$1"
}

wait_for() {
  # wait_for LIMIT INTERVAL COMMAND...: poll until the command succeeds.
  local limit="$1" interval="$2" deadline left
  shift 2
  deadline=$((SECONDS + limit))
  while :; do
    if "$@"; then
      return 0
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      return 1
    fi
    left=$((deadline - SECONDS))
    if [ "${left}" -lt "${interval}" ]; then
      sleep "${left}"
    else
      sleep "${interval}"
    fi
  done
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

RUN_DIR="$(project_path "${MTC_DATA_DIRECTORY}")/run"
LOG_DIR="$(project_path "${MTC_LOG_DIRECTORY}")"
NEO4J_HOSTPORT="${MTC_NEO4J_URI#*://}"
NEO4J_HOSTPORT="${NEO4J_HOSTPORT%%/*}"
NEO4J_HOST="${NEO4J_HOSTPORT%%:*}"
NEO4J_PORT="${NEO4J_HOSTPORT##*:}"
case "${MTC_API_HOST}" in
  0.0.0.0 | :: | "") API_CLIENT_HOST="127.0.0.1" ;;
  *) API_CLIENT_HOST="${MTC_API_HOST}" ;;
esac

FAILED=0
SUMMARY=""

note() {
  SUMMARY="${SUMMARY}  $1
"
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

store_check_text() {
  # The health check of a store, as text for --dry-run.
  case "$1" in
    postgres) echo "pg_isready -q -h ${MTC_POSTGRES_HOST} -p ${MTC_POSTGRES_PORT}" ;;
    qdrant) echo "GET http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/readyz" ;;
    mongodb) echo "mongosh ping at ${MTC_MONGODB_HOST} port ${MTC_MONGODB_PORT} (a TCP probe without mongosh)" ;;
    redis) echo "redis-cli -h ${MTC_REDIS_HOST} -p ${MTC_REDIS_PORT} ping" ;;
    neo4j) echo "a TCP probe of the bolt port ${NEO4J_PORT} at ${NEO4J_HOST}" ;;
  esac
}

probe_limit() {
  # Why the check of store $1 is only a TCP connect (its client tool is not
  # installed, or Neo4j, which is checked by its port), or nothing when a
  # client answers. An open port is not a health answer.
  case "$1" in
    postgres) have pg_isready || echo "pg_isready is not installed" ;;
    qdrant) have curl || echo "curl is not installed" ;;
    mongodb) have mongosh || echo "mongosh is not installed" ;;
    redis) have redis-cli || echo "redis-cli is not installed" ;;
    neo4j) echo "Neo4j is checked by its bolt port only" ;;
  esac
}

store_state() {
  # store_state STORE CONTEXT: "healthy (CONTEXT)", or "port open, CONTEXT
  # (why)" when the only check that could run was a TCP connect.
  local limit
  limit="$(probe_limit "$1")"
  if [ -n "${limit}" ]; then
    printf 'port open, %s (%s, so health was not measured)\n' "$2" "${limit}"
  else
    printf 'healthy (%s)\n' "$2"
  fi
}

store_healthy() {
  local reply
  case "$1" in
    postgres)
      if have pg_isready; then
        pg_isready -q -h "${MTC_POSTGRES_HOST}" -p "${MTC_POSTGRES_PORT}" >/dev/null 2>&1
      else
        tcp_open "${MTC_POSTGRES_HOST}" "${MTC_POSTGRES_PORT}"
      fi
      ;;
    qdrant)
      if have curl; then
        http_ok "http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/readyz"
      else
        tcp_open "${MTC_QDRANT_HOST}" "${MTC_QDRANT_PORT}"
      fi
      ;;
    mongodb)
      if have mongosh; then
        mongosh --quiet --host "${MTC_MONGODB_HOST}" --port "${MTC_MONGODB_PORT}" \
          --eval 'db.adminCommand({ping: 1}).ok' >/dev/null 2>&1
      else
        tcp_open "${MTC_MONGODB_HOST}" "${MTC_MONGODB_PORT}"
      fi
      ;;
    redis)
      if have redis-cli; then
        reply="$(redis-cli -h "${MTC_REDIS_HOST}" -p "${MTC_REDIS_PORT}" ping 2>&1 || true)"
        case "${reply}" in
          # A server that asks for a password is up.
          *PONG* | *NOAUTH*) return 0 ;;
          *) return 1 ;;
        esac
      else
        tcp_open "${MTC_REDIS_HOST}" "${MTC_REDIS_PORT}"
      fi
      ;;
    neo4j) tcp_open "${NEO4J_HOST}" "${NEO4J_PORT}" ;;
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

manual_hint() {
  case "$1" in
    postgres) echo "brew services start postgresql@17, or systemctl start postgresql" ;;
    qdrant) echo "qdrant --config-path config/qdrant.yaml, or the qdrant container" ;;
    mongodb) echo "brew services start mongodb-community, or systemctl start mongod" ;;
    redis) echo "brew services start redis, or systemctl start redis-server" ;;
    neo4j) echo "brew services start neo4j, or systemctl start neo4j" ;;
  esac
}

start_qdrant_binary() {
  # Qdrant has no Homebrew formula: run the release binary in the background.
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "would run: qdrant --config-path config/qdrant.yaml (detached, PID in ${MTC_DATA_DIRECTORY}/run/qdrant.pid)"
    return 0
  fi
  if ! have qdrant; then
    echo "qdrant: there is no Homebrew formula. Put the release binary on PATH as qdrant, or run the qdrant container." >&2
    return 1
  fi
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"
  nohup qdrant --config-path config/qdrant.yaml >>"${LOG_DIR}/qdrant.log" 2>&1 &
  printf '%s\n' "$!" >"${RUN_DIR}/qdrant.pid"
}

store_start() {
  local store="$1"
  case "${MTC_SERVICE_MANAGER}" in
    brew)
      if [ "${store}" = "qdrant" ]; then
        start_qdrant_binary
      else
        run_or_plan brew services start "$(brew_formula "${store}")"
      fi
      ;;
    docker)
      run_or_plan docker compose --env-file .env -f docker/docker-compose.yml \
        up -d "$(compose_service "${store}")"
      ;;
    systemd)
      run_or_plan systemctl start "$(systemd_unit "${store}")"
      ;;
    *)
      echo "${store}: SERVICE_MANAGER is none, so start it yourself (for example: $(manual_hint "${store}"))"
      return 1
      ;;
  esac
}

handle_store() {
  local store="$1"
  if ! store_enabled "${store}"; then
    note "${store}: disabled"
    return 0
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${store}: would check: $(store_check_text "${store}")"
    printf '%s: if it does not answer, ' "${store}"
    store_start "${store}" || true
    note "${store}: planned"
    return 0
  fi
  if store_healthy "${store}"; then
    note "${store}: $(store_state "${store}" "already running")"
    return 0
  fi
  echo "${store}: not answering; starting it (SERVICE_MANAGER=${MTC_SERVICE_MANAGER})"
  if ! store_start "${store}"; then
    note "${store}: down"
    FAILED=1
    return 0
  fi
  if wait_for "${STORE_WAIT}" 2 store_healthy "${store}"; then
    if [ -n "$(probe_limit "${store}")" ]; then
      note "${store}: started, $(store_state "${store}" "now")"
    else
      note "${store}: started"
    fi
  else
    note "${store}: down (not healthy after ${STORE_WAIT} seconds)"
    FAILED=1
  fi
}

brain_healthy() {
  local path
  while IFS= read -r path; do
    path="${path# }"
    path="${path% }"
    [ -n "${path}" ] || continue
    if http_ok "${MTC_LLM_BASE_URL}/${path#/}"; then
      return 0
    fi
  done <<EOF
$(printf '%s' "${MTC_LLM_HEALTH_PATHS}" | tr ',' '\n')
EOF
  return 1
}

handle_brain() {
  if [ "${WITH_BRAIN}" -eq 0 ]; then
    note "brain: left alone (--no-brain)"
    return 0
  fi
  if [ "${MTC_LLM_PROVIDER}" != "local" ]; then
    note "brain: LLM_PROVIDER=${MTC_LLM_PROVIDER}, nothing to start"
    return 0
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "brain: would check the health paths (${MTC_LLM_HEALTH_PATHS}) at ${MTC_LLM_BASE_URL}"
    if [ -n "${MTC_LLM_BACKEND}" ]; then
      echo "brain: if it does not answer, would run: ${SCRIPT_DIR}/start_llm.sh --background, then wait up to ${BRAIN_WAIT} seconds for /v1/models"
    else
      echo "brain: if it does not answer, nothing is started (LLM_BACKEND is empty)"
    fi
    note "brain: planned"
    return 0
  fi
  if brain_healthy; then
    note "brain: healthy (already running)"
    return 0
  fi
  if [ -z "${MTC_LLM_BACKEND}" ]; then
    note "brain: down (LLM_BACKEND is empty: start your model server yourself)"
    FAILED=1
    return 0
  fi
  echo "brain: not answering; starting it (LLM_BACKEND=${MTC_LLM_BACKEND})"
  if ! "${SCRIPT_DIR}/start_llm.sh" --background; then
    note "brain: down (start_llm.sh could not start it)"
    FAILED=1
    return 0
  fi
  if wait_for "${BRAIN_WAIT}" 5 http_ok "${MTC_LLM_BASE_URL}/v1/models"; then
    note "brain: started"
  else
    note "brain: down (no answer after ${BRAIN_WAIT} seconds; a large model may still be loading, see ${MTC_LOG_DIRECTORY}/llm.log)"
    FAILED=1
  fi
}

api_ready() {
  local body
  have curl || return 1
  body="$(curl --silent --fail --max-time 3 \
    "http://${API_CLIENT_HOST}:${MTC_API_PORT}/health" 2>/dev/null || true)"
  case "${body}" in
    *'"ready":true'* | *'"ready": true'*) return 0 ;;
    *) return 1 ;;
  esac
}

handle_api() {
  if [ "${WITH_API}" -eq 0 ]; then
    note "api: left alone (--no-api)"
    return 0
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "api: would check GET http://${API_CLIENT_HOST}:${MTC_API_PORT}/health"
    echo "api: if the port is free, would run: mtc serve (detached, PID in ${MTC_DATA_DIRECTORY}/run/api.pid, log in ${MTC_LOG_DIRECTORY}/api.log), then poll /health for up to ${API_WAIT} seconds"
    note "api: planned"
    return 0
  fi
  if api_ready; then
    note "api: healthy (already running)"
    return 0
  fi
  if tcp_open "${API_CLIENT_HOST}" "${MTC_API_PORT}"; then
    if wait_for "${API_WAIT}" 2 api_ready; then
      note "api: healthy (already running)"
    else
      note "api: down (port ${MTC_API_PORT} is in use by something that is not a ready API)"
      FAILED=1
    fi
    return 0
  fi
  if ! have mtc; then
    note "api: down (the mtc command is not on PATH: activate the virtual environment)"
    FAILED=1
    return 0
  fi
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"
  nohup mtc serve >>"${LOG_DIR}/api.log" 2>&1 &
  printf '%s\n' "$!" >"${RUN_DIR}/api.pid"
  echo "api: started mtc serve; waiting for /health"
  if wait_for "${API_WAIT}" 2 api_ready; then
    note "api: started"
  else
    note "api: down (not ready after ${API_WAIT} seconds, see ${MTC_LOG_DIRECTORY}/api.log)"
    FAILED=1
  fi
}

for STORE in postgres qdrant mongodb redis neo4j; do
  handle_store "${STORE}"
done
handle_brain
handle_api

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "Plan (nothing was started):"
else
  echo "Stack:"
fi
printf '%s' "${SUMMARY}"
exit "${FAILED}"
