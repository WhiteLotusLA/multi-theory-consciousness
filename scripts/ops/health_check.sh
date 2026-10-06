#!/usr/bin/env bash
# Report the health of every enabled component, one line each.
#
# It only looks: it reports through its output, an optional log line and its
# exit code, and leaves every decision to the operator. Settings come from the
# project's Settings through `python3 -m mtc.core.env_export`. Run it from the
# project directory with the project's virtual environment active (a timer
# can run it every few minutes).
set -euo pipefail

SCRIPT_NAME="health_check.sh"

usage() {
  cat <<'EOF'
Usage: scripts/ops/health_check.sh [--quiet] [--log-file PATH] [--dry-run]

Checks each enabled store (PostgreSQL, Qdrant, MongoDB, Redis, Neo4j), the
model server when LLM_PROVIDER is local, and the API server, and prints one
line per component: healthy, DOWN, or "port open" when the only check that
could run was a TCP connect (the store's client tool is not installed, and
always for Neo4j): an open port is not a health answer, so that line says
health was not measured. It changes nothing: a component that is down is
reported, and what to do about it is the operator's call.

Options:
  --quiet          print only the components that are down
  --log-file PATH  append one line per run to PATH (a UTC time, ok,
                   unverified (nothing down, some port open only) or
                   UNHEALTHY, and every component's state)
  --dry-run        print the checks it would make; contact nothing
  --help           show this text

Exit codes: 0 no enabled component is down (a "port open" line is not a
failure, and it is not a health answer either), 1 at least one is down, 2
usage or settings that cannot be read.
EOF
}

QUIET=0
LOG_FILE=""
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help | -h)
      usage
      exit 0
      ;;
    --quiet) QUIET=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --log-file)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "${SCRIPT_NAME}: --log-file needs a path" >&2
        exit 2
      fi
      LOG_FILE="$2"
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

if ! SETTINGS="$(python3 -m mtc.core.env_export PROJECT_ROOT POSTGRES_ENABLED \
  MONGODB_ENABLED QDRANT_ENABLED REDIS_ENABLED NEO4J_ENABLED POSTGRES_HOST \
  POSTGRES_PORT MONGODB_HOST MONGODB_PORT QDRANT_HOST QDRANT_PORT REDIS_HOST \
  REDIS_PORT NEO4J_URI LLM_PROVIDER LLM_BASE_URL LLM_HEALTH_PATHS API_HOST \
  API_PORT)"; then
  echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
  exit 2
fi
eval "${SETTINGS}"

have() {
  command -v "$1" >/dev/null 2>&1
}

tcp_open() {
  # True when something accepts TCP connections at host $1, port $2.
  (exec 3<>"/dev/tcp/$1/$2") >/dev/null 2>&1
}

http_ok() {
  # True when the URL answers with success within three seconds.
  have curl && curl --silent --fail --max-time 3 --output /dev/null "$1"
}

NEO4J_HOSTPORT="${MTC_NEO4J_URI#*://}"
NEO4J_HOSTPORT="${NEO4J_HOSTPORT%%/*}"
NEO4J_HOST="${NEO4J_HOSTPORT%%:*}"
NEO4J_PORT="${NEO4J_HOSTPORT##*:}"
case "${MTC_API_HOST}" in
  0.0.0.0 | :: | "") API_CLIENT_HOST="127.0.0.1" ;;
  *) API_CLIENT_HOST="${MTC_API_HOST}" ;;
esac

component_enabled() {
  case "$1" in
    postgres) [ "${MTC_POSTGRES_ENABLED}" = "true" ] ;;
    qdrant) [ "${MTC_QDRANT_ENABLED}" = "true" ] ;;
    mongodb) [ "${MTC_MONGODB_ENABLED}" = "true" ] ;;
    redis) [ "${MTC_REDIS_ENABLED}" = "true" ] ;;
    neo4j) [ "${MTC_NEO4J_ENABLED}" = "true" ] ;;
    brain) [ "${MTC_LLM_PROVIDER}" = "local" ] ;;
    api) return 0 ;;
    *) return 1 ;;
  esac
}

component_check_text() {
  case "$1" in
    postgres) echo "pg_isready -q -h ${MTC_POSTGRES_HOST} -p ${MTC_POSTGRES_PORT}" ;;
    qdrant) echo "GET http://${MTC_QDRANT_HOST}:${MTC_QDRANT_PORT}/readyz" ;;
    mongodb) echo "mongosh ping at ${MTC_MONGODB_HOST} port ${MTC_MONGODB_PORT} (a TCP probe without mongosh)" ;;
    redis) echo "redis-cli -h ${MTC_REDIS_HOST} -p ${MTC_REDIS_PORT} ping" ;;
    neo4j) echo "a TCP probe of the bolt port ${NEO4J_PORT} at ${NEO4J_HOST}" ;;
    brain) echo "GET the health paths (${MTC_LLM_HEALTH_PATHS}) at ${MTC_LLM_BASE_URL}" ;;
    api) echo "GET http://${API_CLIENT_HOST}:${MTC_API_PORT}/health" ;;
  esac
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

component_healthy() {
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
    brain) brain_healthy ;;
    api) api_ready ;;
    *) return 1 ;;
  esac
}

UNHEALTHY=0
UNVERIFIED=0
STATES=""
for COMPONENT in postgres qdrant mongodb redis neo4j brain api; do
  if ! component_enabled "${COMPONENT}"; then
    continue
  fi
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${COMPONENT}: would check: $(component_check_text "${COMPONENT}")"
    continue
  fi
  if component_healthy "${COMPONENT}"; then
    LIMIT="$(probe_limit "${COMPONENT}")"
    if [ -n "${LIMIT}" ]; then
      STATES="${STATES} ${COMPONENT}=port_open"
      UNVERIFIED=1
      if [ "${QUIET}" -eq 0 ]; then
        echo "${COMPONENT}: port open (${LIMIT}, so health was not measured)"
      fi
    else
      STATES="${STATES} ${COMPONENT}=healthy"
      if [ "${QUIET}" -eq 0 ]; then
        echo "${COMPONENT}: healthy"
      fi
    fi
  else
    STATES="${STATES} ${COMPONENT}=down"
    UNHEALTHY=1
    echo "${COMPONENT}: DOWN"
  fi
done

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "nothing was contacted (--dry-run)"
  exit 0
fi

if [ -n "${LOG_FILE}" ]; then
  if [ "${UNHEALTHY}" -eq 1 ]; then
    VERDICT="UNHEALTHY"
  elif [ "${UNVERIFIED}" -eq 1 ]; then
    VERDICT="unverified"
  else
    VERDICT="ok"
  fi
  printf '%s health: %s%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${VERDICT}" "${STATES}" >>"${LOG_FILE}"
fi
exit "${UNHEALTHY}"
