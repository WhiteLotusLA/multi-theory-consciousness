#!/usr/bin/env bash
# Report listeners that are reachable from other machines and private files
# that other users can read.
#
# It reports, and it prints the exact configuration line to change; it never
# edits a store's configuration, because that is an operator decision. File
# modes are changed only with --fix-permissions. Settings come from the
# project's Settings through `python3 -m mtc.core.env_export`. Run it from the
# project directory with the project's virtual environment active.
set -euo pipefail

SCRIPT_NAME="harden_services.sh"

usage() {
  cat <<'EOF'
Usage: scripts/ops/harden_services.sh [--fix-permissions] [--dry-run]

1. Listeners. Lists the TCP listeners on the five store ports, the model
   server port and the API port (lsof -nP -iTCP -sTCP:LISTEN) and reports
   every one that is not bound to a loopback address, with the configuration
   line that binds it to loopback. Nothing is edited. An expected service
   (an enabled store, the model server under the local provider, the API)
   with no visible listener is named as not measured: it may be stopped, or
   it may run as another user, whose sockets lsof lists only with root.
2. File modes. The environment file and the speakers file should be 0600,
   the agent space and the backup location 0700. A looser mode is reported;
   with --fix-permissions it is corrected.

Options:
  --fix-permissions  set the modes above (the only thing this script changes)
  --dry-run          print what would be listed and changed; change nothing
  --help             show this text

Exit codes: 0 nothing to report, 1 a finding (a listener off loopback, a
loose mode that was not fixed, or a check that could not run), 2 usage or
settings that cannot be read.
EOF
}

FIX=0
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help | -h)
      usage
      exit 0
      ;;
    --fix-permissions) FIX=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *)
      echo "${SCRIPT_NAME}: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! SETTINGS="$(python3 -m mtc.core.env_export PROJECT_ROOT POSTGRES_PORT \
  MONGODB_PORT QDRANT_PORT REDIS_PORT NEO4J_URI LLM_PORT API_PORT \
  AGENT_SPACE_DIR SPEAKERS_PATH BACKUP_LOCATION POSTGRES_ENABLED \
  MONGODB_ENABLED QDRANT_ENABLED REDIS_ENABLED NEO4J_ENABLED LLM_PROVIDER)"; then
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
    ./*) printf '%s/%s\n' "${MTC_PROJECT_ROOT}" "${1#./}" ;;
    *) printf '%s/%s\n' "${MTC_PROJECT_ROOT}" "$1" ;;
  esac
}

NEO4J_PORT="${MTC_NEO4J_URI#*://}"
NEO4J_PORT="${NEO4J_PORT%%/*}"
NEO4J_PORT="${NEO4J_PORT##*:}"
FINDINGS=0
# Ports lsof showed a listener on (space separated), and the expected
# services it showed none for.
SEEN_PORTS=" "
UNSEEN=""

expected_services() {
  # "service:port" for every service this configuration expects to listen.
  if [ "${MTC_POSTGRES_ENABLED}" = "true" ]; then echo "postgres:${MTC_POSTGRES_PORT}"; fi
  if [ "${MTC_MONGODB_ENABLED}" = "true" ]; then echo "mongodb:${MTC_MONGODB_PORT}"; fi
  if [ "${MTC_REDIS_ENABLED}" = "true" ]; then echo "redis:${MTC_REDIS_PORT}"; fi
  if [ "${MTC_QDRANT_ENABLED}" = "true" ]; then echo "qdrant:${MTC_QDRANT_PORT}"; fi
  if [ "${MTC_NEO4J_ENABLED}" = "true" ]; then echo "neo4j:${NEO4J_PORT}"; fi
  if [ "${MTC_LLM_PROVIDER}" = "local" ]; then echo "model server:${MTC_LLM_PORT}"; fi
  echo "api:${MTC_API_PORT}"
}

service_for_port() {
  case "$1" in
    "${MTC_POSTGRES_PORT}") echo "postgres" ;;
    "${MTC_MONGODB_PORT}") echo "mongodb" ;;
    "${MTC_REDIS_PORT}") echo "redis" ;;
    "${MTC_QDRANT_PORT}") echo "qdrant" ;;
    "${NEO4J_PORT}") echo "neo4j" ;;
    "${MTC_LLM_PORT}") echo "model server" ;;
    "${MTC_API_PORT}") echo "api" ;;
    *) echo "" ;;
  esac
}

loopback_hint() {
  # The line that binds a service to loopback, for Homebrew and apt installs.
  case "$1" in
    postgres)
      cat <<'EOF'
    in postgresql.conf set:    listen_addresses = '127.0.0.1'
    (Homebrew: under "$(brew --prefix)/var"; Debian and Ubuntu: under /etc/postgresql)
EOF
      ;;
    mongodb)
      cat <<'EOF'
    in mongod.conf set:        net.bindIp: 127.0.0.1
    (Homebrew: "$(brew --prefix)/etc/mongod.conf"; Debian and Ubuntu: /etc/mongod.conf)
EOF
      ;;
    redis)
      cat <<'EOF'
    in redis.conf set:         bind 127.0.0.1 ::1
                               protected-mode yes
    (Homebrew: "$(brew --prefix)/etc/redis.conf"; Debian and Ubuntu: /etc/redis/redis.conf)
EOF
      ;;
    neo4j)
      cat <<'EOF'
    in neo4j.conf set:         server.default_listen_address=127.0.0.1
    (Homebrew: the conf directory "neo4j --version" reports; Debian and Ubuntu: /etc/neo4j/neo4j.conf)
EOF
      ;;
    qdrant)
      cat <<'EOF'
    in config/qdrant.yaml set: service.host: 127.0.0.1
    (or the environment variable QDRANT__SERVICE__HOST=127.0.0.1)
EOF
      ;;
    "model server")
      cat <<'EOF'
    start it with its host option set to 127.0.0.1 (LLM_HOST in the environment file)
EOF
      ;;
    api)
      cat <<'EOF'
    set MTC_API_HOST=127.0.0.1 in the environment file (a bind off loopback needs authentication)
EOF
      ;;
  esac
}

check_listeners() {
  local listing line name address port service
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "would run: lsof -nP -iTCP -sTCP:LISTEN"
    return 0
  fi
  if ! have lsof; then
    echo "listeners: not checked: lsof is not installed" >&2
    FINDINGS=1
    return 0
  fi
  # lsof exits non-zero when nothing listens at all.
  listing="$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null || true)"
  while IFS= read -r line; do
    case "${line}" in
      *"(LISTEN)"*) ;;
      *) continue ;;
    esac
    name="${line% (LISTEN)*}"
    name="${name##* }"
    port="${name##*:}"
    address="${name%:*}"
    service="$(service_for_port "${port}")"
    if [ -z "${service}" ]; then
      continue
    fi
    SEEN_PORTS="${SEEN_PORTS}${port} "
    case "${address}" in
      127.* | "[::1]")
        echo "listener: ${service} on ${address} port ${port} (loopback)"
        ;;
      *)
        echo "FINDING: ${service} listens on ${address} port ${port}, which other machines can reach"
        loopback_hint "${service}"
        FINDINGS=1
        ;;
    esac
  done <<EOF
${listing}
EOF
  local entry
  while IFS= read -r entry; do
    [ -n "${entry}" ] || continue
    case "${SEEN_PORTS}" in
      *" ${entry##*:} "*) ;;
      *)
        UNSEEN="${UNSEEN:+${UNSEEN}, }${entry%:*} (port ${entry##*:})"
        ;;
    esac
  done <<EOF
$(expected_services)
EOF
  if [ -n "${UNSEEN}" ]; then
    echo "no listener visible: ${UNSEEN}. Not measured: each is stopped, or runs as another user whose sockets lsof lists only when run as root"
  fi
}

mode_of() {
  python3 -c 'import os, stat, sys; print(format(stat.S_IMODE(os.stat(sys.argv[1]).st_mode), "o"))' "$1"
}

check_mode() {
  # check_mode PATH WANTED LABEL: report, and with --fix-permissions set, a
  # mode that gives group or others any access.
  local path="$1" wanted="$2" label="$3" mode
  if [ ! -e "${path}" ]; then
    return 0
  fi
  mode="$(mode_of "${path}")"
  case "${mode}" in
    *00) return 0 ;;
  esac
  if [ "${FIX}" -eq 1 ]; then
    if [ "${DRY_RUN}" -eq 1 ]; then
      echo "would run: chmod ${wanted} ${label}"
      return 0
    fi
    chmod "${wanted}" "${path}"
    echo "fixed: ${label} was mode ${mode}, now ${wanted}"
    return 0
  fi
  echo "FINDING: ${label} has mode ${mode}; other users can read it. Run this again with --fix-permissions (chmod ${wanted})"
  FINDINGS=1
}

check_listeners
check_mode "${MTC_PROJECT_ROOT}/.env" 600 "the environment file"
check_mode "$(project_path "${MTC_SPEAKERS_PATH}")" 600 "the speakers file (${MTC_SPEAKERS_PATH})"
check_mode "$(project_path "${MTC_AGENT_SPACE_DIR}")" 700 "the agent space (${MTC_AGENT_SPACE_DIR})"
if [ -n "${MTC_BACKUP_LOCATION}" ]; then
  check_mode "$(project_path "${MTC_BACKUP_LOCATION}")" 700 "the backup location (${MTC_BACKUP_LOCATION})"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "nothing was changed (--dry-run)"
  exit 0
fi
if [ "${FINDINGS}" -eq 0 ] && [ -n "${UNSEEN}" ]; then
  echo "no finding among what was measured: every visible listener is on loopback and the private files are private; the services named above were not measured"
elif [ "${FINDINGS}" -eq 0 ]; then
  echo "no finding: every listed listener is on loopback and the private files are private"
fi
exit "${FINDINGS}"
