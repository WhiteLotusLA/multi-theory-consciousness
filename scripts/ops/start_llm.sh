#!/usr/bin/env bash
# Start the local model server that LLM_BACKEND names (mlx_lm or llama_cpp).
#
# Settings come from the project's Settings through
# `python3 -m mtc.core.env_export`; nothing is restated here. Run it from the
# project directory with the project's virtual environment active.
set -euo pipefail

SCRIPT_NAME="start_llm.sh"

usage() {
  cat <<'EOF'
Usage: scripts/ops/start_llm.sh [--background] [--wait SECONDS] [--dry-run]

Starts the chat-completions server for LLM_PROVIDER=local.

  LLM_BACKEND=mlx_lm     python3 -m mlx_lm.server --model MODEL_PATH
                         --host LLM_HOST --port LLM_PORT
  LLM_BACKEND=llama_cpp  llama-server -m MODEL_PATH --host LLM_HOST
                         --port LLM_PORT -c CONTEXT_SIZE -b BATCH_SIZE,
                         plus -ngl GPU_LAYERS (left out at -1), -t THREADS
                         (left out at 0) and the memory-mapping flags for
                         USE_MMAP and USE_MLOCK (the form the installed
                         llama-server accepts)

Options:
  --background    start it detached, write its PID under
                  <DATA_DIRECTORY>/run/llm.pid and log to
                  <LOG_DIRECTORY>/llm.log
  --wait SECONDS  with --background, wait until the server answers a health
                  path (a cold load of a large model can take minutes)
  --dry-run       print the command and start nothing
  --help          show this text

Exit codes: 0 started (or already running), 1 it did not come up, 2 usage or
configuration (an empty LLM_BACKEND, a missing MODEL_PATH, a missing server
program).
EOF
}

BACKGROUND=0
WAIT_SECONDS=0
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help | -h)
      usage
      exit 0
      ;;
    --background) BACKGROUND=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --wait)
      if [ "$#" -lt 2 ]; then
        echo "${SCRIPT_NAME}: --wait needs a number of seconds" >&2
        exit 2
      fi
      WAIT_SECONDS="$2"
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
case "${WAIT_SECONDS}" in
  '' | *[!0-9]*)
    echo "${SCRIPT_NAME}: --wait needs a whole number of seconds" >&2
    exit 2
    ;;
esac

if ! SETTINGS="$(python3 -m mtc.core.env_export PROJECT_ROOT DATA_DIRECTORY \
  LOG_DIRECTORY LLM_BACKEND MODEL_PATH LLM_HOST LLM_PORT LLM_BASE_URL \
  LLM_HEALTH_PATHS CONTEXT_SIZE GPU_LAYERS THREADS BATCH_SIZE USE_MMAP \
  USE_MLOCK)"; then
  echo "${SCRIPT_NAME}: the settings could not be read. Activate the project's virtual environment and run this from the project directory." >&2
  exit 2
fi
eval "${SETTINGS}"

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

health_answers() {
  # True when one of LLM_HEALTH_PATHS answers with success.
  local path
  while IFS= read -r path; do
    path="${path# }"
    path="${path% }"
    [ -n "${path}" ] || continue
    if have curl; then
      if curl --silent --fail --max-time 3 --output /dev/null \
        "${MTC_LLM_BASE_URL}/${path#/}"; then
        return 0
      fi
    fi
  done <<EOF
$(printf '%s' "${MTC_LLM_HEALTH_PATHS}" | tr ',' '\n')
EOF
  return 1
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

if [ -z "${MTC_LLM_BACKEND}" ]; then
  echo "${SCRIPT_NAME}: LLM_BACKEND is empty. Set MTC_LLM_BACKEND to mlx_lm or llama_cpp in the environment file, or start your own chat-completions server on LLM_PORT." >&2
  exit 2
fi

if [ "${DRY_RUN}" -eq 0 ] && tcp_open "${MTC_LLM_HOST}" "${MTC_LLM_PORT}"; then
  if health_answers; then
    echo "${SCRIPT_NAME}: a model server already answers at ${MTC_LLM_HOST} port ${MTC_LLM_PORT}; nothing to start"
    exit 0
  fi
  echo "${SCRIPT_NAME}: port ${MTC_LLM_PORT} on ${MTC_LLM_HOST} is in use by something that does not answer the health paths; nothing was started" >&2
  exit 1
fi

MODEL="$(project_path "${MTC_MODEL_PATH}")"
CMD=()
case "${MTC_LLM_BACKEND}" in
  mlx_lm)
    if [ ! -d "${MODEL}" ]; then
      echo "${SCRIPT_NAME}: MODEL_PATH (${MTC_MODEL_PATH}) is not a directory. For mlx_lm it is the directory of a converted model." >&2
      exit 2
    fi
    # mlx_lm.server takes the model, the host and the port; the context size
    # and the other llama.cpp settings have no flag there.
    CMD=(python3 -m mlx_lm.server --model "${MODEL}" --host "${MTC_LLM_HOST}"
      --port "${MTC_LLM_PORT}")
    ;;
  llama_cpp)
    if [ ! -f "${MODEL}" ]; then
      echo "${SCRIPT_NAME}: MODEL_PATH (${MTC_MODEL_PATH}) is not a file. For llama_cpp it is a GGUF model file." >&2
      exit 2
    fi
    CMD=(llama-server -m "${MODEL}" --host "${MTC_LLM_HOST}"
      --port "${MTC_LLM_PORT}" -c "${MTC_CONTEXT_SIZE}" -b "${MTC_BATCH_SIZE}")
    if [ "${MTC_GPU_LAYERS}" != "-1" ]; then
      CMD+=(-ngl "${MTC_GPU_LAYERS}")
    fi
    if [ "${MTC_THREADS}" != "0" ]; then
      CMD+=(-t "${MTC_THREADS}")
    fi
    HELP_TEXT=""
    if have llama-server; then
      HELP_TEXT="$(llama-server --help 2>&1 || true)"
    elif [ "${DRY_RUN}" -eq 0 ]; then
      echo "${SCRIPT_NAME}: llama-server is not on PATH. Install llama.cpp, or set LLM_BACKEND to match the server you run." >&2
      exit 2
    fi
    case "${HELP_TEXT}" in
      *--load-mode*)
        # Newer builds: one loading-mode option; mmap alone is its default.
        case "${MTC_USE_MMAP}:${MTC_USE_MLOCK}" in
          true:true) CMD+=(--load-mode mmap+mlock) ;;
          false:true) CMD+=(--load-mode mlock) ;;
          false:false) CMD+=(--load-mode none) ;;
        esac
        ;;
      *)
        if [ "${MTC_USE_MMAP}" = "false" ]; then
          CMD+=(--no-mmap)
        fi
        if [ "${MTC_USE_MLOCK}" = "true" ]; then
          CMD+=(--mlock)
        fi
        ;;
    esac
    ;;
  *)
    echo "${SCRIPT_NAME}: LLM_BACKEND=${MTC_LLM_BACKEND} is not one of mlx_lm, llama_cpp" >&2
    exit 2
    ;;
esac

if [ "${DRY_RUN}" -eq 1 ]; then
  printf 'would run:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  if [ "${BACKGROUND}" -eq 1 ]; then
    echo "would run it detached, with its PID in <DATA_DIRECTORY>/run/llm.pid and its log in <LOG_DIRECTORY>/llm.log"
  fi
  exit 0
fi

if [ "${MTC_LLM_BACKEND}" = "mlx_lm" ]; then
  if ! python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("mlx_lm") else 1)'; then
    echo "${SCRIPT_NAME}: the mlx_lm package is not installed in this environment (pip install mlx-lm)" >&2
    exit 2
  fi
fi

if [ "${BACKGROUND}" -eq 0 ]; then
  exec "${CMD[@]}"
fi

RUN_DIR="$(project_path "${MTC_DATA_DIRECTORY}")/run"
LOG_DIR="$(project_path "${MTC_LOG_DIRECTORY}")"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
nohup "${CMD[@]}" >>"${LOG_DIR}/llm.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${RUN_DIR}/llm.pid"
echo "${SCRIPT_NAME}: started the model server (PID ${SERVER_PID}); its log is ${MTC_LOG_DIRECTORY}/llm.log"

if [ "${WAIT_SECONDS}" -gt 0 ]; then
  if wait_for "${WAIT_SECONDS}" 5 health_answers; then
    echo "${SCRIPT_NAME}: the model server answers"
  else
    echo "${SCRIPT_NAME}: no answer after ${WAIT_SECONDS} seconds; it may still be loading (see the log)" >&2
    exit 1
  fi
fi
exit 0
