#!/usr/bin/env bash
# Unified production launcher for PAI DSW and ordinary Linux servers.
set -euo pipefail

TRAINING_UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$TRAINING_UI_DIR/nextjs_app/frontend"
ADAPTER_PATH="$TRAINING_UI_DIR/nextjs_app/dsw-adapter.js"

usage() {
  cat <<'EOF'
Usage: bash launch.sh [--local|--dsw]

Modes:
  auto (default)  Detect DSW from its environment or hostname.
  --local         Run at the URL root without the DSW adapter.
  --dsw           Run with the DSW basePath adapter.

Common environment variables:
  NEXT_PORT=8100             Browser-facing port
  BACKEND_PORT=8000          Internal FastAPI port
  VERBOSE=1                  Print complete build and server logs
  SKIP_BUILD=1               Reuse an existing matching Next.js build
  DIFFSYNTH_UI_MODE=auto     auto, local, or dsw
  PYTHON_BIN=python          Python executable for FastAPI

Local-only environment variables:
  DIFFSYNTH_UI_HOST=0.0.0.0  Next.js listen address
  DIFFSYNTH_UI_PUBLIC_URL=   URL printed after startup

DSW-only environment variables:
  DSW_URL=                   Full DSW proxy URL
  DSW_ID=dsw-123456          DSW instance ID
  NEXT_BASE_PATH=            Explicit DSW proxy basePath
  UPSTREAM_PORT=8101         Internal Next.js port
EOF
}

mode="${DIFFSYNTH_UI_MODE:-auto}"
case "${1:-}" in
  "") ;;
  --local) mode="local" ;;
  --dsw) mode="dsw" ;;
  -h|--help) usage; exit 0 ;;
  *) echo "[launch] unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
case "$mode" in
  auto|local|dsw) ;;
  *) echo "[launch] DIFFSYNTH_UI_MODE must be auto, local, or dsw." >&2; exit 2 ;;
esac

current_host="$(hostname)"
if [ "$mode" = "auto" ]; then
  if [ -n "${DSW_URL:-}" ] || [ -n "${DSW_ID:-}" ] || \
     [ -n "${VSCODE_PROXY_URI:-}" ] || [ -n "${NEXT_BASE_PATH:-}" ] || \
     [[ "$current_host" =~ ^dsw-[0-9]+ ]]; then
    mode="dsw"
  else
    mode="local"
  fi
fi

export DIFFSYNTH_STUDIO_ROOT="${DIFFSYNTH_STUDIO_ROOT:-$(cd "$TRAINING_UI_DIR/.." && pwd)}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export NEXT_TELEMETRY_DISABLED=1
export VERBOSE="${VERBOSE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
requested_next_port="${NEXT_PORT:-}"
OPEN_URL=""

if [ "$mode" = "dsw" ]; then
  if [ -z "${DSW_URL:-}" ] && [ -z "${NEXT_BASE_PATH:-}" ] && [ -z "${DSW_ID:-}" ]; then
    if [[ "$current_host" =~ ^(dsw-[0-9]+) ]]; then
      export DSW_ID="${BASH_REMATCH[1]}"
    elif [ -z "${VSCODE_PROXY_URI:-}" ]; then
      echo "[launch] cannot infer the DSW instance from hostname '$current_host'." >&2
      echo "[launch] set DSW_URL, DSW_ID, or NEXT_BASE_PATH." >&2
      exit 1
    fi
  fi

  export NEXT_PORT="${requested_next_port:-8100}"
  if [ -n "${DSW_URL:-}" ]; then
    derived_base="$(echo "$DSW_URL" | sed -E 's#^[a-zA-Z]+://[^/]+##; s#/+$##')"
    derived_port="$(echo "$derived_base" | sed -nE 's#.*/proxy/([0-9]+)$#\1#p')"
    if [ -z "$derived_base" ] || [ -z "$derived_port" ]; then
      echo "[launch] cannot derive basePath and port from DSW_URL='$DSW_URL'." >&2
      exit 1
    fi
    export NEXT_BASE_PATH="${NEXT_BASE_PATH:-$derived_base}"
    if [ -z "$requested_next_port" ]; then
      export NEXT_PORT="$derived_port"
    fi
    OPEN_URL="${DSW_URL%/}/dashboard"
  elif [ -z "${NEXT_BASE_PATH:-}" ] && [ -n "${VSCODE_PROXY_URI:-}" ]; then
    proxy_url="${VSCODE_PROXY_URI//\{\{port\}\}/${NEXT_PORT}}"
    derived_base="$(echo "$proxy_url" | sed -E 's#^[a-zA-Z]+://[^/]+##; s#/+$##')"
    derived_port="$(echo "$derived_base" | sed -nE 's#.*/proxy/([0-9]+)$#\1#p')"
    if [ -n "$derived_base" ] && [ -n "$derived_port" ]; then
      export NEXT_BASE_PATH="$derived_base"
    elif [ -n "${DSW_ID:-}" ]; then
      export NEXT_BASE_PATH="/${DSW_ID}/ide/proxy/${NEXT_PORT}"
    else
      echo "[launch] unrecognized VSCODE_PROXY_URI='$VSCODE_PROXY_URI'." >&2
      exit 1
    fi
    OPEN_URL="${proxy_url%/}/dashboard"
  elif [ -n "${DSW_ID:-}" ]; then
    export NEXT_BASE_PATH="${NEXT_BASE_PATH:-/${DSW_ID}/ide/proxy/${NEXT_PORT}}"
  fi

  if [ -z "${NEXT_BASE_PATH:-}" ]; then
    echo "[launch] DSW mode requires DSW_URL, DSW_ID, VSCODE_PROXY_URI, or NEXT_BASE_PATH." >&2
    exit 1
  fi
  export NEXT_BASE_PATH="${NEXT_BASE_PATH%/}"
  export UPSTREAM_PORT="${UPSTREAM_PORT:-$((NEXT_PORT + 1))}"
  export NEXT_SERVER_PORT="$UPSTREAM_PORT"
  export NEXT_SERVER_HOST=127.0.0.1
  export DIFFSYNTH_LAUNCHED_BY_SCRIPT=1
  OPEN_URL="${OPEN_URL:-http://127.0.0.1:${NEXT_PORT}/dashboard}"

  tail_segment="${NEXT_BASE_PATH##*/}"
  if [[ "$tail_segment" =~ ^[0-9]+$ ]] && [ "$tail_segment" != "$NEXT_PORT" ]; then
    echo "[launch] NEXT_BASE_PATH port $tail_segment differs from NEXT_PORT $NEXT_PORT." >&2
    exit 1
  fi
else
  export NEXT_PORT="${requested_next_port:-8100}"
  export NEXT_BASE_PATH=""
  export NEXT_SERVER_PORT="$NEXT_PORT"
  export NEXT_SERVER_HOST="${DIFFSYNTH_UI_HOST:-0.0.0.0}"
  OPEN_URL="${DIFFSYNTH_UI_PUBLIC_URL:-http://127.0.0.1:${NEXT_PORT}/dashboard}"
fi

export DIFFSYNTH_UI_BACKEND="${DIFFSYNTH_UI_BACKEND:-http://127.0.0.1:${BACKEND_PORT}}"
export LOG_DIR="${DIFFSYNTH_UI_LOG_DIR:-/tmp/diffsynth-train-ui-${NEXT_PORT}}"

for command_name in "$PYTHON_BIN" node npm curl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "[launch] missing command: $command_name" >&2
    exit 1
  fi
done
if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, multipart" >/dev/null 2>&1; then
  echo "[launch] Python Web UI dependencies are incomplete." >&2
  exit 1
fi

check_port_available() {
  local port="$1"
  local status
  set +e
  "$PYTHON_BIN" - "$port" <<'PY'
import errno
import socket
import sys

port = int(sys.argv[1])
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    finally:
        sock.close()
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1 if exc.errno == errno.EADDRINUSE else 2)
PY
  status=$?
  set -e
  if [ "$status" -eq 1 ]; then
    echo "[launch] port $port is already in use." >&2
    exit 1
  elif [ "$status" -ne 0 ]; then
    echo "[launch] unable to verify whether port $port is available." >&2
    exit 1
  fi
}

ports=("$NEXT_PORT" "$BACKEND_PORT")
if [ "$mode" = "dsw" ]; then ports+=("$UPSTREAM_PORT"); fi
for ((i = 0; i < ${#ports[@]}; i++)); do
  for ((j = i + 1; j < ${#ports[@]}; j++)); do
    if [ "${ports[$i]}" = "${ports[$j]}" ]; then
      echo "[launch] service ports must be different: ${ports[$i]}." >&2
      exit 1
    fi
  done
  check_port_available "${ports[$i]}"
done
mkdir -p "$LOG_DIR"

run_logged() {
  local label="$1" logfile="$2"
  shift 2
  if [ "$VERBOSE" = "1" ]; then
    echo "[launch] $label"
    "$@"
  else
    echo "[launch] $label ..."
    if ! "$@" >"$logfile" 2>&1; then
      echo "[launch] $label failed; log: $logfile" >&2
      tail -n 80 "$logfile" >&2 || true
      exit 1
    fi
  fi
}

wait_for_url() {
  local label="$1" url="$2" logfile="$3" code
  for _ in $(seq 1 40); do
    code="$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$url" || true)"
    case "$code" in
      200|302|307|308) echo "[launch] $label ready"; return 0 ;;
    esac
    sleep 0.5
  done
  echo "[launch] $label did not become ready: $url" >&2
  tail -n 80 "$logfile" >&2 || true
  exit 1
}

child_pids=()
cleanup() {
  trap - EXIT INT TERM
  if [ "${#child_pids[@]}" -gt 0 ]; then
    echo "[launch] shutting down..."
    kill "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[launch] mode=${mode}"
if [ "$mode" = "dsw" ]; then
  echo "[launch] external=${NEXT_PORT}, next-internal=${UPSTREAM_PORT}, backend=${BACKEND_PORT}"
else
  echo "[launch] frontend=${NEXT_SERVER_HOST}:${NEXT_PORT}, backend=127.0.0.1:${BACKEND_PORT}"
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  (
    cd "$FRONTEND_DIR"
    if [ ! -d node_modules ]; then
      run_logged "npm install" "$LOG_DIR/npm-install.log" npm install
    fi
    build_path="${NEXT_BASE_PATH:-root}"
    run_logged "next build (basePath=${build_path})" "$LOG_DIR/next-build.log" \
      env NEXT_BASE_PATH="$NEXT_BASE_PATH" DIFFSYNTH_UI_BACKEND="$DIFFSYNTH_UI_BACKEND" \
      npx --no-install next build
    run_logged "write basePath snapshot" "$LOG_DIR/write-base-path.log" \
      env NEXT_BASE_PATH="$NEXT_BASE_PATH" node scripts/write-base-path.js
  )
fi

(
  cd "$TRAINING_UI_DIR"
  if [ "$VERBOSE" = "1" ]; then
    exec "$PYTHON_BIN" -m uvicorn nextjs_app.backend.main:app --host 127.0.0.1 --port "$BACKEND_PORT"
  else
    exec "$PYTHON_BIN" -m uvicorn nextjs_app.backend.main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
      --no-access-log --log-level warning >"$LOG_DIR/backend.log" 2>&1
  fi
) &
child_pids+=("$!")
wait_for_url "backend" "http://127.0.0.1:${BACKEND_PORT}/api/health" "$LOG_DIR/backend.log"

(
  cd "$FRONTEND_DIR"
  run_logged "check build" "$LOG_DIR/check-build.log" env NEXT_BASE_PATH="$NEXT_BASE_PATH" node scripts/check-build.js
  if [ "$VERBOSE" = "1" ]; then
    exec env NEXT_BASE_PATH="$NEXT_BASE_PATH" DIFFSYNTH_UI_BACKEND="$DIFFSYNTH_UI_BACKEND" \
      npx --no-install next start -p "$NEXT_SERVER_PORT" -H "$NEXT_SERVER_HOST"
  else
    exec env NEXT_BASE_PATH="$NEXT_BASE_PATH" DIFFSYNTH_UI_BACKEND="$DIFFSYNTH_UI_BACKEND" \
      npx --no-install next start -p "$NEXT_SERVER_PORT" -H "$NEXT_SERVER_HOST" \
      >"$LOG_DIR/next-start.log" 2>&1
  fi
) &
child_pids+=("$!")
wait_for_url "next" "http://127.0.0.1:${NEXT_SERVER_PORT}${NEXT_BASE_PATH}/dashboard" "$LOG_DIR/next-start.log"

if [ "$mode" = "dsw" ]; then
  (
    if [ "$VERBOSE" = "1" ]; then
      exec env BASE_PATH="$NEXT_BASE_PATH" LISTEN_PORT="$NEXT_PORT" \
        UPSTREAM_PORT="$UPSTREAM_PORT" node "$ADAPTER_PATH"
    else
      exec env BASE_PATH="$NEXT_BASE_PATH" LISTEN_PORT="$NEXT_PORT" \
        UPSTREAM_PORT="$UPSTREAM_PORT" QUIET=1 node "$ADAPTER_PATH" \
        >"$LOG_DIR/dsw-adapter.log" 2>&1
    fi
  ) &
  child_pids+=("$!")
  wait_for_url "adapter" "http://127.0.0.1:${NEXT_PORT}/dashboard" "$LOG_DIR/dsw-adapter.log"
fi

echo "[launch] ready"
echo "[launch] open: ${OPEN_URL}"
wait
