#!/usr/bin/env bash
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="${APP_PATH:-Streamlit_code/app.py}"
HOST="${STREAMLIT_HOST:-127.0.0.1}"
PORT="${STREAMLIT_PORT:-8501}"
RESTART_DELAY="${RESTART_DELAY:-5}"
LOG_DIR="${LOG_DIR:-$APP_DIR/logs}"
LOG_FILE="$LOG_DIR/streamlit_watchdog.log"

mkdir -p "$LOG_DIR"
cd "$APP_DIR" || exit 1

if [[ -n "${STREAMLIT_CMD:-}" ]]; then
    read -r -a STREAMLIT_RUNNER <<< "$STREAMLIT_CMD"
elif command -v streamlit >/dev/null 2>&1; then
    STREAMLIT_RUNNER=(streamlit)
else
    STREAMLIT_RUNNER=(python3 -m streamlit)
fi

echo "[$(date)] Watchdog started for $APP_PATH on http://$HOST:$PORT" | tee -a "$LOG_FILE"
echo "[$(date)] Using command: ${STREAMLIT_RUNNER[*]}" | tee -a "$LOG_FILE"

while true; do
    echo "[$(date)] Starting Streamlit..." | tee -a "$LOG_FILE"
    "${STREAMLIT_RUNNER[@]}" run "$APP_PATH" \
        --server.address "$HOST" \
        --server.port "$PORT" \
        2>&1 | tee -a "$LOG_FILE"

    status=${PIPESTATUS[0]}
    echo "[$(date)] Streamlit exited with status $status. Restarting in ${RESTART_DELAY}s..." | tee -a "$LOG_FILE"
    sleep "$RESTART_DELAY"
done
