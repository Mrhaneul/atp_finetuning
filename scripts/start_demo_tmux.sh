#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${TMUX_SESSION_NAME:-streamlit_demo}"
WATCHDOG="$APP_DIR/scripts/streamlit_watchdog.sh"

cd "$APP_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' is already running."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "$WATCHDOG"

echo "Started Streamlit watchdog in tmux session '$SESSION_NAME'."
echo "App URL: http://${STREAMLIT_HOST:-127.0.0.1}:${STREAMLIT_PORT:-8501}"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Logs: $APP_DIR/logs/streamlit_watchdog.log"
