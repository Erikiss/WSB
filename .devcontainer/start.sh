#!/usr/bin/env bash
# Runs on every Codespace start: (re)start the WSBench server on port 8765. Safe to run by hand.
set -euo pipefail
PORT=8765
LOG=/tmp/wsbench.log
if pgrep -f "wsbench serve" >/dev/null; then
  pkill -f "wsbench serve" || true
  sleep 1
fi
unset WSB_LLM
nohup setsid wsbench serve --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "WSBench is running on port $PORT:"
    curl -fsS "http://127.0.0.1:$PORT/api/health"; echo
    exit 0
  fi
  sleep 1
done
echo "WSBench did not start; see $LOG" >&2
exit 1
