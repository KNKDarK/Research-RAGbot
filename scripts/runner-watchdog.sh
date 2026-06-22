#!/usr/bin/env bash
# runner-watchdog.sh
# Starts gitlab-runner, keeps it alive while pipeline jobs are running,
# and shuts it down when the pipeline is complete (idle timeout).
set -euo pipefail

RUNNER_CONFIG="${HOME}/.gitlab-runner/config.toml"
RUNNER_LOG="${HOME}/.gitlab-runner/runner.log"
IDLE_TIMEOUT=120

if [ ! -f "$RUNNER_CONFIG" ]; then
    echo "[watchdog] Runner config not found at $RUNNER_CONFIG" >> "$RUNNER_LOG"
    exit 1
fi

sleep 5

echo "[watchdog] Starting runner..." >> "$RUNNER_LOG"

gitlab-runner run --config "$RUNNER_CONFIG" >> "$RUNNER_LOG" 2>&1 &
RUNNER_PID=$!

cleanup() {
    kill "$RUNNER_PID" 2>/dev/null || true
    echo "[watchdog] Runner stopped." >> "$RUNNER_LOG"
}
trap cleanup EXIT

LAST_ACTIVITY=$(date +%s)
HAS_RUN_JOBS=false
LOG_POS=0
LOG_INO=0

while kill -0 "$RUNNER_PID" 2>/dev/null; do
    if [ -f "$RUNNER_LOG" ]; then
        CUR_INO=$(stat -c %i "$RUNNER_LOG" 2>/dev/null || echo 0)
        CUR_SIZE=$(stat -c %s "$RUNNER_LOG" 2>/dev/null || echo 0)

        if [ "$CUR_INO" != "$LOG_INO" ]; then
            LOG_POS=0
            LOG_INO=$CUR_INO
        fi

        if [ "$CUR_SIZE" -gt "$LOG_POS" ]; then
            NEW_DATA=$(dd if="$RUNNER_LOG" bs=1 skip="$LOG_POS" 2>/dev/null)
            LOG_POS=$CUR_SIZE

            if echo "$NEW_DATA" | grep -qE "(Job (succeeded|failed)|Checking for jobs.*received|Appending trace to coordinator|builds=[1-9])"; then
                HAS_RUN_JOBS=true
                LAST_ACTIVITY=$(date +%s)
            fi
        fi
    fi

    if [ "$HAS_RUN_JOBS" = true ] && [ $(( $(date +%s) - LAST_ACTIVITY )) -gt "$IDLE_TIMEOUT" ]; then
        echo "[watchdog] No job activity for ${IDLE_TIMEOUT}s, shutting down runner." >> "$RUNNER_LOG"
        kill "$RUNNER_PID" 2>/dev/null
        break
    fi

    sleep 5
done

echo "[watchdog] Runner finished. Pipeline complete." >> "$RUNNER_LOG"
