#!/usr/bin/env bash
# runner-watchdog.sh
# Starts gitlab-runner, keeps it alive while pipeline jobs are running,
# and shuts it down when the pipeline is complete (idle timeout).
set -euo pipefail

RUNNER_CONFIG="${HOME}/.gitlab-runner/config.toml"
RUNNER_LOG="${HOME}/.gitlab-runner/runner.log"
IDLE_TIMEOUT=120  # seconds of no job activity before shutdown

if [ ! -f "$RUNNER_CONFIG" ]; then
    echo "[watchdog] Runner config not found at $RUNNER_CONFIG" >> "$RUNNER_LOG"
    exit 1
fi

# Give GitLab time to receive the push and trigger the pipeline
sleep 5

echo "[watchdog] Starting runner..." >> "$RUNNER_LOG"

# Start runner in background (no max-builds — runs until killed)
gitlab-runner run --config "$RUNNER_CONFIG" >> "$RUNNER_LOG" 2>&1 &
RUNNER_PID=$!

cleanup() {
    kill "$RUNNER_PID" 2>/dev/null || true
    echo "[watchdog] Runner stopped." >> "$RUNNER_LOG"
}
trap cleanup EXIT

LAST_ACTIVITY=$(date +%s)
HAS_RUN_JOBS=false

while kill -0 "$RUNNER_PID" 2>/dev/null; do
    # Check recent log lines for job activity
    RECENT=$(tail -50 "$RUNNER_LOG" 2>/dev/null || true)

    if echo "$RECENT" | grep -qE "(Job succeeded|Job failed|Job.*failed)"; then
        HAS_RUN_JOBS=true
        LAST_ACTIVITY=$(date +%s)
    fi

    if echo "$RECENT" | grep -q "Checking for jobs... received"; then
        HAS_RUN_JOBS=true
        LAST_ACTIVITY=$(date +%s)
    fi

    if echo "$RECENT" | grep -q "Appending trace to coordinator"; then
        LAST_ACTIVITY=$(date +%s)
    fi

    # Check if there are any active builds currently running
    ACTIVE_BUILDS=$(grep -oE "builds=[0-9]+" "$RUNNER_LOG" 2>/dev/null | tail -n 1 | cut -d= -f2 || echo 0)
    ACTIVE_BUILDS=${ACTIVE_BUILDS:-0}
    if [ "$ACTIVE_BUILDS" -gt 0 ]; then
        LAST_ACTIVITY=$(date +%s)
    fi

    # Only enforce idle timeout after at least one job has run
    if [ "$HAS_RUN_JOBS" = true ]; then
        NOW=$(date +%s)
        ELAPSED=$((NOW - LAST_ACTIVITY))
        if [ "$ELAPSED" -gt "$IDLE_TIMEOUT" ]; then
            echo "[watchdog] No job activity for ${IDLE_TIMEOUT}s, shutting down runner." >> "$RUNNER_LOG"
            kill "$RUNNER_PID" 2>/dev/null
            break
        fi
    fi

    sleep 5
done

echo "[watchdog] Runner finished. Pipeline complete." >> "$RUNNER_LOG"
