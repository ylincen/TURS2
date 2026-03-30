#!/usr/bin/env bash
# Use the user's default Bash (via env), more portable than /bin/bash.

# run_settings_local.sh
# Short description of what the script does.

set -euo pipefail
# -e : exit immediately if any command returns non-zero
# -u : error on using unset variables
# -o pipefail : a pipeline fails if any command in it fails

# -------- config knobs (override via env vars) ----------
# how many settings to run at the same time
: "${MAX_JOBS:=$(getconf _NPROCESSORS_ONLN || echo 4)}"
# If MAX_JOBS not set, set it to the number of online CPUs; fall back to 4 if that fails.

# where to write logs
: "${LOG_DIR:=RIVIUM_OUT_20251102}"
# If LOG_DIR not set, default to "slurm_out".

# python executable
: "${PY:=python}"
# If PY not set, default to "python".

# threads per process for BLAS/OpenMP (1 = single-threaded)
: "${BLAS_THREADS:=1}"
# If BLAS_THREADS not set, default to 1 (prevents oversubscription).
# -------------------------------------------------------

cd "$(dirname "$0")"
# Change working directory to the script's location so relative paths are stable.

mkdir -p "$LOG_DIR"
# Ensure the log directory exists.

# cap threading so we don't oversubscribe the machine
export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"
export PYTHONUNBUFFERED=1
# Set thread caps for common math libs; make Python flush output immediately.

# count settings
N="$("$PY" exp_suite.py --list-settings | wc -l)"
# Run your Python script to list settings and count the lines = number of settings.

if [[ "$N" -eq 0 ]]; then
  echo "[ERROR] No settings found (did you implement --list-settings?)" >&2
  exit 1
fi
# Safety check: bail out if there are no settings.

echo "[INFO] Found $N settings"
echo "[INFO] Running up to $MAX_JOBS settings in parallel"
echo "[INFO] Logs -> $LOG_DIR/setting_<idx>.log"
# Informative banners.

pids=()
# Initialize a Bash array to track child process PIDs.

# portable concurrency gate (works without bash 4.3's wait -n)
wait_for_slot() {
  while true; do
    # count running background jobs
    local running
    running=$(jobs -r -p | wc -l | awk '{print $1}')
    # 'jobs -r -p' lists PIDs of running background jobs; count them.
    if (( running < MAX_JOBS )); then
      break
    fi
    sleep 0.2
    # If we've hit the limit, wait briefly and check again.
  done
}
# Function: blocks until fewer than MAX_JOBS background jobs are running.

# clean shutdown: kill all children on Ctrl-C/termination
cleanup() {
  echo
  echo "[INFO] Cleaning up (${#pids[@]} children)..."
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    # Send SIGTERM to all tracked children; ignore errors if some already exited.
    wait || true
    # Wait for them to finish; don't fail the script if wait returns non-zero.
  fi
}
trap cleanup INT TERM
# Register the cleanup function to run on Ctrl-C (INT) or termination (TERM).

for (( i=1; i<=N; i++ )); do
  wait_for_slot
  # Ensure we don't exceed MAX_JOBS concurrent children.
  {
    set +e                                 # don't abort this subshell on failure
    echo "[$(date)] START setting $i"
    echo "CMD: $PY exp_suite.py --setting-idx $i"
    "$PY" exp_suite.py --setting-idx "$i"
    rc=$?
    echo "[$(date)] END setting $i (rc=$rc)"
    exit $rc                               # propagate status so parent 'wait' sees it
  } >"$LOG_DIR/setting_${i}.log" 2>&1 &
  # {
  #   echo "[$(date)] START setting $i"
  #   "$PY" exp_suite.py --setting-idx "$i"
  #   # Run exactly one setting (datasets inside run serially).
  #   rc=$?
  #   # Capture the program's exit code.

  #   echo "[$(date)] END setting $i (rc=$rc)"
  #   exit $rc
  #   # Exit the subshell with the same exit code so the parent can detect failures.
  # } >"$LOG_DIR/setting_${i}.log" 2>&1 &
  # Run the block in the background, redirect stdout+stderr to a per-setting log.

  pids+=("$!")
  # Record the PID ($! is the last background command's PID).

  echo "[LAUNCHED] setting $i (pid ${pids[-1]}) -> $LOG_DIR/setting_${i}.log"
  # Print a quick launch line.
done
# Launch a background job for each setting index 1..N.

# wait for all
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
# Wait on each child; if any exits non-zero, mark overall failure.

if (( fail )); then
  echo "[DONE] Some settings failed. Check logs in $LOG_DIR." >&2
  exit 1
fi
# If any job failed, exit non-zero and point to logs.

echo "[DONE] All settings completed successfully."
echo "Tip: run a merge pass if you want a single CSV (after all runs finish)."
# Happy path: everything succeeded.
