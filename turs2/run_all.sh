#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -u  # (no -e here so we don't die immediately on any background failure)

# --------- CONFIG ---------

# How many jobs to run in parallel (can be overridden from env)
MAX_JOBS="${MAX_JOBS:-32}"

# Python interpreter and path to your unified runner
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-run_all.py}"   # change if needed, e.g. ./run_all.py

# Base options for run_all.py
SPLITS="${SPLITS:-5}"
SEED="${SEED:-2}"

# UCI datasets
UCI_DATASETS=(
  Vehicle glass pendigits HeartCleveland chess iris
  backnote contracept ionosphere car tic-tac-toe wine
  diabetes anuran avila magic waveform DryBeans
)

# ADBench datasets (filenames, as in your original Python)
ADBENCH_DIR="${ADBENCH_DIR:-../ADbench_datasets_Classical}"
ADBENCH_DATASETS=(
  optdigits.npz smtp.npz pendigits.npz WDBC.npz
  satimage-2.npz backdoor.npz
  thyroid.npz Waveform.npz mammography.npz
  vowels.npz musk.npz ALOI.npz glass.npz
)

# Experiments: "name|args..."
# (name is only for logging, args are passed to run_all.py)
EXPERIMENTS=(
  "baseline|--score normalized"
)

# --------- JOB CONTROL ---------

running_jobs=0

wait_for_slot() {
  # Wait until running_jobs < MAX_JOBS
  while (( running_jobs >= MAX_JOBS )); do
    # wait -n: wait for *any* background job to finish (bash >= 4.3)
    if wait -n 2>/dev/null; then
      ((running_jobs--))
    else
      # No more jobs to wait for (shouldn't usually happen here)
      ((running_jobs--))
    fi
  done
}

submit_job() {
  local suite="$1"   # "uci" or "adbench"
  local dataset="$2"
  local exp_name="$3"
  local exp_args="$4"

  echo ">>> SUBMIT: suite=${suite} dataset=${dataset} exp=${exp_name}"

  # Assemble base command
  cmd=(
    "$PYTHON" "$RUNNER"
    --splits "$SPLITS"
    --seed "$SEED"
    --suite "$suite"
    --dataset "$dataset"
  )

  if [[ "$suite" == "adbench" ]]; then
    cmd+=(--adbench_dir "$ADBENCH_DIR")
  fi

  # Append experiment-specific args
  # shellcheck disable=SC2206
  exp_tokens=($exp_args)
  cmd+=("${exp_tokens[@]}")

  # Run in background
  "${cmd[@]}" &
  ((running_jobs++))
}

# --------- MAIN LOOPS ---------

echo "MAX_JOBS = $MAX_JOBS"
echo "PYTHON   = $PYTHON"
echo "RUNNER   = $RUNNER"
echo

# ---- UCI suite first ----
for ds in "${UCI_DATASETS[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    name="${exp%%|*}"   # part before first '|'
    args="${exp#*|}"    # part after first '|'
    wait_for_slot
    submit_job "uci" "$ds" "$name" "$args"
  done
done

# ---- ADBench suite next ----
for ds in "${ADBENCH_DATASETS[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    name="${exp%%|*}"
    args="${exp#*|}"
    wait_for_slot
    submit_job "adbench" "$ds" "$name" "$args"
  done
done

# Wait for all remaining jobs to finish
wait
echo "All jobs finished."
