#!/usr/bin/env bash

# Parallel launcher for run_all_uncertainty_quantification_RF.py
# Uses the same dataset names as your original run_all.sh.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -u  # no undefined vars

# --------- CONFIG ---------

# How many jobs to run in parallel (can be overridden from env)
MAX_JOBS="${MAX_JOBS:-32}"

# Python interpreter and your RF UQ runner
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-run_all_uncertainty_quantification_RF.py}"

# All dataset names from the previous script
DATASETS=(
  # UCI datasets
  Vehicle glass pendigits HeartCleveland chess iris
  backnote contracept ionosphere car tic-tac-toe wine
  diabetes anuran avila magic waveform DryBeans

  # ADBench datasets
  26_optdigits.npz 34_smtp.npz 28_pendigits.npz 43_WDBC.npz
  36_speech.npz 31_satimage-2.npz 3_backdoor.npz
  38_thyroid.npz 41_Waveform.npz 23_mammography.npz
  40_vowels.npz 25_musk.npz 1_ALOI.npz 14_glass.npz
)

# If your RF script needs extra fixed args, set them here, e.g.:
# EXTRA_ARGS=(--splits 5 --seed 2)
EXTRA_ARGS=()

# --------- JOB CONTROL ---------

running_jobs=0

wait_for_slot() {
  while (( running_jobs >= MAX_JOBS )); do
    if wait -n 2>/dev/null; then
      ((running_jobs--))
    else
      ((running_jobs--))
    fi
  done
}

submit_job() {
  local dataset="$1"

  echo ">>> SUBMIT: dataset=${dataset}"

  cmd=(
    "$PYTHON" "$RUNNER"
    --dataset "$dataset"
    "${EXTRA_ARGS[@]}"
  )

  "${cmd[@]}" &
  ((running_jobs++))
}

# --------- MAIN LOOP ---------

echo "MAX_JOBS = $MAX_JOBS"
echo "PYTHON   = $PYTHON"
echo "RUNNER   = $RUNNER"
echo

for ds in "${DATASETS[@]}"; do
  wait_for_slot
  submit_job "$ds"
done

wait
echo "All RF UQ jobs finished."
