#!/usr/bin/env bash
set -euo pipefail

# Max concurrent jobs; override with env var: MAX_JOBS=16 ./run_all_experiments.sh
MAX_JOBS="${MAX_JOBS:-32}"
LOG_DIR="${LOG_DIR:-RIVIUM_OUT_$(date +%Y%m%d)}"
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")"

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -r -p | wc -l)
    if (( running < MAX_JOBS )); then
      break
    fi
    sleep 0.2
  done
}

PY="${PY:-python}"

wait_for_slot
{
  echo "[$(date)] START uci_Vehicle_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_Vehicle_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/001_uci_Vehicle_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_glass_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_glass_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/002_uci_glass_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_pendigits_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_pendigits_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/003_uci_pendigits_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_HeartCleveland_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_HeartCleveland_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/004_uci_HeartCleveland_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_chess_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_chess_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/005_uci_chess_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_iris_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_iris_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/006_uci_iris_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_backnote_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_backnote_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/007_uci_backnote_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_contracept_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_contracept_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/008_uci_contracept_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_ionosphere_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_ionosphere_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/009_uci_ionosphere_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_car_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "car" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "car" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_car_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/010_uci_car_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_tic-tac-toe_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_tic-tac-toe_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/011_uci_tic-tac-toe_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_wine_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_wine_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/012_uci_wine_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_diabetes_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_diabetes_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/013_uci_diabetes_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_anuran_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_anuran_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/014_uci_anuran_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_avila_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_avila_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/015_uci_avila_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_magic_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_magic_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/016_uci_magic_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_waveform_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_waveform_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/017_uci_waveform_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_DryBeans_baseline"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END uci_DryBeans_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/018_uci_DryBeans_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_Vehicle_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized
  rc=$?
  echo "[$(date)] END uci_Vehicle_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/019_uci_Vehicle_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_glass_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized
  rc=$?
  echo "[$(date)] END uci_glass_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/020_uci_glass_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_pendigits_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized
  rc=$?
  echo "[$(date)] END uci_pendigits_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/021_uci_pendigits_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_HeartCleveland_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized
  rc=$?
  echo "[$(date)] END uci_HeartCleveland_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/022_uci_HeartCleveland_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_chess_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized
  rc=$?
  echo "[$(date)] END uci_chess_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/023_uci_chess_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_iris_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized
  rc=$?
  echo "[$(date)] END uci_iris_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/024_uci_iris_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_backnote_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized
  rc=$?
  echo "[$(date)] END uci_backnote_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/025_uci_backnote_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_contracept_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized
  rc=$?
  echo "[$(date)] END uci_contracept_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/026_uci_contracept_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_ionosphere_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized
  rc=$?
  echo "[$(date)] END uci_ionosphere_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/027_uci_ionosphere_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_car_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "car" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "car" --score normalized
  rc=$?
  echo "[$(date)] END uci_car_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/028_uci_car_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_tic-tac-toe_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized
  rc=$?
  echo "[$(date)] END uci_tic-tac-toe_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/029_uci_tic-tac-toe_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_wine_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized
  rc=$?
  echo "[$(date)] END uci_wine_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/030_uci_wine_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_diabetes_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized
  rc=$?
  echo "[$(date)] END uci_diabetes_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/031_uci_diabetes_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_anuran_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized
  rc=$?
  echo "[$(date)] END uci_anuran_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/032_uci_anuran_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_avila_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized
  rc=$?
  echo "[$(date)] END uci_avila_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/033_uci_avila_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_magic_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized
  rc=$?
  echo "[$(date)] END uci_magic_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/034_uci_magic_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_waveform_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized
  rc=$?
  echo "[$(date)] END uci_waveform_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/035_uci_waveform_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_DryBeans_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized"
  "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized
  rc=$?
  echo "[$(date)] END uci_DryBeans_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/036_uci_DryBeans_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_Vehicle_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_Vehicle_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/037_uci_Vehicle_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_glass_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_glass_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/038_uci_glass_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_pendigits_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_pendigits_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/039_uci_pendigits_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_HeartCleveland_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_HeartCleveland_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/040_uci_HeartCleveland_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_chess_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_chess_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/041_uci_chess_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_iris_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_iris_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/042_uci_iris_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_backnote_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_backnote_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/043_uci_backnote_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_contracept_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_contracept_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/044_uci_contracept_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_ionosphere_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_ionosphere_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/045_uci_ionosphere_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_car_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "car" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "car" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_car_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/046_uci_car_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_tic-tac-toe_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_tic-tac-toe_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/047_uci_tic-tac-toe_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_wine_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_wine_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/048_uci_wine_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_diabetes_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_diabetes_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/049_uci_diabetes_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_anuran_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_anuran_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/050_uci_anuran_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_avila_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_avila_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/051_uci_avila_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_magic_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_magic_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/052_uci_magic_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_waveform_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_waveform_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/053_uci_waveform_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_DryBeans_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END uci_DryBeans_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/054_uci_DryBeans_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_Vehicle_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "Vehicle" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_Vehicle_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/055_uci_Vehicle_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_glass_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "glass" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_glass_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/056_uci_glass_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_pendigits_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "pendigits" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_pendigits_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/057_uci_pendigits_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_HeartCleveland_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "HeartCleveland" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_HeartCleveland_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/058_uci_HeartCleveland_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_chess_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "chess" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_chess_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/059_uci_chess_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_iris_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "iris" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_iris_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/060_uci_iris_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_backnote_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "backnote" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_backnote_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/061_uci_backnote_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_contracept_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "contracept" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_contracept_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/062_uci_contracept_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_ionosphere_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "ionosphere" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_ionosphere_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/063_uci_ionosphere_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_car_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "car" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "car" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_car_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/064_uci_car_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_tic-tac-toe_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "tic-tac-toe" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_tic-tac-toe_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/065_uci_tic-tac-toe_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_wine_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "wine" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_wine_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/066_uci_wine_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_diabetes_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "diabetes" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_diabetes_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/067_uci_diabetes_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_anuran_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "anuran" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_anuran_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/068_uci_anuran_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_avila_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "avila" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_avila_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/069_uci_avila_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_magic_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "magic" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_magic_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/070_uci_magic_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_waveform_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "waveform" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_waveform_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/071_uci_waveform_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START uci_DryBeans_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite uci --dataset "DryBeans" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END uci_DryBeans_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/072_uci_DryBeans_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_26_optdigits_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_26_optdigits_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/073_adbench_26_optdigits_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_34_smtp_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_34_smtp_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/074_adbench_34_smtp_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_28_pendigits_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_28_pendigits_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/075_adbench_28_pendigits_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_43_WDBC_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_43_WDBC_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/076_adbench_43_WDBC_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_36_speech_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_36_speech_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/077_adbench_36_speech_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_31_satimage-2_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_31_satimage-2_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/078_adbench_31_satimage-2_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_3_backdoor_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_3_backdoor_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/079_adbench_3_backdoor_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_38_thyroid_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_38_thyroid_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/080_adbench_38_thyroid_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_41_Waveform_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_41_Waveform_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/081_adbench_41_Waveform_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_23_mammography_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_23_mammography_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/082_adbench_23_mammography_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_40_vowels_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_40_vowels_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/083_adbench_40_vowels_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_25_musk_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_25_musk_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/084_adbench_25_musk_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_1_ALOI_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_1_ALOI_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/085_adbench_1_ALOI_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_14_glass_npz_baseline"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized --not_use_excl"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized --not_use_excl
  rc=$?
  echo "[$(date)] END adbench_14_glass_npz_baseline (rc=$rc)"
  exit $rc
} >"$LOG_DIR/086_adbench_14_glass_npz_baseline.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_26_optdigits_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_26_optdigits_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/087_adbench_26_optdigits_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_34_smtp_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_34_smtp_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/088_adbench_34_smtp_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_28_pendigits_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_28_pendigits_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/089_adbench_28_pendigits_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_43_WDBC_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_43_WDBC_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/090_adbench_43_WDBC_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_36_speech_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_36_speech_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/091_adbench_36_speech_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_31_satimage-2_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_31_satimage-2_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/092_adbench_31_satimage-2_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_3_backdoor_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_3_backdoor_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/093_adbench_3_backdoor_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_38_thyroid_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_38_thyroid_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/094_adbench_38_thyroid_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_41_Waveform_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_41_Waveform_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/095_adbench_41_Waveform_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_23_mammography_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_23_mammography_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/096_adbench_23_mammography_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_40_vowels_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_40_vowels_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/097_adbench_40_vowels_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_25_musk_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_25_musk_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/098_adbench_25_musk_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_1_ALOI_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_1_ALOI_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/099_adbench_1_ALOI_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_14_glass_npz_ablation_aux"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized
  rc=$?
  echo "[$(date)] END adbench_14_glass_npz_ablation_aux (rc=$rc)"
  exit $rc
} >"$LOG_DIR/100_adbench_14_glass_npz_ablation_aux.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_26_optdigits_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_26_optdigits_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/101_adbench_26_optdigits_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_34_smtp_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_34_smtp_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/102_adbench_34_smtp_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_28_pendigits_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_28_pendigits_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/103_adbench_28_pendigits_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_43_WDBC_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_43_WDBC_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/104_adbench_43_WDBC_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_36_speech_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_36_speech_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/105_adbench_36_speech_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_31_satimage-2_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_31_satimage-2_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/106_adbench_31_satimage-2_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_3_backdoor_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_3_backdoor_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/107_adbench_3_backdoor_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_38_thyroid_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_38_thyroid_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/108_adbench_38_thyroid_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_41_Waveform_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_41_Waveform_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/109_adbench_41_Waveform_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_23_mammography_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_23_mammography_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/110_adbench_23_mammography_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_40_vowels_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_40_vowels_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/111_adbench_40_vowels_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_25_musk_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_25_musk_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/112_adbench_25_musk_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_1_ALOI_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_1_ALOI_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/113_adbench_1_ALOI_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_14_glass_npz_ablation_no_local"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized --not_use_excl --validity_check none"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized --not_use_excl --validity_check none
  rc=$?
  echo "[$(date)] END adbench_14_glass_npz_ablation_no_local (rc=$rc)"
  exit $rc
} >"$LOG_DIR/114_adbench_14_glass_npz_ablation_no_local.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_26_optdigits_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "26_optdigits.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_26_optdigits_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/115_adbench_26_optdigits_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_34_smtp_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "34_smtp.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_34_smtp_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/116_adbench_34_smtp_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_28_pendigits_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "28_pendigits.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_28_pendigits_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/117_adbench_28_pendigits_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_43_WDBC_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "43_WDBC.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_43_WDBC_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/118_adbench_43_WDBC_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_36_speech_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "36_speech.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_36_speech_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/119_adbench_36_speech_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_31_satimage-2_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "31_satimage-2.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_31_satimage-2_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/120_adbench_31_satimage-2_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_3_backdoor_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "3_backdoor.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_3_backdoor_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/121_adbench_3_backdoor_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_38_thyroid_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "38_thyroid.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_38_thyroid_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/122_adbench_38_thyroid_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_41_Waveform_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "41_Waveform.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_41_Waveform_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/123_adbench_41_Waveform_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_23_mammography_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "23_mammography.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_23_mammography_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/124_adbench_23_mammography_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_40_vowels_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "40_vowels.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_40_vowels_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/125_adbench_40_vowels_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_25_musk_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "25_musk.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_25_musk_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/126_adbench_25_musk_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_1_ALOI_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "1_ALOI.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_1_ALOI_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/127_adbench_1_ALOI_npz_ablation_no_patience.log" 2>&1 &

wait_for_slot
{
  echo "[$(date)] START adbench_14_glass_npz_ablation_no_patience"
  echo "CMD: "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized --not_use_excl --no-patience"
  "$PY" "run_all.py" --suite adbench --adbench_dir "../ADbench_datasets_Classical" --dataset "14_glass.npz" --score normalized --not_use_excl --no-patience
  rc=$?
  echo "[$(date)] END adbench_14_glass_npz_ablation_no_patience (rc=$rc)"
  exit $rc
} >"$LOG_DIR/128_adbench_14_glass_npz_ablation_no_patience.log" 2>&1 &

echo "[INFO] Launched ${#commands[@]:-N} jobs (Bash can’t count here, but logs are in $LOG_DIR)"
wait
echo "[DONE] All background jobs finished (check logs in $LOG_DIR)"