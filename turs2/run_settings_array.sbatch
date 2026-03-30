#!/bin/bash
#SBATCH -J TURS_revision
#SBATCH -p cpu-zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t 5-00:00:00
#SBATCH -o slurm_out/%x_%A_%a.out
#SBATCH -e slurm_out/%x_%A_%a.err
set -euo pipefail

mkdir -p slurm_out

CPUS="${SLURM_CPUS_PER_TASK:-1}"   # default to 1 if not set
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"   # if you use numexpr

# Activate environment if needed
# module load anaconda || true
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate yourenv

IDX="${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Running setting index ${IDX}"
python exp_suite.py --setting-idx "${IDX}"
echo "[$(date)] Done setting index ${IDX}"
