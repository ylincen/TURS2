# (optional) point to your UCI CSVs if you want spurious later
export UCI_DATA_DIR=../datasets/

# safest threading to avoid surprise RAM spikes
export OMP_NUM_THREADS=1

# single dataset, baseline, tiny cut count, short timeout
python run_all.py \
  --suite uci --dataset iris \
  --splits 3 --seed 2 \
  --num_candidate_cuts 20 \
  --score normalized \
  --no-patience  # <- just to exercise the switch once (can drop)
