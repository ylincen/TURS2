TURS: Truly Unordered Probabilistic Rule Sets for Multi-class Classification
=============================================================================

Code for the paper:
  Yang, L. & van Leeuwen, M. "Truly Unordered Probabilistic Rule Sets for
  Multi-class Classification." ECMLPKDD 2022. Springer.


REQUIREMENTS
------------
Python 3.9+ with the packages listed in requirements.txt.

Install with conda (recommended, needed for numba):
  conda create -n turs python=3.9
  conda activate turs
  conda install -c conda-forge numba
  pip install -r requirements.txt


DATASETS
--------
UCI datasets (CSV format, one per file) should be placed in:
  ./datasets/

ADBench datasets (NPZ format) should be placed in:
  ../ADbench_datasets_Classical/    (or pass --adbench_dir to override)


RUNNING EXPERIMENTS
-------------------
The main entry point is turs2/run_all.py.

Run a single dataset (UCI):
  python turs2/run_all.py --suite uci --dataset iris --splits 5 --seed 2

Run a single dataset (ADBench):
  python turs2/run_all.py --suite adbench --dataset 26_optdigits.npz --splits 5 --seed 2

Run all datasets in parallel (edit EXPERIMENTS in run_all.sh first):
  bash turs2/run_all.sh

Key options:
  --out_dir DIR        Output directory for results (default: ./results)
  --score {normalized,absolute}  Scoring mode for beam search (default: normalized)
  --use_aux_beam       Enable the auxiliary (excl) beam (off by default)
  --validity_check {either,none}  Validity check mode (default: either)
  --beam_width N       Beam width (default: 10)
  --splits N           Number of CV folds (default: 5)
  --seed N             Random seed (default: 2)


CONTACT
-------
For questions or issues, please open a GitHub issue or contact:
  l.yang at liacs dot leidenuniv dot nl
