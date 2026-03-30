# TURS: Probabilistic Truly Unordered Rule Sets

Code for the paper:

> **Probabilistic Truly Unordered Rule Sets**
> Lincen Yang and Matthijs van Leeuwen
> LIACS, Leiden University
> *Journal of Machine Learning Research (under review)*

TURS learns **probabilistic rule sets** for multi-class classification. Unlike most
rule set and rule list methods, TURS does not impose any explicit or implicit order
among rules. Rule overlaps are assigned a principled probabilistic semantics that
encourages *overlap consistency*: overlapping rules tend to produce similar class
probability estimates, making each rule independently interpretable.

Learning is formulated as MDL-based model selection, with a heuristic search
algorithm featuring a learning speed score, a diverse-patience beam search, and an
MDL-based local test.

---

## 1. Installation

We recommend conda (required for numba on macOS):

```bash
conda create -n turs python=3.9
conda activate turs
conda install -c conda-forge numba
pip install -r requirements.txt
```

**Optional — compile `nml_regret.py` with Cython for ~2–5× faster NML regret
computation** (requires a C compiler):

```bash
pip install cython
cd turs2
cythonize -i nml_regret.py
cd ..
```

Without compilation the module runs as plain Python automatically.

---

## 2. Repository Structure

```
turs2/
  run_all.py               ← main entry point
  run_all.sh               ← parallel launcher (all datasets)
  DataInfo.py              ← dataset metadata and candidate cuts
  Ruleset.py               ← rule set model: fit(), add_rule(), search
  Rule.py                  ← single rule: grow(), MDL gain
  RuleGrowConstraint.py    ← MDL-based local test, validity check
  Beam.py / ModellingGroup.py
  DataEncoding.py          ← NML data encoding
  ModelEncoding.py         ← MDL model encoding
  nml_regret.py            ← NML regret (Cython-compilable pure Python)
  exp_predictive_perf.py   ← evaluation: ROC-AUC, overlap analysis
  exp_utils.py             ← cover matrix, random-picking, overlap metrics
datasets/                  ← UCI CSV files (not included, see Section 5)
tests/
  test_smoke.py            ← minimal pytest smoke tests
requirements.txt
```

---

## 3. Running Experiments

### Single dataset (UCI)

```bash
cd turs2
python run_all.py \
    --suite uci \
    --dataset iris \
    --splits 5 \
    --seed 2 \
    --out_dir ../results/
```

### Single dataset (ADBench)

```bash
python run_all.py \
    --suite adbench \
    --dataset 26_optdigits.npz \
    --splits 5 \
    --seed 2 \
    --adbench_dir ../ADbench_datasets_Classical/ \
    --out_dir ../results/
```

### All datasets in parallel

Edit `EXPERIMENTS` in `run_all.sh` if needed, then:

```bash
bash run_all.sh
```

`MAX_JOBS` (default 32) controls parallelism. Example to limit to 4 parallel jobs:

```bash
MAX_JOBS=4 bash run_all.sh
```

### Reproducing the paper's main results (Table 3–6)

The paper uses 5-fold stratified cross-validation with the following settings
(all are already the defaults):

```bash
python run_all.py \
    --suite uci \
    --dataset iris \
    --splits 5 \
    --seed 2 \
    --beam_width 10 \
    --num_candidate_cuts 20 \
    --validity_check either \
    --score normalized \
    --out_dir ../results/
```

- `--validity_check either` enables the MDL-based local test (Step 4 in Algorithm 2).
- `--score normalized` uses the learning speed score (Eq. 21).
- Diverse-patience beam search is on by default (`--no-patience` disables it).

---

## 4. Key CLI Options

| Option | Default | Description |
|---|---|---|
| `--suite` | `uci` | Dataset suite: `uci` or `adbench` |
| `--dataset` | `tic-tac-toe` | Dataset name or `.npz` filename |
| `--splits` | `5` | Number of CV folds |
| `--seed` | `2` | Random seed for CV split |
| `--beam_width` | `10` | Beam width *W* (paper: 10) |
| `--num_candidate_cuts` | `20` | Candidate cut points per feature (paper: 20) |
| `--validity_check` | `either` | `either`: MDL local test on; `none`: off |
| `--score` | `normalized` | `normalized`: learning speed score; `absolute`: raw MDL gain |
| `--no-patience` | off | Disable diverse-patience beam search |
| `--use_aux_beam` | off | Enable auxiliary (excl) beam — ablation only |
| `--out_dir` | `RUN_ALL_...` | Output directory |

---

## 5. Datasets

### UCI datasets

Download from the [UCI Machine Learning Repository](https://archive.ics.uci.edu)
and place as CSV files in `datasets/`. The expected filenames are:

| Filename | Header? |
|---|---|
| `chess.csv`, `iris.csv`, `waveform.csv`, `backnote.csv`, `contracept.csv`, `ionosphere.csv`, `magic.csv`, `car.csv`, `tic-tac-toe.csv`, `wine.csv`, `glass.csv`, `pendigits.csv`, `HeartCleveland.csv` | No header |
| `avila.csv`, `anuran.csv`, `diabetes.csv`, `Vehicle.csv`, `DryBeans.csv` | Has header |

All features must be numeric or binary (one-hot-encode categoricals before
placing in `datasets/`). The last column must be the class label.

### ADBench datasets

Download the `.npz` files from the
[ADBench GitHub repository](https://github.com/Minqi824/ADBench) and place them in
`../ADbench_datasets_Classical/` (relative to `turs2/`), or pass `--adbench_dir`
to override.

---

## 6. Output Files and Key Metrics

Each run produces output in `<out_dir>/<setting_tag>/`:

```
<out_dir>/
  <setting_tag>/
    iris.csv               ← per-fold results for iris
    iris_overlap.csv       ← per-instance overlap data (default setting only)
    aggregate_results.csv  ← all datasets and folds combined
```

### Columns in `aggregate_results.csv`

The table below maps each CSV column to the corresponding paper result.

#### Predictive performance (Table 3)

| Column | Description |
|---|---|
| `roc_auc_test` | ROC-AUC on test fold. For multi-class, macro one-vs-rest. Average over folds → **Table 3**. |

#### Model complexity (Tables 4–6)

| Column | Description |
|---|---|
| `num_rules` | Number of rules in the learned model → **Table 5** (avg over folds). |
| `avg_rule_length` | Average number of literals per rule → **Table 6** (avg over folds). |
| `num_rules × avg_rule_length` | Total number of literals → **Table 4** (compute from the two columns above). |

#### Runtime

| Column | Description |
|---|---|
| `runtime` | Wall-clock training time in seconds for one fold. |

#### Reliability of probabilistic outputs (Section 6.3.2, Figure 2)

| Column | Description |
|---|---|
| `train_test_prob_diff` | Coverage-weighted average of the per-rule mean absolute difference between class probability estimates on training vs. test instances (Eq. 24 in paper). **Lower is better.** This is the g-score whose empirical CDF is plotted in Figure 2. |
| `mean_rule_probs_TrainTestDiff` | Unweighted version of the above. |
| `weighted_mean_rule_probs_TrainTestDiff` | Coverage-weighted version (same as `train_test_prob_diff` but for the predicted class only). |

#### Overlap consistency (Section 6.4, Figure 3)

| Column | Description |
|---|---|
| `overlap_perc` | Fraction of test instances covered by more than one rule. |
| `overlap_prob_diffs_mean` | Mean absolute difference between class probabilities of co-covering rules, averaged over all unique overlap groups (training data). |
| `overlap_prob_diffs_max` | Max absolute difference between class probabilities of co-covering rules (training data). **Lower = higher overlap consistency.** |
| `random_picking_roc_auc` | ROC-AUC when predicting by randomly selecting one of the applicable covering rules (averaged over 10 seeds). Close to `roc_auc_test` → high overlap consistency. |

The `<dataset>_overlap.csv` file (written for the default setting only) stores
per-instance individual rule probability vectors for all test instances, enabling
reconstruction of Figure 3 (pairwise ℓ1-distances between overlapping rules'
probability estimates).

---

## 7. Extracting Results in Python

```python
import pandas as pd
import numpy as np

df = pd.read_csv("results/<setting_tag>/aggregate_results.csv")

summary = df.groupby("data_name").agg(
    roc_auc        = ("roc_auc_test",             "mean"),
    num_rules      = ("num_rules",                "mean"),
    avg_rule_len   = ("avg_rule_length",          "mean"),
    total_literals = ("num_rules",                lambda x: (x * df.loc[x.index, "avg_rule_length"]).mean()),
    runtime_s      = ("runtime",                  "sum"),   # total over all folds
    prob_diff      = ("train_test_prob_diff",      "mean"),  # g-score (Figure 2)
    overlap_perc   = ("overlap_perc",             "mean"),
    overlap_cons   = ("overlap_prob_diffs_mean",  "mean"),  # overlap consistency
    rand_roc_auc   = ("random_picking_roc_auc",   "mean"),  # random-pick test
).round(3)

print(summary.to_string())
```

---

## 8. Running the Smoke Tests

```bash
pytest tests/ -v
```

Five tests run on the iris dataset (~5 seconds): fit, predict shape, ROC-AUC > 0.5,
Brier score in [0, 1], and NML regret sanity.

---

<!-- ## 9. Citation

```bibtex
@article{yang2024turs,
  title   = {Probabilistic Truly Unordered Rule Sets},
  author  = {Yang, Lincen and van Leeuwen, Matthijs},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  note    = {Under review}
}
``` -->

---

## Contact

For questions or issues, please open a GitHub issue or contact:
`l.yang at liacs dot leidenuniv dot nl`
