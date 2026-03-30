# turs2/agg_writer.py
from __future__ import annotations
import os, time, json
import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, Tuple

# Keys we usually DO NOT want in CSV (forced to sidecar if present)
DEFAULT_HEAVY_KEYS = {
    "rules_prob_test", "rules_prob_train",
    "cover_matrix_test", "cover_matrix_train",
    "overlap_analysis_full_res",
    "covered_rule_p_per_instance",
    "overlap_count_per_instance", "rule_prob_spread", "rule_prob_var",
    # legacy stringified blobs -> drop from CSV
    "cover_matrix_test_str", "cover_matrix_train_str",
}

def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool, np.number, np.bool_))

def _should_sidecar(k: str, v: Any, heavy_keys: set) -> bool:
    if k in heavy_keys: return True
    # Heuristic: arrays/lists/dicts are heavy
    if isinstance(v, (np.ndarray, list, dict, tuple)): return True
    return False

def make_aggregate_row(
    exp_res: Dict[str, Any],
    out_root: str,
    suite: str,
    dataset: str,
    fold: int | None,
    tag: str | None = None,
    heavy_keys: Iterable[str] = (),
) -> Tuple[Dict[str, Any], str | None]:
    """
    Returns (row_dict, sidecar_path). row_dict has only scalars + a 'sidecar' column.
    Everything non-scalar (or in heavy_keys) is saved to a compressed .npz sidecar.
    """
    heavy_keys = set(DEFAULT_HEAVY_KEYS) | set(heavy_keys)

    # Build sidecar path
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = tag or "exp"
    side_dir = os.path.join(out_root, "sidecars", suite, dataset)
    os.makedirs(side_dir, exist_ok=True)
    fname = f"{dataset}_fold{fold if fold is not None else 'NA'}_{tag}_{ts}.npz"
    sidecar_path = os.path.join(side_dir, fname)

    row: Dict[str, Any] = {}
    side_payload: Dict[str, Any] = {}

    # Always stamp basic identifiers
    row["suite"] = suite
    row["dataset"] = dataset
    row["fold"] = fold

    for k, v in exp_res.items():
        # Skip legacy huge string dumps
        if k.endswith("_str"): 
            continue
        if _should_sidecar(k, v, heavy_keys):
            side_payload[k] = v
        else:
            # Keep only scalars in CSV; stringify tiny dicts if any slip through
            if _is_scalar(v):
                row[k] = v
            elif isinstance(v, dict) and len(v) <= 5:
                row[k] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
            else:
                side_payload[k] = v

    # Write sidecar only if needed
    if side_payload:
        # np.savez_compressed needs arrays; for objects, allow_pickle=True via dtype=object wrapper
        np.savez_compressed(sidecar_path, **{
            key: (np.asarray(val, dtype=object) if not isinstance(val, np.ndarray) else val)
            for key, val in side_payload.items()
        })
        row["sidecar"] = sidecar_path
    else:
        sidecar_path = None
        row["sidecar"] = ""

    return row, sidecar_path


def write_aggregate_csv(rows: Iterable[Dict[str, Any]], out_csv: str) -> None:
    """
    Writes rows to CSV with a stable, readable column order: identifiers, metrics, runtime,
    complexity, overlap summaries, flags, then the sidecar path.
    Unknown keys are appended alphabetically at the end.
    """
    rows = list(rows)
    if not rows:
        # Create an empty file with just a header
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    # Preferred column order (will keep those that exist)
    preferred = [
        # identifiers
        "suite","dataset","fold","data_name","fold_index",
        # performance
        "roc_auc_test","pr_auc_test","logloss_test","Brier_test","accuracy_test",
        "roc_auc_train","pr_auc_train","logloss_train","Brier_train","accuracy_train",
        # complexity
        "num_rules","avg_rule_length","nrow","ncol","avg_num_literals_for_each_datapoint",
        # overlap summaries
        "overlap_perc","train_test_prob_diff",
        "overlap_prob_diffs_mean","overlap_prob_diffs_max","modelling_group_coverage",
        "mean_overlap","median_overlap","mean_rule_prob_spread","median_rule_prob_spread",
        "mean_rule_prob_var","median_rule_prob_var",
        # runtime/config
        "runtime","score_type","use_patience","use_local_test","use_aux_beam",
        "validity_check_","not_use_excl_",
        # sidecar pointer
        "sidecar",
    ]

    df = pd.DataFrame(rows)

    # Build final column order
    existing_pref = [c for c in preferred if c in df.columns]
    remaining = sorted([c for c in df.columns if c not in existing_pref])
    cols = existing_pref + remaining

    df = df[cols]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
