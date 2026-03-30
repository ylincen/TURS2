#!/usr/bin/env python3
"""
test_exp_suite.py

Run all settings on a single UCI dataset (default: iris), then verify that
aggregate_results.csv contains all expected folds for that dataset.

Usage:
  python test_exp_suite.py               # runs on iris, 5 folds
  python test_exp_suite.py --dataset iris --splits 5
  python test_exp_suite.py --dry-run     # only prints the commands
  python test_exp_suite.py --check-only  # just verifies existing results (no running)

Notes:
- This script does NOT require changes to run_all.py, but if you added a
  --out-root flag, you can uncomment the lines that pass it for tidier output.
"""

import os
import sys
import time
import glob
import shlex
import argparse
import subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_all.py")

# ------- Settings to test (mirrors your exp_suite) -------
EXPERIMENTS = [
    {"name": "baseline",
     "args": ["--score", "normalized", "--not_use_excl"]},
    {"name": "ablation_aux",
     "args": ["--score", "normalized"]},
    {"name": "ablation_no_local",
     "args": ["--score", "normalized", "--not_use_excl", "--no-local-test"]},
    {"name": "ablation_no_patience",
     "args": ["--score", "normalized", "--not_use_excl", "--no-patience"]},
]

def runner_base_args(splits, seed):
    return [sys.executable, RUNNER, "--splits", str(splits), "--seed", str(seed)]

def find_new_result_dirs(before_set):
    """Return a list of NEW_exp_all_* dirs created since the 'before_set' snapshot."""
    now_dirs = set(glob.glob(os.path.join("results", "NEW_exp_all_*")))
    return sorted(now_dirs - before_set)

def pick_col(cols, candidates):
    """Pick a column name from 'cols' matching any of the lowercased candidates."""
    cl = [c.lower() for c in cols]
    for cand in candidates:
        if cand in cl:
            return cols[cl.index(cand)]
    return None

def verify_aggregate(root_dir, dataset, expected_splits):
    """
    Return (ok, msg, got_folds, missing_folds).
    ok=False + message if file missing / wrong columns / missing folds.
    """
    agg = os.path.join(root_dir, "aggregate_results.csv")
    if not os.path.isfile(agg):
        return (False, "aggregate_results.csv not found", [], list(range(expected_splits)))

    try:
        df = pd.read_csv(agg)
    except Exception as e:
        return (False, f"read error: {e}", [], list(range(expected_splits)))

    if df.empty:
        return (False, "aggregate is empty", [], list(range(expected_splits)))

    ds_col = pick_col(df.columns, ["dataset", "dataset_name", "data", "name", "ds"])
    fold_col = pick_col(df.columns, ["fold", "split", "cv", "fold_id", "split_id", "kfold", "run"])

    # Normalize dataset name for matching
    def norm(x): return str(x).strip().lower().replace(" ", "").replace("_", "-")
    ds_norm = norm(dataset)

    if ds_col is None:
        return (False, f"no dataset column in aggregate (cols={list(df.columns)})", [], list(range(expected_splits)))

    if fold_col is None:
        # If there's no explicit fold, assume it's not per-fold logging → fail
        return (False, "no fold/split column in aggregate", [], list(range(expected_splits)))

    # Filter to our dataset (robust to case/format)
    df["_ds_norm"] = df[ds_col].map(norm)
    sub = df[df["_ds_norm"] == ds_norm]
    if sub.empty:
        return (False, f"dataset '{dataset}' rows not found in aggregate", [], list(range(expected_splits)))

    try:
        folds = sorted(set(int(v) for v in sub[fold_col].tolist()))
    except Exception:
        return (False, f"fold column '{fold_col}' is not integer-like", [], list(range(expected_splits)))

    missing = [f for f in range(expected_splits) if f not in folds]
    ok = len(missing) == 0
    return (ok, "ok" if ok else "missing folds", folds, missing)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iris", help="UCI dataset name to test (default: iris)")
    ap.add_argument("--splits", type=int, default=5, help="Number of CV splits to expect")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="Only print commands; do not execute")
    ap.add_argument("--check-only", action="store_true", help="Do not run; only verify existing results")
    ap.add_argument("--suite", default="uci", choices=["uci","adbench"])
    ap.add_argument("--adbench-dir", default="../ADbench_datasets_Classical")

    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    report = []  # rows: {setting, root, ok, msg, got, missing}

    if not args.check_only:
        for exp in EXPERIMENTS:
            # Snapshot result dirs to identify the one created by this run
            before = set(glob.glob(os.path.join("results", "NEW_exp_all_*")))

            # Optional: if your run_all.py supports --out-root, uncomment block:
            # tag = f"test_{exp['name']}_{args.dataset}_{int(time.time())}"
            # out_root = os.path.join("results", f"NEW_exp_all_{tag}")
            # base = runner_base_args(args.splits, args.seed) + ["--suite","uci","--dataset", args.dataset, "--out-root", out_root] + exp["args"]

            # base = runner_base_args(args.splits, args.seed) + ["--suite","uci","--dataset", args.dataset] + exp["args"]
            base = runner_base_args(args.splits, args.seed) + ["--suite", args.suite]
            if args.suite == "adbench":
                base += ["--adbench_dir", args.adbench_dir, "--dataset", args.dataset]  # e.g., 34_smtp.npz
            else:
                base += ["--dataset", args.dataset]  # e.g., iris
            base += exp["args"]


            print(f"\n=== RUN setting={exp['name']} dataset={args.dataset} ===")
            print("CMD:", " ".join(shlex.quote(x) for x in base))
            if not args.dry_run:
                rc = subprocess.run(base).returncode
                if rc != 0:
                    print(f"[WARN] runner returned non-zero exit {rc} for setting '{exp['name']}'")
                # Find the new result dir(s) created by this call
                time.sleep(0.2)  # tiny delay to ensure mtime differs
                new_dirs = find_new_result_dirs(before)
                if not new_dirs:
                    # Fall back: pick the latest dir overall
                    candidates = sorted(glob.glob(os.path.join("results", "NEW_exp_all_*")), key=os.path.getmtime)
                    root = candidates[-1] if candidates else None
                    if root is None:
                        report.append({"setting": exp["name"], "root": "-", "ok": False,
                                       "msg": "no results dir found", "got": [], "missing": list(range(args.splits))})
                        continue
                else:
                    # If multiple appeared (rare), take the latest mtime
                    root = sorted(new_dirs, key=os.path.getmtime)[-1]

                ok, msg, got, missing = verify_aggregate(root, args.dataset, args.splits)
                report.append({"setting": exp["name"], "root": os.path.basename(root),
                               "ok": ok, "msg": msg, "got": got, "missing": missing})
    else:
        # check-only: verify the latest N= len(EXPERIMENTS) results
        candidates = sorted(glob.glob(os.path.join("results", "NEW_exp_all_*")), key=os.path.getmtime)[-len(EXPERIMENTS):]
        if not candidates:
            print("[ERROR] No NEW_exp_all_* dirs to check.")
            sys.exit(1)
        for exp, root in zip(EXPERIMENTS, candidates):
            ok, msg, got, missing = verify_aggregate(root, args.dataset, args.splits)
            report.append({"setting": exp["name"], "root": os.path.basename(root),
                           "ok": ok, "msg": msg, "got": got, "missing": missing})

    # --------- print summary ---------
    print("\n=== Summary (dataset: {}, splits: {}) ===".format(args.dataset, args.splits))
    width = max(len(r["setting"]) for r in report) if report else 8
    for r in report:
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['setting']:<{width}}  {status:4s}  root={r['root']}  msg={r['msg']}  got={r['got']}  missing={r['missing']}")

    # Exit non-zero if any failed
    if any(not r["ok"] for r in report):
        sys.exit(2)

if __name__ == "__main__":
    main()
