# audit_results.py
import os, glob, re, pandas as pd

# expected lists
UCI_SET = ["Vehicle","glass","pendigits","HeartCleveland","chess","iris","backnote",
           "contracept","ionosphere","car","tic-tac-toe","wine","diabetes",
           "anuran","avila","magic","waveform","DryBeans"]
EXPECTED_SPLITS = 5

rows = []
for root in sorted(glob.glob("NEW_exp_all_*")):
    agg = os.path.join(root, "aggregate_results.csv")
    if not os.path.isfile(agg):
        continue
    df = pd.read_csv(agg)
    # Try to infer dataset and fold column names you use; tweak if named differently
    # e.g., 'dataset' and 'fold' are common—adjust below if needed:
    ds_col = next((c for c in df.columns if c.lower() in ("dataset","data","name")), None)
    fold_col = next((c for c in df.columns if "fold" in c.lower()), None)
    if ds_col is None or fold_col is None:
        print(f"[WARN] {root}: couldn't find dataset/fold columns; columns={list(df.columns)}")
        continue
    for r in df[[ds_col, fold_col]].itertuples(index=False):
        rows.append((root, str(getattr(r, ds_col)), int(getattr(r, fold_col))))

# Build coverage table
import collections
cover = collections.defaultdict(set)
by_root = collections.defaultdict(list)
for root, ds, fold in rows:
    cover[ds].add(fold)
    by_root[root].append((ds, fold))

print("\n=== Missing by dataset ===")
for ds in UCI_SET:
    got = sorted(list(cover.get(ds, set())))
    missing = [f for f in range(EXPECTED_SPLITS) if f not in got]
    print(f"{ds:15s}  got={got}  missing={missing}")

print("\n=== Roots with few rows (sanity) ===")
for root, lst in sorted(by_root.items(), key=lambda kv: len(kv[1])):
    print(f"{len(lst):4d}  {root}")
