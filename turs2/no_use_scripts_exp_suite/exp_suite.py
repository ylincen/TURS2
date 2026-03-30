#!/usr/bin/env python3
import os, sys, subprocess, shutil, glob, json
from datetime import datetime
import numpy as np
import pandas as pd


# Path to your unified runner
RUNNER = os.path.join(os.path.dirname(__file__), "run_all.py")

# Default dataset sets (tweak as you like)
UCI_SET = ["Vehicle", "glass", "pendigits", "HeartCleveland", "chess", "iris", "backnote", "contracept", 
           "ionosphere", "car", "tic-tac-toe", "wine", "diabetes", 
           "anuran", "avila", "magic", "waveform", "DryBeans"]
ADBENCH_DIR = "../ADbench_datasets_Classical"
ADBENCH_SET = ["26_optdigits.npz", "34_smtp.npz", "28_pendigits.npz", "43_WDBC.npz", "36_speech.npz", "31_satimage-2.npz", 
               "3_backdoor.npz", "38_thyroid.npz", "41_Waveform.npz", "23_mammography.npz", "40_vowels.npz", "25_musk.npz", 
               "1_ALOI.npz", "14_glass.npz"]

# add near the top
import argparse

def build_experiments():
    exp1 = {
        "name": "baseline",
        "args": ["--score", "normalized", "--not_use_excl"],
    }
    exp2_aux = {
        "name": "ablation_aux",
        "args": ["--score", "normalized"]
    }
    exp2_nolocal = {
        "name": "ablation_no_local",
        "args": ["--score", "normalized", "--not_use_excl", "--validity_check", "none"]
    }
    exp2_nopatience = {
        "name": "ablation_no_patience",
        "args": ["--score", "normalized", "--not_use_excl", "--no-patience"]
    }
    return [exp1, exp2_aux, exp2_nolocal, exp2_nopatience]

def make_settings(OUT_BASE):
    """Return an ordered list of (key, func) where func runs that single setting."""
    experiments = build_experiments()

    def run_cmd(cmd):
        print(">>>", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[WARN] command failed (rc={res.returncode}): {' '.join(cmd)}")
            return False
        return True


    settings = []

    # ---- UCI settings (each experiment over all UCI datasets) ----
    for exp in experiments:
        key = f"uci:{exp['name']}"
        def _f(exp=exp):
            failed = []
            cmd_base = runner_base_args() + ["--suite", "uci"] + exp["args"]
            for name in UCI_SET:
                ok = run_cmd(cmd_base + ["--dataset", name])
                if not ok:
                    failed.append(name)
            # optional: write a small manifest next to the results
            if failed:
                print(f"[SETTING WARN] Failed datasets: {failed}")

        settings.append((key, _f))

    # ---- ADBench settings (each experiment over all ADBench datasets) ----
    for exp in experiments:
        key = f"adbench:{exp['name']}"
        def _f(exp=exp):
            cmd_base = (
                runner_base_args()
                + ["--suite", "adbench", "--adbench_dir", ADBENCH_DIR]
                + exp["args"]
            )
            failed = []
            for name in ADBENCH_SET:
                ok = run_cmd(cmd_base + ["--dataset", name])
                if not ok:
                    failed.append(name)
            if failed:
                print(f"[SETTING WARN] ADBench {exp['name']} failed datasets: {failed}")
        settings.append((key, _f))

    # ---- Spurious (two settings) ----
    SPUR_DIR = os.path.join(OUT_BASE, "spurious_datasets")
    os.makedirs(SPUR_DIR, exist_ok=True)
    spurious_pairs = []
    for name in ADBENCH_SET:
        src = os.path.join(ADBENCH_DIR, name)
        injected = os.path.join(SPUR_DIR, name.replace(".npz", "_spurious_indep5.npz"))
        inject_spurious_npz(src, injected, k=5, mode="indep", strength=0.0, seed=123)
        spurious_pairs.append((name, injected))

    def spurious_local_on():
        cmd = runner_base_args() + ["--suite", "adbench", "--adbench_dir", SPUR_DIR, "--score", "normalized", "--not_use_excl"]
        for _, injected in spurious_pairs:
            run_cmd(cmd + ["--dataset", os.path.basename(injected)])

    def spurious_local_off():
        cmd = runner_base_args() + ["--suite", "adbench", "--adbench_dir", SPUR_DIR, "--score", "normalized", "--no-local-test", "--not_use_excl"]
        for _, injected in spurious_pairs:
            run_cmd(cmd + ["--dataset", os.path.basename(injected)])

    settings.append(("spurious:local_on", spurious_local_on))
    settings.append(("spurious:local_off", spurious_local_off))

    return settings

# ---------- Spurious feature injector (ADBench NPZ only) ----------
def inject_spurious_npz(in_npz_path, out_npz_path, k=5, mode="indep", strength=0.1, seed=42):
    """
    mode:
      - 'indep': Bernoulli(0.5) features independent of (X,Y)
      - 'weak':  Bernoulli(0.5 + strength*(Y-0.5)) weakly correlated with Y
    """
    rng = np.random.default_rng(seed)
    data = np.load(in_npz_path)
    X, y = data["X"], data["y"]
    n = X.shape[0]

    if mode == "indep":
        Z = rng.integers(0, 2, size=(n, k))
    elif mode == "weak":
        if y.ndim > 1:
            yb = y.reshape(-1)
        else:
            yb = y
        p = np.clip(0.5 + strength * (yb - 0.5), 0.001, 0.999)
        Z = (rng.random((n, k)) < p[:, None]).astype(int)
    else:
        raise ValueError("mode must be 'indep' or 'weak'")

    X_aug = np.concatenate([X, Z], axis=1)
    np.savez_compressed(out_npz_path, X=X_aug, y=y)

# ---------- Helpers ----------
def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def runner_base_args():
    return [sys.executable, RUNNER, "--splits", "5", "--seed", "2"]

def merge_results(output_roots, merged_csv_path):
    dfs = []
    for root in output_roots:
        agg = os.path.join(root, "aggregate_results.csv")
        if os.path.isfile(agg):
            df = pd.read_csv(agg)
            df["exp_root"] = os.path.basename(root)
            dfs.append(df)
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)
        big.to_csv(merged_csv_path, index=False)
        print("[MERGED]", merged_csv_path)
    else:
        print("[WARN] No aggregate_results.csv found to merge.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-idx", type=int, default=None,
                        help="Run only the Nth setting (1-based index from --list-settings).")
    parser.add_argument("--list-settings", action="store_true",
                        help="List available settings and exit.")
    args = parser.parse_args()

    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    day = datetime.now().strftime("%Y%m%d")
    OUT_BASE = os.path.abspath(os.path.join("results", f"suite_{day}"))
    os.makedirs(OUT_BASE, exist_ok=True)

    settings = make_settings(OUT_BASE)

    # Just list and exit
    if args.list_settings:
        for i, (k, _) in enumerate(settings, 1):
            print(f"{i}\t{k}")
        return

    # Run exactly one setting (datasets serial)
    if args.setting_idx is not None:
        i = args.setting_idx
        if i < 1 or i > len(settings):
            print(f"[ERROR] --setting-idx out of range (1..{len(settings)}). Use --list-settings to see options.")
            return
        key, fn = settings[i - 1]
        print(f"\n=== Running setting [{i}] {key} ===")
        fn()
        return

    # ----------------------------
    # Original behavior: run all settings serially and then merge
    # ----------------------------
    # 1) Baseline: no-aux + local-test ON + beam diversity (patience) ON
    exp1 = {
        "name": "baseline",
        "args": ["--score", "normalized", "--not_use_excl"],
    }

    # 2) Ablations (one-by-one)
    exp2_aux = {"name": "ablation_aux", "args": ["--score", "normalized"]}
    exp2_nolocal = {"name": "ablation_no_local",
                    "args": ["--score", "normalized", "--not_use_excl", "--no-local-test"]}
    exp2_nopatience = {"name": "ablation_no_patience",
                       "args": ["--score", "normalized", "--not_use_excl", "--no-patience"]}

    experiments = [exp1, exp2_aux, exp2_nolocal, exp2_nopatience]
    out_roots = []

    # ---- Run each experiment on UCI ----
    for exp in experiments:
        tag = f"{exp['name']}_uci"
        print(f"\n=== Running {tag} ===")
        cmd = runner_base_args() + ["--suite", "uci"] + exp["args"]
        for name in UCI_SET:
            run(cmd + ["--dataset", name])
    out_roots += sorted(glob.glob(os.path.join("results", "NEW_exp_all_*")),
                        key=os.path.getmtime)[-len(experiments):]

    # ---- Run each experiment on ADBench ----
    for exp in experiments:
        tag = f"{exp['name']}_adbench"
        print(f"\n=== Running {tag} ===")
        cmd = runner_base_args() + ["--suite", "adbench", "--adbench_dir", ADBENCH_DIR, "--not_use_excl"] + exp["args"]
        for name in ADBENCH_SET:
            run(cmd + ["--dataset", name])
    out_roots += sorted(glob.glob(os.path.join("results", "NEW_exp_all_*")),
                        key=os.path.getmtime)[-len(experiments):]

    # 3) Spurious-feature tests (ADBench only)
    SPUR_DIR = os.path.join(OUT_BASE, "spurious_datasets")
    os.makedirs(SPUR_DIR, exist_ok=True)
    spurious_pairs = []
    for name in ADBENCH_SET:
        src = os.path.join(ADBENCH_DIR, name)
        injected = os.path.join(SPUR_DIR, name.replace(".npz", "_spurious_indep5.npz"))
        inject_spurious_npz(src, injected, k=5, mode="indep", strength=0.0, seed=123)
        spurious_pairs.append((name, injected))

    exp3_local_on = {
        "name": "spurious_local_on",
        "args": ["--suite", "adbench", "--adbench_dir", SPUR_DIR, "--score", "normalized", "--not_use_excl"],
    }
    exp3_local_off = {
        "name": "spurious_local_off",
        "args": ["--suite", "adbench", "--adbench_dir", SPUR_DIR, "--score", "normalized", "--no-local-test", "--not_use_excl"],
    }

    for exp in (exp3_local_on, exp3_local_off):
        print(f"\n=== Running {exp['name']} ===")
        cmd = runner_base_args() + exp["args"]
        for _, injected in spurious_pairs:
            run(cmd + ["--dataset", os.path.basename(injected)])
    out_roots += sorted(glob.glob(os.path.join("results", "NEW_exp_all_*")),
                        key=os.path.getmtime)[-2:]

    # 4) Merge all aggregates into one CSV
    merged_csv = os.path.join(OUT_BASE, "all_results_merged.csv")
    merge_results(out_roots, merged_csv)

    print("\nAll done.")
    print("Result roots:")
    for r in out_roots:
        print(" -", r)
    print("Merged CSV:", merged_csv)
    print("\nNotes:")
    print("• Each run folder already contains ROC-AUC/PR-AUC, runtime, rule counts/lengths if your calculate_exp_res exports them.")
    print("• For overlap analysis, use the per-run fields (e.g., rules_prob_test/train, per-instance probs) saved by calculate_exp_res.")
    print("• Spurious tests write augmented NPZs to:", SPUR_DIR)


if __name__ == "__main__":
    main()
