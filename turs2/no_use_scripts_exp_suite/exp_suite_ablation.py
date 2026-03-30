#!/usr/bin/env python3
import os, sys, subprocess, shutil, glob, json
from datetime import datetime
import numpy as np
import pandas as pd
import argparse

# Path to your unified runner
RUNNER = os.path.join(os.path.dirname(__file__), "run_all.py")

# Default dataset sets (tweak as you like)
UCI_SET = [
    "Vehicle", "glass", "pendigits", "HeartCleveland", "chess", "iris",
    "backnote", "contracept", "ionosphere", "car", "tic-tac-toe", "wine",
    "diabetes", "anuran", "avila", "magic", "waveform", "DryBeans",
]
ADBENCH_DIR = "../ADbench_datasets_Classical"
ADBENCH_SET = [
    "26_optdigits.npz", "34_smtp.npz", "28_pendigits.npz", "43_WDBC.npz",
    "36_speech.npz", "31_satimage-2.npz", "3_backdoor.npz",
    "38_thyroid.npz", "41_Waveform.npz", "23_mammography.npz",
    "40_vowels.npz", "25_musk.npz", "1_ALOI.npz", "14_glass.npz",
]

def build_experiments():
    """Define the different configurations (baseline + ablations)."""
    exp1 = {
        "name": "baseline",
        "args": ["--score", "normalized", "--not_use_excl"],
    }
    exp2_aux = {
        "name": "ablation_aux",
        "args": ["--score", "normalized"],
    }
    exp2_nolocal = {
        "name": "ablation_no_local",
        # no local test via validity_check=none
        "args": ["--score", "normalized", "--not_use_excl", "--validity_check", "none"],
    }
    exp2_nopatience = {
        "name": "ablation_no_patience",
        "args": ["--score", "normalized", "--not_use_excl", "--no-patience"],
    }
    return [exp1, exp2_aux, exp2_nolocal, exp2_nopatience]

# ---------- Helpers ----------


def runner_base_args():
    # same base args everywhere; call Python explicitly
    return [sys.executable, RUNNER, "--splits", "5", "--seed", "2"]

def merge_results(output_roots, merged_csv_path):
    """Merge aggregate_results.csv from several experiment roots."""
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

def make_settings(OUT_BASE):
    """
    Return an ordered list of (key, func) where func runs that single setting.

    A "setting" here = one experiment config over:
      - all UCI datasets, or
      - all ADBench datasets.
    """
    experiments = build_experiments()
    time_limit = 24 * 60 * 60  # 24 hours per dataset

    def run_cmd(cmd):
        print(">>>", " ".join(cmd))
        try:
            res = subprocess.run(cmd, timeout=time_limit)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] command exceeded {time_limit} seconds: {' '.join(cmd)}")
            return False

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
            if failed:
                print(f"[SETTING WARN] UCI {exp['name']} failed datasets: {failed}")

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

    return settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-idx", type=int, default=None,
        help="Run only the Nth setting (1-based index from --list-settings).",
    )
    parser.add_argument(
        "--list-settings", action="store_true",
        help="List available settings and exit.",
    )
    args = parser.parse_args()

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
            print(f"[ERROR] --setting-idx out of range (1..{len(settings)}). "
                  f"Use --list-settings to see options.")
            return
        key, fn = settings[i - 1]
        print(f"\n=== Running setting [{i}] {key} ===")
        fn()
        return



if __name__ == "__main__":
    main()
