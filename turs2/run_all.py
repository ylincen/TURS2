import os, sys, glob, time, argparse, types
from datetime import datetime
from typing import Optional

# Add the repo root and turs2 package to the path so this script can be run
# from any working directory on any machine.
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
for _p in [_repo_root, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_RUNTIME_FIRST_FOLD = 24 * 60 * 60  # 24 hours


import numpy as np
import pandas as pd
import signal


from sklearn.model_selection import StratifiedKFold

from turs2.DataInfo import DataInfo
from turs2.Ruleset import Ruleset
from turs2.ModelEncoding import ModelEncodingDependingOnData
from turs2.DataEncoding import NMLencoding
from turs2.exp_utils import *
from turs2.exp_predictive_perf import *

np.seterr(all='raise')

class FirstFoldTimeout(Exception):
    """Raised when the first fold exceeds its time limit."""
    pass

def log_time_limit(
    out_dir: str,
    suite: str,
    dataset_name: str,
    fold_idx: int,
    score_type: str,
    use_patience: bool,
    validity_check: str,
    not_use_excl: bool,
    max_runtime_first_fold: float,
):
    """
    Append a line to BASE_RESULTS_DIR/TIME_LIMIT_OCCUR.txt.
    BASE_RESULTS_DIR is inferred as parent of out_dir.
    """
    
    base_results_dir = os.path.dirname(out_dir)
    os.makedirs(base_results_dir, exist_ok=True)
    marker_path = os.path.join(base_results_dir, "TIME_LIMIT_OCCUR.txt")

    setting_desc = (
        f"score={score_type}, "
        f"patience={use_patience}, "
        f"validity_check={validity_check}, "
        f"not_use_excl={not_use_excl}"
    )

    with open(marker_path, "a") as f:
        f.write(
            f"{datetime.now().isoformat()} | "
            f"suite={suite} | dataset={dataset_name} | fold={fold_idx} | "
            f"setting={setting_desc} | TIME_LIMIT={max_runtime_first_fold}s\n"
        )

    print(
        f"[TIMEOUT] First fold exceeded {max_runtime_first_fold} seconds "
        f"for dataset='{dataset_name}', suite='{suite}'. "
        f"Logged to {marker_path}."
    )



# -------------------- Dataset loaders --------------------

UCI_WITHOUT_HEADER = [
    "chess","iris","waveform","backnote","contracept","ionosphere",
    "magic","car","tic-tac-toe","wine","glass","pendigits","HeartCleveland"
]
UCI_WITH_HEADER = [
    "avila","anuran","diabetes","Vehicle","DryBeans"
]
UCI_DIR = "../datasets/"
ADBENCH_DIR = "../ADbench_datasets_Classical/"

def load_uci_dataset(name: str):
    d = read_data(name, folder_name = UCI_DIR, 
                  datasets_without_header_row=UCI_WITHOUT_HEADER,
                  datasets_with_header_row=UCI_WITH_HEADER)
    d = preprocess_data(d)
    X = d.iloc[:, :-1].to_numpy()
    y = d.iloc[:, -1].to_numpy()
    return d, X, y

def load_adbench_dataset(path_or_name: str, base_dir=ADBENCH_DIR):
    # allow full path or just filename
    if path_or_name.endswith(".npz") and os.path.isfile(path_or_name):
        npz_path = path_or_name
    else:
        npz_path = os.path.join(base_dir, path_or_name)
    d_np = np.load(npz_path)
    X, y = d_np["X"], d_np["y"]
    d_original = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
    d = preprocess_data(d_original)
    return d, X, y, npz_path


# -------------------- Runner --------------------

def run_one_dataset(
    suite: str,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    fold_arg: Optional[int],
    n_splits: int,
    seed: int,
    # config & ablations
    num_candidate_cuts: int = 20,
    beam_width: int = 10,
    max_rules: int = 500,
    max_grow_iter: int = 500,
    validity_check: str = "either",
    not_use_excl: bool = True,
    score_type: str = "normalized",  # {"normalized","absolute"}
    use_patience: bool = True,
    max_runtime_first_fold: Optional[float] = None,
):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(kf.split(X=X, y=y))

    exp_res_all_folds = []
    overlap_res_allfolds = []

    for fold_idx in range(n_splits):
        if fold_arg is not None and fold_idx != fold_arg:
            continue

        train_idx, test_idx = folds[fold_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        alg_config = types.SimpleNamespace()
        alg_config.num_candidate_cuts = num_candidate_cuts
        alg_config.max_num_rules = max_rules
        alg_config.max_grow_iter = max_grow_iter
        alg_config.num_class_as_given = None
        alg_config.beam_width = beam_width
        alg_config.log_learning_process = False
        alg_config.dataset_name = None
        alg_config.X_test = None
        alg_config.y_test = None
        alg_config.feature_names = [f"X{i}" for i in range(X.shape[1])]
        # existing flags you used:
        alg_config.beamsearch_positive_gain_only = False
        alg_config.beamsearch_normalized_gain_must_increase_comparing_rulebase = False
        alg_config.beamsearch_stopping_when_best_normalized_gain_decrease = False
        alg_config.validity_check = validity_check
        alg_config.rerun_on_invalid = False
        alg_config.rerun_positive_control = False
        alg_config.min_sample_each_rule = 2

        # ---- New ablation flags (wire these in your Ruleset/beam code) ----
        alg_config.scoring = score_type            # "normalized" vs "absolute"
        alg_config.use_patience = use_patience     # True/False

        # ---- DataInfo + encodings ----
        data_info = DataInfo(
            X=X_train, y=y_train,
            beam_width=None,
            alg_config=alg_config,
            not_use_excl_=not_use_excl
        )
        data_encoding = NMLencoding(data_info)
        model_encoding = ModelEncodingDependingOnData(data_info)

        ruleset = Ruleset(
            data_info=data_info,
            data_encoding=data_encoding,
            model_encoding=model_encoding
        )

        # ---------- TIME-LIMITED FIT FOR FIRST FOLD ----------
        start_time = time.time()

        # Only use alarm for fold 0 if max_runtime_first_fold is set
        use_alarm = (fold_idx == 0 and max_runtime_first_fold is not None)
        if use_alarm:
            old_handler = signal.getsignal(signal.SIGALRM)

            def _handler(signum, frame):
                raise FirstFoldTimeout()

            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(max_runtime_first_fold))

        try:
            ruleset.fit(max_iter=1000, printing=False)
            end_time = time.time()
        except FirstFoldTimeout:
            end_time = time.time()

            # Cancel alarm + restore handler
            if use_alarm:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

            # Log and BREAK out of the fold loop
            log_time_limit(
                out_dir=out_dir,
                suite=suite,
                dataset_name=dataset_name,
                fold_idx=fold_idx,
                score_type=score_type,
                use_patience=use_patience,
                validity_check=validity_check,
                not_use_excl=not_use_excl,
                max_runtime_first_fold=max_runtime_first_fold,
            )
            break  # <- skip remaining folds for this dataset
        else:
            # if no exception: we’re done with the alarm
            if use_alarm:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        # ---------- END TIME-LIMIT BLOCK ----------


        exp_res_full, exp_res, overlap_prob_diff_analysis_res_IndividualProbsAll_train_test = calculate_exp_res(
            ruleset, X_test, y_test, X_train, y_train,
            dataset_name, fold_idx, start_time, end_time
        )

        exp_res["suite"] = suite
        exp_res["dataset"] = dataset_name
        exp_res["fold"] = fold_idx
        exp_res["not_use_excl_"] = not_use_excl
        exp_res["validity_check_"] = validity_check
        exp_res["score_type"] = score_type
        exp_res["use_patience"] = use_patience
        exp_res["use_local_test"] = (validity_check == "either")
        exp_res["use_aux_beam"] = (not not_use_excl)

        exp_res_all_folds.append(exp_res)
        

        # save overlap_prob_diff_analysis_res_IndividualProbsAll to a separate file
        # only to it to the "normal setting", not abalation study settings
        if (validity_check == "either") and (score_type == "normalized") and use_patience and not_use_excl:
            overlap_df_test = pd.DataFrame({"probs": overlap_prob_diff_analysis_res_IndividualProbsAll_train_test["test"],
                                           "y_pred_prob": list(exp_res_full['y_pred_prob'])})
            overlap_df_test["fold"] = fold_idx
            overlap_df_test["split"] = "test"

            overlap_df_train = pd.DataFrame({"probs": overlap_prob_diff_analysis_res_IndividualProbsAll_train_test["train"],
                                             "y_pred_prob": list(exp_res_full['y_pred_prob_train'])})
            overlap_df_train["fold"] = fold_idx
            overlap_df_train["split"] = "train"


            overlap_df = pd.concat([overlap_df_test, overlap_df_train], ignore_index=True)
            overlap_res_allfolds.append(overlap_df)

    os.makedirs(out_dir, exist_ok=True)

    if not exp_res_all_folds:
        return pd.DataFrame(), out_dir

    # 1) exp_res for this dataset (all folds in one file)
    exp_df = pd.DataFrame(exp_res_all_folds)
    exp_path = os.path.join(out_dir, f"{dataset_name}.csv")
    exp_df.to_csv(exp_path, index=False)

    # 2) overlap analysis for this dataset (if applicable)
    if overlap_res_allfolds:
        overlap_df_all = pd.concat(overlap_res_allfolds, ignore_index=True)
        overlap_path = os.path.join(out_dir, f"{dataset_name}_overlap.csv")
        overlap_df_all.to_csv(overlap_path, index=False)

    return exp_df, out_dir
        


def main():
    parser = argparse.ArgumentParser(description="Unified TURS runner for UCI + ADBench")
    parser.add_argument("--suite", choices=["uci", "adbench", "both"], default="uci")
    parser.add_argument("--dataset", default="tic-tac-toe",
                        help="UCI name (e.g., car) or ADBench .npz name; for ADBench you can pass a glob, e.g., '42_*.npz'")
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold index (0..k-1). Omit to run all.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)

    # paths
    parser.add_argument("--adbench_dir", default="../ADbench_datasets_Classical")

    # config
    parser.add_argument("--num_candidate_cuts", type=int, default=20)
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--max_rules", type=int, default=500)
    parser.add_argument("--max_grow_iter", type=int, default=500)
    parser.add_argument("--validity_check", default="either")
    parser.add_argument("--use_aux_beam", action="store_true",
                        help="Enable auxiliary beam (excl beam); off by default")

    # ablations
    parser.add_argument("--score", choices=["normalized", "absolute"], default="normalized")
    parser.add_argument("--no-patience", action="store_true")
    parser.add_argument("--out_dir", default="RUN_ALL_BW10_OldModelEncoding_Ablation_NormalizedGain",
                        help="Base output directory for results")

    args = parser.parse_args()
    not_use_excl = not args.use_aux_beam
    BASE_RESULTS_DIR = args.out_dir
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    # output folder
    setting_tag = f"score-{args.score}_patience-{not args.no_patience}_local-{args.validity_check}_aux-{args.use_aux_beam}"
    out_root = os.path.join(BASE_RESULTS_DIR, setting_tag)
    os.makedirs(out_root, exist_ok=True)

    all_dfs = []

    def run_uci(name: str):
        _, X, y = load_uci_dataset(name)
        out_dir = out_root
        df, _ = run_one_dataset(
            suite="uci",
            dataset_name=name, X=X, y=y, out_dir=out_dir,
            fold_arg=args.fold, n_splits=args.splits, seed=args.seed,
            num_candidate_cuts=args.num_candidate_cuts,
            beam_width=args.beam_width,
            max_rules=args.max_rules,
            max_grow_iter=args.max_grow_iter,
            validity_check=args.validity_check,
            not_use_excl=not_use_excl,
            score_type=args.score,
            use_patience=(not args.no_patience),
            max_runtime_first_fold=MAX_RUNTIME_FIRST_FOLD,
        )
        all_dfs.append(df)

    def run_adbench(pattern: Optional[str]):
        out_dir = out_root
        if pattern is None:
            targets = [os.path.join(args.adbench_dir, "42_WBC.npz")]
        else:
            # accept either file or glob
            if pattern.endswith(".npz") and os.path.isfile(pattern):
                targets = [pattern]
            else:
                targets = sorted(glob.glob(os.path.join(args.adbench_dir, pattern)))
                if not targets and os.path.isfile(os.path.join(args.adbench_dir, pattern)):
                    targets = [os.path.join(args.adbench_dir, pattern)]
        if not targets:
            print(f"[WARN] No ADBench datasets matched: {pattern}")
            return
        for npz_path in targets:
            _, X, y, fullpath = load_adbench_dataset(npz_path, base_dir=args.adbench_dir)
            dataset_name = os.path.basename(fullpath)
            df, _ = run_one_dataset(
                suite="adbench",
                dataset_name=dataset_name, X=X, y=y, out_dir=out_dir,
                fold_arg=args.fold, n_splits=args.splits, seed=args.seed,
                num_candidate_cuts=args.num_candidate_cuts,
                beam_width=args.beam_width,
                max_rules=args.max_rules,
                max_grow_iter=args.max_grow_iter,
                validity_check=args.validity_check,
                not_use_excl=not_use_excl,
                score_type=args.score,
                use_patience=(not args.no_patience),
                max_runtime_first_fold=MAX_RUNTIME_FIRST_FOLD,
            )
            all_dfs.append(df)

    # dispatch
    if args.suite in ("uci", "both"):
        if args.dataset and args.dataset.endswith(".npz"):
            print("[WARN] --suite=uci but dataset looks like ADBench .npz; ignoring extension.")
        if args.dataset:
            run_uci(args.dataset)
        else:
            for name in ["iris"]:
                run_uci(name)

    if args.suite in ("adbench", "both"):
        run_adbench(args.dataset)

    # write aggregate
    if all_dfs:
        agg = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(out_root, exist_ok=True)
        agg.to_csv(os.path.join(out_root, "aggregate_results.csv"), index=False)
        print(f"[DONE] Wrote aggregate: {os.path.join(out_root, 'aggregate_results.csv')}")
    else:
        print("[DONE] Nothing ran (check --suite/--dataset args).")


if __name__ == "__main__":
    main()
