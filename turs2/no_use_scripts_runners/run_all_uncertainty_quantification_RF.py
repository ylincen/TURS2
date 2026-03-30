# This script reads the same train/test splits of the data used in running experiments of TURS (run_all.py)
# The goal is to test if the overlap with disagreement indicates epistemic uncertainty in Random Forests

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def rf_ensemble_epistemic_uncertainty_from_split(
    X_train,
    y_train,
    X_test,
    y_test,
    n_models=30,
    rf_kwargs=None,
):
    """
    Train multiple RF models and compute:

      - mean predictive probabilities (ensemble)
      - epistemic uncertainty = variance across RF models
      - per-instance (uncalibrated) negative log-likelihood on y_test

    Returns
    -------
    mean_probs : np.ndarray, shape (n_test, n_classes)
        Ensemble mean predicted probabilities for each class.
    var_probs : np.ndarray, shape (n_test, n_classes)
        Variance of predicted probabilities across ensemble members (epistemic).
    nll : np.ndarray, shape (n_test,)
        Per-instance negative log-likelihood based on ensemble mean probabilities.
    """

    if rf_kwargs is None:
        rf_kwargs = dict(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
        )

    prob_list = []
    classes_ = None

    # ---- Train K independent forests ----
    for seed in range(n_models):
        rf = RandomForestClassifier(random_state=seed, **rf_kwargs)
        rf.fit(X_train, y_train)

        probs = rf.predict_proba(X_test)  # shape (n_test, n_classes)

        # record the class order used in predict_proba
        if classes_ is None:
            classes_ = rf.classes_
        else:
            # sanity check (should be identical across ensemble members)
            assert np.array_equal(
                classes_, rf.classes_
            ), "Inconsistent class orders across RF models!"

        prob_list.append(probs)

    # shape -> (K, n_test, n_classes)
    prob_stack = np.stack(prob_list, axis=0)

    # Ensemble mean probs
    mean_probs = prob_stack.mean(axis=0)

    # Epistemic uncertainty = across-model variance
    var_probs = prob_stack.var(axis=0)

    # ---- Uncalibrated negative log-likelihood (per instance) ----
    # Map true labels to column indices in mean_probs according to classes_
    label_to_col = {label: idx for idx, label in enumerate(classes_)}
    col_indices = np.array([label_to_col[label] for label in y_test])

    true_class_probs = mean_probs[np.arange(len(y_test)), col_indices]

    # numerical safety
    true_class_probs = np.clip(true_class_probs, 1e-12, 1.0)

    # NLL = -log p(y | x)
    nll = -np.log(true_class_probs)
    

    return mean_probs, var_probs, nll, prob_list


def run_rf_epistemic_uncertainty_on_dataset(
    dataset_name,
    data_root="./RUN_ALL_GetLabelsOnly_20251210/score-normalized_patience-True_local-either_aux-False/splits_with_features",
    results_dir="./RUN_ALL_RF_EPISTEMIC_UNCERTAINTY_RESULTS/",
    n_models=30,
    rf_kwargs=None,
    n_folds=5,
):
    """
    For each fold:
      - read *train* and *test* CSVs with columns X0,...,Xn,idx,y
      - train RF ensemble on train
      - compute variance of predicted probabilities + NLL on test
      - store results for test instances in global containers keyed by idx

    No full dataset is ever loaded; only per-fold train/test files.
    """

    # global containers: idx -> arrays
    uq_all_folds = {}        # idx -> var_probs (vector over classes)
    mean_p_all_folds = {}    # idx -> mean_probs (vector over classes)
    nll_all_folds = {}       # idx -> scalar NLL

    uq_pred_class_all = {}   # idx -> scalar UQ for predicted class
    uq_mean_var_all = {}     # idx -> scalar UQ as mean variance over classes

    prob_seed0_all = {}      # <--- NEW: idx -> probs from seed 0 (vector over classes)


    n_classes = None         # we will infer from the first fold

    for fold in range(n_folds):
        # fold_dir = os.path.join(data_root, dataset_name, f"fold{fold}")
        data_dir = os.path.join(
            data_root,
            dataset_name,
        )

        # Adjust these filenames if your naming is different:
        train_path = os.path.join(data_dir, f"fold{fold}_train_labels_and_features.csv")
        test_path = os.path.join(data_dir, f"fold{fold}_test_labels_and_features.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Features = all X* columns; idx and y are separate
        feature_cols = [c for c in train_df.columns if c.startswith("X")]

        # ----- train split -----
        X_train = train_df[feature_cols].values
        y_train = train_df["y"].values

        # ----- test split -----
        X_test = test_df[feature_cols].values
        y_test = test_df["y"].values
        idx_test = test_df["idx"].values     # global indices for the dataset

        # REMEMBER: we do NOT use idx as a feature

        # ----- RF ensemble epistemic UQ + NLL on this fold -----
        mean_probs, var_probs, nll, prob_list = rf_ensemble_epistemic_uncertainty_from_split(
            X_train,
            y_train,
            X_test,
            y_test,
            n_models=n_models,
            rf_kwargs=rf_kwargs,
        )

        if n_classes is None:
            n_classes = mean_probs.shape[1]

        # probs of the first RF model (seed = 0) on this fold's test set
        probs_seed0 = prob_list[0]   # shape (n_test, n_classes)

        # ----- derive scalar UQ per instance -----
        # predicted class under ensemble mean
        pred_class = np.argmax(mean_probs, axis=1)

        # variance of predicted class
        uq_pred_class = var_probs[np.arange(len(pred_class)), pred_class]

        # mean variance over all classes
        uq_mean_var = var_probs.mean(axis=1)

        # Update global dicts for test instances
        for i, idx_val in enumerate(idx_test):
            mean_p_all_folds[idx_val] = mean_probs[i]
            uq_all_folds[idx_val] = var_probs[i]
            nll_all_folds[idx_val] = float(nll[i])

            uq_pred_class_all[idx_val] = float(uq_pred_class[i])
            uq_mean_var_all[idx_val] = float(uq_mean_var[i])

            prob_seed0_all[idx_val] = probs_seed0[i]   # <--- NEW


    # -----------------------------------------------------------------
    # Convert dicts to a DataFrame (one row per unique idx)
    # -----------------------------------------------------------------
    all_idx_sorted = sorted(uq_all_folds.keys())

    mean_p_matrix = np.vstack([mean_p_all_folds[i] for i in all_idx_sorted])
    uq_matrix = np.vstack([uq_all_folds[i] for i in all_idx_sorted])
    nll_vector = np.array([nll_all_folds[i] for i in all_idx_sorted])

    uq_pred_class_vec = np.array([uq_pred_class_all[i] for i in all_idx_sorted])
    uq_mean_var_vec = np.array([uq_mean_var_all[i] for i in all_idx_sorted])

    prob_seed0_matrix = np.vstack([prob_seed0_all[i] for i in all_idx_sorted])


    out_df = pd.DataFrame({"idx": all_idx_sorted})

    for c in range(n_classes):
        out_df[f"mean_p_class_{c}"] = mean_p_matrix[:, c]
        out_df[f"var_p_class_{c}"] = uq_matrix[:, c]

    out_df["nll"] = nll_vector
    out_df["uq_pred_class"] = uq_pred_class_vec
    out_df["uq_mean_var"] = uq_mean_var_vec
    out_df["prob_seed_0"] = list(prob_seed0_matrix)

    # out_df["prob_seed_0"] = list(prob_list[0])

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir, f"{dataset_name}_rf_epistemic_uncertainty_all_folds.csv"
    )
    out_df.to_csv(results_path, index=False)

    print(f"Aggregated UQ + NLL for all folds saved to {results_path}")

if __name__ == "__main__":
    # take a single dataset name from the command line
    # user parser
    import argparse
    parser = argparse.ArgumentParser(
        description="Run RF epistemic uncertainty quantification on a dataset with pre-defined splits."
    )
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, required=True,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    run_rf_epistemic_uncertainty_on_dataset(
        dataset_name=dataset_name,
        n_models=30,
        rf_kwargs=dict(
            n_estimators=100,
            max_depth=None,
            n_jobs=1,
            min_samples_leaf=20,
        ),
        n_folds=5,
    )


        