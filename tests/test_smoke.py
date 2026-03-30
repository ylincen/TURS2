"""
Minimal smoke tests: fit TURS on iris and check outputs are sane.
Run with:  pytest tests/
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from turs2.DataInfo import DataInfo
from turs2.Ruleset import Ruleset
from turs2.ModelEncoding import ModelEncodingDependingOnData
from turs2.DataEncoding import NMLencoding
from turs2.nml_regret import regret
from turs2.exp_predictive_perf import calculate_brier_and_prauc, calculate_roc_auc_and_logloss
from turs2.utils_predict import predict_ruleset
from turs2.exp_utils import cover_matrix_fun


@pytest.fixture(scope="module")
def iris_ruleset():
    iris = load_iris()
    X, y = iris.data, iris.target
    # small config so tests stay fast
    import types
    alg_config = types.SimpleNamespace(
        num_candidate_cuts=10, max_num_rules=50, max_grow_iter=20,
        num_class_as_given=None, beam_width=3, log_learning_process=False,
        dataset_name=None, X_test=None, y_test=None,
        feature_names=[f"X{i}" for i in range(X.shape[1])],
        beamsearch_positive_gain_only=False,
        beamsearch_normalized_gain_must_increase_comparing_rulebase=False,
        beamsearch_stopping_when_best_normalized_gain_decrease=False,
        validity_check="either", rerun_on_invalid=False,
        rerun_positive_control=False, min_sample_each_rule=2,
        scoring="normalized", use_patience=True,
    )
    data_info = DataInfo(X=X, y=y, beam_width=None, alg_config=alg_config, not_use_excl_=True)
    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
    ruleset.fit(max_iter=10, printing=False)
    return ruleset, X, y


def test_ruleset_has_rules(iris_ruleset):
    ruleset, X, y = iris_ruleset
    assert len(ruleset.rules) > 0


def test_predict_shape(iris_ruleset):
    ruleset, X, y = iris_ruleset
    probs = predict_ruleset(ruleset, X, y)
    assert probs.shape == (len(y), ruleset.data_info.num_class)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_roc_auc_reasonable(iris_ruleset):
    ruleset, X, y = iris_ruleset
    probs = predict_ruleset(ruleset, X, y)
    roc_auc, _, _, _ = calculate_roc_auc_and_logloss(ruleset, y, probs, y, probs)
    assert roc_auc > 0.5


def test_brier_score_range(iris_ruleset):
    ruleset, X, y = iris_ruleset
    probs = predict_ruleset(ruleset, X, y)
    _, _, brier, _ = calculate_brier_and_prauc(ruleset, y, y, probs, probs)
    assert 0.0 <= brier <= 1.0


def test_nml_regret_positive():
    assert regret(100, 2) > 0
    assert regret(100, 3) > regret(100, 2)
    assert regret(1, 2) >= 0
