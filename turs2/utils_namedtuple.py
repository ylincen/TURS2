from collections import namedtuple

AlgConfig = namedtuple('AlgConfig', [
            "beam_width",
            "num_candidate_cuts", "max_num_rules", "max_grow_iter",
            "num_class_as_given",
            "dataset_name", "feature_names",
            "rf_assist", "rf_oob_decision_function",
            "log_learning_process", "log_folder_name", "X_test", "y_test",
            "beamsearch_positive_gain_only", "beamsearch_normalized_gain_must_increase_comparing_rulebase", "beamsearch_stopping_when_best_normalized_gain_decrease",
            "validity_check", "rerun_on_invalid", "rerun_positive_control"
        ])

RuleRemainingSurrogate = namedtuple("RuleRemainingSurrogate", ["excl", "incl"])

GrowGains = namedtuple("GrowGains", ["incl_abs_gain", "excl_abs_gain", "incl_gain_per_excl_cover",
                                     "excl_gain_per_excl_cover",
                                     "cl_model", "cl_data_incl", "cl_data_excl",
                                     "incl_coverage", "excl_coverage"])