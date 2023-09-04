from collections import namedtuple

AlgConfig = namedtuple('AlgConfig', [
            "beam_width",
            "num_candidate_cuts", "max_num_rules", "max_grow_iter",
            "num_class_as_given", "log_learning_process",
            "dataset_name", "feature_names",
            "validity_check"
        ])

RuleRemainingSurrogate = namedtuple("RuleRemainingSurrogate", ["excl", "incl"])

GrowGains = namedtuple("GrowGains", ["incl_abs_gain", "excl_abs_gain", "incl_gain_per_excl_cover",
                                     "excl_gain_per_excl_cover",
                                     "cl_model", "cl_data_incl", "cl_data_excl",
                                     "incl_coverage", "excl_coverage"])