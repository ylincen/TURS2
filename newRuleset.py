import copy
import numpy as np
from utils import *
from nml_regret import *
from newBeam import *
from newRule import *
import surrogate_tree
from newModelingGroups import *
import time
import pickle


class Ruleset:
    def __init__(self, data_info, features, target, number_of_init_rules,
                 number_of_rules_return=1):
        """
        Init the Ruleset object as an empty ruleset
        :param data_info: meta-data for the original dataset (see more details in newRule.py)
        :param features: feature matrix (numpy 2d array) for the data
        :param target: target (numpy 1d array) for the the data
        """
        self.rules = []
        self.else_rule = ElseRule(bool_array=np.ones(len(target), dtype=bool), data_info=data_info, features=features,
                                  target=target)

        self.data_info = data_info
        self.features = features
        self.target = target

        self.covered_boolarray = np.zeros(data_info.nrow, dtype=bool)  # does not cover anything at first

        self.modeling_groups = ModelingGroupSet(else_rule=self.else_rule,
                                                data_info=self.data_info,
                                                rules=self.rules, target=self.target,
                                                features=self.features)

        self.number_of_init_rules = number_of_init_rules  # search settings: how many rules to return for phase-1 (ignoring overlap)
        self.number_of_rules_return = number_of_rules_return  # search settings: how many rules to return for phase-2 (including overlaps)

        self.grow_history_scores = [self.modeling_groups.else_rule_modeling_group.non_surrogate_score]
        self.grow_history_surroage_scores = [self.modeling_groups.else_rule_modeling_group.surrogate_score]

        self.coverage = np.count_nonzero(self.covered_boolarray)
        self.score = self.modeling_groups.total_score()
        # self.score_per_coverage = self.score / self.coverage

    def update_ruleset(self, rule):
        """
        Given the next "rule", CHECK for STOPPING CRITERION &
        update the ruleset, and also at the same time the modeling_groups
        :param rule:
        :return:
        """

        # update the modeling group
        self.modeling_groups.update(rule)

        # update else_rule
        self.else_rule = ElseRule(bool_array=self.modeling_groups.else_rule_modeling_group.instances_modeling_boolean,
                                  data_info=self.data_info, target=self.target, features=self.features,
                                  get_surrogate_score=False)

        # update the ruleset itself
        self.rules.append(rule)
        self.covered_boolarray = ~self.else_rule.bool_array

        # update the "history" scores
        self.grow_history_surroage_scores.append(self.modeling_groups.total_surrogate_score())
        self.grow_history_scores.append(self.modeling_groups.total_score())


    def build(self, max_iter, beam_width, candidate_cuts, print_or_not=True, dump=False):
        """
        :param max_iter: maximum number of iterations, also the maximum number of rules we may possibly have
        :param beam_width: beam_width for beam search
        :param candidate_cuts: candidate_cuts_for_all_dimensions;  TO DO LATER;
        :return:
        """
        t0 = time.time()
        cl_model_list = []

        buffer_count = 0

        for iter in range(max_iter):
            rule = self.find_next_rule(beam_width, candidate_cuts)
            mdl_score_if_added_this_rule = self.evaluate_rule(rule, surrogate=False)
            cl_model_list.append(rule.cl_model)
            cl_model = np.sum(cl_model_list)
            cl_model_previous = np.sum(cl_model_list[:len(cl_model_list)-1])

            if mdl_score_if_added_this_rule + cl_model < self.grow_history_scores[-1] + cl_model_previous or buffer_count < 5:
                if mdl_score_if_added_this_rule + cl_model < self.grow_history_scores[-1] + cl_model_previous:
                    buffer_count = 0
                else:
                    buffer_count += 1

                # print(rule.condition)

                if rule is None:
                    break
                else:
                    self.update_ruleset(rule)

                if print_or_not:
                    print("iter: ", iter,
                          "rule.prob:", rule.prob, "rule.prob_excl:", rule.prob_excl, "rule.coverage:", rule.nrow,
                          "rule.coverage_excl:", rule.nrow_excl,
                          "rule set coverage: ", np.count_nonzero(self.covered_boolarray) / self.data_info.nrow,
                          "time cost so far: ", time.time() - t0)
                if dump:
                    with open("ruleset_beam10_cuts20.pkl", 'wb') as fp:
                        pickle.dump(obj=self, file=fp)

                if self.modeling_groups.else_rule_modeling_group.surrogate_equal_nonsurrogate:
                    break
            else:
                break

    def find_next_rule(self, beam_width, candidate_cuts):
        beam_ignore_overlap = self.find_beam_ignore_overlap(beam_width=beam_width,
                                                            number_of_init_rules=self.number_of_init_rules,
                                                            candidate_cuts=candidate_cuts)
        # print("rule before refine: ", beam_ignore_overlap[0].condition)
        best_rule = self.find_rule_with_overlap(best_rules_excl=beam_ignore_overlap,
                                                beam_width=beam_width,
                                                candidate_cuts=candidate_cuts)
        return best_rule

    def find_beam_ignore_overlap(self, beam_width, number_of_init_rules, candidate_cuts):
        """
        Given all covered instances, grow a list of beams of rules while ignoring the covered_instances, among which
        then select the best group "beam_collect";
        :param beam_width:
        :param number_of_init_rules:
        :param candidate_cuts:
        :return:
        """
        rule = self.init_emptyrule()

        # init the beam
        beam = Beam(beam_width, data_info=self.data_info)
        beam.rules = [rule]

        rules_alldepth = []

        while len(beam.rules) > 0:  # if an empty beam is returned, the "grow" process is stopped.
            rules_alldepth.extend(beam.rules)

            # grow based on each rule in the beam
            next_beam = Beam(beam_width=beam_width, data_info=self.data_info)
            next_beam.rules = beam.grow_rule_excl(candidate_cuts)

            # update the beam to be the "next_beam"
            beam = next_beam

            # print("beam_collect: ")
            # print([r.condition for r in beam.rules])

        surrogate_scores = []
        # rules_alldepth.pop(1)

        for rule in rules_alldepth:
            rule_score = rule.neglog_likelihood_excl + rule.regret_excl

            else_rule_cover_updated = copy.deepcopy(self.else_rule.bool_array)
            else_rule_cover_updated[rule.indices_excl_overlap] = False

            if any(else_rule_cover_updated) and len(rule.condition["icols"]) != 0:
                surrogate_score_else_rule = \
                    surrogate_tree.get_tree_cl(x_train=self.features[else_rule_cover_updated],
                                               y_train=self.target[else_rule_cover_updated],
                                               num_class=self.data_info.num_class)
            else:
                surrogate_score_else_rule = 0   # Why is here 0 ????

            surrogate_score = rule_score + np.sum(surrogate_score_else_rule)
            surrogate_scores.append(surrogate_score)

        best_rules = []
        surrogate_score_order = np.argsort(surrogate_scores)
        for i in surrogate_score_order:
            if len(best_rules) >= number_of_init_rules:
                break

            best_rules.append(rules_alldepth[i])

        return best_rules

    def find_rule_with_overlap(self, best_rules_excl, beam_width, candidate_cuts):
        beam = Beam(beam_width, data_info=self.data_info)
        beam.rules = best_rules_excl
        
        rules_alldepth = []
        while len(beam.rules) > 0:
            rules_alldepth.extend(beam.rules)
            next_beam = Beam(beam_width, data_info=self.data_info)
            next_beam.rules = beam.grow_rule_incl(candidate_cuts)
            beam = next_beam

        scores = []
        for i, rule in enumerate(rules_alldepth):
            if len(rule.condition["icols"]) == 0:
                ruleset_with_new_rule_score = self.modeling_groups.total_surrogate_score()
            else:
                ruleset_with_new_rule_score = self.evaluate_rule(rule, surrogate=True)
            scores.append(ruleset_with_new_rule_score)
        if len(scores) > 0:
            which_rule = np.argmin(scores)
            return rules_alldepth[which_rule]
        else:
            return None

    def init_emptyrule(self):
        indices = np.arange(self.data_info.nrow)
        indices_excl_overlap = np.where(~self.covered_boolarray)[0]

        condition = {"icols": [], "var_types": [], "cuts": [], "cut_options": []}  # empty condition;
        rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, rule_base=None,
                    features=self.features, target=self.target,
                    features_excl_overlap=self.features[indices_excl_overlap],
                    target_excl_overlap=self.target[indices_excl_overlap],
                    data_info=self.data_info, condition=condition,
                    local_gain=0)
        return rule

    def evaluate_rule(self, rule, surrogate):
        return self.modeling_groups.evaluate_rule(rule, surrogate)

    def self_prune(self, ruleset_size=None):
        if ruleset_size is None:
            cl_model_list = []
            for r in self.rules:
                cl_model_list.append(r.cl_model)

            cl_model_cum = np.cumsum(cl_model_list)
            mdl_score_with_cl_model = self.grow_history_scores + np.append(0, cl_model_cum)
            ruleset_size = np.argmin(mdl_score_with_cl_model)

            if ruleset_size == len(self.grow_history_scores) - 1:
                return self
            else:
                pruned_ruleset = Ruleset(self.data_info, self.data_info.features, self.data_info.target,
                                         number_of_init_rules=self.number_of_init_rules)
                for i in range(ruleset_size):
                    pruned_ruleset.update_ruleset(self.rules[i])
                return pruned_ruleset
        else:
            pruned_ruleset = Ruleset(self.data_info, self.data_info.features, self.data_info.target,
                                     number_of_init_rules=self.number_of_init_rules)
            for i in range(ruleset_size):
                pruned_ruleset.update_ruleset(self.rules[i])
            return pruned_ruleset

