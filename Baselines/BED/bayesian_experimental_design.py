import logging
import numpy as np
import random
from multiprocessing import Pool
import itertools
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
# from copy import deepcopy, copy

from QMR import QMR

logger = logging.getLogger(__name__)


@dataclass
class State:
    act_pos: bool
    action: int
    candidate_diseases: List[int]


class BED:
    def __init__(self, args):
        self.qmr = QMR({'args': args})
        self.args = args
        self.max_episode_len = args.max_episode_len
        self.threshold = args.threshold
        self.search_depth = args.search_depth
        self.util_func = self.utility_func(args.utility_func)

    def utility_func(self, func_name):
        def eps(prob):
            return prob + np.finfo(np.float).eps

        def kl_divergence(prob1, prob2):
            return (np.log(eps(prob1)) * prob1 - np.log(eps(prob2)) * prob1).sum()

        def shannon_information(prob1, prob2):
            return (np.log(eps(prob1)) * prob1 - np.log(eps(prob2)) * prob2).sum()

        if func_name.lower() == 'kl':
            return kl_divergence
        elif func_name.lower() == 'si':
            return shannon_information
        else:
            raise NotImplementedError(
                f'Utility function {func_name} is not defined. Must be one of (KL, SI).')


    def get_state(self, action, qmr):
        act_pos = True if action in qmr.findings else False
        return State(act_pos, action, qmr.candidate_diseases)


    def set_state(self, state, qmr):
        if state.act_pos:
            if isinstance(qmr.findings, set):
                qmr.pos_findings.pop()
            elif not isinstance(qmr.findings[state.action], set):
                qmr.pos_findings.pop()
            else:
                for i in range(len(qmr.findings[state.action])):
                    qmr.pos_findings.pop()
        else:
            qmr.neg_findings.pop()
        qmr.candidate_diseases = state.candidate_diseases


    def get_binary_finding_utility(self, finding, qmr, depth, old_probs, old_joint, pos_findings, neg_findings):
        new_probs_pos = None
        new_probs_neg = None
        if depth > 0:
            state = self.get_state(finding, qmr)
            qmr.step(finding)
            _, new_probs_pos, new_probs_neg = self.act_recursive(qmr, depth - 1)
            self.set_state(state, qmr)

        # When the inquired finding is positive
        new_probs_pos_curr, new_joint = qmr.compute_disease_probs(
            pos_findings + [finding], neg_findings, normalize=True)
        p_pos = new_joint / old_joint

        # When the inquired finding is negative
        new_probs_neg_curr, new_joint = qmr.compute_disease_probs(
            pos_findings, neg_findings + [finding], normalize=True)
        p_neg = new_joint / old_joint

        # Compute utility of the current finding
        if new_probs_pos is None:  # For some findings, cannot take further steps
            new_probs_pos = new_probs_pos_curr
            new_probs_neg = new_probs_neg_curr

        utility = p_pos * self.util_func(new_probs_pos, old_probs) + \
            p_neg * self.util_func(new_probs_neg, old_probs)

        return utility, new_probs_pos, new_probs_neg


    def get_categorical_finding_utility(self, finding, qmr, depth, old_probs, old_joint, pos_findings, neg_findings):
        assert (depth <= 0), "recursive reasoning not yet implemented for categorical findings"

        options = qmr.get_finding_options(finding)

        option_results = []
        utility = 0

        for o in options:
            new_probs_pos_curr, new_joint = qmr.compute_disease_probs(
                pos_findings + [(finding, o)], neg_findings, normalize=True)
            p_pos = new_joint / old_joint
            utility += p_pos * self.util_func(new_probs_pos_curr, old_probs)
            option_results.append([p_pos, new_probs_pos_curr])

        return utility, option_results, None


    def get_multichoice_finding_utility(self, finding, qmr, depth, old_probs, old_joint, pos_findings, neg_findings):
        assert (depth <= 0), "recursive reasoning not yet implemented for multi-choice findings"

        options = qmr.get_finding_options(finding)

        option_results = []

        for o in options:
            # When the inquired finding option is positive
            new_probs_pos_curr, new_joint = qmr.compute_disease_probs(
                pos_findings + [(finding, o)], neg_findings, normalize=True)
            p_pos = new_joint / old_joint

            # When the inquired finding option is negative
            new_probs_neg_curr, new_joint = qmr.compute_disease_probs(
                pos_findings, neg_findings + [(finding, o)], normalize=True)
            p_neg = new_joint / old_joint

            option_utility = (
                p_pos * self.util_func(new_probs_pos_curr, old_probs) + p_neg * self.util_func(new_probs_neg_curr, old_probs)
            )
            option_results.append(option_utility)

        utility = max(option_results)

        return utility, None, None


    def act(self, pos_findings=None, neg_findings=None, candidate_diseases=None):
        action, _, _ = self.act_recursive(self.qmr, self.search_depth - 1, pos_findings, neg_findings, candidate_diseases)
        return action


    def act_recursive(self, qmr, depth, pos_findings=None, neg_findings=None, candidate_diseases=None):
        
        if pos_findings is None or neg_findings is None:
            pos_findings = qmr.pos_findings
            neg_findings = qmr.neg_findings

        # compute valid candidate findings (for categorical or multi-choice, the pos/neg findings is in the form (finding, value)
        candidate_findings = qmr.get_candidate_finding_index(candidate_diseases)
        tmp_pos_findings = [a[0] if isinstance(a, tuple) else a for a in pos_findings]
        tmp_neg_findings = [a[0] if isinstance(a, tuple) else a for a in neg_findings]
        candidate_findings -= set(tmp_pos_findings + tmp_neg_findings)

        old_probs, old_joint = qmr.compute_disease_probs(pos_findings, neg_findings, normalize=True)
        if isinstance(old_probs, (float, int)) and old_joint == 0:
            import json
            all_evi = "\n".join([f"({i}: {a})" for i,a in enumerate(qmr.all_findings)])
            all_options = json.dumps(qmr.finding_info["finding_option_2_idx"], indent=4)
            print(f"disease: {qmr.disease} \n\n pos_findings: {pos_findings} \n\n neg_findings: {neg_findings}  \n\n evidences:\n {all_evi}  \n\n options:\n {all_options} ")
        
        max_utility = 0.0
        action = None
        new_probs_pos = None
        new_probs_neg = None
        for finding in candidate_findings:
        
            finding_type = qmr.get_finding_type(finding)
            
            if finding_type == "B":
                utility, new_probs_pos, new_probs_neg = self.get_binary_finding_utility(finding, qmr, depth, old_probs, old_joint, pos_findings, neg_findings)
            elif finding_type == "C":
                utility, new_probs_pos, new_probs_neg = self.get_categorical_finding_utility(finding, qmr, depth, old_probs, old_joint, pos_findings, neg_findings)
            elif finding_type == "M":
                utility, new_probs_pos, new_probs_neg = self.get_multichoice_finding_utility(finding, qmr, depth, old_probs, old_joint, pos_findings, neg_findings)
            else:
                raise NotImplementedError(f'Finding Type {finding_type} is not defined. Must be one of (B, C, M).')

            if utility > max_utility:
                max_utility = utility
                action = finding

        if max_utility < self.threshold or action is None:
            action = qmr.n_all_findings
        return action, new_probs_pos, new_probs_neg


    def run(self):
        n_correct = [0, 0, 0]
        total_steps = 0
        if self.qmr.test_data is None:
            test_size = self.args.test_size
        else:
            test_size = len(self.qmr.test_data)

        for i in tqdm(range(test_size)):
            if self.qmr.test_data is None:
                self.qmr.reset()
            else:
                self.qmr.reset(i)
            for step in range(self.max_episode_len):
                action = self.act()

                if action == self.qmr.n_all_findings:
                    break
                self.qmr.step(action)

            diag, correctness = self.qmr.inference()
            # print(f" Disesase: {self.qmr.disease} - diagDisease {diag[self.qmr.disease]} - Diag: {np.where(diag>0)}")
            n_correct = [i + j for i, j in zip(n_correct, correctness)]
            total_steps += step + 1
        accuracy = [i / test_size for i in n_correct]
        logger.info(
            f'dataset: {self.args.dataset_name.lower()}, max_episode_len: {self.max_episode_len}, threshold: {self.threshold}\n#experiments: {test_size}; accuracy: {accuracy}; average steps: {total_steps/test_size:.4f}')
        print(
            f'dataset: {self.args.dataset_name.lower()}, max_episode_len: {self.max_episode_len}, threshold: {self.threshold}\n#experiments: {test_size}; accuracy: {accuracy}; average steps: {total_steps/test_size:.4f}')


def job(pack):
    args, max_episode_len, threshold = pack
    bed = BED(args)
    bed.max_episode_len = max_episode_len
    bed.threshold = threshold
    bed.run()


def param_search(args):
    logger.info("Parameter search:")
    pack = [(args, l, t)
            for l, t in itertools.product((20, 15, 10), (0.01, 0.05, 0.1))]
    with Pool() as p:
        p.map(job, pack)
