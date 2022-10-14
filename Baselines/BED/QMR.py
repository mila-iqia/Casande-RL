import ast
import json
import random
from collections import Counter
import itertools
import numpy as np

from data_utils import load_data

class QMR:
    def __init__(self, config):
        # Set arguments
        self.args = config['args']
        self.restrict_flag = True if (not ("no_patho_restriction" in self.args) or self.args.no_patho_restriction == False) else False
        self.max_episode_len = self.args.max_episode_len

        # Set random seed
        random.seed(2021)
        np.random.seed(2021)

        # Load dataset
        data_info = load_data(self.args)
        self.all_diseases, self.all_findings, self.finding2disease, self.disease2finding, self.p_d, self.test_data, self.finding_info = data_info
        self.n_all_findings = len(self.finding2disease)
        self.n_all_diseases = len(self.disease2finding)

    def one_disease_sample(self, floor_prob=0.0):
        findings = set()
        while True:
            disease = random.randrange(self.n_all_diseases)
            for finding, prob in self.disease2finding[disease].items():
                if random.random() <= prob:
                    findings.add(finding)
            if floor_prob > 0:
                for f in range(self.n_all_findings):
                    if random.random() < floor_prob:
                        findings.add(f)
            if len(findings) != 0:
                break

        first_finding = random.choice(list(findings))
        return disease, findings, first_finding

    def update_candidate_diseases(self, finding):
        if self.restrict_flag:
            self.candidate_diseases = [
                d for d in self.candidate_diseases if finding in self.disease2finding[d]]
        else:
            tmp = set(self.candidate_diseases + [d for d in self.disease2finding if finding in self.disease2finding[d]])
            self.candidate_diseases = list(tmp)

    def get_finding_type(self, finding_idx):
        if self.finding_info is None:
            return "B" # default for binary findings
        fingind_name = self.all_findings[finding_idx]
        return self.finding_info["finding_type_and_default"][fingind_name]["data_type"]

    def is_finding_atcd(self, finding_idx):
        if self.finding_info is None:
            return False
        fingind_name = self.all_findings[finding_idx]
        return self.finding_info["finding_type_and_default"][fingind_name]["is_antecedent"]

    def get_disease_severity_mask(self):
        if self.finding_info is None:
            return None
        return self.finding_info["disease_severity_mask"]

    def get_finding_options(self, finding_idx):
        if self.finding_info is None:
            return None # default for binary findings
        fingind_name = self.all_findings[finding_idx]
        dt = self.finding_info["finding_type_and_default"][fingind_name]["data_type"]
        if dt == "B":
            return None # default for binary findings
        else:
            return list(range(len(self.finding_info["finding_option_2_idx"][finding_idx])))

    def get_candidate_finding_index(self, candidate_diseases=None):
        if candidate_diseases is None:
            candidate_diseases = self.candidate_diseases
        return set(itertools.chain.from_iterable(
            [self.disease2finding[d] for d in candidate_diseases]))

    def reset(self, i=None):
        # For simulation data
        if i is None:
            self.disease, self.findings, first_finding = self.one_disease_sample()
            self.pos_findings = [first_finding]
            self.neg_findings = []
            self.candidate_diseases = self.finding2disease[first_finding]
        # For  `real` data
        elif not self.args.dataset_name.lower().startswith("ddxplus"):
            case = self.test_data[i]
            self.disease = case['disease_tag']
            self.findings = set(
                f for f, b in case['goal']['implicit_inform_slots'].items() if b)

            # Get pos_findings and neg_findings
            self.pos_findings = []
            self.neg_findings = []
            for f, b in case['goal']['explicit_inform_slots'].items():
                if b:
                    self.pos_findings.append(f)
                    self.findings.add(f)
                else:
                    self.neg_findings.append(f)
            # Get candidate_diseases
            for i, f in enumerate(self.pos_findings):
                if i == 0:
                    self.candidate_diseases = self.finding2disease[f]
                else:
                    self.update_candidate_diseases(f)
        else:
            self.disease = self.test_data.iloc[i]["PATHOLOGY"]
            differential = self.test_data.iloc[i].get("DIFFERENTIAL_DIAGNOSIS", None)
            differential = ast.literal_eval(differential) if isinstance(differential, str) else differential
            evidences = self.test_data.iloc[i]["EVIDENCES"]
            evidences = ast.literal_eval(evidences) if isinstance(evidences, str) else evidences
            current_set_cat_mult = set()
            options = {}
            present_symptoms = set()
            present_atcds = set()
            present_evidences = set()
            for e in evidences:
                if (("_@_" in e)):
                    idx = e.find("_@_")
                    b = e[:idx]
                    elem_val = e[idx + 3 :]
                    current_set_cat_mult.add(b)
                    evi_idx = self.finding_info['finding_name_2_idx'][b]
                    dt = self.finding_info['finding_type_and_default'][b]['data_type']
                    default_val = self.finding_info['finding_type_and_default'][b]['default_value']
                    is_antecedent = self.finding_info['finding_type_and_default'][b]['is_antecedent']
                    elem_val_idx = self.finding_info['finding_option_2_idx'][evi_idx].get(elem_val, -1)
                    if dt == 'M':
                        l = options.get(b, set())
                        l.add(elem_val_idx)
                        options[evi_idx] = l
                    else:
                        options[evi_idx] = elem_val_idx
                    if not(str(default_val) == str(elem_val)):
                        present_evidences.add(evi_idx)
                        if is_antecedent:
                            present_atcds.add(evi_idx)
                        else:
                            present_symptoms.add(evi_idx)
                else:
                    evi_idx = self.finding_info['finding_name_2_idx'][e]
                    options[evi_idx] = True
                    present_evidences.add(evi_idx)
                    is_antecedent = self.finding_info['finding_type_and_default'][e]['is_antecedent']
                    if is_antecedent:
                        present_atcds.add(evi_idx)
                    else:
                        present_symptoms.add(evi_idx)
            missing_set_cat_mult = self.finding_info["multi_categorical_findings_per_patho"][self.disease] - current_set_cat_mult
            myset = set(evidences)
            for e in missing_set_cat_mult:
                evi_idx = self.finding_info['finding_name_2_idx'][e]
                def_value = self.finding_info['finding_type_and_default'][e].get('default_value', '')
                def_value_idx = self.finding_info['finding_option_2_idx'][evi_idx].get(def_value, -1)
                dt = self.finding_info['finding_type_and_default'][e]['data_type']
                options[evi_idx] = set([def_value_idx]) if dt == "M" else def_value_idx
            self.findings = options
            first_finding = self.test_data.iloc[i]["INITIAL_EVIDENCE"]
            first_finding_idx = self.finding_info['finding_name_2_idx'][first_finding]
            self.pos_findings = [first_finding_idx]
            self.neg_findings = []
            self.candidate_diseases = list(self.finding2disease[first_finding_idx])
            self.disease = self.finding_info['disease_name_2_idx'][self.disease]
            self.differential = None if differential is None else [[self.finding_info['disease_name_2_idx'][a[0]], a[1]] for a in differential]
            self.present_symptoms = present_symptoms
            self.present_atcds = present_atcds
            self.present_evidences = present_evidences
            pass

    def step(self, action):
        if action in self.findings:
            if isinstance(self.findings, set):
                self.pos_findings.append(action)
            else:
                if isinstance(self.findings[action], bool): # binary
                    self.pos_findings.append(action)
                elif not isinstance(self.findings[action], set): # categorical
                    self.pos_findings.append((action, self.findings[action]))
                else: # multi-choice
                    for idx in self.findings[action]:
                        self.pos_findings.append((action, idx))
            self.update_candidate_diseases(action)
        else:
            if isinstance(self.findings, set):
                self.neg_findings.append(action)
            else:
                e = self.all_findings[action]
                dt = self.finding_info['finding_type_and_default'][e]['data_type']
                if dt == "B": # binary
                    self.neg_findings.append(action)
                else: # multi-choice or categorical
                    def_value = self.finding_info['finding_type_and_default'][e].get('default_value', '')
                    def_value_idx = self.finding_info['finding_option_2_idx'][action].get(def_value, -1)
                    self.neg_findings.append((action, def_value_idx))

    def inference(self, pos_findings=None, neg_findings=None):
        disease_probs, _ = self.compute_disease_probs(
            pos_findings=pos_findings, neg_findings=neg_findings, normalize=True)
        top5 = disease_probs.argsort()[-5:][::-1]
        return disease_probs, (self.disease == top5[0], self.disease in top5[:3], self.disease in top5)

    def compute_disease_probs(self, pos_findings=None, neg_findings=None, normalize=False):
        """ Make diagnosis prediction given current observed findings """
        if pos_findings is None:
            pos_findings = self.pos_findings
        if neg_findings is None:
            neg_findings = self.neg_findings

        default_option_proba = 1e-7

        p_f_d = np.empty(self.n_all_diseases)
        for disease in range(self.n_all_diseases):
            prob = 1.0
            f4d = self.disease2finding[disease]

            # Compute negative findings first then the positives
            for finding in neg_findings:
                if isinstance(finding, tuple):
                    evi_idx, option = finding
                else:
                    evi_idx, option = finding, None
                finding_proba = f4d.get(evi_idx, 0) if option is None else f4d.get(evi_idx, {}).get(option, default_option_proba)
                prob *= 1 - finding_proba

            for finding in pos_findings:
                if isinstance(finding, tuple):
                    evi_idx, option = finding
                else:
                    evi_idx, option = finding, None
                if evi_idx in f4d:
                    finding_proba = f4d[evi_idx] if option is None else f4d[evi_idx].get(option, default_option_proba)
                    if finding_proba == 0.0:
                        prob = 0.0
                        break
                    else:
                        prob *= finding_proba
                else:
                    prob = 0.0
                    break
            p_f_d[disease] = prob
        disease_probs = p_f_d * self.p_d

        if normalize:
            joint_prob = sum(disease_probs)
            if joint_prob == 0.0:                
                return 0, 0
            else:
                disease_probs /= joint_prob
                return disease_probs, joint_prob
        else:
            return disease_probs
