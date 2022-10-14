import logging
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
import itertools
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import json
import ast
# from copy import deepcopy, copy

from bayesian_experimental_design import BED

logger = logging.getLogger(__name__)

def write_json(data, fp):
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)

def get_average_state_from_percent(data, percent=0.0, end_percent=1.0, normalize=True):
    """Get the average stat from a data sequence starting at a given percent.
    Parameters
    ----------
    data:  sequence
        the provided data.
    percent: float, list, tuple
        the provided percentage. Default: 0.0.
    end_percent: float, list, tuple
        the provided end percentage. Default: 1.0.
    normalize: boolean
        whether to normalize
    Returns
    -------
    result: float, list
        the computed average stat.
    """
    assert not (percent is None and end_percent is None)
    if percent is None:
        percent = end_percent
    elif end_percent is None:
        end_percent = percent
    assert isinstance(percent, (int, float, list, tuple))
    assert isinstance(end_percent, (int, float, list, tuple))
    extract_flag = False
    if isinstance(percent, (int, float)) and isinstance(end_percent, (int, float)):
        percent = [percent]
        end_percent = [end_percent]
        extract_flag = True
    elif isinstance(percent, (int, float)):
        percent = [percent] * len(end_percent)
    elif isinstance(end_percent, (int, float)):
        end_percent = [end_percent] * len(percent)
    assert len(percent) == len(end_percent)
    length = len(data)
    result = []
    for b, e in zip(percent, end_percent):
        assert b >= 0.0 and b <= 1.0
        assert e >= 0.0 and e <= 1.0
        assert e >= b
        i = int(length * b)
        j = int(length * e)
        j = j + 1 if j == i else j
        j = min(length, j)
        i = i - 1 if i == length else i
        n = j - i
        r = sum(data[i:j])
        r = r / max(1, n) if normalize else r
        result.append(r)
    if len(result) == 1 and extract_flag:
        result = result[0]
    return result

def compute_severity_stats(preds, gt_diff, severity_mask, diff_proba_threshold):
    if severity_mask is None:
        severity_mask = [0] * preds.shape[-1]
    gt_diff = gt_diff.reshape(1, -1)
    severity_mask= np.array(severity_mask).reshape(1, -1)
    gt_severity_mask = ((gt_diff > diff_proba_threshold) * severity_mask).astype(bool)
    pr_severity_mask = ((preds > diff_proba_threshold) * severity_mask).astype(bool)
    # mask of severe patho in pred not in gt
    pr_gt_nonshared_severity_mask = (np.logical_not(gt_severity_mask) * pr_severity_mask).astype(bool)
    # mask of severe patho in gt not in pred
    gt_pr_nonshared_severity_mask = (np.logical_not(pr_severity_mask) * gt_severity_mask).astype(bool)
    nb_out = pr_gt_nonshared_severity_mask.sum(-1)
    nb_in = gt_pr_nonshared_severity_mask.sum(-1)
    gt_nbSev = gt_severity_mask.astype(int).sum(-1)
    max_sev = severity_mask.sum()
    pred_no_gt = (max_sev - gt_nbSev - nb_out) / np.maximum(1, max_sev - gt_nbSev)
    gt_no_pred = (gt_nbSev - nb_in) / np.maximum(1, gt_nbSev)
    gt_pred_f1 = compute_f1(pred_no_gt, gt_no_pred)
    return pred_no_gt, gt_no_pred, gt_pred_f1
    
    

def kl_trajectory_auc(kl_explore, kl_confirm):
    kl_explore = np.array(kl_explore)
    kl_confirm = np.array(kl_confirm)
    result = np.trapz(kl_explore, kl_confirm)
    return result

def kl_confirm_score(probas, dist, c=1):
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    dist = dist.reshape(1, -1)
    cross_entropy = -np.sum(dist * np.log(probas + 1e-10), axis=1)
    kl_val = np.maximum(0, cross_entropy - entropy)
    kl_val = np.exp(-c * kl_val)
    return kl_val


def kl_explore_score(probas, first_proba=None, c=1):
    dist = probas[0:1] if first_proba is None else first_proba
    entropy = -np.sum(dist * np.log(dist + 1e-10), axis=-1)
    dist = dist.reshape(1, -1) if len(dist.shape) == 1 else dist
    cross_entropy = -np.sum(dist * np.log(probas + 1e-10), axis=-1)
    kl_val = np.maximum(0, cross_entropy - entropy)
    kl_val = np.exp(-c * kl_val)
    kl_val = 1.0 - kl_val
    return kl_val
    
def kl_trajectory_score(kl_explore, kl_confirm):
    kl_explore = np.array(kl_explore)
    kl_confirm = np.array(kl_confirm)
    if len(kl_explore) > 1:
        alphas = np.arange(start=len(kl_explore) - 1, stop=-1, step=-1)
        alphas = alphas / (len(kl_explore) - 1)
    else:
        alphas = 0.5
    score = alphas * kl_explore + (1 - alphas) * kl_confirm
    return score

def compute_f1(p, r):
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    if isinstance(r, (list, tuple)):
        r = np.array(r)
    denom = p + r
    return (2 * p * r) / (denom + 1e-10)

def eval_update_candidate_diseases(finding, candidate_diseases, disease2finding, restrict_flag=True):
    if restrict_flag:
        return [d for d in candidate_diseases if finding in disease2finding[d]]
    else:
        return list(set(candidate_diseases + [d for d in disease2finding if finding in disease2finding[d]]))


def eval_step(action, findings, pos_findings, neg_findings, candidate_diseases, finding_info, all_findings, disease2finding, restrict_flag=True):
    if action in findings:
        if isinstance(findings, set):
            pos_findings.append(action)
        else:
            if isinstance(findings[action], bool): # binary
                pos_findings.append(action)
            elif not isinstance(findings[action], set): # categorical
                pos_findings.append((action, findings[action]))
            else: # multi-choice
                for idx in findings[action]:
                    pos_findings.append((action, idx))
        candidate_diseases[:] = eval_update_candidate_diseases(action, candidate_diseases, disease2finding, restrict_flag)
    else:
        if isinstance(findings, set):
            neg_findings.append(action)
        else:
            e = all_findings[action]
            dt = finding_info['finding_type_and_default'][e]['data_type']
            if dt == "B": # binary
                neg_findings.append(action)
            else: # multi-choice or categorical
                def_value = finding_info['finding_type_and_default'][e].get('default_value', '')
                def_value_idx = finding_info['finding_option_2_idx'][action].get(def_value, -1)
                neg_findings.append((action, def_value_idx))
                
def eval_reset(raw_data, qmr):
    disease = raw_data["PATHOLOGY"]
    differential = raw_data.get("DIFFERENTIAL_DIAGNOSIS", None)
    differential = ast.literal_eval(differential) if isinstance(differential, str) else differential
    evidences = raw_data["EVIDENCES"]
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
            evi_idx = qmr.finding_info['finding_name_2_idx'][b]
            dt = qmr.finding_info['finding_type_and_default'][b]['data_type']
            default_val = qmr.finding_info['finding_type_and_default'][b]['default_value']
            is_antecedent = qmr.finding_info['finding_type_and_default'][b]['is_antecedent']
            elem_val_idx = qmr.finding_info['finding_option_2_idx'][evi_idx].get(elem_val, -1)
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
            evi_idx = qmr.finding_info['finding_name_2_idx'][e]
            options[evi_idx] = True
            present_evidences.add(evi_idx)
            is_antecedent = qmr.finding_info['finding_type_and_default'][e]['is_antecedent']
            if is_antecedent:
                present_atcds.add(evi_idx)
            else:
                present_symptoms.add(evi_idx)
    missing_set_cat_mult = qmr.finding_info["multi_categorical_findings_per_patho"][disease] - current_set_cat_mult
    myset = set(evidences)
    for e in missing_set_cat_mult:
        evi_idx = qmr.finding_info['finding_name_2_idx'][e]
        def_value = qmr.finding_info['finding_type_and_default'][e].get('default_value', '')
        def_value_idx = qmr.finding_info['finding_option_2_idx'][evi_idx].get(def_value, -1)
        dt = qmr.finding_info['finding_type_and_default'][e]['data_type']
        options[evi_idx] = set([def_value_idx]) if dt == "M" else def_value_idx
    findings = options
    first_finding = raw_data["INITIAL_EVIDENCE"]
    first_finding_idx = qmr.finding_info['finding_name_2_idx'][first_finding]
    pos_findings = [first_finding_idx]
    neg_findings = []
    candidate_diseases = list(qmr.finding2disease[first_finding_idx])
    disease = qmr.finding_info['disease_name_2_idx'][disease]
    differential = None if differential is None else [[qmr.finding_info['disease_name_2_idx'][a[0]], a[1]] for a in differential]
    output_data = {
        "findings": findings,
        "present_symptoms": present_symptoms,
        "present_evidences": present_evidences,
        "present_atcds": present_atcds,
        "pos_findings": pos_findings,
        "neg_findings": neg_findings,
        "candidate_diseases": candidate_diseases,
        "disease": disease,
        "differential": differential,
    }
    return output_data
    
def compute_metrics(differential, disease, all_diags, present_evidences, present_symptoms, present_atcds, inquired_evidences, inquired_symptoms, inquired_atcds, severity_mask, tres=0.01):
        if differential is not None:
            differential.sort(key=lambda x: x[1], reverse=True)
        gt_diff = [disease] if differential is None else [a[0] for a in differential if a[1] > tres]
        gt_diff_proba = [0.0] * len(all_diags[-1])
        if differential is not None:
            for a in differential:
                gt_diff_proba[a[0]] = a[1]
        else:
            gt_diff_proba[disease] = 1.0

        top5 = all_diags[-1].argsort()[-5:][::-1]
        final_pred = list(zip(range(len(all_diags[-1])), all_diags[-1].tolist()))
        final_pred.sort(key=lambda x: x[1], reverse=True)
        final_pred_diff = [a[0] for a in final_pred if a[1] > tres]

        result = {}
        result["IL"] = len(all_diags)
        result["GTPA"] = all_diags[-1][disease] > tres
        result["GTPA@1"] = disease == top5[0] and all_diags[-1][disease] > tres
        result["GTPA@3"] = disease in top5[:3] and all_diags[-1][disease] > tres
        result["GTPA@5"] = disease in top5 and all_diags[-1][disease] > tres

        result[f"DDR"] = len(set(gt_diff).intersection(final_pred_diff)) / max(1, len(set(gt_diff)))
        result[f"DDP"] = len(set(gt_diff).intersection(final_pred_diff)) / max(1, len(set(final_pred_diff)))
        result[f"DDF1"] = compute_f1(result[f"DDP"], result[f"DDR"])
        for k in [1, 3, 5]:
            result[f"DDR@{k}"] = len(set(gt_diff[:k]).intersection(final_pred_diff[:k])) / max(1, len(set(gt_diff[:k])))
            result[f"DDP@{k}"] = len(set(gt_diff[:k]).intersection(final_pred_diff[:k])) / max(1, len(set(final_pred_diff[:k])))
            result[f"DDF1@{k}"] = compute_f1(result[f"DDP@{k}"], result[f"DDR@{k}"])

        result["PER"] = len(present_evidences.intersection(inquired_evidences)) / max(1, len(present_evidences))
        result["PEP"] = len(present_evidences.intersection(inquired_evidences)) / max(1, len(inquired_evidences))
        result["PEF1"] = compute_f1(result["PEP"], result["PER"])

        result["PSR"] = len(present_symptoms.intersection(inquired_symptoms)) / max(1, len(present_symptoms))
        result["PSP"] = len(present_symptoms.intersection(inquired_symptoms)) / max(1, len(inquired_symptoms))
        result["PSF1"] = compute_f1(result["PSP"], result["PSR"])

        result["PAR"] = len(present_atcds.intersection(inquired_atcds)) / max(1, len(present_atcds))
        result["PAP"] = len(present_atcds.intersection(inquired_atcds)) / max(1, len(inquired_atcds))
        result["PAF1"] = compute_f1(result["PAP"], result["PAR"])
        
        all_diags = np.array(all_diags)
        mini_prob = np.amin(all_diags, axis=0)
        maxi_prob = np.amax(all_diags, axis=0)    
        result["TVD"] = np.absolute(maxi_prob - mini_prob).mean()
        gt_diff_proba = np.array(gt_diff_proba)        
        confirm_score = kl_confirm_score(all_diags, gt_diff_proba)
        explore_score = kl_explore_score(all_diags)
        succesive_explore_score = kl_explore_score(all_diags[1:], first_proba=all_diags[0:-1])
        result["AUCTraj"] = kl_trajectory_auc(explore_score, confirm_score)
        kl_trajectory_values = kl_trajectory_score(explore_score, confirm_score)
        result["TrajScore"] = np.mean(kl_trajectory_values)

        pred_no_gt, gt_no_pred, gt_pred_f1 = compute_severity_stats(all_diags, gt_diff_proba, severity_mask, tres)
        result["DSP"] = pred_no_gt[-1]
        result["DSR"] = gt_no_pred[-1]
        result["DSF1"] = gt_pred_f1[-1]

        p = list(range(0, 105, 5))
        # p_idx = {v: i for i, v in enumerate(p)}
        p = [v / 100.0 for v in p]
        
        explore_score = np.array(get_average_state_from_percent(explore_score, percent=p, end_percent=None))
        succesive_explore_score = np.array(get_average_state_from_percent(succesive_explore_score, percent=p, end_percent=None))
        confirm_score = np.array(get_average_state_from_percent(confirm_score, percent=p, end_percent=None))
        kl_trajectory_values = np.array(get_average_state_from_percent(kl_trajectory_values, percent=p, end_percent=None))
        pred_no_gt = np.array(get_average_state_from_percent(pred_no_gt, percent=p, end_percent=None))
        gt_no_pred = np.array(get_average_state_from_percent(gt_no_pred, percent=p, end_percent=None))
        gt_pred_f1 = np.array(get_average_state_from_percent(gt_pred_f1, percent=p, end_percent=None))

        result["PlotData.x"] = np.array(p)
        result["PlotData.Exploration"] = explore_score
        result["PlotData.SuccessiveExploration"] = succesive_explore_score
        result["PlotData.Confirmation"] = confirm_score
        result["PlotData.Trajectory"] = kl_trajectory_values
        result["PlotData.SevF1"] = gt_pred_f1
        result["PlotData.SevPrecOut"] = pred_no_gt
        result["PlotData.SevRecIn"] = gt_no_pred
        
        return result
        

def aggregate_metrics(metrics):
    result = {}
    all_action_list = []
    all_diff_list = []
    for m in metrics:
        all_action_list.append(m.pop("action_list", [-1]))
        all_diff_list.append(m.pop("diff_list", [[-1]]))
        result = {a: result.get(a, 0) + m[a] for a in m.keys()}
    result = {a: result[a] / max(1, len(metrics)) for a in result.keys()}
    result = {a: result[a].tolist() if hasattr(result[a], "tolist") else result[a] for a in result.keys()}
    return result, all_action_list, all_diff_list


def save_action_list(ds_action_list, all_findings, action_fp=None):
    if action_fp is None:
        return
    print(len(ds_action_list))
    ds_action_list = np.array(ds_action_list).astype('int')
    print(ds_action_list.shape)
    df = pd.DataFrame(ds_action_list, columns = [str(i) for i in range(ds_action_list.shape[1])])
    df = df.applymap(lambda x: 'None' if x == -1 else all_findings[x])
    df.to_csv(action_fp,  sep=',', index=False)

def get_litteral_diff(pred, name_map, name_flag, tres):
    pred = np.array(pred)
    pred_diff_mask = (pred > tres)
    num = pred_diff_mask.sum()
    if num == 0:
        return []
    pred = pred * pred_diff_mask
    top_ind = np.argsort(pred, axis=-1)[::-1]
    klist = top_ind[:num]
    if name_flag:
        result = [name_map[i] for i in klist]
    else:
        result = klist.tolist()
    return result
        
def save_diff_list(ds_diff_list, all_diseases, diff_fp=None):
    if diff_fp is None:
        return
    ds_diff_list = np.array(ds_diff_list).transpose(0,2,1)
    df2 = pd.DataFrame(ds_diff_list.tolist(), columns = [str(i) for i in range(ds_diff_list.shape[1])])
    df2 = df2.applymap(lambda x: get_litteral_diff(x, all_diseases, True, tres=0.01))
    df2.to_csv(diff_fp,  sep=',', index=False)


class BEDBatch(BED):
    def __init__(self, args):
        super().__init__(args)
        
    def batch_run(self, raw_data, max_episode_len):
        assert self.search_depth <= 1, "not implemented for recursive act"
        output_data = eval_reset(raw_data, self.qmr)
        findings = output_data["findings"]
        present_symptoms = output_data["present_symptoms"]
        present_evidences = output_data["present_evidences"]
        present_atcds = output_data["present_atcds"]
        pos_findings = output_data["pos_findings"]
        neg_findings = output_data["neg_findings"]
        candidate_diseases = output_data["candidate_diseases"]
        disease = output_data["disease"]
        differential = output_data["differential"]
        
        action_list = [-1] * (max_episode_len + 1)
        action_list[0] = pos_findings[0]
        
        diff_list = np.ones((self.qmr.n_all_diseases, max_episode_len + 1)) * (-1)

        n_all_findings = len(self.qmr.finding2disease)
        all_diags = []
        diag, _ = self.qmr.compute_disease_probs(pos_findings, neg_findings, normalize=True)
        diff_list[:, 0] = diag
        all_diags.append(diag)
        inquired_evidences = set()
        inquired_symptoms = set()
        inquired_atcds = set()
        for action in pos_findings:
            inquired_evidences.add(action)
            if self.qmr.is_finding_atcd(action):
                inquired_atcds.add(action)
            else:
                inquired_symptoms.add(action)
        for step in range(max_episode_len):
            action = self.act(pos_findings, neg_findings, candidate_diseases)
            if action == n_all_findings:
                break
            inquired_evidences.add(action)
            action_list[step + 1] = action
            if self.qmr.is_finding_atcd(action):
                inquired_atcds.add(action)
            else:
                inquired_symptoms.add(action)
            eval_step(action, findings, pos_findings, neg_findings, candidate_diseases, self.qmr.finding_info, self.qmr.all_findings, self.qmr.disease2finding, self.qmr.restrict_flag)
            diag, _ = self.qmr.compute_disease_probs(pos_findings, neg_findings, normalize=True)
            diff_list[:, step + 1] = diag
            all_diags.append(diag)

        tres = 0.01
        severity_mask = self.qmr.get_disease_severity_mask()

        metrics_dict = compute_metrics(
            differential, disease, all_diags, present_evidences, present_symptoms, present_atcds, inquired_evidences, inquired_symptoms, inquired_atcds, severity_mask, tres
        )
        metrics_dict["action_list"] = action_list
        metrics_dict["diff_list"] = diff_list
        return metrics_dict
        


    def run(self):
        n_correct = [0, 0, 0]
        total_steps = 0
        if self.qmr.test_data is None:
            test_size = self.args.test_size
        else:
            test_size = len(self.qmr.test_data)
            
        if True:
            df = self.qmr.test_data
            self.qmr.test_data = None
            result, all_action_list, all_diff_list = aggregate_metrics(df.apply(lambda raw_data: self.batch_run(raw_data, self.max_episode_len), axis="columns").to_list())
            write_json(result, f"BedMetrics_{self.args.dataset_name.lower()}_{self.max_episode_len}_{self.threshold}.json")
            logger.info(
                f'dataset: {self.args.dataset_name.lower()}, max_episode_len: {self.max_episode_len}, threshold: {self.threshold}\n#experiments: {test_size}; Metrics: \n\n {result}'
            )
            save_action_list(all_action_list, self.qmr.all_findings, f"BedActions_{self.args.dataset_name.lower()}_{self.max_episode_len}_{self.threshold}.csv")
            save_diff_list(all_diff_list, self.qmr.all_diseases, f"BedDifferentials_{self.args.dataset_name.lower()}_{self.max_episode_len}_{self.threshold}.csv")
            return

        end_result_metrics = {}
        all_action_list = []
        for i in tqdm(range(test_size)):
            if self.qmr.test_data is None:
                self.qmr.reset()
            else:
                self.qmr.reset(i)

            all_diags = []
            diag, _ = self.qmr.inference()
            all_diags.append(diag)
            inquired_evidences = set()
            inquired_symptoms = set()
            inquired_atcds = set()
            action_list = [-1] * (self.max_episode_len + 1)
            action_list[0] = self.qmr.pos_findings[0]
            for action in self.qmr.pos_findings:
                inquired_evidences.add(action)
                if self.qmr.is_finding_atcd(action):
                    inquired_atcds.add(action)
                else:
                    inquired_symptoms.add(action)
            for step in range(self.max_episode_len):
                action = self.act()
                if action == self.qmr.n_all_findings:
                    break
                inquired_evidences.add(action)
                action_list[step + 1] = action
                if self.qmr.is_finding_atcd(action):
                    inquired_atcds.add(action)
                else:
                    inquired_symptoms.add(action)
                self.qmr.step(action)
                diag, _ = self.qmr.inference()
                all_diags.append(diag)

            a_metric = compute_metrics(
                self.qmr.differential, self.qmr.disease, all_diags, self.qmr.present_evidences, self.qmr.present_symptoms, self.qmr.present_atcds,
                inquired_evidences, inquired_symptoms, inquired_atcds, self.qmr.get_disease_severity_mask(), tres=0.01
            )
            end_result_metrics = {a: end_result_metrics.get(a, 0) + a_metric[a] for a in a_metric.keys()}
            all_action_list.append(action_list)
        end_result_metrics = {a: end_result_metrics[a] / test_size for a in end_result_metrics.keys()}
        end_result_metrics = {a: end_result_metrics[a].tolist() if hasattr(end_result_metrics[a], "tolist") else end_result_metrics[a] for a in end_result_metrics.keys()}
        logger.info(
            f'dataset: {self.args.dataset_name.lower()}, max_episode_len: {self.max_episode_len}, threshold: {self.threshold}\n#experiments: {test_size}; Metrics: \n\n {end_result_metrics}')
        print(
            f'dataset: {self.args.dataset_name.lower()}, max_episode_len: {self.max_episode_len}, threshold: {self.threshold}\n#experiments: {test_size}; Metrics: \n\n {end_result_metrics}')
        write_json(end_result_metrics, f"BedMetrics_{self.args.dataset_name.lower()}_{self.max_episode_len}_{self.threshold}.json")
        save_action_list(all_action_list, self.qmr.all_findings, f"BedActions_{self.args.dataset_name.lower()}_{self.max_episode_len}_{self.threshold}.csv")

