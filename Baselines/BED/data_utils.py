from collections import defaultdict, Counter
import copy
import logging
import numpy as np
import pickle
import json
import os
import ast
import pandas as pd

from ddxplus_utils import generateBayesianInferenceJsonStats4

logger = logging.getLogger(__name__)


def _HPO_postprocess(data):
    for disease, findings in data.items():
        for finding in findings:
            if isinstance(finding[1], list):
                finding[1] = finding[1][1]


def load_graph_data(args):
    dataset_name = 'SymCAT' if args.dataset_name.lower() == 'symcat' else 'HPO'
    n_diseases = args.n_diseases
    dirname = os.path.dirname(__file__)
    with open(f'{dirname}/{args.input_dir}/{dataset_name}.json') as f:
        data = json.load(f)
    if n_diseases is not None:
        with open(f'{dirname}/{args.input_dir}/disease{n_diseases}_{dataset_name}.txt', 'r') as f:
            lines = f.read().splitlines()
    else:
        lines = list(data)
        n_diseases = len(lines)
    data = {key: data[key] for key in lines}
    if dataset_name == 'HPO':
        _HPO_postprocess(data)

    # Post processes
    disease2finding = defaultdict(dict)
    finding2disease = defaultdict(set)
    all_diseases = []
    for d_idx, (patho, findings) in enumerate(data.items()):
        all_diseases.append(patho)
        for finding in findings:
            disease2finding[d_idx][finding[0]] = finding[1]
            finding2disease[finding[0]].add(d_idx)
    all_findings = list(finding2disease)
    assert len(disease2finding) == n_diseases

    for disease, findings in disease2finding.items():
        disease2finding[disease] = {all_findings.index(
            finding): prob for finding, prob in findings.items()}

    finding2disease = {all_findings.index(
        finding): list(diseases) for finding, diseases in finding2disease.items()}

    logger.info(
        f'#diseases: {n_diseases}, #findings: {len(finding2disease)}')

    # Assume that all disease are equally likely to happen
    p_d = np.ones(n_diseases) / n_diseases
    return (all_diseases, all_findings, finding2disease, disease2finding, p_d, None, None)


def load_dialogue_data(args):
    def get_test_data(data, all_findings, all_diseases):
        for case in data:
            case['disease_tag'] = all_diseases.index(case['disease_tag'])
            case['goal']['implicit_inform_slots'] = {
                all_findings.index(f): b for f, b in case['goal']['implicit_inform_slots'].items()
            }
            case['goal']['explicit_inform_slots'] = {
                all_findings.index(f): b for f, b in case['goal']['explicit_inform_slots'].items()
            }
        return data

    def build_graph(data):
        disease2finding = defaultdict(list)
        finding2disease = defaultdict(set)

        disease_case_counts = defaultdict(int)
        for case in data:
            disease = case['disease_tag']
            disease_case_counts[disease] += 1
            for symp, b in case['goal']['implicit_inform_slots'].items():
                if b:
                    disease2finding[disease].append(symp)
                    finding2disease[symp].add(disease)
            for symp, b in case['goal']['explicit_inform_slots'].items():
                if b:
                    disease2finding[disease].append(symp)
                    finding2disease[symp].add(disease)

        for d, fs in disease2finding.items():
            disease2finding[d] = {f: round(count / disease_case_counts[d], 4)
                                  for f, count in Counter(fs).items()}

        return finding2disease, disease2finding

    logger.info(f'{args.dataset_name} dataset')
    dirname = os.path.dirname(__file__)
    if args.dataset_name.lower() == 'dxy':
        path = f'{dirname}/{args.input_dir}/dxy_dataset/dxy_dialog_data_dialog_v2.pickle'
    else:
        path = f'{dirname}/{args.input_dir}/acl2018-mds/acl2018-mds.p'

    data = pickle.load(open(path, 'rb'))
    finding2disease, disease2finding = build_graph(data['train'])
    all_findings, all_diseases = list(finding2disease), list(disease2finding)
    logger.info(
        f'{args.dataset_name}, #diseases:{len(all_diseases)}, #findings:{len(all_findings)}')
    test_data = get_test_data(data['test'], all_findings, all_diseases)

    finding2disease, disease2finding = update_graph(
        finding2disease, disease2finding, all_findings, all_diseases)

    # Compute disease priors
    n_cases = len(data['train'])
    counter = Counter([case['disease_tag'] for case in data['train']])
    p_d = np.asarray([counter[d] / n_cases for d in all_diseases])

    # Assume that all disease are equally likely to happen
    # n_diseases = len(disease2finding)
    # p_d = np.ones(n_diseases) / n_diseases

    return (all_diseases, all_findings, finding2disease, disease2finding, p_d, test_data, None)

def preprocess_symptoms(symptoms):
    # symptoms is a string of list
    if symptoms.startswith("[") and symptoms.endswith("]"):
        return ast.literal_eval(symptoms)
    data = symptoms.split(";")
    result = []
    for x in data:
        name = x.split(":")[0]
        result.append(name)
    # sort result
    result = sorted(result)
    return result

def load_ddxplus_data(args, prefix):
    def clean(data):
        result = data.replace("\r\n", " ")
        result = result.replace("\r", " ")
        result = result.replace("\n", " ")
        result = result.replace(",", " ")
        return result
    logger.info(f'{args.dataset_name} dataset')
    dirname = os.path.dirname(__file__)
    patient_file = f'{dirname}/{args.input_dir}/ddxplus/release_{prefix}_patients.zip'
    train_stat_file = f'{dirname}/{args.input_dir}/ddxplus/ddxplus_bed_stats_4.json'
    findings_file = f'{dirname}/{args.input_dir}/ddxplus/release_evidences.json'
    conditions_file = f'{dirname}/{args.input_dir}/ddxplus/release_conditions.json'
    if not os.path.exists(train_stat_file):
        generateBayesianInferenceJsonStats4(
            conditions_file, findings_file, f'{dirname}/{args.input_dir}/ddxplus/', 'release_train_patients.zip', f'{dirname}/{args.input_dir}/ddxplus/'
        )
    with open(findings_file) as fp:
        findings = json.load(fp)
        for k in findings.keys():
            findings[k]["name"] = clean(findings[k]["name"])
    with open(conditions_file) as fp:
        conds = json.load(fp)
        for k in conds.keys():
            conds[k]["condition_name"] = clean(conds[k]["condition_name"])
        conds = {conds[k]["condition_name"]: conds[k] for k in conds.keys()}
    with open(train_stat_file) as fp:
        train_stats = json.load(fp)
    prior = {d: train_stats[d]["prior"] for d in train_stats.keys()}
    train_stats = {d: train_stats[d]["findings"] for d in train_stats.keys()}
    all_diseases = list(train_stats.keys())
    all_findings = [findings[k]["name"] for k in findings.keys()]
    disease2finding = {}
    finding2disease = defaultdict(set)
    a_disease_per_finding = {}
    for d_idx, d in enumerate(all_diseases):
        disease2finding[d_idx] = copy.deepcopy(train_stats[d])
        for e in train_stats[d].keys():
            e_idx = all_findings.index(e)
            disease2finding[d_idx][e_idx] = disease2finding[d_idx].pop(e)
            finding2disease[e_idx].add(d_idx)
            a_disease_per_finding[e_idx] = d
    df = pd.read_csv(patient_file)
    cols = {}
    if "AGE_BEGIN" in df.columns:
        cols["AGE_BEGIN"] = "AGE"
    if "GENDER" in df.columns:
        cols["GENDER"] = "SEX"
    if "SYMPTOMS" in df.columns:
        cols["SYMPTOMS"] = "EVIDENCES"
    if "INITIAL_SYMPTOM" in df.columns:
        cols["INITIAL_SYMPTOM"] = "INITIAL_EVIDENCE"
    if "GT_DIFF" in df.columns:
        cols["GT_DIFF"] = "DIFFERENTIAL_DIAGNOSIS"
    if len(cols) > 0:
        df = df.rename(columns=cols)
    df["EVIDENCES"] = df["EVIDENCES"].apply(lambda x: preprocess_symptoms(x))
    p_d = np.asarray([prior[d] for d in all_diseases])
    findings = {findings[k]["name"]: findings[k] for k in findings.keys()}
    severity_threshold = 3
    finding_info = {
       "multi_categorical_findings_per_patho": {
           d : set([
              a for a in train_stats[d].keys() if findings[a]["data_type"] in ['M', 'C']
           ])
           for d in train_stats.keys()
       },
       "finding_type_and_default": {
           e: {"data_type": findings[e].get("data_type", 'B'), 'default_value': findings[e].get("default_value", ''), 'is_antecedent': findings[e].get("is_antecedent", False)}
           for e in findings.keys()
       },
       "disease_severity_mask": [conds[a].get("severity", severity_threshold) < severity_threshold for a in all_diseases],
       "finding_name_2_idx": {a: i for i, a in enumerate(all_findings)},
       "disease_name_2_idx": {a: i for i, a in enumerate(all_diseases)},
       "finding_option_2_idx": {
           e_idx : {a: i for i, a in enumerate(sorted(train_stats[a_disease_per_finding[e_idx]][all_findings[e_idx]].keys())) }
           for e_idx in range(len(all_findings))
           if findings[all_findings[e_idx]].get("data_type", 'B') in ['M', 'C']
       }
    }
    for d_idx in disease2finding.keys():
        for e_idx in disease2finding[d_idx].keys():
            if isinstance(disease2finding[d_idx][e_idx], dict):
                disease2finding[d_idx][e_idx] = {finding_info["finding_option_2_idx"][e_idx][k]: disease2finding[d_idx][e_idx][k] for k in disease2finding[d_idx][e_idx].keys()}
    return (all_diseases, all_findings, finding2disease, disease2finding, p_d, df, finding_info)


def update_graph(finding2disease, disease2finding, all_findings, all_diseases):
    """ Convert the symptom and disease to their indices """
    d = {}
    for disease, findings in disease2finding.items():
        d[all_diseases.index(disease)] = {
            all_findings.index(finding): prob for finding, prob in findings.items()}
    disease2finding = d

    finding2disease = {all_findings.index(finding): [all_diseases.index(d) for d in diseases]
                       for finding, diseases in finding2disease.items()}

    return finding2disease, disease2finding


def load_data(args):
    dataset_name = args.dataset_name.lower()
    if dataset_name == 'symcat' or dataset_name == 'hpo':
        return load_graph_data(args)
    elif dataset_name == 'muzhi' or dataset_name == 'dxy':
        return load_dialogue_data(args)
    elif dataset_name.startswith('ddxplus_'): # dataset_name == 'ddxplus_validate' or dataset_name == 'ddxplus_test'
        prefix = dataset_name[8:]
        return load_ddxplus_data(args, prefix)
    else:
        raise ValueError(f'Dataset name {dataset_name} does not exist.')
