import ast
import json
from collections import OrderedDict

import pandas as pd
import numpy as np
import copy

def clean(data):
    result = data.replace("\r\n", " ")
    result = result.replace("\r", " ")
    result = result.replace("\n", " ")
    result = result.replace(",", " ")
    return result

def load_json_data(data_filepath, key_name):
    with open(data_filepath) as fp:
        data = json.load(fp)
    index_2_key = sorted(list(data.keys()))
    for k in index_2_key:
        data[k][key_name] = clean(data[k][key_name])
    name_2_index = {data[index_2_key[i]][key_name]: i for i in range(len(index_2_key))}
    data_names = [data[k][key_name] for k in index_2_key]    
    return index_2_key, name_2_index, data
    

def computeBayesianInferenceStats4(cond_json_file, symp_json_file, patient_file):
    def getallevidences(evidences, index_2_key, name_2_index, symp, patho, cond, cond_cat_multi_evi):
        if isinstance(evidences, str):
            evidences = ast.literal_eval(evidences)
        symp_indices = set()
        all_multicatevi = set(cond_cat_multi_evi[patho])
        current_set_cat_mult = set()
        for e in evidences:
            if (("_@_" in e)):
                idx = e.find("_@_")
                b = e[:idx]
                current_set_cat_mult.add(b)
        missing_set_cat_mult = all_multicatevi - current_set_cat_mult
        myset = set(evidences)
        for e in missing_set_cat_mult:
            myset.add(f"{e}_@_{symp[index_2_key[name_2_index[e]]].get('default_value', '')}")
        evidences = myset
        result = {}
        for e in evidences:
            if (not ("_@_" in e)):
                result[e] = {}
                result[e]["PRES"] = 1
            else:
                idx = e.find("_@_")
                b = e[:idx]
                elem_val = e[idx + 3 :]
                result[b] = {}
                result[b][elem_val] = 1
        return result
    def update_counts(a, b):
        for k in b.keys():
            if k not in a:
                a[k] = copy.deepcopy(b[k])
            else:
                for j in b[k].keys():
                    a[k][j] = a[k].get(j, 0) + b[k][j]
        return a
    def normalize_counts(a, n):
        for k in a.keys():
            for j in a[k].keys():
                a[k][j] = a[k][j] / n
        return a
    def merge_data(x):
        x = x.tolist()
        result = {}
        for a in x:
            result = update_counts(result, a)
        # normalize
        result = normalize_counts(result, len(x))
        return result
    cond_infos = load_json_data(cond_json_file, key_name="condition_name")
    cond_idx_2_key, cond_name_2_idx, cond_json = cond_infos
    symp_infos = load_json_data(symp_json_file, key_name="name")
    symp_idx_2_key, symp_name_2_idx, symp_json = symp_infos
    cond_cat_multi_evi = {}
    cond_cat_evi = {}
    for d, key in enumerate(cond_idx_2_key):
        defined_symptom_keys = list(cond_json[key]["symptoms"].keys())
        if "antecedents" in cond_json[key]:
            defined_symptom_keys += list(cond_json[key]["antecedents"].keys())
        cond_cat_multi_evi[cond_json[key]["condition_name"]] = [
            symp_json[evi]["name"] for evi in defined_symptom_keys
            if symp_json[evi]["data_type"] in ['M', 'C']
        ]
        cond_cat_evi[cond_json[key]["condition_name"]] = set([
            symp_json[evi]["name"] for evi in defined_symptom_keys
            if symp_json[evi]["data_type"] == 'C'
        ])
    df = pd.read_csv(patient_file)
    prior = df["PATHOLOGY"].value_counts(normalize=True).to_dict()
    df["ALL_EVI_DICT"] = df.apply(lambda x: getallevidences(x.EVIDENCES, symp_idx_2_key, symp_name_2_idx, symp_json, x.PATHOLOGY, cond_json, cond_cat_multi_evi), axis=1)
    df2 = df.groupby(["PATHOLOGY"]).agg({"ALL_EVI_DICT": merge_data})
    pathology_stats = df2.to_dict()["ALL_EVI_DICT"]
    print(json.dumps(pathology_stats, indent=4))
    all_cat_multi_evi = set()
    for k in cond_cat_multi_evi.keys():
        all_cat_multi_evi.update(cond_cat_multi_evi[k])
    all_cat_multi_evi_possible_val = {}
    for e in all_cat_multi_evi:
        all_cat_multi_evi_possible_val[e] = set()
        for k in pathology_stats.keys():
            all_cat_multi_evi_possible_val[e].update(pathology_stats[k].get(e, {}).keys())
    for k in pathology_stats.keys():
        for e in pathology_stats[k].keys():
            if "PRES" in pathology_stats[k][e]:
                pathology_stats[k][e] = pathology_stats[k][e]["PRES"]
            elif e in cond_cat_evi[k]:
                for v in all_cat_multi_evi_possible_val[e]:
                    if str(v) not in pathology_stats[k][e]:
                        pathology_stats[k][e][str(v)] = 1e-7
                # for v in symp_json[symp_idx_2_key[symp_name_2_idx[e]]].get('possible-values', []):
                #     if str(v) not in pathology_stats[k][e]:
                #         pathology_stats[k][e][str(v)] = 0.0
                # pathology_stats[k][e] = {f"{e}_@_{c}": d for c, d in pathology_stats[k][e].items()}
            else:
                for v in all_cat_multi_evi_possible_val[e]:
                    if str(v) not in pathology_stats[k][e]:
                        pathology_stats[k][e][str(v)] = 1e-7
                def_val = symp_json[symp_idx_2_key[symp_name_2_idx[e]]].get('default_value', '')
                if str(def_val) not in pathology_stats[k][e]:
                        pathology_stats[k][e][str(def_val)] = 1e-7
                # for v in symp_json[symp_idx_2_key[symp_name_2_idx[e]]].get('possible-values', []):
                #     if str(v) not in pathology_stats[k][e]:
                #         pathology_stats[k][e][str(v)] = 0.0
                sorted_tuples = sorted(pathology_stats[k][e].items(), key=lambda item: item[1], reverse=True)
                sorted_dict = OrderedDict()
                for c, d in sorted_tuples:
                    # sorted_dict[f"{e}_@_{c}"] = d
                    sorted_dict[f"{c}"] = d
                pathology_stats[k][e] = sorted_dict
    final_results = {
        k: {"prior": prior[k], "findings": pathology_stats[k]}
        for k in pathology_stats.keys()
    }
    return final_results

def generateBayesianInferenceJsonStats4(cond_json_file, symp_json_file, src_dir, patient_file, output_dir):
    pathology_stats = computeBayesianInferenceStats4(cond_json_file, symp_json_file, f"{src_dir}/{patient_file}")
    with open(f"{output_dir}/casande_bed_stats_4.json", 'w') as outfile:
        json.dump(pathology_stats, outfile, indent=4)
    return
