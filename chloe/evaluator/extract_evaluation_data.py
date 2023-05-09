import argparse
import json
import os
import pickle

from chloe.utils.misc_eval_utils import get_pred_truth, get_weight
from chloe.utils.read_utils import read_pkl
from chloe.utils.sim_utils import clean, decode_sex


def create_argument_parser():
    """Sets up the cmdline arguments."""

    parser = argparse.ArgumentParser(description="Launch model training !")

    parser.add_argument(
        "--patients_fp",
        required=True,
        help="Path to the patients file containing the evaluated agent interactions",
    )
    parser.add_argument(
        "--output_fp",
        required=True,
        help="Path to the file to write the outputs of the evaluation",
    )
    parser.add_argument(
        "--name",
        default="Extracted_Trajectory_Info",
        help="Name of the model being evaluated",
    )
    parser.add_argument(
        "--travel_evidence",
        default="trav1",
        help="Code associated with the travel evidence",
    )
    parser.add_argument(
        "--symptoms_fp",
        required=True,
        help="Path to the symptoms file for patients file specified in fp flag.",
        default="./data/evidences.json",
    )
    parser.add_argument(
        "--conditions_fp",
        required=True,
        help="Path to the conditions file for patients file specified in fp flag.",
        default="./data/conditions.json",
    )
    parser.add_argument(
        "--weight_fp",
        default=None,
        help="Path to the file containing the weights of the pathology given "
        " demographic context such as age, sex, and geo region.",
    )

    return parser


def write_json(data, fp):
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)


def load_weight_file(weight_fp):
    if (weight_fp is None) or (weight_fp == ""):
        return None, None

    with open(weight_fp) as fp:
        data = json.load(fp)
    index_2_key = sorted(list(data.keys()))
    return index_2_key, data


def load_cond_and_sympt(condition_fp, symptom_fp, travel_evidence):
    with open(condition_fp) as fp:
        cond_data = json.load(fp)
    cond_index_2_key = sorted(list(cond_data.keys()))
    cond_index_2_name = [
        clean(cond_data[cond_index_2_key[i]]["condition_name"])
        for i in range(len(cond_index_2_key))
    ]
    with open(symptom_fp) as fp:
        symp_data = json.load(fp)
    symp_index_2_key = sorted(list(symp_data.keys()))
    for k in symp_index_2_key:
        symp_data[k]["name"] = clean(symp_data[k]["name"])
    symp_index_2_name = [
        symp_data[symp_index_2_key[i]]["name"]
        for i in range(len(symp_index_2_key))
    ]
    symp_index_2_atcd_flg = [
        symp_data[symp_index_2_key[i]].get("is_antecedent", False)
        for i in range(len(symp_index_2_key))
    ]
    all_locations = None
    if travel_evidence is not None:
        all_locations = symp_data.get(travel_evidence, {}).get("possible-values")
    return cond_index_2_name, symp_index_2_name, symp_index_2_atcd_flg, all_locations


def main(args):
    out = load_cond_and_sympt(args.conditions_fp, args.symptoms_fp, args.travel_evidence)
    cond_index_2_name, symp_index_2_name, symp_index_2_atcd_flg, all_locations = out
    pathoIndex_2_key, weight_data = load_weight_file(args.weight_fp)
    data = read_pkl(args.patients_fp)
    pred_idx_proba, truth_idx_proba, gt_patho = get_pred_truth(data, cond_index_2_name)

    all_sex = data["data"].get("sex", None)
    all_geo = data["data"].get("geo", None)
    all_age = data["data"].get("age", None)

    all_sex = (
        [None] * len(gt_patho)
        if all_sex is None
        else [decode_sex(sex) if not isinstance(sex, str) else sex for sex in all_sex]
    )
    all_geo = (
        [None] * len(gt_patho)
        if all_geo is None
        else [
            (
                all_locations[geo]
                if ((not isinstance(geo, str)) and (all_locations is not None))
                else geo
            )
            for geo in all_geo
        ]
    )
    all_age = [None] * len(gt_patho) if all_age is None else all_age

    w_patient = get_weight(data, pathoIndex_2_key, weight_data, all_locations)

    all_inquired_evidences = data["data"]["inquired_evidences"]
    all_inquired_evidences = [
        [symp_index_2_name[i] for i in evid] for evid in all_inquired_evidences
    ]
    all_atcd_actions = data["data"]["all_atcd_actions"]
    all_relevant_actions = data["data"]["all_relevant_actions"]

    result = {
        "Ground Truth Differential Diagnosis": truth_idx_proba,
        "Pathology": gt_patho,
        "Predicted Differential Diagnosis": pred_idx_proba,
        "Age": all_age,
        "Sex": all_sex,
        "Geo": all_geo,
        "WeightFactor": w_patient,
        "Predicted Symptoms and Antecedents": all_inquired_evidences,
        "Is Antecedent Evidence": all_atcd_actions,
        "Is Experienced Evidence": all_relevant_actions,
    }
    os.makedirs(args.output_fp, exist_ok=True)
    pickle.dump(result, open(f"{args.output_fp}/{args.name}.pkl", "wb"))


if __name__ == "__main__":
    cmdline_args = create_argument_parser().parse_args()
    main(cmdline_args)
