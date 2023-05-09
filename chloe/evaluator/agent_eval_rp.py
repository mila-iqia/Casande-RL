import argparse
import json
import os

import pandas as pd

from chloe.utils.misc_eval_utils import (
    compute_at_k_metrics,
    compute_mass_scenario,
    compute_traj_metrics,
    filter_by_mass,
    get_pred_truth,
    get_weight,
)
from chloe.utils.read_utils import read_pkl


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
        "--model_name", required=True, help="Name of the model being evaluated",
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
    )
    parser.add_argument(
        "--conditions_fp",
        required=True,
        help="Path to the conditions file for patients file specified in fp flag.",
    )
    parser.add_argument(
        "--weight_fp",
        default=None,
        help="Path to the file containing the weights of the pathology given "
        " demographic context such as age, sex, and geo region.",
    )
    parser.add_argument(
        "--pool_size",
        default=64,
        type=int,
        help="Number of processes to be used for evaluation.",
    )
    parser.add_argument(
        "--min_proba",
        default=0.01,
        type=float,
        help="proba threshold needed to include a pathology within the differential.",
    )
    parser.add_argument(
        "--severity_threshold",
        default=3,
        type=int,
        help="Threshold under which a pathology is considered severe.",
    )
    # parser.add_argument(
    #    "--calculate_ncv_metrics",
    #    default=1,
    #    help="A bool indicating if consistency, variability and "
    #    "negative evidence based metrics need to be calculated.",
    # )

    return parser


def write_json(data, fp):
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)


def create_df(pred_idx_proba, truth_idx_proba, gt_patho, w_patient, min_proba=0.01):

    # pred, truth = remove_mass(pred_idx_proba, truth_idx_proba)
    data_dict = {
        "INTERACTION_DIFFERENTIAL_DIAGNOSIS": [
            [f"{val[0]}:{val[1]}" for val in pred] for pred in pred_idx_proba
        ],
        "GROUND_TRUTH_DIFFERENTIAL_DIAGNOSIS": [
            [f"{val[0]}:{val[1]}" for val in gt] for gt in truth_idx_proba
        ],
        "PATHOLOGY": gt_patho,
        "WEIGHT_FACTOR": w_patient,
    }
    data = pd.DataFrame(data_dict)
    # Filter pathos based on mass <=0.01
    data["GROUND_TRUTH_DIFFERENTIAL_DIAGNOSIS"] = data[
        "GROUND_TRUTH_DIFFERENTIAL_DIAGNOSIS"
    ].apply(lambda x: filter_by_mass(x, min_proba))
    data["INTERACTION_DIFFERENTIAL_DIAGNOSIS"] = data[
        "INTERACTION_DIFFERENTIAL_DIAGNOSIS"
    ].apply(lambda x: filter_by_mass(x, min_proba))
    return data


def compute_metrics_diff_diag(data):
    metrics = {}
    compute_at_k_metrics(
        data,
        metrics,
        [1, 3, 5],
        False,
        "INTERACTION_DIFFERENTIAL_DIAGNOSIS",
        "GROUND_TRUTH_DIFFERENTIAL_DIAGNOSIS",
        "WEIGHT_FACTOR",
    )
    compute_mass_scenario(
        data,
        metrics,
        "INTERACTION_DIFFERENTIAL_DIAGNOSIS",
        "GROUND_TRUTH_DIFFERENTIAL_DIAGNOSIS",
        "",
        "WEIGHT_FACTOR",
    )
    return metrics


def load_weight_file(weight_fp):
    if (weight_fp is None) or (weight_fp == ""):
        return None, None

    with open(weight_fp) as fp:
        data = json.load(fp)
    index_2_key = sorted(list(data.keys()))
    return index_2_key, data


def get_all_locations(symptoms_fp, travel_evidence):
    with open(symptom_fp) as fp:
        symp_data = json.load(fp)
    all_locations = None
    if travel_evidence is not None:
        all_locations = symp_data.get(travel_evidence, {}).get("possible-values")
    return all_locations


def main(args):

    pathoIndex_2_key, weight_data = load_weight_file(args.weight_fp)
    data = read_pkl(args.patients_fp)
    pred_idx_proba, truth_idx_proba, gt_patho = get_pred_truth(data)
    all_locations = get_all_locations(args.symptoms_fp, args.travel_evidence)
    w_patient = get_weight(data, pathoIndex_2_key, weight_data, all_locations)
    min_proba = args.min_proba
    data_df = create_df(pred_idx_proba, truth_idx_proba, gt_patho, w_patient, min_proba)
    metrics = compute_metrics_diff_diag(data_df)
    data_df = None
    # metrics = {}
    compute_traj_metrics(
        data,
        w_patient,
        metrics,
        args.symptoms_fp,
        args.conditions_fp,
        args.pool_size,
        min_proba,
        args.severity_threshold,
    )
    os.makedirs(args.output_fp, exist_ok=True)
    write_json(metrics, f"{args.output_fp}/{args.model_name}.json")


if __name__ == "__main__":
    cmdline_args = create_argument_parser().parse_args()
    main(cmdline_args)
