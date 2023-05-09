import argparse
import json
import os

import matplotlib.pyplot as plt

from chloe.utils.misc_eval_utils import get_patient_weight_factor
from chloe.utils.sim_utils import clean, load_csv


def create_argument_parser():
    """Sets up the cmdline arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patients_fp", required=True, help="Path to the patients file",
    )
    parser.add_argument(
        "--output_fp",
        required=True,
        help="Path to the file to write the outputs of the evaluation",
    )
    parser.add_argument(
        "--travel_evidence",
        default="trav1",
        help="Code associated with the travel evidence",
    )
    parser.add_argument(
        "--travel_negative_response",
        default="N",
        help="Code associated with the negative answer of the travel evidence",
    )
    parser.add_argument(
        "--default_location",
        default="AmerN",
        help="Code associated with the default geographical location",
    )
    parser.add_argument(
        "--symptoms_fp",
        default="./data/evidences.json",
        help="Path to the symptoms file for patients file specified in fp flag.",
    )
    parser.add_argument(
        "--conditions_fp",
        default="./data/conditions.json",
        help="Path to the conditions file for patients file specified in fp flag.",
    )
    parser.add_argument(
        "--weight_fp",
        default=None,
        help="Path to the file containing the weights of the pathology given "
        " demographic context such as age, sex, and geo region.",
    )
    parser.add_argument(
        "--prefix", default="", help="Prefix to add to the generated file.",
    )

    return parser


def load_weight_file(weight_fp, condition_fp, symptom_fp, travel_evidence):
    if (weight_fp is None) or (weight_fp == ""):
        return None, None, None, None, None

    with open(weight_fp) as fp:
        data = json.load(fp)
    index_2_key = sorted(list(data.keys()))
    with open(condition_fp) as fp:
        cond_data = json.load(fp)
    name_2_index = {
        clean(cond_data[index_2_key[i]]["condition_name"]): i
        for i in range(len(index_2_key))
    }
    with open(symptom_fp) as fp:
        symp_data = json.load(fp)
    is_geo_present = travel_evidence in symp_data
    all_locations=None
    if is_geo_present:
        all_locations = symp_data.get(travel_evidence, {}).get("possible-values")
    return is_geo_present, name_2_index, index_2_key, data, all_locations


def compute_geo_region(
    symptoms, is_geo_pres, travel_evidence, travel_negative_response, default_location
):
    if not is_geo_pres:
        return None
    indicator = travel_evidence + "_@_"
    values = [a for a in symptoms if a.startswith(indicator)]
    if len(values) == 0:
        geo_value = travel_negative_response
    else:
        idx = values[0].find("_@_")
        geo_value = values[0][idx + 3 :]
    if geo_value == travel_negative_response:
        geo_value = default_location
    return geo_value


def get_weight_map(
    patho_name, name_2_index, index_2_key, weight_data, sex, geo, age, all_locations
):
    index = name_2_index[patho_name]
    patho_key = index_2_key[index]
    return get_patient_weight_factor(
        index_2_key, weight_data, patho_key, sex, geo, age, all_locations
    )


def compute_patient_weight(
    data, is_geo_pres, name_2_index, index_2_key, weight_data, all_locations,
    travel_evidence, travel_negative_response, default_location
):
    data["GEO_REGION"] = data["SYMPTOMS"].apply(
        lambda x: compute_geo_region(
            x, is_geo_pres, travel_evidence, travel_negative_response, default_location
        )
    )
    columns = [
        "PATHOLOGY",
        "GENDER",
        "GEO_REGION",
        "AGE_BEGIN",
    ]
    data["WEIGHT_FACTOR"] = data[columns].apply(
        lambda x: get_weight_map(
            x[0], name_2_index, index_2_key, weight_data, x[1], x[2], x[3], all_locations
        ),
        axis=1,
    )
    return data


def plotHistogram(x, y, filename, x_label="Pathologies", y_label="Frequency"):
    plt.clf()
    plt.figure(figsize=(6, 7))
    plt.bar(x, y)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xticks(fontsize=6, rotation=90)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def main(args):

    out = load_csv(args.patients_fp, False)
    data, unique_pathos = out[0], out[2]

    os.makedirs(args.output_fp, exist_ok=True)

    if args.weight_fp is not None:
        # get the weight data
        [
            is_geo_present,
            patho_name_2_index,
            patho_index_2_key,
            patho_weight_data,
            all_locations,
        ] = load_weight_file(args.weight_fp, args.conditions_fp, args.symptoms_fp, args.travel_evidence)
        # compute weight data
        data = compute_patient_weight(
            data,
            is_geo_present,
            patho_name_2_index,
            patho_index_2_key,
            patho_weight_data,
            all_locations,
            args.travel_evidence,
            args.travel_negative_response,
            args.default_location
        )
        df2 = data.groupby(["PATHOLOGY"]).agg({"WEIGHT_FACTOR": "sum"})
        sum_weights = df2.WEIGHT_FACTOR.agg("sum")
        sum_weights = 1 if sum_weights == 0 else sum_weights
        df2["WEIGHT_FACTOR"] = df2["WEIGHT_FACTOR"].apply(lambda x: x / sum_weights)
        patho_weighted_freq_dict = df2.to_dict()["WEIGHT_FACTOR"]
        values = [patho_weighted_freq_dict.get(a, 0) for a in unique_pathos]
        filename = f"{args.output_fp}/{args.prefix}WeightedHist.png"
        plotHistogram(unique_pathos, values, filename)

    patho_freq_dict = data["PATHOLOGY"].value_counts(normalize=True).to_dict()
    values = [patho_freq_dict.get(a, 0) for a in unique_pathos]
    filename = f"{args.output_fp}/{args.prefix}Hist.png"
    plotHistogram(unique_pathos, values, filename)


if __name__ == "__main__":
    cmdline_args = create_argument_parser().parse_args()
    main(cmdline_args)
