#!/usr/bin/env python

import argparse
import json
import os

import pandas as pd

from chloe.utils.sim_utils import (
    clean,
    only_contain_derivated_symptoms,
    preprocess_symptoms,
)


def only_contain_binary_antecedents(symptoms, binary_symptom_set, antecedent_set):
    """Utility function to check the validity of symptom data.

    Determine if symptoms data as provided only contain
    binary symptoms which are antecedents.

    Parameters
    ----------
    symptoms: list
        list of all symptoms.
    binary_symptom_set: set
        set of all binary symptoms.
    antecedent_set: set
        set of all authorized antecedents.

    Returns
    -------
    result: bool
        True if the provided data only contain binary antecedents, False otherwise.

    """
    tmp = []
    for a in symptoms:
        idx = a.find("_@_")
        if idx == -1:
            val = a
        else:
            val = a[:idx]
        if val in binary_symptom_set:
            tmp.append(val)

    for a in tmp:
        if not (a in antecedent_set):
            return False
    return True


def filter_data(
    symptom_filepath, condition_filepath, patient_filepath, output_dir, output_prefix,
):
    """Filters patient data to eliminate invalid entries.

    The filtering is based on symptoms, antecedents, and pathologies.
    It proceeds by:
        - Eliminating synthesized patients with no symptoms.
        - Eliminating synthesized patients with only derivated symptoms.
        - Eliminating synthesized patients with binary symptoms.
          that are only antecedents.
        - Eliminating synthesized patients whose pathology are not part of
          the provided condition file, if any.

    Parameters
    ----------
    symptom_filepath: str
        path to a json file containing the symptom data to filter from.
    condition_filepath: str
        path to a json file containing the condition data to filter from.
    patient_filepath: str
        path to the csv file containing the patients to filter from.
    output_dir: str
        path to outputs - will store files here.
    output_prefix: str
        prefix to be added to the generated file names.

    Returns
    -------
    None

    """

    patho_data = {}
    symp_data = {}

    # load the condition/patho data of interest
    if condition_filepath is not None:
        with open(condition_filepath) as fp:
            patho_data = json.load(fp)
    # load the symptom data of interest
    assert symptom_filepath is not None
    with open(symptom_filepath) as fp:
        symp_data = json.load(fp)

    atcd_set = set(
        [
            clean(symp_data[k]["name"])
            for k in symp_data.keys()
            if symp_data[k].get("is_antecedent", False)
        ]
    )
    binary_symptom_set = set(
        [
            clean(symp_data[k]["name"])
            for k in symp_data.keys()
            if symp_data[k].get("type-donnes", "B") == "B"
        ]
    )
    patho_list = [clean(patho_data[k]["condition_name"]) for k in patho_data.keys()]

    # load the patient file
    df = pd.read_csv(patient_filepath, sep=",")
    num_rows = len(df)

    # we remove patient with zero symptoms
    df = df[df["NUM_SYMPTOMS"] != 0].reset_index(drop=True)

    # we parse the symptoms to remove the severity level as we do
    # not need it for this project
    df["PRE_PROCESS_SYMPTOMS"] = df["SYMPTOMS"].apply(lambda x: preprocess_symptoms(x))

    # remove rows that only contain derivated symptoms
    df = df[
        ~df["PRE_PROCESS_SYMPTOMS"].map(only_contain_derivated_symptoms)
    ].reset_index(drop=True)

    if len(atcd_set) > 0:
        # remove rows which contain only antecedents
        df = df[
            ~df["PRE_PROCESS_SYMPTOMS"].apply(
                lambda x: only_contain_binary_antecedents(
                    x, binary_symptom_set, atcd_set
                )
            )
        ].reset_index(drop=True)

    if len(patho_list) > 0:
        # remove rows which are not in the patho list
        df = df[df["PATHOLOGY"].isin(patho_list)].reset_index(drop=True)

    num_rows_end = len(df)

    all_columns = [
        "PATIENT",
        "GENDER",
        "RACE",
        "ETHNICITY",
        "AGE_BEGIN",
        "AGE_END",
        "PATHOLOGY",
        "NUM_SYMPTOMS",
        "SYMPTOMS",
    ]
    df = df[all_columns]

    # saving selected patients
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    selected_patient_filename = os.path.join(
        output_dir, f"{output_prefix}_patients.zip"
    )
    df.to_csv(selected_patient_filename, sep=",", index=False)
    print(f"The number of eliminated rows is: {num_rows - num_rows_end}")


def main():
    """This is an utility function for filtering datasets.

    The filtering is based on symptoms, antecedents, and pathologies.
    It proceeds by:
        - Eliminating synthesized patients with no symptoms.
        - Eliminating synthesized patients with only derivated symptoms.
        - Eliminating synthesized patients with binary symptoms.
          that are only antecedents.
        - Eliminating synthesized patients whose pathology are not part of
          the provided condition file, if any.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symptom_filepath",
        help="path to a json file containing the symptom data.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--condition_filepath",
        help="path to a json file containing the condition data.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--patient_filepath",
        help="path to the (zipped) csv file to filter.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_prefix",
        help="prefix to be added to the generated file",
        type=str,
        default="filtered",
    )
    parser.add_argument(
        "--output_dir",
        help="path to outputs - will store files here (default: './')",
        type=str,
        default="./",
    )

    args = parser.parse_args()

    filter_data(
        args.symptom_filepath,
        args.condition_filepath,
        args.patient_filepath,
        args.output_dir,
        args.output_prefix,
    )


if __name__ == "__main__":
    main()
