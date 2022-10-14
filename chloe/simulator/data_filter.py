#!/usr/bin/env python

import argparse
import json
import os

import pandas as pd


def clean_line_breaks(data, replace=" "):
    """Replaces line breaks in the source string with a provided replace string.

    Parameters
    ----------
    data: str
        data string to be cleaned.
    replace: str
        string to be used for replacing data.

    Returns
    -------
    result: str
        the resulting string.

    """
    result = data.replace("\r\n", replace)
    result = result.replace("\r", replace)
    result = result.replace("\n", replace)
    return result


def clean(data, replace=" "):
    """Replaces commas and line breaks in the data string with the replace string.

    Parameters
    ----------
    data: str
        data string to be cleaned.
    replace: str
        string to be used for replacing data.

    Returns
    -------
    result: str
        the resulting string.

    """
    result = clean_line_breaks(data, replace)
    result = result.replace(",", replace)
    return result


def filter_df(df, filter_set, column):
    """Filters a dataframe (df) based on values in filter_set in the specified column.

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe.
    filter_set: set
        set containing column values whose corresponding rows should
        be present in the filtered dataframe.
    column: str
        name of the column in the dataframe on which the filtering is to
        done.

    Returns
    -------
    filtered_df: pandas.DataFrame
        the filtered dataframe.

    """
    filtered_df = df[df[column].isin(filter_set)].reset_index(drop=True)
    return filtered_df


def filter_patients(patient_filepath, clean_patho_set, output_dir):
    """Removes patients with patologies that are not part of the allowed pathologie set.

    This is a wrapper function to remove patients with patologies that are not
    in the allowed pathologies (clean_patho_set) for the
    current chief complaint and save the updated patient list.

    Parameters
    ----------
    patient_filepath: str
        path to the folder containing the train, test and validate patient sets.
    clean_patho_set: set
        set containing relevant pathologies for the input chief complaint.
    output_dir: str
        path to store the filtered patients.

    Returns
    -------
    None

    """
    for patient_set_prefix in ["train", "validate", "test"]:
        patients_file_name = os.path.join(
            patient_filepath, f"{patient_set_prefix}_patients.zip"
        )
        df = pd.read_csv(patients_file_name, sep=",")
        selected_patients = filter_df(df, clean_patho_set, "PATHOLOGY")
        # saving selected patients
        selected_patient_filename = os.path.join(
            output_dir, f"{patient_set_prefix}_patients.zip"
        )
        selected_patients.to_csv(selected_patient_filename, sep=",", index=False)


def filter_data(
    symptom_filepath,
    condition_filepath,
    patient_filepath,
    num_pathos,
    chief_complaint_ids,
    patho_list,
    output_dir,
    output_prefix,
):
    """Filters data to match specified criteria and generate corresponding config files.

    This methods filters data to match specified criteria and generate corresponding
    config files associated to the matching pathologies that will be useful
    for running the simulator.

    Parameters
    ----------
    symptom_filepath: str
        path to a json file containing the symptom data to filter from.
    condition_filepath: str
        path to a json file containing the condition data to filter from.
    patient_filepath: str
        path to the folder containing csv file for the patients to filter from.
    num_pathos: int
        number of pathos to be selected. It will be ignored if patho_list is
        not empty or if chief_complaint_ids is not empty.
    chief_complaint_ids: list of str
        id of the chief complaints associated with the pathologies of interest.
        It will be ignored if patho_list is not empty.
    patho_list: list
        list of pathos of interest (only those pathos will be selected).
    output_dir: str
        path to outputs - will store files here.
    output_prefix: str
        prefix to be added to the generated file names.

    Returns
    -------
    None

    """
    patho_key_name = "condition_name"
    patho_2_symptom_key = "slug"

    with open(condition_filepath) as fp:
        conditions = json.load(fp)

    # retrieve the list of patho to be retrieved (if the provided list is
    # empty)
    if (patho_list is None or len(patho_list) == 0) and (not chief_complaint_ids):
        index_2_key = sorted(list(conditions.keys()))
        num_pathos = min(num_pathos, len(index_2_key))
        patho_list = [
            conditions[index_2_key[k]][patho_key_name] for k in range(num_pathos)
        ]
    elif patho_list is None or len(patho_list) == 0:
        tmp_set = set(chief_complaint_ids)
        patho_list = [
            conditions[k][patho_key_name]
            for k in conditions.keys()
            if tmp_set.intersection(conditions[k].get("id-cc-list", []))
        ]
        msg = (
            f"No pathology were found with the provided "
            f"chief complaints: {chief_complaint_ids}."
        )
        assert len(patho_list) > 0, msg

    patho_set = set(patho_list)
    selected_pathos = {}
    symptoms_set = set()
    for key in conditions.keys():
        if conditions[key][patho_key_name] in patho_set:
            selected_pathos[key] = conditions[key]
            patho_symptoms = conditions[key].get("symptoms", {})
            patho_symptoms.update(conditions[key].get("antecedents", {}))
            if patho_symptoms:
                for sympk in patho_symptoms.keys():
                    symptom_key = patho_symptoms[sympk].get(patho_2_symptom_key, sympk)
                    symptoms_set.add(symptom_key)

    # symptoms file
    with open(symptom_filepath) as fp:
        symptoms = json.load(fp)

    selected_symptoms = {}
    for key in symptoms.keys():
        if key in symptoms_set:
            selected_symptoms[key] = symptoms[key]

    assert len(selected_symptoms) == len(symptoms_set)
    # saving
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # data frame
    clean_patho_set = set([clean(s) for s in patho_set])
    # filtering patients
    filter_patients(patient_filepath, clean_patho_set, output_dir)

    # saving selected pathos
    selected_patho_filename = os.path.join(
        output_dir, "{}_{}_conditions.json".format(output_prefix, len(selected_pathos))
    )
    with open(selected_patho_filename, "w") as fp:
        json.dump(selected_pathos, fp, indent=4)

    # saving selected symptoms
    selected_symptom_filename = os.path.join(
        output_dir, "{}_{}_symptoms.json".format(output_prefix, len(selected_symptoms))
    )
    with open(selected_symptom_filename, "w") as fp:
        json.dump(selected_symptoms, fp, indent=4)


def main():
    """Utility function for filtering datasets based on specified criteria.

    This is an utility function for filtering datasets based on
    specified criteria. At the end, only patients suffering from a pathology
    matching the criteria will be kept.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symptom_filepath",
        help="path to a json file containing the symptom data.",
        type=str,
        default="./chloe/data/symcat/parsed_jsons/symptoms.json",
    )
    parser.add_argument(
        "--condition_filepath",
        help="path to a json file containing the condition data.",
        type=str,
        default="./chloe/data/symcat/parsed_jsons/conditions.json",
    )
    parser.add_argument(
        "--patient_filepath",
        help="path to folder containing train, validate and test set (default: './')"
        "used if the patients are split into train, validate and test sets",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--num_pathos",
        help="number of pathos to be selected. (default 10). It will "
        "be ignored if a patho list is provided using patho_list_conf param",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--patho_list_conf",
        help="path to a txt file containing the list of pathos of interest."
        " one patho per line.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cc",
        help="comma separated list of chief complaints associated with "
        "the pathologies of interest. e.g. cc7, cc2",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_prefix",
        help="prefix to be added to the generated file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="path to outputs - will store files here (default: './')",
        type=str,
        default="./",
    )

    args = parser.parse_args()

    patho_list = []

    # load the patho list of interest
    if args.patho_list_conf is not None:
        file = open(args.patho_list_conf, "r")
        patho_list = file.readlines()
        patho_list = [clean_line_breaks(a, "") for a in patho_list]
        pathos_set = set(patho_list)
        pathos_set.discard("")
        patho_list = list(pathos_set)

    if args.cc is None:
        cc_list = []
    else:
        cc_list = args.cc.split(",")
        cc_list = [a.strip() for a in cc_list]

    filter_data(
        args.symptom_filepath,
        args.condition_filepath,
        args.patient_filepath,
        args.num_pathos,
        cc_list,
        patho_list,
        args.output_dir,
        args.output_prefix,
    )


if __name__ == "__main__":
    main()
