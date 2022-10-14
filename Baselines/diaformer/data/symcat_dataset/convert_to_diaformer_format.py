import argparse, multiprocessing, ast, os, pickle, json
import pandas as pd
from itertools import repeat

def parse_args():
    parser = argparse.ArgumentParser("Convert SymCAT's data to Diaformer's format. The output of this file should be the input to preprocess.py.")

    parser.add_argument("--train_data_path", type=str, required=True, help="path to training data file")
    parser.add_argument("--val_data_path", type=str, required=True, help="path to validation data file")
    parser.add_argument("--test_data_path", type=str, required=True, help="path to test data file")
    parser.add_argument("--evi_meta_path", type=str, required=True, help="path to the evidences' meta data file")

    parser.add_argument("--save_dir", type=str, default=".", help="folder to save converted data")

    return parser.parse_args()

def load_data_from_zip(data_path):
    """Load data from the disk and convert the format.

    The data stored on the disk is a DataFrame. It will be converted into a dictionary whose keys are patient indices.

    Parameters
    ----------
    data_path: str
        path to the data zip file

    Return
    ----------
    patients_dict: dict 
        dictionary containing each patient's data

    """
    # load data from disk
    patients_df = pd.read_csv(data_path, sep=",")

    # convert to dictionary
    patients_dict = patients_df.to_dict(orient="index")

    return patients_dict

def convert_evidence_data_type(meta_data):
    """Convert the data type in json files to the one used in supervised baselines.

    The main difference is that a new data type, "number", is introduced. It corresponds to the integer situation under
    "C" data type in json files.

    Parameter:
    ----------
    meta_data: dict
        the meta data of an evidence, as extracted from the json file
    
    Return:
    data_type: str
        the converted data type
    """
    if meta_data["data_type"] == "B":
        return "binary"
    if meta_data["data_type"] == "M":
        return "multi-label"
    if meta_data["data_type"] == "C":
        if isinstance(meta_data["possible-values"][0], str):
            return "multi-class"
        else:
            return "number"

def to_diaformer_format(patient_data, evi_meta_data):
    """Convert a patient's data to Diaformer format.

    Parameters
    ----------
    patient_data: dict
        a patient's data
    evi_meta_data: dict
        evidence meta data
    
    Return
    ----------
    reformatted_data: dict
        the reformatted patient data

    """
    implicit_symptoms = set(ast.literal_eval(patient_data["EVIDENCES"])) - {patient_data["INITIAL_EVIDENCE"]}
    # parse evidences
    if "_@_" not in patient_data["INITIAL_EVIDENCE"]:
        # binary
        explicit_inform_slots = {patient_data["INITIAL_EVIDENCE"]: {"option": 1, "option_type": "binary"}}
    else:
        evidence, option = evidence.split("_@_")
        if convert_evidence_data_type(evi_meta_data[evidence]) == "number":
            explicit_inform_slots = {evidence: {"option": float(option), "option_type": "number"}}
        elif convert_evidence_data_type(evi_meta_data[evidence]) == "multi-class":
            explicit_inform_slots = {evidence: {"option": evi_meta_data[evidence]["possible-values"].index(option), "option_type": "multi-class"}}
        else:
            raise NotImplementedError()
    implicit_inform_slots = {}
    for evidence in implicit_symptoms:
        if "_@_" not in evidence:
            # binary
            implicit_inform_slots[evidence] = {"option": 1, "option_type": "binary"}
        else:
            evidence, option = evidence.split("_@_")
            if convert_evidence_data_type(evi_meta_data[evidence]) == "number":
                implicit_inform_slots[evidence] = {"option": float(option), "option_type": "number"}
            elif convert_evidence_data_type(evi_meta_data[evidence]) == "multi-class":
                implicit_inform_slots[evidence] = {"option": evi_meta_data[evidence]["possible-values"].index(option), "option_type": "multi-class"}
            else:
                if evidence not in implicit_inform_slots:
                    implicit_inform_slots[evidence] = {"option": [evi_meta_data[evidence]["possible-values"].index(option)], "option_type": "multi-label"}
                else:
                    implicit_inform_slots[evidence]["option"].append(evi_meta_data[evidence]["possible-values"].index(option))
    reformatted_data = {
        "disease_tag": patient_data["PATHOLOGY"],
        "goal": {
            "explicit_inform_slots": explicit_inform_slots,
            "implicit_inform_slots": implicit_inform_slots,
        }
    }
    return reformatted_data

if __name__ == "__main__":
    args = parse_args()

    # load data
    print("Loading training data")
    train_data = load_data_from_zip(args.train_data_path)
    print("Loading validation data")
    val_data = load_data_from_zip(args.val_data_path)
    print("Loading test data")
    test_data = load_data_from_zip(args.test_data_path)
    # read evidence meta data
    with open(args.evi_meta_path, "r") as file:
        evi_meta_data = json.load(file)

    # reformat
    print("Converting training data")
    with multiprocessing.Pool() as pool:
        train_data = pool.starmap(to_diaformer_format, zip(train_data.values(), repeat(evi_meta_data)))
    print("Converting validation data")
    with multiprocessing.Pool() as pool:
        val_data = pool.starmap(to_diaformer_format, zip(val_data.values(), repeat(evi_meta_data)))
    print("Converting test data")
    with multiprocessing.Pool() as pool:
        test_data = pool.starmap(to_diaformer_format, zip(test_data.values(), repeat(evi_meta_data)))

    # save train and validation sets to disk
    print("Saving training and validation data")
    with open(os.path.join(args.save_dir, "train_valid_data.pkl"), "wb") as file:
        pickle.dump(
            {
                "train": train_data,
                "test": val_data,
            }, file
        )
    print("Saving test data")
    with open(os.path.join(args.save_dir, "test_data.pkl"), "wb") as file:
        pickle.dump(
            {
                "test": test_data,
            }, file
        )