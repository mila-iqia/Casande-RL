import argparse
import os

import numpy as np
import pandas as pd


"""This file contains utility functions for splitting a dataset into train,
    valid, and test subsets.
"""


def save_data(output_dir, output_prefix, data):
    """Saves the provided data in the specified output folder.

    Parameters
    ----------
    output_dir: str
        path to output folder where the data will be saved.
    output_prefix: str
        the prefix to be appended to the filename.
    data: pd.DataFrame
        the data to be saved.

    Returns
    -------
    None

    """
    selected_patient_filename = os.path.join(
        output_dir, "{}_patients.zip".format(output_prefix)
    )
    data.to_csv(selected_patient_filename, sep=",", index=False)


def split_train_validate_test(
    patient_filepath, output_dir, proportions=[0.75, 0.85], random_state=12345
):
    """Splits the provided data according to the provided proportions.

    Parameters
    ----------
    patient_filepath: str
        path to the patient file to be splitted.
    output_dir: str
        path to output folder where the files will be stored.
    proportions: list
        list of the proportions on which the split will be based. Default: [0.75, 0.85]
        It means 75% train, 10 % validation, 15% test.
    random_state: int
        the random seed to be used. Default: 12345

    Returns
    -------
    None

    """

    df = pd.read_csv(patient_filepath, sep=",")

    proportions = [int(a * len(df)) for a in proportions]
    train, validate, test = np.split(
        df.sample(frac=1.0, random_state=random_state), proportions
    )

    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)

    save_data(output_dir, "train", train)
    save_data(output_dir, "validate", validate)
    save_data(output_dir, "test", test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="filepath to be splitted", required=True)
    parser.add_argument("--output_dir", help="output dir", default="./")
    parser.add_argument("--seed", type=int, help="seed to be used", default=12345)
    parser.add_argument(
        "--train_percent", type=float, help="percentage of train data", default=0.75
    )
    parser.add_argument(
        "--valid_percent", type=float, help="percentage of validation data", default=0.1
    )

    args = parser.parse_args()

    total_percent = args.train_percent + args.valid_percent
    assert total_percent <= 1.0 and args.train_percent >= 0 and args.valid_percent >= 0

    proportions = [args.train_percent, total_percent]

    split_train_validate_test(
        args.input_file, args.output_dir, proportions, random_state=args.seed,
    )
