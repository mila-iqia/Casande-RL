import json
import pickle as pkl

import pandas as pd
import yaml


def read_yaml(file_name):
    """Reads a yaml file.

    Parameters
    ----------
    file_name: str
        name of the yaml file to be read.

    Returns
    -------
    cc_data_paths: dict
        data in the yaml file as a dict.

    """
    with open(file_name, "r") as stream:
        cc_data_paths = yaml.load(stream, Loader=yaml.FullLoader)
    return cc_data_paths


def read_csv_zip(file_name):
    """Reads a zip/csv file.

    Parameters
    ----------
    file_name: str
        name of the zip/csv file to be read.

    Returns
    -------
    df: pandas.DataFrame
        data as pandas dataframe in the input zip/csv file.

    """
    return pd.read_csv(file_name, sep=",")


def read_json(file_name):
    """Reads a json file.

    Parameters
    ----------
    file_name: str
        name of the json file to be read.

    Returns
    -------
    data: dict
        data in the json file as a dict.

    """
    with open(file_name) as json_file:
        data = json.load(json_file)
    return data


def read_pkl(file_name):
    """Reads a pickle file.

    Parameters
    ----------
    file_name: str
        name of the pickle file to be read.

    Returns
    -------
    data: object
        data in the pickle file as the original saved object.

    """
    with open(file_name, "rb") as pkl_file:
        data = pkl.load(pkl_file)
    return data
