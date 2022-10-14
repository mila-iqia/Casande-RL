import argparse
import logging
from collections import defaultdict

import numpy as np

from chloe.utils.plot_utils import (
    clean,
    conditional_subplots,
    conditional_symptoms_subplots,
    create_dir,
    get_patho_to_id,
    plot_inq_symptoms,
    plot_patho_dist,
    plot_patho_perf,
    print_confusion_matrix,
)
from chloe.utils.read_utils import read_csv_zip, read_json, read_yaml

logging.basicConfig(level=logging.INFO)


def extract_per_patho_metrics_data(data, metric_to_extract):
    """Extracts metrics information for each pathology.

    This functions extracts the specified metrics information
    from the data dictionary for each pathology in the CC of
    interest.

    Parameters
    ----------
    data: dict
        dictionary containg data on the metrics of interest.
    metric_to_extract: list[str]
        list containing metrics that are to be extracted from
        data dictionary.

    Returns
    -------
    extracted_data: dict
        dictionary with metrics of interest as key and values as their
        values for all the pathologies.

    """
    patho_mapping = {
        str(val): key for key, val in data.get("pathos", [[], {}])[1].items()
    }
    extracted_data = defaultdict(list)
    for key in sorted(data.get("per_patho", {}).keys()):
        extracted_data["pathos"].append(patho_mapping[key])
        for metric in metric_to_extract:
            extracted_data[metric].append(data.get("per_patho", None)[key][metric])
    return extracted_data


def get_age_group(age):
    """Buckets the patient ages as specified in Julien's mail.

    Parameters
    ----------
    age: int
        data string to be cleaned.

    Returns
    -------
    result: str
        the resulting string.

    """
    if 0 <= age < 5:
        return "under 5 yo"
    elif 5 <= age < 16:
        return "5-16yo"
    elif 16 <= age < 35:
        return "16-35yo"
    elif 35 <= age < 65:
        return "35-65yo"
    elif 65 <= age < 250:
        return "65 and up"
    else:
        raise ValueError("Age should be in between 0 and 250.")


def create_age_gps(data, colname):
    """Adds a column for age groups to the data.

    Parameters
    ----------
    data: pd.DataFrame
        pandas dataframe containing patients from either
        train/validate/test.zip files.
    colname: str
        name of the new that should have the patients grouped
        by age.

    Returns
    -------
    data: pd.DataFrame
        pandas dataframe containing patients from either
        train/validate/test.zip files with additonal column with name
        as colname and value as the age group.

    """
    assert "AGE_BEGIN" in data  # Pre-condition

    data[colname] = data.apply(lambda x: get_age_group(x.AGE_BEGIN), axis=1)

    assert colname in data  # Post-condition

    return data


def get_patho_urgence_map(data):
    """Creates a dictinary mapping the pathology to its severity.

    Parameters
    ----------
    data: dict
        this a dictionary object from thee corresponding *_conditions.json.
        file.

    Returns
    -------
    patho_urgence_map: dict
        a dictionary containing the mapping between the pathology name and
        their severity.

    """
    patho_urgence_map = {}
    for key, value in data.items():
        assert "condition_name" in value, "condition_name missing"
        assert "urgence" in value, "urgence of the condition is missing"
        patho_urgence_map[clean(value["condition_name"], False)] = (
            1 if (value["urgence"] == 1 or value["urgence"] == 2) else 0
        )
    return patho_urgence_map


def valid_test_plots(
    cc, data_type, metrics_file, metrics_to_plot, cc_output_path, patho_urgence_map
):
    """Plots all the graphs using data from the json metrics file.

    Parameters
    ----------
    cc: str
        chief Complaint for which the graphs are being plotted.
    data_type: str
        type of the dataset under consideration (train/validate/test).
    metrics_file: str
        path to the metrics.json file.
    metrics_to_plot: list[str]
        list of metrics that need to plotted. These metrics need to be present in
        data dictionary.
    cc_output_path: str
        path to save the plotted confusion matrix in.
    patho_urgence_map: dict
        a dictionary containing the mapping between the pathology name and
        their severity.

    Returns
    -------
    None

    """
    eval_metrics = read_json(metrics_file)
    # Plotting the confusion matrix.
    print_confusion_matrix(
        np.array(eval_metrics["global"]["confusion_matrix"]),
        [
            eval_metrics["pathos"][0][i] if i != -1 else "N/A"
            for i in eval_metrics["global"]["confusion_matrix_support"]
        ],
        output_path=cc_output_path,
    )
    # Plotting the distribution  of inquired symptoms.
    inquired_symptoms_count_global = eval_metrics["global"]["inquired_symptoms_count"]
    plot_inq_symptoms(
        inquired_symptoms_count_global,
        eval_metrics["symptoms"][1],
        f"{cc}_{data_type}_inquired_symtpoms",
        output_path=cc_output_path,
    )
    # Plotting performance plots per pathology.
    eval_metrics_to_plot = extract_per_patho_metrics_data(eval_metrics, metrics_to_plot)
    plot_patho_perf(
        eval_metrics_to_plot,
        f"{cc}_{data_type}_perf",
        patho_urgence_map,
        metrics_to_plot,
        output_path=cc_output_path,
    )


def get_field(input_data, key):
    """Returns the value of key in input data.

    If the key is not present then it raises an exception.

    Parameters
    ----------
    input_data: dict
        dictionary containing value for the specified key.
    key: str
        key whose value is to be retrieved from the dictionary.

    Returns
    -------
    data: dict/list/str
        value of the key in input_data. The output type may change depending on the
        key.

    """
    data = input_data.get(key, None)
    if not data:
        raise AttributeError(
            f"{key} not found. Kindly modify the config to add this key"
        )
    return data


def main(config_file):
    """This function handles overall generation of plots.

    Parameters
    ----------
    config_file: str
        path to the file containing the information about the
        data to be plotted, metrics to be plotted, etc.

    Returns
    -------
    None

    """
    # Load plot config from config file.
    plot_config = read_yaml(config_file)
    data_types = get_field(plot_config, "data_types")
    metrics_to_plot = get_field(plot_config, "metrics_to_plot")
    paths = get_field(plot_config, "paths")
    output_path = get_field(plot_config, "output_path")

    for cc, cc_paths in paths.items():
        logging.info(f"at cc {cc}")

        cc_output_path = f"{output_path}{cc}"
        create_dir(cc_output_path)

        patho_data = read_json(cc_paths["condition_file"])
        patho_urgence_map = get_patho_urgence_map(patho_data)

        for data_type in data_types:
            logging.info(f"at cc {data_type}")
            data = read_csv_zip(cc_paths[data_type])

            create_age_gps(data, "AGE_GROUPS")
            # Symptoms distribution plot conditioned on age and gender.
            conditional_symptoms_subplots(
                data, f"{cc}_{data_type}", f"{cc_output_path}/symptoms"
            )
            # Pathology distribution plot conditioned on age and gender.
            conditional_subplots(
                data,
                f"{cc}_{data_type}",
                f"{cc_output_path}/pathology",
                get_patho_to_id(data["PATHOLOGY"]),
                "PATHOLOGY",
                "Pathology",
                "Count",
            )
            # Pathology distribution plot across the dataset.
            plot_patho_dist(
                data, f"{cc}_{data_type}", patho_urgence_map, output_path=cc_output_path
            )
            # Plots for the evaluation metrics for the agent.
            if (
                "valid" in data_type and cc != "all_patients"
            ):  # There is no a eval file for all patients as we train model per CC.
                valid_test_plots(
                    cc,
                    data_type,
                    cc_paths["eval_metrics"],
                    metrics_to_plot,
                    cc_output_path,
                    patho_urgence_map,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_filepath",
        help="path to a yaml file containing information to plot.",
        type=str,
        default="chloe/plotting/plot_config.yaml",
    )
    args = parser.parse_args()
    main(args.config_filepath)
