import os
import string
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLOR = {1: "red", 0: "blue"}


def get_symptom_to_id(symptoms_series):
    """Creates and returns an symptom id to symptom map.

    Parameters
    ----------
    symptoms_series: pd.Series
        pandas series object containging symptoms.

    Returns
    -------
    id_to_symptoms: dict
        dictionary maping a symptom id to corresponding symptom value.

    """
    symptoms_series = list(
        set(
            [symptom for symptoms in symptoms_series.to_numpy() for symptom in symptoms]
        )
    )
    id_to_symptoms = {idx: symptom for idx, symptom in enumerate(symptoms_series)}
    return id_to_symptoms


def get_patho_to_id(pathologies):
    """Creates and returns an pathology id to pathology map.

    Parameters
    ----------
    symptoms_series: pd.Series
        pandas series object containging the pathologies.

    Returns
    -------
    id_to_pathology: dict
        dictionary maping a pathology id to corresponding pathology value.

    """
    id_to_pathology = {
        idx: pathology for idx, pathology in enumerate(list(pathologies.unique()))
    }
    return id_to_pathology


def get_counts(data, key):
    """Return the distribution of data. The data corresponds to the specified key.

    Parameters
    ----------
    data: pd.Series
        pandas series object containing the data for the specified key.

    Returns
    -------
    result: Counter
        counter object containing the distribution of the data.

    """
    if key == "PATHOLOGY":
        return Counter(data.to_numpy())
    return Counter([symptom for symptoms in data.to_numpy() for symptom in symptoms])


def conditional_symptoms_subplots(data, title_prefix, output_path):
    """Plots the distribution of pathology conditioned on the geneder and age.

    Parameters
    ----------
    data: pd.DataFrame
        pandas dataframe containing data about the patients.
    title_prefix: str
        prefix added to name of the file.
    output_path: str
        path to the folder where the plots are to be stored.

    Returns
    -------
    None

    """
    path_gp = data.groupby("PATHOLOGY")
    for pathology, pathology_data in path_gp:
        pathology_data["SYMPTOMS"] = pathology_data["SYMPTOMS"].apply(
            lambda x: clean_symptoms(x)
        )
        id_to_symptoms = get_symptom_to_id(pathology_data["SYMPTOMS"])
        conditional_subplots(
            pathology_data,
            f"{title_prefix}_{pathology}",
            output_path,
            id_to_symptoms,
            "SYMPTOMS",
            "Symptoms",
            "Count",
        )


def conditional_subplots(
    data,
    title_prefix,
    output_path,
    id_to_val,
    dist_key,
    x_label,
    y_label,
    width=0.3,
    fontsize=36,
    label_fontsize=46,
    figsize=(70, 120),
):
    """Plots the distribution of dist_key conditioned on the geneder and age group.

    Parameters
    ----------
    data: pd.DataFrame
        pandas dataframe containing data about the patients.
    title_prefix: str
        prefix added to name of the file.
    output_path: str
        path to the folder where the plots are to be stored.
    id_to_val: dict
        a dictionary mapping value id to its corresponding value. It isused
        for getting symptom/pathology from their corresponding ids.
    dist_key: str
        column name in  the data for which conditional plot is to be generated.
    x_label: str
        x axis label.
    y_label: str
        y axis label.
    width: float
        width of the bars in the plot.
        Default: 0.3
    fontsize: int
        font size for x ticks.
        Default: 36
    label_fontsize: int
        font size for axes labels.
        Default: 46
    figsize: tuple
        a 2-long tuple, the first value determining the horizontal size of the
        ouputted figure, the second determining the vertical size.
        Default: (70, 120)

    Returns
    -------
    None

    """
    gender_colors = {"M": "blue", "F": "red"}

    age_groups = ["under 5 yo", "5-16yo", "16-35yo", "35-65yo", "65 and up"]
    x_vals = np.array((range(len(id_to_val))))

    plot_num = 0
    x_ticks_config = {
        "rotation": "vertical",
        "fontsize": fontsize,
        "ticks": x_vals - 0.1,
        "labels": [id_to_val[idx] for idx in range(len(id_to_val))],
    }

    plt.clf()
    fig2 = plt.figure(constrained_layout=True, figsize=figsize)
    spec2 = fig2.add_gridspec(
        nrows=len(age_groups), ncols=1, left=0.05, right=0.48, wspace=0.05
    )

    for gp in age_groups:
        age_gp_data = data[data["AGE_GROUPS"] == gp]
        if not age_gp_data.shape[0]:
            continue

        gender_gp = age_gp_data.groupby("GENDER")

        ax = fig2.add_subplot(spec2[plot_num, 0])
        ax.set_title(f"{title_prefix}_{gp}", fontsize=label_fontsize)
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        b1, b2 = None, None
        for gender, gp_data in gender_gp:
            count = get_counts(gp_data[dist_key], dist_key)
            y = [count.get(id_to_val[idx], 0) for idx in x_vals]
            if gender == "F":
                b1 = ax.bar(
                    x_vals - width,
                    y,
                    color=gender_colors[gender],
                    width=width,
                    align="center",
                    label=gender,
                )
            else:
                b2 = ax.bar(
                    x_vals,
                    y,
                    color=gender_colors[gender],
                    width=width,
                    align="center",
                    label=gender,
                )
        set_legend(ax, b1, b2, ("F", "M"), fontsize=fontsize)
        plot_num += 1
        plt.xticks(**x_ticks_config)
        plt.yticks(fontsize=label_fontsize)
    create_dir(output_path)
    plt.savefig(f"{output_path}/{title_prefix}.pdf", bbox_inches="tight")


def set_legend(ax, bar1, bar2, legends, fontsize):
    """Sets the legends for different bars sub-plots.

    Parameters
    ----------
    ax: AxesSubplot
        axes subplot containing bars bar1 and bar2.
    bar1: BarContainer
        it contains bars corresponding the 1st element of the
        legends tuple.
    bar2: BarContainer
        it contains bars corresponding the 1st element of the
        legends tuple.
    legends: tuple[str]
        tuple of strings containing legends name for the
        bars bar1 and bar2.
    fontsize: int
        size of font to display the legends.

    Returns
    -------
    None

    """
    if bar1 and bar2:
        ax.legend((bar1, bar2), legends, fontsize=fontsize)
    elif bar2:
        ax.legend((bar2), (legends[1]), fontsize=fontsize)
    elif bar1:
        ax.legend((bar1), (legends[0]), fontsize=fontsize)
    else:
        raise ValueError("Both bar1 and bar2 are None.")


# Helper function.
def create_dir(dir):
    """Creates a directory if it doesn't already exists.

    Parameters
    ----------
    dir: str
        directory name to be created.

    Returns
    -------
    None

    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def remove_punctuation(str_data):
    """Replaces punctuation from the str_data with `_`.

    Parameters
    ----------
    str_data: str
        input string with punctuations.

    Returns
    -------
    result: str
        string with punctuation replaced by `_`.

    """
    exclude = set(string.punctuation)
    result = "".join([ch if ch not in exclude else "_" for ch in str_data])
    return result


def clean(data, remove_punctuation_flag=True):
    """Removes unwanted characters from input data.

    Replaces commas and line breaks in the source string
    with a single space. if remove_punctuation_flag is set to
    true then it replaces the puctuation by `_`. This is
    required to create folders based on the name of the
    pathology.

    Parameters
    ----------
    data: str
        data string to be cleaned.
    remove_punctuation_flag: bool
        flag to determine if punctuation is to be removed.
        Default: True

    Returns
    -------
    result: str
        the resulting string.

    """
    result = data.replace("\r\n", " ")
    result = result.replace("\r", " ")
    result = result.replace("\n", " ")
    result = result.replace(",", " ")
    if remove_punctuation_flag:
        result = remove_punctuation(result)
    return result


def clean_symptoms(symptoms):
    """Cleans the input symptoms.

    Following clean-up operations are performed:
    1. Stripping the value of symptoms after `:`.
    2. Remove int value after `_@_`. for instance,
       douleurxx_intens_@_1 ->  douleurxx_intens.

    Parameters
    ----------
    symptoms: str
        it contains `;` separated string containing the symptoms that a
        patient has.

    Returns
    -------
    symptoms: list[str]
        cleaned list of symptoms.

    """
    symptoms = [symptom.split(":")[0] for symptom in str(symptoms).split(";")]
    symptoms = [
        symptom
        if "@" in symptom and not symptom.split("_@_")[1].isnumeric()
        else symptom.split("_@_")[0]
        for symptom in symptoms
    ]
    return symptoms


def plot_bar(
    x, y, x_ticks_config, xlabel, ylabel, title, fontsize=16, figsize=None, do_clf=True
):
    """Generates a customized bar plot of the input data.

    Parameters
    ----------
    x: list
        x axis values that should be shown on x axis.
    y: y
        y axis values that should be shown on y axis.
    x_ticks_config: dict
        dicitonary specifying the config for how to show the text on x axis.
    xlabel: str
        label for x axis.
    ylabel: str
        label for y axis.
    title: str
        title for the plot.
    fontsize: int
        size of the x labels title and y label in the plot.
        Default: 16
    figsize: tuple(int,int)
        width and height of the plot.
        Default: None
    do_clf: bool
        flag to specify if the plt.clf() needs to be calles or not.
        This is useful as one can set this to true for independent plot and to
        false if 2 plots need to be on the same plot.
        Default: True

    Returns
    -------
    barlist: plt.bar
        list of bars plotted.

    """
    if do_clf:
        plt.clf()
    if figsize:
        plt.figure(figsize=figsize)
    barlist = plt.bar(x, y)
    plt.xticks(**x_ticks_config)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    return barlist


def change_bar_color(barlist, bar_colors):
    """Changes the bar colors of bars in barlist.

    Changes the color of the bars in barlist to the
    corresponding colors in bar_colors.

    Parameters
    ----------
    barlist: plt.bar
        list of bars obtained as an output from a plot_bar function.
    bar_colors: list[str]
        list of colors for the corresponding bar in the barlist.

    Returns
    -------
    None

    """
    for i in range(len(bar_colors)):
        barlist[i].set_color(bar_colors[i])


# Plotting logic functions
def plot_patho_dist(data, title, patho_urgence_map, output_path="."):
    """This function plots the pathology distribution in the data.

    Parameters
    ----------
    data: pd.DataFrame
        pandas dataframe containing the patients.
    title: str
        title for the plot.
    patho_urgence_map: dictionary
        a dicitonary containing the mapping between the pathology name and
        their severity.
    output_path: str
        path to store the plot into.
        Default: "."

    Returns
    -------
    None

    """
    patho_counts = sorted(
        list(data.PATHOLOGY.value_counts().to_dict().items()), key=lambda x: x[0]
    )
    pathos, counts = zip(*patho_counts)
    barlist = plot_bar(
        pathos,
        counts,
        {"rotation": 90},
        "Pathology",
        "Count",
        f"Pathology Count plot for {title}",
        figsize=(18, 5),
    )
    change_bar_color(
        barlist, [COLOR.get(patho_urgence_map[patho], "blue") for patho in pathos]
    )
    create_dir(output_path)
    plt.savefig(f"{output_path}/{title}.pdf", bbox_inches="tight")


def plot_sympt_dist(data, title, output_path="."):
    """Plots the symptom distribution in the data.

    Parameters
    ----------
    data: pd.DataFrame
        pandas dataframe containing the patients.
    title: str
        title for the plot.
    output_path: str
        path to store the plot into.
        Default: "."

    Returns
    -------
    None

    """
    data["SYMPTOMS"] = data["SYMPTOMS"].apply(lambda x: clean_symptoms(x))
    symptoms_count = Counter(
        [symptom for symptoms in data["SYMPTOMS"].to_numpy() for symptom in symptoms]
    )
    symptoms_count = sorted(list(symptoms_count.items()), key=lambda x: x[0])

    symptoms, counts = zip(*symptoms_count)
    plot_bar(
        symptoms,
        counts,
        {"rotation": 90},
        "symptoms",
        "Count",
        f"Symptom Count plot for {title}",
        figsize=(48, 5),
    )
    create_dir(output_path)
    plt.savefig(f"{output_path}/{title} symptom.pdf", bbox_inches="tight")


def plot_group_by(
    data, columns, title, patho_urgence_map, output_path=".", plot_type="pathology"
):
    """Plots conditional symptoms and pathology distribution.

    Groups the data using the column names specified in the columns
    and plot the symptoms nd pathology distribution.

    Parameters
    ----------
    data: pd.DataFrame
        pandas dataframe containing the patients.
    title: str
        title for the plot.
    patho_urgence_map: dictionary
        a dicitonary containing the mapping between the pathology name and
        their severity.
    output_path: str
        path to store the plot into.
        Default: "."
    plot_type: str
        whether to plot pathology or symptom distribution. There is a difference
        in terms of how the counts are calculated and notion of severity only
        applies to pathology.
        Default: "pathology"

    Returns
    -------
    None

    """
    column = columns.pop()
    data_group = data.groupby(column)
    total_rows = 0
    for name, group in data_group:
        name = clean(name)
        total_rows += group.shape[0]
        if not columns:
            if plot_type == "pathology":
                plot_patho_dist(
                    group, f"{title} {column}_{name}", patho_urgence_map, output_path
                )
            else:
                plot_sympt_dist(group, f"{title} {column}_{name}", output_path)
        else:
            plot_group_by(
                group,
                list(columns),
                f"{title} {column}_{name}",
                patho_urgence_map,
                output_path=f"{output_path}/{name}",
                plot_type=plot_type,
            )
    assert total_rows == data.shape[0]


def plot_patho_perf(data, title, patho_urgence_map, metrics_to_plot, output_path="."):
    """Plots all metrics of interest on the same plot using subplots.

    Parameters
    ----------
    data: dictionary
        dictionary with metrics of interest as key and values as their
        values for all the pathologies
    title: str
        title for the plot.
    patho_urgence_map: dictionary
        a dicitonary containing the mapping between the pathology name and
        their severity
    metrics_to_plot: list[str]
        list of metrics that need to plotted. These metrics need to be present in
        data dictionary.
    output_path: str
        path to store the plot into.
        Default: "."

    Returns
    -------
    None

    """
    assert all(True if metric in data else False for metric in metrics_to_plot)
    x = range(len(data.get("precision", [])))
    color_mapping = [
        COLOR.get(patho_urgence_map[patho], "blue") for patho in data.get("pathos", [])
    ]
    plt.clf()
    plt.subplots_adjust(hspace=0.2)

    for plot_num in range(1, len(metrics_to_plot) + 1):
        plt.subplot(len(metrics_to_plot), 1, plot_num)
        # This will create the bar graph for poulation.
        if plot_num != len(metrics_to_plot):
            x_ticks_config, xlabel = {"ticks": [], "labels": []}, ""
        else:
            x_ticks_config, xlabel = (
                {
                    "ticks": x,
                    "labels": data["pathos"],
                    "rotation": "vertical",
                    "fontsize": 26,
                },
                "Pathology",
            )

        pop = plot_bar(
            x,
            data[metrics_to_plot[plot_num - 1]],
            x_ticks_config,
            xlabel,
            metrics_to_plot[plot_num - 1],
            title,
            do_clf=False,
            fontsize=36,
        )

        change_bar_color(pop, color_mapping)
    plt.savefig(f"{output_path}/{title}.png", bbox_inches="tight")

    # plt.show()


def print_confusion_matrix(
    confusion_matrix,
    class_names,
    normalize=False,
    figsize=(30, 27),
    fontsize=25,
    ax=None,
    output_path="./",
):
    """Plots a confusion matrix as a heatmap.

    The confusion matrix is returned by sklearn.metrics.confusion_matrix.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        the numpy.ndarray object returned by sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        an ordered list of class names, in the order they index the given
        confusion matrix.
    figsize: tuple
        a 2-long tuple, the first value determining the horizontal size of the
        ouputted figure, the second determining the vertical size.
        Default: (30, 27)
    fontsize: int
        font size for axes labels.
        Default: 25
    ax: matplotlib.axes
        a matplotlib axes on which the confusion matrix can be plotted.
        Default: None
    output_path: str
        path to save the plotted confusion matrix in.
        Default: "./"

    Returns
    -------
    None

    """
    if normalize:
        confusion_matrix = (
            confusion_matrix.astype("float")
            / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # fig = plt.figure(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, ax=ax)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    heatmap.set_ylabel("True label", fontsize=int(fontsize * 1.5))
    heatmap.set_xlabel("Predicted label", fontsize=int(fontsize * 1.5))
    plt.title("Confusion Matrix plot", fontsize=int(fontsize * 1.5))
    plt.savefig(f"{output_path}/cm.png", bbox_inches="tight")


def plot_inq_symptoms(inquired_symptoms_count, symptom_to_id_map, title, output_path):
    """Plots the inquired symptoms in the a particular CC by the agent.

    Parameters
    ----------
    inquired_symptoms_count: dictionary
        dictionary with keys as the symptom id and values as the normalized
        count of the times that symptom was inquired.
    symptom_to_id_map: dictionary
        dictionary containing the
    title: str
        title for the plot.
    output_path: str
        path to save the plotted confusion matrix in.

    Returns
    -------
    None

    """
    id_tosymptom_map = {val: key for key, val in symptom_to_id_map.items()}
    symptoms, counts = zip(*inquired_symptoms_count.items())
    symptoms = [id_tosymptom_map.get(int(symptom_id), "") for symptom_id in symptoms]
    plot_bar(
        symptoms,
        counts,
        {"rotation": 90},
        "Symptoms",
        "Normalized Count",
        "Plot shows how many patients were asked about a symptom on an average.",
        figsize=(18, 5),
    )

    plt.savefig(f"{output_path}/{title}.png", bbox_inches="tight")
