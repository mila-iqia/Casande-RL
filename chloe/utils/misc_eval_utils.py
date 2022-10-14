import functools
import json
from multiprocessing import Pool

import numpy as np

from chloe.utils.dist_metric import (
    dist_js_div,
    kl_confirm_score,
    kl_explore_score,
    kl_trajectory_auc,
    numpy_get_pathos_inout_ratio,
    numpy_get_severe_pathos_inout_ratio,
    numpy_softmax,
)
from chloe.utils.sim_utils import decode_geo, decode_sex


def get_patient_weight_factor(pathoIndex_2_key, weight_data, patho, sex, geo, age):
    """Utility function to get the weight of the patient given his information.

    Parameters
    ----------
    pathoIndex_2_key :  dict(int -> str)
        map of the patho indices to their key within the condition json file.
    weight_data :  dict
        the data containing the weights of the patho of the form:
        (patho, sex, geo) -> [(AgeMin, AgeMax, Factor)*]
    patho :  int
        index of the patho of interest
    sex :  int
        sex code of the patient of interest
    geo :  int
        geographic code of the patient of interest
    age :  int
        age of the patient of interest
    Returns
    -------
    result: float
        the weight
    """
    if (pathoIndex_2_key is None) or (weight_data is None):
        return 1.0
    if (sex is None) or (geo is None) or (age is None):
        return 1.0
    sex = decode_sex(sex) if not isinstance(sex, str) else sex
    geo = decode_geo(geo) if not isinstance(geo, str) else geo
    patho = pathoIndex_2_key[patho] if not isinstance(patho, str) else patho
    all_options = weight_data.get(patho, {}).get(sex, {}).get(geo, [])
    if len(all_options) == 0:
        assert False, f"({patho}, {sex}, {geo}, {age})"  # that should not happen
        return 1.0
    else:
        possibleWeights = []
        for a in all_options:
            if (a["AgeMin"] <= age) and (age <= a["AgeMax"]):
                possibleWeights.append(a["Factor"])
        if len(possibleWeights) == 0:
            assert False, f"({patho}, {sex}, {geo}, {age})"  # that should not happen
            return 1.0
        elif len(possibleWeights) > 1:
            assert True, f"({patho}, {sex}, {geo}, {age})"  # for age at bins border
            return max(possibleWeights)
        else:
            return possibleWeights[0]


def apply_consecutive_difference_func(data, func, **kwargs):
    """Utility function to apply consecutive difference using `func`.

    Parameters
    ----------
    data :  sequence
        data on which to perform the operation.
    func :  callable
        function to be used.
    kwargs :  dict
        variable arguments.
    Returns
    -------
    result: np.ndarray
        computed difference
    """
    return np.concatenate(
        (
            np.array([0]),
            func(data[1:], data[0:-1], **kwargs) if len(data) > 1 else np.array([]),
        )
    )


def apply_sorting_func(data, axis=-1, arg=False, reverse=False):
    """Utility function to sort data.

    Parameters
    ----------
    data :  sequence
        data on which to perform the operation.
    axis: int
        the axis for sorting.
    arg :  bool
        whether or not to get arg indidices (arg max).
    reverse :  bool
        whether or not the sort is reversed.
    Returns
    -------
    result: np.ndarray
        sorted data
    """
    func = np.sort if not arg else np.argsort
    result = func(data, axis=axis)
    if reverse:
        result = np.swapaxes(result, 0, axis)[::-1]
        result = np.swapaxes(result, 0, axis)
    return result


def get_topk_probability_mass(
    sorted_proba, sorted_indices, true_indices, k=None, normalize=True
):
    """Computes the topk probability mass of succesive predictions of an interaction.

    Parameters
    ----------
    sorted_proba: list
        the sorted predicted probability values for each turn.
        This tensor represents the probability values, not logits.
    sorted_indices: int, np.array
        the sorted indices associated to `sorted_proba`.
    true_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    k: int
        The number of top element to be considered in the predicted distribution.
        if none, then take the number of pathologies in the target distribution.
    normalize: bool
        Flag indicating whether or not to normalize the result. Default: True.

    Return
    ----------
    result: list
        the computed metric for each turn.
    """
    if k is None:
        k = 1 if true_indices is None else np.sum(np.array(true_indices) != -1)
    sorted_proba = np.array(sorted_proba)
    values = (sorted_proba[:, :k]).sum(axis=1)
    result = values.tolist()
    return result


def get_split_topk_probability_mass(
    sorted_proba, sorted_indices, true_indices, k=None, normalize=True
):
    """Computes the topk probability mass of succesive predictions of an interaction.
    Here the computation distinguishes between pathologies in the gt differentials
    and those that are not.
    Parameters
    ----------
    sorted_proba: list
        the sorted predicted probability values for each turn.
        This tensor represents the probability values, not logits.
    sorted_indices: int, np.array
        the sorted indices associated to `sorted_proba`.
    true_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    k: int
        The number of top element to be considered in the predicted distribution.
        if none, then take the number of pathologies in the target distribution.
    normalize: bool
        Flag indicating whether or not to normalize the result. Default: True.

    Return
    ----------
    result: list
        the computed metric for each turn.
    """
    if k is None:
        k = 1 if true_indices is None else np.sum(np.array(true_indices) != -1)
    if true_indices is None:
        true_indices = []
    true_indices = np.array(true_indices)
    kmax = np.sum(true_indices != -1)

    sorted_proba = np.array(sorted_proba)
    sorted_indices = np.array(sorted_indices)

    values = sorted_proba[:, :k]
    consider_indices = true_indices[:kmax]

    considered_indices = [
        np.intersect1d(sorted_indices[i, :k], consider_indices, return_indices=True)[1]
        for i in range(len(sorted_indices))
    ]
    # cum weight of patho within the gt diff, cum weight of patho not in gt diff
    result = [
        (values[i, a].sum(), values[i].sum() - values[i, a].sum())
        for i, a in enumerate(considered_indices)
    ]
    return result


def remove_mass(pred_idx_proba, truth_idx_proba):
    """Utility function to remove mass from the item, mass pair."""
    pred = [[val[0] for val in patient] for patient in pred_idx_proba]
    truth = [[val[0] for val in patient] for patient in truth_idx_proba]
    return pred, truth


def combine_proba_idx_mass(
    all_differential_probas, all_differential_indices, cond_index_2_name=None
):
    """Combines the differential probabilities with their ids."""
    output = []
    for idx in range(len(all_differential_indices)):
        curr_diff_idx = all_differential_indices[idx]
        curr_diff_proba = all_differential_probas[idx]
        if curr_diff_idx is None:
            output.append(None)
            continue
        first_non_zerro_idx = len(curr_diff_idx)
        for patho_idx, val in enumerate(curr_diff_idx):
            if val == -1:
                first_non_zerro_idx = patho_idx
                break
        curr_diff_idx = [
            str(i) if cond_index_2_name is None else cond_index_2_name[i]
            for i in curr_diff_idx[:first_non_zerro_idx]
        ]
        curr_diff_proba = curr_diff_proba[:first_non_zerro_idx]
        if not curr_diff_idx:
            continue
        assert abs(sum(curr_diff_proba) - 1.0) < 0.0001
        combined_pathos_idx_proba = list(zip(curr_diff_idx, curr_diff_proba))
        combined_pathos_idx_proba.sort(key=lambda x: x[1], reverse=True)
        output.append(combined_pathos_idx_proba)
    return output


def get_weight(data_stats, pathoIndex_2_key, weight_data):
    """Utility function to extract the weight factor of each patient from input.

    Parameters
    ----------
    data_stats :  dict
        evaluation stats generated from evaluating a model.
    pathoIndex_2_key :  dict(int -> str)
        map of the patho indices to their key within the condition json file.
    weight_data :  dict
        the data containing the weights of the patho of the form:
        (patho, sex, geo) -> [(AgeMin, AgeMax, Factor)*]
    Returns
    -------
    result: list
        the weight for each patient in the data_stats file
    """
    all_patho = data_stats["data"]["y_true"]
    all_sex = data_stats["data"].get("sex", None)
    all_geo = data_stats["data"].get("geo", None)
    all_age = data_stats["data"].get("age", None)

    if (pathoIndex_2_key is None) or (weight_data is None):
        return [1.0] * len(all_patho)

    data_not_present = (all_age is None) or (all_sex is None) or (all_geo is None)
    if data_not_present:
        return [1.0] * len(all_patho)

    result = [
        get_patient_weight_factor(pathoIndex_2_key, weight_data, patho, sex, geo, age)
        for patho, sex, geo, age in zip(all_patho, all_sex, all_geo, all_age)
    ]
    return result


def get_pred_truth(data_stats, cond_index_2_name=None):
    """Utility function to extract the relevant data from input.

    This function exptracts the predicted and ground truth differential diagnosis
    information from the input data.
    Parameters
    ----------
    data_stats :  dict
        evaluation stats generated from evaluating a model.
    cond_index_2_name :  list
        list of pathologies. Default: None.
    Returns
    -------
    None
    """
    all_differential_indices = data_stats["data"]["y_differential_indices"]
    all_differential_probas = data_stats["data"]["y_differential_probas"]
    combined_pathos_idx_proba = combine_proba_idx_mass(
        all_differential_probas, all_differential_indices, cond_index_2_name
    )
    all_patho = [
        str(i) if cond_index_2_name is None else cond_index_2_name[i]
        for i in data_stats["data"]["y_true"]
    ]
    combined_pathos_idx_proba = [
        a if a is not None else [[all_patho[i], 1.0]]
        for i, a in enumerate(combined_pathos_idx_proba)
    ]
    all_proba_dist = data_stats["data"]["all_proba_dist"]
    all_proba_dist = [
        patient_probas_dist[-1].tolist() for patient_probas_dist in all_proba_dist
    ]
    all_proba_dist = normalize(all_proba_dist)
    all_proba_dist_idx = combine_pred_proba_idx_mass(all_proba_dist, cond_index_2_name)
    return all_proba_dist_idx, combined_pathos_idx_proba, all_patho


def combine_pred_proba_idx_mass(all_proba_dist, cond_index_2_name=None):
    """Utility function to combine item id and its probability for sorting.
    Parameters
    ----------
    all_proba_dist :  list
        a list of probability distributions.
    Returns
    -------
    output: list
        input item ids mapped to item id and its probability pair.
    """
    output = []
    for dist in all_proba_dist:
        idxs = [
            str(i) if cond_index_2_name is None else cond_index_2_name[i]
            for i in range(len(dist))
        ]
        dist_idx = list(zip(idxs, dist))
        assert abs(sum(dist) - 1.0) < 0.001
        dist_idx.sort(key=lambda x: x[1], reverse=True)
        output.append(dist_idx)
    return output


def normalize(data):
    """Normalizes the inputs to sum up to 1."""
    for idx in range(len(data)):
        data[idx] = numpy_softmax(data[idx])
        assert abs(sum(data[idx]) - 1.0) < 0.001
    return data


def filter_by_mass(data, mass, pred_split_token=";", diag_split_token=":"):
    """Filters the input data with the specified mass."""
    if not data:
        return []
    if isinstance(data, str):
        data = data.split(pred_split_token)
    output = []

    for diag_prob in data:
        diag, prob = diag_prob.split(diag_split_token)
        if float(prob) > mass or not output:
            output.append(diag)
    return output


def intersect_cols(
    data, pred_col, truth_col, k, topk_pred_col, topk_truth_col, measure, adjust_k=False
):
    """Calculates intersection between the pred_col and truth_col.

    Parameters
    ----------
    data: pd.Series
        a pandas series containing the pred_col and truth_col columns.
    pred_col: str
        a string corresponding to the prediciton column.
    truth_col: str
        a string corresponding to the ground truth column.
    k: int
        represents the subset size to consider for the
        `pred_col` and `truth_col`.
    topk_pred_col: bool
        a boolean indicating if `pred_col` needs to be sliced
        for the 1st k elements.
    topk_truth_col: bool
        a boolean indicating if `truth_col` needs to be sliced
        for the 1st k elements.
    measure: str
        a string that will decide whether the normalization is
        done using the `truth_col` length (for recall) or
        `pred_col` (for precision).

    Returns
    -------
    result: float
        percentage of pred_col in truth_col.

    """
    assert measure in ["recall", "precision"]
    if topk_truth_col or topk_pred_col:
        assert k > 0
    if adjust_k:
        k = min(k, len(data[truth_col]))
    pred = data[pred_col][:k] if topk_pred_col else data[pred_col]
    truth = data[truth_col][:k] if topk_truth_col else data[truth_col]
    pred = set(pred)
    truth = set(truth)
    denominator = truth if measure == "recall" else pred
    return len(truth.intersection(pred)) / max(len(denominator), 1)


def get_f1(data, pcol, rcol, fillna=True, w_col=None):
    """Calculates the f1 score for the specified column.
    """
    f1 = data[[pcol, rcol]].apply(lambda x: compute_f1(x[pcol], x[rcol]), axis=1)
    if fillna:
        f1 = f1.fillna(0)
    if w_col is None:
        metric = f1.mean()
    else:
        sum_weight = data[w_col].sum()
        if sum_weight == 0:
            sum_weight = 1.0
        metric = data[w_col].multiply(f1).sum() / sum_weight
    return metric, f1


def get_metric(
    data,
    pred_col,
    truth_col,
    k=-1,
    topk_pred_col=False,
    topk_truth_col=False,
    measure="recall",
    adjust_k=False,
    w_col=None,
):
    """Returns the statistics of the `pred_col` for specified measure.

    Parameters
    ----------
    data: pd.DataFrame
        a pandas dataframe containing the pred_col and truth_col columns.
    pred_col: str
        a string corresponding to the prediciton column.
    truth_col: str
        a string corresponding to the ground truth column.
    k: int
        represents the subset size to consider for the
        `pred_col` and `truth_col`.
        Default: -1.
    topk_pred_col: bool
        a boolean indicating if `pred_col` needs to be sliced
        for the 1st k elements.
        Default: False.
    topk_truth_col: bool
        a boolean indicating if `truth_col` needs to be sliced
        for the 1st k elements.
        Default: False.
    measure: str
        a string that will decide whether the normalization is
        done using the `truth_col` length (for recall) or
        `pred_col` (for precision).
        Default: `recall`.
    adjust_k: bool
        Whether or not to adjust k.
        Default: False
    w_col: str
        name of the weight factor column.

    Returns
    -------
    result: dict
        a dictionary containing various statistics like mean, median,
        etc. for the recall of `pred_col`.
    pred_truth_intersection: pd.Series
        a pandas series object that contains the calculated measure
        for each row of the input data.
    """
    if adjust_k:
        print(pred_col, truth_col, "this is imp line")
        print(data[pred_col].apply(lambda x: len(x)).mean())
    pred_truth_intersection = data[[pred_col, truth_col]].apply(
        lambda x: intersect_cols(
            x,
            pred_col,
            truth_col,
            k,
            topk_pred_col,
            topk_truth_col,
            measure=measure,
            adjust_k=adjust_k,
        ),
        axis=1,
    )
    if w_col is None:
        metric = pred_truth_intersection.mean()
    else:
        sum_weight = data[w_col].sum()
        if sum_weight == 0:
            sum_weight = 1.0
        metric = data[w_col].multiply(pred_truth_intersection).sum() / sum_weight
    return metric, pred_truth_intersection


def get_human_accuracy(data, columns, weight_col=None):
    """Calculates the accuracy of the specified column.

    Checks if the array in column at idx 0 of columns contains the
    the elemnts of the column at idx 1 of the columns array.

    Parameters
    ----------
    data: pd.DataFrame
        a dataframe containing columns required to calcualte the
        accuracy from.

    columns: List[str]
        a list of column name required for calculating the accuracy.

    weight_col: str
        column name of the weight factor.
    Returns
    -------
    result: dict
        a dictionary containing the statistcs about the accuracy
        across rows.
    """
    accuracy_series = data[columns].apply(
        lambda x: int(x[columns[1]] in x[columns[0]]), axis=1
    )
    if weight_col is None:
        return accuracy_series.mean()
    else:
        sum_weight = data[weight_col].sum()
        if sum_weight == 0:
            sum_weight = 1.0
        return data[weight_col].multiply(accuracy_series).sum() / sum_weight


def compute_at_k_metrics(data, metrics, ks, hard_check, pred_col, gt_col, wei_col=None):
    """Utility function to calculate topk based differential diagnosis metrics.

    Parameters
    ----------
    data :  pd.DataFrame
        data on which to perform the operation.
    metrics : dict
        a dictionary to which the computed metrics are to be added.
    ks: list
        a list of values of ks for which the metric is to be calculated.
    pred_col: str
        name of the prediction column.
    gt_col: str
        name of the ground truth column.
    wei_col: str
        name of the weight factor column.
    Returns
    -------
    None
    """
    for k_val in ks:
        data[f"{pred_col}@{k_val}"] = data[pred_col].apply(lambda x: x[:k_val])
        metrics[f"GTPA@{k_val}"] = get_human_accuracy(
            data, [f"{pred_col}@{k_val}", "PATHOLOGY"], wei_col
        )
        (metrics[f"DDR@{k_val}"], data[f"DDR@{k_val}"],) = get_metric(
            data, pred_col, gt_col, k_val, True, True, "recall", False, wei_col
        )
        (metrics[f"DDP@{k_val}"], data[f"DDP@{k_val}"],) = get_metric(
            data, pred_col, gt_col, k_val, True, True, "precision", False, wei_col
        )
        metrics[f"DDF1@{k_val}"], data[f"DDF1@{k_val}"] = get_f1(
            data, f"DDP@{k_val}", f"DDR@{k_val}", True, wei_col
        )
        if hard_check:
            assert (
                data[f"{pred_col}@{k_val}"]
                .apply(lambda x: 1 if len(x) == k_val else 0)
                .sum()
                == data.shape[0]
            )
        else:
            assert (
                data[f"{pred_col}@{k_val}"]
                .apply(lambda x: 1 if len(x) <= k_val else 0)
                .sum()
                == data.shape[0]
            )


def compute_mass_scenario(data, metrics, pred_col, gt_col, scenario_prefix, w_col=None):
    """Utility function to calculate mass based differential diagnosis metrics.

    Parameters
    ----------
    data :  pd.DataFrame
        data on which to perform the operation.
    metrics : dict
        a dictionary to which the computed metrics are to be added.
    pred_col: str
        name of the prediction column.
    gt_col: str
        name of the ground truth column.
    scenario_prefix: str
        prefix to be added to calculated metric keys.
    w_col: str
        name of the weight factor column.
    Returns
    -------
    None
    """
    (metrics[f"{scenario_prefix}DDR"], data[f"{scenario_prefix}DDR"],) = get_metric(
        data, pred_col, gt_col, 5, False, False, "recall", False, w_col
    )
    (metrics[f"{scenario_prefix}DDP"], data[f"{scenario_prefix}DDP"],) = get_metric(
        data, pred_col, gt_col, 5, False, False, "precision", False, w_col
    )
    metrics[f"{scenario_prefix}DDF1"], data[f"{scenario_prefix}DDF1"] = get_f1(
        data, f"{scenario_prefix}DDP", f"{scenario_prefix}DDR", True, w_col
    )
    metrics[f"{scenario_prefix}GTPA"] = get_human_accuracy(
        data, [pred_col, "PATHOLOGY"], w_col
    )


def indexed_function_call_keys(index, sequential, params, kwargs={}):
    f = params[0]
    val = [
        global_dict[params[i]][index]
        if (
            isinstance(params[i], str)
            and params[i] in global_dict
            and hasattr(global_dict[params[i]], "__len__")
            and index < len(global_dict[params[i]])
        )
        else (
            global_dict[params[i]]
            if (isinstance(params[i], str) and params[i] in global_dict)
            else params[i]
        )
        for i in range(1, len(params))
    ]
    result = (
        [f(a, *val[1:], **kwargs) if a is not None else None for a in val[0]]
        if sequential
        else f(*val, **kwargs)
    )
    return result


def compute_ratio(data, denominator, flip_flag, *args):
    """Utility function to compute data ratio.

    Parameters
    ----------
    data :  sequence
        data on which to perform the operation.
    denominator: int
        the denominator.
    flip_flag: list
        list of boolean indicating whwther or not to flip variable args
    args: list
        list of arguments
    Returns
    -------
    result: float
        computed ratio
    """
    if len(args) > 0:
        assert len(flip_flag) == len(args)
        args = [data] + list(args)
        flip_flag = [False] + flip_flag
        seq = []
        tot_denom = 0
        for values in zip(*args):
            val = values[0]
            denom = 1
            for i in range(1, len(values)):
                field = values[i] if not flip_flag[i] else (1 - values[i])
                val *= field
                denom *= field
            seq.append(val)
            tot_denom += denom
        final_denom = tot_denom if denominator is None else denominator
        final_denom = 1.0 if final_denom == 0 else final_denom
        return np.sum(seq) / final_denom
    else:
        return np.sum(data) / (1 if denominator == 0 else denominator)


def get_average_state_from_percent(data, percent=0.0, end_percent=1.0, normalize=True):
    """Get the average stat from a data sequence starting at a given percent.
    Parameters
    ----------
    data:  sequence
        the provided data.
    percent: float, list, tuple
        the provided percentage. Default: 0.0.
    end_percent: float, list, tuple
        the provided end percentage. Default: 1.0.
    normalize: boolean
        whether to normalize
    Returns
    -------
    result: float, list
        the computed average stat.
    """
    assert not (percent is None and end_percent is None)
    if percent is None:
        percent = end_percent
    elif end_percent is None:
        end_percent = percent

    assert isinstance(percent, (int, float, list, tuple))
    assert isinstance(end_percent, (int, float, list, tuple))
    extract_flag = False
    if isinstance(percent, (int, float)) and isinstance(end_percent, (int, float)):
        percent = [percent]
        end_percent = [end_percent]
        extract_flag = True
    elif isinstance(percent, (int, float)):
        percent = [percent] * len(end_percent)
    elif isinstance(end_percent, (int, float)):
        end_percent = [end_percent] * len(percent)
    assert len(percent) == len(end_percent)

    length = len(data)
    result = []
    for b, e in zip(percent, end_percent):
        assert b >= 0.0 and b <= 1.0
        assert e >= 0.0 and e <= 1.0
        assert e >= b
        i = int(length * b)
        j = int(length * e)
        j = j + 1 if j == i else j
        j = min(length, j)
        i = i - 1 if i == length else i
        n = j - i
        r = sum(data[i:j])
        r = r / max(1, n) if normalize else r
        result.append(r)
    if len(result) == 1 and extract_flag:
        result = result[0]
    return result


def get_diff_state_from_percent(data, percent=0.0, end_percent=1.0, normalize=True):
    """Get the difference stat from a data sequence starting at a given percent.
    Parameters
    ----------
    data:  sequence
        the provided data.
    percent: float, list, tuple
        the provided percentage. Default: 0.0.
    end_percent: float, list, tuple
        the provided end percentage. Default: 1.0.
    normalize: boolean
        whether to normalize. Default=True
    Returns
    -------
    result: float, list
        the computed average stat.
    """
    assert not (percent is None and end_percent is None)
    if percent is None:
        percent = end_percent
    elif end_percent is None:
        end_percent = percent

    assert isinstance(percent, (int, float, list, tuple))
    assert isinstance(end_percent, (int, float, list, tuple))
    extract_flag = False
    if isinstance(percent, (int, float)) and isinstance(end_percent, (int, float)):
        percent = [percent]
        end_percent = [end_percent]
        extract_flag = True
    elif isinstance(percent, (int, float)):
        percent = [percent] * len(end_percent)
    elif isinstance(end_percent, (int, float)):
        end_percent = [end_percent] * len(percent)
    assert len(percent) == len(end_percent)

    length = len(data)
    result = []
    for b, e in zip(percent, end_percent):
        assert b >= 0.0 and b <= 1.0
        assert e >= 0.0 and e <= 1.0
        assert e >= b
        i = int(length * b)
        j = int(length * e)
        j = j + 1 if j == i else j
        j = min(length - 1, j)
        i = i - 1 if i == length else i
        n = j - i
        r = data[j] - data[i]
        r = r / max(1, n) if normalize else r
        result.append(r)
    if len(result) == 1 and extract_flag:
        result = result[0]
    return result


def get_average_diff_state_from_percent(data, percent=0.0, end_percent=1.0, n_round=2):
    """Get the average difference stat from a data sequence starting at a given percent.
    Parameters
    ----------
    data:  sequence
        the provided data.
    percent: float
        the provided percentage. Default: 0.0.
    end_percent: float
        the provided end percentage. Default: 1.0.
    n_round: int
        number of points for rounding operation. Default: 2.
    Returns
    -------
    result: float
        the computed average stat.
    """
    data = np.around(np.array(data), n_round)
    tmp = (data[1:] - data[0:-1]) >= 0
    tmp = [1] + tmp.tolist()
    return get_average_state_from_percent(tmp, percent, end_percent)


def get_average_diff_state_from_percent_ndim(
    data, percent=0.0, end_percent=1.0, n_round=2, sign_flag=None
):
    """Get the average difference stat from a data sequence starting at a given percent.
    This is done on multi dimension data.
    The average is based on all the dimension of the data.
    The sign_flag is used to imforms if a dimension should be considered
    ascendant (true) or descendant (False).

    Parameters
    ----------
    data:  sequence
        the provided data.
    percent: float
        the provided percentage. Default: 0.0.
    end_percent: float
        the provided end percentage. Default: 1.0.
    n_round: int
        number of points for rounding operation. Default: 2.
    sign_flag: list of boolean
        flags informing the direction (ascendant or descendant)
        for each dimension. Default: None.
    Returns
    -------
    result: float
        the computed average stat.
    """
    data = np.around(np.array(data), n_round)
    if sign_flag is None:
        sign_flag = [True] * data.shape[1]
    assert len(sign_flag) >= data.shape[1]
    tmp = data[1:] - data[0:-1]
    prod = np.array([1] * tmp.shape[0])
    for i in range(tmp.shape[1]):
        flag = sign_flag[i]
        prod *= (tmp[:, i] >= 0) if flag else (tmp[:, i] <= 0)
    tmp = [1] + prod.tolist()
    return get_average_state_from_percent(tmp, percent, end_percent)


def get_slice_data(data, index, col, aggregator=None):
    """Utility function to aggregate sliced data.

    Parameters
    ----------
    data :  sequence
        data on which to perform the operation.
    index: int
        the index element to consider.
    col: int
        the column range to be considered (0:`col`)
    aggregator: callable
        aggregator function. Default: None.
    Returns
    -------
    result: float
        computed aggregated value
    """
    if data is None:
        return 1
    if index is not None:
        data = data[index]
    if col is not None:
        data = data[0:col]
    return data if aggregator is None else aggregator(data)


def compute_f1(p, r):
    """Utility function to compute f1.

    Parameters
    ----------
    p : float
        precision.
    r: float
        recall.
    Returns
    -------
    result: float
        computed f1
    """
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    if isinstance(r, (list, tuple)):
        r = np.array(r)
    denom = p + r
    return (2 * p * r) / (denom + 1e-10)


def load_json(data_filepath):
    """Utility function to load condition/symptom JSON files.

    Parameters
    ----------
    data_filepath :  str
        path to a json file containing the authorized
        symptom/condition data.
    Returns
    -------
    index_2_key: list
        a list containing all the keys of the authorized data.
    data: dict
        the authorized data.
    """
    with open(data_filepath) as fp:
        data = json.load(fp)

    index_2_key = sorted(list(data.keys()))
    return index_2_key, data


def get_topk_hamming_distance(
    sorted_proba, sorted_indices, true_indices, k=None, normalize=True
):
    """Computes the topk Hamming distance between succesive turn predictions.

    Parameters
    ----------
    sorted_proba: list
        the sorted predicted probability values for each turn.
        This tensor represents the probability values, not logits.
    sorted_indices: int, np.array
        the sorted indices associated to `sorted_proba`.
    true_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    k: int
        The number of top element to be considered in the predicted distribution.
        if none, then take the number of pathologies in the target distribution.
    normalize: bool
        Flag indicating whether or not to normalize the result. Default: True.

    Return
    ----------
    result: list
        the computed metric for each turn.
    """
    if k is None:
        k = 1 if true_indices is None else np.sum(np.array(true_indices) != -1)

    denom = 1 if not normalize else k
    sorted_indices = np.array(sorted_indices)
    if sorted_indices.shape[0] > 1:
        values = (sorted_indices[1:, :k] != sorted_indices[:-1, :k]).sum(axis=1) / denom
        values = values.tolist()
    else:
        values = []
    result = [1 if normalize else k] + values
    return result


def get_topk_set_difference(
    sorted_proba, sorted_indices, true_indices, k=None, normalize=True
):
    """Computes the topk set difference between succesive turn predictions.

    Parameters
    ----------
    sorted_proba: list
        the sorted predicted probability values for each turn.
        This tensor represents the probability values, not logits.
    sorted_indices: int, np.array
        the sorted indices associated to `sorted_proba`.
    true_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    k: int
        The number of top element to be considered in the predicted distribution.
        if none, then take the number of pathologies in the target distribution.
    normalize: bool
        Flag indicating whether or not to normalize the result. Default: True.

    Return
    ----------
    result: list
        the computed metric for each turn.
    """
    if k is None:
        k = 1 if true_indices is None else np.sum(np.array(true_indices) != -1)

    denom = 1 if not normalize else k
    values = [
        len(set(sorted_indices[i + 1][:k]) - set(sorted_indices[i][:k])) / denom
        for i in range(len(sorted_indices) - 1)
    ]
    result = [1 if normalize else k] + values
    return result


def get_weighted_mean(data, weights=None):
    if weights is None:
        weights = np.ones(len(data))
    weights = np.array(weights)
    average = np.average(data, weights=weights)
    return average


def weighted_avg_and_std(values, weights=None, axis=None):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    if weights is None:
        weights = np.ones(len(values))
    weights = np.array(weights)
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights, axis=axis)
    v1 = weights.sum()
    v2 = (weights ** 2).sum()
    if v1 != 0:
        variance = (variance * v1) / (v1 - v2 / v1)
    return [average, np.sqrt(variance)]


def kl_trajectory_score(kl_explore, kl_confirm):
    """
    Compute the kl_div based trajectory score from predicted probas.
    """
    kl_explore = np.array(kl_explore)
    kl_confirm = np.array(kl_confirm)
    if len(kl_explore) > 1:
        alphas = np.arange(start=len(kl_explore) - 1, stop=-1, step=-1)
        alphas = alphas / (len(kl_explore) - 1)
    else:
        alphas = 0.5
    score = alphas * kl_explore + (1 - alphas) * kl_confirm
    return score.tolist()


def compute_traj_metrics(
    data_stats,
    weights,
    metrics,
    symp_file,
    cond_file,
    pool_size,
    min_diff_proba=0.01,
    severity_threshold=3,
):
    """Utility function to process an evaluation stat file and generate plots.

    Parameters
    ----------
    data_stats :  str
        evaluation stats generated from evaluating a model.
    weights :  list
        weights of the patient involved in each trajectories.
    metrics : dict
        a dictionary to which the computed metrics are to be added.
    symp_file :  str
        path to a json file containing the authorized symptom data.
    cond_file :  str
        path to a json file containing the authorized condition data.
    pool_size: int
        the pool size to be used.
    min_diff_proba: float
        the threshold under which a pathology is considered to be part of the
        differential. Default: 0.01
    severity_threshold: int
        the threshold under which a pathology is considered severe. Default: 3
    Returns
    -------
    None
    """
    global global_dict
    global_dict = {}
    symp_index_2_key, symp_data = load_json(symp_file)
    cond_index_2_key, cond_data = load_json(cond_file)
    sevpatho_indices = [
        i
        for i in range(len(cond_index_2_key))
        if cond_data[cond_index_2_key[i]].get("severity", severity_threshold)
        < severity_threshold
    ]
    weights = np.array(weights) if weights is not None else None

    all_aux_rewards = data_stats["data"]["all_aux_rewards"]
    all_proba_dist = data_stats["data"]["all_proba_dist"]
    all_q_values = data_stats["data"]["all_q_values"]
    all_atcd_actions = data_stats["data"]["all_atcd_actions"]
    all_relevant_actions = data_stats["data"]["all_relevant_actions"]
    all_differential_indices = data_stats["data"]["y_differential_indices"]
    all_differential_probas = data_stats["data"]["y_differential_probas"]
    all_simulated_pathos = data_stats["data"]["y_true"]
    all_num_experienced_symptoms = data_stats["data"]["num_experienced_symptoms"]
    all_num_experienced_atcds = data_stats["data"]["num_experienced_atcd"]
    all_inquired_evidences = data_stats["data"]["inquired_evidences"]
    # all_first_symptoms = data_stats["data"]["first_symptoms"]
    # ####
    all_num_experienced_evidences = np.array(all_num_experienced_symptoms) + np.array(
        all_num_experienced_atcds
    )
    global_dict = dict(
        all_aux_rewards=all_aux_rewards,
        all_proba_dist=all_proba_dist,
        all_q_values=all_q_values,
        all_atcd_actions=all_atcd_actions,
        all_relevant_actions=all_relevant_actions,
        all_differential_indices=all_differential_indices,
        all_differential_probas=all_differential_probas,
        all_simulated_pathos=all_simulated_pathos,
        all_num_experienced_symptoms=all_num_experienced_symptoms,
        all_num_experienced_atcds=all_num_experienced_atcds,
        all_inquired_evidences=all_inquired_evidences,
        all_num_experienced_evidences=all_num_experienced_evidences,
    )

    # ####################
    def map_func(func, iterable):
        if pool_size == 0:
            result = map(func, iterable)
        else:
            with Pool(pool_size) as p:
                result = p.map(func, iterable)
        return result

    # #####################
    all_js_div_delta = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[apply_consecutive_difference_func, "all_proba_dist"],
                kwargs=dict(func=dist_js_div),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_js_div_delta"] = all_js_div_delta
    all_normalized_proba_dist = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[numpy_softmax, "all_proba_dist"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    # update
    global_dict["all_normalized_proba_dist"] = all_normalized_proba_dist
    kl_confirm = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    kl_confirm_score,
                    "all_normalized_proba_dist",
                    "all_simulated_pathos",
                    "all_differential_indices",
                    "all_differential_probas",
                ],
                kwargs=dict(c=1),
            ),
            range(len(all_proba_dist)),
        )
    )
    # update
    global_dict["kl_confirm"] = kl_confirm
    kl_explore = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[kl_explore_score, "all_normalized_proba_dist"],
                kwargs=dict(c=1),
            ),
            range(len(all_proba_dist)),
        )
    )
    # update
    global_dict["kl_explore"] = kl_explore
    auc_trajectory = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[kl_trajectory_auc, "kl_explore", "kl_confirm"],
                kwargs=dict(mode="none"),
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["AUCTraj"] = get_weighted_mean(auc_trajectory, weights)
    auc_trajectory = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[kl_trajectory_auc, "kl_explore", "kl_confirm"],
                kwargs=dict(mode="sort"),
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["AUCTraj_Sort"] = get_weighted_mean(auc_trajectory, weights)
    kl_trajectory = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[kl_trajectory_score, "kl_explore", "kl_confirm"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    # update
    global_dict["kl_trajectory"] = kl_trajectory
    trajectory_score = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.mean, "kl_trajectory"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["TrajScore"] = get_weighted_mean(trajectory_score, weights)
    all_sorted_desc_normalized_proba_dist = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[apply_sorting_func, "all_normalized_proba_dist"],
                kwargs=dict(axis=-1, arg=False, reverse=True),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict[
        "all_sorted_desc_normalized_proba_dist"
    ] = all_sorted_desc_normalized_proba_dist
    all_sorted_desc_normalized_proba_indices = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[apply_sorting_func, "all_normalized_proba_dist"],
                kwargs=dict(axis=-1, arg=True, reverse=True),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict[
        "all_sorted_desc_normalized_proba_indices"
    ] = all_sorted_desc_normalized_proba_indices

    # ########################### Severe Patho RuleIn RuleOut
    All_IN_Out_Percents = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    numpy_get_severe_pathos_inout_ratio,
                    "all_normalized_proba_dist",
                    "all_simulated_pathos",
                    "all_differential_indices",
                    "all_differential_probas",
                    sevpatho_indices,
                ],
                kwargs=dict(
                    diff_proba_threshold=min_diff_proba, front_broadcast_flag=True
                ),
            ),
            range(len(all_proba_dist)),
        )
    )
    all_out_percents, all_in_percents = zip(*All_IN_Out_Percents)
    global_dict["all_out_percents"] = list(all_out_percents)
    global_dict["all_in_percents"] = list(all_in_percents)
    all_sevf1_percents = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[compute_f1, "all_out_percents", "all_in_percents"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_sevf1_percents"] = list(all_sevf1_percents)
    last_rule_out = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.take, "all_out_percents", -1],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    last_rule_in = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.take, "all_in_percents", -1],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    last_f1_sev = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.take, "all_sevf1_percents", -1],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["DSP"] = get_weighted_mean(last_rule_out, weights)  # Sev_OUT
    metrics["DSR"] = get_weighted_mean(last_rule_in, weights)  # Sev_In
    metrics["DSF1"] = get_weighted_mean(last_f1_sev, weights)  # Sev_F1

    # ########################### All Patho RuleIn RuleOut
    All_patho_IN_Out_Percents = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    numpy_get_pathos_inout_ratio,
                    "all_normalized_proba_dist",
                    "all_simulated_pathos",
                    "all_differential_indices",
                    "all_differential_probas",
                ],
                kwargs=dict(
                    diff_proba_threshold=min_diff_proba, front_broadcast_flag=True
                ),
            ),
            range(len(all_proba_dist)),
        )
    )
    all_patho_out_percents, all_patho_in_percents = zip(*All_patho_IN_Out_Percents)
    global_dict["all_patho_out_percents"] = list(all_patho_out_percents)
    global_dict["all_patho_in_percents"] = list(all_patho_in_percents)
    all_patho_f1_percents = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[compute_f1, "all_patho_out_percents", "all_patho_in_percents"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_patho_f1_percents"] = list(all_patho_f1_percents)
    last_patho_rule_out = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.take, "all_patho_out_percents", -1],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    last_patho_rule_in = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.take, "all_patho_in_percents", -1],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    last_patho_f1 = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.take, "all_patho_f1_percents", -1],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["DPP"] = get_weighted_mean(last_patho_rule_out, weights)  # PathoF1
    metrics["DPR"] = get_weighted_mean(last_patho_rule_in, weights)  # PathoRecall
    metrics["DPF1"] = get_weighted_mean(last_patho_f1, weights)  # PathoF1

    # ###########################
    NQR_JS_evidences = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_js_div_delta",
                    None,
                    [True],
                    "all_relevant_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    NQR_JS_symptoms = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_js_div_delta",
                    None,
                    [True, True],
                    "all_relevant_actions",
                    "all_atcd_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    NQR_JS_atcds = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_js_div_delta",
                    None,
                    [True, False],
                    "all_relevant_actions",
                    "all_atcd_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["NEI"] = get_weighted_mean(NQR_JS_evidences, weights)
    metrics["NSI"] = get_weighted_mean(NQR_JS_symptoms, weights)
    metrics["NAI"] = get_weighted_mean(NQR_JS_atcds, weights)
    # ######################
    interaction_len = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[len, "all_proba_dist"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["IL"] = get_weighted_mean(interaction_len, weights)
    avg_recall_evidences = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_relevant_actions",
                    "all_num_experienced_evidences",
                    [],
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    avg_precision_evidences = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[np.mean, "all_relevant_actions"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["avg_precision_evidences"] = avg_precision_evidences
    global_dict["avg_recall_evidences"] = avg_recall_evidences

    avg_f1_evidences = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[compute_f1, "avg_precision_evidences", "avg_recall_evidences"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["PER"] = get_weighted_mean(avg_recall_evidences, weights)
    metrics["PEP"] = get_weighted_mean(avg_precision_evidences, weights)
    metrics["PEF1"] = get_weighted_mean(avg_f1_evidences, weights)
    avg_recall_symptoms = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_relevant_actions",
                    "all_num_experienced_symptoms",
                    [True],
                    "all_atcd_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    avg_precision_symptoms = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_relevant_actions",
                    None,
                    [True],
                    "all_atcd_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )

    global_dict["avg_recall_symptoms"] = avg_recall_symptoms
    global_dict["avg_precision_symptoms"] = avg_precision_symptoms
    avg_f1_symptoms = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[compute_f1, "avg_precision_symptoms", "avg_recall_symptoms"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["PSR"] = get_weighted_mean(avg_recall_symptoms, weights)
    metrics["PSP"] = get_weighted_mean(avg_precision_symptoms, weights)
    metrics["PSF1"] = get_weighted_mean(avg_f1_symptoms, weights)
    avg_recall_atcds = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_relevant_actions",
                    "all_num_experienced_atcds",
                    [False],
                    "all_atcd_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    avg_precision_atcds = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    compute_ratio,
                    "all_relevant_actions",
                    None,
                    [False],
                    "all_atcd_actions",
                ],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )

    global_dict["avg_recall_atcds"] = avg_recall_atcds
    global_dict["avg_precision_atcds"] = avg_precision_atcds
    avg_f1_atcds = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[compute_f1, "avg_precision_atcds", "avg_recall_atcds"],
                kwargs={},
            ),
            range(len(all_proba_dist)),
        )
    )
    metrics["PAR"] = get_weighted_mean(avg_recall_atcds, weights)
    metrics["PAP"] = get_weighted_mean(avg_precision_atcds, weights)
    metrics["PAF1"] = get_weighted_mean(avg_f1_atcds, weights)

    # #############################
    p = list(range(0, 105, 5))
    p_idx = {v: i for i, v in enumerate(p)}
    p = [v / 100.0 for v in p]

    # ###############################
    JSD_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_js_div_delta"],
                    kwargs=dict(percent=p),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    JSD_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_js_div_delta"],
                    kwargs=dict(end_percent=p),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the average JSD between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    metrics["JSD_{0.00}"] = get_weighted_mean(JSD_Data[:, p_idx[0]], weights)
    metrics["JSD_{0.25}"] = get_weighted_mean(JSD_Data[:, p_idx[25]], weights)
    metrics["JSD_{0.50}"] = get_weighted_mean(JSD_Data[:, p_idx[50]], weights)
    metrics["JSD_{0.75}"] = get_weighted_mean(JSD_Data[:, p_idx[75]], weights)
    # compute the average JSD between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BJSD_{0.25}"] = get_weighted_mean(JSD_B_Data[:, p_idx[25]], weights)
    metrics["BJSD_{0.50}"] = get_weighted_mean(JSD_B_Data[:, p_idx[50]], weights)
    metrics["BJSD_{0.75}"] = get_weighted_mean(JSD_B_Data[:, p_idx[75]], weights)

    # ##########################################
    all_Top5_Hamming_dist = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    get_topk_hamming_distance,
                    "all_sorted_desc_normalized_proba_dist",
                    "all_sorted_desc_normalized_proba_indices",
                    "all_differential_indices",
                ],
                kwargs=dict(k=5, normalize=True),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_Top5_Hamming_dist"] = all_Top5_Hamming_dist

    Ham5_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_Top5_Hamming_dist"],
                    kwargs=dict(percent=p),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    Ham5_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_Top5_Hamming_dist"],
                    kwargs=dict(end_percent=p),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the average HAM between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    metrics["HAM_{0.00}@5"] = get_weighted_mean(Ham5_Data[:, p_idx[0]], weights)
    metrics["HAM_{0.25}@5"] = get_weighted_mean(Ham5_Data[:, p_idx[25]], weights)
    metrics["HAM_{0.50}@5"] = get_weighted_mean(Ham5_Data[:, p_idx[50]], weights)
    metrics["HAM_{0.75}@5"] = get_weighted_mean(Ham5_Data[:, p_idx[75]], weights)
    # compute the average HAM between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BHAM_{0.25}@5"] = get_weighted_mean(Ham5_B_Data[:, p_idx[25]], weights)
    metrics["BHAM_{0.50}@5"] = get_weighted_mean(Ham5_B_Data[:, p_idx[50]], weights)
    metrics["BHAM_{0.75}@5"] = get_weighted_mean(Ham5_B_Data[:, p_idx[75]], weights)

    # ##########################################
    all_Top5_SetDiff_dist = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    get_topk_set_difference,
                    "all_sorted_desc_normalized_proba_dist",
                    "all_sorted_desc_normalized_proba_indices",
                    "all_differential_indices",
                ],
                kwargs=dict(k=5, normalize=True),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_Top5_SetDiff_dist"] = all_Top5_SetDiff_dist

    Dif5_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_Top5_SetDiff_dist"],
                    kwargs=dict(percent=p),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    Dif5_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_Top5_SetDiff_dist"],
                    kwargs=dict(end_percent=p),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the average DIF between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    metrics["DIF_{0.00}@5"] = get_weighted_mean(Dif5_Data[:, p_idx[0]], weights)
    metrics["DIF_{0.25}@5"] = get_weighted_mean(Dif5_Data[:, p_idx[25]], weights)
    metrics["DIF_{0.50}@5"] = get_weighted_mean(Dif5_Data[:, p_idx[50]], weights)
    metrics["DIF_{0.75}@5"] = get_weighted_mean(Dif5_Data[:, p_idx[75]], weights)
    # compute the average DIF between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BDIF_{0.25}@5"] = get_weighted_mean(Dif5_B_Data[:, p_idx[25]], weights)
    metrics["BDIF_{0.50}@5"] = get_weighted_mean(Dif5_B_Data[:, p_idx[50]], weights)
    metrics["BDIF_{0.75}@5"] = get_weighted_mean(Dif5_B_Data[:, p_idx[75]], weights)

    # ####################################################
    all_Top5_ProbMass_dist = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    get_topk_probability_mass,
                    "all_sorted_desc_normalized_proba_dist",
                    "all_sorted_desc_normalized_proba_indices",
                    "all_differential_indices",
                ],
                kwargs=dict(k=5, normalize=True),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_Top5_ProbMass_dist"] = all_Top5_ProbMass_dist

    Con5_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[
                        get_average_diff_state_from_percent,
                        "all_Top5_ProbMass_dist",
                    ],
                    kwargs=dict(percent=p, n_round=2),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    Con5_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[
                        get_average_diff_state_from_percent,
                        "all_Top5_ProbMass_dist",
                    ],
                    kwargs=dict(end_percent=p, n_round=2),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the average CON between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    metrics["CON_{0.00}@5"] = get_weighted_mean(Con5_Data[:, p_idx[0]], weights)
    metrics["CON_{0.25}@5"] = get_weighted_mean(Con5_Data[:, p_idx[25]], weights)
    metrics["CON_{0.50}@5"] = get_weighted_mean(Con5_Data[:, p_idx[50]], weights)
    metrics["CON_{0.75}@5"] = get_weighted_mean(Con5_Data[:, p_idx[75]], weights)
    # compute the average CON between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BCON_{0.25}@5"] = get_weighted_mean(Con5_B_Data[:, p_idx[25]], weights)
    metrics["BCON_{0.50}@5"] = get_weighted_mean(Con5_B_Data[:, p_idx[50]], weights)
    metrics["BCON_{0.75}@5"] = get_weighted_mean(Con5_B_Data[:, p_idx[75]], weights)

    # ###########################################################
    all_Split_Top5_ProbMass_dist = list(
        map_func(
            functools.partial(
                indexed_function_call_keys,
                sequential=False,
                params=[
                    get_split_topk_probability_mass,
                    "all_sorted_desc_normalized_proba_dist",
                    "all_sorted_desc_normalized_proba_indices",
                    "all_differential_indices",
                ],
                kwargs=dict(k=5, normalize=True),
            ),
            range(len(all_proba_dist)),
        )
    )
    global_dict["all_Split_Top5_ProbMass_dist"] = all_Split_Top5_ProbMass_dist

    ConS5_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[
                        get_average_diff_state_from_percent_ndim,
                        "all_Split_Top5_ProbMass_dist",
                    ],
                    kwargs=dict(percent=p, n_round=2, sign_flag=[True, False]),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    ConS5_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[
                        get_average_diff_state_from_percent_ndim,
                        "all_Split_Top5_ProbMass_dist",
                    ],
                    kwargs=dict(end_percent=p, n_round=2, sign_flag=[True, False]),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the average CON2 between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    metrics["CON2_{0.00}@5"] = get_weighted_mean(ConS5_Data[:, p_idx[0]], weights)
    metrics["CON2_{0.25}@5"] = get_weighted_mean(ConS5_Data[:, p_idx[25]], weights)
    metrics["CON2_{0.50}@5"] = get_weighted_mean(ConS5_Data[:, p_idx[50]], weights)
    metrics["CON2_{0.75}@5"] = get_weighted_mean(ConS5_Data[:, p_idx[75]], weights)
    # compute the average CON2 between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BCON2_{0.25}@5"] = get_weighted_mean(ConS5_B_Data[:, p_idx[25]], weights)
    metrics["BCON2_{0.50}@5"] = get_weighted_mean(ConS5_B_Data[:, p_idx[50]], weights)
    metrics["BCON2_{0.75}@5"] = get_weighted_mean(ConS5_B_Data[:, p_idx[75]], weights)

    # ###########################################################

    Con3_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_diff_state_from_percent, "kl_confirm"],
                    kwargs=dict(percent=p, n_round=2),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    Con3_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_diff_state_from_percent, "kl_confirm"],
                    kwargs=dict(end_percent=p, n_round=2),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the average CON2 between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    metrics["CON3_{0.00}@5"] = get_weighted_mean(Con3_Data[:, p_idx[0]], weights)
    metrics["CON3_{0.25}@5"] = get_weighted_mean(Con3_Data[:, p_idx[25]], weights)
    metrics["CON3_{0.50}@5"] = get_weighted_mean(Con3_Data[:, p_idx[50]], weights)
    metrics["CON3_{0.75}@5"] = get_weighted_mean(Con3_Data[:, p_idx[75]], weights)
    # compute the average CON2 between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BCON3_{0.25}@5"] = get_weighted_mean(Con3_B_Data[:, p_idx[25]], weights)
    metrics["BCON3_{0.50}@5"] = get_weighted_mean(Con3_B_Data[:, p_idx[50]], weights)
    metrics["BCON3_{0.75}@5"] = get_weighted_mean(Con3_B_Data[:, p_idx[75]], weights)

    # ##########################################

    Expl_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_diff_state_from_percent, "kl_explore"],
                    kwargs=dict(percent=p, normalize=False),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    Expl_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_diff_state_from_percent, "kl_explore"],
                    kwargs=dict(end_percent=p, normalize=False),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the Info gain (diff of KL) between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    # in [-1, 1]
    metrics["EXPL_{0.00}@5"] = get_weighted_mean(Expl_Data[:, p_idx[0]], weights)
    metrics["EXPL_{0.25}@5"] = get_weighted_mean(Expl_Data[:, p_idx[25]], weights)
    metrics["EXPL_{0.50}@5"] = get_weighted_mean(Expl_Data[:, p_idx[50]], weights)
    metrics["EXPL_{0.75}@5"] = get_weighted_mean(Expl_Data[:, p_idx[75]], weights)
    # compute the average DIF between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BEXPL_{0.25}@5"] = get_weighted_mean(Expl_B_Data[:, p_idx[25]], weights)
    metrics["BEXPL_{0.50}@5"] = get_weighted_mean(Expl_B_Data[:, p_idx[50]], weights)
    metrics["BEXPL_{0.75}@5"] = get_weighted_mean(Expl_B_Data[:, p_idx[75]], weights)

    # ##########################################

    CONF_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_diff_state_from_percent, "kl_confirm"],
                    kwargs=dict(percent=p, normalize=False),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    CONF_B_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_diff_state_from_percent, "kl_confirm"],
                    kwargs=dict(end_percent=p, normalize=False),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # compute the Info gain (diff of KL) between successive agent predictions
    # start from p = [0, 0.25, 0.5, 0.75] up to the end of the interaction session
    # in [-1, 1]
    metrics["CGain_{0.00}@5"] = get_weighted_mean(CONF_Data[:, p_idx[0]], weights)
    metrics["CGain_{0.25}@5"] = get_weighted_mean(CONF_Data[:, p_idx[25]], weights)
    metrics["CGain_{0.50}@5"] = get_weighted_mean(CONF_Data[:, p_idx[50]], weights)
    metrics["CGain_{0.75}@5"] = get_weighted_mean(CONF_Data[:, p_idx[75]], weights)
    # compute the average DIF between successive agent predictions
    # start from the begin of the interaction up to p = [0.25, 0.5, 0.75]
    metrics["BCGain_{0.25}@5"] = get_weighted_mean(CONF_B_Data[:, p_idx[25]], weights)
    metrics["BCGain_{0.50}@5"] = get_weighted_mean(CONF_B_Data[:, p_idx[50]], weights)
    metrics["BCGain_{0.75}@5"] = get_weighted_mean(CONF_B_Data[:, p_idx[75]], weights)

    # ###########################################################
    # get the SevF1 plot data
    SevF1_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_sevf1_percents"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # get the SevPrecision (Out) plot data
    SevPrecOut_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_out_percents"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # get the SevRecall (In) plot data
    SevRecIn_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_in_percents"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )

    SevF1_plot_info = weighted_avg_and_std(SevF1_Data, weights, axis=0)
    SevPrecOut_plot_info = weighted_avg_and_std(SevPrecOut_Data, weights, axis=0)
    SevRecIn_plot_info = weighted_avg_and_std(SevRecIn_Data, weights, axis=0)

    # get the PathoF1 plot data
    PathoF1_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_patho_f1_percents"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # get the SevPrecision (Out) plot data
    PathoPrecOut_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_patho_out_percents"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # get the SevRecall (In) plot data
    PathoRecIn_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "all_patho_in_percents"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )

    PathoF1_plot_info = weighted_avg_and_std(PathoF1_Data, weights, axis=0)
    PathoPrecOut_plot_info = weighted_avg_and_std(PathoPrecOut_Data, weights, axis=0)
    PathoRecIn_plot_info = weighted_avg_and_std(PathoRecIn_Data, weights, axis=0)

    # get the kl_explore plot data
    KLExplore_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "kl_explore"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # get the kl_confirm plot data
    KLConfirm_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "kl_confirm"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )
    # get the kl_trajectory plot data
    KL_Trajectory_Data = np.array(
        list(
            map_func(
                functools.partial(
                    indexed_function_call_keys,
                    sequential=False,
                    params=[get_average_state_from_percent, "kl_trajectory"],
                    kwargs=dict(percent=p, end_percent=None),
                ),
                range(len(all_proba_dist)),
            )
        )
    )

    KLExplore_plot_info = weighted_avg_and_std(KLExplore_Data, weights, axis=0)
    KLConfirm_plot_info = weighted_avg_and_std(KLConfirm_Data, weights, axis=0)
    KL_Trajectory_plot_info = weighted_avg_and_std(KL_Trajectory_Data, weights, axis=0)
    metrics["PlotData"] = {
        "x": p,
        "SevF1": {
            "mean": np.around(SevF1_plot_info[0], 4).tolist(),
            "std": np.around(SevF1_plot_info[1], 4).tolist(),
        },
        "SevPrecOut": {
            "mean": np.around(SevPrecOut_plot_info[0], 4).tolist(),
            "std": np.around(SevPrecOut_plot_info[1], 4).tolist(),
        },
        "SevRecIn": {
            "mean": np.around(SevRecIn_plot_info[0], 4).tolist(),
            "std": np.around(SevRecIn_plot_info[1], 4).tolist(),
        },
        "PathoF1": {
            "mean": np.around(PathoF1_plot_info[0], 4).tolist(),
            "std": np.around(PathoF1_plot_info[1], 4).tolist(),
        },
        "PathoPrecOut": {
            "mean": np.around(PathoPrecOut_plot_info[0], 4).tolist(),
            "std": np.around(PathoPrecOut_plot_info[1], 4).tolist(),
        },
        "PathoRecIn": {
            "mean": np.around(PathoRecIn_plot_info[0], 4).tolist(),
            "std": np.around(PathoRecIn_plot_info[1], 4).tolist(),
        },
        "Exploration": {
            "mean": np.around(KLExplore_plot_info[0], 4).tolist(),
            "std": np.around(KLExplore_plot_info[1], 4).tolist(),
        },
        "Confirmation": {
            "mean": np.around(KLConfirm_plot_info[0], 4).tolist(),
            "std": np.around(KLConfirm_plot_info[1], 4).tolist(),
        },
        "Trajectory": {
            "mean": np.around(KL_Trajectory_plot_info[0], 4).tolist(),
            "std": np.around(KL_Trajectory_plot_info[1], 4).tolist(),
        },
    }
