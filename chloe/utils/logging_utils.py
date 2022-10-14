import datetime
import os
import os.path as osp
import tempfile

import mlflow
import numpy as np
import yaml
from rlpyt.utils.logging import logger

# save the original logger record data function
old_logger_record_tabular_func = logger.record_tabular


def mlflow_record_tabular_func(key, val, *args, **kwargs):
    """Extends the rlpyt logger capabilities by allowing it to record in mlflow.

    Parameters
    ----------
    key: str
        the key string associated with the value to be recorded
    val: float
        the value to be recorded

    Returns
    -------
    None
    """

    old_logger_record_tabular_func(key, val, *args, **kwargs)
    key = logger._tabular_prefix_str + str(key)
    key = key.replace("(", "").replace(")", "")
    mlflow.log_metric(key, val, step=logger._iteration)


def mlflow_log_misc_stats(itr, key, values, sep="_", default=np.nan):
    """Method for logging statistics regarding a sequence of values in mlflow.

    Parameters
    ----------
    itr: int
        step or iteration number.
    key: str
        key to be used.
    values: list, array-like
        values from which the stats will be computed.
    sep: str
        separator (e.g. for regrouping stats together).
        default: '_'
    default: flaot
        value to be used when the `values` is empty.

    Returns
    -------
    None

    """

    prefix = key + sep
    isEmpty = len(values) <= 0
    mlflow.log_metric(
        prefix + "Average", default if isEmpty else np.average(values), step=itr
    )
    mlflow.log_metric(prefix + "Std", default if isEmpty else np.std(values), step=itr)
    mlflow.log_metric(
        prefix + "Median", default if isEmpty else np.median(values), step=itr
    )
    mlflow.log_metric(prefix + "Min", default if isEmpty else np.min(values), step=itr)
    mlflow.log_metric(prefix + "Max", default if isEmpty else np.max(values), step=itr)


def mlflow_log_stats_from_dict(itr, data_dict, sep="_", default=np.nan):
    """Method for logging statistics regarding sequence data contained in a dict.

    Parameters
    ----------
    itr: int
        step or iteration number.
    data_dict: dict
        dictionary contating the data whose statistics will be logged.
    sep: str
        separator (e.g. for regrouping stats together).
        default: '_'
    default: flaot
        value to be used when the `values` is empty.

    Returns
    -------
    None

    """
    if data_dict:
        for k, v in data_dict.items():
            mlflow_log_misc_stats(itr, k, v, sep=sep, default=default)


def suffix_output_dir_with_datetime(output_dir):
    """Suffix an output dir with datetime info of the form <yyyymmdd>/<hhmmss>.

    Parameters
    ----------
    output_dir: str
        the provided ouput dir.

    Returns
    -------
    result: str
        the resulting dir with the datetime suffix append to it.

    """
    yyyymmdd_hhmmss = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    yyyymmdd, hhmmss = yyyymmdd_hhmmss.split("-")
    log_dir = osp.join(output_dir, yyyymmdd, hhmmss)
    return log_dir


def check_and_log_hp(names, hps, allow_extra=True, prefix="", log_hp_flag=True):
    """Method for checking and logging hyper params.

    Parameters
    ----------
    names: list
        list of hyper params to be checked and logged.
    hps: dict
        dictionary containing hyper-params data.
    allow_extra: bool
        flag to indicate wheter the provided dict may contain
        extra params w.r.t the ones provided. default: True
    pefix: str
        prefix to be used when logging the hyper params.
        default: ''
    log_hp_flag: boolean
        whether or not to log the hyper parameter in mlflow.
        default: True

    Returns
    -------
    None
    """
    check_hp(names, hps, allow_extra=allow_extra)
    if log_hp_flag:
        log_hp(names, hps, prefix=prefix)


def check_hp(names, hps, allow_extra=True):
    """Method for checking the presence of hyper params defined in the provided list.

    Parameters
    ----------
    names: list
        list of hyper params to be checked.
    hps: dict
        dictionary containing hyper-params data.
    allow_extra: bool
        flag to indicate wheter the provided dict may contain
        extra params w.r.t the ones provided. default: True

    Returns
    -------
    None

    """
    missing = set()
    for name in names:
        if name not in hps:
            missing.add(name)
    extra = hps.keys() - names

    if len(missing) > 0:
        logger.log("please add the missing hyper-parameters: {}".format(missing))
    if len(extra) > 0 and not allow_extra:
        logger.log("please remove the extra hyper-parameters: {}".format(extra))
    if len(missing) > 0 or (len(extra) > 0 and not allow_extra):
        raise ValueError("fix according to the error message above")


def log_hp(names, hps, prefix=""):
    """Method for logging hyper params.

    Parameters
    ----------
    names: list
        list of hyper params to be logged.
    hps: dict
        dictionary containing hyper-params data.
    pefix: str
        prefix to be used when logging the hyper params.
        default: ''

    Returns
    -------
    None

    """
    for name in sorted(names):
        if type(hps[name]) is dict:
            log_hp(list(hps[name].keys()), hps[name], prefix=f"{prefix}{name}.")
        else:
            mlflow.log_param(prefix + str(name), hps[name])
            logger.log('\thp "{}" => "{}"'.format(prefix + str(name), hps[name]))


def log_metrics(metrics, prefix=""):
    """Method for logging metrics.

    Parameters
    ----------
    metrics: dict
        dictionary containing metrics data.
    pefix: str
        prefix to be used when logging the hyper params.
        default: ''

    Returns
    -------
    None

    """
    for name in sorted(metrics.keys()):
        if type(metrics[name]) is dict:
            log_metrics(metrics[name], prefix=f"{prefix}{name}.")
        elif type(metrics[name]) is list:
            metric_name = prefix + str(name)
            metric_name = metric_name.replace("{", "")
            metric_name = metric_name.replace("}", "")
            metric_name = metric_name.replace("@", "-")
            for i in range(len(metrics[name])):
                mlflow.log_metric(metric_name, metrics[name][i], i)
            logger.log('\tMetric "{}" => "{}"'.format(metric_name, metrics[name]))
        else:
            metric_name = prefix + str(name)
            metric_name = metric_name.replace("{", "")
            metric_name = metric_name.replace("}", "")
            metric_name = metric_name.replace("@", "-")
            mlflow.log_metric(metric_name, metrics[name])
            logger.log('\tMetric "{}" => "{}"'.format(metric_name, metrics[name]))


def log_dict_to_artefact(data, artifact_name, artifact_path):
    """Save dictionary as an artifact.
    Parameters
    ----------
    data : dict
        Dictionary to be saved as an artifact.
    artifact_name : str
        Name of the artifact file in MLFlow
    artifact_path : str
        Where to place the file in the virtual folder structure of MLFlow.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path_local = os.path.join(tmp_dir, artifact_name)
        with open(artifact_path_local, "w") as f:
            yaml.dump(data, f)
        mlflow.log_artifact(artifact_path_local, artifact_path=artifact_path)


def log_config(cfg, prefix=""):
    """Log recursively every config in cfg as an MLFlow param.
    Ex:
    optimizer:
      weight_decay:
        rate: 0.001
    Will be logged as `optimizer.weight_decay.rate : 0.001`.
    Parameters
    ----------
    cfg : dict
        A dictionary containing all configs.
    prefix : str
        A prefix to append in front of the name of the key before logging it.
    """
    for conf_name in cfg.keys():
        if type(cfg[conf_name]) is dict:
            log_config(cfg[conf_name], prefix=f"{prefix}{conf_name}.")
        else:
            mlflow.log_param(f"{prefix}{conf_name}", cfg[conf_name])
