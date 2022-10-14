#!/usr/bin/env python

import argparse
import os
import time

import mlflow
import torch
import yaml
from gym.envs.registration import register
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE
from rlpyt.utils.logging import logger
from yaml import load

from chloe.trainer.trainer import build_and_train
from chloe.utils.dev_utils import (
    check_git_consistency,
    get_current_git_commit_hash,
    get_dev_dependencies,
    get_repo_state,
    log_git_diff,
)
from chloe.utils.logging_utils import (
    check_and_log_hp,
    # log_config,
    log_dict_to_artefact,
    mlflow_record_tabular_func,
    suffix_output_dir_with_datetime,
)
from chloe.utils.sim_utils import store_file_in_shared_memory
from chloe.utils.train_utils import (
    save_run_ids,
    update_config_and_setup_exp_folder,
)

# patch: modifiy the logger function with the new defined one
logger.record_tabular = mlflow_record_tabular_func


def convert_str_to_lst(input, type):
    """Converts a comma separated string input to a list.

    Parameters
    ----------
    input : str
        comma separated string.
    type : type
        datatype in which the comma separated elements need to be converted.

    Returns
    -------
    output: List[type]
        parsed list of the input.

    """
    output = list(map(type, input.split(",")))
    return output


def _parse_orion(cfg):
    """Parse the config for not supported Orion flags and inject the appropriate value.

    Note: This function should be remove when Orion add support for them.

    Parameters
    ----------
    cfg: dict
        dictionary containing the config to train the agent.

    Returns
    -------
    hyper_params: the updated config.
    """
    for name in cfg:
        if type(cfg[name]) is str and cfg[name].startswith("orion#"):
            if cfg[name].endswith("exp.name"):
                cfg[name] = os.environ["ORION_EXPERIMENT_NAME"]
            elif cfg[name].endswith("trial.id"):
                cfg[name] = os.environ["ORION_TRIAL_ID"]
    return cfg


def _rm_cfg_key(cfg, name):
    """
    Remove key from config and return the updated config.
    """
    if name in cfg:
        del cfg[name]
    return cfg


def parse_hyper_params(hyper_params, args):
    """Converts dictionary formats from orion model config file to standard config.

    This function converts the model architecture and embedding dict formats
    from orion model config file to standard config. The model architectures
    in the standard config file is a list but orion cannot take a 2D list
    in its hyperparameter space. So, a comma separated string is used. Orion
    cannot take in keys as int in the config file but embedding_dict has int
    keys in the config file.

    Parameters
    ----------
    hyper_params: dict
        dictionary containing the config to train the agent.
    args: argument namespace
        namespace containing the program arguments.

    Returns
    -------
    hyper_params: the updated hyperparams dict

    """
    if isinstance(hyper_params["architecture_params"].get("hidden_sizes"), str):
        hyper_params["architecture_params"]["hidden_sizes"] = convert_str_to_lst(
            hyper_params["architecture_params"]["hidden_sizes"], int
        )

    if isinstance(hyper_params["architecture_params"].get("dueling_fc_sizes"), str):
        hyper_params["architecture_params"]["dueling_fc_sizes"] = convert_str_to_lst(
            hyper_params["architecture_params"]["dueling_fc_sizes"], int
        )
    if "embedding_dict" in hyper_params["architecture_params"]:
        hyper_params["architecture_params"]["embedding_dict"] = {
            int(key): val
            for key, val in hyper_params["architecture_params"][
                "embedding_dict"
            ].items()
        }
    hyper_params = _parse_orion(hyper_params)
    # Override resuming by removing confs needed for resuming.
    if "start_from_scratch" in args and args.start_from_scratch:
        hyper_params = _rm_cfg_key(hyper_params, "mlflow_uid")
        hyper_params = _rm_cfg_key(hyper_params, "config_uid")
        hyper_params = _rm_cfg_key(hyper_params, "experiment_folder")

    # Set exp name if none given
    if "exp_name" not in hyper_params:
        tmp_name = f"chloe@{hyper_params['agent']}@{hyper_params['algo']}"
        hyper_params["exp_name"] = tmp_name
        print(f"Warning: exp_name not in config file. Setting name to: {tmp_name}")

    # Check previous config for resuming
    if "config_uid" in hyper_params and "output" in args:
        tmp_name = hyper_params["exp_name"]
        tmp_uid = hyper_params["config_uid"]
        run_id = f"run_{args.run_ID}"
        old_conf_path = os.path.join(args.output, tmp_name, tmp_uid, run_id, "cfg.yml")
        if os.path.exists(old_conf_path):
            print(f"\n## Run with config_uid '{tmp_uid}' already in xp '{tmp_name}'")
            if "ignore_existing_conf" in args and args.ignore_existing_conf:
                print(
                    f"## As --ignore_existing_conf as been used, {old_conf_path} will"
                    " be ignored and the given one will be used.\n"
                )
                temp_path = os.path.splitext(old_conf_path)
                tmp_val = time.strftime("%Y_%m_%d_%H%M%S")
                old_conf_path_renamed = f"{temp_path[0]}_before_{tmp_val}{temp_path[1]}"
                os.rename(old_conf_path, old_conf_path_renamed)
            else:
                print(f"## Ignoring given config file and using: {old_conf_path}\n")
                with open(old_conf_path, "r") as f:
                    hyper_params = yaml.load(f, Loader=yaml.Loader)
    return hyper_params


def put_data_in_shared_memory_if_needed(args):
    """Put the data in share memory.

    Parameters
    ----------
    args: argument namespace
        namespace containing the program arguments.

    Returns
    -------
    None
    """
    if args.shared_data_socket is not None:
        tmp_data = args.data
        tmp_val_data = args.eval_data
        data_prefix = "training"
        store_file_in_shared_memory(
            args.shared_data_socket,
            args.data,
            prefix=data_prefix,
            replace_if_present=not args.no_replace_if_present,
        )
        args.data = data_prefix
        if args.eval_data == tmp_data:
            args.eval_data = data_prefix
        else:
            eval_data_prefix = "validate"
            store_file_in_shared_memory(
                args.shared_data_socket,
                args.eval_data,
                prefix=eval_data_prefix,
                replace_if_present=not args.no_replace_if_present,
            )
            args.eval_data = eval_data_prefix
        is_end_data = (
            "end_training_eval_data" in args and args.end_training_eval_data is not None
        )
        if is_end_data and args.end_training_eval_data == tmp_data:
            args.end_training_eval_data = data_prefix
        elif is_end_data and args.end_training_eval_data == tmp_val_data:
            args.end_training_eval_data = args.eval_data
        elif is_end_data:
            end_eval_data_prefix = "endvalid"
            store_file_in_shared_memory(
                args.shared_data_socket,
                args.end_training_eval_data,
                prefix=end_eval_data_prefix,
                replace_if_present=not args.no_replace_if_present,
            )
            args.end_training_eval_data = end_eval_data_prefix


def main():
    """This is a main utility function for training an RL agent.

    It will used the provided data to instantiate simulators
    the agent will interact with in order to learn/evaluate its policy.

    The parameters describing the model architecture, the agent type, the algo to
    be used as well as the metrics to be computed are all provided through the
    specified config file. A special metric (to be specified in the config file) is
    referred to as the performance metric (key 'perf_metric' in the config file) and
    it is used to decide when to eventually early-stop the training process.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config file with generic hyper-parameters,  such as optimizer, "
        "batch_size, ... -  in yaml format",
        type=str,
    )
    parser.add_argument(
        "--data", help="path to the patient data file", type=str, required=True
    )
    parser.add_argument(
        "--eval_data",
        help="path to the patient data file for evaluation purposes."
        " if not set, use the data paremeter as default value.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--end_training_eval_data",
        help="path to the patient data file for evaluation at the end of training.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--shared_data_socket",
        help="pyarrow plasma socket to be used for data sharing.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pretrained_model",
        help="filename of the pretrained model to be used.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        help="path to outputs - will store files here",
        type=str,
        required=True,
    )
    parser.add_argument("--cuda_idx", help="gpu to use", type=int, default=0)
    parser.add_argument(
        "--n_workers",
        help="number of cpu workers. useful for parallel sampler and"
        " multi-cpu optimization.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n_gpus",
        help="number of gup to use. Useful for multi-gpu optimization.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_torch_threads",
        help="number of thread to be used by pytorch.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cpu_list",
        help="list of cpus to be used by this experiments",
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--start_from_scratch",
        action="store_true",
        help="will not load any existing saved model - even if present",
    )
    parser.add_argument(
        "--ignore_existing_conf",
        action="store_true",
        default=False,
        help="ignore loading existing config.",
    )
    parser.add_argument(
        "--no_replace_if_present",
        action="store_true",
        help="if specified, the data will not be stored in the shared data"
        " socket if they are already present",
    )
    parser.add_argument(
        "--datetime_suffix",
        help="add the following datetime suffix to the output dir: "
        "<output_dir>/<yyyymmdd>/<hhmmss>",
        action="store_true",
        # default=True,
    )
    parser.add_argument(
        "--run_ID", help="run identifier for logging purposes", type=int, default=0
    )
    args = parser.parse_args()
    if args.ignore_existing_conf and args.start_from_scratch:
        parser.error(
            "Cannot use --ignore_existing_conf and --start_from_scratch at same time."
        )
    # if eval data is none, set it to data
    if args.eval_data is None:
        args.eval_data = args.data

    args.end_training_eval_data_fp =  args.end_training_eval_data

    # if shared_data_socket is not None
    put_data_in_shared_memory_if_needed(args)

    if not (args.cuda_idx is None):
        if not torch.cuda.is_available():
            logger.log(
                f"No cuda found. Defaulting the cuda_idx param"
                f' from From "{args.cuda_idx}" to "None".'
                f' Defaulting the n_gpus param From "{args.n_gpus}" to 0.'
            )
            args.cuda_idx = None
            args.n_gpus = 0

    # add datetime suffix if required
    if args.datetime_suffix:
        args.output = suffix_output_dir_with_datetime(args.output)

    if args.config is not None:
        with open(args.config, "r") as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)
    else:
        hyper_params = {}
    hyper_params = parse_hyper_params(hyper_params, args)

    # to be done as soon as possible otherwise mlflow
    # will not log with the proper exp. name
    if "exp_name" in hyper_params:
        mlflow.set_experiment(experiment_name=hyper_params["exp_name"])

    start_run(hyper_params)

    run(args, hyper_params)

    mlflow.end_run()


def start_run(cfg):
    mlflow.start_run(run_id=cfg.get("mlflow_uid"))


def run(args, hyper_params):
    """Defines the setup needed for running the training process of the RL agents.

    This is an utility function that properly defines the setup
    needed for running the training process as well as launching the process.

    Parameters
    ----------
    args : dict
        the arguments as provided in the command line.
    hyper_params : dict
        the parameters as provided in the configuration file.

    Returns
    -------
    None

    """

    GYM_ENV_ID = "simPa-v0"

    # get the mflow run id
    mlflow_run_id = mlflow.active_run().info.run_id
    hyper_params = update_config_and_setup_exp_folder(args, hyper_params, mlflow_run_id)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if not os.path.exists(hyper_params["experiment_folder"]):
        os.makedirs(hyper_params["experiment_folder"])

    args.output = hyper_params["experiment_folder"]

    # try to register the env in gym if not yet done
    try:
        register(
            id=GYM_ENV_ID,
            entry_point="chloe.simulator.simulator:PatientInteractionSimulator",
        )
    except Exception:
        logger.log("The environment [simPa-v0] is already registered in gym")
        pass

    run_id = f"run_{args.run_ID}"
    exp_folder = os.path.join(args.output, run_id)
    # saved the running ids for this experiment
    save_run_ids(exp_folder, mlflow_run_id, args.run_ID)

    # set eventually the default performance metric
    hyper_params["perf_metric"] = hyper_params.get("perf_metric", "Reward")

    # Log to MLFlow if in a special mode.
    if "ignore_existing_conf" in args and args.ignore_existing_conf:
        mlflow.log_param("Mode", "Forced Resume")
    elif "start_from_scratch" in args and args.start_from_scratch:
        mlflow.log_param("Mode", "No Resume")

    # get the git commit hash
    git_hash = get_current_git_commit_hash()

    # Log git repo state to MLFlow
    repo_state = get_repo_state()

    # Get previous git state and make sure it's the same before resuming.
    # This enforce reproducibility
    active_run_params = mlflow.active_run().data.params
    if not ("ignore_existing_conf" in args and args.ignore_existing_conf):
        check_git_consistency(repo_state, active_run_params)

    ignore_flag = "ignore_existing_conf" in args and args.ignore_existing_conf
    if (ignore_flag) and active_run_params != {}:
        mlflow_dir = f"force_resumed/{time.strftime('%Y_%m_%d_%H%M%S')}"
        log_dict_to_artefact(repo_state, "repo_state.yml", mlflow_dir)
        log_dict_to_artefact(hyper_params, "cfg.yml", mlflow_dir)
    else:
        mlflow_dir = None
        mlflow.log_params(repo_state)
        # __TODO__ change the hparam that are used from the training algorithm
        # (and NOT the model - these will be specified in the model itself)
        check_and_log_hp(
            [
                "optimizer",
                "architecture",
                "perf_window_size",
                "perf_metric",
                "log_interval_steps",
                "n_steps",
                "exp_name",
                "runner",
                "sampler",
                "algo",
                "agent",
                "n_envs",
                "eval_n_envs",
                "eval_max_steps",
                "eval_max_trajectories",
                "max_decorrelation_steps",
            ],
            hyper_params,
        )
        mlflow.log_param("data", args.data)
        mlflow.log_param("eval_data", args.eval_data)

    if repo_state["Git - Uncommited Changes"]:
        print(
            "Warning : Some uncommited changes were detected. The only way to resume "
            "this experiment will be to use --ignore_existing_conf."
        )
        log_git_diff(exp_folder, mlflow_dir)

    # get the dev dependencies
    dependencies = get_dev_dependencies()

    details = "\ngit code hash: {}\n\ndependencies:\n{}".format(
        git_hash, "\n".join(dependencies)
    )
    mlflow.set_tag(key=MLFLOW_RUN_NOTE, value=details)

    mlflow.log_param("mlflow_uid", hyper_params.get("mlflow_uid"))
    mlflow.log_param("config_uid", hyper_params.get("config_uid"))
    mlflow.log_param("experiment_folder", hyper_params.get("experiment_folder"))

    build_and_train(
        gym_env_id=GYM_ENV_ID, args=args, run_ID=args.run_ID, params=hyper_params,
    )


if __name__ == "__main__":
    main()
