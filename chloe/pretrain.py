#!/usr/bin/env python

import argparse
import os

import mlflow
import torch
import yaml
from gym.envs.registration import register
from yaml import load

from chloe.pretraining.pretrainer import pretrain
from chloe.utils.logging_utils import suffix_output_dir_with_datetime
from chloe.utils.sim_utils import store_file_in_shared_memory


def main():
    """This is an utility function for pretraining MixedDQN like agents.

    Here, the objective is to pretrain the classifier branch of such an agent
    in a supervised way with the hope to significantly reduce the training time in
    the RL settings while boosting performances.

    This utility can also be used to train the classifier in a full observability
    setting where it is possible to obtain the upper bound classification
    performance of the agent assuming it has a perfect knowledge of the
    symptoms/antecedents experienced by the patients. This is done by enabling the
    `no_data_corrupt` flag.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config file with generic agent parameters for its instanciation",
        type=str,
        required=True,
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
        "--shared_data_socket",
        help="pyarrow plasma socket to be used for data sharing.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_workers", help="number of workers for dataloading.", type=int, default=0,
    )
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("--batch_size", help="batch size", type=int, default=256)
    parser.add_argument("--patience", help="patience", type=int, default=10)
    parser.add_argument(
        "--valid_percentage",
        help="the percentage of data to be used for validaton. Must be in [0, 1)."
        " Useful only if eval_data is not provided.",
        type=float,
        default=None,
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument(
        "--metric",
        help="performance metric. If None, the negative loss will be used as a proxy.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output", help="path to outputs - will results here", type=str, default="./",
    )
    parser.add_argument(
        "--topk", help="topk pathologies to be considered", type=int, default=5,
    )
    parser.add_argument("--cuda_idx", help="gpu to use", type=int, default=None)
    parser.add_argument("--seed", help="seed to be used", type=int, default=None)
    parser.add_argument(
        "--no_replace_if_present",
        action="store_true",
        help="if specified, the data will not be stored in the shared data"
        " socket if they are already present",
    )
    parser.add_argument(
        "--no_data_corrupt",
        action="store_true",
        help="if specified, the data will be retrieved with full observability from the"
        " simulator, i.e, they won't be masked.",
    )
    parser.add_argument(
        "--datetime_suffix",
        help="add the following datetime suffix to the output dir: "
        "<output_dir>/<yyyymmdd>/<hhmmss>",
        action="store_true",
    )
    args = parser.parse_args()

    # assert either validation data (eval_data) or non-zero valid_percentage
    assert (args.eval_data is not None) or (
        args.valid_percentage is not None and args.valid_percentage > 0
    )

    # if eval data is none, set it to data
    if args.eval_data is None:
        args.eval_data = args.data

    # if shared_data_socket is not None
    if args.shared_data_socket is not None:
        tmp_data = args.data
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

    if not (args.cuda_idx is None):
        if not torch.cuda.is_available():
            print(
                f"No cuda found. Defaulting the cuda_idx param"
                f' from From "{args.cuda_idx}" to "None".'
            )
            args.cuda_idx = None

    if args.config is not None:
        with open(args.config, "r") as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)
    else:
        hyper_params = {}

    # add datetime suffix if required
    if args.datetime_suffix:
        args.output = suffix_output_dir_with_datetime(args.output)

    # to be done as soon as possible otherwise mlflow
    # will not log with the proper exp. name
    if "exp_name" in hyper_params:
        mlflow.set_experiment(f'{hyper_params["exp_name"]}-Pretraining')

    args.log_params = True

    mlflow.start_run()
    run(args, hyper_params)
    mlflow.end_run()


def run(args, hyper_params):
    """Defines the setup needed for running the pretraining of MixedDQN like agents.

    This is an utility function that properly defines the setup
    needed for running the pre-training process as well as launching the process.

    Parameters
    ----------
    args : dict
        The arguments as provided in the command line.
    hyper_params : dict
        The parameters as provided in the configuration file.

    Returns
    -------
    None

    """

    GYM_ENV_ID = "simPa-v0"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # try to register the env in gym if not yet done
    try:
        register(
            id=GYM_ENV_ID,
            entry_point="chloe.simulator.simulator:PatientInteractionSimulator",
        )
    except Exception:
        print("The environment [simPa-v0] is already registered in gym")
        pass

    pretrain(
        gym_env_id=GYM_ENV_ID, args=args, params=hyper_params,
    )


if __name__ == "__main__":
    main()
