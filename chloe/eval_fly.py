#!/usr/bin/env python

import argparse
import os

import mlflow
import torch
import yaml
from gym.envs.registration import register
from yaml import load

from chloe.trainer.trainer import batch_eval_model_on_the_fly, eval_model_on_the_fly
from chloe.utils.sim_utils import store_file_in_shared_memory


def main():
    """Utility function for evaluating a trained RL agent.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config file with generic agent parameters for its instanciation",
        type=str,
        default="cfg.yml",
    )
    parser.add_argument(
        "--data", help="path to the patient data file", type=str, required=True
    )
    parser.add_argument(
        "--suffix", help="suffix to add to the metric file", type=str, default=""
    )
    parser.add_argument(
        "--shared_data_socket",
        help="pyarrow plasma socket to be used for data sharing.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--end_training_eval_data",
        help="pyarrow plasma socket to be used for data sharing.",
        type=str,
        default="endvalid",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        default=False,
        help="execute last config.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="deterministic execction.",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        default=False,
        help="no logging",
    )
    parser.add_argument(
        "--no_filedeletion",
        action="store_true",
        default=False,
        help="no logging",
    )
    parser.add_argument(
        "--output", help="path to outputs - will results here", type=str, required=True,
    )
    parser.add_argument("--cuda_idx", help="gpu to use", type=int, default=None)
    parser.add_argument(
        "--no_replace_if_present",
        action="store_true",
        help="if specify, the data will not be stored in the shared data"
        " socket if they are already present",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="run the eval in batch mode",
    )
    parser.add_argument(
        "--batch_size", help="eval batch size", type=int, default=None
    )
    parser.add_argument(
        "--run_ID", help="run identifier for logging purposes", type=int, default=0
    )
    parser.add_argument(
        "--n_workers", help="number of cpu workers.", type=int, default=4
    )
    args = parser.parse_args()
    args.end_training_eval_data_fp = args.data

    # if shared_data_socket is not None
    if args.shared_data_socket is not None:
        end_training_eval_data = args.end_training_eval_data
        store_file_in_shared_memory(
            args.shared_data_socket,
            args.data,
            prefix=end_training_eval_data,
            replace_if_present=not args.no_replace_if_present,
        )
        args.data = end_training_eval_data

    if not (args.cuda_idx is None):
        if not torch.cuda.is_available():
            print(
                f"No cuda found. Defaulting the cuda_idx param"
                f' from From "{args.cuda_idx}" to "None".'
            )
            args.cuda_idx = None

    if args.config is not None:
        config = os.path.abspath(
            os.path.join(args.output, f"run_{args.run_ID}/{args.config}")
        )
        with open(config, "r") as stream:
            hyper_params = load(stream, Loader=yaml.FullLoader)
    else:
        hyper_params = {}

    mlflow.start_run(run_id=hyper_params.get("mlflow_uid"))

    run(args, hyper_params)

    mlflow.end_run()


def run(args, hyper_params):
    """Defines the setup needed for running the evaluation process of a trained agent.

    This is an utility function that properly defines the setup
    needed for running the evaluation process as well as launching the process.

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

    # try to register the env in gym if not yet done
    try:
        register(
            id=GYM_ENV_ID,
            entry_point="chloe.simulator.simulator:PatientInteractionSimulator",
        )
    except Exception:
        print("The environment [simPa-v0] is already registered in gym")
        pass

    print(f"output: {args.output} - Last: {args.last}")
    if not args.batch_mode:
        eval_model_on_the_fly(hyper_params, args, args.run_ID, args.last)
    else:
        batch_eval_model_on_the_fly(hyper_params, args, args.run_ID, args.last)

if __name__ == "__main__":
    main()
