#!/usr/bin/env python

import argparse

from chloe.utils.sim_utils import store_file_in_shared_memory


def main():
    """This is an utility function for copying data in a shared memory.

    This data will be used later on by training/evaluating processes
    for instantiating simulator (environment) objects.

    """
    parser = argparse.ArgumentParser()
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
        required=True,
    )
    parser.add_argument(
        "--data_prefix",
        help="prefix to be used for data sharing.",
        type=str,
        default="training",
    )
    parser.add_argument(
        "--eval_data_prefix",
        help="prefix to be used for eval_data sharing.",
        type=str,
        default="validate",
    )
    args = parser.parse_args()

    # if eval data is none, set it to data
    if args.eval_data is None:
        args.eval_data = args.data

    tmp_data = args.data
    data_prefix = args.data_prefix
    store_file_in_shared_memory(
        args.shared_data_socket, args.data, prefix=data_prefix, replace_if_present=True,
    )
    if args.eval_data != tmp_data:
        eval_data_prefix = args.eval_data_prefix
        store_file_in_shared_memory(
            args.shared_data_socket,
            args.eval_data,
            prefix=eval_data_prefix,
            replace_if_present=True,
        )


if __name__ == "__main__":
    main()
