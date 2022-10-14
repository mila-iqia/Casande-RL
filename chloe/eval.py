#!/usr/bin/env python

import argparse
import os

import torch
import yaml
from gym.envs.registration import register
from yaml import load

from chloe.evaluator.evaluator import evaluate
from chloe.utils.logging_utils import suffix_output_dir_with_datetime
from chloe.utils.sim_utils import store_file_in_shared_memory


def main():
    """Utility function for evaluating a trained RL agent.

    This is an utility function for evaluating the policy of an rl agent.
    It will use the provided data to instantiate simulators
    the agent will interact with for the evaluation process.

    The term 'relevancy' is used here to refer to an inquiry regarding a symptom or
    an antecedent that the patient is actually experiencing. Otherwise, the query is
    said to be irrelevant.

    The provided eval_coeffs are used to computed 'aggregated scores' that combine 4
    components:
        - A differential (classification) based metric
        - Symptom Discovery score (recall on symptoms)
        - Antecedent/Risk Factor Discovery score (recall on antecedents)
        - Negative Response Pertinence score (information gain on irrelevant queries)

    There are 4 types of aggregated scores that are computed depending on which
    differential based metric is used:
        - avg_precision_agg_score: the intersection between the top-k of the predicted
          differential and the top-k of the real differential is used as the
          differential metric. This metric is also refer to as "topk-in-topk" metric.
        - avg_precision_full_agg_score: the intersection between the top-k of the
          predicted differential and the whole real differential is used as the
          differential metric. This metric is also refer to as "topk-in-full" metric.
        - avg_ncg_agg_score: the NCG metric between the predicted differential and the
          real differential is used as the differential metric. With respect to the
          previous two metrics, the probabilities within the real differential are taken
          into account while computing the metric.
        - avg_ndcg_agg_score: the NDCG metric between the predicted differential and the
          real differential is used as the differential metric. With respect to the
          previous three metrics, the order and probabilities within the real
          differential are taken into account while computing the metric.

    This script produces two files in the specified output directory:
        - metric_results.json
        - evaluation_stats.pkl

    The 'metric_results.json' file contains the computed metrics that are computed
    per pathology and globally (refer to key `per_patho` and `global` in the json data).
    Statistics are computed for most of the computed metrics. More specifically, the
    following keys are used to refer to them:
        - Min: Minimum
        - Max: Maximum
        - Avg: Average
        - Median: Median
        - Q25: First quartile
        - Q75: Third Quartile

    For e.g., the key `turns_Min` will refer to the mininum number of turns in an
    interaction session during the evaluation process.

    Here is the list of metrics for which statistics are computed:
        - turns: number of turns
        - rewards: sum of rewards received from the environment
        - discounted_rewards: discounted sum of rewards received from the environment
        - num_repeated_symptoms: number of repeated symptoms
        - num_relevant_inquiries: number of relevant inquiries
        - num_simulated_evidences: number of experienced symptoms/antecedents by
          patients
        - relevancy_symptoms_ratio: ratio of relevant queries by episode length
        - simulated_evidence_ratio: ratio of the number experienced symptoms/antecedents
          by episode length
        - num_irrelevant_inquiries: number of irrelevant inquiries
        - num_evidenced_inquiries: number of inquiries that were experienced by patients
        - num_inquired_atcd: number of inquired antecedents
        - num_inquired_symptoms: number of inquired symptoms
        - num_relevant_atcd: number of inquired antecedents experienced by the patients
        - num_relevant_symptoms: number of inquired symptoms experienced by the patients
        - precision_relevant_atcd: precision of the relevancy of the antecedent inquiry
        - precision_relevant_symptoms: precision of the relevancy of the symptom inquiry
        - recall_relevant_atcd: recall of the relevancy of the antecedent inquiry
        - recall_relevant_symptoms: recall of the relevancy of the symptom inquiry
        - num_experienced_atcd: number of experienced antecedents by patient
        - num_experienced_symptoms: number of experienced symptoms by patient
        - avg_info_gain_on_irrelevancy: information gain on irrelevant queries
        - differential_ndcg_metric: differential NDCG metric
        - differential_ncg_metric: differential NCG metric
        - differential_avg_precision_metric: differential topk-in-topk metric
        - differential_avg_precision_full_metric: differential topk-in-full metric
        - avg_ndcg_agg_score: aggragated score based on NDCG metric
        - avg_ncg_agg_score: aggragated score based on NCG metric
        - avg_precision_agg_score: aggragated score based on topk-in-topk metric
        - avg_precision_full_agg_score: aggragated score based on topk-in-full metric

    Besides, classification metrics with respect to the patient underlying simulated
    pathology are computed. These include:
        - accuracy
        - balanced_accuracy
        - precision
        - recall
        - f1
        - top-1-accuracy
        - top-2-accuracy
        - top-3-accuracy
        - top-5-accuracy

    The difference between accuracy and top-1-accuracy is that the former considers the
    situation where the agent reaches the allowed maximum number of turns as a
    misclassification while the latter does not.

    It is important to notice that those metrics are computed only if the flag
    `--compute_metric_flag` is passed to the command.

    The 'evaluation_stats.pkl' file contains the saved trajectories to be able to compute
    or recompute further metrics. Besides the data required to compute the above
    mentioned statistics (using the same key), the following data is also saved:
        - all_proba_dist: the evolution of the probability distribution during the
          interaction session or episode
        - all_q_values: the evolution of the Q-values during the interaction session
          or episode
        - all_aux_rewards: all auxiliary rewards that the agent received during the
          course of the interaction session or episode
        - all_atcd_actions: all indicators (0/1) that a query was about an antecedent
          during the course of the interaction session or episode
        - all_relevant_actions: all indicators (0/1) that a query was relevant
          during the course of the interaction session or episode
        - all_repeated_actions: all indicators (0/1) that a query was a repeated one
          during the course of the interaction session or episode

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config file with generic agent parameters for its instanciation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        help="path to the model to be loaded for this evaluation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data", help="path to the patient data file", type=str, required=True
    )
    parser.add_argument(
        "--shared_data_socket",
        help="pyarrow plasma socket to be used for data sharing.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sharing_prefix",
        help="pyarrow plasma socket to be used for data sharing.",
        type=str,
        default="testing",
    )
    parser.add_argument(
        "--output", help="path to outputs - will results here", type=str, default="./",
    )
    parser.add_argument("--cuda_idx", help="gpu to use", type=int, default=None)
    parser.add_argument(
        "--n_envs", help="number of envs to instanciate", type=int, default=1
    )
    parser.add_argument("--seed", help="seed to be used", type=int, default=None)
    parser.add_argument(
        "--max_trajectories",
        help="max trajectories for evaluation purposes",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max_generation",
        help="max trajectories per pathology to be generated",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--topk",
        help="topk pathologies and their likelihood to output as differential diagnosis",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--eval_coeffs",
        help="the coefficients to be used to weight each component of the aggregated "
        "evaluation metrics (Differential Diagnosis score, Symptom Discovery score, "
        "Risk Factor Discovery score, Negative Response Pertinence score). "
        "If not specified, the ones in the config file, if any, will be used.",
        type=float,
        nargs="*",
    )
    parser.add_argument(
        "--sample_indices_flag",
        action="store_true",
        help="if specify, the patient indices to be evaluated are sampled upfront",
    )
    parser.add_argument(
        "--compute_metric_flag",
        action="store_true",
        help="if specify, the metrics will be computed",
    )
    parser.add_argument(
        "--no_replace_if_present",
        action="store_true",
        help="if specify, the data will not be stored in the shared data"
        " socket if they are already present",
    )
    parser.add_argument(
        "--datetime_suffix",
        help="add the following datetime suffix to the output dir: "
        "<output_dir>/<yyyymmdd>/<hhmmss>",
        action="store_true",
    )
    args = parser.parse_args()

    # args.eval_coeffs
    if args.eval_coeffs is not None:
        assert len(args.eval_coeffs) >= 4, "the coefficient's number must be at least 4"

    # if shared_data_socket is not None
    if args.shared_data_socket is not None:
        data_prefix = args.sharing_prefix
        store_file_in_shared_memory(
            args.shared_data_socket,
            args.data,
            prefix=data_prefix,
            replace_if_present=not args.no_replace_if_present,
        )
        args.data = data_prefix

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

    run(args, hyper_params)


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

    evaluate(
        gym_env_id=GYM_ENV_ID, args=args, params=hyper_params,
    )


if __name__ == "__main__":
    main()
