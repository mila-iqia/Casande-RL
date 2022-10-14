import copy
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from rlpyt.envs.gym import make as gym_make
from rlpyt.utils.seed import make_seed, set_envs_seeds

from chloe.evaluator.custom_serial_eval_collector import CustomSerialEvalCollector
from chloe.simulator.simulator import RewardConfig
from chloe.utils.agent_utils import AgentFactory
from chloe.utils.clf_loss_utils import ClassifierLossFactory
from chloe.utils.dev_utils import (
    get_current_git_commit_hash,
    get_dev_dependencies,
    initialize_seed,
)
from chloe.utils.eval_utils import MetricFactory
from chloe.utils.model_utils import (
    load_component_metadata,
    load_environment_metadata,
    load_model_metadata,
)
from chloe.utils.reward_shaping_utils import RewardShapingFactory
from chloe.utils.sim_utils import load_and_check_data
from chloe.utils.train_utils import SimPaTrajEvalInfo

METRIC_FILE_NAME = "metric_results.json"
DATA_FILE_NAME = "evaluation_stats.pkl"


def create_environments(gym_env_id, args, sim_params):
    """Method for creating the environments for evaluation.

    Parameters
    ----------
    gym_env_id: str
        id of the gym environment.
    args: dict
        the arguments as provided in the command line.
    sim_params: dict
        the simulator parameters as provided in the configuration file.

    Return
    ------
    envs: list
        list of instantiated environments.

    """
    env_kwargs = dict(
        id=gym_env_id,
        patient_filepath=args.data,
        shared_data_socket=args.shared_data_socket,
        **sim_params,
    )
    n_envs = args.n_envs

    env_list = [gym_make(**env_kwargs) for i in range(n_envs)]
    set_envs_seeds(env_list, args.seed)

    return env_list


def create_agent(env, args, params):
    """Method for creating the agent to be used during evaluation.

    Parameters
    ----------
    env: object
        an instance of the simulator on which the agent will interact.
    args: dict
        the arguments as provided in the command line.
    params: dict
        the parameters as provided in the configuration file.

    Return
    ------
    agent: object
        the agent to be used during the evaluation.

    """
    rank = 0
    n_envs = args.n_envs
    global_B = n_envs
    env_ranks = list(range(rank * n_envs, (rank + 1) * n_envs))

    agent_params = load_component_metadata(
        "agent_params", params, prefix="agent_", log_hp_flag=False,
    )
    snapshot = {}
    if os.path.exists(args.model_path):
        snapshot = torch.load(
            args.model_path,
            map_location="cpu" if args.cuda_idx is None else f"cuda:{args.cuda_idx}",
        )
    else:
        raise ValueError(f"the path [{args.model_path}] does not exist!")
    modelCls, model_params = load_model_metadata(params, log_hp_flag=False)

    # eventually get the symptom to observation mapping
    symp_2_obs_map = None
    if hasattr(env, "get_symptom_to_observation_mapping"):
        symp_2_obs_map = env.get_symptom_to_observation_mapping()
    model_params["symptom_2_observation_map"] = symp_2_obs_map
    patho_severity = None
    if hasattr(env, "get_pathology_severity"):
        patho_severity = env.get_pathology_severity()
    model_params["patho_severity"] = patho_severity
    model_params["hierarchical_map"] = (
        copy.deepcopy(env.get_hierarchical_symptom_dependency())
        if hasattr(env, "get_hierarchical_symptom_dependency")
        else None
    )
    model_params["symptom_2_symptom_association"] = (
        copy.deepcopy(env.get_evidence_2_evidence_association())
        if hasattr(env, "get_evidence_2_evidence_association")
        else None
    )
    model_params["symptom_2_patho_association"] = (
        copy.deepcopy(env.get_evidence_2_pathology_association())
        if hasattr(env, "get_evidence_2_pathology_association")
        else None
    )
    model_params["symp_default_values"] = (
        copy.deepcopy(env.get_evidence_default_value_in_obs())
        if hasattr(env, "get_evidence_default_value_in_obs")
        else None
    )
    # check consistency of `include_turns_in_state` between simulator and model
    if hasattr(env, "include_turns_in_state"):
        include_turns_in_state = env.include_turns_in_state
        assert (not ("include_turns_in_state" in model_params)) or (
            include_turns_in_state == model_params["include_turns_in_state"]
        )
    # check consistency between `n_atoms` in agent and model if any
    model_params["n_atoms"] = agent_params.get("n_atoms")
    assert (not ("n_atoms" in agent_params)) or (
        model_params.get("n_atoms") == agent_params["n_atoms"]
    )

    agent_state_dict = snapshot.get("agent_state_dict", None)
    agent = AgentFactory().create(
        params["agent"],
        None,
        None,
        ModelCls=modelCls,
        model_kwargs=model_params,
        initial_model_state_dict=agent_state_dict,
        **agent_params,
    )
    agent.initialize(
        env.spaces, share_memory=False, global_B=global_B, env_ranks=env_ranks
    )
    if hasattr(agent, "give_V_min_max"):
        algo_params = load_component_metadata(
            "algo_params", params, prefix="algo_", log_hp_flag=False,
        )
        v_min = algo_params.get("V_min", None)
        v_max = algo_params.get("V_max", None)
        assert (v_min is not None) and (v_max is not None) and (v_min <= v_max)
        agent.give_V_min_max(v_min, v_max)
    return agent


def processing_collected_trajectories(traj_infos):
    """Preprocesses the trajectories collected by the agent during its evaluation.

    This method preprocesses the trajectories collected by the agent during
    its interaction with the simulator while being evaluated.

    Parameters
    ----------
    traj_infos: list.
        list of trajectoy infos collected by the agent.

    Return
    ------
    infos: dict
        dictionary containing the data needed to compute the evaluation metrics
        from the collected trajectory infos.

    """
    info_dict = {
        "y_true": [],
        "y_pred": [],
        "y_pred_dist": [],
        "y_differential_indices": [],
        "y_differential_probas": [],
        "turns": [],
        "age": [],
        "sex": [],
        "race": [],
        "geo": [],
        "rewards": [],
        "discounted_rewards": [],
        "num_repeated_symptoms": [],
        "num_relevant_inquiries": [],
        "num_irrelevant_inquiries": [],
        "num_evidenced_inquiries": [],
        "num_inquired_atcd": [],
        "num_inquired_symptoms": [],
        "num_relevant_atcd": [],
        "num_relevant_symptoms": [],
        "precision_relevant_atcd": [],
        "precision_relevant_symptoms": [],
        "recall_relevant_atcd": [],
        "recall_relevant_symptoms": [],
        "num_experienced_atcd": [],
        "num_experienced_symptoms": [],
        "avg_info_gain_on_irrelevancy": [],
        "differential_ndcg_metric": [],
        "differential_ncg_metric": [],
        "differential_avg_precision_metric": [],
        "differential_avg_precision_full_metric": [],
        "avg_ndcg_agg_score": [],
        "avg_ncg_agg_score": [],
        "avg_precision_agg_score": [],
        "avg_precision_full_agg_score": [],
        "first_symptoms": [],
        "num_simulated_evidences": [],
        "relevancy_symptoms_ratio": [],
        "simulated_evidence_ratio": [],
        "simulated_patients": [],
        "collected_infos": [],
        "inquired_evidences": [],
        "action_relevancy": [],
        "all_proba_dist": [],
        "all_q_values": [],
        "all_atcd_actions": [],
        "all_relevant_actions": [],
        "all_repeated_actions": [],
        "all_aux_rewards": [],
    }

    zero = np.array([0])
    for info in traj_infos:
        info_dict["age"].append(info.get("_sim_age", None))
        info_dict["sex"].append(info.get("_sim_sex", None))
        info_dict["race"].append(info.get("_sim_race", None))
        info_dict["geo"].append(info.get("_sim_geo", None))

        info_dict["y_true"].append(info.get("_metric_y_true", None))
        info_dict["y_pred"].append(info.get("_metric_y_pred", None))
        info_dict["y_pred_dist"].append(info.get("_dist_info", None))
        info_dict["y_differential_indices"].append(
            info.get("_metric_diffential_indices", None)
        )
        info_dict["y_differential_probas"].append(
            info.get("_metric_diffential_probas", None)
        )
        info_dict["turns"].append(info.get("Length", None))
        info_dict["rewards"].append(info.get("Return", None))
        info_dict["discounted_rewards"].append(info.get("DiscountedReturn", None))
        info_dict["num_repeated_symptoms"].append(info.get("NumRepeatedSymptoms", None))
        info_dict["num_relevant_inquiries"].append(
            sum(info.get("_relevant_actions", zero))
        )
        info_dict["num_irrelevant_inquiries"].append(info.get("_num_irrelevancy", None))
        info_dict["num_evidenced_inquiries"].append(
            info.get("_num_evidenced_inquiries", None)
        )
        info_dict["num_inquired_atcd"].append(info.get("NumInquiredAntecedents", None))
        info_dict["num_inquired_symptoms"].append(info.get("NumInquiredSymptoms", None))
        info_dict["num_relevant_atcd"].append(info.get("NumRelevantAntecedents", None))
        info_dict["num_relevant_symptoms"].append(info.get("NumRelevantSymptoms", None))
        info_dict["recall_relevant_atcd"].append(
            info.get("RecallRelevantAntecedents", None)
        )
        info_dict["recall_relevant_symptoms"].append(
            info.get("RecallRelevantSymptoms", None)
        )
        info_dict["precision_relevant_atcd"].append(
            info.get("PrecisionRelevantAntecedents", None)
        )
        info_dict["precision_relevant_symptoms"].append(
            info.get("PrecisionRelevantSymptoms", None)
        )
        info_dict["num_experienced_atcd"].append(
            info.get("_num_experienced_antecedents", None)
        )
        info_dict["num_experienced_symptoms"].append(
            info.get("_num_experienced_symptoms", None)
        )
        info_dict["avg_info_gain_on_irrelevancy"].append(
            info.get("AvgInfoGainOnIrrelevancy", None)
        )
        info_dict["differential_ndcg_metric"].append(info.get("NdcgDistPred", None))
        info_dict["differential_ncg_metric"].append(info.get("NcgDistPred", None))
        info_dict["differential_avg_precision_metric"].append(
            info.get("AvgPrecDistPred", None)
        )
        info_dict["differential_avg_precision_full_metric"].append(
            info.get("AvgPrecFullDistPred", None)
        )
        info_dict["avg_ndcg_agg_score"].append(info.get("NdcgBasedAggScore", None))
        info_dict["avg_ncg_agg_score"].append(info.get("NcgBasedAggScore", None))
        info_dict["avg_precision_agg_score"].append(info.get("AvgPrecAggScore", None))
        info_dict["avg_precision_full_agg_score"].append(
            info.get("AvgPrecFullAggScore", None)
        )
        info_dict["first_symptoms"].append(info.get("_first_symptom", None))
        info_dict["num_simulated_evidences"].append(info.get("_num_evidences", None))
        info_dict["relevancy_symptoms_ratio"].append(
            sum(info.get("_relevant_actions", zero)) / max(info.get("Length", 0), 1)
        )
        info_dict["simulated_evidence_ratio"].append(
            info.get("_num_evidences", zero) / max(info.get("Length", 0), 1)
        )
        info_dict["simulated_patients"].append(info.get("_simulated_patient", None))
        info_dict["collected_infos"].append(info.get("_gathered_infos", None))
        info_dict["inquired_evidences"].append(info.get("_inquired_symptoms", None))
        info_dict["action_relevancy"].append(info.get("_relevant_actions", None))
        info_dict["all_proba_dist"].append(info.get("_all_proba_distributions", None))
        info_dict["all_q_values"].append(info.get("_all_q_values", None))
        info_dict["all_atcd_actions"].append(info.get("_atcd_actions", None))
        info_dict["all_relevant_actions"].append(info.get("_relevant_actions", None))
        info_dict["all_repeated_actions"].append(info.get("_repeated_actions", None))
        info_dict["all_aux_rewards"].append(info.get("_all_rewards_data", None))

    info_dict["y_pred_dist"] = [
        a.numpy() if isinstance(a, torch.Tensor) else a
        for a in info_dict["y_pred_dist"]
    ]
    info_dict["all_proba_dist"] = [
        None
        if a is None
        else [x.numpy() if isinstance(x, torch.Tensor) else x for x in a]
        for a in info_dict["all_proba_dist"]
    ]
    info_dict["all_q_values"] = [
        None
        if a is None
        else [x.numpy() if isinstance(x, torch.Tensor) else x for x in a]
        for a in info_dict["all_q_values"]
    ]
    not_int_list = [
        "simulated_evidence_ratio",
        "relevancy_symptoms_ratio",
        "inquired_evidences",
        "action_relevancy",
        "y_differential_indices",
        "y_differential_probas",
        "y_pred_dist",
        "precision_relevant_atcd",
        "precision_relevant_symptoms",
        "recall_relevant_atcd",
        "recall_relevant_symptoms",
        "avg_info_gain_on_irrelevancy",
        "differential_ndcg_metric",
        "differential_ncg_metric",
        "differential_avg_precision_metric",
        "differential_avg_precision_full_metric",
        "avg_ndcg_agg_score",
        "avg_ncg_agg_score",
        "avg_precision_agg_score",
        "avg_precision_full_agg_score",
        "simulated_patients",
        "collected_infos",
        "all_proba_dist",
        "all_q_values",
    ]
    not_to_transform = [
        "all_proba_dist",
        "all_q_values",
        "all_atcd_actions",
        "all_relevant_actions",
        "all_repeated_actions",
        "all_aux_rewards",
    ]
    for k in info_dict.keys():
        if k in not_to_transform:
            continue
        if k not in not_int_list:
            if info_dict[k] and info_dict[k][0] is not None:
                info_dict[k] = np.array(info_dict[k], dtype=np.int)
        else:
            info_dict[k] = np.array(info_dict[k])

    return info_dict


def get_traj_info_kwargs(params, agent):
    """Defines the variable arguments needed in trajectory info instances.

    This function provides additional parameters for auxiliary rewards
    involved in the reward shaping procedure.

    Parameters
    ----------
    params: dict
        the parameters as provided in the configuration file.
    agent: object
        the agent to be used during the evaluation.

    Returns
    -------
    result: dict
        the dictionary of variable argument for the trajectory info instances.
    """
    # get the patho severity
    patho_severity = agent.model_kwargs.get("patho_severity", None)
    if patho_severity is not None:
        patho_severity = torch.tensor(patho_severity).float()

    # get the runner params
    runner_params = params.get("runner_params", {})
    # need to always compute auxiliary rewards when evaluating an agent
    traj_aux_reward_flag = True
    topk = runner_params.get("topk", 1)
    eval_coeffs = runner_params.get("eval_coeffs", None)
    if eval_coeffs is None:
        eval_coeffs = [1.0] * 4

    # get the algo params
    algo_params = params.get("algo_params", {})
    discount = algo_params.get("discount", 1.0)
    v_min = getattr(agent, "V_min", None)
    v_max = getattr(agent, "V_max", None)
    atoms = getattr(agent, "n_atoms", None)
    if not (v_min is None or v_max is None or atoms is None):
        p_z = torch.linspace(v_min, v_max, atoms)
    else:
        p_z = None
    aux_reward_info = dict(
        traj_auxiliary_reward_flag=traj_aux_reward_flag,
        env_reward_coef=algo_params.get("env_reward_coef", 1.0),
        clf_reward_coef=algo_params.get("clf_reward_coef", 1.0),
        clf_reward_flag=algo_params.get("clf_reward_flag", False),
        clf_reward_min=algo_params.get("clf_reward_min", None),
        clf_reward_max=algo_params.get("clf_reward_max", None),
        clf_reward_func=algo_params.get("clf_reward_func", None),
        clf_reward_kwargs=algo_params.get("clf_reward_kwargs", {}),
        clf_reward_factory=ClassifierLossFactory(),
        reward_shaping_coef=algo_params.get("reward_shaping_coef", 1.0),
        reward_shaping_flag=algo_params.get("reward_shaping_flag", False),
        reward_shaping_min=algo_params.get("reward_shaping_min", None),
        reward_shaping_max=algo_params.get("reward_shaping_max", None),
        reward_shaping_func=algo_params.get("reward_shaping_func", None),
        reward_shaping_kwargs=algo_params.get("reward_shaping_kwargs", {}),
        reward_shaping_factory=RewardShapingFactory(),
        patho_severity=patho_severity,
    )
    result = {}
    result["aux_reward_info"] = aux_reward_info
    result["discount"] = discount
    result["topk"] = topk
    result["eval_coeffs"] = eval_coeffs
    result["p_z"] = p_z
    return result


def evaluate(gym_env_id, args, params=None):
    """Method for evaluating a policy against a gym environment.

    Parameters
    ----------
    gym_env_id: str
        id of the gym environment.
    args: dict
        the arguments as provided in the command line.
    params: dict
        the parameters as provided in the configuration file. Default: None

    Return
    ------
    None

    """

    if params is None:
        params = {}

    if args.seed is None:
        args.seed = make_seed()

    print(f"Using seed [{args.seed}] for this evaluation.")

    # initialize the random generator
    initialize_seed(args.seed)

    # instantiate the envs
    sim_params, rew_params = load_environment_metadata(params, log_hp_flag=False)
    sim_params["reward_config"] = RewardConfig(**rew_params)
    if "use_initial_symptom_flag" not in sim_params:
        sim_params["use_initial_symptom_flag"] = True
    env_list = create_environments(gym_env_id, args, sim_params)

    # instantiate the agent
    agent = create_agent(env_list[0], args, params)
    agent.to_device(args.cuda_idx)

    # get symptom info
    symptom_infos = load_and_check_data(
        sim_params["symptom_filepath"], [], key_name="name"
    )

    # get patho info
    pathology_infos = load_and_check_data(
        sim_params["condition_filepath"], [], key_name="condition_name"
    )

    # setup the sample collector for this evaluation
    if not args.max_trajectories:
        args.max_trajectories = env_list[0].rb.num_rows

    # setting the max steps
    max_steps = sim_params["max_turns"] * args.max_trajectories

    # defining the trajInfo class
    traj_info_kwargs = get_traj_info_kwargs(params, agent)
    traj_info_kwargs["topk"] = args.topk
    if args.eval_coeffs is not None:
        traj_info_kwargs["eval_coeffs"] = args.eval_coeffs
    TrajInfoCls = SimPaTrajEvalInfo
    if traj_info_kwargs:
        # Avoid passing at init.
        for k, v in traj_info_kwargs.items():
            setattr(TrajInfoCls, "_" + k, v)

    # defining the collector class
    eval_collector = CustomSerialEvalCollector(
        envs=env_list,
        agent=agent,
        TrajInfoCls=TrajInfoCls,
        max_T=max_steps // len(env_list),
        max_trajectories=args.max_trajectories,
        out_path=args.output,
        max_generation=args.max_generation,
        topk=args.topk,
        sample_indices_flag=args.sample_indices_flag,
        seed=args.seed,
    )
    # collect samples
    print("BEGIN SAMPLE COLLECTIONS")
    traj_infos = eval_collector.collect_evaluation(itr=1)

    # processing the collected samples
    print("BEGIN STATS COMPUTATION")
    info_dict = processing_collected_trajectories(traj_infos)

    # compute metrics
    result = (
        {} if not args.compute_metric_flag else compute_metric(info_dict, args.topk)
    )

    # add patho and symptoms info
    result["pathos"] = [pathology_infos[0], pathology_infos[1]]
    result["symptoms"] = [symptom_infos[0], symptom_infos[1]]

    # add git hash
    git_hash = get_current_git_commit_hash()
    result["git_hash"] = git_hash

    # add dev dependencies
    result["dev_dependencies"] = get_dev_dependencies()

    # add evaluation params
    sim_params.pop("reward_config", None)
    result["evaluation_params"] = {
        "seed": int(args.seed),
        "reward_config": {k: float(rew_params[k]) for k in rew_params.keys()},
        "sim_config": sim_params,
    }

    # save results
    params_log_file = os.path.join(args.output, METRIC_FILE_NAME)
    with open(params_log_file, "w") as f:
        json.dump(result, f, indent=4)

    data_to_save = {
        "data": info_dict,
        "metrics": result,
    }
    pickle.dump(data_to_save, open(os.path.join(args.output, DATA_FILE_NAME), "wb"))
    print("END")


def compute_metric(info_dict, topk):
    """Method for computing the evaluation metrics.

    Some of the y_pred are -1 (when max turns is reached) which doesn't belong
    to any class. These appear as warning flags by sklearn.

    Parameters
    ----------
    info_dict: dict
        samples information from the exection of the agent
        on several simulation episodes.
    topk: int
        the top k number of pathologies to be considered in the differential.

    Return
    ------
    result: dict
        dictionary containing the computed metrics.

    """

    patho_set = set(info_dict["y_true"])
    patho_indices = {int(k): np.where(info_dict["y_true"] == k)[0] for k in patho_set}

    metric_factory = MetricFactory()
    eval_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    eval_metrics.extend([f"top-{k}-accuracy" for k in [1, 2, 3, 5]])
    if f"top-{topk}-accuracy" not in eval_metrics:
        eval_metrics.append(f"top-{topk}-accuracy")
    stats_list = [
        "turns",
        "rewards",
        "discounted_rewards",
        "num_repeated_symptoms",
        "num_relevant_inquiries",
        "num_simulated_evidences",
        "relevancy_symptoms_ratio",
        "simulated_evidence_ratio",
        "num_irrelevant_inquiries",
        "num_evidenced_inquiries",
        "num_inquired_atcd",
        "num_inquired_symptoms",
        "num_relevant_atcd",
        "num_relevant_symptoms",
        "precision_relevant_atcd",
        "precision_relevant_symptoms",
        "recall_relevant_atcd",
        "recall_relevant_symptoms",
        "num_experienced_atcd",
        "num_experienced_symptoms",
        "avg_info_gain_on_irrelevancy",
        "differential_ndcg_metric",
        "differential_ncg_metric",
        "differential_avg_precision_metric",
        "differential_avg_precision_full_metric",
        "avg_ndcg_agg_score",
        "avg_ncg_agg_score",
        "avg_precision_agg_score",
        "avg_precision_full_agg_score",
    ]
    unique_count_list = ["first_symptoms", "inquired_evidences"]

    result = {
        "global": OrderedDict(),
        "per_patho": OrderedDict(),
    }

    result["global"]["confusion_matrix"] = (
        metric_factory.evaluate(
            "confusion_matrix", info_dict["y_true"], info_dict["y_pred"]
        )
        .astype(np.int)
        .tolist()
    )
    confusion_matrix_support = sorted(
        [int(a) for a in patho_set.union(set(info_dict["y_pred"]))]
    )
    result["global"]["confusion_matrix_support"] = confusion_matrix_support
    for metric in eval_metrics:
        is_top = metric.startswith("top-")
        tmp_pred = info_dict["y_pred_dist"] if is_top else info_dict["y_pred"]
        result["global"][metric] = (
            0.0
            if is_top and (tmp_pred is None or tmp_pred[0] is None)
            else metric_factory.evaluate(metric, info_dict["y_true"], tmp_pred)
        )
    for field in stats_list:
        stats = metric_factory.misc_stats(info_dict[field], prefix=field + "_")
        result["global"].update(stats)
    for field in unique_count_list:
        count_dict = metric_factory.unique_counts(info_dict[field])
        result["global"][field + "_count"] = count_dict

    for patho in patho_indices.keys():
        result["per_patho"][patho] = OrderedDict()
        indices = patho_indices[patho]

        y_true = info_dict["y_true"][indices]
        y_pred = info_dict["y_pred"][indices]
        y_pred_dist = info_dict["y_pred_dist"][indices]
        for metric in eval_metrics:
            kwargs = {}
            if not metric.endswith("accuracy"):
                kwargs = {"labels": [y_true[0]]}
            is_top = metric.startswith("top-")
            tmp_pred = y_pred_dist if is_top else y_pred
            result["per_patho"][patho][metric] = (
                0.0
                if is_top and (tmp_pred is None or tmp_pred[0] is None)
                else metric_factory.evaluate(metric, y_true, tmp_pred, **kwargs)
            )
        for field in stats_list:
            stats = metric_factory.misc_stats(
                info_dict[field][indices], prefix=field + "_"
            )
            result["per_patho"][patho].update(stats)
        for field in unique_count_list:
            count_dict = metric_factory.unique_counts(info_dict[field][indices])
            result["per_patho"][patho][field + "_count"] = count_dict

    return result
