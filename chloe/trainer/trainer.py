import copy
import json
import os

import mlflow
import numpy as np
import torch
from rlpyt.envs.gym import make as gym_make
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.seed import make_seed

from chloe.eval import run as eval_run
from chloe.evaluator.agent_eval_rp import main as metric_computation
from chloe.evaluator.batchEvaluator import evaluate
from chloe.evaluator.evaluator import DATA_FILE_NAME, METRIC_FILE_NAME, create_agent
from chloe.pretraining.pretrainer import BEST_PRETRAIN_MODEL_NAME, pretrain
from chloe.simulator.simulator import RewardConfig
from chloe.simulator.evalEnv import environment as BatchEvalEnv
from chloe.utils.agent_utils import AgentFactory
from chloe.utils.algo_utils import AlgoFactory
from chloe.utils.collector_utils import CollectorFactory
from chloe.utils.dev_utils import initialize_seed
from chloe.utils.logging_utils import log_metrics
from chloe.utils.model_utils import (
    load_component_metadata,
    load_environment_metadata,
    load_model_metadata,
    load_optimizer_metadata,
)
from chloe.utils.replay_buffer_utils import ReplayBufferFactory
from chloe.utils.runner_utils import BEST_MODEL_NAME, RunnerFactory
from chloe.utils.sampler_utils import SamplerFactory
from chloe.utils.scheduler_utils import numpy_sigmoid_scheduler
from chloe.utils.train_utils import (
    SimPaTrajInfo,
    load_pretrained_snapshot,
    reload_snapshot,
    train,
)

LAST_SNAPSHOT_NAME = "params.pkl"


def createAttrDict(params):
    result = AttrDict({})
    for k in params:
        if isinstance(params.get(k), dict):
            result[k] = createAttrDict(params.get(k))
        else:
            result[k] = params.get(k)
    return result


def pretrain_on_the_fly(gym_env_id, algo_params, agent, params, args):
    """Method for pretraining the agent on the fly against a gym environment.

    Parameters
    ----------
    gym_env_id: str
        Id of the gym environment.
    algo_params: dict
        dictionary defining the pretraining parameters.
    agent: object
        the agent to be initialized with the pretrained model.
    params: dict
        The parameters as provided in the configuration file.
    args: dict
        The arguments as provided in the command line.

    Return
    ------
    None

    """

    pretrain_flag = algo_params.get("pretrain_flag", False)
    if not pretrain_flag:
        return
    pretrain_args = AttrDict(
        {
            "data": args.data,
            "eval_data": args.eval_data,
            "output": args.output,
            "datetime_suffix": False,
            "no_data_corrupt": False,
            "no_replace_if_present": args.no_replace_if_present,
            "num_epochs": algo_params.get("pretrain_epochs", 100),
            "n_workers": args.n_workers,
            "batch_size": algo_params.get("pretrain_batch_size", 256),
            "patience": algo_params.get("pretrain_patience", 5),
            "valid_percentage": algo_params.get("pretrain_validation_percentage", 0.25),
            "lr": algo_params.get("pretrain_clf_learning_rate", 1e-3),
            "metric": algo_params.get("pretrain_perf_metric", None),
            "topk": params["runner_params"]["topk"],
            "seed": params["runner_params"]["seed"],
            "cuda_idx": args.cuda_idx,
            "shared_data_socket": args.shared_data_socket,
        }
    )
    # pretrain
    pretrain(gym_env_id, pretrain_args, params)

    # get the saved model and initialize the agent with it
    filename = os.path.join(pretrain_args.output, BEST_PRETRAIN_MODEL_NAME)
    assert os.path.exists(filename)
    saved_snapshot = torch.load(filename)
    assert saved_snapshot.get("agent_state_dict")
    agent.initial_model_state_dict = saved_snapshot["agent_state_dict"]


def batch_eval_model_on_the_fly(params, args, run_ID, last):
    """Method for evaluating the agent on the fly against a gym environment.

    Parameters
    ----------
    params: dict
        The parameters as provided in the configuration file.
    args: dict
        The arguments as provided in the command line.
    run_ID: int
        the rlpyt run ID

    Return
    ------
    None

    """
    if "end_training_eval_data_fp" not in args or args.end_training_eval_data_fp is None:
        return
    if "suffix" not in args:
        args.suffix = ""
    if "deterministic" not in args:
        args.deterministic = False
    params = copy.deepcopy(params)
    model_dir = os.path.abspath(os.path.join(args.output, f"run_{run_ID}"))
    model_path = model_dir + f"/{LAST_SNAPSHOT_NAME if last else BEST_MODEL_NAME}"
    output_dir = model_dir + f"/{'last_eval' if last else 'best_eval'}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    env_args = AttrDict(
        {
            "no_initial_evidence": False,
            "no_differential": False,
            "train": False,
            "interaction_length": params["simulator_params"]["max_turns"],
            "include_turns_in_state": params["simulator_params"]["include_turns_in_state"],
            "evi_meta_path": params["simulator_params"]["symptom_filepath"],
            "patho_meta_path": params["simulator_params"]["condition_filepath"],
        }
    )
    env = BatchEvalEnv(env_args, args.end_training_eval_data_fp)
    agent_args = AttrDict(
        {
            "n_envs": 1,
            "model_path": model_path,
            "cuda_idx": args.cuda_idx,
        }
    )
    # instantiate the agent
    agent = create_agent(env, agent_args, params)
    if hasattr(agent, "give_V_min_max"):
        if hasattr(agent.model, "set_V_min_max"):
            agent.model.set_V_min_max(agent.V_min, agent.V_max)
    agent.to_device(args.cuda_idx)
    
    batch_size = 44000
    if "batch_size" in args and args.batch_size is not None:
        batch_size = args.batch_size
        
    batchEvalkwargs = {}
    algo_params = createAttrDict(params.get('algo_params', {}))
    if getattr(algo_params, "reward_shaping_flag", False) and getattr(algo_params, "reward_shaping_coef", 1.0) != 0.0:
        batchEvalkwargs['explorationTemporalWeight'] = numpy_sigmoid_scheduler(
            np.arange(env.max_turns + 1), 
            getattr(algo_params, "reward_shaping_kwargs", {}).get('js_alpha', 5),
            env.max_turns,
            0,
            getattr(algo_params, "reward_shaping_kwargs", {}).get('min_map_val', -10),
            getattr(algo_params, "reward_shaping_kwargs", {}).get('max_map_val', 10),
            is_decreasing=True,
        )
        batchEvalkwargs['weightExploration'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('js_weight')
        batchEvalkwargs['min_exploration_reward'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('js_min')
        batchEvalkwargs['max_exploration_reward'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('js_max')

        batchEvalkwargs['confirmationTemporalWeight'] = numpy_sigmoid_scheduler(
            np.arange(env.max_turns + 1), 
            getattr(algo_params, "reward_shaping_kwargs", {}).get('ce_alpha', 5),
            env.max_turns,
            0,
            getattr(algo_params, "reward_shaping_kwargs", {}).get('min_map_val', -10),
            getattr(algo_params, "reward_shaping_kwargs", {}).get('max_map_val', 10),
        )
        batchEvalkwargs['weightConfirmation'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('ce_weight')
        batchEvalkwargs['min_confirmation_reward'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('ce_min')
        batchEvalkwargs['max_confirmation_reward'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('ce_max')

        batchEvalkwargs['weightSeverity'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('sev_out_weight')
        batchEvalkwargs['min_severity_reward'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('sev_out_min')
        batchEvalkwargs['max_severity_reward'] = getattr(algo_params, "reward_shaping_kwargs", {}).get('bounds_dict', {}).get('sev_out_max')

    if getattr(algo_params, "clf_reward_flag", False) and getattr(algo_params, "clf_reward_coef", 1.0) != 0.0:
        batchEvalkwargs['weightClassification'] = getattr(algo_params, "clf_reward_coef", 1.0)
        batchEvalkwargs['min_classification_reward'] = None
        batchEvalkwargs['max_classification_reward'] = None
        batchEvalkwargs['weightSevIn'] = getattr(algo_params, "clf_reward_kwargs", {}).get('sev_in_weight')

    batchEvalkwargs['discount'] = getattr(algo_params, "discount", 1)
    
    
    agent.eval_mode(1)
    result = evaluate(
        env,
        agent,
        params["simulator_params"]["max_turns"],
        seed=params["runner_params"]["seed"],
        compute_metrics_flag=True,
        batch_size=batch_size,
        deterministic=args.deterministic,
        output_fp=os.path.join(output_dir, f"BatchMetrics{args.suffix}.json"),
        action_fp=os.path.join(output_dir, f"BatchActions{args.suffix}.csv"),
        diff_fp=os.path.join(output_dir, f"BatchDifferential{args.suffix}.csv"),
        **batchEvalkwargs
    )


def eval_model_on_the_fly(params, args, run_ID, last):
    """Method for evaluating the agent on the fly against a gym environment.

    Parameters
    ----------
    params: dict
        The parameters as provided in the configuration file.
    args: dict
        The arguments as provided in the command line.
    run_ID: int
        the rlpyt run ID

    Return
    ------
    None

    """
    if "end_training_eval_data" not in args or args.end_training_eval_data is None:
        return
    if "suffix" not in args:
        args.suffix = ""
    params = copy.deepcopy(params)
    model_dir = os.path.abspath(os.path.join(args.output, f"run_{run_ID}"))
    model_path = model_dir + f"/{LAST_SNAPSHOT_NAME if last else BEST_MODEL_NAME}"
    output_dir = model_dir + f"/{'last_eval' if last else 'best_eval'}"
    eval_args = AttrDict(
        {
            "data": args.end_training_eval_data,
            "model_path": model_path,
            "output": output_dir,
            "datetime_suffix": False,
            "no_replace_if_present": args.no_replace_if_present,
            "max_trajectories": None,
            "seed": params["runner_params"]["seed"],
            "n_envs": 1,
            "sharing_prefix": args.end_training_eval_data,
            "cuda_idx": args.cuda_idx,
            "shared_data_socket": args.shared_data_socket,
            "max_generation": 100,
            "topk": params["runner_params"]["topk"],
            "eval_coeffs": params["runner_params"]["eval_coeffs"],
            "sample_indices_flag": True,
            "compute_metric_flag": False,
        }
    )
    # use the differential for evaluation whenever available
    params["simulator_params"]["use_differential_diagnosis"] = True
    # use the provided first symptom for evaluation whenever available
    params["simulator_params"]["use_initial_symptom_flag"] = True

    # get the trajectory file
    filename = os.path.join(eval_args.output, DATA_FILE_NAME)
    filename2 = os.path.join(eval_args.output, METRIC_FILE_NAME)
    
    metric_dir = eval_args.output
    metric_args = AttrDict(
        {
            "patients_fp": filename,
            "output_fp": metric_dir,
            "model_name": f"Metrics{args.suffix}",
            "symptoms_fp": params["simulator_params"]["symptom_filepath"],
            "conditions_fp": params["simulator_params"]["condition_filepath"],
            "weight_fp": None,
            "pool_size": max(1, args.n_workers * 2),
            "min_proba": 0.01,
            "severity_threshold": 3,
        }
    )
    metric_file = os.path.join(metric_args.output_fp, f"{metric_args.model_name}.json")
    if os.path.exists(metric_file):
        return

    # eval
    if not os.path.exists(filename) or not os.path.exists(filename2):
        eval_run(eval_args, params)

    assert os.path.exists(filename)
    assert os.path.exists(filename2)

    # metric computation
    metric_computation(metric_args)

    # get the metric file
    assert os.path.exists(metric_file)

    # load it and log it
    with open(metric_file) as fp:
        metric_data = json.load(fp)
    if not ("no_log" in args and args.no_log):
        log_metrics(metric_data, f"{'eval_last_' if last else 'eval_best_'}")

    # delete trajectory file
    if not ("no_filedeletion" in args and args.no_filedeletion):
        os.remove(filename)
        os.remove(filename2)


def build_and_train(gym_env_id, args, run_ID=0, params=None):
    """Method for building and training an agent against a gym environment.

    Parameters
    ----------
    gym_env_id: str
        Id of the gym environment.
    args: dict
        The arguments as provided in the command line.
    run_ID: int
        The ID of the run. Default: 0
    params: dict
        The parameters as provided in the configuration file.
        Default: None.

    Return
    ------
    None

    """

    log_dir = args.output

    exp_name = mlflow.get_experiment(mlflow.active_run().info.experiment_id).name

    if params is None:
        params = {}

    sampler_params = load_component_metadata(
        "sampler_params", params, prefix="sampler_"
    )
    runner_params = params.get("runner_params", {})
    if runner_params.get("seed", None) is None:
        runner_params["seed"] = make_seed()
        params["runner_params"] = runner_params

    # initialize the random generator
    initialize_seed(runner_params["seed"])

    algo_params = load_component_metadata("algo_params", params, prefix="algo_")
    agent_params = load_component_metadata("agent_params", params, prefix="agent_")
    mandatory_runner_params = ["topk", "eval_coeffs", "traj_auxiliary_reward_flag"]
    params["runner_params"]['eval_batch_size'] = runner_params.get(
        'eval_batch_size', 33112
    )
    runner_params = load_component_metadata(
        "runner_params", params, mandatory_runner_params, prefix="runner_"
    )
    eval_metrics = params.get("eval_metrics", None)  # list of metrics

    replay_buffer_cls = algo_params.get("ReplayBufferCls", None)
    if not (replay_buffer_cls is None):
        replay_buffer_cls = ReplayBufferFactory().get_replay_buffer_class(
            replay_buffer_cls
        )
        algo_params["ReplayBufferCls"] = replay_buffer_cls

    collector_cls = sampler_params.get("CollectorCls", None)
    eval_collector_cls = sampler_params.get("eval_CollectorCls", None)
    if not ((collector_cls is None) and (eval_collector_cls is None)):
        collector_factory = CollectorFactory()
        if collector_cls is not None:
            collector_cls = collector_factory.get_collector_class(collector_cls)
            sampler_params["CollectorCls"] = collector_cls
        if eval_collector_cls is not None:
            eval_collector_cls = collector_factory.get_collector_class(
                eval_collector_cls
            )
            sampler_params["eval_CollectorCls"] = eval_collector_cls

    sim_params, rew_params = load_environment_metadata(params)
    sim_params["reward_config"] = RewardConfig(**rew_params)
    sim_params["use_initial_symptom_flag"] = (
        False
        if "use_initial_symptom_flag" not in sim_params
        else sim_params["use_initial_symptom_flag"]
    )
    eval_sim_params = copy.deepcopy(sim_params)
    eval_sim_params["use_initial_symptom_flag"] = True

    sampler = SamplerFactory().create(
        params["sampler"],
        EnvCls=gym_make,
        TrajInfoCls=SimPaTrajInfo,
        env_kwargs=dict(
            id=gym_env_id,
            patient_filepath=args.data,
            shared_data_socket=args.shared_data_socket,
            **sim_params,
        ),
        eval_env_kwargs=dict(
            id=gym_env_id,
            patient_filepath=args.eval_data,
            shared_data_socket=args.shared_data_socket,
            **eval_sim_params,
        ),
        batch_T=sampler_params.pop("batch_T", 1),  # One time-step per sampler iteration
        batch_B=params["n_envs"],
        max_decorrelation_steps=params["max_decorrelation_steps"],
        eval_n_envs=params["eval_n_envs"],
        eval_max_steps=params["eval_max_steps"],
        eval_max_trajectories=params["eval_max_trajectories"],
        **sampler_params,
    )

    snapshot = reload_snapshot(
        os.path.abspath(os.path.join(log_dir, f"run_{run_ID}")),
        LAST_SNAPSHOT_NAME,
        args.start_from_scratch,
    )

    # load the pretrained filename
    pretrained_filename = getattr(args, "pretrained_model", None)
    snapshot = load_pretrained_snapshot(snapshot, pretrained_filename)

    modelCls, model_params = load_model_metadata(params)
    optimCls, optim_params = load_optimizer_metadata(params)

    # eventually get the symp_to_obs_mapping
    env_0 = gym_make(
        id=gym_env_id,
        patient_filepath=args.data,
        shared_data_socket=args.shared_data_socket,
        **sim_params,
    )
    symp_2_obs_map = None
    if hasattr(env_0, "get_symptom_to_observation_mapping"):
        symp_2_obs_map = env_0.get_symptom_to_observation_mapping()
    model_params["symptom_2_observation_map"] = symp_2_obs_map
    patho_severity = None
    if hasattr(env_0, "get_pathology_severity"):
        patho_severity = env_0.get_pathology_severity()
    model_params["patho_severity"] = patho_severity
    model_params["hierarchical_map"] = (
        env_0.get_hierarchical_symptom_dependency()
        if hasattr(env_0, "get_hierarchical_symptom_dependency")
        else None
    )
    model_params["symptom_2_symptom_association"] = (
        env_0.get_evidence_2_evidence_association()
        if hasattr(env_0, "get_evidence_2_evidence_association")
        else None
    )
    model_params["symptom_2_patho_association"] = (
        env_0.get_evidence_2_pathology_association()
        if hasattr(env_0, "get_evidence_2_pathology_association")
        else None
    )
    model_params["symp_default_values"] = (
        env_0.get_evidence_default_value_in_obs()
        if hasattr(env_0, "get_evidence_default_value_in_obs")
        else None
    )
    # check consistency of `include_turns_in_state` between simulator and model
    if hasattr(env_0, "include_turns_in_state"):
        include_turns_in_state = env_0.include_turns_in_state
        assert (not ("include_turns_in_state" in model_params)) or (
            include_turns_in_state == model_params["include_turns_in_state"]
        )
    # check consistency between `n_atoms` in agent and model if any
    model_params["n_atoms"] = agent_params.get("n_atoms")
    assert (not ("n_atoms" in agent_params)) or (
        model_params.get("n_atoms") == agent_params["n_atoms"]
    )
    # check consistency with `use_stop_action` in model and exit_loss in algo
    assert model_params.get("use_stop_action", True) or (
        not model_params.get("use_stop_action", True)
        and (
            algo_params.get("clf_reward_kwargs", {}).get("exit_loss_coeff", 1.0) == 0.0
        )
    )
    batch_env_args = AttrDict(
        {
            "no_initial_evidence": False,
            "no_differential": not params["simulator_params"]["use_differential_diagnosis"],
            "train": False,
            "interaction_length": params["simulator_params"]["max_turns"],
            "include_turns_in_state": params["simulator_params"]["include_turns_in_state"],
            "evi_meta_path": params["simulator_params"]["symptom_filepath"],
            "patho_meta_path": params["simulator_params"]["condition_filepath"],
        }
    )
    batch_env = (
        BatchEvalEnv(batch_env_args, args.end_training_eval_data_fp) 
        if ('end_training_eval_data_fp' in args and args.end_training_eval_data_fp is not None)
        else None
    )
    runner_params['batch_env'] = batch_env

    algo = AlgoFactory().create(
        params["algo"],
        OptimCls=optimCls,
        optim_kwargs=optim_params,
        initial_optim_state_dict=snapshot.get("optimizer_state_dict", None),
        **algo_params,
    )
    agent_state_dict = snapshot.get("agent_state_dict", None)
    agent = AgentFactory().create(
        params["agent"],
        algo.__class__,
        sampler.__class__,
        ModelCls=modelCls,
        model_kwargs=model_params,
        initial_model_state_dict=agent_state_dict,
        **agent_params,
    )
    runnerFactory = RunnerFactory()
    affinity = runnerFactory.get_affinity(
        params["runner"],
        sampler,
        args.cuda_idx,
        args.n_workers,
        args.n_gpus,
        args.cpu_list,
        args.num_torch_threads,
    )
    resume_info = snapshot.get("resume_info", {})
    runner = runnerFactory.create(
        params["runner"],
        resume_info=resume_info,
        metrics=eval_metrics,
        perf_window_size=params.get("perf_window_size", 10),
        patience=params.get("patience"),
        perf_metric=params.get("perf_metric", "Reward"),
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=max(params["n_steps"] - snapshot.get("cum_steps", 0), 0),
        log_interval_steps=params["log_interval_steps"],
        affinity=affinity,
        **runner_params,
    )
    params["env_id"] = gym_env_id
    with logger_context(
        log_dir,
        run_ID,
        exp_name,
        params,
        snapshot_mode="last",
        override_prefix=True,
        use_summary_writer=False,
    ):
        pretrain_on_the_fly(gym_env_id, algo_params, runner.agent, params, args)
        train(runner)
