import os

import mlflow
import torch
import torch.nn.functional as F
from rlpyt.agents.dqn.r2d1_agent import R2d1AgentBase
from rlpyt.envs.gym import make as gym_make
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.utils.seed import make_seed, set_envs_seeds
from torch.utils.data import DataLoader

from chloe.pretraining.dataset import SimPaDataset
from chloe.simulator.simulator import RewardConfig
from chloe.utils.agent_utils import AgentFactory
from chloe.utils.dev_utils import get_current_git_commit_hash, initialize_seed
from chloe.utils.dist_metric import dist_accuracy, dist_ncg, dist_ndcg
from chloe.utils.eval_utils import MetricFactory
from chloe.utils.model_utils import (
    load_component_metadata,
    load_environment_metadata,
    load_model_metadata,
    load_optimizer_metadata,
)
from chloe.utils.tensor_utils import soft_cross_entropy

BEST_PRETRAIN_MODEL_NAME = "pretrain_best_model_params.pkl"


def create_environments(gym_env_id, args, sim_params):
    """Method for creating the environments for pretraining.

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
    env_list = [gym_make(**env_kwargs)]

    is_eval_off = args.eval_data is None or args.eval_data == args.data
    if not is_eval_off:
        env_kwargs = dict(
            id=gym_env_id,
            patient_filepath=args.eval_data,
            shared_data_socket=args.shared_data_socket,
            **sim_params,
        )
        env_list.append(gym_make(**env_kwargs))
    set_envs_seeds(env_list, args.seed)

    if is_eval_off:
        env_list.append(None)

    return env_list


def create_datasets(env_train, env_valid, args):
    """Method for creating the training and valid datasets.

    Parameters
    ----------
    env_train: object
        the enviroment to be used for training.
    env_valid: object
        the enviroment to be used for validation.
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    train_ds: dataset
        the training dataset.
    valid_ds: dataset
        the valid dataset.

    """
    ds_train = SimPaDataset(env_train, not args.no_data_corrupt)
    ds_valid = None
    if env_valid is not None:
        ds_valid = SimPaDataset(env_valid, not args.no_data_corrupt)
    elif not ((args.valid_percentage is None) or (args.valid_percentage == 0)):
        valid_size = int(len(ds_train) * args.valid_percentage)
        train_size = len(ds_train) - valid_size
        if valid_size > 0:
            ds_train, ds_valid = torch.utils.data.random_split(
                ds_train, [train_size, valid_size],
            )
    return ds_train, ds_valid


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
    # get log flag
    log_flag = getattr(args, "log_params", False)

    rank = 0
    n_envs = 1
    global_B = n_envs
    env_ranks = list(range(rank * n_envs, (rank + 1) * n_envs))

    agent_params = load_component_metadata(
        "agent_params", params, prefix="agent_", log_hp_flag=log_flag,
    )
    modelCls, model_params = load_model_metadata(params, log_hp_flag=log_flag)

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
        env.get_hierarchical_symptom_dependency()
        if hasattr(env, "get_hierarchical_symptom_dependency")
        else None
    )
    model_params["symptom_2_symptom_association"] = (
        env.get_evidence_2_evidence_association()
        if hasattr(env, "get_evidence_2_evidence_association")
        else None
    )
    model_params["symptom_2_patho_association"] = (
        env.get_evidence_2_pathology_association()
        if hasattr(env, "get_evidence_2_pathology_association")
        else None
    )
    model_params["symp_default_values"] = (
        env.get_evidence_default_value_in_obs()
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
    assert (not ("n_atoms" in agent_params)) or (
        model_params.get("n_atoms") == agent_params["n_atoms"]
    )

    agent_state_dict = None
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
        assert (v_min is not None) and (v_max is not None)
        agent.give_V_min_max(v_min, v_max)
    return agent


def train_epoch(epoch, agent, optimizer, train_dl, metric_factory, args):
    """Train Epoch for the pretraining process.

    Parameters
    ----------
    epoch: int
        the epoch number.
    agent: agent
        the agent whose classifier should be pretrained.
    optimizer: optimizer
        the optimizer to be used.
    train_dl: dataloader
        the training dataloader.
    metric_factory: object
        the metric factory.
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    avg_loss: float
        the obtained average loss value.
    avg_metric: float
        the obtained average performance metric value.

    """
    avg_loss = 0.0
    avg_metric = None
    num_elts = 0
    act, rew = None, None
    allowed_dist_metrics = dict(
        dist_accuracy=dist_accuracy, dist_ndcg=dist_ndcg, dist_ncg=dist_ncg
    )
    for _, (x, y) in enumerate(train_dl):
        num_elts += x.size(0)
        target, diff_indices, diff_probas = y
        diff_indices = None if diff_indices.nelement() == 0 else diff_indices
        diff_probas = None if diff_probas.nelement() == 0 else diff_probas

        act = (
            torch.tensor([0] * x.size(0)).long()
            if act is None or x.size(0) != act.size(0)
            else act
        )
        rew = (
            torch.tensor([0.0] * x.size(0))
            if rew is None or x.size(0) != rew.size(0)
            else rew
        )

        data, target, diff_indices, diff_probas, act, rew = buffer_to(
            (x, target, diff_indices, diff_probas, act, rew), device=agent.device,
        )
        optimizer.zero_grad()
        init_state = [None] if isinstance(agent, R2d1AgentBase) else []
        arguments = [data, act, rew] + init_state
        out = agent.classify(*arguments)
        logits = out[0] if isinstance(out, (list, tuple)) else out

        # permute axis
        logits = logits.transpose(1, -1)
        target = target.long()
        diff_indices = None if diff_indices is None else diff_indices.transpose(1, -1)
        diff_probas = None if diff_probas is None else diff_probas.transpose(1, -1)
        is_soft = (diff_indices is not None) and (diff_probas is not None)

        # compute loss
        loss = (
            soft_cross_entropy(
                logits, diff_indices, diff_probas, weight=None, reduction="mean",
            )
            if is_soft
            else F.cross_entropy(logits, target, weight=None, reduction="mean")
        )
        loss.backward()

        # optimize
        optimizer.step()
        avg_loss += loss.item() * x.size(0)
        if args.metric is not None:
            avg_metric = 0.0 if avg_metric is None else avg_metric
            top_flag = args.metric.startswith("top-")
            dist_flag = args.metric.startswith("dist_")
            assert (not dist_flag) or (args.metric in allowed_dist_metrics)
            pred_info = (
                torch.argmax(logits, dim=1).view(-1)
                if not top_flag and not dist_flag
                else logits.detach().transpose(1, -1).reshape(-1, logits.size(1))
            )
            pred_info = pred_info.cpu().numpy()
            target = target.cpu().view(-1).numpy()
            tmp_size = None if diff_indices is None else diff_indices.size(1)
            diff_indices = (
                diff_indices.cpu().transpose(1, -1).reshape(-1, tmp_size).numpy()
                if diff_indices is not None
                else None
            )
            tmp_size = None if diff_probas is None else diff_probas.size(1)
            diff_probas = (
                diff_probas.cpu().transpose(1, -1).reshape(-1, tmp_size).numpy()
                if diff_probas is not None
                else None
            )
            perf = (
                metric_factory.evaluate(args.metric, target, pred_info)
                if not dist_flag
                else allowed_dist_metrics[args.metric](
                    pred_info, target, diff_indices, diff_probas, args.topk, True
                )
            )
            avg_metric += perf * x.size(0)
    avg_loss /= max(1, num_elts)
    if avg_metric is not None:
        avg_metric /= max(1, num_elts)
    else:
        avg_metric = -avg_loss
    return avg_loss, avg_metric


def eval_epoch(epoch, agent, eval_dl, metric_factory, args):
    """Train Epoch for the pretraining process.

    Parameters
    ----------
    epoch: int
        the epoch number.
    agent: agent
        the agent whose classifier should be pretrained.
    eval_dl: dataloader
        the validate dataloader.
    metric_factory: object
        the metric factory.
    args: dict
        The arguments as provided in the command line.

    Return
    ------
    avg_loss: float
        the obtained average loss value.
    avg_metric: float
        the obtained average performance metric value.

    """
    avg_loss = 0.0
    avg_metric = None
    num_elts = 0
    allowed_dist_metrics = dict(
        dist_accuracy=dist_accuracy, dist_ndcg=dist_ndcg, dist_ncg=dist_ncg
    )
    with torch.no_grad():
        act, rew = None, None
        for _, (x, y) in enumerate(eval_dl):
            num_elts += x.size(0)
            target, diff_indices, diff_probas = y
            diff_indices = None if diff_indices.nelement() == 0 else diff_indices
            diff_probas = None if diff_probas.nelement() == 0 else diff_probas

            act = (
                torch.tensor([0] * x.size(0)).long()
                if act is None or x.size(0) != act.size(0)
                else act
            )
            rew = (
                torch.tensor([0.0] * x.size(0))
                if rew is None or x.size(0) != rew.size(0)
                else rew
            )

            data, target, diff_indices, diff_probas, act, rew = buffer_to(
                (x, target, diff_indices, diff_probas, act, rew), device=agent.device,
            )
            init_state = [None] if isinstance(agent, R2d1AgentBase) else []
            arguments = [data, act, rew] + init_state
            out = agent.classify(*arguments)
            logits = out[0] if isinstance(out, (list, tuple)) else out

            # permute axis
            logits = logits.transpose(1, -1)
            target = target.long()
            diff_indices = (
                None if diff_indices is None else diff_indices.transpose(1, -1)
            )
            diff_probas = None if diff_probas is None else diff_probas.transpose(1, -1)
            is_soft = (diff_indices is not None) and (diff_probas is not None)

            # compute loss
            loss = (
                soft_cross_entropy(
                    logits, diff_indices, diff_probas, weight=None, reduction="mean",
                )
                if is_soft
                else F.cross_entropy(logits, target, weight=None, reduction="mean")
            )
            avg_loss += loss.item() * x.size(0)
            if args.metric is not None:
                avg_metric = 0.0 if avg_metric is None else avg_metric
                top_flag = args.metric.startswith("top-")
                dist_flag = args.metric.startswith("dist_")
                assert (not dist_flag) or (args.metric in allowed_dist_metrics)
                pred_info = (
                    torch.argmax(logits, dim=1).view(-1)
                    if not top_flag and not dist_flag
                    else logits.detach().transpose(1, -1).reshape(-1, logits.size(1))
                )
                pred_info = pred_info.cpu().numpy()
                target = target.cpu().view(-1).numpy()
                tmp_size = None if diff_indices is None else diff_indices.size(1)
                diff_indices = (
                    diff_indices.cpu().transpose(1, -1).reshape(-1, tmp_size).numpy()
                    if diff_indices is not None
                    else None
                )
                tmp_size = None if diff_probas is None else diff_probas.size(1)
                diff_probas = (
                    diff_probas.cpu().transpose(1, -1).reshape(-1, tmp_size).numpy()
                    if diff_probas is not None
                    else None
                )
                perf = (
                    metric_factory.evaluate(args.metric, target, pred_info)
                    if not dist_flag
                    else allowed_dist_metrics[args.metric](
                        pred_info, target, diff_indices, diff_probas, args.topk, True
                    )
                )
                avg_metric += perf * x.size(0)
    avg_loss /= max(1, num_elts)
    if avg_metric is not None:
        avg_metric /= max(1, num_elts)
    else:
        avg_metric = -avg_loss
    return avg_loss, avg_metric


def pretrain_classifier(agent, optimizer, train_ds, eval_ds, args):
    """Pretrain the agent classifier.

    Parameters
    ----------
    agent: agent
        the agent whose classifier should be pretrained.
    optimizer: optimizer
        the optimizer to be used.
    train_ds: dataset
        the training dataset.
    valid_ds: dataset
        the valid dataset.
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    None

    """
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    eval_dl = (
        None
        if eval_ds is None
        else DataLoader(eval_ds, batch_size=args.batch_size, num_workers=args.n_workers)
    )
    metric_factory = MetricFactory()
    best_performance = None
    remaining_patience = args.patience
    for i in range(args.num_epochs):
        agent.train_mode(0)
        avg_loss, avg_perf = train_epoch(
            i, agent, optimizer, train_dl, metric_factory, args
        )
        mlflow.log_metric("pretraining-train-loss", avg_loss, i)
        mlflow.log_metric("pretraining-train-perf", avg_perf, i)
        if eval_dl is not None:
            agent.eval_mode(0)
            avg_val_loss, avg_val_perf = eval_epoch(
                i, agent, eval_dl, metric_factory, args
            )
            mlflow.log_metric("pretraining-val-loss", avg_val_loss, i)
            mlflow.log_metric("pretraining-val-perf", avg_val_perf, i)
            if (best_performance is None) or (avg_val_perf > best_performance):
                best_performance = avg_val_perf
                params = dict(
                    itr=i,
                    agent_state_dict=agent.state_dict(),
                    pretrain_optimizer_state_dict=optimizer.state_dict(),
                )
                file_name = os.path.join(args.output, BEST_PRETRAIN_MODEL_NAME)
                torch.save(params, file_name)
                remaining_patience = args.patience
            else:
                remaining_patience = (
                    remaining_patience - 1
                    if args.patience is not None
                    else args.patience
                )
        logger.log(
            f"Pretraining Epoch {i}: train_loss: {avg_loss} train_perf: {avg_perf} "
            f"valid_loss: {avg_val_loss} valid_perf: {avg_val_perf}."
        )
        if (remaining_patience is not None) and (remaining_patience < 0):
            break
    if hasattr(agent, "update_target"):
        agent.update_target()
    if best_performance is None:
        params = dict(
            itr=i,
            agent_state_dict=agent.state_dict(),
            pretrain_optimizer_state_dict=optimizer.state_dict(),
        )
        file_name = os.path.join(args.output, BEST_PRETRAIN_MODEL_NAME)
        torch.save(params, file_name)
    else:
        func_metric = "loss" if not args.metric else args.metric
        logger.log(
            f"End Pretraining Epoch with best {func_metric} performance of: "
            f"{best_performance}."
        )


def pretrain(gym_env_id, args, params=None):
    """Method for pretraining an agent against a gym environment.

    Parameters
    ----------
    gym_env_id: str
        Id of the gym environment.
    args: dict
        The arguments as provided in the command line.
    params: dict
        The parameters as provided in the configuration file.
        Default: None

    Return
    ------
    None

    """

    if params is None:
        params = {}

    archi = params.get("architecture", "").lower()
    if ("mixed" not in archi) and ("mixreb" not in archi):
        logger.log("Pretraining is only valid for MixedDQN/MixReb like agents.")
        return

    assert (args.valid_percentage is None) or (
        (args.valid_percentage >= 0) and (args.valid_percentage < 1)
    )
    assert args.batch_size > 0
    assert args.patience >= 0

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.seed is None:
        args.seed = make_seed()

    logger.log(f"Pretraining: Using seed [{args.seed}] for this process.")

    args.metric = None if args.metric is None else args.metric.lower()

    # initialize the random generator
    initialize_seed(args.seed)

    # get log flag
    log_flag = getattr(args, "log_params", False)

    # instantiate the envs
    sim_params, rew_params = load_environment_metadata(params, log_hp_flag=log_flag)
    sim_params["reward_config"] = RewardConfig(**rew_params)

    env_train, env_valid = create_environments(gym_env_id, args, sim_params)

    # instantiate the agent
    agent = create_agent(env_train, args, params)
    if args.cuda_idx is not None:
        agent.to_device(args.cuda_idx)

    # get the optimizer info
    optimCls, optim_params = load_optimizer_metadata(params, log_hp_flag=log_flag)

    # create the datasets
    ds_train, ds_valid = create_datasets(env_train, env_valid, args)

    # define the optimizer
    optimizer = optimCls(agent.parameters(), lr=args.lr, **optim_params,)

    # pretrain
    pretrain_classifier(agent, optimizer, ds_train, ds_valid, args)

    # add git hash
    git_hash = get_current_git_commit_hash()
    if log_flag:
        mlflow.log_param("git_hash", git_hash)
