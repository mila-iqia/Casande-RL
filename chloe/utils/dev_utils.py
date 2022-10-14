import os
import random

import git
import mlflow
import numpy as np
import torch
from pip._internal.operations import freeze


def get_current_git_commit_hash():
    """Method for getting the current commit hash of the repository.

    Method for getting the git hash of the current commit/version of the code.

    Parameters
    ----------

    Returns
    -------
    result: str
        the hex string corresponing to the hash.

    """
    if False:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    else:
        sha = "Anonymous"
    return sha


def get_repo_state():
    """Returns state of the git repository which contains.
    It contains:
    - The active branch name
    - The hash of the active commit
    - The hash of the last pushed commit
    - If there are some uncommited changes
    Returns
    -------
    state: dict
        Dictionary of Git status info.
    """
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch

    state = {"Git - Active branch name": branch.name}
    state["Git - Active commit"] = branch.commit.hexsha
    state["Git - Uncommited Changes"] = repo.is_dirty()

    remote_name = f"origin/{branch.name}"
    if remote_name in repo.refs:
        state["Git - Last pushed commit"] = repo.refs[remote_name].commit.hexsha
    else:
        state["Git - Last pushed commit"] = "UNPUSHED BRANCH"
    return state


def log_git_diff(experiment_folder, mlflow_dir):
    """Log the git diff to mlflow active run as an artifact and to current exp_folder.
    Parameters
    ----------
    experiment_folder: str
        Path to the current experiment folder.
    """
    repo = git.Repo(search_parent_directories=True)
    diff = repo.git.diff()

    diff_path = os.path.join(experiment_folder, "git_diff.txt")
    with open(diff_path, "w") as f:
        f.write(diff)
    mlflow.log_artifact(diff_path, mlflow_dir)


def check_git_consistency(repo_state, active_run_params):
    """Check consistency of the git repo with the current mlflow params.
    Parameters
    ----------
    repo_state: dict
        repo state.
    active_run_params: dict
        mlflow run params
    """
    for g in active_run_params:
        if g.startswith("Git -") and (g in repo_state):
            if str(repo_state[g]) != str(active_run_params[g]):
                err_message = "Cannot resume experiment from different code.\n"
                err_message += (
                    f"{g} does not match. {repo_state[g]} => {active_run_params[g]}"
                )
                raise Exception(err_message)
            elif g == "Git - Uncommited Changes" and repo_state[g] is True:
                err_message = (
                    "Cannot resume experiment that was started on uncommited code."
                )
                raise Exception(err_message)


def get_dev_dependencies():
    """Get the package dependencies of the repository.

    Method for getting the package dependencies installed in the
    development environment.

    Parameters
    ----------

    Returns
    -------
    result: list
        the list of installed packages.

    """
    dependencies = list(freeze.freeze())
    return dependencies


def initialize_seed(seed, cuda={}, log_flag=False):
    """Method for seeding the random generators.
    Parameters
    ----------
    seed: int
        the seed to be used.
    cuda : dict
        determininistic: bool
            To what should we put torch.backends.cudnn.deterministic.
        backends: bool
            To what should we put torch.backends.cudnn.benchmark.
    Returns
    -------
    None
    """
    if log_flag:
        mlflow.log_param("seed", seed)
    torch.backends.cudnn.deterministic = cuda.get("deterministic", False)
    torch.backends.cudnn.benchmark = cuda.get("benchmark", True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
