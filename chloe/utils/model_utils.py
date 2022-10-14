from rlpyt.utils.logging import logger
from torch import optim

from chloe.models.dqn_baseline_models import (
    BaselineCatDQNModel,
    BaselineDQNModel,
    BaselineR2D1DQNModel,
)
from chloe.models.dqn_proba_models import (
    MixRebCatDQNModel,
    MixRebDQNModel,
    MixRebR2D1DQNModel,
    MixedCatDQNModel,
    MixedDQNModel,
    MixedR2D1DQNModel,
)
from chloe.models.dqn_rebuild_models import (
    RebuildCatDQNModel,
    RebuildDQNModel,
    RebuildR2D1DQNModel,
)
from chloe.models.mlp_model import MyMlpCatDQNModel, MyMlpDQNModel, MyMlpPGModel
from chloe.utils.logging_utils import check_and_log_hp


def load_model_metadata(hyper_params, mandatory_params=None, log_hp_flag=True):
    """Method for loading the metadata of the model to use.

    The currently allowed models are:
        - baseline_dqn_model
        - baseline_cat_dqn_model
        - baseline_r2d1_dqn_model
        - mixed_dqn_model
        - mixed_cat_dqn_model
        - mixed_r2d1_dqn_model
        - rebuild_dqn_model
        - rebuild_cat_dqn_model
        - rebuild_r2d1_dqn_model
        - mixreb_dqn_model
        - mixreb_cat_dqn_model
        - mixreb_r2d1_dqn_model
        - my_mlp_dqn_model
        - my_mlp_cat_dqn_model
        - my_mlp_pg_model

    Parameters
    ----------
    hyper_params: dict
        dictionary containing hyper-params data.
    mandatory_params: list, None
        list of expected mandatory params for the model to be loaded.
        Default: None
    log_hp_flag: boolean
        whether or not to log the hyper parameter in mlflow.
        Default: True

    Returns
    -------
    model_class: class
        the class of the optimizer.
    model_params: dict
        the parameters to be used when instanciating the model.

    """
    mandatory_params = [] if mandatory_params is None else mandatory_params
    mlp_mandatory_params = [
        "input_size",
        "hidden_sizes",
        "output_size",
        "dueling",
        "dueling_fc_sizes",
    ]
    baseline_mandatory_params = [
        "input_size",
        "hidden_sizes",
        "output_size",
        "num_symptoms",
        "dueling",
        "dueling_fc_sizes",
        "mask_inquired_symptoms",
        "not_inquired_value",
        "include_turns_in_state",
        "use_turn_just_for_masking",
        "min_turns_ratio_for_decision",
    ]
    architecture = hyper_params["architecture"]
    valid_model_dict = {
        "my_mlp_dqn_model": MyMlpDQNModel,
        "my_mlp_cat_dqn_model": MyMlpCatDQNModel,
        "my_mlp_pg_model": MyMlpPGModel,
        "baseline_dqn_model": BaselineDQNModel,
        "baseline_cat_dqn_model": BaselineCatDQNModel,
        "baseline_r2d1_dqn_model": BaselineR2D1DQNModel,
        "mixed_dqn_model": MixedDQNModel,
        "mixed_cat_dqn_model": MixedCatDQNModel,
        "mixed_r2d1_dqn_model": MixedR2D1DQNModel,
        "rebuild_dqn_model": RebuildDQNModel,
        "rebuild_cat_dqn_model": RebuildCatDQNModel,
        "rebuild_r2d1_dqn_model": RebuildR2D1DQNModel,
        "mixreb_dqn_model": MixRebDQNModel,
        "mixreb_cat_dqn_model": MixRebCatDQNModel,
        "mixreb_r2d1_dqn_model": MixRebR2D1DQNModel,
    }
    mandatory_model_params_dict = {
        "my_mlp_dqn_model": mlp_mandatory_params,
        "my_mlp_cat_dqn_model": mlp_mandatory_params + ["n_atoms"],
        "my_mlp_pg_model": ["input_size", "hidden_sizes", "output_size"],
        "baseline_dqn_model": baseline_mandatory_params,
        "baseline_cat_dqn_model": baseline_mandatory_params + ["n_atoms"],
        "baseline_r2d1_dqn_model": baseline_mandatory_params + ["lstm_size"],
        "mixed_dqn_model": baseline_mandatory_params + ["pi_hidden_sizes"],
        "mixed_cat_dqn_model": baseline_mandatory_params
        + ["pi_hidden_sizes", "n_atoms"],
        "mixed_r2d1_dqn_model": baseline_mandatory_params
        + ["pi_hidden_sizes", "lstm_size"],
        "rebuild_dqn_model": baseline_mandatory_params
        + ["reb_size", "reb_hidden_sizes"],
        "rebuild_cat_dqn_model": baseline_mandatory_params
        + ["reb_size", "reb_hidden_sizes", "n_atoms"],
        "rebuild_r2d1_dqn_model": baseline_mandatory_params
        + ["reb_size", "reb_hidden_sizes", "lstm_size"],
        "mixreb_dqn_model": baseline_mandatory_params
        + ["reb_size", "reb_hidden_sizes", "pi_hidden_sizes"],
        "mixreb_cat_dqn_model": baseline_mandatory_params
        + ["reb_size", "reb_hidden_sizes", "pi_hidden_sizes", "n_atoms"],
        "mixreb_r2d1_dqn_model": baseline_mandatory_params
        + ["reb_size", "reb_hidden_sizes", "pi_hidden_sizes", "lstm_size"],
    }
    # __TODO__ fix architecture list
    if architecture.lower() in valid_model_dict:
        model_class = valid_model_dict[architecture.lower()]
        mandatory_params += mandatory_model_params_dict[architecture.lower()]
    else:
        raise ValueError("architecture {} not supported".format(architecture))

    logger.log("selected architecture: {}".format(architecture))

    # get the architeture params
    model_params = hyper_params.get("architecture_params", {})

    # eventually add mandatory params
    keys = list(model_params.keys())
    if not (mandatory_params is None) and (len(mandatory_params) > 0):
        keys = list(set(keys + mandatory_params))

    # check and log params
    if len(keys) > 0:
        check_and_log_hp(keys, model_params, prefix="model_", log_hp_flag=log_hp_flag)

    return model_class, model_params


def load_optimizer_metadata(hyper_params, mandatory_params=None, log_hp_flag=True):
    """Method for loading the metadata of the optimizer to use.

    The currently allowed values are `Adam` and `sgd`.

    Parameters
    ----------
    hyper_params: dict
        dictionary containing hyper-params data.
    mandatory_params: list, None
        list of expected mandatory params for the model to be loaded.
        Default: None
    log_hp_flag: boolean
        whether or not to log the hyper parameter in mlflow.
        Default: True

    Returns
    -------
    optim_class: class
        the class of the optimizer.
    optim_params: dict
        the optim parameters to be used when instanciating it.

    """
    optimizer_name = hyper_params["optimizer"]
    optimizers = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
        # "radam": optim.RAdam,
        # "nadam": optim.NAdam,
        "rmsprop": optim.RMSprop,
    }
    # __TODO__ fix optimizer list
    if optimizer_name.lower() in optimizers:
        optimizer_class = optimizers[optimizer_name.lower()]
    else:
        raise ValueError("optimizer {} not supported".format(optimizer_name))

    logger.log("selected optimizer: {}".format(optimizer_class))

    # get the optimizer params
    optim_params = hyper_params.get("optimizer_params", {})

    # eventually add mandatory params
    keys = list(optim_params.keys())
    if not (mandatory_params is None) and (len(mandatory_params) > 0):
        keys = list(set(keys + mandatory_params))

    # check and log params
    if len(keys) > 0:
        check_and_log_hp(keys, optim_params, prefix="optim_", log_hp_flag=log_hp_flag)
    return optimizer_class, optim_params


def load_environment_metadata(hyper_params, log_hp_flag=True):
    """Method for loading the metadata of the environment to use.

    Parameters
    ----------
    hyper_params: dict
        dictionary containing hyper-params data.
    log_hp_flag: boolean
        whether or not to log the hyper parameter in mlflow.
        Default: True

    Returns
    -------
    sim_params: dict
        the parameter dict for instantiating the environment.
    rew_params: dict
        the parameter dict of the reward function to be used.

    """
    sim_params = load_component_metadata(
        "simulator_params",
        hyper_params,
        mandatory_params=[
            "symptom_filepath",
            "condition_filepath",
            "max_turns",
            "action_type",
            "stop_if_repeated_question",
            "include_turns_in_state",
            "include_race_in_state",
            "include_ethnicity_in_state",
            "is_reward_relevancy_patient_specific",
        ],
        prefix="simulator_",
        log_hp_flag=log_hp_flag,
    )
    rew_params = load_component_metadata(
        "reward_config",
        hyper_params,
        mandatory_params=[
            "reward_on_repeated_action",
            "reward_on_missing_diagnosis",
            "reward_on_correct_diagnosis",
            "reward_on_intermediary_turns",
            "reward_on_relevant_symptom_inquiry",
            "reward_on_irrelevant_symptom_inquiry",
        ],
        prefix="simulator_",
        log_hp_flag=log_hp_flag,
    )
    return sim_params, rew_params


def load_component_metadata(
    component, hyper_params, mandatory_params=None, prefix="", log_hp_flag=True
):
    """Method for loading the metadata of the provided component.

    Parameters
    ----------
    component: str
        component for which the metadata will be loaded.
    hyper_params: dict
        dictionary containing hyper-params data.
    mandatory_params: list, None
        list of expected mandatory params for the model to be loaded.
        Default: None
    pefix: str
        prefix to be used when logging the metadata. Default: ""
    log_hp_flag: boolean
        whether or not to log the hyper parameter in mlflow.
        Default: True

    Returns
    -------
    metadata: dict
        the loaded metadata.

    """

    data = hyper_params.get(component, {})

    # eventually add mandatory params
    keys = list(data.keys())
    if not (mandatory_params is None) and (len(mandatory_params) > 0):
        keys = list(set(keys + mandatory_params))

    # check and log params
    if len(keys) > 0:
        check_and_log_hp(keys, data, prefix=prefix, log_hp_flag=log_hp_flag)
    return data
