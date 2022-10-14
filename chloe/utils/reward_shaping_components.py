from collections import namedtuple

import torch
from rlpyt.agents.base import AgentInputs
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.buffer import buffer_method, buffer_to
from rlpyt.utils.tensor import select_at_indexes

from chloe.utils.clf_loss_utils import ClassifierLossFactory
from chloe.utils.rebuild_loss_utils import RebuildLossFactory
from chloe.utils.reward_shaping_utils import RewardShapingFactory
from chloe.utils.tensor_utils import (
    _clamp_utils,
    _get_target_categorical_distributional,
    _negate_tensor,
)


class _fake_context:
    """A Fake context manager that does nothing.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _get_context(back_propagate_flag):
    """Determines the context in which to perform the computation.

    This method helps activate `torch.nograd()` when necessary.

    Parameters
    ----------
    back_propagate_flag: bool
        Flag indicating whether to backpropagate or not.

    Return
    ---------
    ctx: object
        a context manager where the Autograd is eventually deactivated
        depending on the value of `back_propagate_flag`.

    """
    if not back_propagate_flag:
        return torch.no_grad()
    else:
        return _fake_context()


class PretrainClassifierMixin:
    """Mixin class to pretrain classifiers in Mixed DQN-like Algos.

    """

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """Initializes the replay buffer for training the agent.

        This methods is overloaded to eventually request to the replay buffer
        to return data from intermediate states when sampling from it.

        Parameters
        ----------
        examples: obj
            example data for initializing the buffer with.
        batch_spec: tuple
            specification related to the sampler used for collecting data.
        async_: bool
            flag indicating if the buffer will operate in asynchronous mode.
            Default: False

        Return
        ---------
        None

        """
        super().initialize_replay_buffer(examples, batch_spec, async_)
        if hasattr(self.replay_buffer, "_set_intermediate_data_flag"):
            self.replay_buffer._set_intermediate_data_flag(
                self.replay_intermediate_data_flag
            )

    def optim_initialize(self, rank=0):
        """Ititializes the optimizer for training the agent.

        This methods is overloaded to eventually create the optimizer for
        the classifier. This method is called in `initialize` or by
        async runner after forking sampler.

        Parameters
        ----------
        rank: int
            the rank attributed by `rlpyt` to the agent. Default: 0

        Return
        ---------
        None

        """
        super().optim_initialize(rank)
        if self.separate_classifier_optimizer:
            self.clf_optimizer = self.OptimCls(
                self.agent.parameters(),
                lr=self.clf_learning_rate,
                **self.clf_optim_params,
            )
        else:
            self.clf_optimizer = self.optimizer

    def set_pretrain_params(
        self,
        pretrain_flag=True,
        separate_classifier_optimizer=False,
        pretrain_validation_percentage=0.25,
        pretrain_epochs=10,
        pretrain_batch_size=32,
        pretrain_clf_learning_rate=None,
        clf_learning_rate=None,
        clf_optim_params=None,
        pretrain_perf_metric="Accuracy",
        pretrain_patience=5,
        pretrain_loss_func="cross_entropy",
        pretrain_loss_kwargs=None,
    ):
        """Set the parameters for pretraining the classifier in Mixed DQN-like Algos.

        Parameters
        ----------
        pretrain_flag: bool
            whether or not to pretrain the classifier. Default: True
        separate_classifier_optimizer: bool
            whether or not the classifier has its own optimizer. Default: False
        pretrain_validation_percentage: float
            the percentage of data to be used for validaton. Must be in [0, 1) with 1
            excluded. Default: 0.25
        pretrain_epochs: int
            the number of epochs to pretrain the classifier. Default: 10
        pretrain_batch_size: int
            the batch size to be used during pretraining. Default: 32
        pretrain_clf_learning_rate: float
            the learning rate to use during the classifier pretraining. if None,
            the learning rate of the agent will be used. Default: None
        clf_learning_rate: float
            the learning rate to use during the classifier training. if None,
            the learning rate of the agent will be used. Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        clf_optim_params: dict
            the params for initializing the classifier optimizer.  Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        pretrain_perf_metric: str
            the performance metric used for pretraining. Default: Accuracy
        pretrain_patience: int
            the authorized patience for early stopping when pretraining. Default: 5
        pretrain_loss_func: str
            the loss function used for pretraining. Default: cross_entropy
        pretrain_loss_kwargs: dict
            variable arguments for the pretrain loss function. Default: None

        Return
        ----------
        None

        """
        validation_percentage = pretrain_validation_percentage
        assert (validation_percentage >= 0) and (validation_percentage < 1)
        assert pretrain_batch_size > 0
        self.pretrain_flag = pretrain_flag
        self.pretrain_validation_percentage = validation_percentage
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_batch_size = pretrain_batch_size
        if pretrain_clf_learning_rate is None:
            pretrain_clf_learning_rate = self.learning_rate
        if clf_learning_rate is None:
            clf_learning_rate = self.learning_rate
        self.pretrain_clf_learning_rate = pretrain_clf_learning_rate
        self.clf_learning_rate = clf_learning_rate
        self.separate_classifier_optimizer = separate_classifier_optimizer
        self.clf_optim_params = {} if clf_optim_params is None else clf_optim_params
        self.pretrain_perf_metric = pretrain_perf_metric
        self.pretrain_patience = pretrain_patience
        self.pretrain_loss_func = pretrain_loss_func
        if pretrain_loss_kwargs is None:
            pretrain_loss_kwargs = {}
        self.pretrain_loss_kwargs = pretrain_loss_kwargs
        self.clf_loss_factory = ClassifierLossFactory()
        self.pretrain_validation_data = []


class RebuildDQNLossMixin:
    """Mixin class to include feature rebuilding in DQN-like Algos.

    This class overloads the `loss` function such as to integrate the reconstruction
    loss when computing the loss of the DQN-like agent being trained.

    """

    def set_feature_rebuild_loss_params(
        self,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
    ):
        """Set the parameters for adding feature rebuilding loss in DQN-like Algos.

        Feature rebuilding loss is added each time we compute DQN loss.

        Parameters
        ----------
        feature_rebuild_loss_flag: bool
            whether or not to add feature rebuilding loss. Default: False
        feature_rebuild_loss_min: float
            min value of the feature rebuilding loss. Default: None
        feature_rebuild_loss_max: float
            max value of the feature rebuilding loss. Default: None
        feature_rebuild_loss_coef: float
            coefficient of the feature rebuilding loss. Default: 1.0
        feature_rebuild_loss_func: str
            the loss function used for computing the reconstruction loss.
            Default: bce
        feature_rebuild_loss_kwargs: dict
            variable arguments for the loss function. Default: None

        Return
        ----------
        None

        """
        self.feature_rebuild_loss_flag = feature_rebuild_loss_flag
        self.feature_rebuild_loss_min = feature_rebuild_loss_min
        self.feature_rebuild_loss_max = feature_rebuild_loss_max
        self.feature_rebuild_loss_func = feature_rebuild_loss_func
        self.feature_rebuild_loss_coef = feature_rebuild_loss_coef
        if feature_rebuild_loss_kwargs is None:
            feature_rebuild_loss_kwargs = dict()
        self.feature_rebuild_loss_kwargs = feature_rebuild_loss_kwargs
        if getattr(self, "rebuild_loss_factory", None) is None:
            self.rebuild_loss_factory = RebuildLossFactory()
        self.feature_rebuild_loss_stats = {}

    def _define_rebuild_opt_info(self):
        """Define the fields of the optimization data info.
        """
        result = []
        if self.feature_rebuild_loss_flag:
            result.append("rebuild_loss")
            self.opt_info_fields = tuple(self.opt_info_fields + tuple(result))

        self.rebuild_variable_logs = result
        self.optInfoCls = namedtuple("AugOptInfo", self.opt_info_fields)

    def _update_rebuild_loss_stats(self, stats, values, log_dict):
        """Update the stats of a computed classifier loss.

        Parameters
        ----------
        stats: dict
            the stats object to be updated.
        values: tensor
            the (main) values to update the stats with.
        log_dict: dict
            the dictionnary containing the (sub-component) data to
            update the stats with.

        Return
        ----------
        result: dict
            the updated stats.

        """
        prefix = "rebuild_loss"
        keys = (
            list(log_dict.keys())
            if log_dict
            and self.feature_rebuild_loss_kwargs.get("log_component_flag", False)
            else []
        )
        keys += [""]
        for k in keys:
            val = values if k == "" else log_dict[k]
            t = k if k == "" else f"_{k}"
            stats[f"{prefix}{t}"] = val.item()
        return stats

    def _get_empty_optim_info(self):
        """Returns an empty optimization info object.

        Parameters
        ----------

        Return
        ----------
        opt_info: obj
            the empty optimization info object.

        """
        return self.optInfoCls(*([] for _ in range(len(self.optInfoCls._fields))))

    def initialize(self, *args, **kwargs):
        """Initializes the Rebuild DQN like Algos.

        """
        super().initialize(*args, **kwargs)
        # define the opt info class
        self._define_rebuild_opt_info()

    def _compute_rebuild_loss(self, samples):
        """Computes the feature reconstruction loss.

        This method is specific to Rebuild DQN Algo. It defines the loss
        while integrating eventually the reconstructed feature.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the loss is computed.

        Return
        ----------
        loss: obj
            the computed loss information.

        """
        self.feature_rebuild_loss_stats.clear()

        if not hasattr(self.agent, "saved_reconstruct"):
            return 0.0
        if not self.feature_rebuild_loss_flag:
            return 0.0

        gt_data = samples.sim_patient
        pred_data = self.agent.saved_reconstruct

        # tranfer data into agent device
        gt_data = buffer_to(gt_data, device=self.agent.device)
        value, reb_loss_dict = self.rebuild_loss_factory.evaluate(
            self.feature_rebuild_loss_func,
            pred_data,
            gt_data[..., -pred_data.size(-1) :],
            weight=None,
            reduction="mean",
            **self.feature_rebuild_loss_kwargs,
        )
        value = _clamp_utils(
            value, self.feature_rebuild_loss_min, self.feature_rebuild_loss_max
        )
        self._update_rebuild_loss_stats(
            self.feature_rebuild_loss_stats, value, reb_loss_dict
        )
        return value

    def loss(self, samples):
        """Computes the Q-learning loss while integrating the rebuild loss.

        This method is specific to Rebuild DQN Algo. It defines the loss
        while integrating eventually the rebuild loss.
        This is mainly inspired by `rlpyt.algos.dqn.dqn.loss(samples)`
        function with the adaptation that allows to deal with Mixed output, that is,
        Q-values and reconstructed feature.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the loss is computed.

        Return
        ----------
        loss: obj
            the computed loss information.

        """
        data_loss_info = list(super().loss(samples))

        # get reconstruction loss coef
        rebuild_loss_coeff = self.feature_rebuild_loss_coef
        # compute the loss on reconstructed data
        rebuild_loss = (
            0.0
            if rebuild_loss_coeff == 0.0 or not self.feature_rebuild_loss_flag
            else self._compute_rebuild_loss(samples)
        )
        # update the loss
        data_loss_info[0] += rebuild_loss_coeff * rebuild_loss

        return data_loss_info

    def _update_rebuild_loss_info(self, opt_info):
        """Updates the optimization info with the rebuild loss stats.

        This method is called in the `optimize_agent` method.

        Parameters
        ----------
        opt_info: obj
            the information about the optimization performed so far.

        Return
        ----------
        opt_info: obj
            the updated information with the auxiliary reward stats.

        """
        for k in self.rebuild_variable_logs:
            if k.startswith("rebuild_loss"):
                getattr(opt_info, k).append(self.feature_rebuild_loss_stats.get(k, 0.0))

        return opt_info

    def _apply_optimization(self, samples, opt_info):
        """Applies the optimization for Mixed DQN-like Algos.

        This method is called in the `optimize_agent` method.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the
            optimization is performed.
        opt_info: obj
            the information about the optimization performed so far.

        Return
        ----------
        opt_info: obj
            the updated information with the applied optimization.

        """
        opt_info = super()._apply_optimization(samples, opt_info)
        opt_info = self._update_rebuild_loss_info(opt_info)

        return opt_info


class MixedDQNRewardShapingLossMixin:
    """Mixin class to include reward shaping in Mixed DQN-like Algos.

    This class overloads the `loss` function such as to integrate the auxiliary
    rewards when computing the loss of the DQN-like agent being trained.

    """

    def set_clf_reward_params(
        self,
        clf_reward_flag=False,
        clf_reward_min=None,
        clf_reward_max=None,
        clf_reward_coef=1.0,
        clf_reward_func="cross_entropy",
        clf_reward_kwargs=None,
    ):
        """Set the parameters for adding classification reward in Mixed DQN-like Algos.

        Classification rewards are added at the end of an episode when the agent makes
        a prediction or when it consumes all its budget.

        Parameters
        ----------
        clf_reward_flag: bool
            whether or not to add classification reward. Default: False
        clf_reward_min: float
            min value of the classifcation reward. Default: None
        clf_reward_max: float
            max value of the classifcation reward. Default: None
        clf_reward_coef: float
            coefficient of the auxiliary reward. Default: 1.0
        clf_reward_func: str
            the reward function used for rewading the agent at the end of the
            interaction session. Default: cross_entropy
        clf_reward_kwargs: dict
            variable arguments for the reward shaping function. Default: None

        Return
        ----------
        None

        """
        self.clf_reward_flag = clf_reward_flag
        self.clf_reward_min = clf_reward_min
        self.clf_reward_max = clf_reward_max
        self.clf_reward_func = clf_reward_func
        self.clf_reward_coef = clf_reward_coef
        if clf_reward_kwargs is None:
            clf_reward_kwargs = dict()
        self.clf_reward_kwargs = clf_reward_kwargs
        if getattr(self, "clf_loss_factory", None) is None:
            self.clf_loss_factory = ClassifierLossFactory()
        self.delta_timestep = None
        self.clf_reward_stats = {}

    def set_clf_loss_params(
        self,
        clf_loss_flag=False,
        clf_loss_complete_data_flag=False,
        clf_loss_only_at_end_episode_flag=True,
        clf_loss_func="cross_entropy",
        clf_loss_kwargs=None,
    ):
        """Set the parameters for classification loss in Mixed DQN-like Algos.

        Parameters
        ----------
        clf_loss_flag: bool
            whether or not to perform classification loss update. Default: False
        clf_loss_complete_data_flag: bool
            whether or not to consider the complete simulated patients  when
            computing classification loss. Default: False
        clf_loss_only_at_end_episode_flag: bool
            whether or not to provide classification error signals
            at end of episode. Default: True
        clf_loss_func: str
            the loss function used for training. Default: cross_entropy
        clf_loss_kwargs: dict
            variable arguments for the classifier loss function. Default: None

        Return
        ----------
        None

        """
        self.clf_loss_flag = clf_loss_flag
        self.clf_loss_complete_data_flag = clf_loss_complete_data_flag
        self.clf_loss_only_at_end_episode_flag = clf_loss_only_at_end_episode_flag
        self.clf_loss_func = clf_loss_func
        if clf_loss_kwargs is None:
            clf_loss_kwargs = dict()
        self.clf_loss_kwargs = clf_loss_kwargs
        if getattr(self, "clf_loss_factory", None) is None:
            self.clf_loss_factory = ClassifierLossFactory()
        self.clf_loss_stats = {}

    def set_reward_shaping_params(
        self,
        reward_shaping_flag=False,
        reward_shaping_min=None,
        reward_shaping_max=None,
        reward_shaping_coef=1.0,
        env_reward_coef=1.0,
        reward_shaping_back_propagate_flag=False,
        reward_shaping_func="cross_entropy",
        reward_shaping_kwargs=None,
    ):
        """Set the parameters for adding reward shaping in Mixed DQN-like Algos.

        Parameters
        ----------
        reward_shaping_flag: bool
            whether or not to add reward shaping. Default: False
        reward_shaping_min: float
            min value of the auxiliary reward. Default: None
        reward_shaping_max: float
            max value of the auxiliary reward. Default: None
        reward_shaping_coef: float
            coefficient of the auxiliary reward. Default: 1.0
        env_reward_coef: float
            coefficient of the classic reward. Default: 1.0
        reward_shaping_back_propagate_flag: bool
            whether or not to backpropagate on auxiliary rewards. Default: False
        reward_shaping_func: str
            the function used to compute the auxiliary reward. Default: cross_entropy
        reward_shaping_kwargs: dict
            variable arguments for the reward shaping function. Default: None

        Return
        ----------
        None

        """
        self.reward_shaping_flag = reward_shaping_flag
        self.reward_shaping_min = reward_shaping_min
        self.reward_shaping_max = reward_shaping_max
        self.reward_shaping_coef = reward_shaping_coef
        self.env_reward_coef = env_reward_coef
        self.reward_shaping_back_propagate_flag = reward_shaping_back_propagate_flag
        self.reward_shaping_func = reward_shaping_func
        if reward_shaping_kwargs is None:
            reward_shaping_kwargs = dict()
        self.reward_shaping_kwargs = reward_shaping_kwargs
        self.reward_shaping_factory = RewardShapingFactory()
        self.reward_shaping_stats = {}

    def _define_opt_info(self):
        """Define the fields of the optimization data info.
        """
        result = []
        if self.reward_shaping_flag or self.clf_reward_flag or self.clf_loss_flag:
            num_cls = 2 if self.patho_severity is None else len(self.patho_severity)
            with torch.no_grad():
                tmp_dist = torch.rand(num_cls, num_cls, device=self.agent.device)
                # we consider all the pathologies
                sim_patho = torch.arange(num_cls, device=self.agent.device)
                timestep = torch.zeros(num_cls, device=self.agent.device)
                differential_indices = sim_patho.unsqueeze(1)
                differential_probas = torch.ones(
                    sim_patho.unsqueeze(1).size(), device=self.agent.device
                )
                evidence = torch.zeros(num_cls, device=self.agent.device)
                aux_keys, clf_rew_keys, clf_loss_keys = [], [], []
                if self.reward_shaping_flag:
                    _, aux_dict = self.reward_shaping_factory.evaluate(
                        self.reward_shaping_func,
                        tmp_dist,
                        tmp_dist,
                        sim_patho,
                        differential_indices=differential_indices,
                        differential_probas=differential_probas,
                        evidence=evidence,
                        discount=self.discount,
                        severity=self.patho_severity,
                        timestep=timestep,
                        **self.reward_shaping_kwargs,
                    )
                    tmp_flag = self.reward_shaping_kwargs.get(
                        "log_component_flag", False
                    )
                    aux_keys = list(aux_dict.keys()) if aux_dict and tmp_flag else []
                    aux_keys += [""]
                if self.clf_reward_flag:
                    _, clf_rew_dict = self.clf_loss_factory.evaluate(
                        self.clf_reward_func,
                        tmp_dist,
                        sim_patho,
                        differential_indices=differential_indices,
                        differential_probas=differential_probas,
                        reduction="none",
                        severity=self.patho_severity,
                        timestep=timestep,
                        **self.clf_reward_kwargs,
                    )
                    tmp_flag = self.clf_reward_kwargs.get("log_component_flag", False)
                    clf_rew_keys = (
                        list(clf_rew_dict.keys()) if clf_rew_dict and tmp_flag else []
                    )
                    clf_rew_keys += [""]
                if self.clf_loss_flag:
                    _, clf_loss_dict = self.clf_loss_factory.evaluate(
                        self.clf_loss_func,
                        tmp_dist,
                        sim_patho,
                        differential_indices=differential_indices,
                        differential_probas=differential_probas,
                        reduction="none",
                        severity=self.patho_severity,
                        timestep=timestep,
                        **self.clf_loss_kwargs,
                    )
                    tmp_flag = self.clf_loss_kwargs.get("log_component_flag", False)
                    clf_loss_keys = (
                        list(clf_loss_dict.keys()) if clf_loss_dict and tmp_flag else []
                    )
                    clf_loss_keys += [""]

                stats = ["min", "max", "avg", "median"]
                for k in aux_keys:
                    t = k if k == "" else f"_{k}"
                    result.extend([f"aux_rew{t}_{a}_info_" for a in stats])
                for k in clf_rew_keys:
                    t = k if k == "" else f"_{k}"
                    result.extend([f"clf_rew{t}_{a}_info_" for a in stats])
                for k in clf_loss_keys:
                    t = k if k == "" else f"_{k}"
                    result.extend([f"clf_loss{t}"])
                if len(result) > 0:
                    self.opt_info_fields = tuple(self.opt_info_fields + tuple(result))

        self.variable_logs = result
        self.optInfoCls = namedtuple("AugOptInfo", self.opt_info_fields)

    def _update_auxiliary_rew_stats(self, stats, values, log_dict, prefix):
        """Update the stats of a computed auxiliary reward.

        Parameters
        ----------
        stats: dict
            the stats object to be updated.
        values: tensor
            the (main) values to update the stats with.
        log_dict: dict
            the dictionnary containing the (sub-component) data to
            update the stats with.
        prefix: str
            the prefix used to log the data. one of ['clf_rew', 'aux_rew'].

        Return
        ----------
        result: dict
            the updated stats.

        """
        assert prefix in ("clf_rew", "aux_rew")
        log_data_component_flag = (
            self.clf_reward_kwargs.get("log_component_flag", False)
            if prefix == "clf_rew"
            else self.reward_shaping_kwargs.get("log_component_flag", False)
        )
        keys = list(log_dict.keys()) if log_dict and log_data_component_flag else []
        keys += [""]
        for k in keys:
            val = values if k == "" else log_dict[k]
            t = k if k == "" else f"_{k}"
            stats[f"{prefix}{t}_min_info_"] = val.min().item()
            stats[f"{prefix}{t}_max_info_"] = val.max().item()
            stats[f"{prefix}{t}_avg_info_"] = val.mean().item()
            stats[f"{prefix}{t}_median_info_"] = val.median().item()
        return stats

    def _update_clf_loss_stats(self, stats, values, log_dict):
        """Update the stats of a computed classifier loss.

        Parameters
        ----------
        stats: dict
            the stats object to be updated.
        values: tensor
            the (main) values to update the stats with.
        log_dict: dict
            the dictionnary containing the (sub-component) data to
            update the stats with.

        Return
        ----------
        result: dict
            the updated stats.

        """
        prefix = "clf_loss"
        keys = (
            list(log_dict.keys())
            if log_dict and self.clf_loss_kwargs.get("log_component_flag", False)
            else []
        )
        keys += [""]
        for k in keys:
            val = values if k == "" else log_dict[k]
            t = k if k == "" else f"_{k}"
            stats[f"{prefix}{t}"] = val.item()
        return stats

    def _get_empty_optim_info(self):
        """Returns an empty optimization info object.

        Parameters
        ----------

        Return
        ----------
        opt_info: obj
            the empty optimization info object.

        """
        return self.optInfoCls(*([] for _ in range(len(self.optInfoCls._fields))))

    def initialize(self, *args, **kwargs):
        """Initializes the Mixed DQN like Algos.

        """
        super().initialize(*args, **kwargs)
        self.patho_severity = getattr(self.agent.model, "patho_severity", None)
        self.min_turn_ratio = getattr(
            self.agent.model, "min_turns_ratio_for_decision", None
        )
        if self.patho_severity is not None:
            self.patho_severity = torch.tensor(self.patho_severity)
            self.patho_severity = self.patho_severity.to(self.agent.device)
        # define the opt info class
        self._define_opt_info()

    def select_at_indexes(self, indexes, tensor):
        """Returns the `tensor` data at the multi-dimensional integer array `indexes`.

        Parameters
        ----------
        indexes: tensor
            a tensor of indexes.
        tensor: tensor
            a tensor from which to retrieve the data of interest.

        Return
        ----------
        result: tensor
            the resulting data.

        """
        dim = len(indexes.shape)
        num_values = tensor.size(dim)

        # alters indexes to match num_values
        if getattr(self.agent.model, "use_stop_action", True):
            mask = indexes > num_values - 1
            indexes[mask] = num_values - 1

        return select_at_indexes(indexes, tensor)

    def loss(self, samples):
        """Computes the Q-learning loss while integrating auxiliary rewards.

        This method is specific to Mixed DQN Algo. It defines the loss
        while integrating eventually the auxilirary reward based on reward shaping.
        This is mainly inspired by `rlpyt.algos.dqn.dqn.loss(samples)`
        function with the adaptation that allows to deal with Mixed output, that is,
        Q-values and posterior probability distribution.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the loss is computed.

        Return
        ----------
        loss: obj
            the computed loss information.

        """
        # add the auxiliary rewards
        if self.reward_shaping_flag or self.clf_reward_flag:
            samples = self._add_auxiliary_rewards(samples)

        data_loss_info = list(super().loss(samples))

        # get exit loss coef
        exit_loss_coeff = self.clf_reward_kwargs.get("exit_loss_coeff", 1.0)
        # compute the loss on exit token
        exit_loss = (
            0.0
            if exit_loss_coeff == 0.0
            else self._compute_exit_loss_on_target(samples)
        )
        # update the loss
        data_loss_info[0] += exit_loss_coeff * exit_loss

        return data_loss_info

    def _update_loss_and_auxiliary_reward_info(self, opt_info):
        """Updates the optimization info with the auxiliary reward and loss stats.

        This method is called in the `optimize_agent` method.

        Parameters
        ----------
        opt_info: obj
            the information about the optimization performed so far.

        Return
        ----------
        opt_info: obj
            the updated information with the auxiliary reward stats.

        """
        for k in self.variable_logs:
            if k.startswith("clf_rew"):
                getattr(opt_info, k).append(self.clf_reward_stats.get(k, 0.0))
            if k.startswith("aux_rew"):
                getattr(opt_info, k).append(self.reward_shaping_stats.get(k, 0.0))
            if k.startswith("clf_loss"):
                getattr(opt_info, k).append(self.clf_loss_stats.get(k, 0.0))

        return opt_info

    def _apply_optimization(self, samples, opt_info):
        """Applies the optimization for Mixed DQN-like Algos.

        This method is called in the `optimize_agent` method.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the
            optimization is performed.
        opt_info: obj
            the information about the optimization performed so far.

        Return
        ----------
        opt_info: obj
            the updated information with the applied optimization.

        """
        opt_info = super()._apply_optimization(samples, opt_info)
        if self.clf_loss_flag:
            self.clf_optimizer.zero_grad()
            clf_loss = self._loss_classifier(samples)
            if clf_loss is not None:
                clf_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm
                )
                self.clf_optimizer.step()
        opt_info = self._update_loss_and_auxiliary_reward_info(opt_info)

        return opt_info


class MixedDQNRewardShapingNonSeqMixin:
    """Mixin class to include reward shaping using non sequential replay buffers.

    It defines a method to add auxiliary rewards as well as a method
    to compute classification loss.

    """

    def _loss_classifier(self, samples):
        """Computes the classification loss.

        This method computes the classification loss of the agent.
        All these operations are performed on sample data from
        the replay buffer.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which
            the classification loss is computed.

        Return
        ----------
        loss: tensor
            the computed loss.

        """
        # clear previous stats
        self.clf_loss_stats.clear()

        is_void_simpatho = getattr(samples, "sim_patho", None) is None
        is_void_differential = (
            getattr(samples, "sim_differential_indices", None) is None
        ) or (getattr(samples, "sim_differential_probas", None) is None)
        if is_void_simpatho and is_void_differential:
            return None
        data = [*samples.agent_inputs]
        target = samples.sim_patho
        timestep = samples.sim_timestep
        differential_indices = samples.sim_differential_indices
        differential_probas = samples.sim_differential_probas
        done = samples.done
        if self.clf_loss_complete_data_flag:
            # concatenate sample and real simulated patient symptoms
            data[0] = torch.cat([data[0], samples.sim_patient])
            data[1] = torch.cat([data[1], data[1]])
            data[2] = torch.cat([data[2], data[2]])

            target = torch.cat([target, target])
            timestep = None if timestep is None else torch.cat([timestep, timestep])
            differential_indices = (
                None
                if differential_indices is None
                else torch.cat([differential_indices, differential_indices])
            )
            differential_probas = (
                None
                if differential_probas is None
                else torch.cat([differential_probas, differential_probas])
            )
            done = torch.cat([done, torch.ones_like(done)])

        # tranfer data into agent device
        [
            data[0],
            data[1],
            data[2],
            target,
            timestep,
            done,
            differential_indices,
            differential_probas,
        ] = buffer_to(
            (
                data[0],
                data[1],
                data[2],
                target,
                timestep,
                done,
                differential_indices,
                differential_probas,
            ),
            device=self.agent.device,
        )

        pis = self.agent.classify(*data)
        pis = pis.transpose(1, -1)

        differential_indices = (
            None
            if differential_indices is None
            else differential_indices.transpose(1, -1)
        )
        differential_probas = (
            None
            if differential_probas is None
            else differential_probas.transpose(1, -1)
        )

        if not self.clf_loss_only_at_end_episode_flag:
            loss, loss_dict = self.clf_loss_factory.evaluate(
                self.clf_loss_func,
                pis,
                target,
                differential_indices=differential_indices,
                differential_probas=differential_probas,
                severity=self.patho_severity,
                timestep=timestep,
                **self.clf_loss_kwargs,
            )
        else:
            loss, loss_dict = self.clf_loss_factory.evaluate(
                self.clf_loss_func,
                pis,
                target,
                differential_indices=differential_indices,
                differential_probas=differential_probas,
                reduction="none",
                severity=self.patho_severity,
                timestep=timestep,
                **self.clf_loss_kwargs,
            )
            num_elts = done.float().sum().item()
            loss *= done.float()
            loss = torch.sum(loss) / max(num_elts, 1)
            if loss_dict:
                for k in loss_dict:
                    loss_dict[k] *= done.float()
                    loss_dict[k] = torch.sum(loss_dict[k]) / max(num_elts, 1)
        self.clf_loss_stats = self._update_clf_loss_stats(
            self.clf_loss_stats, loss, loss_dict
        )
        return loss

    def _compute_discounted_auxiliary_rewards(
        self, aux_reward, clf_reward, steps_done, s_done_n, intermediate_flag
    ):
        """Computes the nstep discounted return for auxiliary rewards.

        Parameters
        ----------
        aux_reward: tensor
            auxiliary rewards induced by reward shaping.
        clf_reward: tensor
            auxiliary rewards induced by classification errors.
        steps_done: tensor
            signal indicating if a trajectory ends or not.
        s_done_n: tensor
            signal indicating if a trajectory ends at any future time
            before the n-step or not.
        intermediate_flag: bool
            flag indicating the presence of intermediate data or not.

        Return
        ----------
        discounted_aux_return: tensor
            the discounted return for auxiliary rewards induced by reward shaping.
        discounted_clf_return: tensor
            the discounted return for auxiliary rewards induced by classification
            errors.

        """
        if aux_reward is None and clf_reward is None:
            return 0.0, 0.0

        # compute discounted rewards
        discount = self.discount
        n_step = self.n_step_return

        discounted_aux_return = aux_reward[0] if aux_reward is not None else 0
        discounted_clf_return = clf_reward[0] if clf_reward is not None else 0

        # done_n is used to get n-step done signals, which is
        # True if `done=True` at any future time before the n-step
        # target bootstrap
        if intermediate_flag:
            done_n = steps_done[0]
            # make sure we consider this reward if the current state is not done
            discounted_aux_return *= 1 - done_n.float()

            # make sure we consider this reward only if the current state s_done is done
            discounted_clf_return *= done_n.float()

            for n in range(1, n_step):
                next_aux_reward = aux_reward[n] if aux_reward is not None else 0
                next_clf_reward = clf_reward[n] if clf_reward is not None else 0

                # make sure we consider this reward if the current state is not done
                next_aux_reward *= 1 - steps_done[n].float()
                # make sure we consider this reward if the current state is done
                next_clf_reward *= steps_done[n].float()

                discounted_aux_return += (
                    (discount ** n) * next_aux_reward * (1 - done_n.float())
                )
                discounted_clf_return += (
                    (discount ** n) * next_clf_reward * (1 - done_n.float())
                )

                done_n = torch.max(done_n, steps_done[n])
        else:
            # use the sample done_n if no intermediate data is available
            done_n = s_done_n
            # make sure we consider this reward if the current state is not done
            discounted_aux_return *= 1 - done_n.float()

            # make sure we consider this reward only if the current state s_done is done
            discounted_clf_return *= steps_done[0].float()

        return discounted_aux_return, discounted_clf_return

    def _add_auxiliary_rewards(self, samples):
        """Integrates auxiliary rewards (reward shaping) into the environment rewards.

        This method adds the auxiliary rewards into the classical rewards
        returned by the environment. These auxiliary rewards are based on
        potential reward shaping functions or the classification errors.
        Here, the discounted auxiliary reward is computed and added to the
        discounted rewards from the environment. All these operations are performed on
        sample data from the replay buffer.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which
            the auxiliary reward is added.

        Return
        ----------
        samples: obj
            the altered sample data in which the auxiliary rewards
            have been integrated.

        """
        clf_reward = None
        aux_reward = None
        intermediate_data = (
            None
            if getattr(samples, "intermediate_data") is None
            else samples.intermediate_data
        )

        # clear previous stats
        self.clf_reward_stats.clear()
        self.reward_shaping_stats.clear()

        # tranfer data into agent device
        (intermediate_data,) = buffer_to((intermediate_data,), device=self.agent.device)
        # tranfer data into agent device
        agent_inputs, target_inputs, s_sim_patho, s_tstep, s_done, s_done_n = buffer_to(
            (
                samples.agent_inputs,
                samples.target_inputs,
                samples.sim_patho,
                samples.sim_timestep,
                samples.done,
                samples.done_n,
            ),
            device=self.agent.device,
        )
        s_differential_indices, s_differential_probas, s_evidence = buffer_to(
            (
                samples.sim_differential_indices,
                samples.sim_differential_probas,
                samples.sim_evidence,
            ),
            device=self.agent.device,
        )

        # put all the data together
        steps_done = [s_done]
        all_data = [*agent_inputs]
        all_data = [[all_data[i]] for i in range(len(all_data))]
        if intermediate_data is not None:
            steps_done.append(intermediate_data.done)
            for i, data in enumerate(intermediate_data.inputs):
                all_data[i].append(data)
        for i, data in enumerate(target_inputs):
            all_data[i].append(data)

        # concat the data
        steps_done = torch.cat(steps_done, dim=0)
        for i in range(len(all_data)):
            all_data[i] = torch.cat(all_data[i], dim=0)

        B = s_done.size(0)
        N = all_data[0].size(0) // B

        # initialize delta time step
        if self.delta_timestep is None:
            self.delta_timestep = torch.arange(
                N - 1, device=self.agent.device
            ).unsqueeze(1)

        with _get_context(self.reward_shaping_back_propagate_flag):
            all_pis = self.agent.classify(*all_data)
            all_pis_size = all_pis.size()
            all_pis = all_pis.reshape(N, B, *all_pis_size[1:])

            # reshape the sim_patho to match the desired size
            sim_patho = s_sim_patho
            sim_patho = (
                None
                if sim_patho is None
                else sim_patho.unsqueeze(0).expand(N - 1, *sim_patho.size())
            )

            # reshape the differential_indices to match the desired size
            differential_indices = s_differential_indices
            dind_flag = differential_indices is not None
            d_size = differential_indices.size() if dind_flag else None
            differential_indices = (
                differential_indices.unsqueeze(0).expand(N - 1, *d_size)
                if dind_flag
                else None
            )

            # reshape the differential_probas to match the desired size
            differential_probas = s_differential_probas
            dprob_flag = differential_probas is not None
            d_size = differential_probas.size() if dprob_flag else None
            differential_probas = (
                differential_probas.unsqueeze(0).expand(N - 1, *d_size)
                if dprob_flag
                else None
            )

            # reshape the sim_evidence to match the desired size
            evidence = s_evidence
            evid_flag = s_evidence is not None
            d_size = evidence.size() if evid_flag else None
            evidence = (
                evidence.unsqueeze(0).expand(N - 1, *d_size) if evid_flag else None
            )

            # reshape the timestep to match the desired size
            timestep = s_tstep
            if timestep is not None:
                timestep = timestep.unsqueeze(0).expand(N - 1, *timestep.size())
                timestep = timestep + self.delta_timestep

            # reshape steps_done to match the desired size
            steps_done = steps_done.reshape(-1, *s_done.size()).bool()

            # resulting auxiliary reward initialized with zeros
            result = torch.zeros(*steps_done.size(), device=self.agent.device)

            is_on_simpatho = sim_patho is not None
            is_on_differential = (differential_indices is not None) and (
                differential_probas is not None
            )
            is_on_clf = is_on_simpatho or is_on_differential
            if self.clf_reward_flag and is_on_clf and (steps_done.any()):
                # classification loss (e.g., negative cross entropy)
                tmp_pis = all_pis[0:-1]
                diff_indices = (
                    None
                    if differential_indices is None
                    else differential_indices[steps_done].transpose(1, -1)
                )
                diff_probas = (
                    None
                    if differential_probas is None
                    else differential_probas[steps_done].transpose(1, -1)
                )
                clf_reward_support, clf_rew_dic = self.clf_loss_factory.evaluate(
                    self.clf_reward_func,
                    tmp_pis[steps_done].transpose(1, -1),
                    sim_patho[steps_done],
                    differential_indices=diff_indices,
                    differential_probas=diff_probas,
                    reduction="none",
                    severity=self.patho_severity,
                    timestep=None if timestep is None else timestep[steps_done],
                    **self.clf_reward_kwargs,
                )
                clf_reward_support = _negate_tensor(clf_reward_support)
                clf_rew_dic = _negate_tensor(clf_rew_dic)

                # clamp it if needed
                clf_reward_support = _clamp_utils(
                    clf_reward_support, self.clf_reward_min, self.clf_reward_max
                )

                # compute raw aux reward statistics
                self.clf_reward_stats = self._update_auxiliary_rew_stats(
                    self.clf_reward_stats, clf_reward_support, clf_rew_dic, "clf_rew"
                )

                # save it
                result[steps_done] = clf_reward_support
                # Only consider this reward when the current state is done
                clf_reward = result * steps_done.float()

            tmp_all_data = all_data[0].reshape(N, B, *all_data[0].size()[1:])
            tmp_tgt_data = tmp_all_data[-1]
            tmp_done_n = (
                s_done_n.bool()
                if self.min_turn_ratio is None
                else (s_done_n.bool() | (tmp_tgt_data[:, 0] < self.min_turn_ratio))
            )
            # resulting clf reward on target initialized with zeros
            self.clf_rew_tgt = torch.zeros(*tmp_done_n.size(), device=self.agent.device)
            self.clf_rew_tgt_mask = tmp_done_n
            if self.clf_reward_flag and is_on_clf and not (tmp_done_n.all()):
                tmp_pis = all_pis[-1]
                diff_indices = (
                    None
                    if differential_indices is None
                    else differential_indices[-1][~tmp_done_n].transpose(1, -1)
                )
                diff_probas = (
                    None
                    if differential_probas is None
                    else differential_probas[-1][~tmp_done_n].transpose(1, -1)
                )
                tmp_pathos = None if sim_patho is None else sim_patho[-1][~tmp_done_n]
                tmp_timestep = (
                    None if timestep is None else timestep[-1][~tmp_done_n] + 1
                )
                clf_reward_on_target, _ = self.clf_loss_factory.evaluate(
                    self.clf_reward_func,
                    tmp_pis[~tmp_done_n].transpose(1, -1),
                    tmp_pathos,
                    differential_indices=diff_indices,
                    differential_probas=diff_probas,
                    reduction="none",
                    severity=self.patho_severity,
                    timestep=tmp_timestep,
                    **self.clf_reward_kwargs,
                )
                clf_reward_on_target = _negate_tensor(clf_reward_on_target)

                # clamp it if needed
                clf_reward_on_target = _clamp_utils(
                    clf_reward_on_target, self.clf_reward_min, self.clf_reward_max
                )
                # save it
                self.clf_rew_tgt[~tmp_done_n] = clf_reward_on_target

            if self.reward_shaping_flag and not (steps_done.all()):
                next_pis = all_pis[1:]
                prev_pis = all_pis[0:-1]
                diff_indices = (
                    None
                    if differential_indices is None
                    else differential_indices[~steps_done]
                )
                diff_probas = (
                    None
                    if differential_probas is None
                    else differential_probas[~steps_done]
                )
                aux_reward_support, aux_rew_dic = self.reward_shaping_factory.evaluate(
                    self.reward_shaping_func,
                    next_pis[~steps_done],
                    prev_pis[~steps_done],
                    None if sim_patho is None else sim_patho[~steps_done],
                    differential_indices=diff_indices,
                    differential_probas=diff_probas,
                    evidence=None if evidence is None else evidence[~steps_done],
                    discount=self.discount,
                    severity=self.patho_severity,
                    timestep=None if timestep is None else timestep[~steps_done],
                    **self.reward_shaping_kwargs,
                )

                # clamp it if needed
                aux_reward_support = _clamp_utils(
                    aux_reward_support, self.reward_shaping_min, self.reward_shaping_max
                )

                # compute raw aux reward statistics
                self.reward_shaping_stats = self._update_auxiliary_rew_stats(
                    self.reward_shaping_stats,
                    aux_reward_support,
                    aux_rew_dic,
                    "aux_rew",
                )

                # save it
                result[~steps_done] = aux_reward_support
                # Only consider this reward when the current state is not done
                aux_reward = result * (1.0 - steps_done.float())

            # compute discounted rewards
            intermediate_flag = intermediate_data is not None
            outputs_rew = self._compute_discounted_auxiliary_rewards(
                aux_reward, clf_reward, steps_done, s_done_n, intermediate_flag
            )
            discounted_aux_return, discounted_clf_return = outputs_rew

            # combined both rewards
            weighted_aux_return = self.reward_shaping_coef * discounted_aux_return
            weighted_cla_return = (
                self.env_reward_coef * samples.return_.to(self.agent.device)
                + self.clf_reward_coef * discounted_clf_return
            )
            final_return = weighted_aux_return + weighted_cla_return

        # modify the return in samples
        tmp_dict = dict(samples.items())
        tmp_dict["return_"] = final_return
        # set the transfered data in order to avoid transferring twice
        tmp_dict["agent_inputs"] = agent_inputs
        tmp_dict["target_inputs"] = target_inputs
        tmp_dict["sim_patho"] = s_sim_patho
        tmp_dict["sim_differential_indices"] = s_differential_indices
        tmp_dict["sim_differential_probas"] = s_differential_probas
        tmp_dict["sim_timestep"] = s_tstep
        tmp_dict["sim_evidence"] = s_evidence
        tmp_dict["done"] = s_done
        tmp_dict["done_n"] = s_done_n
        samples = samples.__class__(**tmp_dict)

        return samples

    def _compute_exit_loss_on_target(self, samples):
        """Computes the Q-learning loss assuming the exit token has been chosen.

        This method is specific to Mixed DQN Algo. It defines the loss
        while integrating eventually the auxilirary reward based on reward shaping
        assuming the exit token has been chosen. This methods mainly focuses on
        samples for which the `done_n` signal is not True.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the loss is computed.

        Return
        ----------
        loss: obj
            the computed loss information.

        """
        if not hasattr(self, "clf_rew_tgt"):
            return 0.0
        if not getattr(self.agent.model, "use_stop_action", True):
            return 0.0

        s_done = samples.done
        return_ = samples.return_
        is_weights = None if not self.prioritized_replay else samples.is_weights

        # tranfer data into agent device
        (is_weights,) = buffer_to((is_weights,), device=self.agent.device)

        # update the return with the computed classifier reward on targets
        return_ += (self.discount ** self.n_step_return) * self.clf_rew_tgt

        # get the saved q from the agent
        q = self.agent.saved_q
        # q values of exit token
        is_cat_agent = isinstance(self.agent, CatDqnAgent)
        q_exit = (
            q.transpose(-1, -2)[..., -1] if is_cat_agent else q[..., -1]  # B x P  # B
        )

        tmp_done_n = self.clf_rew_tgt_mask
        if not (tmp_done_n.all()):
            if not is_cat_agent:
                loss_info = self._get_loss_values(
                    q_exit[~tmp_done_n],
                    s_done[~tmp_done_n],
                    None if is_weights is None else is_weights[~tmp_done_n],
                    return_[~tmp_done_n],
                )
                return loss_info[0]
            else:
                if not hasattr(self, "lin_z"):
                    self.lin_z = torch.linspace(
                        self.agent.V_min,
                        self.agent.V_max,
                        self.agent.n_atoms,
                        device=self.agent.device,
                    )
                q_target = q_exit[~tmp_done_n].detach()  # B'x P
                target_p = _get_target_categorical_distributional(
                    return_[~tmp_done_n],
                    q_target,
                    self.agent.V_min,
                    self.agent.V_max,
                    self.lin_z,
                )
                loss_info = self._get_loss_values(
                    q_target,
                    s_done[~tmp_done_n],
                    None if is_weights is None else is_weights[~tmp_done_n],
                    target_p,
                )
                return loss_info[0]
        else:
            return 0.0


class MixedDQNRewardShapingSeqMixin:
    """Mixin class to include reward shaping using sequential replay buffers.

    It defines a method to add auxiliary rewards as well as a method
    to compute classification loss.

    """

    def _warm_up_rnn_states(
        self, all_observation, all_action, all_reward, all_done, init_rnn_state
    ):
        """Advances the RNN states by warming up the model on wanrn up sample data.

        Parameters
        ----------
        all_observation: tensor
            the observation as returned by the environment.
        all_action: tensor
            the action as performed within the environment.
        all_action: tensor
            the reward as returned by the environment.
        all_done: tensor
            the done signal as returned by the environment.
        init_rnn_state: tensor
            the rnn state to be warmed up/advanced.

        Return
        ----------
        init_rnn_state: obj
            the advanced rnn states.

        """
        wT = self.warmup_T
        if wT > 0:
            warmup_slice = slice(None, wT)  # Same for agent and target.
            warmup_inputs = AgentInputs(
                observation=all_observation[warmup_slice],
                prev_action=all_action[warmup_slice],
                prev_reward=all_reward[warmup_slice],
            )

        # define the init rnn state
        if self.store_rnn_state_interval == 0:
            init_rnn_state = None
        else:
            # [B,N,H]-->[N,B,H] cudnn.
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")

        if wT > 0:  # Do warmup.
            with torch.no_grad():
                _, init_rnn_state = self.agent(*warmup_inputs, init_rnn_state)
            # Recommend aligning sampling batch_T and store_rnn_interval with
            # warmup_T (and no mid_batch_reset), so that end of trajectory
            # during warmup leads to new trajectory beginning at start of
            # training segment of replay.
            warmup_invalid_mask = valid_from_done(all_done[:wT])[-1] == 0  # [B]
            init_rnn_state[:, warmup_invalid_mask] = 0  # [N,B,H] (cudnn)

        return init_rnn_state

    def _loss_classifier(self, samples):
        """Computes the classification loss for sequential replay memory.

        This method computes the classification loss of the agent.
        All these operations are performed on sample data from
        the replay buffer.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which
            the classification loss is computed.

        Return
        ----------
        loss: tensor
            the computed loss.

        """
        # clear previous stats
        self.clf_loss_stats.clear()

        is_void_simpatho = getattr(samples, "sim_patho", None) is None
        is_void_differential = (
            getattr(samples, "sim_differential_indices", None) is None
        ) or (getattr(samples, "sim_differential_probas", None) is None)
        if is_void_simpatho and is_void_differential:
            return None

        # tranfer data into agent device
        tmp = buffer_to(
            (
                samples.all_observation,
                samples.all_action,
                samples.all_reward,
                samples.done,
                samples.sim_patho,
                samples.sim_timestep,
                samples.sim_differential_indices,
                samples.sim_differential_probas,
                samples.init_rnn_state,
            ),
            device=self.agent.device,
        )
        [
            all_observation,
            all_action,
            all_reward,
            all_done,
            s_sim_patho,
            tstep,
            s_sim_differential_indices,
            s_sim_differential_probas,
            hs,
        ] = tmp
        wT, bT = self.warmup_T, self.batch_T
        init_rnn_state = self._warm_up_rnn_states(
            all_observation, all_action, all_reward, samples.done, hs
        )
        # now we start the computation of the classifier loss
        # all the previous code was inspired by how the r2d1 algo is boostraped
        # in rlpyt

        agent_slice = slice(wT, wT + bT)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice],
            prev_action=all_action[agent_slice],
            prev_reward=all_reward[agent_slice],
        )
        done = all_done[agent_slice]
        target = s_sim_patho[agent_slice]
        timestep = tstep[agent_slice] if tstep is not None else None
        differential_indices = (
            None
            if s_sim_differential_indices is None
            else s_sim_differential_indices[agent_slice]
        )
        differential_probas = (
            None
            if s_sim_differential_probas is None
            else s_sim_differential_probas[agent_slice]
        )

        data = [*agent_inputs]
        if self.clf_loss_complete_data_flag:
            s_sim_patient = buffer_to((samples.sim_patient,), device=self.agent.device)
            sim_patient = s_sim_patient[agent_slice]

            # concatenate sample and real simulated patient symptoms
            data[0] = torch.cat([data[0], sim_patient], dim=1)
            data[1] = torch.cat([data[1], data[1]], dim=1)
            data[2] = torch.cat([data[2], data[2]], dim=1)

            target = torch.cat([target, target], dim=1)
            timestep = (
                None if timestep is None else torch.cat([timestep, timestep], dim=1)
            )
            differential_indices = (
                None
                if differential_indices is None
                else torch.cat([differential_indices, differential_indices], dim=1)
            )
            differential_probas = (
                None
                if differential_probas is None
                else torch.cat([differential_probas, differential_probas], dim=1)
            )
            done = torch.cat([done, torch.ones_like(done)], dim=1)
            if init_rnn_state is not None:
                init_rnn_state = (
                    init_rnn_state.__class__(
                        *[torch.cat([c, c], dim=1) for c in init_rnn_state]
                    )
                    if isinstance(init_rnn_state, (tuple, list))
                    else torch.cat([init_rnn_state, init_rnn_state], dim=1)
                )

        pis, _ = self.agent.classify(*data, init_rnn_state)
        pis = pis.transpose(1, -1)
        differential_indices = (
            None
            if differential_indices is None
            else differential_indices.transpose(1, -1)
        )
        differential_probas = (
            None
            if differential_probas is None
            else differential_probas.transpose(1, -1)
        )

        if not self.clf_loss_only_at_end_episode_flag:
            loss, loss_dict = self.clf_loss_factory.evaluate(
                self.clf_loss_func,
                pis,
                target,
                differential_indices=differential_indices,
                differential_probas=differential_probas,
                severity=self.patho_severity,
                timestep=timestep,
                **self.clf_loss_kwargs,
            )
        else:
            loss, loss_dict = self.clf_loss_factory.evaluate(
                self.clf_loss_func,
                pis,
                target,
                differential_indices=differential_indices,
                differential_probas=differential_probas,
                reduction="none",
                severity=self.patho_severity,
                timestep=timestep,
                **self.clf_loss_kwargs,
            )
            loss *= done.float()
            sum_done = done.float().sum(dim=0)
            max_sum_done = torch.max(sum_done, torch.ones_like(sum_done))
            loss = torch.sum(loss, dim=0) / max_sum_done
            done_mask = sum_done.gt(0)
            num_elts = done_mask.float().sum().item()
            loss = torch.sum(loss) / max(num_elts, 1)
            if loss_dict:
                for k in loss_dict:
                    loss_dict[k] *= done.float()
                    loss_dict[k] = torch.sum(loss_dict[k], dim=0) / max_sum_done
                    loss_dict[k] = torch.sum(loss_dict[k]) / max(num_elts, 1)
        self.clf_loss_stats = self._update_clf_loss_stats(
            self.clf_loss_stats, loss, loss_dict
        )
        return loss

    def _compute_discounted_auxiliary_rewards(
        self, aux_reward, clf_reward, steps_done, all_done_n, intermediate_flag
    ):
        """Computes the nstep discounted return for auxiliary rewards.

        Parameters
        ----------
        aux_reward: tensor
            auxiliary rewards induced by reward shaping.
        clf_reward: tensor
            auxiliary rewards induced by classification errors.
        steps_done: tensor
            signal indicating if a trajectory ends or not.
        all_done_n: tensor
            signal indicating if a trajectory ends at any future time
            before the n-step or not.
        intermediate_flag: bool
            flag indicating the presence of intermediate data or not.

        Return
        ----------
        discounted_aux_return: tensor
            the discounted return for auxiliary rewards induced by reward shaping.
        discounted_clf_return: tensor
            the discounted return for auxiliary rewards induced by classification
            errors.

        """
        if aux_reward is None and clf_reward is None:
            return 0.0, 0.0

        # compute discounted rewards
        discount = self.discount
        wT, bT, nsr = self.warmup_T, self.batch_T, self.n_step_return

        discounted_aux_return = aux_reward[0] if aux_reward is not None else 0
        discounted_clf_return = clf_reward[0] if clf_reward is not None else 0

        # done_n is used to get n-step done signals, which is
        # True if `done=True` at any future time before the n-step
        # target bootstrap
        done_n = steps_done[0] if intermediate_flag else all_done_n[wT : wT + bT]

        val_step = nsr if intermediate_flag else 1

        # make sure we consider this reward if the current state is not done
        discounted_aux_return *= 1 - done_n.float()
        # make sure we consider this reward only if the current state is done
        discounted_clf_return *= done_n.float()

        for n in range(1, val_step):
            next_aux_reward = aux_reward[n] if aux_reward is not None else 0
            next_clf_reward = clf_reward[n] if clf_reward is not None else 0

            int_done = steps_done[n]
            # make sure we consider this reward if the current state is not done
            next_aux_reward *= 1 - int_done.float()
            # make sure we consider this reward if the current state is done
            next_clf_reward *= int_done.float()
            done_n_f = done_n.float()

            discounted_aux_return += (discount ** n) * next_aux_reward * (1 - done_n_f)
            discounted_clf_return += (discount ** n) * next_clf_reward * (1 - done_n_f)

            done_n = torch.max(done_n, int_done)

        return discounted_aux_return, discounted_clf_return

    def _add_auxiliary_rewards(self, samples):
        """Integrates auxiliary rewards (reward shaping) into the environment rewards.

        This method adds the auxiliary rewards into the classical rewards
        returned by the environment. The auxiliary rewards are based on
        potential reward shaping functions or the classification errors. Here, the
        discounted auxiliary reward is computed and added to the discounted rewards from
        the environment. All these operations are performed on sample data from the
        sequential replay buffer.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which
            the auxiliary reward is added.

        Return
        ----------
        samples: obj
            the altered sample data in which the auxiliary rewards
            have been integrated.

        """
        # clear previous stats
        self.clf_reward_stats.clear()
        self.reward_shaping_stats.clear()

        # transfer data to the device
        all_observation, all_action, all_reward, s_init_rnn_state = buffer_to(
            (
                samples.all_observation,
                samples.all_action,
                samples.all_reward,
                samples.init_rnn_state,
            ),
            device=self.agent.device,
        )
        wT, bT, nsr = self.warmup_T, self.batch_T, self.n_step_return
        init_rnn_state = self._warm_up_rnn_states(
            all_observation, all_action, all_reward, samples.done, s_init_rnn_state
        )

        # check if we have intermediate data
        int_data_flag = getattr(self.replay_buffer, "intermediate_data_flag", False)
        val_step = nsr if int_data_flag else 1

        # initialize delta time step
        if self.delta_timestep is None:
            self.delta_timestep = (
                torch.arange(val_step, device=self.agent.device)
                .unsqueeze(1)
                .unsqueeze(2)
            )

        # now we start the computation of auxiliary rewards
        # all the previous code was inspired by how the r2d1 algo is boostraped
        # in rlpyt
        all_pis = []
        with _get_context(self.reward_shaping_back_propagate_flag):
            all_slice = slice(wT, wT + bT + nsr)
            tmp_inputs = AgentInputs(
                observation=all_observation[all_slice],
                prev_action=all_action[all_slice],
                prev_reward=all_reward[all_slice],
            )
            pis, _ = self.agent.classify(*tmp_inputs, init_rnn_state)
            for i in range(nsr + 1):
                if not int_data_flag and (i > 0 and i < nsr):
                    continue
                all_pis.append(pis[i : i + bT].unsqueeze(0))  # keep the next bT elts.

            # cat the obtained distributions
            all_pis = torch.cat(all_pis, dim=0)

            # obtained the env returns, done signals as well as sim_pathos
            [
                all_returns,
                all_done,
                all_done_n,
                all_sim_patho,
                all_timesteps,
                all_differential_indices,
                all_differential_probas,
                all_evidences,
            ] = buffer_to(
                (
                    samples.return_,
                    samples.done,
                    samples.done_n,
                    samples.sim_patho,
                    samples.sim_timestep,
                    samples.sim_differential_indices,
                    samples.sim_differential_probas,
                    samples.sim_evidence,
                ),
                device=self.agent.device,
            )
            # restrict the value to the ones corresponding to the state being evaluated
            return_ = all_returns[wT : wT + bT]
            s_done_n = all_done_n[wT : wT + bT]
            s_sim_patho = (
                all_sim_patho[wT : wT + bT] if all_sim_patho is not None else None
            )
            s_timestep = (
                all_timesteps[wT : wT + bT] if all_timesteps is not None else None
            )
            s_sim_differential_indices = (
                None
                if all_differential_indices is None
                else all_differential_indices[wT : wT + bT]
            )
            s_sim_differential_probas = (
                None
                if all_differential_probas is None
                else all_differential_probas[wT : wT + bT]
            )
            s_evidence = (
                all_evidences[wT : wT + bT] if all_evidences is not None else None
            )

            # compute the auxiliary reward
            sim_patho = s_sim_patho
            if sim_patho is not None:
                sim_size = sim_patho.size()
                sim_patho = sim_patho.unsqueeze(0)
                sim_patho = sim_patho.expand(val_step, *sim_size)

            sim_differential_indices = s_sim_differential_indices
            dind_flag = sim_differential_indices is not None
            d_size = sim_differential_indices.size() if dind_flag else None
            sim_differential_indices = (
                sim_differential_indices.unsqueeze(0).expand(val_step, *d_size)
                if dind_flag
                else None
            )

            sim_differential_probas = s_sim_differential_probas
            dprob_flag = sim_differential_probas is not None
            d_size = sim_differential_probas.size() if dprob_flag else None
            sim_differential_probas = (
                sim_differential_probas.unsqueeze(0).expand(val_step, *d_size)
                if dprob_flag
                else None
            )

            evidence = s_evidence
            evid_flag = evidence is not None
            d_size = evidence.size() if evid_flag else None
            evidence = (
                evidence.unsqueeze(0).expand(val_step, *d_size) if evid_flag else None
            )

            timestep = s_timestep
            ts_flag = timestep is not None
            ts_size = timestep.size() if ts_flag else None
            timestep = (
                timestep.unsqueeze(0).expand(val_step, *ts_size) + self.delta_timestep
                if ts_flag
                else None
            )

            aux_reward = None
            clf_reward = None

            steps_done = []
            for n in range(val_step):
                done = all_done[wT + n : wT + n + bT]
                steps_done.append(done.unsqueeze(0))
            steps_done = torch.cat(steps_done, dim=0).bool()

            # resulting auxiliary reward initialized with zeros
            result = torch.zeros(*steps_done.size(), device=self.agent.device)

            if self.reward_shaping_flag and not (steps_done.all()):
                next_pis = all_pis[1:]
                prev_pis = all_pis[0:-1]
                diff_indices = (
                    None
                    if sim_differential_indices is None
                    else sim_differential_indices[~steps_done]
                )
                diff_probas = (
                    None
                    if sim_differential_probas is None
                    else sim_differential_probas[~steps_done]
                )
                aux_reward_support, aux_rew_dic = self.reward_shaping_factory.evaluate(
                    self.reward_shaping_func,
                    next_pis[~steps_done],
                    prev_pis[~steps_done],
                    None if sim_patho is None else sim_patho[~steps_done],
                    differential_indices=diff_indices,
                    differential_probas=diff_probas,
                    evidence=None if evidence is None else evidence[~steps_done],
                    discount=self.discount,
                    severity=self.patho_severity,
                    timestep=None if timestep is None else timestep[~steps_done],
                    **self.reward_shaping_kwargs,
                )

                # clamp it if needed
                aux_reward_support = _clamp_utils(
                    aux_reward_support, self.reward_shaping_min, self.reward_shaping_max
                )

                # compute raw aux reward statistics
                self.reward_shaping_stats = self._update_auxiliary_rew_stats(
                    self.reward_shaping_stats,
                    aux_reward_support,
                    aux_rew_dic,
                    "aux_rew",
                )

                # save it
                result[~steps_done] = aux_reward_support
                # Only consider this reward when the current state is not done
                aux_reward = result * (1.0 - steps_done.float())

            is_on_simpatho = sim_patho is not None
            is_on_differential = (sim_differential_indices is not None) and (
                sim_differential_probas is not None
            )
            is_on_clf = is_on_simpatho or is_on_differential
            if self.clf_reward_flag and is_on_clf and (steps_done.any()):
                # classification loss (e.g., negative cross entropy)
                tmp_pis = all_pis[0:-1]
                diff_indices = (
                    None
                    if sim_differential_indices is None
                    else sim_differential_indices[steps_done].transpose(1, -1)
                )
                diff_probas = (
                    None
                    if sim_differential_probas is None
                    else sim_differential_probas[steps_done].transpose(1, -1)
                )
                clf_reward_support, clf_rew_dic = self.clf_loss_factory.evaluate(
                    self.clf_reward_func,
                    tmp_pis[steps_done].transpose(1, -1),
                    sim_patho[steps_done],
                    differential_indices=diff_indices,
                    differential_probas=diff_probas,
                    reduction="none",
                    severity=self.patho_severity,
                    timestep=None if timestep is None else timestep[steps_done],
                    **self.clf_reward_kwargs,
                )
                clf_reward_support = _negate_tensor(clf_reward_support)
                clf_rew_dic = _negate_tensor(clf_rew_dic)

                # clamp it if needed
                clf_reward_support = _clamp_utils(
                    clf_reward_support, self.clf_reward_min, self.clf_reward_max
                )

                # compute raw aux reward statistics
                self.clf_reward_stats = self._update_auxiliary_rew_stats(
                    self.clf_reward_stats, clf_reward_support, clf_rew_dic, "clf_rew"
                )

                # save it
                result[steps_done] = clf_reward_support
                # Only consider this reward when the current state is done
                clf_reward = result * steps_done.float()

            tmp_done_n = (
                s_done_n.bool()
                if self.min_turn_ratio is None
                else (
                    s_done_n.bool()
                    | (tmp_inputs.observation[-bT:, :, 0] < self.min_turn_ratio)
                )
            )
            # resulting clf reward on target initialized with zeros
            self.clf_rew_tgt = torch.zeros(*tmp_done_n.size(), device=self.agent.device)
            self.clf_rew_tgt_mask = tmp_done_n
            if self.clf_reward_flag and is_on_clf and not (tmp_done_n.all()):
                tmp_pis = all_pis[-1]
                diff_indices = (
                    None
                    if sim_differential_indices is None
                    else sim_differential_indices[-1][~tmp_done_n].transpose(1, -1)
                )
                diff_probas = (
                    None
                    if sim_differential_probas is None
                    else sim_differential_probas[-1][~tmp_done_n].transpose(1, -1)
                )
                tmp_pathos = None if sim_patho is None else sim_patho[-1][~tmp_done_n]
                tmp_timestep = (
                    None if timestep is None else timestep[-1][~tmp_done_n] + 1
                )
                clf_reward_on_target, _ = self.clf_loss_factory.evaluate(
                    self.clf_reward_func,
                    tmp_pis[~tmp_done_n].transpose(1, -1),
                    tmp_pathos,
                    differential_indices=diff_indices,
                    differential_probas=diff_probas,
                    reduction="none",
                    severity=self.patho_severity,
                    timestep=tmp_timestep,
                    **self.clf_reward_kwargs,
                )
                clf_reward_on_target = _negate_tensor(clf_reward_on_target)

                # clamp it if needed
                clf_reward_on_target = _clamp_utils(
                    clf_reward_on_target, self.clf_reward_min, self.clf_reward_max
                )
                # save it
                self.clf_rew_tgt[~tmp_done_n] = clf_reward_on_target

            # compute discounted rewards
            tmp_output = self._compute_discounted_auxiliary_rewards(
                aux_reward, clf_reward, steps_done, all_done_n, int_data_flag
            )
            discounted_aux_return, discounted_clf_return = tmp_output

            # combined both rewards
            weighted_aux_return = self.reward_shaping_coef * discounted_aux_return
            weighted_cla_return = (
                self.env_reward_coef * return_
                + self.clf_reward_coef * discounted_clf_return
            )
            final_return = weighted_aux_return + weighted_cla_return

        # modify the return in samples
        tmp_dict = dict(samples.items())
        all_ret = []
        if wT > 0:
            all_ret.append(all_returns[0:wT])
        all_ret.append(final_return)
        all_ret.append(all_returns[wT + bT :])
        modified_return = torch.cat(all_ret, dim=0)
        tmp_dict["return_"] = modified_return
        # set the transfered data in order to avoid transferring twice
        tmp_dict["done"] = all_done
        tmp_dict["done_n"] = all_done_n
        tmp_dict["all_observation"] = all_observation
        tmp_dict["all_action"] = all_action
        tmp_dict["all_reward"] = all_reward
        tmp_dict["sim_patho"] = all_sim_patho
        tmp_dict["sim_differential_indices"] = all_differential_indices
        tmp_dict["sim_differential_probas"] = all_differential_probas
        tmp_dict["sim_timestep"] = all_timesteps
        tmp_dict["sim_evidence"] = all_evidences
        samples = samples.__class__(**tmp_dict)

        return samples

    def _compute_exit_loss_on_target(self, samples):
        """Computes the Q-learning loss assuming the exit token has been chosen.

        This method is specific to Mixed DQN Algo. It defines the loss
        while integrating eventually the auxilirary reward based on reward shaping
        assuming the exit token has been chosen. This methods mainly focuses on
        samples for which the `done_n` signal is not True.

        Parameters
        ----------
        samples: obj
            the sample data from the replay memory against which the loss is computed.

        Return
        ----------
        loss: obj
            the computed loss information.

        """
        if not hasattr(self, "clf_rew_tgt"):
            return 0.0
        if not getattr(self.agent.model, "use_stop_action", True):
            return 0.0

        wT, bT = self.warmup_T, self.batch_T
        s_done = samples.done[wT : wT + bT]
        return_ = samples.return_[wT : wT + bT]
        is_weights = None if not self.prioritized_replay else samples.is_weights
        is_weights = (
            None if is_weights is None else is_weights.unsqueeze(0).expand(bT, -1)
        )

        # tranfer data into agent device
        (is_weights,) = buffer_to((is_weights,), device=self.agent.device)

        # update the return with the computed classifier reward on targets
        return_ += (self.discount ** self.n_step_return) * self.clf_rew_tgt

        # get the saved q from the agent
        q = self.agent.saved_q
        # q_values on exit token
        q_exit = q[..., -1]  # T x B

        tmp_done_n = self.clf_rew_tgt_mask
        if not (tmp_done_n.all()):
            loss_info = self._get_loss_values(
                q_exit[~tmp_done_n],
                s_done[~tmp_done_n],
                None if is_weights is None else is_weights[~tmp_done_n],
                self.value_scale(return_[~tmp_done_n]),
            )
            return loss_info[0]
        else:
            return 0.0
