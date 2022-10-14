from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.algos.dqn.r2d1 import R2D1, PrioritiesSamplesToBuffer, SamplesToBufferRnn
from rlpyt.algos.pg.ppo import PPO, LossInputs
from rlpyt.utils.collections import namedarraytuple

from chloe.utils.reward_shaping_components import (
    MixedDQNRewardShapingLossMixin,
    MixedDQNRewardShapingNonSeqMixin,
    MixedDQNRewardShapingSeqMixin,
    PretrainClassifierMixin,
    RebuildDQNLossMixin,
)

AugSamplesToBuffer = namedarraytuple(
    "AugSamplesToBuffer",
    SamplesToBuffer._fields
    + (
        "sim_patho",
        "sim_patient",
        "sim_severity",
        "sim_evidence",
        "sim_timestep",
        "sim_differential_indices",
        "sim_differential_probas",
    ),
)
"""This class overloads `SamplesToBuffer` with relevant data.

    The data added are the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the simulated patient experiencing the inquired\
        action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the simulated\
        differential
        - sim_differential_probas: the probas of pathos within the simulated\
        differential

"""
AugSamplesToBufferRnn = namedarraytuple(
    "AugSamplesToBufferRnn",
    SamplesToBufferRnn._fields
    + (
        "sim_patho",
        "sim_patient",
        "sim_severity",
        "sim_evidence",
        "sim_timestep",
        "sim_differential_indices",
        "sim_differential_probas",
    ),
)
"""This class overloads `SamplesToBufferRnn` with relevant data.

    The data added are the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the simulated patient experiencing the inquired\
        action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the simulated\
        differential
        - sim_differential_probas: the probas of pathos within the simulated\
        differential

"""
AugLossInputs = namedarraytuple(
    "AugLossInputs",
    LossInputs._fields
    + (
        "sim_patho",
        "sim_patient",
        "sim_severity",
        "sim_evidence",
        "sim_timestep",
        "sim_differential_indices",
        "sim_differential_probas",
    ),
)
"""This class overloads PPO `LossInputs` with relevant data.

    The data added are the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the simulated patient experiencing the inquired\
        action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the simulated\
        differential
        - sim_differential_probas: the probas of pathos within the simulated\
        differential

"""


class DQNReplayBufferMixin:
    """Mixin class to properly handle the use of Replay buffer in DQN-like Algos.

    This class overloads the `samples_to_buffer` and `examples_to_buffer` methods
    to retrieve additional information from the data that need to be stored in the
    replay buffer. The extra data includes the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
    """

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer.

        This method is called in optimize_agent() if samples are provided
        to that method. In asynchronous mode, it will be called in the
        memory_copier process.

        It retrieves additional information from the data that need to be stored in the
        replay buffer. The extra data includes the following:
            - sim_patho: the simulated pathology
            - sim_patient: the simulated patient
            - sim_severity: the severity of the simulated pathology
            - sim_evidence: indicator of the patient experiencing the inquired action
            - sim_timestep: the timestep in the interaction session
            - sim_differential_indices: the indices of pathos within the differential
            - sim_differential_probas: the probas of pathos within the differential

        Parameters
        ----------
        samples: obj
            the data to be added to the replay buffer.

        Return
        ----------
        result: obj
            the formatted data, including the pathology as well as the real
            symptoms being simulated, to be added to the replay buffer.

        """
        data = super().samples_to_buffer(samples)
        sim_patho = getattr(samples.env.env_info, "sim_patho", None)
        sim_patient = getattr(samples.env.env_info, "sim_patient", None)
        sim_severity = getattr(samples.env.env_info, "sim_severity", None)
        sim_evidence = getattr(samples.env.env_info, "sim_evidence", None)
        sim_timestep = getattr(samples.env.env_info, "sim_timestep", None)
        sim_differential_indices = getattr(
            samples.env.env_info, "sim_differential_indices", None
        )
        sim_differential_probas = getattr(
            samples.env.env_info, "sim_differential_probas", None
        )
        if isinstance(data, SamplesToBuffer):
            return AugSamplesToBuffer(
                *data,
                sim_patho=sim_patho,
                sim_patient=sim_patient,
                sim_severity=sim_severity,
                sim_evidence=sim_evidence,
                sim_timestep=sim_timestep,
                sim_differential_indices=sim_differential_indices,
                sim_differential_probas=sim_differential_probas,
            )
        elif isinstance(data, SamplesToBufferRnn):
            return AugSamplesToBufferRnn(
                *data,
                sim_patho=sim_patho,
                sim_patient=sim_patient,
                sim_severity=sim_severity,
                sim_evidence=sim_evidence,
                sim_timestep=sim_timestep,
                sim_differential_indices=sim_differential_indices,
                sim_differential_probas=sim_differential_probas,
            )
        elif isinstance(data, PrioritiesSamplesToBuffer):
            priorities = data.priorities
            samples = data.samples
            if isinstance(samples, SamplesToBuffer):
                samples = AugSamplesToBuffer(
                    *samples,
                    sim_patho=sim_patho,
                    sim_patient=sim_patient,
                    sim_severity=sim_severity,
                    sim_evidence=sim_evidence,
                    sim_timestep=sim_timestep,
                    sim_differential_indices=sim_differential_indices,
                    sim_differential_probas=sim_differential_probas,
                )
            elif isinstance(samples, SamplesToBufferRnn):
                samples = AugSamplesToBufferRnn(
                    *samples,
                    sim_patho=sim_patho,
                    sim_patient=sim_patient,
                    sim_severity=sim_severity,
                    sim_evidence=sim_evidence,
                    sim_timestep=sim_timestep,
                    sim_differential_indices=sim_differential_indices,
                    sim_differential_probas=sim_differential_probas,
                )
            return PrioritiesSamplesToBuffer(priorities=priorities, samples=samples)
        else:
            return data

    def examples_to_buffer(self, examples):
        """Defines how to initialize the replay buffer from examples.

        This method is called in initialize_replay_buffer().

        It retrieves additional information from the data that need to be stored in the
        replay buffer. The extra data includes the following:
            - sim_patho: the simulated pathology
            - sim_patient: the simulated patient
            - sim_severity: the severity of the simulated pathology
            - sim_evidence: indicator of the patient experiencing the inquired action
            - sim_timestep: the timestep in the interaction session
            - sim_differential_indices: the indices of pathos within the differential
            - sim_differential_probas: the probas of pathos within the differential

        Parameters
        ----------
        examples: obj
            the examples of data to be added to the replay buffer.

        Return
        ----------
        result: obj
            the formatted data, including the pathology as well as the real
            symptoms being simulated, to be added to the replay buffer.

        """
        data = super().examples_to_buffer(examples)
        sim_patho = getattr(examples["env_info"], "sim_patho", None)
        sim_patient = getattr(examples["env_info"], "sim_patient", None)
        sim_severity = getattr(examples["env_info"], "sim_severity", None)
        sim_evidence = getattr(examples["env_info"], "sim_evidence", None)
        sim_timestep = getattr(examples["env_info"], "sim_timestep", None)
        sim_differential_indices = getattr(
            examples["env_info"], "sim_differential_indices", None
        )
        sim_differential_probas = getattr(
            examples["env_info"], "sim_differential_probas", None
        )
        if isinstance(data, SamplesToBuffer):
            return AugSamplesToBuffer(
                *data,
                sim_patho=sim_patho,
                sim_patient=sim_patient,
                sim_severity=sim_severity,
                sim_evidence=sim_evidence,
                sim_timestep=sim_timestep,
                sim_differential_indices=sim_differential_indices,
                sim_differential_probas=sim_differential_probas,
            )
        elif isinstance(data, SamplesToBufferRnn):
            return AugSamplesToBufferRnn(
                *data,
                sim_patho=sim_patho,
                sim_patient=sim_patient,
                sim_severity=sim_severity,
                sim_evidence=sim_evidence,
                sim_timestep=sim_timestep,
                sim_differential_indices=sim_differential_indices,
                sim_differential_probas=sim_differential_probas,
            )
        elif isinstance(data, PrioritiesSamplesToBuffer):
            priorities = data.priorities
            samples = data.samples
            if isinstance(samples, SamplesToBuffer):
                samples = AugSamplesToBuffer(
                    *samples,
                    sim_patho=sim_patho,
                    sim_patient=sim_patient,
                    sim_severity=sim_severity,
                    sim_evidence=sim_evidence,
                    sim_timestep=sim_timestep,
                    sim_differential_indices=sim_differential_indices,
                    sim_differential_probas=sim_differential_probas,
                )
            elif isinstance(samples, SamplesToBufferRnn):
                samples = AugSamplesToBufferRnn(
                    *samples,
                    sim_patho=sim_patho,
                    sim_patient=sim_patient,
                    sim_severity=sim_severity,
                    sim_evidence=sim_evidence,
                    sim_timestep=sim_timestep,
                    sim_differential_indices=sim_differential_indices,
                    sim_differential_probas=sim_differential_probas,
                )
            return PrioritiesSamplesToBuffer(priorities=priorities, samples=samples)
        else:
            return data


class PPOLossInputMixin:
    """Mixin class to properly handle inputs needed for loss computations in PPO Algos.

    This class overloads the `loss_input_from_samples` method to provide
    additional information that may be useful for loss computation.
    The extra data includes the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
    """

    def loss_input_from_samples(self, samples, agent_inputs, return_, advantage, valid):
        """Defines the input data needed for loss computation.

        This method overloads the `loss_input_from_samples` method.

        It provides additional information that may be useful for loss computation.
        The extra data includes the following:
            - sim_patho: the simulated pathology
            - sim_patient: the simulated patient
            - sim_severity: the severity of the simulated pathology
            - sim_evidence: indicator of the patient experiencing the inquired action
            - sim_timestep: the timestep in the interaction session
            - sim_differential_indices: the indices of pathos within the differential
            - sim_differential_probas: the probas of pathos within the differential

        Parameters
        ----------
        samples: obj
            the samples data as provided by the environments.
        agent_inputs: obj
            the data as provided to the agent as input.
        return_: tensor
            the expeted returns.
        advantage: tensor
            the expeted advantages.
        valid: tensor
            data indicating which samples are valid or not.

        Return
        ----------
        result: obj
            the tuple of data needed for loss computatiion, including the pathology
            as well as the real symptoms being simulated.

        """
        data = super().loss_input_from_samples(
            samples, agent_inputs, return_, advantage, valid
        )
        sim_patho = getattr(samples.env.env_info, "sim_patho", None)
        sim_patient = getattr(samples.env.env_info, "sim_patient", None)
        sim_severity = getattr(samples.env.env_info, "sim_severity", None)
        sim_evidence = getattr(samples.env.env_info, "sim_evidence", None)
        sim_timestep = getattr(samples.env.env_info, "sim_timestep", None)
        sim_differential_indices = getattr(
            samples.env.env_info, "sim_differential_indices", None
        )
        sim_differential_probas = getattr(
            samples.env.env_info, "sim_differential_probas", None
        )
        return AugLossInputs(
            *data,
            sim_patho=sim_patho,
            sim_patient=sim_patient,
            sim_severity=sim_severity,
            sim_evidence=sim_evidence,
            sim_timestep=sim_timestep,
            sim_differential_indices=sim_differential_indices,
            sim_differential_probas=sim_differential_probas,
        )


class AugDQN(DQNReplayBufferMixin, DQN):
    """DQN Algo with custom functions for adding data in the replay buffer.

    The `samples_to_buffer` and `examples_to_buffer` methods are overloaded
    to retrieve additional information from the data that need to be stored in the
    replay buffer. The extra data includes the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
    """

    pass


class AugCategoricalDQN(DQNReplayBufferMixin, CategoricalDQN):
    """CategoricalDQN Algo with custom functions for adding data in the replay buffer.

    The `samples_to_buffer` and `examples_to_buffer` methods are overloaded
    to retrieve additional information from the data that need to be stored in the
    replay buffer. The extra data includes the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
    """

    pass


class AugR2D1(DQNReplayBufferMixin, R2D1):
    """R2D1 Algo with custom functions for adding data in the replay buffer.

    The `samples_to_buffer` and `examples_to_buffer` methods are overloaded
    to retrieve additional information from the data that need to be stored in the
    replay buffer. The extra data includes the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
    """

    pass


class AugPPO(PPOLossInputMixin, PPO):
    """PPO Algo with custom functions for building data needed for loss computation.

    The `loss_input_from_samples` method is overloaded to provide
    additional information that may be useful for loss computation.
    The extra data includes the following:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
    """

    pass


class AugMixedDQN(
    RebuildDQNLossMixin,
    MixedDQNRewardShapingNonSeqMixin,
    MixedDQNRewardShapingLossMixin,
    PretrainClassifierMixin,
    AugDQN,
):
    """DQN Algo with mixed outputs: Q-Values and probability distributions.

    The class extend DQN algorithm in a way to allow dealing with 'classifier
    pretraining', 'reward reshaping', 'classifier-based reward', as well as
    'classifier learning'.
    """

    def __init__(
        self,
        *args,
        replay_intermediate_data_flag=True,
        separate_classifier_optimizer=False,
        clf_learning_rate=None,
        clf_optim_params=None,
        pretrain_flag=True,
        pretrain_validation_percentage=0.25,
        pretrain_epochs=10,
        pretrain_batch_size=32,
        pretrain_clf_learning_rate=None,
        pretrain_perf_metric="Accuracy",
        pretrain_patience=5,
        pretrain_loss_func="cross_entropy",
        pretrain_loss_kwargs=None,
        reward_shaping_flag=False,
        reward_shaping_min=None,
        reward_shaping_max=None,
        reward_shaping_coef=1.0,
        env_reward_coef=1.0,
        reward_shaping_back_propagate_flag=False,
        reward_shaping_func="cross_entropy",
        reward_shaping_kwargs=None,
        clf_reward_flag=False,
        clf_reward_min=None,
        clf_reward_max=None,
        clf_reward_coef=1.0,
        clf_reward_func="cross_entropy",
        clf_reward_kwargs=None,
        clf_loss_flag=False,
        clf_loss_complete_data_flag=False,
        clf_loss_only_at_end_episode_flag=True,
        clf_loss_func="cross_entropy",
        clf_loss_kwargs=None,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
        **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        replay_intermediate_data_flag: bool
            whether or not sampling from the replay buffer will also collect
            intermediate data when `n_step_returns > 1`. Default: True
        separate_classifier_optimizer: bool
            whether or not the classifier has its own optimizer. Default: False
        clf_learning_rate: float
            the learning rate to use during the classifier training. if None,
            the learning rate of the agent will be used. Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        clf_optim_params: dict
            the params for initializing the classifier optimizer.  Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        pretrain_flag: bool
            whether or not to pretrain the classifier. Default: True
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
        pretrain_perf_metric: str
            the performance metric used for pretraining. Default: Accuracy
        pretrain_patience: int
            the authorized patience for early stopping when pretraining. Default: 5
        pretrain_loss_func: str
            the loss function used for pretraining. Default: cross_entropy.
        pretrain_loss_kwargs: dict
            variable arguments for the pretrain loss function. Default: None
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

        """
        super().__init__(*args, **kwargs)
        self.set_pretrain_params(
            pretrain_flag,
            separate_classifier_optimizer,
            pretrain_validation_percentage,
            pretrain_epochs,
            pretrain_batch_size,
            pretrain_clf_learning_rate,
            clf_learning_rate,
            clf_optim_params,
            pretrain_perf_metric,
            pretrain_patience,
            pretrain_loss_func,
            pretrain_loss_kwargs,
        )
        self.set_reward_shaping_params(
            reward_shaping_flag,
            reward_shaping_min,
            reward_shaping_max,
            reward_shaping_coef,
            env_reward_coef,
            reward_shaping_back_propagate_flag,
            reward_shaping_func,
            reward_shaping_kwargs,
        )
        self.set_clf_reward_params(
            clf_reward_flag,
            clf_reward_min,
            clf_reward_max,
            clf_reward_coef,
            clf_reward_func,
            clf_reward_kwargs,
        )
        self.set_clf_loss_params(
            clf_loss_flag,
            clf_loss_complete_data_flag,
            clf_loss_only_at_end_episode_flag,
            clf_loss_func,
            clf_loss_kwargs,
        )
        self.set_feature_rebuild_loss_params(
            feature_rebuild_loss_flag,
            feature_rebuild_loss_min,
            feature_rebuild_loss_max,
            feature_rebuild_loss_coef,
            feature_rebuild_loss_func,
            feature_rebuild_loss_kwargs,
        )
        self.replay_intermediate_data_flag = replay_intermediate_data_flag


class AugMixedCategoricalDQN(
    RebuildDQNLossMixin,
    MixedDQNRewardShapingNonSeqMixin,
    MixedDQNRewardShapingLossMixin,
    PretrainClassifierMixin,
    AugCategoricalDQN,
):
    """Categorical DQN Algo with mixed outputs: Q-Values and probability distributions.

    The class extend Categorical DQN algorithm in a way to allow dealing with
    'classifier pretraining', 'reward reshaping', 'classifier-based reward',
    as well as 'classifier learning'.
    """

    def __init__(
        self,
        *args,
        replay_intermediate_data_flag=True,
        separate_classifier_optimizer=False,
        clf_learning_rate=None,
        clf_optim_params=None,
        pretrain_flag=True,
        pretrain_validation_percentage=0.25,
        pretrain_epochs=10,
        pretrain_batch_size=32,
        pretrain_clf_learning_rate=None,
        pretrain_perf_metric="Accuracy",
        pretrain_patience=5,
        pretrain_loss_func="cross_entropy",
        pretrain_loss_kwargs=None,
        reward_shaping_flag=False,
        reward_shaping_min=None,
        reward_shaping_max=None,
        reward_shaping_coef=1.0,
        env_reward_coef=1.0,
        reward_shaping_back_propagate_flag=False,
        reward_shaping_func="cross_entropy",
        reward_shaping_kwargs=None,
        clf_reward_flag=False,
        clf_reward_min=None,
        clf_reward_max=None,
        clf_reward_coef=1.0,
        clf_reward_func="cross_entropy",
        clf_reward_kwargs=None,
        clf_loss_flag=False,
        clf_loss_complete_data_flag=False,
        clf_loss_only_at_end_episode_flag=True,
        clf_loss_func="cross_entropy",
        clf_loss_kwargs=None,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
        **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        replay_intermediate_data_flag: bool
            whether or not sampling from the replay buffer will also collect
            intermediate data when `n_step_returns > 1`. Default: True
        separate_classifier_optimizer: bool
            whether or not the classifier has its own optimizer. Default: False
        clf_learning_rate: float
            the learning rate to use during the classifier training. if None,
            the learning rate of the agent will be used. Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        clf_optim_params: dict
            the params for initializing the classifier optimizer.  Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        pretrain_flag: bool
            whether or not to pretrain the classifier. Default: True
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
        pretrain_perf_metric: str
            the performance metric used for pretraining. Default: Accuracy
        pretrain_patience: int
            the authorized patience for early stopping when pretraining. Default: 5
        pretrain_loss_func: str
            the loss function used for pretraining. Default: cross_entropy
        pretrain_loss_kwargs: dict
            variable arguments for the pretrain loss function. Default: None
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
        clf_loss_flag: bool
            whether or not to perform classification loss update. Default: False
        clf_loss_complete_data_flag: bool
            whether or not to consider the complete simulated patients  when
            computing classification loss. Default: False
        clf_loss_only_at_end_episode_flag: bool
            whether or not to provide classification error signals
            at end of episode. Default: True.
        clf_loss_func: str
            the loss function used for training. Default: cross_entropy
        clf_loss_kwargs: dict
            variable arguments for the classifier loss function. Default: None
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

        """
        super().__init__(*args, **kwargs)
        self.set_pretrain_params(
            pretrain_flag,
            separate_classifier_optimizer,
            pretrain_validation_percentage,
            pretrain_epochs,
            pretrain_batch_size,
            pretrain_clf_learning_rate,
            clf_learning_rate,
            clf_optim_params,
            pretrain_perf_metric,
            pretrain_patience,
            pretrain_loss_func,
            pretrain_loss_kwargs,
        )
        self.set_reward_shaping_params(
            reward_shaping_flag,
            reward_shaping_min,
            reward_shaping_max,
            reward_shaping_coef,
            env_reward_coef,
            reward_shaping_back_propagate_flag,
            reward_shaping_func,
            reward_shaping_kwargs,
        )
        self.set_clf_reward_params(
            clf_reward_flag,
            clf_reward_min,
            clf_reward_max,
            clf_reward_coef,
            clf_reward_func,
            clf_reward_kwargs,
        )
        self.set_clf_loss_params(
            clf_loss_flag,
            clf_loss_complete_data_flag,
            clf_loss_only_at_end_episode_flag,
            clf_loss_func,
            clf_loss_kwargs,
        )
        self.set_feature_rebuild_loss_params(
            feature_rebuild_loss_flag,
            feature_rebuild_loss_min,
            feature_rebuild_loss_max,
            feature_rebuild_loss_coef,
            feature_rebuild_loss_func,
            feature_rebuild_loss_kwargs,
        )
        self.replay_intermediate_data_flag = replay_intermediate_data_flag


class AugMixedR2D1(
    RebuildDQNLossMixin,
    MixedDQNRewardShapingSeqMixin,
    MixedDQNRewardShapingLossMixin,
    PretrainClassifierMixin,
    AugR2D1,
):
    """R2D1 Algo with mixed outputs: Q-Values and probability distributions.

    The class extend R2D1 algorithm in a way to allow dealing with
    'classifier pretraining', 'reward reshaping', 'classifier-based reward',
    as well as 'classifier learning'.
    """

    def __init__(
        self,
        *args,
        replay_intermediate_data_flag=True,
        separate_classifier_optimizer=False,
        clf_learning_rate=None,
        clf_optim_params=None,
        pretrain_flag=True,
        pretrain_validation_percentage=0.25,
        pretrain_epochs=10,
        pretrain_batch_size=32,
        pretrain_clf_learning_rate=None,
        pretrain_perf_metric="Accuracy",
        pretrain_patience=5,
        pretrain_loss_func="cross_entropy",
        pretrain_loss_kwargs=None,
        reward_shaping_flag=False,
        reward_shaping_min=None,
        reward_shaping_max=None,
        reward_shaping_coef=1.0,
        env_reward_coef=1.0,
        reward_shaping_back_propagate_flag=False,
        reward_shaping_func="cross_entropy",
        reward_shaping_kwargs=None,
        clf_reward_flag=False,
        clf_reward_min=None,
        clf_reward_max=None,
        clf_reward_coef=1.0,
        clf_reward_func="cross_entropy",
        clf_reward_kwargs=None,
        clf_loss_flag=False,
        clf_loss_complete_data_flag=False,
        clf_loss_only_at_end_episode_flag=True,
        clf_loss_func="cross_entropy",
        clf_loss_kwargs=None,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
        **kwargs,
    ):
        """Instantiates a class object.

        Parameters
        ----------
        replay_intermediate_data_flag: bool
            whether or not sampling from the replay buffer will also collect
            intermediate data when `n_step_returns > 1`. Default: True
        separate_classifier_optimizer: bool
            whether or not the classifier has its own optimizer. Default: False
        clf_learning_rate: float
            the learning rate to use during the classifier training. if None,
            the learning rate of the agent will be used. Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        clf_optim_params: dict
            the params for initializing the classifier optimizer.  Only useful when
            `separate_classifier_optimizer` is set to True. Default: None
        pretrain_flag: bool
            whether or not to pretrain the classifier. Default: True
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
        pretrain_perf_metric: str
            the performance metric used for pretraining. Default: Accuracy
        pretrain_patience: int
            the authorized patience for early stopping when pretraining. Default: 5
        pretrain_loss_func: str
            the loss function used for pretraining. Default: cross_entropy
        pretrain_loss_kwargs: dict
            variable arguments for the pretrain loss function. Default: None
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

        """
        super().__init__(*args, **kwargs)
        self.set_pretrain_params(
            pretrain_flag,
            separate_classifier_optimizer,
            pretrain_validation_percentage,
            pretrain_epochs,
            pretrain_batch_size,
            pretrain_clf_learning_rate,
            clf_learning_rate,
            clf_optim_params,
            pretrain_perf_metric,
            pretrain_patience,
            pretrain_loss_func,
            pretrain_loss_kwargs,
        )
        self.set_reward_shaping_params(
            reward_shaping_flag,
            reward_shaping_min,
            reward_shaping_max,
            reward_shaping_coef,
            env_reward_coef,
            reward_shaping_back_propagate_flag,
            reward_shaping_func,
            reward_shaping_kwargs,
        )
        self.set_clf_reward_params(
            clf_reward_flag,
            clf_reward_min,
            clf_reward_max,
            clf_reward_coef,
            clf_reward_func,
            clf_reward_kwargs,
        )
        self.set_clf_loss_params(
            clf_loss_flag,
            clf_loss_complete_data_flag,
            clf_loss_only_at_end_episode_flag,
            clf_loss_func,
            clf_loss_kwargs,
        )
        self.set_feature_rebuild_loss_params(
            feature_rebuild_loss_flag,
            feature_rebuild_loss_min,
            feature_rebuild_loss_max,
            feature_rebuild_loss_coef,
            feature_rebuild_loss_func,
            feature_rebuild_loss_kwargs,
        )
        self.replay_intermediate_data_flag = replay_intermediate_data_flag


class AugRebuildDQN(
    RebuildDQNLossMixin, AugDQN,
):
    """DQN Algo with mixed outputs: Q-Values and reconstruted features.

    The class extend DQN algorithm in a way to allow dealing with
    'reconstruction learning'.
    """

    def __init__(
        self,
        *args,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
        **kwargs,
    ):
        """Instantiates a class object.

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

        """
        super().__init__(*args, **kwargs)
        self.set_feature_rebuild_loss_params(
            feature_rebuild_loss_flag,
            feature_rebuild_loss_min,
            feature_rebuild_loss_max,
            feature_rebuild_loss_coef,
            feature_rebuild_loss_func,
            feature_rebuild_loss_kwargs,
        )


class AugRebuildCategoricalDQN(
    RebuildDQNLossMixin, AugCategoricalDQN,
):
    """Categorical DQN Algo with mixed outputs: Q-Values and probability distributions.

    The class extend Categorical DQN algorithm in a way to allow dealing with
    'classifier pretraining', 'reward reshaping', 'classifier-based reward',
    as well as 'classifier learning'.
    """

    def __init__(
        self,
        *args,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
        **kwargs,
    ):
        """Instantiates a class object.

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

        """
        super().__init__(*args, **kwargs)
        self.set_feature_rebuild_loss_params(
            feature_rebuild_loss_flag,
            feature_rebuild_loss_min,
            feature_rebuild_loss_max,
            feature_rebuild_loss_coef,
            feature_rebuild_loss_func,
            feature_rebuild_loss_kwargs,
        )


class AugRebuildR2D1(
    RebuildDQNLossMixin, AugR2D1,
):
    """R2D1 Algo with mixed outputs: Q-Values and probability distributions.

    The class extend R2D1 algorithm in a way to allow dealing with
    'classifier pretraining', 'reward reshaping', 'classifier-based reward',
    as well as 'classifier learning'.
    """

    def __init__(
        self,
        *args,
        feature_rebuild_loss_flag=False,
        feature_rebuild_loss_min=None,
        feature_rebuild_loss_max=None,
        feature_rebuild_loss_coef=1.0,
        feature_rebuild_loss_func="bce",
        feature_rebuild_loss_kwargs=None,
        **kwargs,
    ):
        """Instantiates a class object.

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

        """
        super().__init__(*args, **kwargs)
        self.set_feature_rebuild_loss_params(
            feature_rebuild_loss_flag,
            feature_rebuild_loss_min,
            feature_rebuild_loss_max,
            feature_rebuild_loss_coef,
            feature_rebuild_loss_func,
            feature_rebuild_loss_kwargs,
        )
