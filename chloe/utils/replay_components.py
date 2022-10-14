import numpy as np
from rlpyt.agents.base import AgentInputs
from rlpyt.replays.non_sequence.n_step import SamplesFromReplay
from rlpyt.replays.non_sequence.time_limit import SamplesFromReplayTL
from rlpyt.replays.sequence.n_step import SamplesFromReplay as SeqSamplesFromReplay
from rlpyt.utils.buffer import buffer_func, torchify_buffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import extract_sequences

IntermediateSamplesFromReplay = namedarraytuple(
    "IntermediateSamplesFromReplay", ("inputs", "reward", "done")
)
"""Data structure used for intermediate samples.

    It contains the following:
        - input: the state data at intermediate stages
        - reward: the reward data at intermediate stages
        - done: flag indicating if the simulation ended at intermediate states

"""

AugSamplesFromReplay = namedarraytuple(
    "AugSamplesFromReplay",
    SamplesFromReplay._fields
    + (
        "sim_patho",
        "sim_patient",
        "sim_severity",
        "sim_evidence",
        "sim_timestep",
        "sim_differential_indices",
        "sim_differential_probas",
        "intermediate_data",
    ),
)
"""This class overloads `SamplesFromReplay` with relevant data.

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
        - intermediate_data: data for intermediate states

"""

AugSamplesFromReplayTL = namedarraytuple(
    "AugSamplesFromReplayTL",
    SamplesFromReplayTL._fields
    + (
        "sim_patho",
        "sim_patient",
        "sim_severity",
        "sim_evidence",
        "sim_timestep",
        "sim_differential_indices",
        "sim_differential_probas",
        "intermediate_data",
    ),
)
"""This class overloads `SamplesFromReplayTL` with relevant data.

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
        - intermediate_data: data for intermediate states

"""

AugSeqSamplesFromReplay = namedarraytuple(
    "AugSeqSamplesFromReplay",
    SeqSamplesFromReplay._fields
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
"""This class overloads `SeqSamplesFromReplay` with relevant data.

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

AugSamplesFromReplayPri = namedarraytuple(
    "AugSamplesFromReplayPri", AugSamplesFromReplay._fields + ("is_weights",)
)
"""This class overloads prioritized `SamplesFromReplay` with relevant data.

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
        - intermediate_data: data for intermediate states

"""

AugSeqSamplesFromReplayPri = namedarraytuple(
    "AugSeqSamplesFromReplayPri", AugSeqSamplesFromReplay._fields + ("is_weights",)
)
"""This class overloads prioritized `SeqSamplesFromReplay` with relevant data.

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


class AugReplayBufferExtractMixin:
    """Mixin class to properly extract relevant info from replay buffer.

    This class overloads the `extract_batch` method to retrieve additional information
    from the replay buffer such as the `simulated_pathology`, the `simulated_patients`,
    the `simulated_severity` and eventually the `intermediate_states` in cases
    `n_step_return` > 1. The exhaustive list of additional extracted data is:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential
        - intermediate_data: data for intermediate states

    """

    def _set_intermediate_data_flag(self, flag):
        """Sets the flag for extracting intermediate state data from replay buffer.

        Parameters
        ----------
        flag: bool
            the value to be setted.

        Returns
        -------
        None

        """
        self.intermediate_data_flag = flag

    def _get_intermediate_data(self, T_idxs, B_idxs):
        """Method for extracting intermediate state data from replay buffer.

        Parameters
        ----------
        T_idxs: tensor of int
            indices at the `T` dimension.
        B_idxs: tensor of int
            indices at the `B` dimension.

        Returns
        -------
        data: IntermediateSamplesFromReplay
            tuple of extracted info data.

        """
        if self.n_step_return > 1:

            target_T_idxs = T_idxs + self.n_step_return

            intermediate_T_idxs = np.linspace(
                T_idxs + 1, target_T_idxs - 1, num=self.n_step_return - 1, dtype=np.int,
            )
            intermediate_T_idxs = intermediate_T_idxs % self.T

            intermediate_B_idxs = np.tile(B_idxs, self.n_step_return - 1).reshape(
                (self.n_step_return - 1, -1)
            )

            intermediate_T_idxs = intermediate_T_idxs.reshape((-1,))
            intermediate_B_idxs = intermediate_B_idxs.reshape((-1,))

            intermediate_inputs = AgentInputs(
                observation=self.extract_observation(
                    intermediate_T_idxs, intermediate_B_idxs
                ),
                prev_action=self.samples.action[
                    intermediate_T_idxs - 1, intermediate_B_idxs
                ],
                prev_reward=self.samples.reward[
                    intermediate_T_idxs - 1, intermediate_B_idxs
                ],
            )
            intermediate_reward = self.samples.reward[
                intermediate_T_idxs, intermediate_B_idxs
            ]
            intermediate_done = self.samples.done[
                intermediate_T_idxs, intermediate_B_idxs
            ]
            # self.n_step_return - 1 x len(T_idx). reshape => [self.n_step_return -
            # 1, len(T_idx)]

            return IntermediateSamplesFromReplay(
                inputs=intermediate_inputs,
                reward=intermediate_reward,
                done=intermediate_done,
            )

        else:
            return None

    def extract_batch(self, T_idxs, B_idxs):
        """Method for extraction relevant data from replay buffer.

        This function overloads the `extract_batch` method to retrieve additional
        information from the replay buffer such as the `simulated_pathology`,
        the `simulated_patients`, the `simulated_severity` and eventually the
        `intermediate_states` in cases `n_step_return` > 1. The exhaustive list of
        additional extracted data is:
            - sim_patho: the simulated pathology
            - sim_patient: the simulated patient
            - sim_severity: the severity of the simulated pathology
            - sim_evidence: indicator of the patient experiencing the inquired action
            - sim_timestep: the timestep in the interaction session
            - sim_differential_indices: the indices of pathos within the differential
            - sim_differential_probas: the probas of pathos within the differential
            - intermediate_data: data for intermediate states

        Parameters
        ----------
        T_idxs: tensor of int
            indices at the `T` dimension.
        B_idxs: tensor of int
            indices at the `B` dimension.

        Returns
        -------
        data: tuple
            tuple of extracted info data.

        """
        result = super().extract_batch(T_idxs, B_idxs)
        sim_patho = None
        sim_patient = None
        sim_severity = None
        sim_evidence = None
        sim_timestep = None
        sim_differential_indices = None
        sim_differential_probas = None
        intermediate_data = None
        if hasattr(self.samples, "sim_patho") and (self.samples.sim_patho is not None):
            sim_patho = self.samples.sim_patho[T_idxs, B_idxs]
        if hasattr(self.samples, "sim_patient") and (
            self.samples.sim_patient is not None
        ):
            sim_patient = self.samples.sim_patient[T_idxs, B_idxs]
        if hasattr(self.samples, "sim_severity") and (
            self.samples.sim_severity is not None
        ):
            sim_severity = self.samples.sim_severity[T_idxs, B_idxs]
        if hasattr(self.samples, "sim_evidence") and (
            self.samples.sim_evidence is not None
        ):
            sim_evidence = self.samples.sim_evidence[T_idxs, B_idxs]
        if hasattr(self.samples, "sim_timestep") and (
            self.samples.sim_timestep is not None
        ):
            sim_timestep = self.samples.sim_timestep[T_idxs, B_idxs]
        if hasattr(self.samples, "sim_differential_indices") and (
            self.samples.sim_differential_indices is not None
        ):
            sim_differential_indices = self.samples.sim_differential_indices[
                T_idxs, B_idxs
            ]
        if hasattr(self.samples, "sim_differential_probas") and (
            self.samples.sim_differential_probas is not None
        ):
            sim_differential_probas = self.samples.sim_differential_probas[
                T_idxs, B_idxs
            ]
        if hasattr(self, "intermediate_data_flag") and (self.intermediate_data_flag):
            intermediate_data = self._get_intermediate_data(T_idxs, B_idxs)
        if isinstance(result, SamplesFromReplay):
            result = AugSamplesFromReplay(
                *result,
                sim_patho=sim_patho,
                sim_patient=sim_patient,
                sim_severity=sim_severity,
                sim_evidence=sim_evidence,
                sim_timestep=sim_timestep,
                sim_differential_indices=sim_differential_indices,
                sim_differential_probas=sim_differential_probas,
                intermediate_data=intermediate_data,
            )
        else:
            result = AugSamplesFromReplayTL(
                *result,
                sim_patho=sim_patho,
                sim_patient=sim_patient,
                sim_severity=sim_severity,
                sim_evidence=sim_evidence,
                sim_timestep=sim_timestep,
                sim_differential_indices=sim_differential_indices,
                sim_differential_probas=sim_differential_probas,
                intermediate_data=intermediate_data,
            )

        return torchify_buffer(result)

    def weight_batch_samples(self, batch, is_weights):
        """Method for weighting sampled data from prioritized replay buffer.

        Parameters
        ----------
        batch: tuple
            tuple of data representing the sampled batch.
        is_weights: tensor of float
            the weights to be assigned to the batch data.

        Returns
        -------
        data: tuple
            the weighted batch data.

        """
        return AugSamplesFromReplayPri(*batch, is_weights=is_weights)


class AugSeqReplayBufferExtractMixin:
    """Mixin class to properly extract relevant info from sequential replay buffer.

    This class overloads the `extract_batch` method to retrieve additional information
    from the sequential replay buffer such as the `simulated_pathology`,
    the `simulated_patients`, and the `simulated_severity`. The exhaustive list of
    additional extracted data is:
        - sim_patho: the simulated pathology
        - sim_patient: the simulated patient
        - sim_severity: the severity of the simulated pathology
        - sim_evidence: indicator of the patient experiencing the inquired action
        - sim_timestep: the timestep in the interaction session
        - sim_differential_indices: the indices of pathos within the differential
        - sim_differential_probas: the probas of pathos within the differential

    """

    def _set_intermediate_data_flag(self, flag):
        """Sets the flag for extracting intermediate state data from replay buffer.

        Parameters
        ----------
        flag: bool
            the value to be setted.

        Returns
        -------
        None

        """
        self.intermediate_data_flag = flag

    def extract_batch(self, T_idxs, B_idxs, T):
        """Method for extraction relevant data from sequential replay buffer.

        This function overloads the `extract_batch` method to retrieve additional
        information from the replay buffer such as the `simulated_pathology`,
        the `simulated_patients`, the `simulated_severity`. The exhaustive list of
        additional extracted data is:
            - sim_patho: the simulated pathology
            - sim_patient: the simulated patient
            - sim_severity: the severity of the simulated pathology
            - sim_evidence: indicator of the patient experiencing the inquired action
            - sim_timestep: the timestep in the interaction session
            - sim_differential_indices: the indices of pathos within the differential
            - sim_differential_probas: the probas of pathos within the differential

        Parameters
        ----------
        T_idxs: tensor of int
            indices at the `T` dimension.
        B_idxs: tensor of int
            indices at the `B` dimension.
        T: int
            length of the retrieved sample sequence.

        Returns
        -------
        data: tuple
            tuple of extracted info data.

        """
        result = super().extract_batch(T_idxs, B_idxs, T)
        flag = hasattr(self, "intermediate_data_flag") and (self.intermediate_data_flag)
        if flag and self.n_step_return > 1:
            tmp_dict = dict(result.items())
            tmp_dict["done"] = torchify_buffer(
                extract_sequences(
                    self.samples.done, T_idxs, B_idxs, T + self.n_step_return - 1
                )
            )
            result = result.__class__(**tmp_dict)
        sim_patho = None
        sim_patient = None
        sim_severity = None
        sim_evidence = None
        sim_timestep = None
        sim_differential_indices = None
        sim_differential_probas = None
        if hasattr(self.samples, "sim_patho") and (self.samples.sim_patho is not None):
            sim_patho = buffer_func(
                self.samples.sim_patho,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        if hasattr(self.samples, "sim_patient") and (
            self.samples.sim_patient is not None
        ):
            sim_patient = buffer_func(
                self.samples.sim_patient,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        if hasattr(self.samples, "sim_severity") and (
            self.samples.sim_severity is not None
        ):
            sim_severity = buffer_func(
                self.samples.sim_severity,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        if hasattr(self.samples, "sim_evidence") and (
            self.samples.sim_evidence is not None
        ):
            sim_evidence = buffer_func(
                self.samples.sim_evidence,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        if hasattr(self.samples, "sim_timestep") and (
            self.samples.sim_timestep is not None
        ):
            sim_timestep = buffer_func(
                self.samples.sim_timestep,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        if hasattr(self.samples, "sim_differential_indices") and (
            self.samples.sim_differential_indices is not None
        ):
            sim_differential_indices = buffer_func(
                self.samples.sim_differential_indices,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        if hasattr(self.samples, "sim_differential_probas") and (
            self.samples.sim_differential_probas is not None
        ):
            sim_differential_probas = buffer_func(
                self.samples.sim_differential_probas,
                extract_sequences,
                T_idxs,
                B_idxs,
                T + self.n_step_return,
            )
        result = AugSeqSamplesFromReplay(
            *result,
            sim_patho=sim_patho,
            sim_patient=sim_patient,
            sim_severity=sim_severity,
            sim_evidence=sim_evidence,
            sim_timestep=sim_timestep,
            sim_differential_indices=sim_differential_indices,
            sim_differential_probas=sim_differential_probas,
        )

        return torchify_buffer(result)

    def weight_batch_samples(self, batch, is_weights):
        """Method for weighting sampled data from prioritized sequential replay buffer.

        Parameters
        ----------
        batch: tuple
            tuple of data representing the sampled batch.
        is_weights: tensor of float
            the weights to be assigned to the batch data.

        Returns
        -------
        data: tuple
            the weighted batch data.

        """
        return AugSeqSamplesFromReplayPri(*batch, is_weights=is_weights)
