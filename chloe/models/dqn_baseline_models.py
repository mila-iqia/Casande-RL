import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel, DuelingHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from chloe.agents.base import AbstractAgent

RnnState = namedarraytuple("RnnState", ["h", "c"])
"""This class defines a data structure to keep the LSTM state components.

    It contains the following elements:
        - h: the `h` component of the lstm hidden state.
        - c: the `c` component of the lstm hidden state.

"""


class BaselineDQNModel(nn.Module, AbstractAgent):
    """Class representing a simple model used for DQN based training.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_symptoms,
        nonlinearity=nn.ReLU,
        dueling=False,
        dueling_fc_sizes=None,
        embedding_dict=None,
        freeze_one_hot_encoding=True,
        mask_inquired_symptoms=True,
        not_inquired_value=0,
        symptom_2_observation_map=None,
        patho_severity=None,
        include_turns_in_state=False,
        use_turn_just_for_masking=True,
        min_turns_ratio_for_decision=None,
        hierarchical_map=None,
        mask_unrelated_symptoms=False,
        symptom_2_symptom_association=None,
        mask_unrelated_pathos=False,
        symptom_2_patho_association=None,
        symp_default_values=None,
        **kwargs
    ):
        """Initializes the class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers.
        output_size: int
            the dimension of the output layer.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            the dimensions of the duelling branch if instantiated (`dueling` is True).
            If None, then the last value in `hidden_sizes` will be used. Default: None
        embedding_dict: dict
            a dictionary corresponding to the embeddings definition for the provided
            features (feature_index => [num_possible_values, embedding_dim]).
            If instead of a list of two elements we have an int or a list of a single
            element corresponding to `num_possible_values`, then the `embedding_dim`
            will be equal to that same value.
            Default: None
        freeze_one_hot_encoding: boolean
            flag indicating whether or not to use one-hot encoding for the embeddings.
            Default: True
        mask_inquired_symptoms: boolean
            flag indicating whether or not to mask the symptoms which have
            been already inquired. Default: True
        not_inquired_value: int
            value representing symptoms that have not been inquired yet. Default: 0
        symptom_2_observation_map: dict
            a mapping from symptom index to index range associated to
            that symptom in the observation space. Default: None
        patho_severity: list
            the severity associated to each pathology. Default: None
        include_turns_in_state: boolean
            flag indicating whether or not the state contains information regarding
            the current turn in the interaction session. Default: False
        use_turn_just_for_masking: boolean
            flag indicating whether or not to use the turn in the state just for
            masking based on `min_turns_ratio_for_decision`. This is valid only if
            `include_turns_in_state` is True. if `include_turns_in_state` is False,
            then it is defaulted to False no matter what. Default: True
        min_turns_ratio_for_decision: float
            minimum turn ratio for allowing decision making. if specified and
            `include_turns_in_state` is True, then the actions corresponding to
            decision making (here, pathologies) will be articificially deactivated
            until `min_turns_ratio_for_decision` is reached. Default: None
        hierarchical_map: dict (int -> list of int)
            a mapping from master symptom index to dependent symptom indices.
            Default: None
        mask_unrelated_symptoms: boolean
            flag indicating whether or not to mask unrelated symptoms to ones which have
            been already inquired. Default: False
        symptom_2_symptom_association: numpy array
            an NxN boolean array whit N being the number of symptoms where the entry
            [i, j] is True means that symptoms i and j are related. Default: None
        mask_unrelated_pathos: boolean
            flag indicating whether or not to mask unrelated pathos to symptoms which
            have been already inquired. Default: False
        symptom_2_patho_association: numpy array
            an NxM boolean array whit N and M being respectively the number of symptoms
            and the number of pathologies where the entry [i, j] is True means that
            symptom i and pathology j are related. Default: None
        symp_default_values: list
            list of tuple (pos, value) where the ith entry correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        """
        super(BaselineDQNModel, self).__init__()
        self.dueling = dueling
        self.include_turns_in_state = include_turns_in_state
        if min_turns_ratio_for_decision is not None:
            assert min_turns_ratio_for_decision >= 0
            assert min_turns_ratio_for_decision <= 1
            assert include_turns_in_state
        self.min_turns_ratio_for_decision = min_turns_ratio_for_decision
        self.use_turn_just_for_masking = (
            use_turn_just_for_masking and include_turns_in_state
        )
        if embedding_dict is None:
            embedding_dict = {}

        for a in embedding_dict.keys():
            if isinstance(embedding_dict[a], collections.abc.Sequence):
                embedding_dict[a] = list(embedding_dict[a])
            if not isinstance(embedding_dict[a], list):
                embedding_dict[a] = [embedding_dict[a]]
            if len(embedding_dict[a]) < 2:
                embedding_dict[a].append(embedding_dict[a][-1])

        self.embedding_dict = embedding_dict
        self.freeze_one_hot_encoding = freeze_one_hot_encoding
        self.mask_inquired_symptoms = mask_inquired_symptoms
        self.not_inquired_value = not_inquired_value
        self.num_symptoms = num_symptoms
        self.output_size = output_size
        self.patho_severity = patho_severity
        self.symptom_2_observation_map = self._preprocess_symp_2_obs_map(
            symptom_2_observation_map
        )
        self.hierarchical_map = self._preprocess_hierarchical_map(hierarchical_map)

        self._preprocess_symp_default_values(symp_default_values)
        self.symp2symp, self.symp2patho = self._preprocess_symptom_patho_associations(
            symptom_2_symptom_association, symptom_2_patho_association
        )
        self.mask_unrelated_symptoms = (
            mask_unrelated_symptoms and self.symp2symp is not None
        )
        self.mask_unrelated_pathos = (
            mask_unrelated_pathos and self.symp2patho is not None
        )

        self.embeddings = None
        total_embedding_dims = 0
        input_2_embedings = 0
        if len(self.embedding_dict) > 0:
            tmp_embeddings = {
                str(a): nn.Embedding.from_pretrained(
                    torch.eye(self.embedding_dict[a][0], self.embedding_dict[a][1]),
                    freeze=freeze_one_hot_encoding,
                )
                for a in self.embedding_dict.keys()
            }
            self.embeddings = nn.ModuleDict(tmp_embeddings)
            total_embedding_dims = sum(
                [self.embedding_dict[a][1] for a in self.embedding_dict.keys()]
            )
            input_2_embedings = len(self.embedding_dict.keys())
            indices = set(range(input_size)) - set(self.embedding_dict.keys())
            indices = list(indices)
            indices.sort()
            self.selected_indices = torch.LongTensor(indices)
            self.embed_indices = sorted(list(self.embedding_dict.keys()))
            self.embed_indice_tensor = torch.LongTensor(self.embed_indices)

        input_size = input_size - input_2_embedings + total_embedding_dims
        if dueling:
            if dueling_fc_sizes is None:
                dueling_fc_sizes = [hidden_sizes[-1]]
            self.fc_out = MlpModel(input_size, hidden_sizes, nonlinearity=nonlinearity)
            self.head = DuelingHeadModel(
                hidden_sizes[-1], dueling_fc_sizes, output_size
            )
        else:
            self.fc_out = MlpModel(
                input_size, [hidden_sizes[0]], nonlinearity=nonlinearity
            )
            self.head = MlpModel(
                hidden_sizes[0], hidden_sizes[1:], output_size, nonlinearity
            )

    def cuda(self, device=None):
        """Moves all model parameters and buffers to the GPU.

        Parameters
        ----------
        device: int
            id of the gpu device to transfer the model on. Default: None
        """
        super().cuda(device)
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.cuda(device)
            self.embed_indice_tensor = self.embed_indice_tensor.cuda(device)
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.cuda(device)
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].cuda(device)
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.cuda(device)
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.cuda(device)
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.cuda(device)
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.cuda(device)
        return self

    def cpu(self):
        """Moves all model parameters and buffers to the CPU.
        """
        super().cpu()
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.cpu()
            self.embed_indice_tensor = self.embed_indice_tensor.cpu()
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.cpu()
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].cpu()
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.cpu()
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.cpu()
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.cpu()
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.cpu()
        return self

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers.
        """
        super().to(*args, **kwargs)
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.to(*args, **kwargs)
            self.embed_indice_tensor = self.embed_indice_tensor.to(*args, **kwargs)
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.to(*args, **kwargs)
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].to(*args, **kwargs)
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.to(*args, **kwargs)
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.to(*args, **kwargs)
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.to(*args, **kwargs)
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.to(*args, **kwargs)
        return self

    def _preprocess_hierarchical_map(self, hierarchical_map):
        """Preprocess the provided hierarchical_map to make it ready for computations.

        Parameters
        ----------
        hierarchical_map: dict (int -> list of int)
            a mapping from master symptom index to dependent symptom indices.

        Returns
        -------
        result: dict
            the preprocessed map.

        """
        if hierarchical_map is None:
            return None
        for k in hierarchical_map:
            hierarchical_map[k] = torch.LongTensor(hierarchical_map[k])
        return hierarchical_map

    def _preprocess_symptom_patho_associations(self, symp2symp, symp2patho):
        """Preprocess the provided associations to make it ready for computations.

        Parameters
        ----------
        symp2symp: numpy array
            an NxN boolean array whit N being the number of symptoms where the entry
            [i, j] is True means that symptoms i and j are related.
        symp2patho: numpy array
            an NxM boolean array whit N and M being respectively the number of symptoms
            and the number of pathologies where the entry [i, j] is True means that
            symptom i and pathology j are related.

        Returns
        -------
        result: tuple
            the preprocessed assocations.

        """
        res_symp2symp = None
        res_symp2patho = None
        if symp2symp is not None:
            res_symp2symp = torch.from_numpy(np.array(symp2symp)).float()
        if symp2patho is not None:
            res_symp2patho = torch.from_numpy(np.array(symp2patho)).float()
        return res_symp2symp, res_symp2patho

    def _preprocess_symp_default_values(self, symp_default_values):
        """Preprocess the provided default values to make it ready for computations.

        Here the `symp_init_index` property is created and it contains the
        index in the state space for each symptom id.

        Parameters
        ----------
        symp_default_values: list
            list of tuple (pos, value) where the ith entry correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing.

        Returns
        -------
        None

        """
        if symp_default_values is None:
            self.symp_default_values = None
            self.symp_default_indices = None
        else:
            diff = 1 if self.use_turn_just_for_masking else 0
            self.symp_default_indices = torch.LongTensor(
                [a[0] - diff for a in symp_default_values]
            )
            self.symp_default_values = torch.FloatTensor(
                [a[1] for a in symp_default_values]
            )

    def _preprocess_symp_2_obs_map(self, symptom_2_observation_map):
        """Determine the index in the state space associated to each symptom.

        Here the `symp_init_index` property is created and it contains the
        index in the state space for each symptom id.

        Parameters
        ----------
        symptom_2_observation_map: dict
            mapping of each symptom id to the index span in the obsevation/state space.
            if a symptom is associated with a span [2, 6), that means the entries
            2, 3, 4, and 5 of the observation data are dedicated to that symptom.

        Returns
        -------
        result: list
            a sorted representation of the provided dictionary based on the symptom id.

        """
        if symptom_2_observation_map is None:
            self.symp_init_index = None
            return None
        else:
            diff = 1 if self.use_turn_just_for_masking else 0
            self.symp_init_index = torch.LongTensor(
                [
                    symptom_2_observation_map[idx][0] - diff
                    for idx in range(len(symptom_2_observation_map))
                ]
            )
            return [
                [a - diff for a in symptom_2_observation_map[idx]]
                for idx in range(len(symptom_2_observation_map))
            ]

    def _compute_hierarchical_mask(self, mask):
        """Compute the mask of the dependent symptoms based on master symptoms.

        For each dependent symptom, return True if the master is False or keep it
        unchanged otherwise.

        Parameters
        ----------
        mask: bool tensor
            mask of the currently inquired symptoms.

        Returns
        -------
        result: bool tensor
            the updated mask.

        """
        if not self.hierarchical_map:
            return mask
        else:
            for idx in self.hierarchical_map:
                indices = self.hierarchical_map[idx]
                mask[:, indices] = (
                    torch.logical_not(mask[:, idx : idx + 1]) + mask[:, indices]
                ).bool()
        return mask

    def _compute_positively_inquired_mask(self, observation, inquired_mask=None):
        """Compute the mask of the symptoms that are positively inquired.

        For each symptom, return True if positively inquired, False otherwise.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.
        inquired_mask: tensor
            mask of inquired symptoms

        Returns
        -------
        result: tensor
            the computed mask for inquired symptoms.

        """
        if self.symp_default_values is None or self.symp_default_indices is None:
            return None
        data = observation[:, self.symp_default_indices]
        # True if not default value, False otherwise.
        mask = data != self.symp_default_values.reshape(
            1, -1, *([1] * (len(data.size()) - 2))
        )
        if inquired_mask is None:
            inquired_mask = self._compute_inquired_mask(observation)
        return torch.logical_and(mask, inquired_mask)

    def _compute_inquired_mask(self, observation):
        """Compute the mask of the symptoms that have been already inquired.

        For each symptom, return True if already inquired, False otherwise.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the computed mask for inquired symptoms.

        """
        if self.symptom_2_observation_map is None:
            # we assume boolean feature
            data = observation[:, -self.num_symptoms :]

        else:
            data = observation[:, self.symp_init_index]
        # True if already inquired, False otherwise.
        mask = data != self.not_inquired_value
        return mask

    def _embed_observation(self, obs):
        """Preprocess the provided observation by performing embedding operations.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the preprocessed observation data.

        """
        if self.embeddings is not None:
            parts = []
            embed_indices = self.embed_indice_tensor
            data = torch.index_select(obs, 1, embed_indices).long()
            for k, i in enumerate(self.embed_indices):
                parts.append(self.embeddings[str(i)](data[:, k]))
            indices = self.selected_indices
            parts.append(torch.index_select(obs, 1, indices))
            obs = torch.cat(parts, dim=1)

        return obs

    def _mask_q_val_for_inquired_symptoms(self, q, observation):
        """Mask Q-values corresponding to already inquired symptoms.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned symptoms.

        """
        if self.mask_inquired_symptoms:
            mask = self._compute_inquired_mask(observation)
            mask = self._compute_hierarchical_mask(mask)
            q = torch.cat(
                [
                    q[:, 0 : self.num_symptoms].masked_fill(mask, torch.finfo().min),
                    q[:, self.num_symptoms :],
                ],
                dim=1,
            )
        return q

    def _mask_pi_val_for_unrelated_pathos(self, pi, observation):
        """Mask policy entries corresponding to unrelated pathology actions.

        Parameters
        ----------
        pi: tensor
            policy data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        return self._mask_q_val_for_unrelated_pathos(pi, observation)

    def _mask_q_val_for_unrelated_pathos(self, q, observation):
        """Mask Q-values corresponding to unrelated pathology actions.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        return q

    def _mask_q_val_for_decision_making(self, q, turns):
        """Mask Q-values corresponding to pathology actions.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        turns: tensor
            the turns as contained in the observation data.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        if self.include_turns_in_state and (
            self.min_turns_ratio_for_decision is not None
        ):
            mask = turns < self.min_turns_ratio_for_decision
            mask = mask.expand(-1, q.size(1) - self.num_symptoms)
            q = torch.cat(
                [
                    q[:, 0 : self.num_symptoms],
                    q[:, self.num_symptoms :].masked_fill(mask, torch.finfo().min),
                ],
                dim=1,
            )
        return q

    def _retrieve_time_and_redefine_observation(self, observation):
        """Retrieves turns from observation and eventually modify the observations.

        The observation will be modified if `use_turn_just_for_masking` is True.
        In this case, the turns will be removed from the observation data.

        Parameters
        ----------
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        turns: tensor
            the turns as contained in the observation data.
        observation: tensor
            the modified observation data.

        """
        turns = None
        if self.include_turns_in_state:
            turns = observation[:, 0:1]
            if self.use_turn_just_for_masking:
                observation = observation[:, 1:]
        return turns, observation

    def forward(self, observation, prev_action=None, prev_reward=None):
        """Computes the Q-values given the provided input data.

        Parameters
        ----------
        observation: tensor
            observation data as provided to the model.
        prev_action: tensor
            previous action performed by the agent. Default: None
        prev_reward: tensor
            previous reward received by the agent. Default: None

        Returns
        -------
        result: tensor
            the computed Q-Values.

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)

        if self.fc_out is not None:
            obs = self.fc_out(obs.view(T * B, -1))

        obs = obs.view(T * B, -1)
        q = self.head(obs)

        q = self._mask_q_val_for_inquired_symptoms(q, observation)

        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)

        # mask unrelated patho
        q = self._mask_q_val_for_unrelated_pathos(q, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        return q

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionary. The action is accessible using the key
        `action`.

        Parameters
        ----------
        observations: tensor
            observation data as provided to the model.

        Returns
        -------
        result: dict
            a dictionary containing the computed action and some
            extra information eventually.

        """
        qvalues = self.forward(observations)
        actions = torch.argmax(qvalues, dim=-1)

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "q": qvalues}


class BaselineCatDQNModel(nn.Module, AbstractAgent):
    """Class representing a simple model used for Categorical DQN based training.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_symptoms,
        nonlinearity=nn.ReLU,
        dueling=False,
        dueling_fc_sizes=None,
        embedding_dict=None,
        freeze_one_hot_encoding=True,
        mask_inquired_symptoms=True,
        not_inquired_value=0,
        symptom_2_observation_map=None,
        patho_severity=None,
        include_turns_in_state=False,
        use_turn_just_for_masking=True,
        min_turns_ratio_for_decision=None,
        hierarchical_map=None,
        mask_unrelated_symptoms=False,
        symptom_2_symptom_association=None,
        mask_unrelated_pathos=False,
        symptom_2_patho_association=None,
        symp_default_values=None,
        n_atoms=51,
        **kwargs
    ):
        """Initilizes a class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers.
        output_size: int
            the dimension of the output layer.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            the dimensions of the duelling branch if instantiated (`dueling` is True).
            If None, then the last value in `hidden_sizes` will be used. Default: None
        embedding_dict: dict
            a dictionary corresponding to the embeddings definition for the provided
            features (feature_index => [num_possible_values, embedding_dim]).
            If instead of a list of two elements we have an int or a list of a single
            element corresponding to `num_possible_values`, then the `embedding_dim`
            will be equal to that same value.
            Default: None
        freeze_one_hot_encoding: boolean
            flag indicating whether or not to use one-hot encoding for the embeddings.
            Default: True
        mask_inquired_symptoms: boolean
            flag indicating whether or not to mask the symptoms which have
            been already inquired. Default: True
        not_inquired_value: int
            value representing symptoms that have not been inquired yet. Default: 0
        symptom_2_observation_map: dict
            A mapping from symptom index to index range associated to
            that symptom in the observation space. Default: None
        patho_severity: list
            the severity associated to each pathology. Default: None
        include_turns_in_state: boolean
            flag indicating whether or not the state contains information regarding
            the current turn in the interaction session. Default: False
        use_turn_just_for_masking: boolean
            flag indicating whether or not to use the turn in the state just for
            masking based on `min_turns_ratio_for_decision`. This is valid only if
            `include_turns_in_state` is True. if `include_turns_in_state` is False,
            then it is defaulted to False no matter what. Default: True
        min_turns_ratio_for_decision: int
            minimum turn ratio for allowing decision making. if specified and
            `include_turns_in_state` is True, then the actions corresponding to
            decision making (here, pathologies) will be articificially deactivated
            until `min_turns_ratio_for_decision` is reached. Default: None
        hierarchical_map: dict (int -> list of int)
            a mapping from master symptom index to dependent symptom indices.
            Default: None
        mask_unrelated_symptoms: boolean
            flag indicating whether or not to mask unrelated symptoms to ones which have
            been already inquired. Default: False
        symptom_2_symptom_association: numpy array
            an NxN boolean array whit N being the number of symptoms where the entry
            [i, j] is True means that symptoms i and j are related. Default: None
        mask_unrelated_pathos: boolean
            flag indicating whether or not to mask unrelated pathos to symptoms which
            have been already inquired. Default: False
        symptom_2_patho_association: numpy array
            an NxM boolean array whit N and M being respectively the number of symptoms
            and the number of pathologies where the entry [i, j] is True means that
            symptom i and pathology j are related. Default: None
        symp_default_values: list
            list of tuple (pos, value) where the ith entry correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        n_atoms: int
            the number of atoms to be used by the categorical DQN algo. Default: 51
        """
        super(BaselineCatDQNModel, self).__init__()
        self.dueling = dueling
        self.n_atoms = n_atoms
        self.include_turns_in_state = include_turns_in_state
        if min_turns_ratio_for_decision is not None:
            assert min_turns_ratio_for_decision >= 0
            assert min_turns_ratio_for_decision <= 1
            assert include_turns_in_state
        self.min_turns_ratio_for_decision = min_turns_ratio_for_decision
        self.use_turn_just_for_masking = (
            use_turn_just_for_masking and include_turns_in_state
        )
        self.output_size = output_size
        self.patho_severity = patho_severity
        self.symptom_2_observation_map = self._preprocess_symp_2_obs_map(
            symptom_2_observation_map
        )
        self.hierarchical_map = self._preprocess_hierarchical_map(hierarchical_map)

        self._preprocess_symp_default_values(symp_default_values)
        self.symp2symp, self.symp2patho = self._preprocess_symptom_patho_associations(
            symptom_2_symptom_association, symptom_2_patho_association
        )
        self.mask_unrelated_symptoms = (
            mask_unrelated_symptoms and self.symp2symp is not None
        )
        self.mask_unrelated_pathos = (
            mask_unrelated_pathos and self.symp2patho is not None
        )

        if embedding_dict is None:
            embedding_dict = {}

        for a in embedding_dict.keys():
            if isinstance(embedding_dict[a], collections.abc.Sequence):
                embedding_dict[a] = list(embedding_dict[a])
            if not isinstance(embedding_dict[a], list):
                embedding_dict[a] = [embedding_dict[a]]
            if len(embedding_dict[a]) < 2:
                embedding_dict[a].append(embedding_dict[a][-1])

        self.embedding_dict = embedding_dict
        self.freeze_one_hot_encoding = freeze_one_hot_encoding
        self.mask_inquired_symptoms = mask_inquired_symptoms
        self.not_inquired_value = not_inquired_value
        self.num_symptoms = num_symptoms
        self.zero_index = torch.LongTensor([0])
        # categorical distribution properties - useful for inference
        self.V_min = None
        self.V_max = None
        self.dist_z = None

        self.embeddings = None
        total_embedding_dims = 0
        input_2_embedings = 0
        if len(self.embedding_dict) > 0:
            tmp_embeddings = {
                str(a): nn.Embedding.from_pretrained(
                    torch.eye(self.embedding_dict[a][0], self.embedding_dict[a][1]),
                    freeze=freeze_one_hot_encoding,
                )
                for a in self.embedding_dict.keys()
            }
            self.embeddings = nn.ModuleDict(tmp_embeddings)
            total_embedding_dims = sum(
                [self.embedding_dict[a][1] for a in self.embedding_dict.keys()]
            )
            input_2_embedings = len(self.embedding_dict.keys())
            indices = set(range(input_size)) - set(self.embedding_dict.keys())
            indices = list(indices)
            indices.sort()
            self.selected_indices = torch.LongTensor(indices)
            self.embed_indices = sorted(list(self.embedding_dict.keys()))
            self.embed_indice_tensor = torch.LongTensor(self.embed_indices)

        input_size = input_size - input_2_embedings + total_embedding_dims
        if dueling:
            if dueling_fc_sizes is None:
                dueling_fc_sizes = [hidden_sizes[-1]]
            self.fc_out = MlpModel(input_size, hidden_sizes, nonlinearity=nonlinearity)
            self.head = DistributionalDuelingHeadModel(
                hidden_sizes[-1], dueling_fc_sizes, output_size, n_atoms
            )
        else:
            self.fc_out = MlpModel(
                input_size, [hidden_sizes[0]], nonlinearity=nonlinearity
            )
            self.head = MlpModel(
                hidden_sizes[0], hidden_sizes[1:], output_size * n_atoms, nonlinearity
            )

    def cuda(self, device=None):
        """Moves all model parameters and buffers to the GPU.

        Parameters
        ----------
        device: int
            id of the gpu device to transfer the model on. Default: None
        """
        super().cuda(device)
        self.zero_index = self.zero_index.cuda(device)
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.cuda(device)
            self.embed_indice_tensor = self.embed_indice_tensor.cuda(device)
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.cuda(device)
        if self.dist_z is not None:
            self.dist_z = self.dist_z.cuda(device)
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].cuda(device)
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.cuda(device)
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.cuda(device)
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.cuda(device)
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.cuda(device)
        return self

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers.
        """
        super().to(*args, **kwargs)
        self.zero_index = self.zero_index.to(*args, **kwargs)
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.to(*args, **kwargs)
            self.embed_indice_tensor = self.embed_indice_tensor.to(*args, **kwargs)
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.to(*args, **kwargs)
        if self.dist_z is not None:
            self.dist_z = self.dist_z.to(*args, **kwargs)
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].to(*args, **kwargs)
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.to(*args, **kwargs)
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.to(*args, **kwargs)
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.to(*args, **kwargs)
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.to(*args, **kwargs)
        return self

    def cpu(self):
        """Moves all model parameters and buffers to the CPU.
        """
        super().cpu()
        self.zero_index = self.zero_index.cpu()
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.cpu()
            self.embed_indice_tensor = self.embed_indice_tensor.cpu()
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.cpu()
        if self.dist_z is not None:
            self.dist_z = self.dist_z.cpu()
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].cpu()
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.cpu()
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.cpu()
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.cpu()
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.cpu()
        return self

    def _preprocess_hierarchical_map(self, hierarchical_map):
        """Preprocess the provided hierarchical_map to make it ready for computations.

        Parameters
        ----------
        hierarchical_map: dict (int -> list of int)
            a mapping from master symptom index to dependent symptom indices.

        Returns
        -------
        result: dict
            the preprocessed map.

        """
        if hierarchical_map is None:
            return None
        for k in hierarchical_map:
            hierarchical_map[k] = torch.LongTensor(hierarchical_map[k])
        return hierarchical_map

    def _preprocess_symptom_patho_associations(self, symp2symp, symp2patho):
        """Preprocess the provided associations to make it ready for computations.

        Parameters
        ----------
        symp2symp: numpy array
            an NxN boolean array whit N being the number of symptoms where the entry
            [i, j] is True means that symptoms i and j are related.
        symp2patho: numpy array
            an NxM boolean array whit N and M being respectively the number of symptoms
            and the number of pathologies where the entry [i, j] is True means that
            symptom i and pathology j are related.

        Returns
        -------
        result: tuple
            the preprocessed assocations.

        """
        res_symp2symp = None
        res_symp2patho = None
        if symp2symp is not None:
            res_symp2symp = torch.from_numpy(np.array(symp2symp)).float()
        if symp2patho is not None:
            res_symp2patho = torch.from_numpy(np.array(symp2patho)).float()
        return res_symp2symp, res_symp2patho

    def _preprocess_symp_default_values(self, symp_default_values):
        """Preprocess the provided default values to make it ready for computations.

        Here the `symp_init_index` property is created and it contains the
        index in the state space for each symptom id.

        Parameters
        ----------
        symp_default_values: list
            list of tuple (pos, value) where the ith entry correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing.

        Returns
        -------
        None

        """
        if symp_default_values is None:
            self.symp_default_values = None
            self.symp_default_indices = None
        else:
            diff = 1 if self.use_turn_just_for_masking else 0
            self.symp_default_indices = torch.LongTensor(
                [a[0] - diff for a in symp_default_values]
            )
            self.symp_default_values = torch.FloatTensor(
                [a[1] for a in symp_default_values]
            )

    def _preprocess_symp_2_obs_map(self, symptom_2_observation_map):
        """Determine the index in the state space associated to each symptom.

        Here the `symp_init_index` property is created and it contains the
        index in the state space for each symptom id.

        Parameters
        ----------
        symptom_2_observation_map: dict
            mapping of each symptom id to the index span in the obsevation/state space.
            if a symptom is associated with a span [2, 6), that means the entries
            2, 3, 4, and 5 of the observation data are dedicated to that symptom.

        Returns
        -------
        result: list
            a sorted representation of the provided dictionary based on the symptom id.

        """
        if symptom_2_observation_map is None:
            self.symp_init_index = None
            return None
        else:
            diff = 1 if self.use_turn_just_for_masking else 0
            self.symp_init_index = torch.LongTensor(
                [
                    symptom_2_observation_map[idx][0] - diff
                    for idx in range(len(symptom_2_observation_map))
                ]
            )
            return [
                [a - diff for a in symptom_2_observation_map[idx]]
                for idx in range(len(symptom_2_observation_map))
            ]

    def _compute_hierarchical_mask(self, mask):
        """Compute the mask of the dependent symptoms based on master symptoms.

        For each dependent symptom, return True if the master is False or keep it
        unchanged otherwise.

        Parameters
        ----------
        mask: bool tensor
            mask of the currently inquired symptoms.

        Returns
        -------
        result: bool tensor
            the updated mask.

        """
        if not self.hierarchical_map:
            return mask
        else:
            for idx in self.hierarchical_map:
                indices = self.hierarchical_map[idx]
                mask[:, indices] = (
                    torch.logical_not(mask[:, idx : idx + 1]) + mask[:, indices]
                ).bool()
        return mask

    def _compute_positively_inquired_mask(self, observation, inquired_mask=None):
        """Compute the mask of the symptoms that are positively inquired.

        For each symptom, return True if positively inquired, False otherwise.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.
        inquired_mask: tensor
            mask of inquired symptoms

        Returns
        -------
        result: tensor
            the computed mask for inquired symptoms.

        """
        if self.symp_default_values is None or self.symp_default_indices is None:
            return None
        data = observation[:, self.symp_default_indices]
        # True if not default value, False otherwise.
        mask = data != self.symp_default_values.reshape(
            1, -1, *([1] * (len(data.size()) - 2))
        )
        if inquired_mask is None:
            inquired_mask = self._compute_inquired_mask(observation)
        return torch.logical_and(mask, inquired_mask)

    def _compute_inquired_mask(self, observation):
        """Compute the mask of the symptoms that have been already inquired.

        For each symptom, return True if already inquired, False otherwise.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the computed mask for inquired symptoms.

        """
        if self.symptom_2_observation_map is None:
            # we assume boolean feature
            data = observation[:, -self.num_symptoms :]

        else:
            data = observation[:, self.symp_init_index]
        # True if already inquired, False otherwise
        mask = data != self.not_inquired_value
        return mask

    def _embed_observation(self, obs):
        """Preprocess the provided observation by performing embedding operations.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the preprocessed observation data.
        """
        if self.embeddings is not None:
            parts = []
            embed_indices = self.embed_indice_tensor
            data = torch.index_select(obs, 1, embed_indices).long()
            for k, i in enumerate(self.embed_indices):
                parts.append(self.embeddings[str(i)](data[:, k]))
            indices = self.selected_indices
            parts.append(torch.index_select(obs, 1, indices))
            obs = torch.cat(parts, dim=1)

        return obs

    def _mask_q_val_for_inquired_symptoms(self, q, observation):
        """Mask Q-values corresponding to already inquired symptoms.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned symptoms.

        """
        if self.mask_inquired_symptoms:
            mask = self._compute_inquired_mask(observation)
            mask = self._compute_hierarchical_mask(mask)
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, self.n_atoms)
            zero_index = self.zero_index
            mask = mask.index_fill(2, zero_index, False)
            q = torch.cat(
                [
                    q[:, 0 : self.num_symptoms].masked_fill(mask, torch.finfo().min),
                    q[:, self.num_symptoms :],
                ],
                dim=1,
            )
        return q

    def _mask_pi_val_for_unrelated_pathos(self, pi, observation):
        """Mask policy entries corresponding to unrelated pathology actions.

        Parameters
        ----------
        pi: tensor
            policy data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        return pi

    def _mask_q_val_for_unrelated_pathos(self, q, observation):
        """Mask Q-values corresponding to unrelated pathology actions.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        return q

    def _mask_q_val_for_decision_making(self, q, turns):
        """Mask Q-values corresponding to pathology actions.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        turns: tensor
            the turns as contained in the observation data.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        if self.include_turns_in_state and (
            self.min_turns_ratio_for_decision is not None
        ):
            mask = turns < self.min_turns_ratio_for_decision
            mask = mask.expand(-1, q.size(1) - self.num_symptoms)
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, self.n_atoms)
            zero_index = self.zero_index
            mask = mask.index_fill(2, zero_index, False)
            q = torch.cat(
                [
                    q[:, 0 : self.num_symptoms],
                    q[:, self.num_symptoms :].masked_fill(mask, torch.finfo().min),
                ],
                dim=1,
            )
        return q

    def _retrieve_time_and_redefine_observation(self, observation):
        """Retrieves turns from observation and eventually modify the observations.

        The observation will be modified if `use_turn_just_for_masking` is True.
        In this case, the turns will be removed from the observation data.

        Parameters
        ----------
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        turns: tensor
            the turns as contained in the observation data.
        observation: tensor
            the modified observation data.

        """
        turns = None
        if self.include_turns_in_state:
            turns = observation[:, 0:1]
            if self.use_turn_just_for_masking:
                observation = observation[:, 1:]
        return turns, observation

    def set_V_min_max(self, V_min, V_max):
        """Set the minimum and maximum values of the expected Q-values.

        This function should be called prior the call of the `predict`
        method during inference.

        Parameters
        ----------
        V_min: float
            minimum value of the expected Q-values.
        V_max: float
            maximum value of the expected Q-values.

        Returns
        -------
        None

        """
        self.V_min = V_min
        self.V_max = V_max
        self.dist_z = torch.linspace(
            V_min, V_max, self.n_atoms, device=self.zero_index.device
        )

    def forward(self, observation, prev_action=None, prev_reward=None):
        """Computes the distributional Q-values given the provided input data.

        Parameters
        ----------
        observation: tensor
            observation data as provided to the model.
        prev_action: tensor
            previous action performed by the agent. Default: None
        prev_reward: tensor
            previous reward received by the agent. Default: None

        Returns
        -------
        result: tensor
            the computed distributional Q-Values.

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)

        if self.fc_out is not None:
            obs = self.fc_out(obs.view(T * B, -1))

        obs = obs.view(T * B, -1)
        q = self.head(obs)

        q = q.view(-1, self.output_size, self.n_atoms)
        q = self._mask_q_val_for_inquired_symptoms(q, observation)
        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)
        # mask unrelated patho
        q = self._mask_q_val_for_unrelated_pathos(q, observation)
        q = F.softmax(q, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        return q

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionary. The action is accessible using the key
        `action`.

        Parameters
        ----------
        observations: tensor
            observation data as provided to the model.

        Returns
        -------
        result: dict
            a dictionary containing the computed action and some
            extra information eventually.

        """
        if self.dist_z is None:
            raise ValueError(
                "the function 'set_V_min_max' must be called prior to call 'predict'."
            )
        p = self.forward(observations)

        qvalues = torch.tensordot(p, self.dist_z, dims=1)
        actions = torch.argmax(qvalues, dim=-1)

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "q": qvalues, "p": p}


class BaselineR2D1DQNModel(nn.Module, AbstractAgent):
    """Class representing a simple model used for R2D1 DQN based training.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        lstm_size,
        num_symptoms,
        nonlinearity=nn.ReLU,
        dueling=False,
        dueling_fc_sizes=None,
        embedding_dict=None,
        freeze_one_hot_encoding=True,
        mask_inquired_symptoms=True,
        not_inquired_value=0,
        symptom_2_observation_map=None,
        patho_severity=None,
        include_turns_in_state=False,
        use_turn_just_for_masking=True,
        min_turns_ratio_for_decision=None,
        hierarchical_map=None,
        mask_unrelated_symptoms=False,
        symptom_2_symptom_association=None,
        mask_unrelated_pathos=False,
        symptom_2_patho_association=None,
        symp_default_values=None,
        **kwargs
    ):
        """Initializes the class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers.
        output_size: int
            the dimension of the output layer.
        lstm_size: int
            the dimension of the lstm internal size.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            If None, then the last value in `hidden_sizes` will be used. Default: None
        embedding_dict: dict
            a dictionary corresponding to the embeddings definition for the provided
            features (feature_index => [num_possible_values, embedding_dim]).
            If instead of a list of two elements we have an int or a list of a single
            element corresponding to `num_possible_values`, then the `embedding_dim`
            will be equal to that same value.
            Default: None
        freeze_one_hot_encoding: boolean
            flag indicating whether or not to use one-hot encoding for the embeddings.
            Default: True
        mask_inquired_symptoms: boolean
            flag indicating whether or not to mask the symptoms which have
            been already inquired. Default: True
        not_inquired_value: int
            value representing symptoms that have not been inquired yet. Default: 0
        symptom_2_observation_map: dict
            A mapping from symptom index to index range associated to
            that symptom in the observation space. Default: None
        patho_severity: list
            the severity associated to each pathology. Default: None
        include_turns_in_state: boolean
            flag indicating whether or not the state contains information regarding
            the current turn in the interaction session. Default: False
        use_turn_just_for_masking: boolean
            flag indicating whether or not to use the turn in the state just for
            masking based on `min_turns_ratio_for_decision`. This is valid only if
            `include_turns_in_state` is True. if `include_turns_in_state` is False,
            then it is defaulted to False no matter what. Default: True
        min_turns_ratio_for_decision: float
            minimum turn ratio for allowing decision making. if specified and
            `include_turns_in_state` is True, then the actions corresponding to
            decision making (here, pathologies) will be articificially deactivated
            until `min_turns_ratio_for_decision` is reached. Default: None
        hierarchical_map: dict (int -> list of int)
            a mapping from master symptom index to dependent symptom indices.
            Default: None
        mask_unrelated_symptoms: boolean
            flag indicating whether or not to mask unrelated symptoms to ones which have
            been already inquired. Default: False
        symptom_2_symptom_association: numpy array
            an NxN boolean array whit N being the number of symptoms where the entry
            [i, j] is True means that symptoms i and j are related. Default: None
        mask_unrelated_pathos: boolean
            flag indicating whether or not to mask unrelated pathos to symptoms which
            have been already inquired. Default: False
        symptom_2_patho_association: numpy array
            an NxM boolean array whit N and M being respectively the number of symptoms
            and the number of pathologies where the entry [i, j] is True means that
            symptom i and pathology j are related. Default: None
        symp_default_values: list
            list of tuple (pos, value) where the ith entry correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        """
        super(BaselineR2D1DQNModel, self).__init__()
        self.dueling = dueling
        self.include_turns_in_state = include_turns_in_state
        if min_turns_ratio_for_decision is not None:
            assert min_turns_ratio_for_decision >= 0
            assert min_turns_ratio_for_decision <= 1
            assert include_turns_in_state
        self.min_turns_ratio_for_decision = min_turns_ratio_for_decision
        self.use_turn_just_for_masking = (
            use_turn_just_for_masking and include_turns_in_state
        )
        if embedding_dict is None:
            embedding_dict = {}

        for a in embedding_dict.keys():
            if isinstance(embedding_dict[a], collections.abc.Sequence):
                embedding_dict[a] = list(embedding_dict[a])
            if not isinstance(embedding_dict[a], list):
                embedding_dict[a] = [embedding_dict[a]]
            if len(embedding_dict[a]) < 2:
                embedding_dict[a].append(embedding_dict[a][-1])

        self.embedding_dict = embedding_dict
        self.freeze_one_hot_encoding = freeze_one_hot_encoding
        self.mask_inquired_symptoms = mask_inquired_symptoms
        self.not_inquired_value = not_inquired_value
        self.num_symptoms = num_symptoms
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.patho_severity = patho_severity
        self.symptom_2_observation_map = self._preprocess_symp_2_obs_map(
            symptom_2_observation_map
        )
        self.hierarchical_map = self._preprocess_hierarchical_map(hierarchical_map)

        self._preprocess_symp_default_values(symp_default_values)
        self.symp2symp, self.symp2patho = self._preprocess_symptom_patho_associations(
            symptom_2_symptom_association, symptom_2_patho_association
        )
        self.mask_unrelated_symptoms = (
            mask_unrelated_symptoms and self.symp2symp is not None
        )
        self.mask_unrelated_pathos = (
            mask_unrelated_pathos and self.symp2patho is not None
        )

        self.embeddings = None
        total_embedding_dims = 0
        input_2_embedings = 0
        if len(self.embedding_dict) > 0:
            tmp_embeddings = {
                str(a): nn.Embedding.from_pretrained(
                    torch.eye(self.embedding_dict[a][0], self.embedding_dict[a][1]),
                    freeze=freeze_one_hot_encoding,
                )
                for a in self.embedding_dict.keys()
            }
            self.embeddings = nn.ModuleDict(tmp_embeddings)
            total_embedding_dims = sum(
                [self.embedding_dict[a][1] for a in self.embedding_dict.keys()]
            )
            input_2_embedings = len(self.embedding_dict.keys())
            indices = set(range(input_size)) - set(self.embedding_dict.keys())
            indices = list(indices)
            indices.sort()
            self.selected_indices = torch.LongTensor(indices)
            self.embed_indices = sorted(list(self.embedding_dict.keys()))
            self.embed_indice_tensor = torch.LongTensor(self.embed_indices)

        input_size = input_size - input_2_embedings + total_embedding_dims
        if dueling:
            if dueling_fc_sizes is None:
                dueling_fc_sizes = [hidden_sizes[-1]]
            self.fc_out = MlpModel(input_size, hidden_sizes, nonlinearity=nonlinearity)
            self.lstm = torch.nn.LSTM(hidden_sizes[-1], lstm_size)
            self.head = DuelingHeadModel(lstm_size, dueling_fc_sizes, output_size)
        else:
            self.fc_out = MlpModel(
                input_size, [hidden_sizes[0]], nonlinearity=nonlinearity
            )
            self.lstm = torch.nn.LSTM(hidden_sizes[0], lstm_size)
            self.head = MlpModel(lstm_size, hidden_sizes[1:], output_size, nonlinearity)

    def cuda(self, device=None):
        """Moves all model parameters and buffers to the GPU.

        Parameters
        ----------
        device: int
            Id of the gpu device to transfer the model on. Default: None
        """
        super().cuda(device)
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.cuda(device)
            self.embed_indice_tensor = self.embed_indice_tensor.cuda(device)
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.cuda(device)
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].cuda(device)
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.cuda(device)
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.cuda(device)
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.cuda(device)
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.cuda(device)
        return self

    def cpu(self):
        """Moves all model parameters and buffers to the CPU.
        """
        super().cpu()
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.cpu()
            self.embed_indice_tensor = self.embed_indice_tensor.cpu()
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.cpu()
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].cpu()
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.cpu()
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.cpu()
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.cpu()
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.cpu()
        return self

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers.
        """
        super().to(*args, **kwargs)
        if len(self.embedding_dict) > 0:
            self.selected_indices = self.selected_indices.to(*args, **kwargs)
            self.embed_indice_tensor = self.embed_indice_tensor.to(*args, **kwargs)
        if self.symp_init_index is not None:
            self.symp_init_index = self.symp_init_index.to(*args, **kwargs)
        if self.hierarchical_map:
            for k in self.hierarchical_map.keys():
                self.hierarchical_map[k] = self.hierarchical_map[k].to(*args, **kwargs)
        if self.symp2symp is not None:
            self.symp2symp = self.symp2symp.to(*args, **kwargs)
        if self.symp2patho is not None:
            self.symp2patho = self.symp2patho.to(*args, **kwargs)
        if self.symp_default_values is not None:
            self.symp_default_values = self.symp_default_values.to(*args, **kwargs)
        if self.symp_default_indices is not None:
            self.symp_default_indices = self.symp_default_indices.to(*args, **kwargs)
        return self

    def _preprocess_hierarchical_map(self, hierarchical_map):
        """Preprocess the provided hierarchical_map to make it ready for computations.

        Parameters
        ----------
        hierarchical_map: dict (int -> list of int)
            a mapping from master symptom index to dependent symptom indices.

        Returns
        -------
        result: dict
            the preprocessed map.

        """
        if hierarchical_map is None:
            return None
        for k in hierarchical_map:
            hierarchical_map[k] = torch.LongTensor(hierarchical_map[k])
        return hierarchical_map

    def _preprocess_symptom_patho_associations(self, symp2symp, symp2patho):
        """Preprocess the provided associations to make it ready for computations.

        Parameters
        ----------
        symp2symp: numpy array
            an NxN boolean array whit N being the number of symptoms where the entry
            [i, j] is True means that symptoms i and j are related.
        symp2patho: numpy array
            an NxM boolean array whit N and M being respectively the number of symptoms
            and the number of pathologies where the entry [i, j] is True means that
            symptom i and pathology j are related.

        Returns
        -------
        result: tuple
            the preprocessed assocations.

        """
        res_symp2symp = None
        res_symp2patho = None
        if symp2symp is not None:
            res_symp2symp = torch.from_numpy(np.array(symp2symp)).float()
        if symp2patho is not None:
            res_symp2patho = torch.from_numpy(np.array(symp2patho)).float()
        return res_symp2symp, res_symp2patho

    def _preprocess_symp_default_values(self, symp_default_values):
        """Preprocess the provided default values to make it ready for computations.

        Here the `symp_init_index` property is created and it contains the
        index in the state space for each symptom id.

        Parameters
        ----------
        symp_default_values: list
            list of tuple (pos, value) where the ith entry correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing.

        Returns
        -------
        None

        """
        if symp_default_values is None:
            self.symp_default_values = None
            self.symp_default_indices = None
        else:
            diff = 1 if self.use_turn_just_for_masking else 0
            self.symp_default_indices = torch.LongTensor(
                [a[0] - diff for a in symp_default_values]
            )
            self.symp_default_values = torch.FloatTensor(
                [a[1] for a in symp_default_values]
            )

    def _preprocess_symp_2_obs_map(self, symptom_2_observation_map):
        """Determine the index in the state space associated to each symptom.

        Here the `symp_init_index` property is created and it contains the
        index in the state space for each symptom id.

        Parameters
        ----------
        symptom_2_observation_map: dict
            mapping of each symptom id to the index span in the obsevation/state space.
            if a symptom is associated with a span [2, 6), that means the entries
            2, 3, 4, and 5 of the observation data are dedicated to that symptom.

        Returns
        -------
        result: list
            a sorted representation of the provided dictionary based on the symptom id.

        """
        if symptom_2_observation_map is None:
            self.symp_init_index = None
            return None
        else:
            diff = 1 if self.use_turn_just_for_masking else 0
            self.symp_init_index = torch.LongTensor(
                [
                    symptom_2_observation_map[idx][0] - diff
                    for idx in range(len(symptom_2_observation_map))
                ]
            )
            return [
                [a - diff for a in symptom_2_observation_map[idx]]
                for idx in range(len(symptom_2_observation_map))
            ]

    def _compute_hierarchical_mask(self, mask):
        """Compute the mask of the dependent symptoms based on master symptoms.

        For each dependent symptom, return True if the master is False or keep it
        unchanged otherwise.

        Parameters
        ----------
        mask: bool tensor
            mask of the currently inquired symptoms.

        Returns
        -------
        result: bool tensor
            the updated mask.

        """
        if not self.hierarchical_map:
            return mask
        else:
            for idx in self.hierarchical_map:
                indices = self.hierarchical_map[idx]
                mask[:, indices] = (
                    torch.logical_not(mask[:, idx : idx + 1]) + mask[:, indices]
                ).bool()
        return mask

    def _compute_positively_inquired_mask(self, observation, inquired_mask=None):
        """Compute the mask of the symptoms that are positively inquired.

        For each symptom, return True if positively inquired, False otherwise.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.
        inquired_mask: tensor
            mask of inquired symptoms

        Returns
        -------
        result: tensor
            the computed mask for inquired symptoms.

        """
        if self.symp_default_values is None or self.symp_default_indices is None:
            return None
        data = observation[:, self.symp_default_indices]
        # True if not default value, False otherwise.
        mask = data != self.symp_default_values.reshape(
            1, -1, *([1] * (len(data.size()) - 2))
        )
        if inquired_mask is None:
            inquired_mask = self._compute_inquired_mask(observation)
        return torch.logical_and(mask, inquired_mask)

    def _compute_inquired_mask(self, observation):
        """Compute the mask of the symptoms that have been already inquired.

        For each symptom, return True if already inquired, False otherwise.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the computed mask for inquired symptoms.

        """
        if self.symptom_2_observation_map is None:
            # we assume boolean feature
            data = observation[:, -self.num_symptoms :]

        else:
            data = observation[:, self.symp_init_index]
        # True if already inquired, False otherwise
        mask = data != self.not_inquired_value
        return mask

    def _embed_observation(self, obs):
        """Preprocess the provided observation by performing embedding operations.

        Parameters
        ----------
        obs: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the preprocessed observation data.
        """
        if self.embeddings is not None:
            parts = []
            embed_indices = self.embed_indice_tensor
            data = torch.index_select(obs, 1, embed_indices).long()
            for k, i in enumerate(self.embed_indices):
                parts.append(self.embeddings[str(i)](data[:, k]))
            indices = self.selected_indices
            parts.append(torch.index_select(obs, 1, indices))
            obs = torch.cat(parts, dim=1)

        return obs

    def _mask_q_val_for_inquired_symptoms(self, q, observation):
        """Mask Q-values corresponding to already inquired symptoms.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned symptoms.
        """
        if self.mask_inquired_symptoms:
            mask = self._compute_inquired_mask(observation)
            mask = self._compute_hierarchical_mask(mask)
            q = torch.cat(
                [
                    q[:, 0 : self.num_symptoms].masked_fill(mask, torch.finfo().min),
                    q[:, self.num_symptoms :],
                ],
                dim=1,
            )
        return q

    def _mask_pi_val_for_unrelated_pathos(self, pi, observation):
        """Mask policy entries corresponding to unrelated pathology actions.

        Parameters
        ----------
        pi: tensor
            policy data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        return self._mask_q_val_for_unrelated_pathos(pi, observation)

    def _mask_q_val_for_unrelated_pathos(self, q, observation):
        """Mask Q-values corresponding to unrelated pathology actions.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.

        """
        return q

    def _mask_q_val_for_decision_making(self, q, turns):
        """Mask Q-values corresponding to pathology actions.

        Parameters
        ----------
        q: tensor
            Q-value data as initially computed by the model.
        turns: tensor
            the turns as contained in the observation data.

        Returns
        -------
        result: tensor
            the corresponding Q-values with masked values (-inf) for concerned actions.
        """
        if self.include_turns_in_state and (
            self.min_turns_ratio_for_decision is not None
        ):
            mask = turns < self.min_turns_ratio_for_decision
            mask = mask.expand(-1, q.size(1) - self.num_symptoms)
            q = torch.cat(
                [
                    q[:, 0 : self.num_symptoms],
                    q[:, self.num_symptoms :].masked_fill(mask, torch.finfo().min),
                ],
                dim=1,
            )
        return q

    def _retrieve_time_and_redefine_observation(self, observation):
        """Retrieves turns from observation and eventually modify the observations.

        The observation will be modified if `use_turn_just_for_masking` is True.
        In this case, the turns will be removed from the observation data.

        Parameters
        ----------
        observation: tensor
            observation data as provided to the model.

        Returns
        -------
        turns: tensor
            the turns as contained in the observation data.
        observation: tensor
            the modified observation data.
        """
        turns = None
        if self.include_turns_in_state:
            turns = observation[:, 0:1]
            if self.use_turn_just_for_masking:
                observation = observation[:, 1:]
        return turns, observation

    def forward(
        self, observation, prev_action=None, prev_reward=None, init_rnn_state=None
    ):
        """Computes the Q-values given the provided input data.

        Parameters
        ----------
        observation: tensor
            observation data as provided to the model.
        prev_action: tensor
            previous action performed by the agent. Default: None
        prev_reward: tensor
            previous reward received by the agent. Default: None
        init_rnn_state: tensor
            init state of the model. Default: None

        Returns
        -------
        q: tensor
            the computed Q-Values.
        next_rnn_state: tensor
            the next state of the lstm.

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)

        if self.fc_out is not None:
            obs = self.fc_out(obs.view(T * B, -1))

        lstm_input = obs.view(T, B, -1)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        q = self.head(lstm_out.view(T * B, -1))

        q = self._mask_q_val_for_inquired_symptoms(q, observation)

        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)

        # mask unrelated patho
        q = self._mask_q_val_for_unrelated_pathos(q, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, next_rnn_state

    def reset(self):
        """Function for resseting the internal state of the agent.

           Useful when the agent works in a reccurrent setting.

        """
        self.predict_rnn_state = None

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionary. The action is accessible using the key
        `action`.

        Parameters
        ----------
        observations: tensor
            observation data as provided to the model.

        Returns
        -------
        result: dict
            a dictionary containing the computed action and some
            extra information eventually.

        """
        init_rnn_state = (
            self.predict_rnn_state if hasattr(self, "predict_rnn_state") else None
        )
        out = self.forward(observations, init_rnn_state=init_rnn_state)
        qvalues, next_rnn_state = out

        self.predict_rnn_state = next_rnn_state
        actions = torch.argmax(qvalues, dim=-1)

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "rnn_state": next_rnn_state, "q": qvalues}
