import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from chloe.models.dqn_baseline_models import (
    BaselineCatDQNModel,
    BaselineDQNModel,
    BaselineR2D1DQNModel,
    RnnState,
)
from chloe.models.dqn_rebuild_models import (
    RebuildCatDQNModel,
    RebuildDQNModel,
    RebuildR2D1DQNModel,
)


class MixedDQNModel(BaselineDQNModel):
    """Class representing a mixed model used for DQN based training.

       The difference with `BaselineDQNModel` model is that there is two types
       of outputs for this model:
           - Q-values for symptoms
           - Probability distribution for pathologies.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_symptoms,
        pi_hidden_sizes=None,
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
        use_stop_action=True,
        **kwargs
    ):
        """Initializes the class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers for the Q-values branch.
        output_size: int
            the dimension of the output layer.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        pi_hidden_sizes: list of ints
            the dimension of the hidden layers for the probability distribution branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            Default: None
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
            list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        use_stop_action:
            a flag indicating if the agent should implement a stop action instead of
            diagnosis prediction.
            Default: True
        """
        super(MixedDQNModel, self).__init__(
            input_size,
            hidden_sizes,
            num_symptoms + 1 if use_stop_action else output_size,
            num_symptoms,
            nonlinearity,
            dueling,
            dueling_fc_sizes,
            embedding_dict,
            freeze_one_hot_encoding,
            mask_inquired_symptoms,
            not_inquired_value,
            symptom_2_observation_map,
            patho_severity,
            include_turns_in_state,
            use_turn_just_for_masking,
            min_turns_ratio_for_decision,
            hierarchical_map,
            mask_unrelated_symptoms,
            symptom_2_symptom_association,
            mask_unrelated_pathos,
            symptom_2_patho_association,
            symp_default_values,
            **kwargs
        )
        if pi_hidden_sizes is None:
            pi_hidden_sizes = [hidden_sizes[-1]]
        self.use_stop_action = use_stop_action
        num_pathos = output_size - num_symptoms
        self.pi = MlpModel(
            self.fc_out.output_size, pi_hidden_sizes, num_pathos, nonlinearity
        )

    def forward(self, observation, prev_action=None, prev_reward=None):
        """Computes the predicted values given the provided input data.

        It computes the Q-values for symptoms and probability distribution
        (unnormalized - before softmax) for pathologies given the provided
        input data.

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
        q: tensor
            the computed Q-Values for symptoms.
        pi: tensor
            the computed unnormalized score probabilities for pathologies.

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)
        obs = self.fc_out(obs.view(T * B, -1))

        q = self.head(obs)
        pi = self.pi(obs)

        q = self._mask_q_val_for_inquired_symptoms(q, observation)

        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)

        # mask unrelated patho
        if not self.use_stop_action:
            q = self._mask_q_val_for_unrelated_pathos(q, observation)
        pi = self._mask_pi_val_for_unrelated_pathos(pi, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q, pi = restore_leading_dims((q, pi), lead_dim, T, B)

        return q, pi

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
        qvalues, pi = self.forward(observations)
        actions = torch.argmax(qvalues, dim=-1)
        pi_actions = torch.argmax(pi, dim=-1) + self.num_symptoms

        mask = actions == self.num_symptoms
        actions[mask] = pi_actions[mask]

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "pi": pi, "q": qvalues}


class MixedCatDQNModel(BaselineCatDQNModel):
    """Class representing a simple model used for Categorical DQN based training.

       The difference with `BaselineCatDQNModel` model is that there is two types
       of outputs for this model:
           - Q-values for symptoms.
           - Probability distribution for pathologies.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_symptoms,
        pi_hidden_sizes=None,
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
        use_stop_action=True,
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
        pi_hidden_sizes: list of ints
            the dimension of the hidden layers for the probability distribution branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            Default: None
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
            the current turn in the interaction session. Default: False.
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
            list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        use_stop_action:
            a flag indicating if the agent should implement a stop action instead of
            diagnosis prediction.
            Default: True
        n_atoms: int
            the number of atoms to be used by the categorical DQN algo. Default: 51
        """
        super(MixedCatDQNModel, self).__init__(
            input_size,
            hidden_sizes,
            num_symptoms + 1 if use_stop_action else output_size,
            num_symptoms,
            nonlinearity,
            dueling,
            dueling_fc_sizes,
            embedding_dict,
            freeze_one_hot_encoding,
            mask_inquired_symptoms,
            not_inquired_value,
            symptom_2_observation_map,
            patho_severity,
            include_turns_in_state,
            use_turn_just_for_masking,
            min_turns_ratio_for_decision,
            hierarchical_map,
            mask_unrelated_symptoms,
            symptom_2_symptom_association,
            mask_unrelated_pathos,
            symptom_2_patho_association,
            symp_default_values,
            n_atoms,
            **kwargs
        )
        if pi_hidden_sizes is None:
            pi_hidden_sizes = [hidden_sizes[-1]]
        self.use_stop_action = use_stop_action
        num_pathos = output_size - num_symptoms
        self.pi = MlpModel(
            self.fc_out.output_size, pi_hidden_sizes, num_pathos, nonlinearity
        )

    def forward(self, observation, prev_action=None, prev_reward=None):
        """Computes the predicted values given the provided input data.

        It computes the distributional Q-values for symptoms and the
        probability distribution (unnormalized - before softmax) for
        pathologies given the provided input data.

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
        q: tensor
            the computed Q-Values for symptoms.
        pi: tensor
            the computed unnormalized score probabilities for pathologies.

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)
        obs = self.fc_out(obs.view(T * B, -1))

        q = self.head(obs)
        pi = self.pi(obs)

        q = q.view(-1, self.output_size, self.n_atoms)
        q = self._mask_q_val_for_inquired_symptoms(q, observation)
        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)
        # mask unrelated patho
        if not self.use_stop_action:
            q = self._mask_q_val_for_unrelated_pathos(q, observation)
        q = F.softmax(q, dim=-1)
        pi = self._mask_pi_val_for_unrelated_pathos(pi, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q, pi = restore_leading_dims((q, pi), lead_dim, T, B)

        return q, pi

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
        p, pi = self.forward(observations)

        qvalues = torch.tensordot(p, self.dist_z, dims=1)
        actions = torch.argmax(qvalues, dim=-1)
        pi_actions = torch.argmax(pi, dim=-1) + self.num_symptoms

        mask = actions == self.num_symptoms
        actions[mask] = pi_actions[mask]

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "pi": pi, "q": qvalues, "p": p}


class MixedR2D1DQNModel(BaselineR2D1DQNModel):
    """Class representing a mixed model used for R2D1 DQN based training.

       The difference with `BaselineR2D1DQNModel` model is that there is two types
       of outputs for this model:
           - Q-values for symptoms
           - Probability distribution for pathologies.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        lstm_size,
        num_symptoms,
        pi_hidden_sizes=None,
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
        use_stop_action=True,
        **kwargs
    ):
        """Initializes the class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers for the Q-values branch.
        output_size: int
            the dimension of the output layer.
        lstm_size: int
            the dimension of the lstm internal size.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        pi_hidden_sizes: list of ints
            the dimension of the hidden layers for the probability distribution branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            Default: None
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
            list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        use_stop_action:
            a flag indicating if the agent should implement a stop action instead of
            diagnosis prediction.
            Default: True
        """
        super(MixedR2D1DQNModel, self).__init__(
            input_size,
            hidden_sizes,
            num_symptoms + 1 if use_stop_action else output_size,
            lstm_size,
            num_symptoms,
            nonlinearity,
            dueling,
            dueling_fc_sizes,
            embedding_dict,
            freeze_one_hot_encoding,
            mask_inquired_symptoms,
            not_inquired_value,
            symptom_2_observation_map,
            patho_severity,
            include_turns_in_state,
            use_turn_just_for_masking,
            min_turns_ratio_for_decision,
            hierarchical_map,
            mask_unrelated_symptoms,
            symptom_2_symptom_association,
            mask_unrelated_pathos,
            symptom_2_patho_association,
            symp_default_values,
            **kwargs
        )
        if pi_hidden_sizes is None:
            pi_hidden_sizes = [hidden_sizes[-1]]
        self.use_stop_action = use_stop_action
        num_pathos = output_size - num_symptoms
        self.pi = MlpModel(lstm_size, pi_hidden_sizes, num_pathos, nonlinearity)

    def forward(
        self, observation, prev_action=None, prev_reward=None, init_rnn_state=None
    ):
        """Computes the predicted values given the provided input data.

        It computes the Q-values for symptoms and probability distribution
        (unnormalized - before softmax) for pathologies given the provided
        input data.

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
            the computed Q-Values for symptoms.
        pi: tensor
            the computed unnormalized score probabilities for pathologies.
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
        obs = self.fc_out(obs.view(T * B, -1))

        lstm_input = obs.view(T, B, -1)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        lstm_out = lstm_out.view(T * B, -1)

        q = self.head(lstm_out)
        pi = self.pi(lstm_out)

        q = self._mask_q_val_for_inquired_symptoms(q, observation)

        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)

        # mask unrelated patho
        if not self.use_stop_action:
            q = self._mask_q_val_for_unrelated_pathos(q, observation)
        pi = self._mask_pi_val_for_unrelated_pathos(pi, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q, pi = restore_leading_dims((q, pi), lead_dim, T, B)

        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, pi, next_rnn_state

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
        qvalues, pi, next_rnn_state = out

        self.predict_rnn_state = next_rnn_state
        actions = torch.argmax(qvalues, dim=-1)
        pi_actions = torch.argmax(pi, dim=-1) + self.num_symptoms

        mask = actions == self.num_symptoms
        actions[mask] = pi_actions[mask]

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "pi": pi, "rnn_state": next_rnn_state, "q": qvalues}


class MixRebDQNModel(RebuildDQNModel):
    """Class representing a mixed and rebuild model used for DQN based training.

       The difference with `BaselineDQNModel` model is that there is two types
       of outputs for this model:
           - Q-values for symptoms
           - Probability distribution for pathologies.
           - Output for rebuilding the input features.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_symptoms,
        reb_size,
        reb_hidden_sizes=None,
        pi_hidden_sizes=None,
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
        use_stop_action=True,
        **kwargs
    ):
        """Initializes the class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers for the Q-values branch.
        output_size: int
            the dimension of the output layer.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        reb_size: int
            the dimension of the rebuild branch layer.
        reb_hidden_sizes: list of ints
            the dimension of the hidden layers for the rebuild branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        pi_hidden_sizes: list of ints
            the dimension of the hidden layers for the probability distribution branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            Default: None
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
            list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        use_stop_action:
            a flag indicating if the agent should implement a stop action instead of
            diagnosis prediction.
            Default: True
        """
        super(MixRebDQNModel, self).__init__(
            input_size,
            hidden_sizes,
            num_symptoms + 1 if use_stop_action else output_size,
            num_symptoms,
            reb_size,
            reb_hidden_sizes,
            nonlinearity,
            dueling,
            dueling_fc_sizes,
            embedding_dict,
            freeze_one_hot_encoding,
            mask_inquired_symptoms,
            not_inquired_value,
            symptom_2_observation_map,
            patho_severity,
            include_turns_in_state,
            use_turn_just_for_masking,
            min_turns_ratio_for_decision,
            hierarchical_map,
            mask_unrelated_symptoms,
            symptom_2_symptom_association,
            mask_unrelated_pathos,
            symptom_2_patho_association,
            symp_default_values,
            **kwargs
        )
        if pi_hidden_sizes is None:
            pi_hidden_sizes = [hidden_sizes[-1]]
        self.use_stop_action = use_stop_action
        num_pathos = output_size - num_symptoms
        self.pi = MlpModel(
            self.fc_out.output_size, pi_hidden_sizes, num_pathos, nonlinearity
        )

    def forward(self, observation, prev_action=None, prev_reward=None):
        """Computes the predicted values given the provided input data.

        It computes the Q-values for symptoms and probability distribution
        (unnormalized - before softmax) for pathologies given the provided
        input data as well as the rebuild input feature.

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
        q: tensor
            the computed Q-Values for symptoms.
        pi: tensor
            the computed unnormalized score probabilities for pathologies.
        reb: tensor
            the reconstructed output data in [0, 1].

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)
        obs = self.fc_out(obs.view(T * B, -1))

        q = self.head(obs)
        pi = self.pi(obs)
        reb = self.rebuild(obs)

        q = self._mask_q_val_for_inquired_symptoms(q, observation)

        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)

        # mask unrelated patho
        if not self.use_stop_action:
            q = self._mask_q_val_for_unrelated_pathos(q, observation)
        pi = self._mask_pi_val_for_unrelated_pathos(pi, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q, pi, reb = restore_leading_dims((q, pi, reb), lead_dim, T, B)

        return q, pi, reb

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
        qvalues, pi, reb = self.forward(observations)
        actions = torch.argmax(qvalues, dim=-1)
        pi_actions = torch.argmax(pi, dim=-1) + self.num_symptoms

        mask = actions == self.num_symptoms
        actions[mask] = pi_actions[mask]

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "pi": pi, "rebuild": reb, "q": qvalues}


class MixRebCatDQNModel(RebuildCatDQNModel):
    """Class representing a mixed/rebuild model used for Categorical DQN based training.

       The difference with `BaselineCatDQNModel` model is that there is two types
       of outputs for this model:
           - Q-values for symptoms.
           - Probability distribution for pathologies.
           - Output for rebuilding the input features.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        num_symptoms,
        reb_size,
        reb_hidden_sizes=None,
        pi_hidden_sizes=None,
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
        use_stop_action=True,
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
        reb_size: int
            the dimension of the rebuild branch layer.
        reb_hidden_sizes: list of ints
            the dimension of the hidden layers for the rebuild branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        pi_hidden_sizes: list of ints
            the dimension of the hidden layers for the probability distribution branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            Default: None
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
            the current turn in the interaction session. Default: False.
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
            list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        use_stop_action:
            a flag indicating if the agent should implement a stop action instead of
            diagnosis prediction.
            Default: True
        n_atoms: int
            the number of atoms to be used by the categorical DQN algo. Default: 51
        """
        super(MixRebCatDQNModel, self).__init__(
            input_size,
            hidden_sizes,
            num_symptoms + 1 if use_stop_action else output_size,
            num_symptoms,
            reb_size,
            reb_hidden_sizes,
            nonlinearity,
            dueling,
            dueling_fc_sizes,
            embedding_dict,
            freeze_one_hot_encoding,
            mask_inquired_symptoms,
            not_inquired_value,
            symptom_2_observation_map,
            patho_severity,
            include_turns_in_state,
            use_turn_just_for_masking,
            min_turns_ratio_for_decision,
            hierarchical_map,
            mask_unrelated_symptoms,
            symptom_2_symptom_association,
            mask_unrelated_pathos,
            symptom_2_patho_association,
            symp_default_values,
            n_atoms,
            **kwargs
        )
        if pi_hidden_sizes is None:
            pi_hidden_sizes = [hidden_sizes[-1]]
        self.use_stop_action = use_stop_action
        num_pathos = output_size - num_symptoms
        self.pi = MlpModel(
            self.fc_out.output_size, pi_hidden_sizes, num_pathos, nonlinearity
        )

    def forward(self, observation, prev_action=None, prev_reward=None):
        """Computes the predicted values given the provided input data.

        It computes the distributional Q-values for symptoms and the
        probability distribution (unnormalized - before softmax) for
        pathologies given the provided input data as well as the rebuild input feature.

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
        q: tensor
            the computed Q-Values for symptoms.
        pi: tensor
            the computed unnormalized score probabilities for pathologies.
        reb: tensor
            the reconstructed output data in [0, 1].

        """
        observation = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, 1)
        observation = observation.view(T * B, *obs_shape)

        # retrieve the turns and alter the obs if necessary
        turns, observation = self._retrieve_time_and_redefine_observation(observation)

        obs = self._embed_observation(observation)
        obs = self.fc_out(obs.view(T * B, -1))

        q = self.head(obs)
        pi = self.pi(obs)
        reb = self.rebuild(obs)

        q = q.view(-1, self.output_size, self.n_atoms)
        q = self._mask_q_val_for_inquired_symptoms(q, observation)
        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)
        # mask unrelated patho
        if not self.use_stop_action:
            q = self._mask_q_val_for_unrelated_pathos(q, observation)
        q = F.softmax(q, dim=-1)
        pi = self._mask_pi_val_for_unrelated_pathos(pi, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q, pi, reb = restore_leading_dims((q, pi, reb), lead_dim, T, B)

        return q, pi, reb

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
        p, pi, reb = self.forward(observations)

        qvalues = torch.tensordot(p, self.dist_z, dims=1)
        actions = torch.argmax(qvalues, dim=-1)
        pi_actions = torch.argmax(pi, dim=-1) + self.num_symptoms

        mask = actions == self.num_symptoms
        actions[mask] = pi_actions[mask]

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {"action": actions, "pi": pi, "rebuild": reb, "q": qvalues, "p": p}


class MixRebR2D1DQNModel(RebuildR2D1DQNModel):
    """Class representing a mixed and rebuild model used for R2D1 DQN based training.

       The difference with `BaselineR2D1DQNModel` model is that there is two types
       of outputs for this model:
           - Q-values for symptoms
           - Probability distribution for pathologies.
           - Output for rebuilding the input features.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        lstm_size,
        num_symptoms,
        reb_size,
        reb_hidden_sizes=None,
        pi_hidden_sizes=None,
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
        use_stop_action=True,
        **kwargs
    ):
        """Initializes the class object.

        Parameters
        ----------
        input_size: int
            the dimension of the input data.
        hidden_sizes: list of ints
            the dimension of the hidden layers for the Q-values branch.
        output_size: int
            the dimension of the output layer.
        lstm_size: int
            the dimension of the lstm internal size.
        num_symptoms: int
            the number of symptoms the agent is able to inquire.
        reb_size: int
            the dimension of the rebuild branch layer.
        reb_hidden_sizes: list of ints
            the dimension of the hidden layers for the rebuild branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        pi_hidden_sizes: list of ints
            the dimension of the hidden layers for the probability distribution branch.
            if None, then the last value in `hidden_sizes` will be used. Default: None
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            Default: None
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
            list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing. Default: None
        use_stop_action:
            a flag indicating if the agent should implement a stop action instead of
            diagnosis prediction.
            Default: True
        """
        super(MixRebR2D1DQNModel, self).__init__(
            input_size,
            hidden_sizes,
            num_symptoms + 1 if use_stop_action else output_size,
            lstm_size,
            num_symptoms,
            reb_size,
            reb_hidden_sizes,
            nonlinearity,
            dueling,
            dueling_fc_sizes,
            embedding_dict,
            freeze_one_hot_encoding,
            mask_inquired_symptoms,
            not_inquired_value,
            symptom_2_observation_map,
            patho_severity,
            include_turns_in_state,
            use_turn_just_for_masking,
            min_turns_ratio_for_decision,
            hierarchical_map,
            mask_unrelated_symptoms,
            symptom_2_symptom_association,
            mask_unrelated_pathos,
            symptom_2_patho_association,
            symp_default_values,
            **kwargs
        )
        if pi_hidden_sizes is None:
            pi_hidden_sizes = [hidden_sizes[-1]]
        self.use_stop_action = use_stop_action
        num_pathos = output_size - num_symptoms
        self.pi = MlpModel(lstm_size, pi_hidden_sizes, num_pathos, nonlinearity)

    def forward(
        self, observation, prev_action=None, prev_reward=None, init_rnn_state=None
    ):
        """Computes the predicted values given the provided input data.

        It computes the Q-values for symptoms and probability distribution
        (unnormalized - before softmax) for pathologies given the provided
        input data as well as the rebuild input feature.

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
            the computed Q-Values for symptoms.
        pi: tensor
            the computed unnormalized score probabilities for pathologies.
        reb: tensor
            the reconstructed output data in [0, 1].
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
        obs = self.fc_out(obs.view(T * B, -1))

        lstm_input = obs.view(T, B, -1)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        lstm_out = lstm_out.view(T * B, -1)

        q = self.head(lstm_out)
        pi = self.pi(lstm_out)
        reb = self.rebuild(lstm_out)

        q = self._mask_q_val_for_inquired_symptoms(q, observation)

        # mask for decision making
        q = self._mask_q_val_for_decision_making(q, turns)

        # mask unrelated patho
        if not self.use_stop_action:
            q = self._mask_q_val_for_unrelated_pathos(q, observation)
        pi = self._mask_pi_val_for_unrelated_pathos(pi, observation)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q, pi, reb = restore_leading_dims((q, pi, reb), lead_dim, T, B)

        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, pi, reb, next_rnn_state

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
        qvalues, pi, reb, next_rnn_state = out

        self.predict_rnn_state = next_rnn_state
        actions = torch.argmax(qvalues, dim=-1)
        pi_actions = torch.argmax(pi, dim=-1) + self.num_symptoms

        mask = actions == self.num_symptoms
        actions[mask] = pi_actions[mask]

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.cpu().numpy()

        return {
            "action": actions,
            "pi": pi,
            "rebuild": reb,
            "rnn_state": next_rnn_state,
            "q": qvalues,
        }
