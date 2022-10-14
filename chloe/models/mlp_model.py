import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel, DuelingHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from chloe.agents.base import AbstractAgent


class MyMlpDQNModel(nn.Module, AbstractAgent):
    """Class representing a simple model used for DQN based training.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        nonlinearity=nn.ReLU,
        dueling=False,
        dueling_fc_sizes=None,
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
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            If None, then the last value in `hidden_sizes` will be used. Default: None

        """
        super(MyMlpDQNModel, self).__init__()
        self.dueling = dueling
        if dueling:
            if dueling_fc_sizes is None:
                dueling_fc_sizes = [hidden_sizes[-1]]
            self.fc_out = MlpModel(input_size, hidden_sizes, nonlinearity=nonlinearity)
            self.head = DuelingHeadModel(
                hidden_sizes[-1], dueling_fc_sizes, output_size
            )
        else:
            self.fc_out = None
            self.head = MlpModel(input_size, hidden_sizes, output_size, nonlinearity)

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
        obs = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(obs, 1)

        if self.fc_out is not None:
            obs = self.fc_out(obs.view(T * B, *obs_shape))
            obs = obs.view(T * B, -1)
        else:
            obs = obs.view(T * B, *obs_shape)

        q = self.head(obs)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        return q

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionary. The action is accessible using the key `action`.

        Parameters
        ----------
        observations: tensor
            observation data as provided to the model.

        Returns
        -------
        result: dict
            a dictionnary containing the computed action and some
            extra information eventually.

        """
        qvalues = self.forward(observations)
        actions = torch.argmax(qvalues, dim=-1)

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.numpy()

        return {"action": actions, "q": qvalues}


class MyMlpCatDQNModel(nn.Module, AbstractAgent):
    """Class representing a simple model used for Categorical DQN based training.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        nonlinearity=nn.ReLU,
        dueling=False,
        dueling_fc_sizes=None,
        n_atoms=51,
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
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        dueling: boolean
            flag indicating whether or not to have a dueling model. Default: False
        dueling_fc_sizes: list of ints, None
            The dimensions of the duelling branch if instantiated (`dueling` is True).
            If None, then the last value in `hidden_sizes` will be used. Default: None
        n_atoms: int
            the number of atoms to be used by the categorical DQN algo. Default: 51

        """
        super(MyMlpCatDQNModel, self).__init__()
        self.dueling = dueling
        self.n_atoms = n_atoms
        self.output_size = output_size
        # categorical distribution properties - useful for inference
        self.V_min = None
        self.V_max = None
        self.dist_z = None
        if dueling:
            if dueling_fc_sizes is None:
                dueling_fc_sizes = [hidden_sizes[-1]]
            self.fc_out = MlpModel(input_size, hidden_sizes, nonlinearity=nonlinearity)
            self.head = DistributionalDuelingHeadModel(
                hidden_sizes[-1], dueling_fc_sizes, output_size, n_atoms
            )
        else:
            self.fc_out = None
            self.head = MlpModel(
                input_size, hidden_sizes, output_size * n_atoms, nonlinearity
            )

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
        self.dist_z = torch.linspace(V_min, V_max, self.n_atoms)

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
        obs = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(obs, 1)

        if self.fc_out is not None:
            obs = self.fc_out(obs.view(T * B, *obs_shape))
            obs = obs.view(T * B, -1)
        else:
            obs = obs.view(T * B, *obs_shape)

        q = self.head(obs)
        q = q.view(-1, self.output_size, self.n_atoms)
        q = F.softmax(q, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        return q

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionnary. The action is accessible using the key `action`.

        Parameters
        ----------
        observations: tensor
            observation data as provided to the model.

        Returns
        -------
        result: dict
            a dictionnary containing the computed action and some
            extra information eventually.

        """
        if self.dist_z is None:
            raise ValueError(
                "the function 'set_V_min_max' must be called prior to call 'predict'."
            )
        p = self.forward(observations)

        self.dist_z = self.dist_z.to(p.device)
        qvalues = torch.tensordot(p, self.dist_z, dim=1)
        actions = torch.argmax(qvalues, dim=-1)

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.numpy()

        return {"action": actions, "q": qvalues}


class MyMlpPGModel(nn.Module, AbstractAgent):
    """Class representing a simple model used for Policy Gradient based training.
    """

    def __init__(
        self, input_size, hidden_sizes, output_size, nonlinearity=nn.ReLU, **kwargs
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
        nonlinearity: non_linearity function
            the non_linearity function to be used in the network. Default: `nn.ReLU`
        """
        super(MyMlpPGModel, self).__init__()
        self.fc_out = MlpModel(input_size, hidden_sizes, None, nonlinearity)
        self.pi = nn.Linear(hidden_sizes[-1], output_size)
        self.value = nn.Linear(hidden_sizes[-1], 1)

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
        pi: tensor
            the probability distribution on the possible actions.
        v: tensor
            the expected values associated to each action.

        """
        obs = observation.float()

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, obs_shape = infer_leading_dims(obs, 1)

        q = self.fc_out(obs.view(T * B, *obs_shape))

        pi = F.softmax(self.pi(q.view(T * B, -1)), dim=-1)
        v = self.value(q.view(T * B, -1)).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionnary. The action is accessible using the key `action`.

        Parameters
        ----------
        observations: tensor
            observation data as provided to the model.

        Returns
        -------
        result: dict
            a dictionnary containing the computed action and some
            extra information eventually.

        """
        pi, _ = self.forward(observations)
        actions = torch.argmax(pi, dim=-1)

        if torch.numel(actions) == 1:
            actions = actions.view(-1).item()
        else:
            actions = actions.numpy()

        return {"action": actions}
