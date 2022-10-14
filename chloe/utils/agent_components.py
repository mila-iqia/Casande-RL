import torch
from rlpyt.agents.dqn.catdqn_agent import AgentInfo as CatAgentInfo
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.dqn_agent import AgentInfo
from rlpyt.agents.dqn.r2d1_agent import AgentInfo as R2D1AgentInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

MixedAgentInfo = namedarraytuple("MixedAgentInfo", AgentInfo._fields + ("dist_info",),)
"""This class overloads `dqn_agent.AgentInfo` with relevant data.

    The data added are the following:
        - dist_info: the predicted probability distribution.
"""
MixedCatAgentInfo = namedarraytuple(
    "MixedCatAgentInfo", CatAgentInfo._fields + ("dist_info",),
)
"""This class overloads `catdqn_agent.AgentInfo` with relevant data.

    The data added are the following:
        - dist_info: the predicted probability distribution.
"""
MixedR2D1AgentInfo = namedarraytuple(
    "MixedR2D1AgentInfo", R2D1AgentInfo._fields + ("dist_info",)
)
"""This class overloads `r2d1_agent.AgentInfo` with relevant data.

    The data added are the following:
        - dist_info: the predicted probability distribution.
"""
RebuildAgentInfo = namedarraytuple(
    "RebuildAgentInfo", AgentInfo._fields + ("rebuild_info",),
)
"""This class overloads `dqn_agent.AgentInfo` with relevant data.

    The data added are the following:
        - rebuild_info: the reconstructed data.
"""
RebuildCatAgentInfo = namedarraytuple(
    "RebuildCatAgentInfo", CatAgentInfo._fields + ("rebuild_info",),
)
"""This class overloads `catdqn_agent.AgentInfo` with relevant data.

    The data added are the following:
        - rebuild_info: the reconstructed data.
"""
RebuildR2D1AgentInfo = namedarraytuple(
    "RebuildR2D1AgentInfo", R2D1AgentInfo._fields + ("rebuild_info",)
)
"""This class overloads `r2d1_agent.AgentInfo` with relevant data.

    The data added are the following:
        - rebuild_info: the reconstructed data.
"""
MixRebAgentInfo = namedarraytuple(
    "MixRebAgentInfo", AgentInfo._fields + ("dist_info", "rebuild_info",),
)
"""This class overloads `dqn_agent.AgentInfo` with relevant data.

    The data added are the following:
        - dist_info: the predicted probability distribution.
        - rebuild_info: the reconstructed data.
"""
MixRebCatAgentInfo = namedarraytuple(
    "MixRebCatAgentInfo", CatAgentInfo._fields + ("dist_info", "rebuild_info",),
)
"""This class overloads `catdqn_agent.AgentInfo` with relevant data.

    The data added are the following:
        - dist_info: the predicted probability distribution.
        - rebuild_info: the reconstructed data.
"""
MixRebR2D1AgentInfo = namedarraytuple(
    "MixRebR2D1AgentInfo", R2D1AgentInfo._fields + ("dist_info", "rebuild_info",)
)
"""This class overloads `r2d1_agent.AgentInfo` with relevant data.

    The data added are the following:
        - dist_info: the predicted probability distribution.
        - rebuild_info: the reconstructed data.
"""


class MixedDQNAgentMixin:
    """Mixin class to properly handle MixedDQN model outputs by DQN like Agents.

    This class overloads the `__call__` and `target` methods
    to return only the Q-values computed by the model. It also introduces the
    `classify` method to provide the probability distribution computed by the model.
    It also includes the `reconstruct` method for getting the reconstructed data if any.
    Finally, it overloads the `to_agent_step` to refine the information returned
    to the agent when executing the policy by including the probability distribution
    and the reconstructed data if any.
    """

    def __call__(self, observation, prev_action, prev_reward):
        """Defines the method of the agent for computing the Q-Values.

        This method is overloaded in such a way that it returns only the
        Q-values from the model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The computed Q-values.

        """
        output = super().__call__(observation, prev_action, prev_reward)
        reb = None
        if len(output) == 2:
            q, _ = output
        else:
            q, _, reb = output
        # saved q for later loss computation (if grad is enabled)
        if torch.is_grad_enabled():
            self.saved_q = q
        # saved reb for later loss computation (if grad is enabled)
        if torch.is_grad_enabled() and reb is not None:
            self.saved_reconstruct = reb
        return q

    def target(self, observation, prev_action, prev_reward):
        """Returns the target Q-values for states/observations..

        This method is overloaded in such a way that it returns only the
        Q-values from the target model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The computed target Q-values.

        """
        output = super().target(observation, prev_action, prev_reward)
        q = output[0]
        return q

    def _get_pi_actions(self, pi):
        """Epsilon sampling to get the max indices from the provided pi.

        Parameters
        ----------
        pi: tensor
            the predicted distribution.

        Return
        ----------
        result: tensor
            the sampled indices.

        """
        arg_select = torch.argmax(pi, dim=-1)
        mask = torch.rand(arg_select.shape) < self.distribution._epsilon
        arg_rand = torch.randint(low=0, high=pi.shape[-1], size=(mask.sum(),))
        arg_select[mask] = arg_rand
        return arg_select

    def to_agent_step(self, output):
        """Convert the model outputs into Agent step information.

        The agent step info includes the computed Q-values, the selected actions
        to be executed by the agent as well as the computed probability distribution.

        Parameters
        ----------
        output: tensor, tuple of tensor
            the output of the agent model.

        Return
        ----------
        result: object
            The Agent step information.

        """
        q, pi = output[0], output[1]
        reb = None if len(output) < 3 else output[2]
        agent_step = super().to_agent_step(q)
        pi, reb = buffer_to((pi, reb), device="cpu")
        selected_actions = agent_step.action
        pi_actions = self._get_pi_actions(pi)
        use_stop_action = getattr(self.model, "use_stop_action", True)
        if isinstance(self, CatDqnAgent):
            num_q_actions = q.size(-2) - 1 if use_stop_action else q.size(-2)
            agent_info = (
                MixedCatAgentInfo(*agent_step.agent_info, dist_info=pi)
                if reb is None
                else MixRebCatAgentInfo(
                    *agent_step.agent_info, dist_info=pi, rebuild_info=reb
                )
            )
        else:
            num_q_actions = q.size(-1) - 1 if use_stop_action else q.size(-1)
            agent_info = (
                MixedAgentInfo(*agent_step.agent_info, dist_info=pi)
                if reb is None
                else MixRebAgentInfo(
                    *agent_step.agent_info, dist_info=pi, rebuild_info=reb
                )
            )

        if use_stop_action:
            pi_actions = pi_actions + num_q_actions
            mask = selected_actions == num_q_actions
            selected_actions[mask] = pi_actions[mask]

        tmp_dict = dict(agent_step.items())
        tmp_dict["agent_info"] = agent_info
        tmp_dict["action"] = selected_actions
        return agent_step.__class__(**tmp_dict)

    def classify(self, observation, prev_action, prev_reward):
        """Defines the method of the agent for computing the probability distribution.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The computed probability distributions (logits).

        """
        output = super().__call__(observation, prev_action, prev_reward)
        pi = output[1]
        return pi

    def reconstruct(self, observation, prev_action, prev_reward):
        """Defines the method of the agent for computing the reconstructed data.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The reconstructed data.

        """
        output = super().__call__(observation, prev_action, prev_reward)
        return None if len(output) < 3 else output[2]


class MixedSeqDQNAgentMixin:
    """Mixin class to properly handle MixedDQN model outputs by sequential DQN Agents.

    This class overloads the `__call__` and `target` methods
    to return only the Q-values computed by the model. It also introduces the
    `classify` method to provide the probability distribution computed by the model.
    It also includes the `reconstruct` method for getting the reconstructed data if any.
    Finally, it overloads the `to_agent_step` to refine the information returned
    to the agent when executing the policy by including the probability distribution
    and the reconstructed data if any.
    """

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        """Defines the method of the agent for computing the Q-Values.

        This method is overloaded in such a way that it returns only the
        Q-values from the model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        q: tensor
            The computed Q-values.
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().__call__(observation, prev_action, prev_reward, init_rnn_state)
        next_rnn_state = output[-1]
        tmp_out = output[0:-1]
        reb = None
        if len(tmp_out) == 2:
            q, _ = tmp_out
        else:
            q, _, reb = tmp_out
        # saved q for later loss computation (if grad is enabled)
        if torch.is_grad_enabled():
            self.saved_q = q
        # saved reb for later loss computation (if grad is enabled)
        if torch.is_grad_enabled() and reb is not None:
            self.saved_reconstruct = reb
        return q, next_rnn_state

    def target(self, observation, prev_action, prev_reward, init_rnn_state):
        """Returns the target Q-values for states/observations.

        This method is overloaded in such a way that it returns only the
        Q-values from the target model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        q: tensor
            The computed target Q-values.
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().target(observation, prev_action, prev_reward, init_rnn_state)
        q = output[0]
        next_rnn_state = output[-1]
        return q, next_rnn_state

    def _get_pi_actions(self, pi):
        """Epsilon sampling to get the max indices from the provided pi.

        Parameters
        ----------
        pi: tensor
            the predicted distribution.

        Return
        ----------
        result: tensor
            the sampled indices.
        """
        arg_select = torch.argmax(pi, dim=-1)
        mask = torch.rand(arg_select.shape) < self.distribution._epsilon
        arg_rand = torch.randint(low=0, high=pi.shape[-1], size=(mask.sum(),))
        arg_select[mask] = arg_rand
        return arg_select

    def to_agent_step(self, output):
        """Convert the model outputs into Agent step information.

        The agent step info includes the computed Q-values, the selected actions
        to be executed by the agent, the next rnn state, as well as the computed
        probability distribution.

        Parameters
        ----------
        output: tensor, tuple of tensor
            the output of the agent model.

        Return
        ----------
        result: object
            The Agent step information.

        """
        q, pi, next_rnn_state = output[0], output[1], output[-1]
        reb = None if len(output) < 4 else output[2]
        agent_step = super().to_agent_step((q, next_rnn_state))
        pi, reb = buffer_to((pi, reb), device="cpu")
        selected_actions = agent_step.action
        pi_actions = self._get_pi_actions(pi)
        use_stop_action = getattr(self.model, "use_stop_action", True)
        num_q_actions = q.size(-1) - 1 if use_stop_action else q.size(-1)
        agent_info = (
            MixedR2D1AgentInfo(*agent_step.agent_info, dist_info=pi)
            if reb is None
            else MixRebR2D1AgentInfo(
                *agent_step.agent_info, dist_info=pi, rebuild_info=reb
            )
        )

        if use_stop_action:
            pi_actions = pi_actions + num_q_actions
            mask = selected_actions == num_q_actions
            selected_actions[mask] = pi_actions[mask]

        tmp_dict = dict(agent_step.items())
        tmp_dict["agent_info"] = agent_info
        tmp_dict["action"] = selected_actions
        return agent_step.__class__(**tmp_dict)

    def classify(self, observation, prev_action, prev_reward, init_rnn_state):
        """Defines the method of the agent for computing the probability distribution.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        pi: tensor
            The computed probability distributions (logits).
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().__call__(observation, prev_action, prev_reward, init_rnn_state)
        next_rnn_state = output[-1]
        pi = output[1]
        return pi, next_rnn_state

    def reconstruct(self, observation, prev_action, prev_reward, init_rnn_state):
        """Defines the method of the agent for computing the reconstructed data.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        result: tensor
            The reconstructed data.
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().__call__(observation, prev_action, prev_reward, init_rnn_state)
        next_rnn_state = output[-1]
        tmp_out = output[0:-1]
        reb = None if len(tmp_out) < 3 else tmp_out[2]
        return reb, next_rnn_state


class RebuildDQNAgentMixin:
    """Mixin class to properly handle RebuildDQN model outputs by DQN like Agents.

    This class overloads the `__call__` and `target` methods
    to return only the Q-values computed by the model. It also introduces the
    `reconstruct` method to provide the reconstructed data computed by the model.
    Finally, it overloads the `to_agent_step` to refine the information returned
    to the agent when executing the policy by including the reconstructed data.
    """

    def __call__(self, observation, prev_action, prev_reward):
        """Defines the method of the agent for computing the Q-Values.

        This method is overloaded in such a way that it returns only the
        Q-values from the model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The computed Q-values.

        """
        output = super().__call__(observation, prev_action, prev_reward)
        q, reb = output
        # saved reb for later loss computation (if grad is enabled)
        if torch.is_grad_enabled():
            self.saved_reconstruct = reb
        return q

    def target(self, observation, prev_action, prev_reward):
        """Returns the target Q-values for states/observations..

        This method is overloaded in such a way that it returns only the
        Q-values from the target model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The computed target Q-values.

        """
        output = super().target(observation, prev_action, prev_reward)
        q, _ = output
        return q

    def to_agent_step(self, output):
        """Convert the model outputs into Agent step information.

        The agent step info includes the computed Q-values, the selected actions
        to be executed by the agent as well as the reconstructed data.

        Parameters
        ----------
        output: tensor, tuple of tensor
            the output of the agent model.

        Return
        ----------
        result: object
            The Agent step information.

        """
        q, reb = output
        agent_step = super().to_agent_step(q)
        reb = buffer_to(reb, device="cpu")
        selected_actions = agent_step.action
        if isinstance(self, CatDqnAgent):
            agent_info = RebuildCatAgentInfo(*agent_step.agent_info, rebuild_info=reb)
        else:
            agent_info = RebuildAgentInfo(*agent_step.agent_info, rebuild_info=reb)

        tmp_dict = dict(agent_step.items())
        tmp_dict["agent_info"] = agent_info
        tmp_dict["action"] = selected_actions
        return agent_step.__class__(**tmp_dict)

    def reconstruct(self, observation, prev_action, prev_reward):
        """Defines the method of the agent for computing the reconstructed data.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.

        Return
        ----------
        result: tensor
            The reconstructed data.

        """
        output = super().__call__(observation, prev_action, prev_reward)
        _, reb = output
        return reb


class RebuildSeqDQNAgentMixin:
    """Mixin class to properly handle RebuildDQN model outputs by sequential DQN Agents.

    This class overloads the `__call__` and `target` methods
    to return only the Q-values computed by the model. It also introduces the
    `reconstruct` method to provide the reconstructed data computed by the model.
    Finally, it overloads the `to_agent_step` to refine the information returned
    to the agent when executing the policy by including the reconstructed data.
    """

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        """Defines the method of the agent for computing the Q-Values.

        This method is overloaded in such a way that it returns only the
        Q-values from the model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        q: tensor
            The computed Q-values.
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().__call__(observation, prev_action, prev_reward, init_rnn_state)
        q, reb, next_rnn_state = output
        # saved reb for later loss computation (if grad is enabled)
        if torch.is_grad_enabled():
            self.saved_reconstruct = reb
        return q, next_rnn_state

    def target(self, observation, prev_action, prev_reward, init_rnn_state):
        """Returns the target Q-values for states/observations.

        This method is overloaded in such a way that it returns only the
        Q-values from the target model.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        q: tensor
            The computed target Q-values.
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().target(observation, prev_action, prev_reward, init_rnn_state)
        q, _, next_rnn_state = output
        return q, next_rnn_state

    def to_agent_step(self, output):
        """Convert the model outputs into Agent step information.

        The agent step info includes the computed Q-values, the selected actions
        to be executed by the agent, the next rnn state, as well as the computed
        probability distribution.

        Parameters
        ----------
        output: tensor, tuple of tensor
            the output of the agent model.

        Return
        ----------
        result: object
            The Agent step information.

        """
        q, reb, next_rnn_state = output
        agent_step = super().to_agent_step((q, next_rnn_state))
        reb = buffer_to(reb, device="cpu")
        selected_actions = agent_step.action
        agent_info = RebuildR2D1AgentInfo(*agent_step.agent_info, rebuild_info=reb)

        tmp_dict = dict(agent_step.items())
        tmp_dict["agent_info"] = agent_info
        tmp_dict["action"] = selected_actions
        return agent_step.__class__(**tmp_dict)

    def reconstruct(self, observation, prev_action, prev_reward, init_rnn_state):
        """Defines the method of the agent for computing the reconstructed data.

        Parameters
        ----------
        observation: tensor
            the data describing the observation the agent is evaluated on.
        prev_action: tensor
            the data describing the previous action performed by the agent.
        prev_reward: tensor
            the data describing the previous reward received by the agent.
        init_rnn_state: tensor, tuple of tensors
            the data describing the initial state of the model.

        Return
        ----------
        result: tensor
            The reconstructed data.
        next_rnn_state: tensor
            The next rnn state after the computation.

        """
        output = super().__call__(observation, prev_action, prev_reward, init_rnn_state)
        _, reb, next_rnn_state = output
        return reb, next_rnn_state
