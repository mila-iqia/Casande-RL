import torch

from rlpyt.agents.base import AlternatingRecurrentAgentMixin
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent, R2d1AlternatingAgent
from rlpyt.agents.pg.categorical import (
    AlternatingRecurrentCategoricalPgAgent,
    CategoricalPgAgent,
    RecurrentCategoricalPgAgent,
    RecurrentCategoricalPgAgentBase,
)
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.pg.base import PolicyGradientAlgo
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSamplerBase
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSamplerBase
from rlpyt.utils.logging import logger

from chloe.utils.agent_components import (
    MixedDQNAgentMixin,
    MixedSeqDQNAgentMixin,
    RebuildDQNAgentMixin,
    RebuildSeqDQNAgentMixin,
)
from chloe.utils.reward_shaping_components import (
    MixedDQNRewardShapingLossMixin,
    RebuildDQNLossMixin,
)


class NonInfinityEpsilonGreedy(EpsilonGreedy):
    """Extend EpsilonGreedy to avoid randomly sample actions with infinity Qvalues.
    """

    def sample(self, q):
        """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
        B will apply across the Batch dimension (same epsilon for all T)."""
        arg_select = torch.argmax(q, dim=-1)
        mask = torch.rand(arg_select.shape, device=q.device) < self._epsilon
        arg_rand = torch.randint(
            low=0, high=q.shape[-1], size=arg_select.shape, device=q.device
        )
        with torch.no_grad():
            mask2 = torch.logical_and(
                mask,
                torch.gather(q, -1, arg_rand.unsqueeze(-1)).squeeze(-1)
                != torch.finfo().min,
            )
            arg_rand[~mask2] = arg_select[~mask2]
        arg_select[mask] = arg_rand[mask]
        return arg_select


class DQNAgentParamLoadMixin:
    """Properly handle DQN-like agent initialization based on provided state dict.
    """

    def initialize(self, *args, **kwargs):
        """Method for initializing the agent.

        Parameters
        ----------
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Returns
        -------
        None

        """
        _initial_model_state_dict = self.initial_model_state_dict
        # Don't let base agent try to load.
        self.initial_model_state_dict = None
        super().initialize(*args, **kwargs)
        self.initial_model_state_dict = _initial_model_state_dict
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)

    def load_state_dict(self, state_dict):
        """Method for loading the network params from a state_dict.

        This method overloads the rlpyt package implementation where
        the state_dict of the target model is not saved/loaded when
        checkpointing the agent. In this implementation, the state_dict
        of the target model is explicitly saved and therefore, it can be
        loaded.

        Parameters
        ----------
        state_dict: dict
            dictionary containing the params to be loaded.

        Returns
        -------
        None

        """
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target"])


class ParamReloadDQNAgent(DQNAgentParamLoadMixin, DqnAgent):
    """DQN Agent with enhanced parameter loading functionality.

       The `load_state_dict` function is overloaded to be able to load
       target model from the provided data.
    """

    def initialize(self, env_spaces, *args, **kwargs):
        """Method for initializing the agent.
        """
        super().initialize(env_spaces, *args, **kwargs)
        self.distribution = NonInfinityEpsilonGreedy(dim=env_spaces.action.n)

    pass


class ParamReloadCatDqnAgent(DQNAgentParamLoadMixin, CatDqnAgent):
    """CatDQN Agent with enhanced parameter loading functionality.

       The `load_state_dict` function is overloaded to be able to load
       target model from the provided data.

    """

    pass


class ParamReloadR2d1Agent(DQNAgentParamLoadMixin, R2d1Agent):
    """R2D1 Agent with enhanced parameter loading functionality.

       The `load_state_dict` function is overloaded to be able to load
       target model from the provided data.

    """

    def initialize(self, env_spaces, *args, **kwargs):
        """Method for initializing the agent.
        """
        super().initialize(env_spaces, *args, **kwargs)
        self.distribution = NonInfinityEpsilonGreedy(dim=env_spaces.action.n)

    pass


class ParamReloadR2d1AlternatingAgent(DQNAgentParamLoadMixin, R2d1AlternatingAgent):
    """Alternating R2D1 Agent with enhanced parameter loading functionality.

       The `load_state_dict` function is overloaded to be able to load
       target model from the provided data.

    """

    def initialize(self, env_spaces, *args, **kwargs):
        """Method for initializing the agent.
        """
        super().initialize(env_spaces, *args, **kwargs)
        self.distribution = NonInfinityEpsilonGreedy(dim=env_spaces.action.n)

    pass


class MixedDqnAgent(MixedDQNAgentMixin, ParamReloadDQNAgent):
    """DQN Agent able to deal with mixed outputs: Q-values and prob. distributions.

    """

    pass


class MixedCatDqnAgent(MixedDQNAgentMixin, ParamReloadCatDqnAgent):
    """CatDQN Agent able to deal with mixed outputs: Q-values and prob. distributions.

    """

    pass


class MixedR2d1Agent(MixedSeqDQNAgentMixin, ParamReloadR2d1Agent):
    """R2D1 Agent able to deal with mixed outputs: Q-values and prob. distributions.

    """

    pass


class MixedR2d1AlternatingAgent(MixedSeqDQNAgentMixin, ParamReloadR2d1AlternatingAgent):
    """R2D1 Agent able to deal with mixed outputs: Q-values and prob. distributions.

    This agent differs from the `MixedR2d1Agent` as it works in an alternating
    behavior. That is, it maintains an alternating pair of recurrent states to
    use when stepping in the sampler. It automatically swaps them out when
    ``advance_rnn_state()`` is called, so it otherwise behaves like regular
    recurrent agent.  It should be used only in alternating samplers, where two sets of
    environment instances take turns stepping.

    """

    pass


class RebuildDqnAgent(RebuildDQNAgentMixin, ParamReloadDQNAgent):
    """DQN Agent able to deal with mixed outputs: Q-values and reconstructed data.

    """

    pass


class RebuildCatDqnAgent(RebuildDQNAgentMixin, ParamReloadCatDqnAgent):
    """CatDQN Agent able to deal with mixed outputs: Q-values and reconstructed data.

    """

    pass


class RebuildR2d1Agent(RebuildSeqDQNAgentMixin, ParamReloadR2d1Agent):
    """R2D1 Agent able to deal with mixed outputs: Q-values and reconstructed data.

    """

    pass


class RebuildR2d1AlternatingAgent(
    RebuildSeqDQNAgentMixin, ParamReloadR2d1AlternatingAgent
):
    """R2D1 Agent able to deal with mixed outputs: Q-values and reconstructed data.

    This agent differs from the `RebuildR2d1Agent` as it works in an alternating
    behavior. That is, it maintains an alternating pair of recurrent states to
    use when stepping in the sampler. It automatically swaps them out when
    ``advance_rnn_state()`` is called, so it otherwise behaves like regular
    recurrent agent.  It should be used only in alternating samplers, where two sets of
    environment instances take turns stepping.

    """

    pass


class AgentFactory:
    """A factory for instanciating agent object.

    The predefined agent classes are:
        - DQNAgent
        - CatDqnAgent
        - R2D1Agent
        - R2d1AlternatingAgent
        - Mixed_DQNAgent
        - Mixed_CatDqnAgent
        - Mixed_R2D1Agent
        - Mixed_R2d1AlternatingAgent
        - Rebuild_DQNAgent
        - Rebuild_CatDqnAgent
        - Rebuild_R2D1Agent
        - Rebuild_R2d1AlternatingAgent
        - CategoricalPgAgent
        - RecurrentCategoricalPgAgent
        - AlternatingRecurrentCategoricalPgAgent

    Please, refer to https://rlpyt.readthedocs.io/en/latest/pages/dqn.html
    and https://rlpyt.readthedocs.io/en/latest/pages/pg.html#agents
    for mor details.
    """

    def __init__(self):
        altPgKey = "alternatingrecurrentcategoricalpgagent"
        altDqnKey = "r2d1alternatingagent"
        altMixedDqnKey = "mixed_r2d1alternatingagent"
        altRebuildDqnKey = "rebuild_r2d1alternatingagent"
        self._builders = {
            "categoricalpgagent": CategoricalPgAgent,
            "recurrentcategoricalpgagent": RecurrentCategoricalPgAgent,
            altPgKey: AlternatingRecurrentCategoricalPgAgent,
            "dqnagent": ParamReloadDQNAgent,
            "catdqnagent": ParamReloadCatDqnAgent,
            "r2d1agent": ParamReloadR2d1Agent,
            "mixed_dqnagent": MixedDqnAgent,
            "mixed_catdqnagent": MixedCatDqnAgent,
            "mixed_r2d1agent": MixedR2d1Agent,
            "rebuild_dqnagent": RebuildDqnAgent,
            "rebuild_catdqnagent": RebuildCatDqnAgent,
            "rebuild_r2d1agent": RebuildR2d1Agent,
            altDqnKey: ParamReloadR2d1AlternatingAgent,
            altRebuildDqnKey: RebuildR2d1AlternatingAgent,
            altMixedDqnKey: MixedR2d1AlternatingAgent,
        }

    def register_builder(self, key, builder, force_replace=False):
        """Register an instance within the factory.

        Parameters
        ----------
        key: str
            registration key.
        builder: class
            class to be registered in the factory.
        force_replace: boolean
            Indicate whether to overwrite the key if it
            is already present in the factory. Default: False

        Return
        ------
        None

        """
        assert key is not None
        if not (key.lower() in self._builders):
            logger.log('register the key "{}".'.format(key))
            self._builders[key.lower()] = builder
        else:
            if force_replace:
                logger.log('"{}" already exists - force to erase'.format(key))
                self._builders[key.lower()] = builder
            else:
                logger.log('"{}" already exists - no registration'.format(key))

    def check_policy_gradient_constraints(self, agent_cls, algo_cls):
        """Checks the validity of the agent wrt the provided PG algo.

        Check if the agent to create is compliant with policy gradient algos if needed.

        Parameters
        ----------
        agent_cls: class
            class of the agent to create.
        algo_cls: class
            class of the algo object to be used with the agent to create.

        Return
        ------
        None

        """
        if algo_cls is None:
            return
        if issubclass(algo_cls, PolicyGradientAlgo):
            pg_cls = (CategoricalPgAgent, RecurrentCategoricalPgAgentBase)
            if not issubclass(agent_cls, pg_cls):
                raise ValueError(
                    f'The provided agent class name "{agent_cls}" is not '
                    f'compliant with the provided algorithm class name "{algo_cls}".'
                )

    def check_dqn_constraints(self, agent_cls, algo_cls):
        """Checks the validity of the agent wrt the provided DQN algo.

        Check if the agent to create is compliant with DQN based algos if needed.

        Parameters
        ----------
        agent_cls: class
            class of the agent to create.
        algo_cls: class
            class of the algo object to be used with the agent to create.

        Return
        ------
        None

        """
        if algo_cls is None:
            return
        if issubclass(algo_cls, DQN):
            dqn_cls = (DqnAgent,)
            if not issubclass(agent_cls, dqn_cls):
                raise ValueError(
                    f'The provided agent class name "{agent_cls}" is not '
                    f'compliant with the provided algorithm class name "{algo_cls}".'
                )
            is_mixed = issubclass(algo_cls, MixedDQNRewardShapingLossMixin)
            is_rebuild = issubclass(algo_cls, RebuildDQNLossMixin)
            mixed_cls = (
                MixedDqnAgent,
                MixedCatDqnAgent,
                MixedR2d1Agent,
                MixedR2d1AlternatingAgent,
            )
            rebuild_cls = (
                RebuildDqnAgent,
                RebuildCatDqnAgent,
                RebuildR2d1Agent,
                RebuildR2d1AlternatingAgent,
                MixedDqnAgent,
                MixedCatDqnAgent,
                MixedR2d1Agent,
                MixedR2d1AlternatingAgent,
            )
            if is_mixed:
                if not issubclass(agent_cls, mixed_cls):
                    raise ValueError(
                        f'The provided agent class name "{agent_cls}" is not '
                        f"compliant with the provided algorithm class name"
                        f'"{algo_cls}". Please consider using a Mixed DQN agent.'
                    )
            else:
                if issubclass(agent_cls, mixed_cls):
                    raise ValueError(
                        f'The provided agent class name "{agent_cls}" is not '
                        f"compliant with the provided algorithm class name"
                        f'"{algo_cls}". Please consider using a NON Mixed DQN agent.'
                    )
            if is_rebuild:
                if not issubclass(agent_cls, rebuild_cls):
                    raise ValueError(
                        f'The provided agent class name "{agent_cls}" is not '
                        f"compliant with the provided algorithm class name"
                        f'"{algo_cls}". Please consider using a Rebuild DQN agent.'
                    )
            else:
                if issubclass(agent_cls, rebuild_cls):
                    raise ValueError(
                        f'The provided agent class name "{agent_cls}" is not '
                        f"compliant with the provided algorithm class name"
                        f'"{algo_cls}". Please consider using a NON Rebuild DQN agent.'
                    )

    def check_sampler_constraints(self, agent_cls, sampler_cls):
        """Checks the validity of the agent wrt the provided sampler.

        Check if the agent to create is compliant with
        the sampler to be used.

        Parameters
        ----------
        agent_cls: class
            class of the agent to create.
        sampler_cls: class
            class of the sampler object to be used with the agent to create.

        Return
        ------
        None

        """
        if sampler_cls is None:
            return
        if issubclass(agent_cls, AlternatingRecurrentAgentMixin):
            alt_sampler_cls = (AlternatingSamplerBase, AsyncAlternatingSamplerBase)
            if not issubclass(sampler_cls, alt_sampler_cls):
                raise ValueError(
                    f'The provided agent class name "{agent_cls}" is not compliant '
                    f'with the provided sampler class name "{sampler_cls}".'
                )

    def get_agent_class(self, key):
        """Get an agent class based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which agent class to retrieve.

        Return
        ------
        builder: agent
            the class the agent specified by the key.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid agent key")
        return builder

    def create(self, key, algo_cls, sampler_cls, *args, **kwargs):
        """Get an agent instance based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which agent instance to create.
        algo_cls: class
            class of the algo object to be used with the agent to create.
        sampler_cls: class
            class of the sampler object to be used with the agent to create.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Return
        ------
        result: agent
            an instance of the agent specified by the key.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid agent key")
        self.check_policy_gradient_constraints(builder, algo_cls)
        self.check_dqn_constraints(builder, algo_cls)
        self.check_sampler_constraints(builder, sampler_cls)
        return builder(*args, **kwargs)
