import random

from chloe.agents.base import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, env):
        """Init method for the agent class.

        Parameters
        ----------
        env : object
            the environment to interact with.
        """
        super(RandomAgent).__init__()
        self.num_actions = env.action_space.n

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionary. The action is accessible using the key
        `action`.

        Parameters
        ----------
        observations : object
            the provided observation.

        Returns
        -------
        result: dict
            the returned dictionary.
        """
        action = random.randint(0, self.num_actions - 1)
        return {"action": action}
