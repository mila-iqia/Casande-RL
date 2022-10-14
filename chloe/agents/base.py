class AbstractAgent:
    def reset(self):
        """Function for resseting the internal state of the agent.

        This function is useful when the agent works in a reccurrent setting.

        Parameters
        ----------
        Returns
        -------
        None
        """
        pass

    def predict(self, observations) -> dict:
        """Function for predicting the action given the observation.

        It returns actions and potentially some extra information
        in a dictionary. The action is accessible using the key `action`.

        """
        raise NotImplementedError
