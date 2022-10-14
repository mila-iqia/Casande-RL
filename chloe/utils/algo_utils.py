from rlpyt.algos.pg.a2c import A2C
from rlpyt.utils.logging import logger

from chloe.utils.algo_components import (
    AugCategoricalDQN,
    AugDQN,
    AugMixedCategoricalDQN,
    AugMixedDQN,
    AugMixedR2D1,
    AugPPO,
    AugR2D1,
    AugRebuildCategoricalDQN,
    AugRebuildDQN,
    AugRebuildR2D1,
)


class AlgoFactory:
    """A factory for instanciating algo object.

    The predefined algorithms are:
        - A2C
        - PPO
        - DQN
        - CategoricalDQN
        - R2D1
        - MixedDQN
        - MixedCategoricalDQN
        - MixedR2D1

    Please, refer to https://rlpyt.readthedocs.io/en/latest/pages/dqn.html
    and https://rlpyt.readthedocs.io/en/latest/pages/pg.html
    for mor details.
    """

    def __init__(self):
        """Instantiates the factory.
        """
        self._builders = {
            "a2c": A2C,
            "ppo": AugPPO,
            "dqn": AugDQN,
            "categoricaldqn": AugCategoricalDQN,
            "r2d1": AugR2D1,
            "mixed_dqn": AugMixedDQN,
            "mixed_categoricaldqn": AugMixedCategoricalDQN,
            "mixed_r2d1": AugMixedR2D1,
            "rebuild_dqn": AugRebuildDQN,
            "rebuild_categoricaldqn": AugRebuildCategoricalDQN,
            "rebuild_r2d1": AugRebuildR2D1,
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

    def create(self, key, *args, **kwargs):
        """Get an algo instance based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which algo instance to create.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Return
        ------
        algo: object
            the algo instance.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid algo key")
        return builder(*args, **kwargs)
