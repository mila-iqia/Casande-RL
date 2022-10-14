from rlpyt.samplers.async_.collectors import (
    DbCpuResetCollector,
    DbCpuWaitResetCollector,
    DbGpuResetCollector,
    DbGpuWaitResetCollector,
)
from rlpyt.samplers.parallel.cpu.collectors import (
    CpuEvalCollector,
    CpuResetCollector,
    CpuWaitResetCollector,
)
from rlpyt.samplers.parallel.gpu.collectors import (
    GpuEvalCollector,
    GpuResetCollector,
    GpuWaitResetCollector,
)
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.logging import logger


class CollectorFactory:
    """A factory for instanciating sample collector object.

    The predefined sample collector classes are:
        - CpuEvalCollector
        - CpuResetCollector
        - CpuWaitResetCollector
        - GpuEvalCollector
        - GpuResetCollector
        - GpuWaitResetCollector
        - SerialEvalCollector
        - DbCpuResetCollector
        - DbCpuWaitResetCollector
        - DbGpuResetCollector
        - DbGpuWaitResetCollector

    Please, refer to https://rlpyt.readthedocs.io/en/latest/pages/collector.html
    for mor details.
    """

    def __init__(self):
        self._builders = {
            "CpuEvalCollector".lower(): CpuEvalCollector,
            "CpuResetCollector".lower(): CpuResetCollector,
            "CpuWaitResetCollector".lower(): CpuWaitResetCollector,
            "GpuEvalCollector".lower(): GpuEvalCollector,
            "GpuResetCollector".lower(): GpuResetCollector,
            "GpuWaitResetCollector".lower(): GpuWaitResetCollector,
            "SerialEvalCollector".lower(): SerialEvalCollector,
            "DbCpuResetCollector".lower(): DbCpuResetCollector,
            "DbCpuWaitResetCollector".lower(): DbCpuWaitResetCollector,
            "DbGpuResetCollector".lower(): DbGpuResetCollector,
            "DbGpuWaitResetCollector".lower(): DbGpuWaitResetCollector,
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

    def get_collector_class(self, key):
        """Get a sample collector class based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which sample collector to retrieve.

        Return
        ------
        cls: class
            the class of the collectore associated with the provided key.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid collector key")
        return builder

    def create(self, key, *args, **kwargs):
        """Get a sample collector instance based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which sample collector instance to create.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Return
        ------
        collector: object
            the instantiated sample collector.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid collector key")
        return builder(*args, **kwargs)
