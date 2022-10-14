from rlpyt.samplers.async_.alternating_sampler import (
    AsyncAlternatingSampler,
    AsyncAlternatingSamplerBase,
    AsyncNoOverlapAlternatingSampler,
)
from rlpyt.samplers.async_.base import AsyncParallelSamplerMixin, AsyncSamplerMixin
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler, AsyncGpuSamplerBase
from rlpyt.samplers.async_.serial_sampler import AsyncSerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import (
    AlternatingSampler,
    AlternatingSamplerBase,
    NoOverlapAlternatingSampler,
)
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger


def is_aternating_sampler(sampler):
    """Check if a sampler is an alternating sampler.

    Parameters
    ----------
    sampler: object
        the provided sampler object.

    Return
    ------
    result: bool
        True if the provided sample is an alternating sampler, False otherwise.

    """
    alt_sampler_cls = (AlternatingSamplerBase, AsyncAlternatingSamplerBase)
    return isinstance(sampler, alt_sampler_cls)


def is_async_sampler(sampler):
    """Check if a sampler is an async sampler.

    Parameters
    ----------
    sampler: object
        the provided sampler object.

    Return
    ------
    result: bool
        True if the provided sample is an async sampler, False otherwise.

    """
    asyn_sampler_cls = (
        AsyncGpuSamplerBase,
        AsyncAlternatingSamplerBase,
        AsyncSerialSampler,
        AsyncGpuSampler,
        AsyncCpuSampler,
        AsyncSamplerMixin,
        AsyncParallelSamplerMixin,
    )
    return isinstance(sampler, asyn_sampler_cls)


class SamplerFactory:
    """A factory for instanciating sampler object.

    The predefined sampler classes are:
        - SerialSampler
        - CpuSampler
        - GpuSampler
        - AlternatingSampler
        - NoOverlapAlternatingSampler
        - AsyncSerialSampler
        - AsyncCpuSampler
        - AsyncGpuSampler
        - AsyncAlternatingSampler
        - AsyncNoOverlapAlternatingSampler

    Please, refer to https://rlpyt.readthedocs.io/en/latest/pages/sampler.html
    for mor details.
    """

    def __init__(self):
        self._builders = {
            "serialsampler": SerialSampler,
            "cpusampler": CpuSampler,
            "gpusampler": GpuSampler,
            "alternatingsampler": AlternatingSampler,
            "nooverlapalternatingsampler": NoOverlapAlternatingSampler,
            "asyncserialsampler": AsyncSerialSampler,
            "asynccpusampler": AsyncCpuSampler,
            "asyncgpusampler": AsyncGpuSampler,
            "asyncalternatingsampler": AsyncAlternatingSampler,
            "asyncnooverlapalternatingsampler": AsyncNoOverlapAlternatingSampler,
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
            indicate whether to overwrite the key if it
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
        """Get a sampler instance based on the provided key.

        Parameters
        ----------
        key: str
            key indication of which sampler instance to create.
        args: list
            list of arguments.
        kwargs: dict
            dict of  arguments.

        Return
        ------
        sampler: object
            the created sampler.

        """
        builder = self._builders.get(key.lower())
        if not builder:
            raise ValueError(key + " is not a valid sampler key")
        return builder(*args, **kwargs)
